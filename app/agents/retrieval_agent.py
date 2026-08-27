"""Retrieval agent: fetch context and produce a grounded, cited answer."""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from app.agents.base import Agent
from app.agents.grounding import split_sentences
from app.prompts import QUERY_ANSWER_SYSTEM
from app.retrieval.retriever import Retriever, build_citations
from app.schemas.retrieval import QueryAnswer, RetrievalResult, RetrievedChunk


class _AnswerDraft(BaseModel):
    answer: str
    used_sources: list[int] = Field(
        default_factory=list, description="1-based indices of chunks cited"
    )
    confident: bool = True
    notes: list[str] = Field(default_factory=list)


class RetrievalAgent(Agent):
    name = "retrieval"

    def __init__(self, settings, llm, retriever: Retriever) -> None:
        super().__init__(settings, llm)
        self._retriever = retriever

    def gather(self, query: str, *, top_k: int | None = None) -> RetrievalResult:
        return self._retriever.retrieve(query, top_k=top_k)

    def answer_query(self, query: str, *, top_k: int | None = None) -> tuple[QueryAnswer, int]:
        result = self._retriever.retrieve(query, top_k=top_k)
        if not result.chunks:
            return (
                QueryAnswer(
                    query=query,
                    answer="No documents have been ingested yet, so this cannot be answered.",
                    citations=[],
                    supporting_chunks=[],
                    confident=False,
                    notes=["corpus is empty"],
                ),
                0,
            )

        if self.uses_claude:
            draft, retries, error = self.with_retry(
                lambda: self._claude_answer(query, result.chunks),
                on_error="retrieval-answer",
            )
            if draft is not None:
                return self._assemble(query, result, draft), retries
            # fall back to extractive answer if the model keeps failing
            answer = self._extractive_answer(query, result)
            answer.notes.append(f"LLM answer unavailable ({error}); showing extractive fallback")
            return answer, retries

        return self._extractive_answer(query, result), 0

    # -- claude path ---------------------------------------------------
    def _claude_answer(self, query: str, chunks: list[RetrievedChunk]) -> _AnswerDraft:
        context = "\n\n".join(
            f"[{i}] ({c.filename}, {c.locator})\n{c.text}" for i, c in enumerate(chunks, start=1)
        )
        return self.llm.structured(
            system=QUERY_ANSWER_SYSTEM,
            user=f"Question: {query}\n\nContext:\n{context}",
            model=_AnswerDraft,
        )

    def _assemble(self, query: str, result: RetrievalResult, draft: _AnswerDraft) -> QueryAnswer:
        chunks = result.chunks
        cited_idx = [i for i in draft.used_sources if 1 <= i <= len(chunks)]
        if not cited_idx:
            cited_idx = [i for i in range(1, len(chunks) + 1) if f"[{i}]" in draft.answer]
        supporting = [chunks[i - 1] for i in cited_idx] or chunks[:1]
        citations = build_citations(supporting)
        confident = draft.confident and result.confident and bool(cited_idx)
        notes = list(draft.notes)
        if not cited_idx:
            notes.append("model did not cite any source; answer treated as low-confidence")
        return QueryAnswer(
            query=query,
            answer=draft.answer.strip(),
            citations=citations,
            supporting_chunks=supporting,
            confident=confident,
            notes=notes,
        )

    # -- mock / fallback path ---------------------------------------
    def _extractive_answer(self, query: str, result: RetrievalResult) -> QueryAnswer:
        query_words = set(re.findall(r"[a-z0-9]+", query.lower()))
        scored: list[tuple[float, str, int]] = []
        for idx, chunk in enumerate(result.chunks, start=1):
            for sentence in split_sentences(chunk.text):
                words = re.findall(r"[a-z0-9]+", sentence.lower())
                s_words = set(words)
                if len(words) < 4 or sentence.rstrip().endswith(":"):
                    continue  # skip heading fragments ("Steps:", "Rollback")
                overlap = len(query_words & s_words) / len(query_words or {1})
                scored.append((overlap, sentence, idx))
        scored.sort(key=lambda t: t[0], reverse=True)

        picked: list[tuple[str, int]] = [(s, i) for score, s, i in scored[:3] if score > 0]
        if not picked:
            # nothing overlapped the query - fall back to the lead sentence of the best chunk
            lead = split_sentences(result.chunks[0].text) or [result.chunks[0].text[:240]]
            picked = [(lead[0], 1)]

        used_indices = sorted({i for _, i in picked})
        supporting = [result.chunks[i - 1] for i in used_indices]
        citations = build_citations(supporting)
        marker_by_chunk = {c.chunk_id: cit.marker for c, cit in zip(supporting, citations)}

        answer_parts = []
        for sentence, chunk_idx in picked:
            marker = marker_by_chunk[result.chunks[chunk_idx - 1].chunk_id]
            answer_parts.append(f"{sentence} {marker}")
        answer = " ".join(answer_parts)

        corpus_words_all = set(
            re.findall(r"[a-z0-9]+", " ".join(c.text for c in result.chunks).lower())
        )
        meaningful_q = {w for w in query_words if len(w) > 3}
        coverage_all = (
            len(meaningful_q & corpus_words_all) / len(meaningful_q) if meaningful_q else 1.0
        )
        best_overlap = scored[0][0] if scored else 0.0
        if coverage_all < 0.5 or (not result.confident and best_overlap < 0.2):
            return QueryAnswer(
                query=query,
                answer=(
                    "The ingested documents do not appear to cover this question. "
                    "No grounded answer can be given from the current corpus."
                ),
                citations=[],
                supporting_chunks=[],
                confident=False,
                notes=[
                    f"query-term coverage {coverage_all:.2f}, best sentence overlap "
                    f"{best_overlap:.2f} - below the threshold to answer"
                ],
            )
        confident = result.confident and best_overlap >= 0.3 and coverage_all >= 0.5
        notes: list[str] = []
        if not confident:
            notes.append(
                f"low confidence (top similarity {result.top_score:.2f}, "
                f"best sentence overlap {best_overlap:.2f}); the corpus may not "
                "cover this question"
            )
        return QueryAnswer(
            query=query,
            answer=answer,
            citations=citations,
            supporting_chunks=supporting,
            confident=confident,
            notes=notes,
        )
