"""Query the vector store and decide whether the corpus can answer at all."""

from __future__ import annotations

from app.retrieval.backends.base import VectorBackend
from app.schemas.retrieval import Citation, RetrievalResult, RetrievedChunk


class Retriever:
    def __init__(self, store: VectorBackend, *, top_k: int, min_score: float) -> None:
        self._store = store
        self._top_k = top_k
        self._min_score = min_score

    def retrieve(self, query: str, *, top_k: int | None = None) -> RetrievalResult:
        query = query.strip()
        if not query:
            raise ValueError("query must not be empty")

        hits = self._store.search(query, top_k or self._top_k)
        hits = _dedupe(hits)
        top_score = hits[0].score if hits else 0.0
        confident = top_score >= self._min_score
        # Drop the long tail of weak matches, but always keep the best one so the
        # caller can show *something* and explain the low confidence.
        kept = [h for h in hits if h.score >= self._min_score] or hits[:1]
        return RetrievalResult(query=query, chunks=kept, top_score=top_score, confident=confident)


def _dedupe(hits: list[RetrievedChunk]) -> list[RetrievedChunk]:
    seen: set[str] = set()
    out: list[RetrievedChunk] = []
    for h in hits:
        if h.chunk_id in seen:
            continue
        seen.add(h.chunk_id)
        out.append(h)
    return out


def build_citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    return [
        Citation(
            marker=f"[{i}]",
            filename=c.filename,
            locator=c.locator,
            chunk_id=c.chunk_id,
        )
        for i, c in enumerate(chunks, start=1)
    ]
