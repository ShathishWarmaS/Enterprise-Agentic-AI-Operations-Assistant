"""Action agent: call MCP tools and assemble the structured operational decision."""

from __future__ import annotations

import re

from app.agents.base import Agent
from app.agents.grounding import split_sentences
from app.prompts import ACTION_SYSTEM
from app.retrieval.retriever import build_citations
from app.schemas.agents import (
    ChecklistItem,
    DataAnalysisResult,
    IncidentSummary,
    OperationalDecision,
    ToolCall,
)
from app.schemas.retrieval import RetrievalResult
from app.tools.registry import ToolRegistry

_SEVERITY_KEYWORDS = {
    "critical": ("outage", "down", "data loss", "breach", "cannot", "all users"),
    "high": ("error", "failing", "degraded", "spike", "elevated", "5xx", "timeout"),
    "medium": ("slow", "intermittent", "warning", "some users", "increase"),
}
_RECOMMEND_RE = re.compile(
    r"\b(should|recommend|must|need to|ensure|rotate|roll back|restart|scale)\b", re.I
)
_CAUSE_RE = re.compile(
    r"\b(because|due to|caused by|root cause|resulted from|triggered by)\b", re.I
)


class ActionAgent(Agent):
    name = "action"

    def __init__(self, settings, llm, tools: ToolRegistry) -> None:
        super().__init__(settings, llm)
        self._tools = tools

    def decide(
        self,
        *,
        request: str,
        retrieval: RetrievalResult,
        data: DataAnalysisResult | None,
    ) -> tuple[OperationalDecision, list[ToolCall]]:
        chunks = retrieval.chunks
        citations = build_citations(chunks)
        severity = self._severity(request, data)

        observations = self._observations(request, retrieval, data, citations)
        likely_cause = self._likely_cause(chunks, citations, request)
        impact = self._impact(request, data)

        summary_call = self._tools.get("draft_incident_summary").invoke(
            {
                "title": _title(request),
                "severity": severity,
                "impact": impact,
                "likely_cause": likely_cause,
                "observations": observations,
                "evidence": [c.model_dump() for c in citations],
            }
        )
        checklist_call = self._tools.get("generate_checklist").invoke(
            {"scenario": request, "severity": severity}
        )
        tool_calls = [summary_call, checklist_call]

        incident = self._incident_from_call(
            summary_call, request, severity, impact, likely_cause, observations, citations
        )
        checklist = self._checklist_from_call(checklist_call, severity)
        next_steps = self._next_steps(retrieval, checklist)
        confidence, open_questions = self._confidence(retrieval, data)

        decision = OperationalDecision(
            request=request,
            incident=incident,
            recommended_next_steps=next_steps,
            remediation_checklist=checklist,
            citations=citations,
            data_findings=list(data.findings) if data else [],
            open_questions=open_questions,
            confidence=confidence,
        )

        if self.uses_claude:
            refined, _, _ = self.with_retry(
                lambda: self._claude_refine(request, retrieval, data, decision),
                on_error="action-refine",
            )
            if refined is not None:
                decision = refined

        return decision, tool_calls

    # -- claude path -------------------------------------------------
    def _claude_refine(
        self, request, retrieval, data, base: OperationalDecision
    ) -> OperationalDecision:
        context = "\n\n".join(
            f"[{i}] ({c.filename}, {c.locator}) {c.text}"
            for i, c in enumerate(retrieval.chunks, start=1)
        )
        data_json = data.model_dump_json(indent=2) if data else "null"
        draft = self.llm.structured(
            system=ACTION_SYSTEM,
            user=(
                f"Request: {request}\n\nRetrieval context:\n{context}\n\n"
                f"Data findings JSON:\n{data_json}\n\n"
                f"Deterministic draft to improve (keep the checklist and citations):\n"
                f"{base.model_dump_json(indent=2)}"
            ),
            model=OperationalDecision,
        )
        # citations and checklist come from tools, not the model
        draft.citations = base.citations
        draft.incident.evidence = base.citations
        if not draft.remediation_checklist:
            draft.remediation_checklist = base.remediation_checklist
        return draft

    # -- deterministic helpers -------------------------------------
    @staticmethod
    def _severity(request: str, data: DataAnalysisResult | None) -> str:
        lower = request.lower()
        for level, keywords in _SEVERITY_KEYWORDS.items():
            if any(k in lower for k in keywords):
                return level
        if data and data.anomalies:
            return "medium"
        return "low"

    def _observations(self, request, retrieval, data, citations) -> list[str]:
        obs: list[str] = []
        marker = {c.chunk_id: cit.marker for c, cit in zip(retrieval.chunks, citations)}
        # Only cite document sentences when retrieval was actually confident;
        # otherwise we'd be attaching authoritative-looking markers to noise.
        if retrieval.confident:
            for chunk in retrieval.chunks[:3]:
                sentence = _best_sentence(chunk.text, request, min_overlap=0.18)
                if sentence:
                    obs.append(f"{sentence} {marker[chunk.chunk_id]}")
        if data:
            for finding in data.findings[:4]:
                obs.append(finding.observation)
            for anomaly in data.anomalies[:3]:
                obs.append(f"Data quality: {anomaly}")
        if not obs:
            obs.append(
                "No confident supporting context was retrieved; this summary is based "
                "only on the operator's description of the situation."
            )
        return obs

    @staticmethod
    def _likely_cause(chunks, citations, request: str = "") -> str:
        marker = {c.chunk_id: cit.marker for c, cit in zip(chunks, citations)}
        query_words = set(re.findall(r"[a-z0-9]+", request.lower()))
        best: tuple[float, str, str] | None = None
        for chunk in chunks:
            for sentence in split_sentences(chunk.text):
                if not _CAUSE_RE.search(sentence):
                    continue
                # a sentence that only says "until the root cause is confirmed"
                # is about process, not the cause itself - require an actual
                # causal connective and some overlap with the request
                if not re.search(
                    r"\b(because|due to|caused by|resulted from|triggered by)\b", sentence, re.I
                ):
                    continue
                s_words = set(re.findall(r"[a-z0-9]+", sentence.lower()))
                overlap = len(query_words & s_words) / (len(query_words) or 1)
                score = overlap + 0.1 * len(s_words) ** 0.5
                if best is None or score > best[0]:
                    best = (score, sentence, chunk.chunk_id)
        if best is not None:
            return f"{best[1]} {marker[best[2]]}"
        if chunks:
            lead = split_sentences(chunks[0].text)[:1] or [chunks[0].text[:200]]
            return f"{lead[0]} {marker[chunks[0].chunk_id]}"
        return "Undetermined - no supporting documentation was retrieved."

    @staticmethod
    def _impact(request: str, data: DataAnalysisResult | None) -> str:
        if data and data.findings:
            headline = data.findings[0].observation
            return f"Based on ingested data: {headline}."
        return f"Scope inferred from the request only: {_title(request)}."

    def _incident_from_call(
        self, call: ToolCall, request, severity, impact, cause, obs, citations
    ) -> IncidentSummary:
        if call.ok and call.result:
            return IncidentSummary.model_validate(call.result)
        # tool failed (shouldn't for valid args) - build directly so the run still completes
        return IncidentSummary(
            title=_title(request),
            severity=severity,
            summary=" ".join(obs),
            impact=impact,
            likely_cause=cause,
            evidence=citations,
        )

    @staticmethod
    def _checklist_from_call(call: ToolCall, severity: str) -> list[ChecklistItem]:
        if call.ok and call.result:
            return [ChecklistItem.model_validate(i) for i in call.result["items"]]
        return [
            ChecklistItem(
                order=1,
                action="Mitigate to restore service",
                owner_role="on-call-engineer",
                blocking=True,
            ),
            ChecklistItem(
                order=2,
                action="Verify recovery via monitoring",
                owner_role="on-call-engineer",
                blocking=True,
            ),
            ChecklistItem(
                order=3, action="Schedule a blameless postmortem", owner_role="service-owner"
            ),
        ]

    @staticmethod
    def _next_steps(retrieval: RetrievalResult, checklist: list[ChecklistItem]) -> list[str]:
        steps: list[str] = []
        for chunk in retrieval.chunks:
            for sentence in split_sentences(chunk.text):
                if _RECOMMEND_RE.search(sentence) and sentence not in steps:
                    steps.append(sentence)
        steps.extend(item.action for item in checklist if item.blocking)
        # de-dupe preserving order, cap the list
        ordered = list(dict.fromkeys(steps))
        return ordered[:6] or ["Follow the remediation checklist below."]

    @staticmethod
    def _confidence(
        retrieval: RetrievalResult, data: DataAnalysisResult | None
    ) -> tuple[str, list[str]]:
        questions: list[str] = []
        if not retrieval.confident or len(retrieval.chunks) < 2:
            questions.append(
                "Retrieved context is thin; confirm against a current runbook before acting."
            )
        if data and data.missing_fields:
            questions.append(
                f"Data columns with missing values: {', '.join(data.missing_fields[:5])}."
            )
        if retrieval.confident and len(retrieval.chunks) >= 2 and not questions:
            return "high", questions
        if retrieval.chunks:
            return ("medium" if retrieval.confident else "low"), questions
        return "low", questions or ["No supporting evidence was found."]


def _title(request: str) -> str:
    text = " ".join(request.strip().split())
    return (text[:80] + "…") if len(text) > 80 else text


def _best_sentence(text: str, request: str, *, min_overlap: float = 0.0) -> str | None:
    query_words = set(re.findall(r"[a-z0-9]+", request.lower()))
    best, best_score, best_overlap = "", -1.0, 0.0
    for sentence in split_sentences(text):
        s_words = set(re.findall(r"[a-z0-9]+", sentence.lower()))
        if not s_words:
            continue
        overlap = len(query_words & s_words) / len(s_words)
        score = overlap / len(s_words) ** 0.25
        if score > best_score:
            best, best_score, best_overlap = sentence, score, overlap
    if best_overlap < min_overlap:
        return None
    return best or (text[:200] if min_overlap == 0.0 else None)
