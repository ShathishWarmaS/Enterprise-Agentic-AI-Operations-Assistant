"""Planner agent: turn a request into an ordered agent/tool plan."""

from __future__ import annotations

from app.agents.base import Agent
from app.prompts import PLANNER_SYSTEM
from app.schemas.agents import AgentName, Plan, PlanStep

_DATA_HINTS = (
    "how many",
    "count",
    "rate",
    "average",
    "trend",
    "percent",
    "%",
    "metric",
    "table",
    "csv",
    "rows",
    "column",
    "missing value",
    "data quality",
    "anomal",
    "spike",
    "increase",
    "decrease",
    "top ",
    "sum of",
)
_DOC_HINTS = (
    "runbook",
    "postmortem",
    "incident",
    "policy",
    "procedure",
    "why",
    "cause",
    "how do we",
    "how to",
    "what is the",
    "guidance",
    "recommend",
    "should we",
)


class PlannerAgent(Agent):
    name = "planner"

    def plan(self, request: str, *, has_tables: bool) -> tuple[Plan, int, str | None]:
        request = request.strip()
        if not request:
            raise ValueError("request must not be empty")

        if self.uses_claude:
            result, retries, error = self.with_retry(
                lambda: self.llm.structured(
                    system=PLANNER_SYSTEM,
                    user=f"Request: {request}\nIngested tables available: {has_tables}",
                    model=Plan,
                ),
                on_error="planner",
            )
            if result is not None:
                return _sanitise(result, request), retries, error
            # fall through to the deterministic plan on repeated LLM failure

        return self._mock_plan(request, has_tables), 0, None

    def _mock_plan(self, request: str, has_tables: bool) -> Plan:
        lower = request.lower()
        wants_data = has_tables and any(h in lower for h in _DATA_HINTS)
        wants_docs = any(h in lower for h in _DOC_HINTS) or not wants_data

        steps: list[PlanStep] = []
        n = 1
        if wants_docs:
            steps.append(
                PlanStep(
                    step=n,
                    agent=AgentName.retrieval,
                    objective="Retrieve documentation relevant to the request",
                    tool="search_documents",
                )
            )
            n += 1
        if wants_data:
            steps.append(
                PlanStep(
                    step=n,
                    agent=AgentName.data_analysis,
                    objective="Query the ingested table(s) for the metrics the request implies",
                    tool="query_data",
                )
            )
            n += 1
        steps.append(
            PlanStep(
                step=n,
                agent=AgentName.action,
                objective="Draft an incident summary, next steps, and a remediation checklist",
                tool="draft_incident_summary",
            )
        )
        steps.append(
            PlanStep(
                step=n + 1,
                agent=AgentName.validation,
                objective="Validate that every conclusion is grounded in retrieved context",
            )
        )
        rationale = (
            "Request looks data-oriented; "
            if wants_data
            else "Request looks documentation-oriented; "
        ) + "action step always runs to produce the decision, then validation gates it."
        return Plan(request=request, steps=steps, rationale=rationale)


def _sanitise(plan: Plan, request: str) -> Plan:
    """Ensure the LLM plan always ends with action + validation."""
    steps = [s for s in plan.steps if s.agent in AgentName.__members__.values()]
    agents = {s.agent for s in steps}
    n = len(steps)
    if AgentName.action not in agents:
        n += 1
        steps.append(
            PlanStep(
                step=n,
                agent=AgentName.action,
                objective="Produce the operational decision",
                tool="draft_incident_summary",
            )
        )
    if AgentName.validation not in agents:
        n += 1
        steps.append(
            PlanStep(
                step=n, agent=AgentName.validation, objective="Validate grounding of the decision"
            )
        )
    for i, step in enumerate(steps, start=1):
        step.step = i
    return Plan(request=request, steps=steps, rationale=plan.rationale)
