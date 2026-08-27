"""Wire the five agents together and execute a plan end to end.

Design choices:
- The Planner proposes a plan, but the orchestrator owns control flow. It always
  ensures retrieval context exists before the Action step and always runs
  Validation last, regardless of what the plan said.
- A failure in one step degrades the run (sets `degraded=True`, records the
  error on that step) instead of aborting - the operator still gets partial
  output plus an honest validation verdict.
"""

from __future__ import annotations

import logging

from app.agents.action_agent import ActionAgent
from app.agents.data_agent import DataAnalysisAgent
from app.agents.planner import PlannerAgent
from app.agents.retrieval_agent import RetrievalAgent
from app.agents.validation_agent import ValidationAgent
from app.config import Settings
from app.data.tables import TableStore
from app.retrieval.retriever import Retriever
from app.schemas.agents import (
    AgentName,
    AgentRunResult,
    AgentStep,
    DataAnalysisResult,
    ValidationReport,
)
from app.schemas.retrieval import RetrievalResult
from app.services.llm import LLMClient
from app.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(
        self,
        *,
        settings: Settings,
        llm: LLMClient,
        retriever: Retriever,
        table_store: TableStore,
        tools: ToolRegistry,
    ) -> None:
        self._settings = settings
        self._llm = llm
        self._table_store = table_store
        self.planner = PlannerAgent(settings, llm)
        self.retrieval = RetrievalAgent(settings, llm, retriever)
        self.data = DataAnalysisAgent(settings, llm, table_store)
        self.action = ActionAgent(settings, llm, tools)
        self.validation = ValidationAgent(settings, llm)

    def run(self, request: str, *, session_id: str) -> AgentRunResult:
        request = request.strip()
        if not request:
            raise ValueError("request must not be empty")

        has_tables = bool(self._table_store.list_tables())
        plan, plan_retries, plan_error = self.planner.plan(request, has_tables=has_tables)

        steps: list[AgentStep] = []
        degraded = plan_error is not None
        retrieval_result: RetrievalResult | None = None
        data_result: DataAnalysisResult | None = None

        planned_agents = {s.agent for s in plan.steps}

        # 1. Retrieval (always, so Action has grounding)
        try:
            retrieval_result = self.retrieval.gather(request)
            steps.append(
                AgentStep(
                    agent=AgentName.retrieval,
                    summary=(
                        f"retrieved {len(retrieval_result.chunks)} chunk(s); "
                        f"top score {retrieval_result.top_score:.2f}; "
                        f"confident={retrieval_result.confident}"
                    ),
                    retrieved=retrieval_result.chunks,
                )
            )
        except Exception as exc:  # noqa: BLE001 - recorded, then degrade
            logger.exception("retrieval step failed")
            degraded = True
            steps.append(
                AgentStep(agent=AgentName.retrieval, summary="retrieval failed", error=str(exc))
            )
            retrieval_result = RetrievalResult(
                query=request, chunks=[], top_score=0.0, confident=False
            )

        # 2. Data analysis (only if planned and tables exist)
        if AgentName.data_analysis in planned_agents and has_tables:
            try:
                data_result, calls = self.data.analyse(request)
                steps.append(
                    AgentStep(
                        agent=AgentName.data_analysis,
                        summary=(
                            f"analysed table {data_result.table!r}: "
                            f"{len(data_result.findings)} finding(s), "
                            f"{len(data_result.anomalies)} anomaly(ies)"
                        ),
                        tool_calls=calls,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("data analysis step failed")
                degraded = True
                steps.append(
                    AgentStep(
                        agent=AgentName.data_analysis,
                        summary="data analysis failed",
                        error=str(exc),
                    )
                )

        # 3. Action
        decision = None
        try:
            decision, tool_calls = self.action.decide(
                request=request, retrieval=retrieval_result, data=data_result
            )
            steps.append(
                AgentStep(
                    agent=AgentName.action,
                    summary=(
                        f"produced decision (severity={decision.incident.severity}, "
                        f"confidence={decision.confidence}, "
                        f"{len(decision.remediation_checklist)} checklist item(s))"
                    ),
                    tool_calls=tool_calls,
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("action step failed")
            degraded = True
            steps.append(AgentStep(agent=AgentName.action, summary="action failed", error=str(exc)))

        # 4. Validation (always)
        if decision is not None:
            data_context = None
            if data_result:
                data_context = [f.observation for f in data_result.findings] + data_result.anomalies
            validation = self.validation.validate(
                decision,
                retrieval_result.chunks,
                expect_grounding=True,
                data_context=data_context,
            )
        else:
            validation = ValidationReport(
                passed=False,
                grounded=False,
                issues=[],
            )
        steps.append(
            AgentStep(
                agent=AgentName.validation,
                summary=(
                    f"passed={validation.passed}, grounded={validation.grounded}, "
                    f"{validation.supported_claims}/{validation.checked_claims} claims supported"
                ),
            )
        )
        if steps and plan_retries:
            steps[0].retries = plan_retries

        return AgentRunResult(
            session_id=session_id,
            request=request,
            plan=plan,
            steps=steps,
            decision=decision,
            validation=validation,
            llm_mode=self._llm.mode,
            degraded=degraded or not validation.passed,
        )
