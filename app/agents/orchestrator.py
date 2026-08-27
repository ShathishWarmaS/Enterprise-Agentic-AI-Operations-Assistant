"""Wire the five agents together and execute a plan end to end.

Design choices:
- The Planner proposes a plan, but the orchestrator owns control flow. It always
  ensures retrieval context exists before the Action step and always runs
  Validation last, regardless of what the plan said.
- A failure in one step degrades the run (sets `degraded=True`, records the
  error on that step) instead of aborting - the operator still gets partial
  output plus an honest validation verdict.
- `run_streaming()` is the single execution path; `run()` collects it. Retrieval
  and data-analysis are independent, so when `settings.agent_parallel` is set
  and both are needed they run concurrently on a 2-worker pool. The emitted
  `steps` are always ordered [retrieval, data_analysis, action, validation] so
  the output is byte-identical to the sequential path regardless of which
  future finishes first.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor

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
        result: AgentRunResult | None = None
        for kind, payload in self.run_streaming(request, session_id=session_id):
            if kind == "result":
                result = AgentRunResult.model_validate(payload)
        assert result is not None  # run_streaming always yields a final result
        return result

    def run_streaming(self, request: str, *, session_id: str) -> Iterator[tuple[str, dict]]:
        request = request.strip()
        if not request:
            raise ValueError("request must not be empty")

        has_tables = bool(self._table_store.list_tables())
        plan, plan_retries, plan_error = self.planner.plan(request, has_tables=has_tables)
        planned_agents = {s.agent for s in plan.steps}
        want_data = AgentName.data_analysis in planned_agents and has_tables

        degraded = plan_error is not None
        steps: list[AgentStep] = []

        # 1 + 2. Retrieval and (optionally) data analysis - independent, so run
        # them concurrently when asked. The results are consumed in a fixed
        # order below so the output does not depend on completion order.
        if want_data and self._settings.agent_parallel:
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="orch") as pool:
                fut_retrieval = pool.submit(self._retrieval_step, request)
                fut_data = pool.submit(self._data_step, request)
                retrieval_step, retrieval_result, r_degraded = fut_retrieval.result()
                data_step, data_result, d_degraded = fut_data.result()
        else:
            retrieval_step, retrieval_result, r_degraded = self._retrieval_step(request)
            if want_data:
                data_step, data_result, d_degraded = self._data_step(request)
            else:
                data_step, data_result, d_degraded = None, None, False

        degraded = degraded or r_degraded or d_degraded
        if plan_retries:
            retrieval_step.retries = plan_retries
        steps.append(retrieval_step)
        yield "step", retrieval_step.model_dump(mode="json")
        if data_step is not None:
            steps.append(data_step)
            yield "step", data_step.model_dump(mode="json")

        # 3. Action
        decision, action_step, a_degraded = self._action_step(
            request, retrieval_result, data_result
        )
        degraded = degraded or a_degraded
        steps.append(action_step)
        yield "step", action_step.model_dump(mode="json")

        # 4. Validation (always)
        validation, validation_step = self._validation_step(decision, retrieval_result, data_result)
        steps.append(validation_step)
        yield "step", validation_step.model_dump(mode="json")

        result = AgentRunResult(
            session_id=session_id,
            request=request,
            plan=plan,
            steps=steps,
            decision=decision,
            validation=validation,
            llm_mode=self._llm.mode,
            degraded=degraded or not validation.passed,
        )
        yield "result", result.model_dump(mode="json")

    # -- individual steps -------------------------------------------
    def _retrieval_step(self, request: str) -> tuple[AgentStep, RetrievalResult, bool]:
        try:
            retrieval_result = self.retrieval.gather(request)
            step = AgentStep(
                agent=AgentName.retrieval,
                summary=(
                    f"retrieved {len(retrieval_result.chunks)} chunk(s); "
                    f"top score {retrieval_result.top_score:.2f}; "
                    f"confident={retrieval_result.confident}"
                ),
                retrieved=retrieval_result.chunks,
            )
            return step, retrieval_result, False
        except Exception as exc:  # noqa: BLE001 - recorded, then degrade
            logger.exception("retrieval step failed")
            step = AgentStep(agent=AgentName.retrieval, summary="retrieval failed", error=str(exc))
            return (
                step,
                RetrievalResult(query=request, chunks=[], top_score=0.0, confident=False),
                True,
            )

    def _data_step(self, request: str) -> tuple[AgentStep, DataAnalysisResult | None, bool]:
        try:
            data_result, calls = self.data.analyse(request)
            step = AgentStep(
                agent=AgentName.data_analysis,
                summary=(
                    f"analysed table {data_result.table!r}: "
                    f"{len(data_result.findings)} finding(s), "
                    f"{len(data_result.anomalies)} anomaly(ies)"
                ),
                tool_calls=calls,
            )
            return step, data_result, False
        except Exception as exc:  # noqa: BLE001
            logger.exception("data analysis step failed")
            step = AgentStep(
                agent=AgentName.data_analysis,
                summary="data analysis failed",
                error=str(exc),
            )
            return step, None, True

    def _action_step(self, request, retrieval_result, data_result):
        try:
            decision, tool_calls = self.action.decide(
                request=request, retrieval=retrieval_result, data=data_result
            )
            step = AgentStep(
                agent=AgentName.action,
                summary=(
                    f"produced decision (severity={decision.incident.severity}, "
                    f"confidence={decision.confidence}, "
                    f"{len(decision.remediation_checklist)} checklist item(s))"
                ),
                tool_calls=tool_calls,
            )
            return decision, step, False
        except Exception as exc:  # noqa: BLE001
            logger.exception("action step failed")
            step = AgentStep(agent=AgentName.action, summary="action failed", error=str(exc))
            return None, step, True

    def _validation_step(self, decision, retrieval_result, data_result):
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
            validation = ValidationReport(passed=False, grounded=False, issues=[])
        step = AgentStep(
            agent=AgentName.validation,
            summary=(
                f"passed={validation.passed}, grounded={validation.grounded}, "
                f"{validation.supported_claims}/{validation.checked_claims} claims supported"
            ),
        )
        return validation, step
