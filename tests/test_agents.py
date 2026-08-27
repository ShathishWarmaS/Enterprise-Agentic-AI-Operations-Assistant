"""Agent behaviour: planning, grounding checks, and orchestrator degradation."""

from __future__ import annotations

from app.agents.grounding import is_supported, split_sentences
from app.schemas.agents import (
    AgentName,
    ChecklistItem,
    Citation,
    IncidentSummary,
    OperationalDecision,
)


def test_planner_data_request_includes_data_agent(seeded_container):
    plan, _, _ = seeded_container.orchestrator().planner.plan(
        "how many incidents per service in the incidents table?", has_tables=True
    )
    agents = [s.agent for s in plan.steps]
    assert AgentName.data_analysis in agents
    assert agents[-1] == AgentName.validation
    assert AgentName.action in agents


def test_planner_always_ends_with_action_then_validation(seeded_container):
    plan, _, _ = seeded_container.orchestrator().planner.plan(
        "why did checkout break?", has_tables=False
    )
    assert [s.agent for s in plan.steps][-2:] == [AgentName.action, AgentName.validation]


def test_grounding_rejects_fabricated_claim():
    context = ["payment-service holds a database connection pool of 40 connections per pod"]
    assert is_supported("the connection pool has 40 connections", context)
    assert not is_supported("the service was migrated to GraphQL last quarter", context)


def test_split_sentences_joins_wrapped_lines():
    text = "The pool was\nexhausted within\nfour minutes.\n\n- roll back first"
    sentences = split_sentences(text)
    assert "The pool was exhausted within four minutes." in sentences
    assert "roll back first" in sentences


def test_validation_flags_unsupported_and_missing_fields(seeded_container):
    decision = OperationalDecision(
        request="r",
        incident=IncidentSummary(
            title="T",
            severity="high",
            summary="The database was moved to Aurora and the team switched to Kafka.",
            impact="",
            likely_cause="Aliens.",
            evidence=[],
        ),
        recommended_next_steps=["do the thing"],
        remediation_checklist=[ChecklistItem(order=1, action="a", owner_role="o")],
        citations=[
            Citation(marker="[1]", filename="runbook_payment_service.md", locator="x", chunk_id="c")
        ],
        confidence="high",
    )
    context = seeded_container.retriever.retrieve("payment-service 5xx").chunks
    report = seeded_container.orchestrator().validation.validate(decision, context)
    assert report.passed is False
    assert any(i.kind == "missing_field" for i in report.issues)  # empty impact
    assert report.unsupported_sentences


def test_orchestrator_end_to_end_mock(seeded_container):
    run = seeded_container.orchestrator().run(
        "payment-service is throwing 5xx after a deploy 10 minutes ago; give me next steps",
        session_id="t-e2e",
    )
    assert run.decision is not None
    assert run.decision.remediation_checklist
    assert run.decision.citations
    assert run.validation.passed is True
    assert run.llm_mode == "mock"


def test_orchestrator_handles_unanswerable_gracefully(seeded_container):
    run = seeded_container.orchestrator().run(
        "what is the airspeed velocity of an unladen swallow?", session_id="t-nonsense"
    )
    # It still completes and produces a decision, but validation should not
    # bless it as grounded.
    assert run.decision is not None
    assert run.validation.grounded is False or run.decision.confidence == "low"
