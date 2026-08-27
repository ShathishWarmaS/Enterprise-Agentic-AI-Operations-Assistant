"""Validation & guardrail agent: gate the decision on grounding and completeness."""

from __future__ import annotations

import re

from app.agents.base import Agent
from app.agents.grounding import is_supported, split_sentences, support_score
from app.prompts import VALIDATION_SYSTEM
from app.schemas.agents import OperationalDecision, ValidationIssue, ValidationReport
from app.schemas.retrieval import RetrievedChunk

_MARKER_RE = re.compile(r"\[\d+\]")


class ValidationAgent(Agent):
    name = "validation"

    def validate(
        self,
        decision: OperationalDecision,
        context_chunks: list[RetrievedChunk],
        *,
        expect_grounding: bool = True,
        data_context: list[str] | None = None,
    ) -> ValidationReport:
        # Data-analysis observations are first-class evidence too, not just docs.
        contexts = [c.text for c in context_chunks] + list(data_context or [])
        issues: list[ValidationIssue] = []

        self._check_required_fields(decision, issues)

        claims = self._claims(decision)
        unsupported: list[str] = []
        supported = 0
        for field, sentence in claims:
            clean = _MARKER_RE.sub("", sentence).strip()
            if not clean:
                continue
            if contexts and is_supported(clean, contexts):
                supported += 1
            else:
                unsupported.append(sentence)
                issues.append(
                    ValidationIssue(
                        field=field,
                        kind="unsupported_claim",
                        detail=(
                            f"not supported by retrieved context "
                            f"(best overlap {support_score(clean, contexts):.2f}): {sentence[:120]}"
                        ),
                    )
                )

        checked = len(claims)
        grounded = bool(contexts) and (supported / checked >= 0.6 if checked else False)

        if expect_grounding and not decision.citations and not data_context:
            grounded = False
            issues.append(
                ValidationIssue(
                    field="citations",
                    kind="missing_field",
                    detail="no citations and no data evidence attached",
                )
            )

        if decision.confidence == "high" and (unsupported or not grounded):
            issues.append(
                ValidationIssue(
                    field="confidence",
                    kind="low_confidence",
                    detail="'high' confidence but claims are unsupported or grounding is weak",
                )
            )

        if self.uses_claude and contexts:
            self._augment_with_claude(decision, context_chunks, issues)

        # Missing/invalid fields always block. Individual unsupported sentences are
        # recorded but do not by themselves fail the run - the overall grounding
        # ratio does. A run is rejected if it is largely ungrounded or if more
        # than a third of its claims are unsupported.
        hard_blocking = [i for i in issues if i.kind in ("missing_field", "schema")]
        too_many_unsupported = checked > 0 and (len(unsupported) / checked) > 0.34
        passed = (
            not hard_blocking and not too_many_unsupported and (grounded or not expect_grounding)
        )

        return ValidationReport(
            passed=passed,
            grounded=grounded,
            issues=issues,
            unsupported_sentences=unsupported,
            checked_claims=checked,
            supported_claims=supported,
        )

    # -- checks ------------------------------------------------------
    @staticmethod
    def _check_required_fields(
        decision: OperationalDecision, issues: list[ValidationIssue]
    ) -> None:
        inc = decision.incident
        required = {
            "incident.title": inc.title,
            "incident.summary": inc.summary,
            "incident.impact": inc.impact,
            "incident.likely_cause": inc.likely_cause,
        }
        for field, value in required.items():
            if not value or not value.strip():
                issues.append(ValidationIssue(field=field, kind="missing_field", detail="empty"))
        if inc.severity not in ("low", "medium", "high", "critical"):
            issues.append(
                ValidationIssue(
                    field="incident.severity",
                    kind="schema",
                    detail=f"invalid severity {inc.severity!r}",
                )
            )
        if not decision.recommended_next_steps:
            issues.append(
                ValidationIssue(
                    field="recommended_next_steps",
                    kind="missing_field",
                    detail="no next steps produced",
                )
            )
        if not decision.remediation_checklist:
            issues.append(
                ValidationIssue(
                    field="remediation_checklist", kind="missing_field", detail="checklist is empty"
                )
            )

    @staticmethod
    def _claims(decision: OperationalDecision) -> list[tuple[str, str]]:
        """Factual assertions that must be grounded.

        We check the incident *analysis* (summary + likely cause), not the
        recommended steps or checklist - those are advice, and the standard
        incident-response spine (declare an incident, postmortem, ...) is
        procedure, not a claim about this system that needs a citation.
        """
        claims: list[tuple[str, str]] = [
            ("incident.summary", s) for s in split_sentences(decision.incident.summary)
        ]
        if decision.incident.likely_cause.strip():
            claims.append(("incident.likely_cause", decision.incident.likely_cause))
        return claims

    def _augment_with_claude(self, decision, context_chunks, issues) -> None:
        context = "\n\n".join(f"[{i}] {c.text}" for i, c in enumerate(context_chunks, start=1))
        report, _, _ = self.with_retry(
            lambda: self.llm.structured(
                system=VALIDATION_SYSTEM,
                user=f"Decision:\n{decision.model_dump_json(indent=2)}\n\nContext:\n{context}",
                model=ValidationReport,
            ),
            on_error="validation-second-opinion",
        )
        if report is None:
            return
        for issue in report.issues:
            # only add new concerns; the deterministic pass owns pass/fail
            if issue.detail not in {i.detail for i in issues}:
                issues.append(issue)
