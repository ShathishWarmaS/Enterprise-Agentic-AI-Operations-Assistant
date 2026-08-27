"""The evaluation harness itself: real scores, bounded, and reproducible."""

from __future__ import annotations

import pytest

from app.services.evaluation import load_cases, run_evaluation


def test_eval_runs_on_sample_dataset(seeded_container):
    summary = run_evaluation(seeded_container, load_cases())
    assert summary.cases == len(load_cases())
    for field in (
        "retrieval_relevance",
        "citation_presence",
        "tool_selection_accuracy",
        "structured_output_validity",
        "groundedness",
        "missing_info_handling",
        "response_consistency",
        "pass_rate",
    ):
        value = getattr(summary, field)
        assert 0.0 <= value <= 1.0, f"{field} out of range: {value}"


def test_eval_is_deterministic_in_mock_mode(seeded_container):
    cases = load_cases()
    a = run_evaluation(seeded_container, cases)
    b = run_evaluation(seeded_container, cases)
    assert a.pass_rate == b.pass_rate
    assert a.response_consistency == 1.0


def test_eval_requires_cases(seeded_container):
    with pytest.raises(ValueError):
        run_evaluation(seeded_container, [])
