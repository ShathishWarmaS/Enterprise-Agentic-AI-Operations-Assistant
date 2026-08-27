"""Schemas for the offline evaluation harness."""

from __future__ import annotations

from pydantic import BaseModel, Field


class EvalCase(BaseModel):
    id: str
    request: str
    kind: str = Field(description="query | agent")
    # Filenames that a correct answer must cite at least one of.
    expected_sources: list[str] = Field(default_factory=list)
    # Substrings expected to appear in a grounded answer (case-insensitive).
    expected_answer_contains: list[str] = Field(default_factory=list)
    # Tools a correct plan is expected to use.
    expected_tools: list[str] = Field(default_factory=list)
    # True when the corpus cannot answer and the system should say so.
    unanswerable: bool = False


class EvalCaseResult(BaseModel):
    id: str
    retrieval_relevance: float
    citation_present: bool
    tool_selection_accuracy: float
    structured_output_valid: bool
    grounded: bool
    handled_missing_info: bool
    passed: bool
    detail: str = ""


class EvalSummary(BaseModel):
    cases: int
    retrieval_relevance: float
    citation_presence: float
    tool_selection_accuracy: float
    structured_output_validity: float
    groundedness: float
    missing_info_handling: float
    response_consistency: float
    pass_rate: float
    results: list[EvalCaseResult]
