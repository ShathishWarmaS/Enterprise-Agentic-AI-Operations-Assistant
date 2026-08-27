"""Schemas for the multi-agent workflow: plans, tool calls, and structured output."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from app.schemas.retrieval import Citation, RetrievedChunk


class AgentName(str, Enum):
    planner = "planner"
    retrieval = "retrieval"
    data_analysis = "data_analysis"
    action = "action"
    validation = "validation"


class PlanStep(BaseModel):
    step: int
    agent: AgentName
    objective: str
    tool: str | None = Field(default=None, description="MCP tool the agent should call, if any")


class Plan(BaseModel):
    request: str
    steps: list[PlanStep]
    rationale: str


class ToolCall(BaseModel):
    tool: str
    arguments: dict
    ok: bool
    result: dict | None = None
    error: str | None = None
    duration_ms: int = 0


class DataFinding(BaseModel):
    metric: str
    value: float | int | str
    unit: str | None = None
    observation: str


class DataAnalysisResult(BaseModel):
    table: str | None = None
    row_count: int = 0
    findings: list[DataFinding] = Field(default_factory=list)
    anomalies: list[str] = Field(default_factory=list)
    missing_fields: list[str] = Field(default_factory=list)


class IncidentSummary(BaseModel):
    title: str
    severity: str = Field(description="low | medium | high | critical")
    summary: str
    impact: str
    likely_cause: str
    evidence: list[Citation]


class ChecklistItem(BaseModel):
    order: int
    action: str
    owner_role: str
    blocking: bool = False


class OperationalDecision(BaseModel):
    """The final structured output the Action agent produces and Validation gates."""

    request: str
    incident: IncidentSummary
    recommended_next_steps: list[str]
    remediation_checklist: list[ChecklistItem]
    citations: list[Citation]
    data_findings: list[DataFinding] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    confidence: str = Field(description="high | medium | low")


class ValidationIssue(BaseModel):
    field: str
    kind: str = Field(description="unsupported_claim | missing_field | low_confidence | schema")
    detail: str


class ValidationReport(BaseModel):
    passed: bool
    grounded: bool
    issues: list[ValidationIssue] = Field(default_factory=list)
    unsupported_sentences: list[str] = Field(default_factory=list)
    checked_claims: int = 0
    supported_claims: int = 0


class AgentStep(BaseModel):
    agent: AgentName
    summary: str
    tool_calls: list[ToolCall] = Field(default_factory=list)
    retrieved: list[RetrievedChunk] = Field(default_factory=list)
    retries: int = 0
    error: str | None = None


class AgentRunResult(BaseModel):
    session_id: str
    request: str
    plan: Plan
    steps: list[AgentStep]
    decision: OperationalDecision | None
    validation: ValidationReport
    llm_mode: str
    degraded: bool = Field(default=False, description="true if a step failed and the run fell back")
