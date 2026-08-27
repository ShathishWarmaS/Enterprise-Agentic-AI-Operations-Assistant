"""Request/response bodies for the HTTP API (thin wrappers over domain schemas)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.agents import AgentRunResult
from app.schemas.documents import IngestResult, UploadedDocument
from app.schemas.evaluation import EvalSummary
from app.schemas.retrieval import QueryAnswer


class HealthResponse(BaseModel):
    status: str = "ok"
    llm_mode: str
    vector_backend: str
    session_backend: str
    documents: int
    chunks: int
    tables: list[str]


class IngestRequest(BaseModel):
    document_id: str | None = Field(default=None, description="omit with ingest_all=true")
    ingest_all: bool = False


class IngestResponse(BaseModel):
    results: list[IngestResult]


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class QueryResponse(BaseModel):
    session_id: str
    answer: QueryAnswer


class AgentRunRequest(BaseModel):
    request: str = Field(min_length=1, max_length=2000)


class AgentRunResponse(BaseModel):
    session_id: str
    result: AgentRunResult


class EvaluateRequest(BaseModel):
    cases_path: str | None = None


class SessionResponse(BaseModel):
    session_id: str
    kind: str
    request: str
    llm_mode: str
    created_at: str
    response: dict


class UploadResponse(BaseModel):
    document: UploadedDocument
    next: str = "POST /documents/ingest with this document_id"


__all__ = [
    "AgentRunRequest",
    "AgentRunResponse",
    "EvaluateRequest",
    "EvalSummary",
    "HealthResponse",
    "IngestRequest",
    "IngestResponse",
    "QueryRequest",
    "QueryResponse",
    "SessionResponse",
    "UploadResponse",
]
