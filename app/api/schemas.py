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
    async_: bool = Field(
        default=False,
        alias="async",
        description="run ingestion on a background worker; poll GET /jobs/{job_id}",
    )

    model_config = {"populate_by_name": True}


class IngestResponse(BaseModel):
    results: list[IngestResult]


class IngestJobRef(BaseModel):
    document_id: str
    job_id: str


class IngestAcceptedResponse(BaseModel):
    jobs: list[IngestJobRef]


class JobResponse(BaseModel):
    job_id: str
    kind: str
    status: str
    result: dict | None = None
    error: str | None = None
    created_at: str
    updated_at: str


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
    "IngestAcceptedResponse",
    "IngestJobRef",
    "IngestRequest",
    "IngestResponse",
    "JobResponse",
    "QueryRequest",
    "QueryResponse",
    "SessionResponse",
    "UploadResponse",
]
