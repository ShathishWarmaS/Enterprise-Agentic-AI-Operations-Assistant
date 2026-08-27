"""Schemas for vector retrieval and grounded query answers."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.documents import SourceType


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    source_type: SourceType
    locator: str
    text: str
    score: float = Field(description="cosine similarity in [0, 1]")


class Citation(BaseModel):
    """A pointer a reader can follow back to the source."""

    marker: str = Field(description="e.g. [1], used inline in answer text")
    filename: str
    locator: str
    chunk_id: str


class RetrievalResult(BaseModel):
    query: str
    chunks: list[RetrievedChunk]
    # max score across chunks; low values mean the corpus probably lacks an answer
    top_score: float
    confident: bool


class QueryAnswer(BaseModel):
    query: str
    answer: str
    citations: list[Citation]
    supporting_chunks: list[RetrievedChunk]
    confident: bool
    notes: list[str] = Field(default_factory=list)
