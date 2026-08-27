"""Schemas for ingestion: source files, cleaning reports, and chunks."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    pdf = "pdf"
    csv = "csv"
    json = "json"
    markdown = "markdown"
    text = "text"


class DocumentStatus(str, Enum):
    uploaded = "uploaded"
    ingested = "ingested"
    failed = "failed"


class UploadedDocument(BaseModel):
    document_id: str
    filename: str
    source_type: SourceType
    size_bytes: int
    stored_path: str
    status: DocumentStatus = DocumentStatus.uploaded
    created_at: datetime


class CleaningIssue(BaseModel):
    """A single problem found while cleaning/validating a source."""

    severity: str = Field(description="info | warning | error")
    location: str = Field(description="row index, column name, page, or line range")
    message: str


class CleaningReport(BaseModel):
    rows_in: int = 0
    rows_out: int = 0
    dropped_rows: int = 0
    coerced_cells: int = 0
    issues: list[CleaningIssue] = Field(default_factory=list)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    source_type: SourceType
    ordinal: int = Field(description="position of this chunk within its document")
    text: str
    # Free-form provenance: page number, row range, sheet name, JSON path.
    locator: str


class IngestResult(BaseModel):
    document_id: str
    filename: str
    source_type: SourceType
    status: DocumentStatus
    chunks_created: int
    cleaning: CleaningReport
    # Present for CSV/JSON tabular sources so the Data Analysis agent can query them.
    table_registered: bool = False
