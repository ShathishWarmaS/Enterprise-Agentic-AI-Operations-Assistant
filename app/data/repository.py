"""Data-access helpers. Thin functions over SQLAlchemy sessions, no ORM leakage
above this layer beyond plain dicts / pydantic models.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.data.models import DocumentRow, JobRow, SessionRow
from app.schemas.documents import (
    DocumentStatus,
    IngestResult,
    SourceType,
    UploadedDocument,
)


def save_uploaded_document(session: Session, doc: UploadedDocument) -> None:
    session.add(
        DocumentRow(
            id=doc.document_id,
            filename=doc.filename,
            source_type=doc.source_type.value,
            size_bytes=doc.size_bytes,
            stored_path=doc.stored_path,
            status=doc.status.value,
        )
    )


def get_document(session: Session, document_id: str) -> DocumentRow | None:
    return session.get(DocumentRow, document_id)


def list_documents(session: Session) -> list[DocumentRow]:
    return list(session.scalars(select(DocumentRow).order_by(DocumentRow.created_at.desc())))


def mark_document_ingested(session: Session, result: IngestResult, table_name: str | None) -> None:
    row = session.get(DocumentRow, result.document_id)
    if row is None:
        raise KeyError(f"unknown document_id {result.document_id!r}")
    row.status = result.status.value
    row.chunks = result.chunks_created
    row.table_name = table_name
    row.cleaning_report = result.cleaning.model_dump(mode="json")


def mark_document_failed(session: Session, document_id: str, reason: str) -> None:
    row = session.get(DocumentRow, document_id)
    if row is not None:
        row.status = DocumentStatus.failed.value
        row.cleaning_report = {"error": reason}


def save_session(
    session: Session,
    *,
    session_id: str,
    kind: str,
    request: str,
    llm_mode: str,
    response: dict,
) -> None:
    session.add(
        SessionRow(
            id=session_id,
            kind=kind,
            request=request,
            llm_mode=llm_mode,
            response=response,
        )
    )


def get_session_record(session: Session, session_id: str) -> SessionRow | None:
    return session.get(SessionRow, session_id)


def create_job(session: Session, *, job_id: str, kind: str, payload: dict) -> JobRow:
    row = JobRow(id=job_id, kind=kind, status="queued", payload=payload)
    session.add(row)
    return row


def get_job(session: Session, job_id: str) -> JobRow | None:
    return session.get(JobRow, job_id)


def update_job_status(
    session: Session,
    job_id: str,
    status: str,
    result: dict | None = None,
    error: str | None = None,
) -> None:
    row = session.get(JobRow, job_id)
    if row is None:
        raise KeyError(f"unknown job_id {job_id!r}")
    row.status = status
    if result is not None:
        row.result = result
    if error is not None:
        row.error = error
    row.updated_at = datetime.now(UTC)


def document_to_schema(row: DocumentRow) -> UploadedDocument:
    return UploadedDocument(
        document_id=row.id,
        filename=row.filename,
        source_type=SourceType(row.source_type),
        size_bytes=row.size_bytes,
        stored_path=row.stored_path,
        status=DocumentStatus(row.status),
        created_at=row.created_at or datetime.now(UTC),
    )
