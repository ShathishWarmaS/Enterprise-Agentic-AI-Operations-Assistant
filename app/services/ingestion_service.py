"""Coordinate ingestion: run the pipeline, then persist to the three stores."""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import UTC, datetime
from pathlib import Path

from app.data import repository
from app.data.database import session_scope
from app.data.tables import TableStore
from app.ingestion.loaders import LoaderError, detect_source_type
from app.ingestion.pipeline import run_pipeline
from app.retrieval.vector_store import VectorStore
from app.schemas.documents import (
    DocumentStatus,
    IngestResult,
    UploadedDocument,
)

logger = logging.getLogger(__name__)


class IngestionError(RuntimeError):
    pass


class IngestionService:
    def __init__(self, *, settings, vector_store: VectorStore, table_store: TableStore) -> None:
        self._settings = settings
        self._vectors = vector_store
        self._tables = table_store

    def save_upload(self, *, filename: str, content: bytes) -> UploadedDocument:
        if not content:
            raise IngestionError("uploaded file is empty")
        safe_name = _safe_filename(filename)
        try:
            source_type = detect_source_type(Path(safe_name))
        except LoaderError as exc:
            raise IngestionError(str(exc)) from exc

        document_id = hashlib.sha256(content).hexdigest()[:16]
        dest = self._settings.upload_dir / f"{document_id}_{safe_name}"
        dest.write_bytes(content)

        doc = UploadedDocument(
            document_id=document_id,
            filename=safe_name,
            source_type=source_type,
            size_bytes=len(content),
            stored_path=str(dest),
            status=DocumentStatus.uploaded,
            created_at=datetime.now(UTC),
        )
        with session_scope() as session:
            if repository.get_document(session, document_id) is None:
                repository.save_uploaded_document(session, doc)
        return doc

    def ingest(self, document_id: str) -> IngestResult:
        with session_scope() as session:
            row = repository.get_document(session, document_id)
            if row is None:
                raise IngestionError(f"unknown document_id {document_id!r}")
            path = Path(row.stored_path)
            filename = row.filename

        # remove any previous chunks for this doc so re-ingest is idempotent
        self._vectors.delete_document(document_id)

        try:
            output = run_pipeline(
                path=path,
                document_id=document_id,
                filename=filename,
                chunk_size=self._settings.chunk_size,
                chunk_overlap=self._settings.chunk_overlap,
                ocr_pdf=self._settings.pdf_ocr_fallback,
            )
        except (LoaderError, ValueError) as exc:
            with session_scope() as session:
                repository.mark_document_failed(session, document_id, str(exc))
            raise IngestionError(f"ingest failed for {filename}: {exc}") from exc

        added = self._vectors.add(output.chunks)

        table_name: str | None = None
        if output.frame is not None and not output.frame.empty:
            table_name = self._tables.register(_table_name(filename), output.frame)

        result = IngestResult(
            document_id=document_id,
            filename=filename,
            source_type=output.source_type,
            status=DocumentStatus.ingested,
            chunks_created=added,
            cleaning=output.cleaning,
            table_registered=table_name is not None,
        )
        with session_scope() as session:
            repository.mark_document_ingested(session, result, table_name)
        logger.info("ingested %s: %d chunks, table=%s", filename, added, table_name)
        return result


def _safe_filename(name: str) -> str:
    base = Path(name).name.strip() or "upload"
    return re.sub(r"[^A-Za-z0-9._-]+", "_", base)


def _table_name(filename: str) -> str:
    stem = Path(filename).stem.lower()
    return re.sub(r"[^a-z0-9_]+", "_", stem).strip("_") or "table"
