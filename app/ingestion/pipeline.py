"""Pure ingestion transform: bytes on disk -> chunks (+ optional clean frame).

No I/O beyond reading the source file. Storage (vector index, table store, DB)
is the caller's job - see app/services/ingestion_service.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from app.ingestion.chunking import chunk_source
from app.ingestion.cleaning import clean_frame
from app.ingestion.loaders import LoadedSource, load
from app.schemas.documents import Chunk, CleaningReport, SourceType


@dataclass
class IngestionOutput:
    source_type: SourceType
    chunks: list[Chunk]
    cleaning: CleaningReport
    frame: pd.DataFrame | None  # cleaned; present for csv / json-array sources


def run_pipeline(
    *,
    path: Path,
    document_id: str,
    filename: str,
    chunk_size: int,
    chunk_overlap: int,
    ocr_pdf: bool = False,
) -> IngestionOutput:
    source: LoadedSource = load(path, ocr_pdf=ocr_pdf)

    cleaning = CleaningReport()
    if source.frame is not None:
        cleaned, cleaning = clean_frame(source.frame)
        source = LoadedSource(
            source_type=source.source_type,
            text=_frame_preview(cleaned),
            frame=cleaned,
        )
    else:
        cleaning = _text_quality(source.text)

    chunks = chunk_source(
        source=source,
        document_id=document_id,
        filename=filename,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    if not chunks:
        raise ValueError(f"{filename} produced no chunks after cleaning")

    return IngestionOutput(
        source_type=source.source_type,
        chunks=chunks,
        cleaning=cleaning,
        frame=source.frame,
    )


def _frame_preview(frame: pd.DataFrame) -> str:
    return f"columns: {', '.join(map(str, frame.columns))}\n{frame.head(200).to_csv(index=False)}"


def _text_quality(text: str) -> CleaningReport:
    report = CleaningReport(rows_in=1, rows_out=1)
    stripped = text.strip()
    if len(stripped) < 40:
        from app.schemas.documents import CleaningIssue

        report.issues.append(
            CleaningIssue(
                severity="warning",
                location="document",
                message="very short document; retrieval quality will be limited",
            )
        )
    return report
