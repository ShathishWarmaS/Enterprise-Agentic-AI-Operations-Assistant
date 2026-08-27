"""Turn a raw file on disk into text and/or a tabular frame.

Each loader is deliberately small and total: it either returns a `LoadedSource`
or raises `LoaderError` with a specific reason. No loader swallows exceptions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from app.schemas.documents import SourceType

_EXT_TO_TYPE: dict[str, SourceType] = {
    ".pdf": SourceType.pdf,
    ".csv": SourceType.csv,
    ".tsv": SourceType.csv,
    ".json": SourceType.json,
    ".md": SourceType.markdown,
    ".markdown": SourceType.markdown,
    ".txt": SourceType.text,
    ".log": SourceType.text,
}


class LoaderError(ValueError):
    """Raised when a file cannot be parsed as its declared type."""


@dataclass
class LoadedSource:
    source_type: SourceType
    # Plain text view of the document (always populated; tabular sources get a
    # CSV-ish rendering so they are still searchable by the retrieval agent).
    text: str
    # Structured view, present only for CSV and JSON-array sources.
    frame: pd.DataFrame | None = None
    # Per-segment locators aligned with paragraph splits of `text`
    # (e.g. "page 3"). Optional; chunking falls back to char offsets.
    segments: list[tuple[str, str]] = field(default_factory=list)


def detect_source_type(path: Path) -> SourceType:
    try:
        return _EXT_TO_TYPE[path.suffix.lower()]
    except KeyError:
        raise LoaderError(
            f"unsupported file extension {path.suffix!r}; "
            f"supported: {sorted(set(_EXT_TO_TYPE))}"
        ) from None


def load(path: Path) -> LoadedSource:
    if not path.exists():
        raise LoaderError(f"file not found: {path}")
    if path.stat().st_size == 0:
        raise LoaderError(f"file is empty: {path.name}")

    source_type = detect_source_type(path)
    loader = {
        SourceType.pdf: _load_pdf,
        SourceType.csv: _load_csv,
        SourceType.json: _load_json,
        SourceType.markdown: _load_text,
        SourceType.text: _load_text,
    }[source_type]
    return loader(path)


def _load_text(path: Path) -> LoadedSource:
    raw = path.read_text(encoding="utf-8", errors="replace").strip()
    if not raw:
        raise LoaderError(f"{path.name} contains no readable text")
    stype = detect_source_type(path)
    # No segments: let the chunker apply its section/paragraph splitter, which
    # handles headings and bullet lists better than a fixed-width guess.
    return LoadedSource(source_type=stype, text=raw)


def _load_pdf(path: Path) -> LoadedSource:
    try:
        import pymupdf
    except ImportError as exc:  # pragma: no cover - dependency guaranteed by pyproject
        raise LoaderError("PyMuPDF (pymupdf) is required to read PDF files") from exc

    parts: list[str] = []
    segments: list[tuple[str, str]] = []
    with pymupdf.open(path) as doc:
        if doc.page_count == 0:
            raise LoaderError(f"{path.name} has no pages")
        for page_index in range(doc.page_count):
            page_text = doc.load_page(page_index).get_text("text").strip()
            if page_text:
                locator = f"page {page_index + 1}"
                parts.append(page_text)
                segments.append((locator, page_text))
    if not parts:
        raise LoaderError(f"{path.name} has pages but no extractable text (scanned image?)")
    return LoadedSource(source_type=SourceType.pdf, text="\n\n".join(parts), segments=segments)


def _load_csv(path: Path) -> LoadedSource:
    sep = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        frame = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=True, skip_blank_lines=True)
    except pd.errors.EmptyDataError as exc:
        raise LoaderError(f"{path.name} has no columns to parse") from exc
    except pd.errors.ParserError:
        # Ragged rows (unquoted delimiters in a free-text column are common in
        # real exports). Recover by capping splits at the header width and
        # folding any overflow back into the last column.
        frame = _read_ragged_csv(path, sep)
    if frame.empty:
        raise LoaderError(f"{path.name} parsed to zero rows")
    return LoadedSource(
        source_type=SourceType.csv,
        text=_frame_to_text(frame),
        frame=frame,
    )


def _load_json(path: Path) -> LoadedSource:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LoaderError(f"{path.name} is not valid JSON: {exc}") from exc

    # A list of flat objects becomes a table; anything else stays as pretty text.
    if isinstance(data, list) and data and all(isinstance(row, dict) for row in data):
        frame = pd.json_normalize(data).astype(str)
        return LoadedSource(
            source_type=SourceType.json,
            text=_frame_to_text(frame),
            frame=frame,
        )
    pretty = json.dumps(data, indent=2, ensure_ascii=False)
    if not pretty.strip() or pretty.strip() in {"{}", "[]", "null"}:
        raise LoaderError(f"{path.name} contains no usable data")
    return LoadedSource(source_type=SourceType.json, text=pretty)


def _read_ragged_csv(path: Path, sep: str) -> pd.DataFrame:
    import csv
    import io

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    reader = csv.reader(io.StringIO("\n".join(lines)), delimiter=sep)
    rows = [r for r in reader if any(cell.strip() for cell in r)]
    if not rows:
        raise LoaderError(f"{path.name} parsed to zero rows")
    header = rows[0]
    width = len(header)
    fixed: list[list[str]] = []
    for row in rows[1:]:
        if len(row) > width:
            row = row[: width - 1] + [sep.join(row[width - 1 :])]
        elif len(row) < width:
            row = row + [""] * (width - len(row))
        fixed.append(row)
    return pd.DataFrame(fixed, columns=header).astype("string")


def _frame_to_text(frame: pd.DataFrame) -> str:
    header = f"columns: {', '.join(map(str, frame.columns))}"
    body = frame.head(200).to_csv(index=False).strip()
    return f"{header}\n{body}"
