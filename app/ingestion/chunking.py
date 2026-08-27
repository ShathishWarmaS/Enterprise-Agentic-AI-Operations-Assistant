"""Split loaded sources into overlapping chunks for embedding.

Text is split on paragraph boundaries and packed up to `chunk_size` characters
with `chunk_overlap` carried between chunks. Tabular sources get a schema
summary chunk plus row-window chunks so both "what columns exist" and "which
rows say X" style questions can retrieve something useful.
"""

from __future__ import annotations

import pandas as pd

from app.ingestion.loaders import LoadedSource
from app.schemas.documents import Chunk


def chunk_source(
    *,
    source: LoadedSource,
    document_id: str,
    filename: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    if source.frame is not None:
        pieces = list(_chunk_frame(source.frame, chunk_size))
    elif source.segments:
        pieces = list(_chunk_segments(source.segments, chunk_size, chunk_overlap))
    else:
        pieces = list(_chunk_plain(source.text, chunk_size, chunk_overlap))

    return [
        Chunk(
            chunk_id=f"{document_id}:{ordinal}",
            document_id=document_id,
            filename=filename,
            source_type=source.source_type,
            ordinal=ordinal,
            text=text,
            locator=locator,
        )
        for ordinal, (locator, text) in enumerate(pieces)
        if text.strip()
    ]


def _pack(blocks: list[tuple[str, str]], chunk_size: int, overlap: int):
    """Greedily pack (locator, text) blocks into <= chunk_size windows."""
    buf: list[str] = []
    buf_len = 0
    first_locator: str | None = None
    last_locator: str | None = None

    for locator, text in blocks:
        if first_locator is None:
            first_locator = locator
        if buf_len + len(text) > chunk_size and buf:
            yield _span(first_locator, last_locator), "\n\n".join(buf)
            carry = "\n\n".join(buf)[-overlap:] if overlap else ""
            buf = [carry] if carry else []
            buf_len = len(carry)
            first_locator = locator
        buf.append(text)
        buf_len += len(text)
        last_locator = locator

    if buf and "".join(buf).strip():
        yield _span(first_locator, last_locator), "\n\n".join(buf)


def _span(first: str | None, last: str | None) -> str:
    if not first:
        return "unknown"
    return first if (last is None or last == first) else f"{first}–{last}"


def _chunk_segments(segments: list[tuple[str, str]], chunk_size: int, overlap: int):
    # Expand each segment (e.g. a PDF page) into finer blocks but keep the
    # segment locator so citations still point at the page.
    expanded: list[tuple[str, str]] = []
    for locator, body in segments:
        parts = _split_sections(body) or [(locator, body)]
        for _, part in parts:
            expanded.append((locator, part))
    yield from _pack(expanded, chunk_size, overlap)


def _chunk_plain(text: str, chunk_size: int, overlap: int):
    sections = _split_sections(text) or [("para 0", text)]
    yield from _pack(sections, chunk_size, overlap)


def _split_sections(text: str) -> list[tuple[str, str]]:
    """Split on Markdown headings first, then on blank lines within a section.

    Keeps a heading attached to the prose that follows it and keeps bullet
    lists together, which produces far cleaner sentences downstream than a
    naive ``split("\\n\\n")``.
    """
    lines = text.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_heading = "intro"
    buf: list[str] = []

    def flush() -> None:
        if buf and "".join(buf).strip():
            sections.append((current_heading, buf.copy()))
        buf.clear()

    for line in lines:
        if line.lstrip().startswith("#"):
            flush()
            current_heading = line.lstrip("#").strip().lower()[:40] or "section"
        else:
            buf.append(line)
    flush()

    out: list[tuple[str, str]] = []
    for heading, body_lines in sections:
        body = "\n".join(body_lines).strip()
        for i, para in enumerate(p.strip() for p in body.split("\n\n") if p.strip()):
            label = heading if i == 0 else f"{heading} (cont.)"
            out.append((label, para))
    return out


def _chunk_frame(frame: pd.DataFrame, chunk_size: int):
    summary = _schema_summary(frame)
    yield "schema", summary

    rows_per_chunk = max(5, chunk_size // 120)
    for start in range(0, len(frame), rows_per_chunk):
        window = frame.iloc[start : start + rows_per_chunk]
        end = start + len(window) - 1
        yield f"rows {start}–{end}", window.to_csv(index=False).strip()


def _schema_summary(frame: pd.DataFrame) -> str:
    lines = [f"table with {len(frame)} rows and {len(frame.columns)} columns"]
    for col in frame.columns:
        dtype = str(frame[col].dtype)
        non_null = int(frame[col].notna().sum())
        sample = frame[col].dropna().astype(str).head(3).tolist()
        lines.append(f"- {col} ({dtype}): {non_null} non-null; e.g. {sample}")
    return "\n".join(lines)
