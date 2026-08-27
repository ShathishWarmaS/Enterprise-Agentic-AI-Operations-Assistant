"""Ingestion: loaders, cleaning, chunking - especially the failure paths."""

from __future__ import annotations

import pytest

from app.ingestion.chunking import chunk_source
from app.ingestion.cleaning import clean_frame
from app.ingestion.loaders import LoadedSource, LoaderError, load
from app.schemas.documents import SourceType


def test_missing_file_raises(tmp_dir):
    with pytest.raises(LoaderError, match="not found"):
        load(tmp_dir / "nope.txt")


def test_empty_file_raises(tmp_dir):
    p = tmp_dir / "empty.txt"
    p.write_text("")
    with pytest.raises(LoaderError, match="empty"):
        load(p)


def test_unsupported_extension(tmp_dir):
    p = tmp_dir / "thing.xyz"
    p.write_text("data")
    with pytest.raises(LoaderError, match="unsupported file extension"):
        load(p)


def test_malformed_json(tmp_dir):
    p = tmp_dir / "bad.json"
    p.write_text('{"a": 1,,}')
    with pytest.raises(LoaderError, match="not valid JSON"):
        load(p)


def test_ragged_csv_is_recovered(tmp_dir):
    p = tmp_dir / "ragged.csv"
    p.write_text("id,note\n1,hello\n2,a note, with a comma\n3,fine\n")
    source = load(p)
    assert source.frame is not None
    assert len(source.frame) == 3
    assert "with a comma" in source.frame.iloc[1]["note"]


def test_cleaning_coerces_and_flags_nulls():
    import pandas as pd

    frame = pd.DataFrame(
        {
            "Amount ($)": ["1,000", "2,500", "oops", "3,000"],
            "Empty": [None, None, None, None],
            "Name": [" a ", "b", "b", None],
        }
    )
    cleaned, report = clean_frame(frame)
    assert "amount" in cleaned.columns
    assert pd.api.types.is_numeric_dtype(cleaned["amount"])
    assert "empty" not in cleaned.columns  # entirely-null column dropped
    assert report.coerced_cells >= 1
    assert any(i.severity == "error" for i in report.issues) or any(
        "missing" in i.message for i in report.issues
    )


def test_chunking_produces_ordered_chunks():
    body = "# Title\n\nFirst paragraph on databases.\n\n## Section\n\nSecond paragraph on caches."
    source = LoadedSource(source_type=SourceType.text, text=body)
    chunks = chunk_source(
        source=source, document_id="d1", filename="f.md", chunk_size=200, chunk_overlap=20
    )
    assert chunks
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    assert all(c.document_id == "d1" for c in chunks)
