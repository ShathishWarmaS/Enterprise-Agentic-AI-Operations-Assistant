"""Ingest everything in sample_data/ into the local stores.

    python scripts/seed.py [--dir sample_data]

Idempotent: re-running re-ingests each file (chunks are replaced, not duplicated).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.data.database import init_db
from app.services.container import get_container
from app.services.ingestion_service import IngestionError

SKIP = {".json"}  # eval_cases.json is not a document
DOC_SUFFIXES = {".pdf", ".csv", ".tsv", ".md", ".markdown", ".txt", ".log", ".json"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="sample_data", type=Path)
    args = parser.parse_args()

    init_db()
    container = get_container()

    files = sorted(
        p
        for p in args.dir.iterdir()
        if p.is_file() and p.suffix.lower() in DOC_SUFFIXES and p.name != "eval_cases.json"
    )
    if not files:
        print(f"no ingestable files in {args.dir}")
        return 1

    failures = 0
    for path in files:
        try:
            doc = container.ingestion.save_upload(filename=path.name, content=path.read_bytes())
            result = container.ingestion.ingest(doc.document_id)
            table = " (table registered)" if result.table_registered else ""
            issues = result.cleaning.error_count
            print(
                f"  {path.name}: {result.chunks_created} chunks{table}"
                + (f", {issues} data errors" if issues else "")
            )
        except IngestionError as exc:
            failures += 1
            print(f"  {path.name}: FAILED - {exc}", file=sys.stderr)

    stats = container.vector_store.stats()
    print(
        f"\nvector store: {stats['chunks']} chunks from {stats['documents']} documents "
        f"({stats['backend']} backend)"
    )
    print(f"tables: {container.table_store.list_tables()}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
