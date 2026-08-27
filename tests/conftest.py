"""Test fixtures.

Every test run gets its own storage directory and SQLite file (set via env
before app modules import), and a container seeded with the sample corpus.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

_TMP = Path(tempfile.mkdtemp(prefix="eai-tests-"))
# Force the offline test profile...
os.environ.update(LLM_MODE="mock", REDIS_URL="", LOG_LEVEL="WARNING")
# ...but let CI (or a developer) point storage at a real Postgres by pre-setting
# these - the pgvector integration job does exactly that.
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_TMP / 'test.sqlite3'}")
os.environ.setdefault("VECTOR_STORE_DIR", str(_TMP / "vector"))
os.environ.setdefault("UPLOAD_DIR", str(_TMP / "uploads"))

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = REPO_ROOT / "sample_data"


@pytest.fixture(scope="session")
def seeded_container():
    from app.data.database import init_db
    from app.services.container import get_container

    init_db()
    container = get_container()
    for path in sorted(SAMPLE_DIR.iterdir()):
        if (
            path.suffix.lower() in {".md", ".txt", ".csv", ".pdf", ".json"}
            and path.name != "eval_cases.json"
        ):
            doc = container.ingestion.save_upload(filename=path.name, content=path.read_bytes())
            container.ingestion.ingest(doc.document_id)
    return container


@pytest.fixture(scope="session")
def client(seeded_container):
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def tmp_dir() -> Path:
    d = Path(tempfile.mkdtemp(prefix="eai-case-", dir=_TMP))
    return d
