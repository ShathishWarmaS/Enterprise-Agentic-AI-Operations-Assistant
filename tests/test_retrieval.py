"""Vector store + retriever behaviour, including empty-corpus and no-match cases."""

from __future__ import annotations

import os

import numpy as np
import pytest

from app.config import Settings
from app.retrieval.backends.local import DimMismatchError, LocalVectorBackend
from app.retrieval.embeddings import (
    Embedder,
    HashingEmbedder,
    SentenceTransformerEmbedder,
    build_embedder,
    embed,
)
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import VectorStore, build_vector_store
from app.schemas.documents import Chunk, SourceType

_PG_URL = os.environ.get("TEST_DATABASE_URL") or "postgresql+psycopg2://ops:ops@localhost:5432/ops"


def _chunk(cid: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=cid,
        document_id="d",
        filename="d.md",
        source_type=SourceType.markdown,
        ordinal=int(cid[-1]),
        text=text,
        locator="x",
    )


def _store(tmp_dir) -> LocalVectorBackend:
    return LocalVectorBackend(tmp_dir / "v", HashingEmbedder())


# -- embedder ---------------------------------------------------------------
def test_embeddings_are_deterministic_and_normalised():
    v1, v2 = embed("connection pool exhaustion"), embed("connection pool exhaustion")
    assert (v1 == v2).all()
    assert abs(float((v1**2).sum()) - 1.0) < 1e-5


def test_hashing_embedder_dim_and_determinism():
    e = HashingEmbedder()
    assert e.dim == 512
    a = e.embed_batch(["rollback plan", "redis ttl"])
    b = e.embed_batch(["rollback plan", "redis ttl"])
    assert a.shape == (2, 512)
    assert a.dtype == np.float32
    assert np.array_equal(a, b)


def test_build_embedder_defaults_to_hashing():
    e = build_embedder(Settings())
    assert isinstance(e, HashingEmbedder)
    assert isinstance(e, Embedder)


def test_sentence_transformer_embedder_roundtrip():
    pytest.importorskip("sentence_transformers")
    e = SentenceTransformerEmbedder("sentence-transformers/all-MiniLM-L6-v2")
    v = e.embed("hello world")
    assert v.shape == (e.dim,)
    assert abs(float((v**2).sum()) - 1.0) < 1e-4


# -- local backend --------------------------------------------------------
def test_empty_store_returns_nothing(tmp_dir):
    assert _store(tmp_dir).search("anything", 5) == []


def test_add_search_delete_roundtrip(tmp_dir):
    store = _store(tmp_dir)
    store.add([_chunk("c0", "rollback takes ninety seconds"), _chunk("c1", "redis cache ttl")])
    hits = store.search("how long does rollback take", 2)
    assert hits and hits[0].chunk_id == "c0"
    assert store.delete_document("d") == 2
    assert len(store) == 0


def test_vectorstore_alias_still_constructs(tmp_dir):
    assert isinstance(VectorStore(tmp_dir / "v", HashingEmbedder()), LocalVectorBackend)


def test_build_vector_store_defaults_to_local(tmp_dir):
    s = Settings(vector_store_dir=tmp_dir / "v")
    store = build_vector_store(s, HashingEmbedder())
    assert isinstance(store, LocalVectorBackend)
    assert store.backend in {"local:faiss", "local:numpy"}


def test_local_backend_refuses_dim_mismatch(tmp_dir):
    store = _store(tmp_dir)
    store.add([_chunk("c0", "hello")])

    class _Wide(HashingEmbedder):
        @property
        def dim(self) -> int:
            return 384

        def embed(self, text):  # pragma: no cover - not reached
            return np.zeros(384, dtype=np.float32)

    with pytest.raises(DimMismatchError):
        LocalVectorBackend(tmp_dir / "v", _Wide())


# -- retriever ----------------------------------------------------------
def test_retriever_flags_low_confidence(tmp_dir):
    store = _store(tmp_dir)
    store.add([_chunk("c0", "the quick brown fox jumps over the lazy dog")])
    result = Retriever(store, top_k=5, min_score=0.5).retrieve("kubernetes ingress TLS termination")
    assert result.confident is False
    assert result.chunks  # still returns the best effort


def test_retriever_rejects_empty_query(tmp_dir):
    store = _store(tmp_dir)
    with pytest.raises(ValueError):
        Retriever(store, top_k=5, min_score=0.2).retrieve("   ")


# -- pgvector backend (skipped unless a Postgres+pgvector is reachable) ----
def _pgvector_backend(embedder):
    pytest.importorskip("pgvector")
    from sqlalchemy import text

    import app.data.database as db

    try:
        eng = db.create_engine(_PG_URL, future=True)
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:  # pragma: no cover - env dependent
        pytest.skip(f"no pgvector Postgres reachable at {_PG_URL}: {exc}")

    orig = db.engine
    db.engine = eng
    from app.retrieval.backends.pgvector import PgVectorBackend

    try:
        backend = PgVectorBackend(embedder)
        backend.clear()
    except Exception as exc:  # pragma: no cover - env dependent
        db.engine = orig
        pytest.skip(f"pgvector unavailable: {exc}")
    return backend, (lambda: setattr(db, "engine", orig))


def test_pgvector_add_search_delete_roundtrip():
    backend, restore = _pgvector_backend(HashingEmbedder())
    try:
        backend.add(
            [_chunk("c0", "rollback takes ninety seconds"), _chunk("c1", "redis cache ttl")]
        )
        hits = backend.search("how long does rollback take", 2)
        assert hits and hits[0].chunk_id == "c0"
        assert 0.0 <= hits[0].score <= 1.0
        assert backend.stats()["backend"] == "pgvector"
        assert backend.delete_document("d") == 2
        assert len(backend) == 0
    finally:
        backend.clear()
        restore()
