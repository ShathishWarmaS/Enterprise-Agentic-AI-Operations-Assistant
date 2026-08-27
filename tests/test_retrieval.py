"""Vector store + retriever behaviour, including empty-corpus and no-match cases."""

from __future__ import annotations

import pytest

from app.retrieval.embeddings import embed
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import VectorStore
from app.schemas.documents import Chunk, SourceType


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


def test_embeddings_are_deterministic_and_normalised():
    v1, v2 = embed("connection pool exhaustion"), embed("connection pool exhaustion")
    assert (v1 == v2).all()
    assert abs(float((v1**2).sum()) - 1.0) < 1e-5


def test_empty_store_returns_nothing(tmp_dir):
    store = VectorStore(tmp_dir / "v")
    assert store.search("anything", 5) == []


def test_add_search_delete_roundtrip(tmp_dir):
    store = VectorStore(tmp_dir / "v")
    store.add([_chunk("c0", "rollback takes ninety seconds"), _chunk("c1", "redis cache ttl")])
    hits = store.search("how long does rollback take", 2)
    assert hits and hits[0].chunk_id == "c0"
    assert store.delete_document("d") == 2
    assert len(store) == 0


def test_retriever_flags_low_confidence(tmp_dir):
    store = VectorStore(tmp_dir / "v")
    store.add([_chunk("c0", "the quick brown fox jumps over the lazy dog")])
    result = Retriever(store, top_k=5, min_score=0.5).retrieve("kubernetes ingress TLS termination")
    assert result.confident is False
    assert result.chunks  # still returns the best effort


def test_retriever_rejects_empty_query(tmp_dir):
    store = VectorStore(tmp_dir / "v")
    with pytest.raises(ValueError):
        Retriever(store, top_k=5, min_score=0.2).retrieve("   ")
