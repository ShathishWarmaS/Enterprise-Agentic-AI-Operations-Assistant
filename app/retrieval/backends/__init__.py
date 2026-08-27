"""Pluggable vector-store backends.

`LocalVectorBackend` is an in-process FAISS/numpy index (single-worker).
`PgVectorBackend` is a shared Postgres/pgvector store, safe for concurrent
writers across multiple workers. Choose with `VECTOR_BACKEND`; the
`app.retrieval.vector_store.build_vector_store` factory returns the right one.
"""

from __future__ import annotations

from app.retrieval.backends.base import VectorBackend
from app.retrieval.backends.local import LocalVectorBackend

__all__ = ["VectorBackend", "LocalVectorBackend", "PgVectorBackend"]


def __getattr__(name: str):  # pragma: no cover - thin lazy import
    if name == "PgVectorBackend":
        from app.retrieval.backends.pgvector import PgVectorBackend

        return PgVectorBackend
    raise AttributeError(name)
