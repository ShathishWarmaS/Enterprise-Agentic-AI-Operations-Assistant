"""Vector-store factory + backwards-compatible aliases.

The real implementations live in `app.retrieval.backends`. This module keeps a
small factory (`build_vector_store`) and the historical `VectorStore` name
(now an alias of `LocalVectorBackend`) so older imports keep working.
"""

from __future__ import annotations

from app.config import Settings
from app.config import VectorBackend as VectorBackendKind
from app.retrieval.backends.base import VectorBackend
from app.retrieval.backends.local import LocalVectorBackend
from app.retrieval.embeddings import Embedder

# Backwards-compatible alias (kept to reduce churn in callers/tests).
VectorStore = LocalVectorBackend

__all__ = ["VectorBackend", "LocalVectorBackend", "VectorStore", "build_vector_store"]


def build_vector_store(settings: Settings, embedder: Embedder) -> VectorBackend:
    if settings.vector_backend is VectorBackendKind.pgvector:
        from app.retrieval.backends.pgvector import PgVectorBackend

        return PgVectorBackend(embedder)
    return LocalVectorBackend(settings.vector_store_dir, embedder)
