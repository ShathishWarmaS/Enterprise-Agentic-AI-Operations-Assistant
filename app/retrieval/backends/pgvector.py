"""Shared Postgres/pgvector vector store.

Unlike `LocalVectorBackend` (an in-process index that only one worker can own),
this backend keeps all vectors in Postgres, so any number of API workers can
read and write the same corpus concurrently. That is the fix for the
multi-worker bottleneck.

Concurrency: every operation runs in its own short transaction against the
app-wide engine. Writes use `INSERT ... ON CONFLICT (chunk_id) DO NOTHING`, so
two workers ingesting overlapping chunks cannot deadlock or duplicate rows.
"""

from __future__ import annotations

import logging

from sqlalchemy import text

from app.data import database as _db
from app.retrieval.backends.base import VectorBackend
from app.retrieval.embeddings import Embedder
from app.schemas.documents import Chunk
from app.schemas.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)


class PgVectorBackend(VectorBackend):
    def __init__(self, embedder: Embedder) -> None:
        try:
            import pgvector.sqlalchemy  # noqa: F401
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                'VECTOR_BACKEND=pgvector needs the `pgvector` extra: pip install -e ".[pgvector]"'
            ) from exc
        self._embedder = embedder
        self._dim = embedder.dim
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with _db.engine.begin() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.execute(text(f"""
                    CREATE TABLE IF NOT EXISTS chunk_embeddings (
                        chunk_id     TEXT PRIMARY KEY,
                        document_id  TEXT NOT NULL,
                        filename     TEXT NOT NULL,
                        source_type  TEXT NOT NULL,
                        locator      TEXT NOT NULL,
                        text         TEXT NOT NULL,
                        embedding    vector({self._dim}) NOT NULL
                    )
                    """))
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS chunk_embeddings_document_id_idx "
                    "ON chunk_embeddings (document_id)"
                )
            )

    @property
    def backend(self) -> str:
        return "pgvector"

    @staticmethod
    def _vec_literal(vec) -> str:
        return "[" + ",".join(f"{float(x):.8f}" for x in vec) + "]"

    def add(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        vectors = self._embedder.embed_batch([c.text for c in chunks])
        rows = [
            {
                "chunk_id": c.chunk_id,
                "document_id": c.document_id,
                "filename": c.filename,
                "source_type": c.source_type.value,
                "locator": c.locator,
                "text": c.text,
                "embedding": self._vec_literal(v),
            }
            for c, v in zip(chunks, vectors)
        ]
        stmt = text("""
            INSERT INTO chunk_embeddings
                (chunk_id, document_id, filename, source_type, locator, text, embedding)
            VALUES
                (:chunk_id, :document_id, :filename, :source_type, :locator, :text, :embedding)
            ON CONFLICT (chunk_id) DO NOTHING
            """)
        with _db.engine.begin() as conn:
            conn.execute(stmt, rows)
        return len(chunks)

    def delete_document(self, document_id: str) -> int:
        with _db.engine.begin() as conn:
            result = conn.execute(
                text("DELETE FROM chunk_embeddings WHERE document_id = :d"),
                {"d": document_id},
            )
        return int(result.rowcount or 0)

    def clear(self) -> None:
        with _db.engine.begin() as conn:
            conn.execute(text("TRUNCATE chunk_embeddings"))

    def __len__(self) -> int:
        with _db.engine.connect() as conn:
            return int(conn.execute(text("SELECT count(*) FROM chunk_embeddings")).scalar_one())

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        q = self._vec_literal(self._embedder.embed(query))
        stmt = text("""
            SELECT chunk_id, document_id, filename, source_type, locator, text,
                   embedding <=> :q AS distance
            FROM chunk_embeddings
            ORDER BY embedding <=> :q ASC
            LIMIT :k
            """)
        with _db.engine.connect() as conn:
            rows = conn.execute(stmt, {"q": q, "k": top_k}).mappings().all()
        results: list[RetrievedChunk] = []
        for r in rows:
            similarity = max(0.0, min(1.0, round(1.0 - float(r["distance"]), 4)))
            results.append(
                RetrievedChunk(
                    chunk_id=r["chunk_id"],
                    document_id=r["document_id"],
                    filename=r["filename"],
                    source_type=r["source_type"],
                    locator=r["locator"],
                    text=r["text"],
                    score=similarity,
                )
            )
        return results

    def stats(self) -> dict:
        with _db.engine.connect() as conn:
            chunks = int(conn.execute(text("SELECT count(*) FROM chunk_embeddings")).scalar_one())
            docs = int(
                conn.execute(
                    text("SELECT count(DISTINCT document_id) FROM chunk_embeddings")
                ).scalar_one()
            )
        return {"chunks": chunks, "documents": docs, "backend": self.backend}
