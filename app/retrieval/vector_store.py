"""Persistent vector index over ingested chunks.

Uses FAISS (`IndexFlatIP` on L2-normalised vectors == cosine similarity) when it
is importable, and falls back to a pure-numpy brute-force search otherwise so
the test suite and `import` checks work even if the wheel is unavailable on a
given platform. Chunk metadata is stored alongside the vectors in a JSON
sidecar; positions in the index line up with positions in that list.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

import numpy as np

from app.retrieval.embeddings import EMBED_DIM, embed, embed_batch
from app.schemas.documents import Chunk
from app.schemas.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)

try:  # pragma: no cover - exercised implicitly
    import faiss

    _HAVE_FAISS = True
except ImportError:  # pragma: no cover
    faiss = None  # type: ignore[assignment]
    _HAVE_FAISS = False


class VectorStore:
    def __init__(self, directory: Path) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._vectors_path = self._dir / "vectors.npy"
        self._meta_path = self._dir / "chunks.json"
        self._lock = threading.Lock()
        self._meta: list[dict] = []
        self._matrix = np.zeros((0, EMBED_DIM), dtype=np.float32)
        self._index = faiss.IndexFlatIP(EMBED_DIM) if _HAVE_FAISS else None
        self._load()

    # -- persistence -----------------------------------------------------
    def _load(self) -> None:
        if self._meta_path.exists() and self._vectors_path.exists():
            self._meta = json.loads(self._meta_path.read_text("utf-8"))
            self._matrix = np.load(self._vectors_path).astype(np.float32)
            if self._index is not None and len(self._matrix):
                self._index.add(self._matrix)
            logger.info("vector store loaded: %d chunks", len(self._meta))

    def _persist(self) -> None:
        np.save(self._vectors_path, self._matrix)
        self._meta_path.write_text(json.dumps(self._meta), encoding="utf-8")

    # -- writes --------------------------------------------------------
    def add(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        vectors = embed_batch([c.text for c in chunks])
        with self._lock:
            self._matrix = np.vstack([self._matrix, vectors]) if len(self._matrix) else vectors
            self._meta.extend(c.model_dump(mode="json") for c in chunks)
            if self._index is not None:
                self._index.add(vectors)
            self._persist()
        return len(chunks)

    def delete_document(self, document_id: str) -> int:
        with self._lock:
            keep = [i for i, m in enumerate(self._meta) if m["document_id"] != document_id]
            removed = len(self._meta) - len(keep)
            if not removed:
                return 0
            self._meta = [self._meta[i] for i in keep]
            self._matrix = self._matrix[keep] if keep else np.zeros((0, EMBED_DIM), np.float32)
            if self._index is not None:
                self._index.reset()
                if len(self._matrix):
                    self._index.add(self._matrix)
            self._persist()
        return removed

    def clear(self) -> None:
        with self._lock:
            self._meta = []
            self._matrix = np.zeros((0, EMBED_DIM), dtype=np.float32)
            if self._index is not None:
                self._index.reset()
            self._persist()

    # -- reads --------------------------------------------------------
    def __len__(self) -> int:
        return len(self._meta)

    @property
    def backend(self) -> str:
        return "faiss" if self._index is not None else "numpy"

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        if not self._meta:
            return []
        q = embed(query).reshape(1, -1)
        k = min(top_k, len(self._meta))
        pairs: list[tuple[int, float]]
        if self._index is not None:
            scores, idxs = self._index.search(q, k)
            pairs = list(zip(idxs[0].tolist(), scores[0].tolist()))
        else:
            sims = (self._matrix @ q.T).ravel()
            order = np.argsort(-sims)[:k]
            pairs = [(int(i), float(sims[i])) for i in order]

        results: list[RetrievedChunk] = []
        for pos, score in pairs:
            if pos < 0:
                continue
            meta = self._meta[pos]
            results.append(
                RetrievedChunk(
                    chunk_id=meta["chunk_id"],
                    document_id=meta["document_id"],
                    filename=meta["filename"],
                    source_type=meta["source_type"],
                    locator=meta["locator"],
                    text=meta["text"],
                    # clamp: tiny negative dot products are just floating-point noise
                    score=max(0.0, min(1.0, round(score, 4))),
                )
            )
        return results

    def stats(self) -> dict:
        docs = {m["document_id"] for m in self._meta}
        return {"chunks": len(self._meta), "documents": len(docs), "backend": self.backend}
