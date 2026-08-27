"""In-process vector index over ingested chunks (single-worker).

Uses FAISS (`IndexFlatIP` on L2-normalised vectors == cosine similarity) when it
is importable, and falls back to a pure-numpy brute-force search otherwise so
the test suite and `import` checks work even if the wheel is unavailable on a
given platform. Chunk metadata is stored alongside the vectors in a JSON
sidecar; positions in the index line up with positions in that list.

The embedding dimension is persisted next to the index. Loading an index whose
dim does not match the current embedder is refused with a clear error, because
mixing vector spaces silently produces garbage retrieval.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

import numpy as np

from app.retrieval.backends.base import VectorBackend
from app.retrieval.embeddings import Embedder
from app.schemas.documents import Chunk
from app.schemas.retrieval import RetrievedChunk

logger = logging.getLogger(__name__)

try:  # pragma: no cover - exercised implicitly
    import faiss

    _HAVE_FAISS = True
except ImportError:  # pragma: no cover
    faiss = None  # type: ignore[assignment]
    _HAVE_FAISS = False


class DimMismatchError(RuntimeError):
    pass


class LocalVectorBackend(VectorBackend):
    def __init__(self, directory: Path, embedder: Embedder) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._embedder = embedder
        self._dim = embedder.dim
        self._vectors_path = self._dir / "vectors.npy"
        self._meta_path = self._dir / "chunks.json"
        self._dim_path = self._dir / "dim.txt"
        self._lock = threading.Lock()
        self._meta: list[dict] = []
        self._matrix = np.zeros((0, self._dim), dtype=np.float32)
        self._index = faiss.IndexFlatIP(self._dim) if _HAVE_FAISS else None
        self._load()

    # -- persistence -----------------------------------------------------
    def _load(self) -> None:
        if self._dim_path.exists():
            stored = int(self._dim_path.read_text("utf-8").strip())
            if stored != self._dim:
                raise DimMismatchError(
                    f"persisted vector index has dim {stored} but the current embedding "
                    f"backend produces dim {self._dim}. Run `rm -rf {self._dir}` "
                    "(or the whole storage/vector dir) and re-seed after switching "
                    "EMBEDDING_BACKEND."
                )
        if self._meta_path.exists() and self._vectors_path.exists():
            self._meta = json.loads(self._meta_path.read_text("utf-8"))
            self._matrix = np.load(self._vectors_path).astype(np.float32)
            if self._matrix.shape[1] and self._matrix.shape[1] != self._dim:
                raise DimMismatchError(
                    f"persisted vectors have dim {self._matrix.shape[1]} but the current "
                    f"embedding backend produces dim {self._dim}. Run `rm -rf {self._dir}` "
                    "and re-seed after switching EMBEDDING_BACKEND."
                )
            if self._index is not None and len(self._matrix):
                self._index.add(self._matrix)
            logger.info("vector store loaded: %d chunks", len(self._meta))

    def _persist(self) -> None:
        np.save(self._vectors_path, self._matrix)
        self._meta_path.write_text(json.dumps(self._meta), encoding="utf-8")
        self._dim_path.write_text(str(self._dim), encoding="utf-8")

    # -- writes --------------------------------------------------------
    def add(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        vectors = self._embedder.embed_batch([c.text for c in chunks])
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
            self._matrix = self._matrix[keep] if keep else np.zeros((0, self._dim), np.float32)
            if self._index is not None:
                self._index.reset()
                if len(self._matrix):
                    self._index.add(self._matrix)
            self._persist()
        return removed

    def clear(self) -> None:
        with self._lock:
            self._meta = []
            self._matrix = np.zeros((0, self._dim), dtype=np.float32)
            if self._index is not None:
                self._index.reset()
            self._persist()

    # -- reads --------------------------------------------------------
    def __len__(self) -> int:
        return len(self._meta)

    @property
    def backend(self) -> str:
        return "local:faiss" if self._index is not None else "local:numpy"

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        if not self._meta:
            return []
        q = self._embedder.embed(query).reshape(1, -1)
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
