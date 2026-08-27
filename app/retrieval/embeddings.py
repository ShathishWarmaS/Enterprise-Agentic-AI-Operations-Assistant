"""Pluggable text embeddings.

The default backend is a deterministic, offline hashed bag-of-n-grams embedding
(`HashingEmbedder`): no network, no model downloads, fully reproducible (which
keeps the evaluation harness stable), numpy-only. It captures lexical overlap
well but has no real semantic understanding (synonyms, paraphrase).

`SentenceTransformerEmbedder` swaps in a real neural encoder when the optional
`embeddings` extra is installed (`pip install -e ".[embeddings]"`). It is never
used in CI (torch is too heavy).

Pick the backend with `EMBEDDING_BACKEND` (see `app.config.Settings`); the
`build_embedder(settings)` factory returns the right one.
"""

from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from collections import Counter

import numpy as np

from app.config import EmbeddingBackend, Settings

_TOKEN_RE = re.compile(r"[a-z0-9]+")
EMBED_DIM = 512


class Embedder(ABC):
    """Turns text into L2-normalised float32 vectors."""

    @property
    @abstractmethod
    def dim(self) -> int: ...

    @abstractmethod
    def embed(self, text: str) -> np.ndarray: ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> np.ndarray: ...


def _l2_normalise(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    return (vec / norm).astype(np.float32) if norm else vec.astype(np.float32)


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _ngrams(tokens: list[str]) -> list[str]:
    grams = list(tokens)
    grams += [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
    # Character trigrams over the whitespace-joined tokens. These make lexical
    # variants ("rollback" vs "roll back", "5xx" vs "5 xx") land near each other,
    # which the pure word-level bag cannot do. Prefixed so they never collide
    # with word features; slightly down-weighted by being more numerous.
    joined = f" {' '.join(tokens)} "
    grams += [f"#{joined[i : i + 3]}" for i in range(len(joined) - 2)]
    return grams


def _bucket(term: str) -> tuple[int, float]:
    digest = hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest()
    idx = int.from_bytes(digest[:4], "big") % EMBED_DIM
    sign = 1.0 if digest[4] & 1 else -1.0
    return idx, sign


class HashingEmbedder(Embedder):
    """Deterministic hashed n-gram + char-trigram embedding (dim 512)."""

    @property
    def dim(self) -> int:
        return EMBED_DIM

    def embed(self, text: str) -> np.ndarray:
        vec = np.zeros(EMBED_DIM, dtype=np.float32)
        tokens = _tokens(text)
        if not tokens:
            return vec
        counts = Counter(_ngrams(tokens))
        for term, count in counts.items():
            idx, sign = _bucket(term)
            # sublinear term frequency dampening, same idea as TF-IDF's log-tf
            vec[idx] += sign * (1.0 + math.log(count))
        return _l2_normalise(vec)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, EMBED_DIM), dtype=np.float32)
        return np.vstack([self.embed(t) for t in texts])


class SentenceTransformerEmbedder(Embedder):
    """Neural sentence embeddings via the optional `sentence_transformers` package."""

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ImportError(
                "EMBEDDING_BACKEND=sentence_transformers needs the `embeddings` extra: "
                'pip install -e ".[embeddings]"'
            ) from exc
        self._model = SentenceTransformer(model_name)
        self._dim = int(self._model.get_sentence_embedding_dimension())

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        vecs = self._model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )
        return np.asarray(vecs, dtype=np.float32)


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_backend is EmbeddingBackend.sentence_transformers:
        return SentenceTransformerEmbedder(settings.embedding_model)
    return HashingEmbedder()


_DEFAULT = HashingEmbedder()


def embed(text: str) -> np.ndarray:
    """Module-level convenience wrapper over the default `HashingEmbedder`."""
    return _DEFAULT.embed(text)


def embed_batch(texts: list[str]) -> np.ndarray:
    """Module-level convenience wrapper over the default `HashingEmbedder`."""
    return _DEFAULT.embed_batch(texts)
