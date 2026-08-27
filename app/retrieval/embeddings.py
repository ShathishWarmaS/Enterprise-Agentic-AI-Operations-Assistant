"""Deterministic, offline text embeddings.

This project must run with no network access and no model downloads, so we use
a hashed bag-of-n-grams embedding rather than a neural encoder. It captures
lexical overlap well enough for retrieval over the sample corpus, is fully
deterministic (which keeps the evaluation harness stable), and has no
dependencies beyond numpy.

Trade-off: it has no real semantic understanding (synonyms, paraphrase). The
`VectorStore` interface is intentionally small so a sentence-transformers or
Anthropic-embedding backend can be swapped in without touching callers.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")
EMBED_DIM = 512


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


def embed(text: str) -> np.ndarray:
    vec = np.zeros(EMBED_DIM, dtype=np.float32)
    tokens = _tokens(text)
    if not tokens:
        return vec
    counts = Counter(_ngrams(tokens))
    for term, count in counts.items():
        idx, sign = _bucket(term)
        # sublinear term frequency dampening, same idea as TF-IDF's log-tf
        vec[idx] += sign * (1.0 + math.log(count))
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm else vec


def embed_batch(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)
    return np.vstack([embed(t) for t in texts])
