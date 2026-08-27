"""Lightweight groundedness check used by the Validation agent (and mock Action).

We can't run an NLI model offline, so we approximate "is this sentence supported
by the context" with content-word overlap: the fraction of a claim's meaningful
words that also appear in some single context chunk. It is deliberately
conservative - short claims need near-total overlap - so the failure mode is
rejecting a weak-but-true claim, not passing a fabricated one.
"""

from __future__ import annotations

import re

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_WORD = re.compile(r"[a-z][a-z0-9\-]+")
_STOP = {
    "the",
    "and",
    "for",
    "that",
    "this",
    "with",
    "from",
    "was",
    "were",
    "are",
    "has",
    "have",
    "had",
    "will",
    "should",
    "would",
    "could",
    "may",
    "might",
    "into",
    "over",
    "under",
    "than",
    "then",
    "them",
    "they",
    "there",
    "here",
    "which",
    "while",
    "also",
    "been",
    "being",
    "such",
    "each",
    "some",
    "most",
    "more",
    "very",
    "when",
    "what",
    "who",
    "how",
    "why",
    "its",
    "it's",
    "a",
}


_LIST_PREFIX = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
_TERMINATED = re.compile(r"[.!?:]\s*$")


def _blocks(text: str) -> list[str]:
    """Group physical lines into logical blocks.

    A new block starts on a blank line, a heading, or a list marker. Lines
    inside a block are joined with spaces - crucial for PDF and wrapped
    Markdown, where sentences routinely span several physical lines.
    """
    blocks: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            joined = " ".join(b.strip() for b in buf).strip()
            if joined:
                blocks.append(joined)
            buf.clear()

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            flush()
            continue
        if line.startswith("#") or _LIST_PREFIX.match(line):
            flush()
            line = _LIST_PREFIX.sub("", line.lstrip("#").strip())
        buf.append(line)
        if _TERMINATED.search(line):
            flush()
    flush()
    return blocks


_CITE_MARKER = re.compile(r"\s*\[\d+\]")


def split_sentences(text: str) -> list[str]:
    out: list[str] = []
    text = _CITE_MARKER.sub("", text.replace("`", ""))
    for block in _blocks(text):
        for piece in _SENT_SPLIT.split(block):
            piece = piece.strip(" -*\t")
            if len(piece) > 2:
                out.append(piece)
    return out


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if w not in _STOP}


def _trigrams(text: str) -> set[str]:
    squashed = re.sub(r"[^a-z0-9]+", "", text.lower())
    return {squashed[i : i + 3] for i in range(len(squashed) - 2)}


def support_score(claim: str, contexts: list[str]) -> float:
    claim_words = _content_words(claim)
    if not claim_words:
        return 1.0  # nothing to support (e.g. "Next steps:")
    claim_tri = _trigrams(claim)
    best = 0.0
    for ctx in contexts:
        ctx_words = _content_words(ctx)
        if not ctx_words:
            continue
        word_cov = len(claim_words & ctx_words) / len(claim_words)
        # character-level cover catches "roll back" ~ "rollback" style variants
        tri_cov = len(claim_tri & _trigrams(ctx)) / len(claim_tri) if claim_tri else 0.0
        best = max(best, max(word_cov, 0.85 * tri_cov))
    return round(best, 3)


def is_supported(claim: str, contexts: list[str], *, threshold: float = 0.6) -> bool:
    words = _content_words(claim)
    # Very short claims must be almost entirely covered to count as supported.
    effective = threshold if len(words) >= 5 else max(threshold, 0.8)
    return support_score(claim, contexts) >= effective
