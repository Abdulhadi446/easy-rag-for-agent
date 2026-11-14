"""Lexical scoring and rank fusion for hybrid retrieval.

Dense embeddings are good at meaning, BM25 is good at exact words, identifiers and
acronyms. Hybrid search runs both and fuses the rankings with reciprocal rank fusion
(RRF), which needs no score calibration between the two signals.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Sequence

TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)

# A small English stop list keeps BM25 from rewarding "the" and "of". Documents in
# other languages simply have no stop words removed.
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "then",
        "there",
        "these",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
    }
)


def tokenize(text: str, *, keep_stopwords: bool = False) -> list[str]:
    """Lower-case word tokens, with an optional English stop-word filter."""
    tokens = TOKEN_RE.findall(text.lower())
    if keep_stopwords:
        return tokens
    filtered = [token for token in tokens if token not in STOPWORDS]
    return filtered or tokens


class BM25:
    """Okapi BM25 over a fixed set of documents."""

    def __init__(self, documents: Sequence[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.documents = [tokenize(document) for document in documents]
        self.frequencies = [Counter(tokens) for tokens in self.documents]
        self.lengths = [len(tokens) for tokens in self.documents]
        self.average_length = (sum(self.lengths) / len(self.lengths)) if self.lengths else 0.0
        document_frequency: Counter[str] = Counter()
        for frequency in self.frequencies:
            document_frequency.update(frequency.keys())
        total = len(self.documents)
        self.idf = {
            term: math.log(1.0 + (total - count + 0.5) / (count + 0.5))
            for term, count in document_frequency.items()
        }

    def scores(self, query: str) -> list[float]:
        """One BM25 score per document (higher is better)."""
        if not self.documents:
            return []
        terms = tokenize(query)
        if not terms:
            return [0.0] * len(self.documents)
        average = self.average_length or 1.0
        results: list[float] = []
        for frequency, length in zip(self.frequencies, self.lengths):
            score = 0.0
            for term in terms:
                count = frequency.get(term, 0)
                if not count:
                    continue
                denominator = count + self.k1 * (
                    1.0 - self.b + self.b * (length / average)
                )
                score += self.idf.get(term, 0.0) * (count * (self.k1 + 1.0)) / denominator
            results.append(score)
        return results

    def top(self, query: str, limit: int) -> list[tuple[int, float]]:
        """Best ``limit`` document indexes, skipping documents with no term overlap."""
        ranked = [
            (index, score)
            for index, score in enumerate(self.scores(query))
            if score > 0.0
        ]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[: max(0, limit)]


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[int]],
    *,
    weights: Sequence[float] | None = None,
    k: int = 60,
) -> dict[int, float]:
    """Fuse ranked index lists: score(i) = sum(weight / (k + rank))."""
    if weights is None:
        weights = [1.0] * len(rankings)
    fused: dict[int, float] = defaultdict(float)
    for ranking, weight in zip(rankings, weights):
        seen: set[int] = set()
        for rank, index in enumerate(ranking, start=1):
            if index in seen:
                continue
            seen.add(index)
            fused[index] += weight * (1.0 / (k + rank))
    return dict(fused)
