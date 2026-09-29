"""Cross-encoder reranking.

The retriever is fast but reads question and passage separately. A
cross-encoder (BAAI/bge-reranker-base, run locally) reads them together, so it
is much better at two jobs: ordering the retrieved chunks, and saying when none
of them actually answers the question (the relevance gate).
"""
from __future__ import annotations


class CrossEncoderReranker:
    def __init__(self, model: str = "BAAI/bge-reranker-base"):
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        self.name = model
        self._model = TextCrossEncoder(model)

    def scores(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        return [float(s) for s in self._model.rerank(query, texts)]


class OverlapReranker:
    """Dependency-free stand-in for tests: share of query terms found in the text."""
    name = "overlap"

    def scores(self, query: str, texts: list[str]) -> list[float]:
        from .bm25 import tokenize
        q = set(tokenize(query))
        return [10 * len(q & set(tokenize(t))) / max(len(q), 1) - 6 for t in texts]
