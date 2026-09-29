"""Hybrid retrieval: dense vectors (Qdrant) + BM25, fused with Reciprocal Rank Fusion.

RRF only uses ranks, so the two very different score scales never need to be
calibrated against each other. Each hit keeps both original scores; the dense
score of the best hit is what decides whether the documents cover the
question at all.
"""
from __future__ import annotations

from dataclasses import dataclass

from .bm25 import BM25


@dataclass
class Hit:
    chunk: dict
    score: float           # fused score (or the single method's score)
    dense: float | None = None
    bm25: float | None = None
    rerank: float | None = None


class HybridRetriever:
    def __init__(self, store, embedder, rrf_k: int = 60, candidates: int = 20):
        self.store, self.embedder = store, embedder
        self.rrf_k, self.candidates = rrf_k, candidates
        self._chunks: list[dict] = []
        self._bm25: BM25 | None = None
        self.refresh()

    def refresh(self) -> None:
        """Rebuild the keyword index after documents are added or removed."""
        self._chunks = self.store.all_chunks()
        self._bm25 = BM25([c["text"] for c in self._chunks]) if self._chunks else None

    def dense(self, query: str, k: int, doc_ids: list[str] | None = None) -> list[Hit]:
        qv = self.embedder.embed_query(query)
        return [Hit(p, s, dense=s) for p, s in self.store.search(qv, k, doc_ids)]

    def keyword(self, query: str, k: int, doc_ids: list[str] | None = None) -> list[Hit]:
        if not self._bm25:
            return []
        scored = [(c, s) for c, s in zip(self._chunks, self._bm25.scores(query))
                  if s > 0 and (not doc_ids or c["doc_id"] in doc_ids)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [Hit(c, s, bm25=s) for c, s in scored[:k]]

    def hybrid(self, query: str, k: int, doc_ids: list[str] | None = None) -> list[Hit]:
        dense = self.dense(query, self.candidates, doc_ids)
        kw = self.keyword(query, self.candidates, doc_ids)
        fused: dict[str, Hit] = {}
        for rank, h in enumerate(dense):
            cid = h.chunk["chunk_id"]
            fused[cid] = Hit(h.chunk, 1 / (self.rrf_k + rank + 1), dense=h.dense)
        for rank, h in enumerate(kw):
            cid = h.chunk["chunk_id"]
            cur = fused.setdefault(cid, Hit(h.chunk, 0.0))
            cur.score += 1 / (self.rrf_k + rank + 1)
            cur.bm25 = h.bm25
        # chunks found only by BM25 still need a dense score for the relevance gate
        missing = [h for h in fused.values() if h.dense is None]
        if missing:
            qv = self.embedder.embed_query(query)
            vecs = self.embedder.embed_documents([h.chunk["text"] for h in missing])
            for h, v in zip(missing, vecs):
                h.dense = float(v @ qv)
        return sorted(fused.values(), key=lambda h: h.score, reverse=True)[:k]

    def search(self, query: str, k: int = 5, method: str = "hybrid", doc_ids: list[str] | None = None) -> list[Hit]:
        return {"hybrid": self.hybrid, "dense": self.dense, "bm25": self.keyword}[method](query, k, doc_ids)
