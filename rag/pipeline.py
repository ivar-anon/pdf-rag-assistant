"""The whole flow in one object: add PDFs, list them, ask questions."""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

from .answer import NOT_FOUND, Answer, ExtractiveAnswerer, LLMAnswerer
from .config import Settings, get_settings
from .embeddings import FastEmbedEmbedder
from .ingest import doc_id_for, ingest_pdf
from .rerank import CrossEncoderReranker
from .retrieve import HybridRetriever
from .store import QdrantStore


class RAGPipeline:
    def __init__(self, settings: Settings | None = None, *, embedder=None, reranker=None, answerer=None,
                 in_memory: bool = False):
        self.s = settings or get_settings()
        self.embedder = embedder or FastEmbedEmbedder(self.s.embed_model)
        if reranker is not None:
            self.reranker = reranker
        elif self.s.rerank_model.lower() != "none":
            self.reranker = CrossEncoderReranker(self.s.rerank_model)
        else:
            self.reranker = None
        if in_memory:
            self.store = QdrantStore(self.embedder.dim, self.s.collection)
        elif self.s.qdrant_url:
            self.store = QdrantStore(self.embedder.dim, self.s.collection, url=self.s.qdrant_url)
        else:
            self.s.data_dir.mkdir(parents=True, exist_ok=True)
            self.store = QdrantStore(self.embedder.dim, self.s.collection, path=str(self.s.data_dir / "qdrant"))
        self.retriever = HybridRetriever(self.store, self.embedder)
        if answerer is not None:
            self.answerer = answerer
        elif self.s.openai_api_key:
            self.answerer = LLMAnswerer(self.s.openai_api_key, self.s.openai_model)
        else:
            self.answerer = ExtractiveAnswerer(self.embedder, self.reranker)

    # ---- documents -------------------------------------------------------------
    def add_pdf(self, path: str | Path) -> dict:
        doc_id = doc_id_for(path)
        self.store.delete_document(doc_id)  # re-uploading the same file replaces it
        chunks = ingest_pdf(path, max_chars=self.s.chunk_chars)
        if not chunks:
            raise ValueError(f"No extractable text in {Path(path).name}. Scanned PDFs need OCR first.")
        vectors = self.embedder.embed_documents([f"{c.doc_title}\n{c.text}" for c in chunks])
        self.store.upsert(chunks, vectors)
        self.retriever.refresh()
        return {"doc_id": doc_id, "title": chunks[0].doc_title, "source": chunks[0].source,
                "pages": max(c.page for c in chunks), "chunks": len(chunks)}

    def documents(self) -> list[dict]:
        docs: dict[str, dict] = {}
        for c in self.retriever._chunks:
            d = docs.setdefault(c["doc_id"], {"doc_id": c["doc_id"], "title": c["doc_title"], "source": c["source"],
                                              "pages": 0, "chunks": 0})
            d["pages"] = max(d["pages"], c["page"])
            d["chunks"] += 1
        return sorted(docs.values(), key=lambda d: d["source"])

    def remove(self, doc_id: str) -> None:
        self.store.delete_document(doc_id)
        self.retriever.refresh()

    # ---- questions -----------------------------------------------------------------
    def ask(self, question: str, doc_ids: list[str] | None = None, k: int | None = None) -> dict:
        t0 = time.perf_counter()
        k = k or self.s.top_k
        if self.reranker:
            hits = self.retriever.search(question, max(self.s.candidates, k), "hybrid", doc_ids)
            for h, sc in zip(hits, self.reranker.scores(question, [h.chunk["text"] for h in hits])):
                h.rerank = sc
            hits = sorted(hits, key=lambda h: h.rerank, reverse=True)[:k]
            covered = bool(hits) and hits[0].rerank >= self.s.min_rerank
        else:
            hits = self.retriever.search(question, k, "hybrid", doc_ids)
            covered = bool(hits) and max((h.dense or 0.0) for h in hits) >= self.s.min_relevance
        if not covered:
            ans = Answer(NOT_FOUND, False, "gate")
        else:
            ans = self.answerer.answer(question, hits)
        return {
            "question": question,
            **asdict(ans),
            "retrieval": [{"chunk_id": h.chunk["chunk_id"], "source": h.chunk["source"], "page": h.chunk["page"],
                           "fused": round(h.score, 4), "dense": None if h.dense is None else round(h.dense, 4),
                           "bm25": None if h.bm25 is None else round(h.bm25, 3),
                           "rerank": None if h.rerank is None else round(h.rerank, 3)} for h in hits],
            "latency_ms": round((time.perf_counter() - t0) * 1000),
        }

    def close(self) -> None:
        self.store.close()


if __name__ == "__main__":  # python -m rag.pipeline "question"
    import sys
    rag = RAGPipeline()
    if not rag.documents():
        for pdf in sorted(Path("sample_docs/pdfs").glob("*.pdf")):
            print("indexed", rag.add_pdf(pdf))
    print(json.dumps(rag.ask(" ".join(sys.argv[1:]) or "How many vacation days do employees get?"), indent=2))
