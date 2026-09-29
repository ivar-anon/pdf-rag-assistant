"""Vector store on Qdrant.

By default Qdrant runs embedded (local mode, persisted under data/qdrant), so
the project works with `pip install` and nothing else. Set QDRANT_URL to use a
Qdrant server or Qdrant Cloud instead; the code path is the same.
"""
from __future__ import annotations

import uuid
from dataclasses import asdict

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (Distance, FieldCondition, Filter, FilterSelector, MatchValue, PointStruct,
                                  VectorParams)

from .ingest import Chunk


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class QdrantStore:
    def __init__(self, dim: int, collection: str = "documents", *, url: str | None = None, path: str | None = None):
        if url:
            self.client = QdrantClient(url=url)
        elif path:
            self.client = QdrantClient(path=path)
        else:
            self.client = QdrantClient(":memory:")
        self.collection = collection
        if not self.client.collection_exists(collection):
            self.client.create_collection(collection, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        points = [PointStruct(id=_point_id(c.chunk_id), vector=v.tolist(), payload={**asdict(c), "chunk_id": c.chunk_id})
                  for c, v in zip(chunks, vectors)]
        for i in range(0, len(points), 256):
            self.client.upsert(self.collection, points=points[i:i + 256], wait=True)

    def delete_document(self, doc_id: str) -> None:
        self.client.delete(self.collection, points_selector=FilterSelector(
            filter=Filter(must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))])), wait=True)

    def search(self, vector: np.ndarray, limit: int, doc_ids: list[str] | None = None) -> list[tuple[dict, float]]:
        flt = None
        if doc_ids:
            flt = Filter(should=[FieldCondition(key="doc_id", match=MatchValue(value=d)) for d in doc_ids])
        res = self.client.query_points(self.collection, query=vector.tolist(), limit=limit, query_filter=flt,
                                       with_payload=True)
        return [(p.payload, float(p.score)) for p in res.points]

    def all_chunks(self) -> list[dict]:
        out, offset = [], None
        while True:
            points, offset = self.client.scroll(self.collection, limit=512, offset=offset, with_payload=True,
                                                with_vectors=False)
            out += [p.payload for p in points]
            if offset is None:
                return out

    def close(self) -> None:
        self.client.close()
