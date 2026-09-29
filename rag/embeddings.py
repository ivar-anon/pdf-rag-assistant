"""Embedding models.

FastEmbedEmbedder runs BAAI/bge-small-en-v1.5 locally (ONNX, 384 dims): no API
key and no per-document cost. HashEmbedder is a tiny deterministic stand-in
used by the unit tests so they run in milliseconds.
"""
from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


def _normalise(v: np.ndarray) -> np.ndarray:
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-12)


class FastEmbedEmbedder:
    def __init__(self, model: str = "BAAI/bge-small-en-v1.5"):
        from fastembed import TextEmbedding
        self.name = model
        self._model = TextEmbedding(model)
        self.dim = len(next(iter(self._model.embed(["dimension probe"]))))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return _normalise(np.array(list(self._model.embed(texts)), dtype=np.float32))

    def embed_query(self, text: str) -> np.ndarray:
        prefix = QUERY_PREFIX if "bge" in self.name.lower() else ""
        return _normalise(np.array(next(iter(self._model.embed([prefix + text]))), dtype=np.float32))


class HashEmbedder:
    name = "hash-512"
    dim = 512

    def _one(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(tok.encode()).hexdigest(), 16) % self.dim] += 1
        return v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return _normalise(np.stack([self._one(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return _normalise(self._one(text))
