"""Settings, read from environment variables (or a .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv()


@dataclass(frozen=True)
class Settings:
    data_dir: Path = Path(os.getenv("RAG_DATA_DIR", "data"))
    qdrant_url: str | None = os.getenv("QDRANT_URL") or None          # e.g. http://localhost:6333; empty = embedded
    collection: str = os.getenv("RAG_COLLECTION", "documents")
    embed_model: str = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY") or None
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-6-luna")
    chunk_chars: int = int(os.getenv("RAG_CHUNK_CHARS", "600"))
    top_k: int = int(os.getenv("RAG_TOP_K", "5"))
    candidates: int = int(os.getenv("RAG_CANDIDATES", "8"))            # chunks sent to the reranker
    rerank_model: str = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")  # "none" disables reranking
    # relevance gate: if the best chunk scores below this, the documents don't cover the question
    min_rerank: float = float(os.getenv("RAG_MIN_RERANK", "-5.0"))
    min_relevance: float = float(os.getenv("RAG_MIN_RELEVANCE", "0.62"))  # dense-similarity gate, used without a reranker


def get_settings() -> Settings:
    return Settings()
