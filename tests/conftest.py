import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.config import Settings  # noqa: E402
from rag.embeddings import HashEmbedder  # noqa: E402
from rag.pipeline import RAGPipeline  # noqa: E402
from rag.rerank import OverlapReranker  # noqa: E402

PDFS = sorted((ROOT / "sample_docs" / "pdfs").glob("*.pdf"))


@pytest.fixture(scope="session", autouse=True)
def sample_pdfs():
    if len(PDFS) < 3:
        import subprocess
        subprocess.run([sys.executable, str(ROOT / "sample_docs" / "generate_docs.py")], check=True)
    return sorted((ROOT / "sample_docs" / "pdfs").glob("*.pdf"))


class FakeLLM:
    """Stands in for the OpenAI client: records the call and returns a canned answer."""

    def __init__(self, text):
        self.text, self.calls = text, []
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(output_text=self.text)


def make_rag(answerer=None, **settings):
    s = Settings(openai_api_key=None, rerank_model="none", **settings)
    return RAGPipeline(s, embedder=HashEmbedder(), reranker=OverlapReranker(), answerer=answerer, in_memory=True)


@pytest.fixture
def rag(sample_pdfs):
    r = make_rag()
    for p in sample_pdfs:
        r.add_pdf(p)
    yield r
    r.close()
