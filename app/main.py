"""HTTP API and web UI.

    uvicorn app.main:app --reload         # then open http://localhost:8000

Endpoints
    GET    /api/health                    mode (llm / extractive) and models in use
    GET    /api/documents                 indexed documents
    POST   /api/documents                 upload a PDF (multipart field "file")
    DELETE /api/documents/{doc_id}        remove a document
    GET    /api/documents/{doc_id}/file   the original PDF (the UI opens it at the cited page)
    POST   /api/ask                       {"question": "...", "doc_ids": ["..."]?}
"""
from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag.ingest import doc_id_for
from rag.pipeline import RAGPipeline

ROOT = Path(__file__).resolve().parent.parent
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
STATE: dict = {}


class AskIn(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    doc_ids: list[str] | None = None


def files_dir(rag: RAGPipeline) -> Path:
    d = rag.s.data_dir / "files"
    d.mkdir(parents=True, exist_ok=True)
    return d


def create_app(pipeline: RAGPipeline | None = None, seed_samples: bool | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        rag = pipeline or RAGPipeline()
        STATE["rag"] = rag
        seed = seed_samples if seed_samples is not None else os.getenv("SEED_SAMPLES", "1") == "1"
        if seed and not rag.documents():
            for pdf in sorted((ROOT / "sample_docs" / "pdfs").glob("*.pdf")):
                info = rag.add_pdf(pdf)
                shutil.copy(pdf, files_dir(rag) / f"{info['doc_id']}.pdf")
        yield
        rag.close()

    app = FastAPI(title="PDF RAG Assistant", version="1.0.0", lifespan=lifespan)

    @app.get("/api/health")
    def health():
        rag: RAGPipeline = STATE["rag"]
        return {"status": "ok", "mode": rag.answerer.mode,
                "llm_model": rag.s.openai_model if rag.answerer.mode == "llm" else None,
                "embedding_model": rag.embedder.name,
                "reranker": rag.reranker.name if rag.reranker else None,
                "vector_store": "qdrant (server)" if rag.s.qdrant_url else "qdrant (embedded)",
                "documents": len(rag.documents())}

    @app.get("/api/documents")
    def documents():
        return STATE["rag"].documents()

    @app.post("/api/documents", status_code=201)
    async def upload(file: UploadFile = File(...)):
        rag: RAGPipeline = STATE["rag"]
        data = await file.read()
        if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(413, f"File larger than {MAX_UPLOAD_MB} MB.")
        if not data.startswith(b"%PDF"):
            raise HTTPException(415, "Only PDF files are supported.")
        name = Path(file.filename or "document.pdf").name
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / name
            path.write_bytes(data)
            try:
                info = rag.add_pdf(path)
            except ValueError as e:
                raise HTTPException(422, str(e))
            shutil.copy(path, files_dir(rag) / f"{doc_id_for(path)}.pdf")
        return info

    @app.delete("/api/documents/{doc_id}", status_code=204)
    def delete(doc_id: str):
        rag: RAGPipeline = STATE["rag"]
        if doc_id not in {d["doc_id"] for d in rag.documents()}:
            raise HTTPException(404, "Unknown document.")
        rag.remove(doc_id)
        (files_dir(rag) / f"{doc_id}.pdf").unlink(missing_ok=True)

    @app.get("/api/documents/{doc_id}/file")
    def original(doc_id: str):
        path = files_dir(STATE["rag"]) / f"{Path(doc_id).name}.pdf"
        if not path.exists():
            raise HTTPException(404, "File not stored.")
        return FileResponse(path, media_type="application/pdf")

    @app.post("/api/ask")
    def ask(body: AskIn):
        rag: RAGPipeline = STATE["rag"]
        if not rag.documents():
            raise HTTPException(409, "Upload at least one PDF first.")
        return rag.ask(body.question.strip(), body.doc_ids or None)

    app.mount("/", StaticFiles(directory=ROOT / "app" / "static", html=True), name="static")
    return app


app = create_app()
