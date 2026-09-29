import io

import pytest
from fastapi.testclient import TestClient
from reportlab.pdfgen import canvas

from app.main import create_app
from conftest import make_rag


@pytest.fixture
def client(tmp_path):
    rag = make_rag(data_dir=tmp_path)
    with TestClient(create_app(pipeline=rag, seed_samples=True)) as c:
        yield c


def small_pdf(text: str) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(72, 720, text)
    c.save()
    return buf.getvalue()


def test_health_reports_mode(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["mode"] == "extractive" and h["documents"] == 3


def test_ask_returns_answer_with_citations(client):
    r = client.post("/api/ask", json={"question": "What is the monthly fee per robot?"})
    assert r.status_code == 200
    body = r.json()
    assert "145 USD" in body["text"] and body["citations"][0]["source"] == "services-agreement.pdf"
    assert {"retrieval", "latency_ms", "mode", "found"} <= body.keys()


def test_upload_ask_and_delete(client):
    pdf = small_pdf("Onboarding: every new hire gets a buddy for the first 30 days.")
    r = client.post("/api/documents", files={"file": ("onboarding.pdf", pdf, "application/pdf")})
    assert r.status_code == 201
    doc = r.json()
    assert doc["title"] == "Onboarding" and doc["pages"] == 1
    a = client.post("/api/ask", json={"question": "How long does a new hire have a buddy?", "doc_ids": [doc["doc_id"]]}).json()
    assert "30 days" in a["text"]
    assert client.get(f"/api/documents/{doc['doc_id']}/file").headers["content-type"] == "application/pdf"
    assert client.delete(f"/api/documents/{doc['doc_id']}").status_code == 204
    assert doc["doc_id"] not in [d["doc_id"] for d in client.get("/api/documents").json()]


def test_rejects_non_pdf(client):
    r = client.post("/api/documents", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_rejects_pdf_without_text(client):
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.showPage()
    c.save()
    r = client.post("/api/documents", files={"file": ("scan.pdf", buf.getvalue(), "application/pdf")})
    assert r.status_code == 422 and "OCR" in r.json()["detail"]


def test_validation(client):
    assert client.post("/api/ask", json={"question": "hi"}).status_code == 422
    assert client.delete("/api/documents/nope").status_code == 404


def test_ui_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "PDF RAG Assistant" in r.text
