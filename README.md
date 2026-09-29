# PDF RAG Assistant

Ask questions about your PDFs and get answers that come **only** from the documents, with a citation to the exact page for every claim. When the documents don't cover a question, it says so instead of guessing.

![Answers with page citations](docs/ui-answers.png)

- **Hybrid retrieval:** BM25 keyword search plus dense vectors in Qdrant, fused with Reciprocal Rank Fusion.
- **Cross-encoder reranking** (`BAAI/bge-reranker-base`) for ordering and for a relevance gate that refuses out-of-scope questions.
- **Grounded answers:** generated with OpenAI (Responses API) from numbered excerpts. Citations are validated, and an uncited answer is flagged.
- **Runs without an API key:** extractive mode quotes the best sentences or table rows, with the same citations.
- **Tables survive extraction.** Rows are rebuilt as `Code | Meaning | Action`, so a question about error E07 finds that row.
- **Measured:** an evaluation set of 36 questions, reported per retrieval method (below).
- **FastAPI backend, a small web UI, Docker, 29 tests and CI.**

## Quick start

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload        # open http://localhost:8000
```

On first start the app indexes three sample PDFs. The embedding and reranker models (about 1.2 GB in total) download once and are cached. Drop your own PDFs into the sidebar.

For generated answers instead of quotes:

```bash
cp .env.example .env                 # then set OPENAI_API_KEY (and OPENAI_MODEL if you like)
```

With Docker (the models are baked into the image at build time):

```bash
docker build -t pdf-rag-assistant .
docker run -p 8000:8000 -v rag-data:/data --env-file .env pdf-rag-assistant
```

## How it works

```mermaid
flowchart LR
  A[PDF upload] --> B["Text per page<br/>layout mode, tables as rows"]
  B --> C["Chunks ≤ 600 chars<br/>never cross a page"]
  C --> D[(Qdrant<br/>bge-small vectors)]
  C --> E[BM25 index]
  Q[Question] --> D & E
  D & E --> F["RRF fusion<br/>top 8"]
  F --> G[Cross-encoder rerank]
  G -->|best score below gate| N["I couldn't find this<br/>in the documents."]
  G -->|top 5 excerpts| H["OpenAI answer with [n] citations<br/>or extractive quotes"]
  H --> V[Citation check] --> R[Answer + sources + page links]
```

1. **Extraction** (`rag/ingest.py`): pypdf in layout mode. Hard-wrapped lines are joined back into paragraphs, and table rows are kept as rows. Page footers are dropped.
2. **Chunking**: paragraphs are packed into chunks of about 600 characters. A heading stays with its paragraph, one block overlaps between neighbouring chunks, and no chunk crosses a page, so every citation points to one page.
3. **Retrieval** (`rag/retrieve.py`): dense search in Qdrant (embedded by default, or a server via `QDRANT_URL`) and BM25, combined with RRF. BM25 catches exact codes, numbers and names; vectors catch paraphrases.
4. **Reranking and gate** (`rag/rerank.py`): a cross-encoder reads the question and each candidate together. If the best score is below `RAG_MIN_RERANK`, the app refuses without calling the LLM.
5. **Answer** (`rag/answer.py`): the model sees numbered excerpts with document and page, and must cite them as `[1]`. Citations to excerpts that don't exist are removed, and an answer with no citation is flagged instead of being presented as grounded.

## Evaluation

`python eval/run_eval.py` runs 30 answerable questions, worded differently from the documents, plus 6 questions the documents don't cover. A hit means the right document **and** the right page. Full output is in [eval/results.md](eval/results.md).

| retrieval | hit@1 | hit@3 | MRR@5 |
|---|---|---|---|
| BM25 | 93% | 100% | 0.961 |
| dense | 93% | 100% | 0.961 |
| hybrid (RRF) | 93% | 100% | 0.967 |
| **hybrid + reranker** | **100%** | **100%** | **1.000** |

In extractive mode (no LLM), 28/30 answers contain the expected fact and 30/30 cite the right page. The relevance gate refuses 4 of the 6 out-of-scope questions while keeping all 30 answerable ones. The two it lets through ("Who is the CEO?", "What does the robot cost to buy?") land on passages that mention the company or prices. In LLM mode those are handled by the instruction to answer only from the excerpts, which extractive mode cannot do.

## API

| method | path | |
|---|---|---|
| GET | `/api/health` | mode (`llm` / `extractive`), models, vector store |
| GET | `/api/documents` | indexed documents |
| POST | `/api/documents` | upload a PDF (`multipart/form-data`, field `file`) |
| DELETE | `/api/documents/{doc_id}` | remove a document |
| GET | `/api/documents/{doc_id}/file` | the original PDF (the UI opens it at the cited page) |
| POST | `/api/ask` | `{"question": "...", "doc_ids": ["..."]}`. `doc_ids` is optional and limits the search |

```json
{
  "text": "Up to five unused vacation days may be carried over to the next year; carried-over days expire on 31 March. [1]",
  "found": true,
  "mode": "extractive",
  "citations": [{"n": 1, "source": "employee-handbook.pdf", "page": 2, "doc_title": "...", "snippet": "..."}],
  "retrieval": [{"chunk_id": "c2a593b892:p2:c0", "fused": 0.0328, "dense": 0.759, "bm25": 9.783, "rerank": 5.309}],
  "latency_ms": 1618
}
```

## Configuration

All settings are environment variables (see `.env.example`).

| variable | default | |
|---|---|---|
| `OPENAI_API_KEY` | empty | enables generated answers |
| `OPENAI_MODEL` | `gpt-6-luna` | any Responses API model |
| `QDRANT_URL` | empty | empty = embedded Qdrant under `data/` |
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | any fastembed text model |
| `RERANK_MODEL` | `BAAI/bge-reranker-base` | `none` falls back to a dense-similarity gate |
| `RAG_MIN_RERANK` | `-5.0` | relevance gate (see the evaluation for the trade-off) |
| `RAG_CHUNK_CHARS` / `RAG_TOP_K` / `RAG_CANDIDATES` | `600` / `5` / `8` | chunk size, excerpts sent to the model, chunks reranked |

## Project layout

```
rag/          ingest · embeddings · store (Qdrant) · bm25 · retrieve · rerank · answer · pipeline · config
app/          FastAPI app and the single-page UI
eval/         questions.jsonl, run_eval.py, results.md
sample_docs/  generator for the sample PDFs (a fictional company) and the PDFs themselves
tests/        29 tests (ingest, retrieval, gate, LLM contract with a mocked client, API)
```

## Limits and next steps

- **Scanned PDFs** have no text layer and are rejected with a clear message. Adding OCR (e.g. Tesseract via `ocrmypdf`) before ingestion is the usual fix.
- The LLM path is covered by tests with a mocked OpenAI client; point it at your own key to see generated answers.
- Single-tenant: there is no login or per-user document separation. Qdrant payload filters by user or workspace are the natural extension.
- The reranker runs on CPU (about 1.5 s per question). A GPU, a smaller reranker or reranking fewer candidates bring that down.
- The sample corpus is small and synthetic. On a real corpus, re-run the evaluation with your own questions and tune `RAG_MIN_RERANK` from the table it prints.

## License

MIT
