FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 RAG_DATA_DIR=/data FASTEMBED_CACHE_PATH=/models
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY rag rag
COPY app app
COPY sample_docs sample_docs
# download the embedding and reranker models at build time so the first request is fast
RUN python -c "from fastembed import TextEmbedding; from fastembed.rerank.cross_encoder import TextCrossEncoder; TextEmbedding('BAAI/bge-small-en-v1.5'); TextCrossEncoder('BAAI/bge-reranker-base')"
VOLUME ["/data"]
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
