# Vector DB Benchmark

OpenSearch vs Qdrant vector search benchmark with RAG Chatbot, powered by Upstage Embedding API.

## Features

- **Benchmark**: Qdrant vs OpenSearch side-by-side latency & result comparison
- **RAG Chatbot**: Document-grounded Q&A (hallucination-guarded, streaming)
- **Multi-format ingestion**: PDF, DOCX support with sliding-window chunking
- **KB Index management**: Multiple knowledge bases, document deduplication

## Architecture

```
PDF/DOCX → Upstage embedding-passage → Qdrant + OpenSearch
User query → Upstage embedding-query → Vector search → LLM (streaming)
```

## Quick Start

### 1. Clone & install dependencies

```bash
git clone https://github.com/YOUR_USERNAME/vector-db-benchmark.git
cd vector-db-benchmark

# Install uv (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

### 2. Set up environment variables

```bash
cp .env.example .env
# Edit .env and fill in your API keys
```

Required keys:
- `UPSTAGE_API_KEY` — get from [console.upstage.ai](https://console.upstage.ai)
- `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` — optional, for RAG Chatbot LLM

### 3. Start OpenSearch (optional, for benchmark comparison)

```bash
make start-opensearch   # downloads ~400MB on first run
```

### 4. Run

```bash
# Streamlit UI (Benchmark + RAG Chatbot)
make app

# CLI benchmark (PDF ingest + Qdrant vs OpenSearch comparison)
make run
```

## Commands

| Command | Description |
|---------|-------------|
| `make start-opensearch` | Download and start OpenSearch in background |
| `make stop-opensearch` | Stop OpenSearch |
| `make app` | Launch Streamlit UI (`localhost:8501`) |
| `make run` | Run CLI benchmark |

## Embedding Models

| Model | Usage | Dimension |
|-------|-------|-----------|
| `embedding-passage` | Document ingestion | 4096 |
| `embedding-query` | Search queries | 4096 |

## Notes

- **WSL users**: Must use `OPENSEARCH_KNN_ENGINE=lucene` (nmslib/faiss crash without native libs)
- **Qdrant modes**: `memory` (fastest), `local` (persistent, no server), `server` (remote)
- Score ranges differ: OpenSearch scores are normalized `(1+cos)/2 → [0,1]`, Qdrant returns raw cosine

## Tech Stack

- Python 3.12+, uv
- [Upstage Embeddings API](https://console.upstage.ai)
- [Qdrant](https://qdrant.tech) (local or server mode)
- [OpenSearch](https://opensearch.org) (HNSW via knn_vector)
- Streamlit
