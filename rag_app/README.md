# 🔍 RAG Application — Retrieval-Augmented Generation

A production-ready RAG (Retrieval-Augmented Generation) application built with:
- **LangChain** — orchestration & document processing
- **ChromaDB** — local vector store
- **OpenAI** — embeddings (`text-embedding-3-small`) + LLM (`gpt-4o-mini`)
- **FastAPI** — REST API backend
- **Streamlit** — interactive chat UI
- **PyMuPDF / python-docx / BeautifulSoup** — multi-format document ingestion

---

## 📁 Project Structure

```
rag_app/
├── app/
│   ├── __init__.py
│   ├── config.py           # All configuration & env vars
│   ├── ingestor.py         # Document loading & chunking
│   ├── embedder.py         # Embedding model wrapper
│   ├── vectorstore.py      # ChromaDB CRUD operations
│   ├── retriever.py        # Hybrid retrieval (semantic + BM25)
│   ├── generator.py        # LLM chain + prompt engineering
│   ├── rag_pipeline.py     # End-to-end RAG orchestrator
│   └── api.py              # FastAPI routes
├── data/                   # Drop your documents here
│   └── sample.txt          # Sample document for quick test
├── vectorstore/            # ChromaDB persisted here (auto-created)
├── scripts/
│   ├── ingest_docs.py      # CLI: ingest documents into vector DB
│   └── query_cli.py        # CLI: ask questions from terminal
├── tests/
│   ├── test_ingestor.py
│   ├── test_retriever.py
│   └── test_pipeline.py
├── ui.py                   # Streamlit chat interface
├── main.py                 # FastAPI app entrypoint
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
cd rag_app
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### 3. Add Your Documents

Drop any supported files into the `data/` folder:
- `.pdf` — PDF documents
- `.txt` — Plain text files
- `.md` — Markdown files
- `.docx` — Word documents
- `.html` — Web pages
- `.csv` — CSV data files

### 4. Ingest Documents

```bash
python scripts/ingest_docs.py --source data/
```

### 5a. Launch Chat UI (Streamlit)

```bash
streamlit run ui.py
```
Open http://localhost:8501

### 5b. Launch REST API (FastAPI)

```bash
uvicorn main:app --reload --port 8000
```
Open http://localhost:8000/docs

### 5c. Use CLI

```bash
python scripts/query_cli.py --question "What is this document about?"
```

---

## 🔌 API Reference

### POST `/api/query`
Ask a question against the knowledge base.

```json
// Request
{
  "question": "What are the key findings?",
  "top_k": 5,
  "filter_source": null
}

// Response
{
  "answer": "The key findings are ...",
  "sources": [
    {
      "content": "...",
      "source": "data/report.pdf",
      "page": 3,
      "score": 0.91
    }
  ],
  "question": "What are the key findings?",
  "model": "gpt-4o-mini",
  "tokens_used": 842
}
```

### POST `/api/ingest`
Ingest a document at runtime via API.

```json
// Request (multipart/form-data)
file: <uploaded file>

// Response
{
  "message": "Successfully ingested 47 chunks from report.pdf",
  "chunks": 47,
  "filename": "report.pdf"
}
```

### GET `/api/sources`
List all ingested document sources.

### DELETE `/api/sources/{source_name}`
Remove a document from the knowledge base.

### GET `/api/health`
Health check.

---

## ⚙️ Configuration

All settings are in `.env` or `app/config.py`:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | required | Your OpenAI API key |
| `OPENAI_MODEL` | `gpt-4o-mini` | LLM model |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `CHUNK_SIZE` | `1000` | Token size per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K` | `5` | Retrieval top-k |
| `CHROMA_PATH` | `vectorstore/` | Vector DB path |
| `TEMPERATURE` | `0.2` | LLM temperature |

---

## 🏗️ Architecture

```
User Query
    │
    ▼
[Query Embedding]          ← text-embedding-3-small
    │
    ▼
[Hybrid Retrieval]         ← ChromaDB (semantic) + BM25 (keyword)
    │
    ▼
[Re-ranking / Filtering]   ← MMR diversity + score threshold
    │
    ▼
[Prompt Assembly]          ← Context + System prompt + Question
    │
    ▼
[LLM Generation]           ← gpt-4o-mini
    │
    ▼
[Answer + Sources]         ← Returned to user
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 📝 Switching LLM / Embedding Providers

The app is designed to swap providers easily. In `app/config.py`:

```python
# Use Anthropic Claude instead of OpenAI
LLM_PROVIDER = "anthropic"   # "openai" | "anthropic" | "ollama"

# Use local Ollama instead of cloud
LLM_PROVIDER = "ollama"
OLLAMA_MODEL = "llama3.2"
```

---

## 🔒 Notes

- API keys are never stored in code — use `.env` only
- ChromaDB runs entirely locally — your documents stay private
- The vectorstore persists between runs automatically
