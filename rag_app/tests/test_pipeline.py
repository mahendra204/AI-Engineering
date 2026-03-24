"""
tests/test_pipeline.py
Integration-style tests for the RAG pipeline.
LLM and vector store are mocked so no API keys are needed.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.documents import Document
from app.rag_pipeline import RAGPipeline, QueryResult, SourceDocument


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_doc(content: str, filename: str = "test.pdf", page: int = 1) -> Document:
    return Document(
        page_content=content,
        metadata={
            "source": f"data/{filename}",
            "filename": filename,
            "page": page,
            "file_type": "pdf",
            "chunk_index": 0,
            "chunk_total": 3,
        },
    )


MOCK_DOCS = [
    make_doc("RAG stands for Retrieval-Augmented Generation.", "rag_guide.pdf", 1),
    make_doc("Vector databases enable semantic search at scale.", "rag_guide.pdf", 2),
    make_doc("LangChain orchestrates LLM pipelines efficiently.", "langchain.pdf", 1),
]


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_vector_store():
    vs = MagicMock()
    vs.count.return_value = 50
    vs.list_sources.return_value = [
        {"filename": "rag_guide.pdf", "source": "data/rag_guide.pdf", "file_type": "pdf", "chunks": 30},
        {"filename": "langchain.pdf", "source": "data/langchain.pdf", "file_type": "pdf", "chunks": 20},
    ]
    vs.add_documents.return_value = 10
    vs.delete_source.return_value = 5
    return vs


@pytest.fixture
def mock_retriever(mock_vector_store):
    retriever = MagicMock()
    retriever.mode = "hybrid"
    retriever.retrieve.return_value = MOCK_DOCS
    retriever.retrieve_with_scores.return_value = [
        (MOCK_DOCS[0], 0.92),
        (MOCK_DOCS[1], 0.85),
        (MOCK_DOCS[2], 0.78),
    ]
    retriever.invalidate_bm25 = MagicMock()
    return retriever


@pytest.fixture
def pipeline(mock_vector_store, mock_retriever):
    """RAGPipeline with mocked vector store and retriever."""
    return RAGPipeline(
        vector_store=mock_vector_store,
        retriever=mock_retriever,
    )


# ── query() tests ─────────────────────────────────────────────────────────────

class TestQuery:
    @patch("app.rag_pipeline.generate_answer", return_value="RAG combines retrieval and generation.")
    @patch("app.rag_pipeline.count_tokens", return_value=250)
    def test_query_returns_result(self, mock_tokens, mock_gen, pipeline):
        result = pipeline.query("What is RAG?")
        assert isinstance(result, QueryResult)
        assert result.answer == "RAG combines retrieval and generation."
        assert result.question == "What is RAG?"

    @patch("app.rag_pipeline.generate_answer", return_value="Answer text here.")
    @patch("app.rag_pipeline.count_tokens", return_value=100)
    def test_query_sources_populated(self, mock_tokens, mock_gen, pipeline):
        result = pipeline.query("Explain vector databases")
        assert len(result.sources) == 3
        assert all(isinstance(s, SourceDocument) for s in result.sources)
        assert result.sources[0].score == 0.92
        assert result.sources[0].filename == "rag_guide.pdf"
        assert result.sources[0].page == 1

    @patch("app.rag_pipeline.generate_answer", return_value="Answer.")
    @patch("app.rag_pipeline.count_tokens", return_value=50)
    def test_query_latency_recorded(self, mock_tokens, mock_gen, pipeline):
        result = pipeline.query("Test question")
        assert result.latency_ms > 0

    @patch("app.rag_pipeline.generate_answer", return_value="Answer.")
    @patch("app.rag_pipeline.count_tokens", return_value=50)
    def test_query_passes_top_k(self, mock_tokens, mock_gen, pipeline, mock_retriever):
        pipeline.query("Question?", top_k=10)
        mock_retriever.retrieve_with_scores.assert_called_once_with(
            "Question?", k=10, filter_source=None
        )

    @patch("app.rag_pipeline.generate_answer", return_value="Filtered answer.")
    @patch("app.rag_pipeline.count_tokens", return_value=50)
    def test_query_with_filter_source(self, mock_tokens, mock_gen, pipeline, mock_retriever):
        pipeline.query("Question?", filter_source="rag_guide.pdf")
        mock_retriever.retrieve_with_scores.assert_called_once_with(
            "Question?", k=5, filter_source="rag_guide.pdf"
        )


# ── ingest() tests ────────────────────────────────────────────────────────────

class TestIngest:
    @patch("app.rag_pipeline.ingest_from_path")
    def test_ingest_calls_add_documents(self, mock_ingest, pipeline, mock_vector_store):
        mock_ingest.return_value = MOCK_DOCS
        result = pipeline.ingest("data/test.pdf")
        assert result == 10  # mock_vector_store.add_documents returns 10
        mock_vector_store.add_documents.assert_called_once_with(MOCK_DOCS)

    @patch("app.rag_pipeline.ingest_from_path")
    def test_ingest_empty_returns_zero(self, mock_ingest, pipeline, mock_vector_store):
        mock_ingest.return_value = []
        result = pipeline.ingest("data/empty.txt")
        assert result == 0
        mock_vector_store.add_documents.assert_not_called()

    @patch("app.rag_pipeline.ingest_from_path")
    def test_ingest_invalidates_bm25(self, mock_ingest, pipeline, mock_retriever):
        mock_ingest.return_value = MOCK_DOCS
        pipeline.ingest("data/test.pdf")
        mock_retriever.invalidate_bm25.assert_called_once()


# ── Knowledge base management tests ──────────────────────────────────────────

class TestKnowledgeBase:
    def test_list_sources(self, pipeline):
        sources = pipeline.list_sources()
        assert len(sources) == 2
        assert sources[0]["filename"] == "rag_guide.pdf"

    def test_count(self, pipeline):
        assert pipeline.count() == 50

    def test_delete_source(self, pipeline, mock_vector_store, mock_retriever):
        deleted = pipeline.delete_source("rag_guide.pdf")
        assert deleted == 5
        mock_vector_store.delete_source.assert_called_once_with("rag_guide.pdf")
        mock_retriever.invalidate_bm25.assert_called()


# ── SourceDocument dataclass ──────────────────────────────────────────────────

class TestSourceDocument:
    def test_creation(self):
        src = SourceDocument(
            content="Some chunk text",
            source="data/doc.pdf",
            filename="doc.pdf",
            page=3,
            score=0.88,
            file_type="pdf",
        )
        assert src.filename == "doc.pdf"
        assert src.score == 0.88
        assert src.page == 3

    def test_optional_page(self):
        src = SourceDocument(
            content="Text", source="f.txt", filename="f.txt",
            page=None, score=0.5
        )
        assert src.page is None
