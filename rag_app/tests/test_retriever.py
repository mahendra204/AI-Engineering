"""
tests/test_retriever.py
Unit tests for BM25 and hybrid retrieval logic.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from langchain_core.documents import Document
from app.retriever import BM25Index, _reciprocal_rank_fusion, _tokenise


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_doc(content: str, filename: str = "test.txt", page: int = 1) -> Document:
    return Document(
        page_content=content,
        metadata={"source": filename, "filename": filename, "page": page, "file_type": "txt"},
    )


CORPUS = [
    make_doc("The quick brown fox jumps over the lazy dog", "fox.txt"),
    make_doc("Python is a versatile programming language used in data science", "python.txt"),
    make_doc("Machine learning models require large datasets for training", "ml.txt"),
    make_doc("RAG combines retrieval and generation for better AI answers", "rag.txt"),
    make_doc("Vector databases store embeddings for semantic search", "vectors.txt"),
]


# ── _tokenise ─────────────────────────────────────────────────────────────────

def test_tokenise_basic():
    tokens = _tokenise("Hello World")
    assert tokens == ["hello", "world"]


def test_tokenise_empty():
    assert _tokenise("") == []


def test_tokenise_punctuation():
    # Current tokeniser splits on whitespace only
    tokens = _tokenise("Hello, World!")
    assert "hello," in tokens or "hello" in tokens  # acceptable either way


# ── BM25Index ─────────────────────────────────────────────────────────────────

class TestBM25Index:
    def test_search_returns_results(self):
        idx = BM25Index(CORPUS)
        results = idx.search("Python programming", k=3)
        assert len(results) > 0
        assert all(isinstance(doc, Document) for doc, _ in results)
        assert all(isinstance(score, float) for _, score in results)

    def test_search_relevance_ordering(self):
        idx = BM25Index(CORPUS)
        results = idx.search("Python programming language", k=5)
        # The python.txt doc should rank highest
        top_doc = results[0][0]
        assert "Python" in top_doc.page_content or "programming" in top_doc.page_content

    def test_search_rag_query(self):
        idx = BM25Index(CORPUS)
        results = idx.search("RAG retrieval generation", k=3)
        top_contents = [doc.page_content for doc, _ in results]
        assert any("RAG" in c or "retrieval" in c for c in top_contents)

    def test_search_empty_corpus(self):
        idx = BM25Index([])
        results = idx.search("anything", k=5)
        assert results == []

    def test_search_k_limit(self):
        idx = BM25Index(CORPUS)
        results = idx.search("the", k=2)
        assert len(results) <= 2

    def test_scores_are_non_negative(self):
        idx = BM25Index(CORPUS)
        results = idx.search("machine learning", k=5)
        assert all(score >= 0 for _, score in results)


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

class TestRRF:
    def test_merges_two_lists(self):
        list_a = [(make_doc("doc A about cats"), 0.9), (make_doc("doc B about dogs"), 0.8)]
        list_b = [(make_doc("doc B about dogs"), 0.7), (make_doc("doc C about birds"), 0.6)]
        fused = _reciprocal_rank_fusion(list_a, list_b)
        assert len(fused) >= 2

    def test_scores_are_positive(self):
        list_a = [(make_doc("alpha content here"), 0.9)]
        list_b = [(make_doc("beta content here"), 0.8)]
        fused = _reciprocal_rank_fusion(list_a, list_b)
        assert all(score > 0 for _, score in fused)

    def test_shared_document_scores_higher(self):
        shared = make_doc("shared document appears in both lists")
        unique = make_doc("unique document only in one list xyz")
        list_a = [(shared, 0.9), (unique, 0.5)]
        list_b = [(shared, 0.8), (make_doc("another unique doc abc"), 0.4)]
        fused = _reciprocal_rank_fusion(list_a, list_b)
        fused_contents = [doc.page_content[:20] for doc, _ in fused]
        # shared doc should be first
        assert fused[0][0].page_content.startswith("shared")

    def test_empty_lists(self):
        fused = _reciprocal_rank_fusion([], [])
        assert fused == []

    def test_one_empty_list(self):
        list_a = [(make_doc("only list content abc"), 0.9)]
        fused = _reciprocal_rank_fusion(list_a, [])
        assert len(fused) == 1
