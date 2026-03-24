"""
tests/test_ingestor.py
Unit tests for document loading and chunking.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ingestor import load_file, chunk_documents, ingest_from_path
from langchain_core.documents import Document


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_txt(tmp_path):
    f = tmp_path / "sample.txt"
    f.write_text(
        "This is paragraph one.\n\nThis is paragraph two with more content.\n\n"
        "This is paragraph three which has even more content to ensure chunking works correctly."
    )
    return f


@pytest.fixture
def sample_csv(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("name,age,city\nAlice,30,NYC\nBob,25,LA\nCharlie,35,Chicago")
    return f


@pytest.fixture
def sample_html(tmp_path):
    f = tmp_path / "page.html"
    f.write_text("""
    <html><body>
    <nav>Nav content</nav>
    <h1>Main Title</h1>
    <p>This is the main content paragraph.</p>
    <script>alert('remove me')</script>
    </body></html>
    """)
    return f


# ── load_file tests ───────────────────────────────────────────────────────────

def test_load_txt(sample_txt):
    docs = load_file(sample_txt)
    assert len(docs) == 1
    assert "paragraph one" in docs[0].page_content
    assert docs[0].metadata["file_type"] == "txt"
    assert docs[0].metadata["filename"] == "sample.txt"


def test_load_csv(sample_csv):
    docs = load_file(sample_csv)
    assert len(docs) == 3  # 3 data rows
    assert any("Alice" in d.page_content for d in docs)
    assert docs[0].metadata["file_type"] == "csv"


def test_load_html(sample_html):
    docs = load_file(sample_html)
    assert len(docs) == 1
    assert "Main Title" in docs[0].page_content
    assert "alert" not in docs[0].page_content  # script removed
    assert "Nav content" not in docs[0].page_content  # nav removed


def test_load_unsupported_type(tmp_path):
    f = tmp_path / "file.xyz"
    f.write_text("content")
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_file(f)


# ── chunk_documents tests ─────────────────────────────────────────────────────

def test_chunk_recursive():
    docs = [Document(page_content="word " * 500, metadata={"source": "test.txt", "filename": "test.txt"})]
    chunks = chunk_documents(docs, strategy="recursive", chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    assert all(len(c.page_content) <= 200 for c in chunks)  # rough upper bound
    assert chunks[0].metadata["chunk_index"] == 0


def test_chunk_preserves_metadata():
    docs = [Document(
        page_content="Test content " * 100,
        metadata={"source": "doc.pdf", "filename": "doc.pdf", "page": 5}
    )]
    chunks = chunk_documents(docs, strategy="recursive", chunk_size=50, chunk_overlap=10)
    for chunk in chunks:
        assert chunk.metadata["page"] == 5
        assert chunk.metadata["filename"] == "doc.pdf"
        assert "chunk_index" in chunk.metadata
        assert "chunk_total" in chunk.metadata


def test_chunk_empty_input():
    chunks = chunk_documents([])
    assert chunks == []


# ── ingest_from_path tests ────────────────────────────────────────────────────

def test_ingest_from_file(sample_txt):
    chunks = ingest_from_path(sample_txt)
    assert len(chunks) >= 1


def test_ingest_from_directory(tmp_path):
    (tmp_path / "a.txt").write_text("Content A " * 50)
    (tmp_path / "b.txt").write_text("Content B " * 50)
    (tmp_path / "ignore.xyz").write_text("ignored")
    chunks = ingest_from_path(tmp_path)
    sources = {c.metadata["filename"] for c in chunks}
    assert "a.txt" in sources
    assert "b.txt" in sources
    assert "ignore.xyz" not in sources


def test_ingest_nonexistent_path():
    with pytest.raises(FileNotFoundError):
        ingest_from_path("/nonexistent/path/file.txt")
