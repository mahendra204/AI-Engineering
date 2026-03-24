"""
app/ingestor.py
Multi-format document loader and text splitter.

Supported formats:
  .pdf   → PyMuPDF (fitz)
  .docx  → python-docx
  .txt   → plain text
  .md    → Markdown (treated as text)
  .html  → BeautifulSoup
  .csv   → pandas → text rows
"""

import csv
import logging
import os
from pathlib import Path
from typing import Iterator, List, Optional

from langchain_core.documents import Document
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

from app.config import settings

logger = logging.getLogger(__name__)


# ── Low-level loaders ─────────────────────────────────────────────────────────

def _load_pdf(path: Path) -> List[Document]:
    """Load a PDF file page-by-page using PyMuPDF."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("Install pymupdf: pip install pymupdf")

    docs = []
    with fitz.open(str(path)) as pdf:
        for page_num, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            if text:
                docs.append(Document(
                    page_content=text,
                    metadata={
                        "source": str(path),
                        "filename": path.name,
                        "page": page_num,
                        "total_pages": len(pdf),
                        "file_type": "pdf",
                    },
                ))
    logger.info(f"PDF '{path.name}': loaded {len(docs)} pages")
    return docs


def _load_docx(path: Path) -> List[Document]:
    """Load a DOCX file, preserving paragraph structure."""
    try:
        from docx import Document as DocxDocument
    except ImportError:
        raise ImportError("Install python-docx: pip install python-docx")

    doc = DocxDocument(str(path))
    full_text = "\n\n".join(
        p.text.strip() for p in doc.paragraphs if p.text.strip()
    )
    return [Document(
        page_content=full_text,
        metadata={"source": str(path), "filename": path.name, "file_type": "docx"},
    )]


def _load_txt(path: Path) -> List[Document]:
    """Load a plain text or Markdown file."""
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    return [Document(
        page_content=text,
        metadata={
            "source": str(path),
            "filename": path.name,
            "file_type": path.suffix.lstrip("."),
        },
    )]


def _load_html(path: Path) -> List[Document]:
    """Load an HTML file, stripping tags with BeautifulSoup."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError("Install beautifulsoup4: pip install beautifulsoup4")

    raw = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(raw, "lxml")
    # Remove script / style noise
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    return [Document(
        page_content=text,
        metadata={"source": str(path), "filename": path.name, "file_type": "html"},
    )]


def _load_csv(path: Path) -> List[Document]:
    """Load a CSV, converting each row to a text document."""
    docs = []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            text = "\n".join(f"{k}: {v}" for k, v in row.items() if v)
            if text:
                docs.append(Document(
                    page_content=text,
                    metadata={
                        "source": str(path),
                        "filename": path.name,
                        "row": i + 1,
                        "file_type": "csv",
                    },
                ))
    logger.info(f"CSV '{path.name}': loaded {len(docs)} rows")
    return docs


# ── Dispatcher ────────────────────────────────────────────────────────────────

_LOADERS = {
    ".pdf":  _load_pdf,
    ".docx": _load_docx,
    ".txt":  _load_txt,
    ".md":   _load_txt,
    ".html": _load_html,
    ".htm":  _load_html,
    ".csv":  _load_csv,
}


def load_file(path: Path) -> List[Document]:
    """Load a single file into a list of LangChain Documents."""
    ext = path.suffix.lower()
    loader = _LOADERS.get(ext)
    if loader is None:
        raise ValueError(
            f"Unsupported file type '{ext}'. "
            f"Supported: {', '.join(_LOADERS)}"
        )
    return loader(path)


def load_directory(
    directory: str | Path,
    glob: Optional[str] = None,
    recursive: bool = True,
) -> List[Document]:
    """Recursively load all supported documents from a directory."""
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    docs: List[Document] = []
    pattern = glob or "**/*" if recursive else "*"
    
    for path in sorted(directory.glob(pattern)):
        if not path.is_file():
            continue
        if path.suffix.lower() not in _LOADERS:
            continue
        try:
            file_docs = load_file(path)
            docs.extend(file_docs)
            logger.info(f"Loaded {len(file_docs)} document(s) from {path.name}")
        except Exception as e:
            logger.warning(f"Skipped '{path.name}': {e}")

    logger.info(f"Total documents loaded from '{directory}': {len(docs)}")
    return docs


# ── Chunking ──────────────────────────────────────────────────────────────────

def chunk_documents(
    documents: List[Document],
    strategy: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[Document]:
    """
    Split documents into chunks using the configured strategy.

    Strategies:
      recursive → RecursiveCharacterTextSplitter (default, works on all content)
      semantic  → SentenceTransformersTokenTextSplitter (token-aware)
      sentence  → sentence-aware splitting
    """
    strategy   = strategy or settings.chunk_strategy
    chunk_size    = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    if strategy == "recursive":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )
    elif strategy == "semantic":
        splitter = SentenceTransformersTokenTextSplitter(
            chunk_overlap=chunk_overlap,
            tokens_per_chunk=chunk_size,
        )
    elif strategy == "sentence":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[". ", "! ", "? ", "\n\n", "\n", " "],
        )
    else:
        raise ValueError(f"Unknown chunk strategy: {strategy}")

    chunks = splitter.split_documents(documents)

    # Add chunk index metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["chunk_total"] = len(chunks)

    logger.info(
        f"Chunked {len(documents)} documents into {len(chunks)} chunks "
        f"(strategy={strategy}, size={chunk_size}, overlap={chunk_overlap})"
    )
    return chunks


def ingest_from_path(
    source: str | Path,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    strategy: Optional[str] = None,
) -> List[Document]:
    """
    Convenience function: load + chunk from a file or directory path.
    Returns chunked Documents ready to be embedded.
    """
    source = Path(source)

    if source.is_file():
        raw_docs = load_file(source)
    elif source.is_dir():
        raw_docs = load_directory(source)
    else:
        raise FileNotFoundError(f"Path not found: {source}")

    if not raw_docs:
        logger.warning("No documents loaded — nothing to chunk.")
        return []

    return chunk_documents(raw_docs, strategy=strategy,
                           chunk_size=chunk_size, chunk_overlap=chunk_overlap)
