"""
app/rag_pipeline.py
End-to-end RAG orchestrator — the single entry point for both ingest and query.

Usage:
    pipeline = RAGPipeline()

    # Ingest
    n = pipeline.ingest("data/report.pdf")

    # Query
    result = pipeline.query("What are the key findings?")
    print(result.answer)
    for src in result.sources:
        print(src.filename, src.score)

    # Stream
    for token in pipeline.stream("Summarise the methodology."):
        print(token, end="", flush=True)
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Iterator, List, Optional

from langchain_core.documents import Document

from app.config import settings
from app.ingestor import ingest_from_path
from app.retriever import HybridRetriever, get_retriever
from app.vectorstore import VectorStore, get_vector_store
from app.generator import (
    generate_answer,
    stream_answer,
    astream_answer,
    count_tokens,
    format_context,
)

logger = logging.getLogger(__name__)


# ── Response models ───────────────────────────────────────────────────────────

@dataclass
class SourceDocument:
    content: str
    source: str
    filename: str
    page: Optional[int]
    score: float
    file_type: str = "unknown"


@dataclass
class QueryResult:
    question: str
    answer: str
    sources: List[SourceDocument]
    retrieval_mode: str
    model: str
    latency_ms: float
    tokens_used: int = 0


# ── Pipeline ──────────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    High-level RAG pipeline.

    Wires together:
      ingestor  → vector store       (write path)
      retriever → generator          (read / query path)
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        retriever: Optional[HybridRetriever] = None,
    ):
        self.vs = vector_store or get_vector_store()
        self.retriever = retriever or get_retriever()

    # ── Ingest ────────────────────────────────────────────────────────────

    def ingest(
        self,
        source: str | Path,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        chunk_strategy: Optional[str] = None,
    ) -> int:
        """
        Load, chunk and embed documents from a file or directory.

        Args:
            source: Path to a file or directory.

        Returns:
            Number of chunks added to the vector store.
        """
        logger.info(f"Ingesting: {source}")
        t0 = time.perf_counter()

        chunks = ingest_from_path(
            source,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy=chunk_strategy,
        )
        if not chunks:
            logger.warning(f"No chunks produced from {source}")
            return 0

        added = self.vs.add_documents(chunks)

        # Rebuild BM25 index after adding new data
        self.retriever.invalidate_bm25()

        elapsed = (time.perf_counter() - t0) * 1000
        logger.info(f"Ingested {added} chunks in {elapsed:.0f}ms from '{source}'")
        return added

    def ingest_bytes(
        self,
        content: bytes,
        filename: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> int:
        """
        Ingest a file from raw bytes (e.g. an API upload).
        Writes to a temp file then delegates to ingest().
        """
        import tempfile
        suffix = Path(filename).suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = Path(tmp.name)

        try:
            # Preserve original filename in metadata by renaming
            named_path = tmp_path.parent / filename
            tmp_path.rename(named_path)
            return self.ingest(named_path, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        finally:
            named_path.unlink(missing_ok=True)

    # ── Query ─────────────────────────────────────────────────────────────

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        filter_source: Optional[str] = None,
        score_threshold: Optional[float] = None,
    ) -> QueryResult:
        """
        Full RAG query: retrieve + generate.

        Args:
            question:         The user's question.
            top_k:            Number of chunks to retrieve (default: settings.top_k).
            filter_source:    Only search within this document filename.
            score_threshold:  Minimum relevance score filter.

        Returns:
            QueryResult with answer, sources, and metadata.
        """
        t0 = time.perf_counter()
        k = top_k or settings.top_k

        # 1. Retrieve
        scored_docs = self.retriever.retrieve_with_scores(
            question,
            k=k,
            filter_source=filter_source,
        )

        docs = [doc for doc, _ in scored_docs]

        # 2. Generate
        answer = generate_answer(question, docs)

        # 3. Build result
        sources = [
            SourceDocument(
                content=doc.page_content,
                source=doc.metadata.get("source", ""),
                filename=doc.metadata.get("filename", "unknown"),
                page=doc.metadata.get("page"),
                score=round(score, 4),
                file_type=doc.metadata.get("file_type", "unknown"),
            )
            for doc, score in scored_docs
        ]

        context_text = format_context(docs)
        tokens = count_tokens(question + context_text + answer)

        latency = (time.perf_counter() - t0) * 1000
        logger.info(f"Query answered in {latency:.0f}ms | {tokens} tokens | {len(docs)} sources")

        return QueryResult(
            question=question,
            answer=answer,
            sources=sources,
            retrieval_mode=settings.retrieval_mode,
            model=settings.openai_model if settings.llm_provider == "openai" else settings.anthropic_model,
            latency_ms=round(latency, 1),
            tokens_used=tokens,
        )

    def stream(
        self,
        question: str,
        top_k: Optional[int] = None,
        filter_source: Optional[str] = None,
    ) -> Iterator[str]:
        """
        Streaming RAG query — yields answer tokens as they are generated.

        Note: sources are not returned in streaming mode; call query() for full metadata.
        """
        k = top_k or settings.top_k
        docs = self.retriever.retrieve(question, k=k, filter_source=filter_source)
        yield from stream_answer(question, docs)

    async def astream(
        self,
        question: str,
        top_k: Optional[int] = None,
        filter_source: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Async streaming for FastAPI SSE."""
        k = top_k or settings.top_k
        docs = self.retriever.retrieve(question, k=k, filter_source=filter_source)
        async for token in astream_answer(question, docs):
            yield token

    # ── Knowledge base management ─────────────────────────────────────────

    def list_sources(self) -> List[dict]:
        """List all documents in the knowledge base."""
        return self.vs.list_sources()

    def delete_source(self, filename: str) -> int:
        """Remove a document from the knowledge base by filename."""
        deleted = self.vs.delete_source(filename)
        self.retriever.invalidate_bm25()
        return deleted

    def count(self) -> int:
        """Total chunks in the knowledge base."""
        return self.vs.count()


# ── Singleton ─────────────────────────────────────────────────────────────────

_pipeline: Optional[RAGPipeline] = None


def get_pipeline() -> RAGPipeline:
    """Return a cached RAGPipeline singleton."""
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline
