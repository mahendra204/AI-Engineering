"""
app/vectorstore.py
ChromaDB vector store — CRUD operations for the knowledge base.

Responsibilities:
  - Initialise / connect to the persistent ChromaDB collection
  - Add document chunks (deduplicating by source)
  - Delete chunks by source filename
  - List all ingested sources
  - Expose a retriever interface consumed by retriever.py
"""

import hashlib
import logging
from pathlib import Path
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.config import settings
from app.embedder import get_embeddings

logger = logging.getLogger(__name__)


def _doc_id(doc: Document) -> str:
    """
    Generate a stable deterministic ID for a chunk so re-ingesting
    the same file doesn't create duplicate vectors.
    """
    content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()
    source = doc.metadata.get("source", "unknown")
    chunk_idx = doc.metadata.get("chunk_index", 0)
    return f"{Path(source).stem}_{chunk_idx}_{content_hash[:8]}"


class VectorStore:
    """Wrapper around LangChain's Chroma integration."""

    def __init__(self, embeddings: Optional[Embeddings] = None):
        self.embeddings = embeddings or get_embeddings()
        self._db: Optional[Chroma] = None
        self._init_db()

    # ── Initialisation ────────────────────────────────────────────────────

    def _init_db(self) -> None:
        persist_dir = settings.chroma_path
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        self._db = Chroma(
            collection_name=settings.chroma_collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_dir,
        )
        count = self._db._collection.count()
        logger.info(
            f"ChromaDB initialised at '{persist_dir}' "
            f"(collection='{settings.chroma_collection_name}', {count} vectors)"
        )

    # ── Write ─────────────────────────────────────────────────────────────

    def add_documents(
        self,
        documents: List[Document],
        deduplicate: bool = True,
    ) -> int:
        """
        Add chunked documents to the vector store.

        Args:
            documents: List of LangChain Document chunks.
            deduplicate: If True, skip chunks whose ID already exists.

        Returns:
            Number of chunks actually added.
        """
        if not documents:
            logger.warning("add_documents called with empty list.")
            return 0

        ids = [_doc_id(doc) for doc in documents]

        if deduplicate:
            existing = set(self._db._collection.get(ids=ids)["ids"])
            new_pairs = [
                (doc, id_) for doc, id_ in zip(documents, ids)
                if id_ not in existing
            ]
            if not new_pairs:
                logger.info("All chunks already exist — nothing to add.")
                return 0
            documents, ids = zip(*new_pairs)
            documents, ids = list(documents), list(ids)

        self._db.add_documents(documents=documents, ids=ids)
        logger.info(f"Added {len(documents)} chunk(s) to ChromaDB")
        return len(documents)

    # ── Delete ────────────────────────────────────────────────────────────

    def delete_source(self, source_name: str) -> int:
        """
        Remove all chunks that came from a given source file.

        Args:
            source_name: filename (e.g. 'report.pdf') or full path.

        Returns:
            Number of chunks deleted.
        """
        # Search by both 'source' and 'filename' metadata fields
        results = self._db._collection.get(
            where={"filename": {"$eq": source_name}}
        )
        ids = results.get("ids", [])

        if not ids:
            logger.warning(f"No chunks found for source '{source_name}'")
            return 0

        self._db._collection.delete(ids=ids)
        logger.info(f"Deleted {len(ids)} chunk(s) for source '{source_name}'")
        return len(ids)

    # ── Query ─────────────────────────────────────────────────────────────

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter_source: Optional[str] = None,
        score_threshold: Optional[float] = None,
    ) -> List[tuple[Document, float]]:
        """
        Semantic similarity search with optional source filter.

        Returns:
            List of (Document, score) tuples, sorted by relevance.
        """
        where = {"filename": {"$eq": filter_source}} if filter_source else None

        results = self._db.similarity_search_with_relevance_scores(
            query=query,
            k=k,
            filter=where,
        )

        if score_threshold is not None:
            results = [(doc, score) for doc, score in results if score >= score_threshold]

        return results

    def as_retriever(self, k: Optional[int] = None, filter_source: Optional[str] = None):
        """Return a LangChain retriever (for use in chains)."""
        search_kwargs = {"k": k or settings.top_k}
        if filter_source:
            search_kwargs["filter"] = {"filename": {"$eq": filter_source}}

        return self._db.as_retriever(
            search_type="mmr",
            search_kwargs={
                **search_kwargs,
                "fetch_k": (k or settings.top_k) * 3,
                "lambda_mult": 1 - settings.mmr_diversity,
            },
        )

    # ── Inspect ───────────────────────────────────────────────────────────

    def list_sources(self) -> List[dict]:
        """
        Return a deduplicated list of ingested sources with chunk counts.
        """
        all_meta = self._db._collection.get(include=["metadatas"])["metadatas"]

        sources: dict[str, dict] = {}
        for meta in all_meta:
            fname = meta.get("filename", "unknown")
            if fname not in sources:
                sources[fname] = {
                    "filename": fname,
                    "source": meta.get("source", fname),
                    "file_type": meta.get("file_type", "unknown"),
                    "chunks": 0,
                }
            sources[fname]["chunks"] += 1

        return sorted(sources.values(), key=lambda x: x["filename"])

    def count(self) -> int:
        """Total number of chunks in the store."""
        return self._db._collection.count()

    def reset(self) -> None:
        """⚠️  Delete ALL vectors from the collection."""
        self._db._collection.delete(where={"chunk_index": {"$gte": 0}})
        logger.warning("Vector store RESET — all chunks deleted.")


# ── Module-level singleton ────────────────────────────────────────────────────

_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    """Return a cached VectorStore singleton."""
    global _store
    if _store is None:
        _store = VectorStore()
    return _store
