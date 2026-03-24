"""
app/retriever.py
Hybrid retrieval: combines ChromaDB semantic search with BM25 keyword search.

Modes (controlled by settings.retrieval_mode):
  semantic → pure vector similarity (ChromaDB)
  keyword  → pure BM25 keyword match
  hybrid   → Reciprocal Rank Fusion of both (default)
"""

import logging
from typing import List, Optional

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi

from app.config import settings
from app.vectorstore import VectorStore, get_vector_store

logger = logging.getLogger(__name__)


# ── BM25 helpers ──────────────────────────────────────────────────────────────

def _tokenise(text: str) -> List[str]:
    """Simple whitespace + lowercase tokeniser for BM25."""
    return text.lower().split()


class BM25Index:
    """In-memory BM25 index built from the current vector store corpus."""

    def __init__(self, corpus: List[Document]):
        self.docs = corpus
        tokenised = [_tokenise(d.page_content) for d in corpus]
        self.bm25 = BM25Okapi(tokenised) if tokenised else None

    def search(self, query: str, k: int = 10) -> List[tuple[Document, float]]:
        if self.bm25 is None or not self.docs:
            return []
        scores = self.bm25.get_scores(_tokenise(query))
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.docs[idx], float(scores[idx])) for idx in top_indices]


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def _reciprocal_rank_fusion(
    results_a: List[tuple[Document, float]],
    results_b: List[tuple[Document, float]],
    k: int = 60,
) -> List[tuple[Document, float]]:
    """
    Combine two ranked result lists using Reciprocal Rank Fusion.
    Higher k → less penalty for low ranks.
    """
    scores: dict[str, float] = {}
    docs:   dict[str, Document] = {}

    for rank, (doc, _) in enumerate(results_a):
        key = doc.page_content[:200]  # Use content prefix as identity key
        scores[key] = scores.get(key, 0) + 1 / (rank + k)
        docs[key] = doc

    for rank, (doc, _) in enumerate(results_b):
        key = doc.page_content[:200]
        scores[key] = scores.get(key, 0) + 1 / (rank + k)
        docs[key] = doc

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(docs[key], score) for key, score in ranked]


# ── Main Retriever ────────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Retriever that blends vector and keyword search.
    """

    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        mode: Optional[str] = None,
    ):
        self.vs = vector_store or get_vector_store()
        self.mode = mode or settings.retrieval_mode
        self._bm25: Optional[BM25Index] = None

    def _get_bm25(self) -> BM25Index:
        """Lazily build a BM25 index from the entire corpus."""
        if self._bm25 is None:
            logger.info("Building BM25 index from vector store corpus...")
            raw = self.vs._db._collection.get(include=["documents", "metadatas"])
            docs = [
                Document(page_content=content, metadata=meta)
                for content, meta in zip(raw["documents"], raw["metadatas"])
            ]
            self._bm25 = BM25Index(docs)
            logger.info(f"BM25 index built with {len(docs)} documents")
        return self._bm25

    def invalidate_bm25(self) -> None:
        """Call this after ingesting new documents to rebuild the index."""
        self._bm25 = None

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        filter_source: Optional[str] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Document]:
        """
        Retrieve the top-k most relevant documents for a query.

        Args:
            query:            The user's question.
            k:                Number of results (default: settings.top_k).
            filter_source:    Only return chunks from this filename.
            score_threshold:  Minimum relevance score for semantic results.

        Returns:
            Ordered list of Document chunks.
        """
        k = k or settings.top_k
        threshold = score_threshold if score_threshold is not None else settings.score_threshold

        if self.mode == "semantic":
            semantic_results = self.vs.similarity_search(
                query, k=k, filter_source=filter_source, score_threshold=threshold
            )
            return [doc for doc, _ in semantic_results]

        elif self.mode == "keyword":
            bm25_results = self._get_bm25().search(query, k=k)
            return [doc for doc, _ in bm25_results[:k]]

        else:  # hybrid (default)
            fetch_k = k * 3  # Over-fetch then merge

            semantic_results = self.vs.similarity_search(
                query, k=fetch_k, filter_source=filter_source, score_threshold=0.0
            )
            bm25_results = self._get_bm25().search(query, k=fetch_k)

            fused = _reciprocal_rank_fusion(semantic_results, bm25_results)

            # Apply score threshold on fused scores (normalised to 0-1 range)
            if fused:
                max_score = max(s for _, s in fused)
                fused = [
                    (doc, score / max_score)
                    for doc, score in fused
                    if (score / max_score) >= threshold
                ]

            return [doc for doc, _ in fused[:k]]

    def retrieve_with_scores(
        self,
        query: str,
        k: Optional[int] = None,
        filter_source: Optional[str] = None,
    ) -> List[tuple[Document, float]]:
        """Same as retrieve() but also returns relevance scores."""
        k = k or settings.top_k

        if self.mode == "semantic":
            return self.vs.similarity_search(query, k=k, filter_source=filter_source)

        elif self.mode == "keyword":
            return self._get_bm25().search(query, k=k)

        else:  # hybrid
            semantic_results = self.vs.similarity_search(query, k=k * 3)
            bm25_results = self._get_bm25().search(query, k=k * 3)
            fused = _reciprocal_rank_fusion(semantic_results, bm25_results)

            if fused:
                max_score = max(s for _, s in fused)
                fused = [(doc, score / max_score) for doc, score in fused]

            return fused[:k]


# ── Singleton ─────────────────────────────────────────────────────────────────

_retriever: Optional[HybridRetriever] = None


def get_retriever(mode: Optional[str] = None) -> HybridRetriever:
    """Return a cached HybridRetriever singleton."""
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever(mode=mode)
    return _retriever
