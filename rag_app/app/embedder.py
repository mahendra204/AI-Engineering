"""
app/embedder.py
Embedding model factory.

Supports:
  openai   → text-embedding-3-small / text-embedding-3-large
  anthropic → (uses OpenAI embeddings; Anthropic doesn't expose an embedding API)
  ollama   → nomic-embed-text (local)

Usage:
    from app.embedder import get_embeddings
    embeddings = get_embeddings()
"""

import logging
from functools import lru_cache
from typing import Optional

from langchain_core.embeddings import Embeddings

from app.config import settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_embeddings(provider: Optional[str] = None) -> Embeddings:
    """
    Return a cached embedding model instance.

    Args:
        provider: Override the provider from settings.
                  One of "openai", "ollama".
    """
    provider = provider or settings.llm_provider

    # Anthropic doesn't have an embedding API → fall back to OpenAI
    if provider == "anthropic":
        logger.info(
            "Anthropic doesn't provide embedding models. "
            "Falling back to OpenAI embeddings."
        )
        provider = "openai"

    if provider == "openai":
        return _openai_embeddings()
    elif provider == "ollama":
        return _ollama_embeddings()
    else:
        raise ValueError(
            f"Unknown embedding provider: '{provider}'. "
            "Choose from: openai, ollama"
        )


def _openai_embeddings() -> Embeddings:
    """OpenAI text-embedding-3-small (1536 dims, cheap + fast)."""
    if not settings.openai_api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. "
            "Add it to your .env file."
        )
    try:
        from langchain_openai import OpenAIEmbeddings
    except ImportError:
        raise ImportError("pip install langchain-openai")

    logger.info(f"Using OpenAI embedding model: {settings.embedding_model}")
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        openai_api_key=settings.openai_api_key,
    )


def _ollama_embeddings() -> Embeddings:
    """Local Ollama embeddings (e.g. nomic-embed-text)."""
    try:
        from langchain_community.embeddings import OllamaEmbeddings
    except ImportError:
        raise ImportError("pip install langchain-community")

    logger.info(
        f"Using Ollama embedding model: {settings.ollama_embedding_model} "
        f"at {settings.ollama_base_url}"
    )
    return OllamaEmbeddings(
        base_url=settings.ollama_base_url,
        model=settings.ollama_embedding_model,
    )
