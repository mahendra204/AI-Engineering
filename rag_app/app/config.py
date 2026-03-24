"""
app/config.py
Centralised configuration via Pydantic-Settings.
All values can be overridden through the .env file or real environment variables.
"""

from functools import lru_cache
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── App ───────────────────────────────────────────────────────────────
    app_title: str = Field("RAG Knowledge Assistant", alias="APP_TITLE")
    app_host: str = Field("0.0.0.0", alias="APP_HOST")
    app_port: int = Field(8000, alias="APP_PORT")
    log_level: str = Field("INFO", alias="LOG_LEVEL")

    # ── LLM Provider ──────────────────────────────────────────────────────
    llm_provider: Literal["openai", "anthropic", "ollama"] = Field(
        "openai", alias="LLM_PROVIDER"
    )

    # ── OpenAI ────────────────────────────────────────────────────────────
    openai_api_key: str = Field("", alias="OPENAI_API_KEY")
    openai_model: str = Field("gpt-4o-mini", alias="OPENAI_MODEL")
    embedding_model: str = Field("text-embedding-3-small", alias="EMBEDDING_MODEL")

    # ── Anthropic ─────────────────────────────────────────────────────────
    anthropic_api_key: str = Field("", alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field("claude-3-5-haiku-20241022", alias="ANTHROPIC_MODEL")

    # ── Ollama ────────────────────────────────────────────────────────────
    ollama_base_url: str = Field("http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field("llama3.2", alias="OLLAMA_MODEL")
    ollama_embedding_model: str = Field("nomic-embed-text", alias="OLLAMA_EMBEDDING_MODEL")

    # ── Retrieval ─────────────────────────────────────────────────────────
    top_k: int = Field(5, alias="TOP_K")
    retrieval_mode: Literal["semantic", "keyword", "hybrid"] = Field(
        "hybrid", alias="RETRIEVAL_MODE"
    )
    score_threshold: float = Field(0.3, alias="SCORE_THRESHOLD")
    mmr_diversity: float = Field(0.3, alias="MMR_DIVERSITY")

    # ── Chunking ──────────────────────────────────────────────────────────
    chunk_size: int = Field(1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(200, alias="CHUNK_OVERLAP")
    chunk_strategy: Literal["recursive", "semantic", "sentence"] = Field(
        "recursive", alias="CHUNK_STRATEGY"
    )

    # ── Vector Store ──────────────────────────────────────────────────────
    chroma_path: str = Field("vectorstore", alias="CHROMA_PATH")
    chroma_collection_name: str = Field("rag_collection", alias="CHROMA_COLLECTION_NAME")

    # ── Generation ────────────────────────────────────────────────────────
    temperature: float = Field(0.2, alias="TEMPERATURE")
    max_tokens: int = Field(1024, alias="MAX_TOKENS")
    streaming: bool = Field(True, alias="STREAMING")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()


# Convenient module-level shortcut
settings = get_settings()
