"""
app/generator.py
LLM answer generation with prompt engineering.

Supports:
  openai    → gpt-4o-mini (default), gpt-4o, gpt-3.5-turbo, …
  anthropic → claude-3-5-haiku-20241022, claude-3-5-sonnet, …
  ollama    → llama3.2, mistral, …

Usage:
    from app.generator import get_llm, build_rag_chain
"""

import logging
from functools import lru_cache
from typing import AsyncIterator, Iterator, List, Optional

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough

from app.config import settings

logger = logging.getLogger(__name__)


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a knowledgeable and precise AI assistant.
Your job is to answer questions using ONLY the information provided in the context below.

Guidelines:
- Answer directly and concisely based on the context.
- If the context doesn't contain enough information to answer, say: "I don't have enough information in the provided documents to answer that."
- Never make up facts or use knowledge outside the provided context.
- When possible, cite the source document (filename and page if available).
- Use bullet points or numbered lists for clarity when listing multiple items.
- Keep answers focused and avoid unnecessary repetition.

Context:
{context}
"""

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "{question}"),
])


# ── Context formatter ─────────────────────────────────────────────────────────

def format_context(docs: List[Document]) -> str:
    """
    Format retrieved documents into a readable context block.
    Includes source metadata as citations.
    """
    if not docs:
        return "No relevant context found."

    parts = []
    for i, doc in enumerate(docs, start=1):
        meta = doc.metadata
        source = meta.get("filename", meta.get("source", "unknown"))
        page = meta.get("page")
        citation = f"[Source {i}: {source}" + (f", page {page}]" if page else "]")
        parts.append(f"{citation}\n{doc.page_content.strip()}")

    return "\n\n---\n\n".join(parts)


# ── LLM factory ───────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_llm(provider: Optional[str] = None) -> BaseChatModel:
    """Return a cached LLM instance."""
    provider = provider or settings.llm_provider

    if provider == "openai":
        return _openai_llm()
    elif provider == "anthropic":
        return _anthropic_llm()
    elif provider == "ollama":
        return _ollama_llm()
    else:
        raise ValueError(f"Unknown LLM provider: '{provider}'")


def _openai_llm() -> BaseChatModel:
    if not settings.openai_api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set in .env")
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        raise ImportError("pip install langchain-openai")

    logger.info(f"Using OpenAI LLM: {settings.openai_model}")
    return ChatOpenAI(
        model=settings.openai_model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        openai_api_key=settings.openai_api_key,
        streaming=settings.streaming,
    )


def _anthropic_llm() -> BaseChatModel:
    if not settings.anthropic_api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY is not set in .env")
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError:
        raise ImportError("pip install langchain-anthropic")

    logger.info(f"Using Anthropic LLM: {settings.anthropic_model}")
    return ChatAnthropic(
        model=settings.anthropic_model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
        anthropic_api_key=settings.anthropic_api_key,
    )


def _ollama_llm() -> BaseChatModel:
    try:
        from langchain_community.chat_models import ChatOllama
    except ImportError:
        raise ImportError("pip install langchain-community")

    logger.info(f"Using Ollama LLM: {settings.ollama_model} @ {settings.ollama_base_url}")
    return ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=settings.temperature,
    )


# ── Generation functions ──────────────────────────────────────────────────────

def generate_answer(
    question: str,
    docs: List[Document],
    llm: Optional[BaseChatModel] = None,
) -> str:
    """
    Generate a grounded answer from retrieved documents.

    Args:
        question: The user's question.
        docs:     Retrieved context documents.
        llm:      Optional LLM override.

    Returns:
        The generated answer as a string.
    """
    llm = llm or get_llm()
    context = format_context(docs)

    chain = RAG_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"question": question, "context": context})
    return answer


def stream_answer(
    question: str,
    docs: List[Document],
    llm: Optional[BaseChatModel] = None,
) -> Iterator[str]:
    """
    Stream the answer token-by-token.

    Yields:
        String tokens as they are generated.
    """
    llm = llm or get_llm()
    context = format_context(docs)

    chain = RAG_PROMPT | llm | StrOutputParser()
    for token in chain.stream({"question": question, "context": context}):
        yield token


async def astream_answer(
    question: str,
    docs: List[Document],
    llm: Optional[BaseChatModel] = None,
) -> AsyncIterator[str]:
    """
    Async streaming version for use with FastAPI SSE endpoints.
    """
    llm = llm or get_llm()
    context = format_context(docs)
    chain = RAG_PROMPT | llm | StrOutputParser()

    async for token in chain.astream({"question": question, "context": context}):
        yield token


def count_tokens(text: str, model: Optional[str] = None) -> int:
    """Estimate token count using tiktoken."""
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model(model or settings.openai_model)
        return len(enc.encode(text))
    except Exception:
        # Rough estimate: ~4 chars per token
        return len(text) // 4
