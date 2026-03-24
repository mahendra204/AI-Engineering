"""
app/api.py
FastAPI route definitions for the RAG application.

Routes:
  GET  /api/health              → health check
  POST /api/query               → ask a question (JSON)
  GET  /api/query/stream        → streaming query (SSE)
  POST /api/ingest              → upload & ingest a document
  GET  /api/sources             → list all ingested sources
  DELETE /api/sources/{name}    → remove a source
  GET  /api/stats               → knowledge base statistics
"""

import logging
import time
from typing import AsyncIterator, List, Optional

from fastapi import APIRouter, HTTPException, Query, UploadFile, File, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.rag_pipeline import get_pipeline

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["RAG"])


# ── Request / Response schemas ────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question to ask")
    top_k: int = Field(5, ge=1, le=20, description="Number of chunks to retrieve")
    filter_source: Optional[str] = Field(None, description="Limit search to this filename")
    score_threshold: Optional[float] = Field(None, ge=0.0, le=1.0)


class SourceInfo(BaseModel):
    content: str
    source: str
    filename: str
    page: Optional[int]
    score: float
    file_type: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceInfo]
    retrieval_mode: str
    model: str
    latency_ms: float
    tokens_used: int


class IngestResponse(BaseModel):
    message: str
    filename: str
    chunks_added: int


class SourceListItem(BaseModel):
    filename: str
    source: str
    file_type: str
    chunks: int


class StatsResponse(BaseModel):
    total_chunks: int
    total_sources: int
    sources: List[SourceListItem]


class HealthResponse(BaseModel):
    status: str
    total_chunks: int
    timestamp: float


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse)
def health_check():
    """Check if the API is running and report knowledge base size."""
    pipeline = get_pipeline()
    return HealthResponse(
        status="ok",
        total_chunks=pipeline.count(),
        timestamp=time.time(),
    )


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """
    Ask a question against the knowledge base.

    Returns the generated answer plus the source chunks used.
    """
    pipeline = get_pipeline()

    if pipeline.count() == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Knowledge base is empty. Please ingest documents first.",
        )

    try:
        result = pipeline.query(
            question=req.question,
            top_k=req.top_k,
            filter_source=req.filter_source,
            score_threshold=req.score_threshold,
        )
    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    return QueryResponse(
        question=result.question,
        answer=result.answer,
        sources=[
            SourceInfo(
                content=s.content,
                source=s.source,
                filename=s.filename,
                page=s.page,
                score=s.score,
                file_type=s.file_type,
            )
            for s in result.sources
        ],
        retrieval_mode=result.retrieval_mode,
        model=result.model,
        latency_ms=result.latency_ms,
        tokens_used=result.tokens_used,
    )


@router.get("/query/stream")
async def query_stream(
    question: str = Query(..., min_length=1),
    top_k: int = Query(5, ge=1, le=20),
    filter_source: Optional[str] = Query(None),
):
    """
    Streaming query via Server-Sent Events (SSE).
    Connect with EventSource or fetch with ReadableStream.
    """
    pipeline = get_pipeline()

    if pipeline.count() == 0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Knowledge base is empty. Please ingest documents first.",
        )

    async def event_generator() -> AsyncIterator[str]:
        try:
            async for token in pipeline.astream(question, top_k=top_k, filter_source=filter_source):
                # SSE format: "data: <token>\n\n"
                yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Stream error: {e}", exc_info=True)
            yield f"data: [ERROR] {e}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_document(file: UploadFile = File(...)):
    """
    Upload and ingest a document into the knowledge base.

    Supported formats: .pdf, .txt, .md, .docx, .html, .csv
    """
    supported = {".pdf", ".txt", ".md", ".docx", ".html", ".htm", ".csv"}
    from pathlib import Path
    ext = Path(file.filename).suffix.lower()

    if ext not in supported:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported file type '{ext}'. Supported: {', '.join(supported)}",
        )

    try:
        content = await file.read()
        pipeline = get_pipeline()
        added = pipeline.ingest_bytes(content, file.filename)
    except Exception as e:
        logger.error(f"Ingest error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    return IngestResponse(
        message=f"Successfully ingested '{file.filename}' — {added} chunks added.",
        filename=file.filename,
        chunks_added=added,
    )


@router.get("/sources", response_model=List[SourceListItem])
def list_sources():
    """List all documents currently in the knowledge base."""
    pipeline = get_pipeline()
    return pipeline.list_sources()


@router.delete("/sources/{source_name}", status_code=status.HTTP_200_OK)
def delete_source(source_name: str):
    """Remove a document and all its chunks from the knowledge base."""
    pipeline = get_pipeline()
    deleted = pipeline.delete_source(source_name)
    if deleted == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Source '{source_name}' not found in knowledge base.",
        )
    return {"message": f"Deleted {deleted} chunks for '{source_name}'"}


@router.get("/stats", response_model=StatsResponse)
def get_stats():
    """Return knowledge base statistics."""
    pipeline = get_pipeline()
    sources = pipeline.list_sources()
    return StatsResponse(
        total_chunks=pipeline.count(),
        total_sources=len(sources),
        sources=sources,
    )
