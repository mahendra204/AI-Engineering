"""
main.py
FastAPI application entrypoint.

Run:
    uvicorn main:app --reload --port 8000

Then visit:
    http://localhost:8000/docs  → Swagger UI
    http://localhost:8000/redoc → ReDoc
"""

import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import router
from app.config import settings

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialise heavy singletons on startup so first request is fast."""
    logger.info("🚀  Starting RAG API...")
    from app.rag_pipeline import get_pipeline
    pipeline = get_pipeline()
    logger.info(f"✅  Knowledge base ready — {pipeline.count()} chunks loaded")
    yield
    logger.info("🛑  Shutting down RAG API")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title=settings.app_title,
    description=(
        "A production-ready Retrieval-Augmented Generation API. "
        "Ingest documents, then ask questions against your knowledge base."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — adjust origins for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(router)


# ── Root ──────────────────────────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    return JSONResponse({
        "message": f"Welcome to {settings.app_title}",
        "docs": "/docs",
        "health": "/api/health",
    })


# ── Dev runner ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=True,
        log_level=settings.log_level.lower(),
    )
