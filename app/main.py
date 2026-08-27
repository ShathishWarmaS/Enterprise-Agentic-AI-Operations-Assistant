"""FastAPI application entrypoint.

Run: uvicorn app.main:app --reload
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.routes import router
from app.config import settings
from app.data.database import init_db
from app.services.container import get_container

logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.configure_logging()
    init_db()
    container = get_container()
    logger.info(
        "startup: llm_mode=%s vector_backend=%s session_backend=%s",
        container.llm.mode,
        container.vector_store.backend,
        container.session_store.backend_name,
    )
    try:
        yield
    finally:
        container.job_queue.shutdown()


app = FastAPI(
    title="Enterprise Agentic AI Operations Assistant",
    version="0.1.0",
    summary="Multi-agent RAG over messy enterprise data with grounded, cited outputs.",
    lifespan=lifespan,
)
app.include_router(router)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # Log the full trace server-side; return a generic message to the client.
    logger.exception("unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"detail": "internal error", "type": exc.__class__.__name__},
    )


@app.get("/", include_in_schema=False)
async def root() -> dict:
    return {"service": "enterprise-agentic-ai", "docs": "/docs", "health": "/health"}
