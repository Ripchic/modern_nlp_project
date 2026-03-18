"""reviewmind/main.py — FastAPI app factory + lifespan + structlog logging."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

import structlog
from fastapi import FastAPI

from reviewmind.api.router import api_router


def configure_logging() -> None:
    """Configure structlog for JSON logging to stdout."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
    # Reduce noise from third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Manage application lifecycle: connect/disconnect resources."""
    from reviewmind.config import settings

    log = structlog.get_logger("reviewmind.lifespan")
    log.info("startup_begin")

    # ── PostgreSQL ────────────────────────────────────────────
    try:
        from sqlalchemy.ext.asyncio import create_async_engine

        engine = create_async_engine(
            settings.database_url,
            pool_pre_ping=True,
            echo=False,
        )
        app.state.db_engine = engine
        log.info("postgres_engine_created")
    except Exception as exc:
        log.warning("postgres_init_failed", error=str(exc))
        app.state.db_engine = None

    # ── Redis ─────────────────────────────────────────────────
    try:
        from redis.asyncio import from_url as redis_from_url

        client = redis_from_url(settings.redis_url, decode_responses=True)
        app.state.redis = client
        log.info("redis_client_created")
    except Exception as exc:
        log.warning("redis_init_failed", error=str(exc))
        app.state.redis = None

    # ── Qdrant ────────────────────────────────────────────────
    try:
        from qdrant_client import AsyncQdrantClient

        qdrant = AsyncQdrantClient(url=settings.qdrant_url, timeout=5)
        app.state.qdrant = qdrant
        log.info("qdrant_client_created")
    except Exception as exc:
        log.warning("qdrant_init_failed", error=str(exc))
        app.state.qdrant = None

    log.info("startup_complete")
    yield

    # ── Shutdown ──────────────────────────────────────────────
    if getattr(app.state, "db_engine", None) is not None:
        await app.state.db_engine.dispose()
        log.info("postgres_disconnected")

    if getattr(app.state, "redis", None) is not None:
        await app.state.redis.aclose()
        log.info("redis_disconnected")

    if getattr(app.state, "qdrant", None) is not None:
        await app.state.qdrant.close()
        log.info("qdrant_disconnected")

    log.info("shutdown_complete")


def create_app() -> FastAPI:
    """FastAPI application factory."""
    configure_logging()

    application = FastAPI(
        title="ReviewMind API",
        description="AI-агрегатор обзоров для принятия решений о покупке",
        version="0.1.0",
        lifespan=lifespan,
    )
    application.include_router(api_router)

    # ── Rate limiting (slowapi) ───────────────────────────────
    from reviewmind.api.rate_limit import setup_rate_limiting

    setup_rate_limiting(application)

    return application


app = create_app()
