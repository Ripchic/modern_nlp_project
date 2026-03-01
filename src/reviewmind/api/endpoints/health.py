"""reviewmind/api/endpoints/health.py — GET /health endpoint."""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter()
logger = structlog.get_logger("reviewmind.health")


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str
    postgres: bool
    redis: bool
    qdrant: bool


async def _check_postgres(request: Request) -> bool:
    """Check PostgreSQL connectivity by executing a simple query."""
    engine = getattr(request.app.state, "db_engine", None)
    if engine is None:
        return False
    try:
        from sqlalchemy import text

        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


async def _check_redis(request: Request) -> bool:
    """Check Redis connectivity via PING command."""
    client = getattr(request.app.state, "redis", None)
    if client is None:
        return False
    try:
        result = await client.ping()
        return bool(result)
    except Exception:
        return False


async def _check_qdrant(request: Request) -> bool:
    """Check Qdrant connectivity by listing collections."""
    client = getattr(request.app.state, "qdrant", None)
    if client is None:
        return False
    try:
        await client.get_collections()
        return True
    except Exception:
        return False


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """Return health status of all backing services."""
    postgres = await _check_postgres(request)
    redis = await _check_redis(request)
    qdrant = await _check_qdrant(request)

    all_healthy = postgres and redis and qdrant
    status = "ok" if all_healthy else "degraded"

    logger.info("health_check", status=status, postgres=postgres, redis=redis, qdrant=qdrant)

    return HealthResponse(
        status=status,
        postgres=postgres,
        redis=redis,
        qdrant=qdrant,
    )
