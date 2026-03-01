"""reviewmind/api/dependencies.py — FastAPI dependency injection."""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import Request

if TYPE_CHECKING:
    from qdrant_client import AsyncQdrantClient
    from redis.asyncio import Redis
    from sqlalchemy.ext.asyncio import AsyncEngine


def get_db_engine(request: Request) -> AsyncEngine | None:
    """Get the database engine from application state."""
    return getattr(request.app.state, "db_engine", None)


def get_redis(request: Request) -> Redis | None:
    """Get the Redis client from application state."""
    return getattr(request.app.state, "redis", None)


def get_qdrant(request: Request) -> AsyncQdrantClient | None:
    """Get the Qdrant client from application state."""
    return getattr(request.app.state, "qdrant", None)
