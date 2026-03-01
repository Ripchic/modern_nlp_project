# reviewmind/db/session.py — Async session factory (asyncpg + SQLAlchemy)
from __future__ import annotations

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from reviewmind.db.models import Base


def build_engine(database_url: str, *, echo: bool = False, pool_size: int = 10, max_overflow: int = 20) -> AsyncEngine:
    """Create an async SQLAlchemy engine from a DATABASE_URL.

    The URL must use ``postgresql+asyncpg://`` scheme.
    """
    return create_async_engine(
        database_url,
        echo=echo,
        pool_size=pool_size,
        max_overflow=max_overflow,
        pool_pre_ping=True,
    )


def build_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Return a session factory bound to the given engine."""
    return async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=False,
        autocommit=False,
    )


async def get_session(session_factory: async_sessionmaker[AsyncSession]) -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency: yield a session, then close it."""
    async with session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def create_all_tables(engine: AsyncEngine) -> None:
    """Create all tables defined in Base metadata (for testing / dev)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def drop_all_tables(engine: AsyncEngine) -> None:
    """Drop all tables defined in Base metadata (for testing / dev)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
