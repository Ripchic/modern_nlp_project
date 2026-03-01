"""Unit tests for db/session.py (TASK-009)."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from reviewmind.db.session import (
    build_engine,
    build_session_factory,
    create_all_tables,
    drop_all_tables,
)


class TestBuildEngine:
    def test_returns_async_engine(self):
        engine = build_engine("postgresql+asyncpg://user:pass@localhost/testdb")
        assert isinstance(engine, AsyncEngine)

    def test_pool_size_and_overflow(self):
        engine = build_engine(
            "postgresql+asyncpg://user:pass@localhost/testdb",
            pool_size=5,
            max_overflow=10,
        )
        assert engine.pool.size() == 5

    def test_echo_false_by_default(self):
        engine = build_engine("postgresql+asyncpg://user:pass@localhost/testdb")
        assert engine.echo is False

    def test_echo_can_be_enabled(self):
        engine = build_engine("postgresql+asyncpg://user:pass@localhost/testdb", echo=True)
        # echo can be True or callable that returns True
        assert engine.echo


class TestBuildSessionFactory:
    def test_returns_session_maker(self):
        engine = build_engine("postgresql+asyncpg://user:pass@localhost/testdb")
        factory = build_session_factory(engine)
        assert isinstance(factory, async_sessionmaker)

    def test_session_factory_expire_on_commit_false(self):
        engine = build_engine("postgresql+asyncpg://user:pass@localhost/testdb")
        factory = build_session_factory(engine)
        # Check the kw settings on the factory
        assert factory.kw.get("expire_on_commit") is False


class TestTableOps:
    @pytest.mark.asyncio
    async def test_create_all_tables_calls_run_sync(self):
        engine = MagicMock()
        conn = AsyncMock()
        conn.run_sync = AsyncMock()

        engine.begin = MagicMock()
        engine.begin.return_value.__aenter__ = AsyncMock(return_value=conn)
        engine.begin.return_value.__aexit__ = AsyncMock(return_value=False)

        await create_all_tables(engine)
        conn.run_sync.assert_called_once()

    @pytest.mark.asyncio
    async def test_drop_all_tables_calls_run_sync(self):
        engine = MagicMock()
        conn = AsyncMock()
        conn.run_sync = AsyncMock()

        engine.begin = MagicMock()
        engine.begin.return_value.__aenter__ = AsyncMock(return_value=conn)
        engine.begin.return_value.__aexit__ = AsyncMock(return_value=False)

        await drop_all_tables(engine)
        conn.run_sync.assert_called_once()


class TestRepositoryImports:
    def test_repositories_package_exports(self):
        from reviewmind.db.repositories import (
            JobRepository,
            QueryLogRepository,
            SourceRepository,
            SubscriptionRepository,
            UserLimitRepository,
            UserRepository,
        )

        assert UserRepository.__name__ == "UserRepository"
        assert UserLimitRepository.__name__ == "UserLimitRepository"
        assert SourceRepository.__name__ == "SourceRepository"
        assert JobRepository.__name__ == "JobRepository"
        assert SubscriptionRepository.__name__ == "SubscriptionRepository"
        assert QueryLogRepository.__name__ == "QueryLogRepository"
