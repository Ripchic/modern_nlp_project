"""Tests for reviewmind/main.py — app factory, configure_logging, lifespan."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import structlog
from fastapi import FastAPI

# ── configure_logging ────────────────────────────────────────


class TestConfigureLogging:
    """Test structlog configuration."""

    def test_runs_without_error(self):
        from reviewmind.main import configure_logging

        configure_logging()  # should not raise

    def test_json_output(self, capsys):
        from reviewmind.main import configure_logging

        configure_logging()
        log = structlog.get_logger("test_json")
        log.info("test_event", key="value")
        out = capsys.readouterr().out
        assert "test_event" in out
        assert "key" in out

    def test_timestamp_present(self, capsys):
        from reviewmind.main import configure_logging

        configure_logging()
        log = structlog.get_logger("test_ts")
        log.info("ts_event")
        out = capsys.readouterr().out
        assert "timestamp" in out

    def test_log_level_present(self, capsys):
        from reviewmind.main import configure_logging

        configure_logging()
        log = structlog.get_logger("test_level")
        log.info("level_event")
        out = capsys.readouterr().out
        assert "info" in out.lower()


# ── create_app ───────────────────────────────────────────────


class TestCreateApp:
    """Test FastAPI app factory."""

    def test_returns_fastapi_instance(self):
        with patch("reviewmind.main.configure_logging"):
            from reviewmind.main import create_app

            app = create_app()
        assert isinstance(app, FastAPI)

    def test_app_title(self):
        with patch("reviewmind.main.configure_logging"):
            from reviewmind.main import create_app

            app = create_app()
        assert app.title == "ReviewMind API"

    def test_app_version(self):
        with patch("reviewmind.main.configure_logging"):
            from reviewmind.main import create_app

            app = create_app()
        assert app.version == "0.1.0"

    def test_health_route_registered(self):
        with patch("reviewmind.main.configure_logging"):
            from reviewmind.main import create_app

            app = create_app()
        route_paths = [getattr(r, "path", None) for r in app.routes]
        assert "/health" in route_paths

    def test_docs_route_registered(self):
        with patch("reviewmind.main.configure_logging"):
            from reviewmind.main import create_app

            app = create_app()
        route_paths = [getattr(r, "path", None) for r in app.routes]
        assert "/docs" in route_paths

    def test_module_level_app_exists(self):
        from reviewmind.main import app

        assert isinstance(app, FastAPI)


# ── lifespan ─────────────────────────────────────────────────


class TestLifespan:
    """Test lifespan startup/shutdown resource management."""

    def _make_app(self) -> FastAPI:
        with patch("reviewmind.main.configure_logging"):
            from reviewmind.main import create_app

            return create_app()

    def test_startup_creates_all_clients(self):
        """Lifespan creates engine, redis, qdrant clients when services available."""
        mock_settings = MagicMock()
        mock_settings.database_url = "postgresql+asyncpg://u:p@localhost/db"
        mock_settings.redis_url = "redis://localhost:6379/0"
        mock_settings.qdrant_url = "http://localhost:6333"

        mock_engine = MagicMock()
        mock_engine.dispose = AsyncMock()
        mock_redis = AsyncMock()
        mock_redis.aclose = AsyncMock()
        mock_qdrant = AsyncMock()
        mock_qdrant.close = AsyncMock()

        app = self._make_app()

        with (
            patch("reviewmind.config.settings", mock_settings),
            patch("sqlalchemy.ext.asyncio.create_async_engine", return_value=mock_engine),
            patch("redis.asyncio.from_url", return_value=mock_redis),
            patch("qdrant_client.AsyncQdrantClient", return_value=mock_qdrant),
        ):
            from starlette.testclient import TestClient

            with TestClient(app):
                assert app.state.db_engine is mock_engine
                assert app.state.redis is mock_redis
                assert app.state.qdrant is mock_qdrant

    def test_startup_handles_postgres_failure(self):
        """Lifespan sets db_engine=None if PostgreSQL init fails."""
        mock_settings = MagicMock()
        mock_settings.database_url = "bad://url"
        mock_settings.redis_url = "redis://localhost:6379/0"
        mock_settings.qdrant_url = "http://localhost:6333"

        mock_redis = AsyncMock()
        mock_redis.aclose = AsyncMock()
        mock_qdrant = AsyncMock()
        mock_qdrant.close = AsyncMock()

        app = self._make_app()

        with (
            patch("reviewmind.config.settings", mock_settings),
            patch("sqlalchemy.ext.asyncio.create_async_engine", side_effect=Exception("pg fail")),
            patch("redis.asyncio.from_url", return_value=mock_redis),
            patch("qdrant_client.AsyncQdrantClient", return_value=mock_qdrant),
        ):
            from starlette.testclient import TestClient

            with TestClient(app):
                assert app.state.db_engine is None
                assert app.state.redis is mock_redis
                assert app.state.qdrant is mock_qdrant

    def test_startup_handles_all_failures(self):
        """Lifespan sets all clients to None when all services fail."""
        mock_settings = MagicMock()

        app = self._make_app()

        with (
            patch("reviewmind.config.settings", mock_settings),
            patch("sqlalchemy.ext.asyncio.create_async_engine", side_effect=Exception("pg")),
            patch("redis.asyncio.from_url", side_effect=Exception("redis")),
            patch("qdrant_client.AsyncQdrantClient", side_effect=Exception("qdrant")),
        ):
            from starlette.testclient import TestClient

            with TestClient(app):
                assert app.state.db_engine is None
                assert app.state.redis is None
                assert app.state.qdrant is None

    def test_shutdown_disposes_engine(self):
        """Lifespan calls dispose() on engine at shutdown."""
        mock_settings = MagicMock()
        mock_engine = MagicMock()
        mock_engine.dispose = AsyncMock()
        mock_redis = AsyncMock()
        mock_redis.aclose = AsyncMock()
        mock_qdrant = AsyncMock()
        mock_qdrant.close = AsyncMock()

        app = self._make_app()

        with (
            patch("reviewmind.config.settings", mock_settings),
            patch("sqlalchemy.ext.asyncio.create_async_engine", return_value=mock_engine),
            patch("redis.asyncio.from_url", return_value=mock_redis),
            patch("qdrant_client.AsyncQdrantClient", return_value=mock_qdrant),
        ):
            from starlette.testclient import TestClient

            with TestClient(app):
                pass  # lifespan enters and exits

        mock_engine.dispose.assert_awaited_once()
        mock_redis.aclose.assert_awaited_once()
        mock_qdrant.close.assert_awaited_once()
