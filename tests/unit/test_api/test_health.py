"""Tests for reviewmind/api/endpoints/health.py — GET /health endpoint."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from reviewmind.api.endpoints.health import HealthResponse, router


@pytest.fixture()
def simple_app() -> FastAPI:
    """Create a minimal app with health route — no lifespan, no external deps."""
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture()
def client(simple_app: FastAPI) -> TestClient:
    """TestClient bound to the simple app."""
    return TestClient(simple_app)


# ── Basic response tests ────────────────────────────────────


class TestHealthResponseSchema:
    """Test the HealthResponse pydantic model."""

    def test_schema_fields(self):
        resp = HealthResponse(status="ok", postgres=True, redis=True, qdrant=True)
        assert resp.status == "ok"
        assert resp.postgres is True

    def test_schema_degraded(self):
        resp = HealthResponse(status="degraded", postgres=False, redis=True, qdrant=False)
        assert resp.status == "degraded"
        assert resp.postgres is False


# ── All services down ────────────────────────────────────────


class TestHealthAllDown:
    """Test health when no services are available (state not set)."""

    def test_returns_200(self, client: TestClient):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_status_degraded(self, client: TestClient):
        resp = client.get("/health")
        data = resp.json()
        assert data["status"] == "degraded"

    def test_all_booleans_false(self, client: TestClient):
        resp = client.get("/health")
        data = resp.json()
        assert data["postgres"] is False
        assert data["redis"] is False
        assert data["qdrant"] is False

    def test_response_content_type(self, client: TestClient):
        resp = client.get("/health")
        assert resp.headers["content-type"] == "application/json"

    def test_response_has_exactly_four_fields(self, client: TestClient):
        resp = client.get("/health")
        assert set(resp.json().keys()) == {"status", "postgres", "redis", "qdrant"}


# ── All services up ──────────────────────────────────────────


class TestHealthAllUp:
    """Test health when all services respond."""

    @pytest.fixture()
    def all_up_app(self, simple_app: FastAPI) -> FastAPI:
        # Mock postgres engine
        mock_engine = MagicMock()
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_conn)
        mock_ctx.__aexit__ = AsyncMock(return_value=None)
        mock_engine.connect.return_value = mock_ctx

        # Mock redis
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=True)

        # Mock qdrant
        mock_qdrant = AsyncMock()
        mock_qdrant.get_collections = AsyncMock(return_value=[])

        simple_app.state.db_engine = mock_engine
        simple_app.state.redis = mock_redis
        simple_app.state.qdrant = mock_qdrant
        return simple_app

    def test_status_ok(self, all_up_app: FastAPI):
        with TestClient(all_up_app) as c:
            data = c.get("/health").json()
        assert data["status"] == "ok"

    def test_all_booleans_true(self, all_up_app: FastAPI):
        with TestClient(all_up_app) as c:
            data = c.get("/health").json()
        assert data["postgres"] is True
        assert data["redis"] is True
        assert data["qdrant"] is True


# ── Partial failures ─────────────────────────────────────────


class TestHealthPartial:
    """Test health when some services are down."""

    def test_only_redis_up(self, simple_app: FastAPI):
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(return_value=True)
        simple_app.state.redis = mock_redis

        with TestClient(simple_app) as c:
            data = c.get("/health").json()
        assert data["status"] == "degraded"
        assert data["redis"] is True
        assert data["postgres"] is False
        assert data["qdrant"] is False

    def test_only_qdrant_up(self, simple_app: FastAPI):
        mock_qdrant = AsyncMock()
        mock_qdrant.get_collections = AsyncMock(return_value=[])
        simple_app.state.qdrant = mock_qdrant

        with TestClient(simple_app) as c:
            data = c.get("/health").json()
        assert data["status"] == "degraded"
        assert data["qdrant"] is True
        assert data["postgres"] is False
        assert data["redis"] is False


# ── Exception handling ───────────────────────────────────────


class TestHealthExceptions:
    """Test health when services raise exceptions."""

    def test_postgres_connection_error(self, simple_app: FastAPI):
        mock_engine = MagicMock()
        mock_engine.connect.side_effect = Exception("Connection refused")
        simple_app.state.db_engine = mock_engine

        with TestClient(simple_app) as c:
            data = c.get("/health").json()
        assert data["postgres"] is False

    def test_redis_connection_error(self, simple_app: FastAPI):
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(side_effect=Exception("Connection refused"))
        simple_app.state.redis = mock_redis

        with TestClient(simple_app) as c:
            data = c.get("/health").json()
        assert data["redis"] is False

    def test_qdrant_connection_error(self, simple_app: FastAPI):
        mock_qdrant = AsyncMock()
        mock_qdrant.get_collections = AsyncMock(side_effect=Exception("Connection refused"))
        simple_app.state.qdrant = mock_qdrant

        with TestClient(simple_app) as c:
            data = c.get("/health").json()
        assert data["qdrant"] is False

    def test_all_services_raise_still_returns_200(self, simple_app: FastAPI):
        mock_engine = MagicMock()
        mock_engine.connect.side_effect = Exception("pg fail")
        simple_app.state.db_engine = mock_engine

        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock(side_effect=Exception("redis fail"))
        simple_app.state.redis = mock_redis

        mock_qdrant = AsyncMock()
        mock_qdrant.get_collections = AsyncMock(side_effect=Exception("qdrant fail"))
        simple_app.state.qdrant = mock_qdrant

        with TestClient(simple_app) as c:
            resp = c.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["postgres"] is False
        assert data["redis"] is False
        assert data["qdrant"] is False
