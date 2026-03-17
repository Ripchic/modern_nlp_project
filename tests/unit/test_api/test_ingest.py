"""Unit tests for TASK-028 — POST /ingest endpoint.

Tests cover:
- IngestRequest / IngestResponse / IngestURLResult schema validation
- POST /ingest endpoint: success, partial failure, Qdrant unavailable, no DB, validation errors
- Router wiring: ingest_router registered in api_router
- Full app integration: /ingest available via create_app()
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from reviewmind.api.endpoints.ingest import router
from reviewmind.api.schemas import IngestRequest, IngestResponse, IngestURLResult
from reviewmind.ingestion.pipeline import IngestionResult, SourceIngestionResult

# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════


def _make_app(**state_attrs) -> FastAPI:
    """Create a minimal FastAPI app with the ingest router and optional state."""
    app = FastAPI()
    app.include_router(router)
    for k, v in state_attrs.items():
        setattr(app.state, k, v)
    return app


def _mock_qdrant() -> MagicMock:
    return MagicMock(name="qdrant")


def _mock_engine() -> MagicMock:
    engine = MagicMock(name="db_engine")
    return engine


def _success_result(url: str, chunks: int = 5) -> SourceIngestionResult:
    return SourceIngestionResult(
        url=url, success=True, source_type="web", chunks_count=chunks, source_id=1,
    )


def _failed_result(url: str, error: str = "Scrape failed") -> SourceIngestionResult:
    return SourceIngestionResult(
        url=url, success=False, source_type="web", error=error,
    )


def _ingestion_result(results: list[SourceIngestionResult]) -> IngestionResult:
    ir = IngestionResult()
    for r in results:
        ir.results.append(r)
        if r.success:
            ir.success_count += 1
            ir.chunks_count += r.chunks_count
        else:
            ir.failed_count += 1
            ir.failed_urls.append(r.url)
    return ir


# ══════════════════════════════════════════════════════════════
# Tests — Schema models
# ══════════════════════════════════════════════════════════════


class TestIngestRequest:
    """Test IngestRequest pydantic model."""

    def test_valid_request(self):
        req = IngestRequest(user_id=123, urls=["https://example.com"])
        assert req.user_id == 123
        assert req.urls == ["https://example.com"]
        assert req.session_id is None
        assert req.product_query == ""

    def test_with_all_fields(self):
        req = IngestRequest(
            user_id=123,
            session_id="sess-1",
            urls=["https://a.com", "https://b.com"],
            product_query="Sony WH-1000XM5",
        )
        assert req.session_id == "sess-1"
        assert len(req.urls) == 2
        assert req.product_query == "Sony WH-1000XM5"

    def test_urls_required(self):
        with pytest.raises(Exception):
            IngestRequest(user_id=123)

    def test_empty_urls_rejected(self):
        with pytest.raises(Exception):
            IngestRequest(user_id=123, urls=[])

    def test_user_id_required(self):
        with pytest.raises(Exception):
            IngestRequest(urls=["https://example.com"])


class TestIngestURLResult:
    """Test IngestURLResult pydantic model."""

    def test_success(self):
        r = IngestURLResult(url="https://a.com", status="success", source_type="web", chunks_count=5)
        assert r.status == "success"
        assert r.error is None

    def test_failed(self):
        r = IngestURLResult(url="https://a.com", status="failed", error="Scrape failed")
        assert r.status == "failed"
        assert r.error == "Scrape failed"

    def test_defaults(self):
        r = IngestURLResult(url="https://a.com", status="success")
        assert r.source_type == ""
        assert r.chunks_count == 0
        assert r.error is None


class TestIngestResponse:
    """Test IngestResponse pydantic model."""

    def test_defaults(self):
        resp = IngestResponse()
        assert resp.results == []
        assert resp.success_count == 0
        assert resp.failed_count == 0
        assert resp.chunks_count == 0

    def test_with_results(self):
        resp = IngestResponse(
            results=[IngestURLResult(url="https://a.com", status="success", chunks_count=3)],
            success_count=1,
            chunks_count=3,
        )
        assert len(resp.results) == 1
        assert resp.success_count == 1
        assert resp.chunks_count == 3

    def test_serialization(self):
        resp = IngestResponse(success_count=2, failed_count=1, chunks_count=10)
        data = resp.model_dump()
        assert data["success_count"] == 2
        assert data["failed_count"] == 1
        assert data["chunks_count"] == 10


# ══════════════════════════════════════════════════════════════
# Tests — POST /ingest endpoint
# ══════════════════════════════════════════════════════════════


class TestIngestEndpointSuccess:
    """Test successful ingestion scenarios."""

    @patch("reviewmind.api.endpoints.ingest.IngestionPipeline")
    def test_single_url_success(self, mock_pipeline_cls):
        pipeline = AsyncMock()
        pipeline.ingest_urls = AsyncMock(return_value=_ingestion_result([
            _success_result("https://example.com", chunks=5),
        ]))
        pipeline.__aenter__ = AsyncMock(return_value=pipeline)
        pipeline.__aexit__ = AsyncMock(return_value=None)
        mock_pipeline_cls.return_value = pipeline

        app = _make_app(qdrant=_mock_qdrant())
        with TestClient(app) as client:
            resp = client.post("/ingest", json={
                "user_id": 123,
                "urls": ["https://example.com"],
                "product_query": "test product",
            })

        assert resp.status_code == 200
        data = resp.json()
        assert data["success_count"] == 1
        assert data["failed_count"] == 0
        assert data["chunks_count"] == 5
        assert len(data["results"]) == 1
        assert data["results"][0]["status"] == "success"
        assert data["results"][0]["url"] == "https://example.com"

    @patch("reviewmind.api.endpoints.ingest.IngestionPipeline")
    def test_multiple_urls_success(self, mock_pipeline_cls):
        pipeline = AsyncMock()
        pipeline.ingest_urls = AsyncMock(return_value=_ingestion_result([
            _success_result("https://youtube.com/watch?v=abc", chunks=3),
            _success_result("https://reddit.com/r/tech/123", chunks=7),
            _success_result("https://rtings.com/review", chunks=10),
        ]))
        pipeline.__aenter__ = AsyncMock(return_value=pipeline)
        pipeline.__aexit__ = AsyncMock(return_value=None)
        mock_pipeline_cls.return_value = pipeline

        app = _make_app(qdrant=_mock_qdrant())
        with TestClient(app) as client:
            resp = client.post("/ingest", json={
                "user_id": 456,
                "urls": [
                    "https://youtube.com/watch?v=abc",
                    "https://reddit.com/r/tech/123",
                    "https://rtings.com/review",
                ],
            })

        data = resp.json()
        assert data["success_count"] == 3
        assert data["chunks_count"] == 20

    @patch("reviewmind.api.endpoints.ingest.IngestionPipeline")
    def test_source_type_propagated(self, mock_pipeline_cls):
        result = SourceIngestionResult(
            url="https://youtube.com/watch?v=x",
            success=True,
            source_type="youtube",
            chunks_count=4,
        )
        pipeline = AsyncMock()
        pipeline.ingest_urls = AsyncMock(return_value=_ingestion_result([result]))
        pipeline.__aenter__ = AsyncMock(return_value=pipeline)
        pipeline.__aexit__ = AsyncMock(return_value=None)
        mock_pipeline_cls.return_value = pipeline

        app = _make_app(qdrant=_mock_qdrant())
        with TestClient(app) as client:
            resp = client.post("/ingest", json={"user_id": 1, "urls": ["https://youtube.com/watch?v=x"]})

        assert resp.json()["results"][0]["source_type"] == "youtube"


class TestIngestEndpointPartialFailure:
    """Test that one URL failure does not block others."""

    @patch("reviewmind.api.endpoints.ingest.IngestionPipeline")
    def test_one_success_one_failure(self, mock_pipeline_cls):
        pipeline = AsyncMock()
        pipeline.ingest_urls = AsyncMock(return_value=_ingestion_result([
            _success_result("https://good.com", chunks=5),
            _failed_result("https://bad.com", error="не удалось загрузить"),
        ]))
        pipeline.__aenter__ = AsyncMock(return_value=pipeline)
        pipeline.__aexit__ = AsyncMock(return_value=None)
        mock_pipeline_cls.return_value = pipeline

        app = _make_app(qdrant=_mock_qdrant())
        with TestClient(app) as client:
            resp = client.post("/ingest", json={
                "user_id": 1,
                "urls": ["https://good.com", "https://bad.com"],
            })

        data = resp.json()
        assert data["success_count"] == 1
        assert data["failed_count"] == 1
        assert data["chunks_count"] == 5

        good = next(r for r in data["results"] if r["url"] == "https://good.com")
        bad = next(r for r in data["results"] if r["url"] == "https://bad.com")
        assert good["status"] == "success"
        assert bad["status"] == "failed"
        assert bad["error"] == "не удалось загрузить"

    @patch("reviewmind.api.endpoints.ingest.IngestionPipeline")
    def test_all_failed(self, mock_pipeline_cls):
        pipeline = AsyncMock()
        pipeline.ingest_urls = AsyncMock(return_value=_ingestion_result([
            _failed_result("https://a.com"),
            _failed_result("https://b.com"),
        ]))
        pipeline.__aenter__ = AsyncMock(return_value=pipeline)
        pipeline.__aexit__ = AsyncMock(return_value=None)
        mock_pipeline_cls.return_value = pipeline

        app = _make_app(qdrant=_mock_qdrant())
        with TestClient(app) as client:
            resp = client.post("/ingest", json={"user_id": 1, "urls": ["https://a.com", "https://b.com"]})

        data = resp.json()
        assert data["success_count"] == 0
        assert data["failed_count"] == 2
        assert all(r["status"] == "failed" for r in data["results"])


class TestIngestEndpointQdrantUnavailable:
    """Test behavior when Qdrant is not available."""

    def test_no_qdrant_returns_all_failed(self):
        app = _make_app()  # no qdrant in state
        with TestClient(app) as client:
            resp = client.post("/ingest", json={"user_id": 1, "urls": ["https://a.com", "https://b.com"]})

        assert resp.status_code == 200
        data = resp.json()
        assert data["failed_count"] == 2
        assert data["success_count"] == 0
        assert all(r["status"] == "failed" for r in data["results"])
        assert all("unavailable" in r["error"].lower() for r in data["results"])


class TestIngestEndpointNoDB:
    """Test ingestion works without a database engine (no source metadata persistence)."""

    @patch("reviewmind.api.endpoints.ingest.IngestionPipeline")
    def test_no_db_engine_still_works(self, mock_pipeline_cls):
        pipeline = AsyncMock()
        pipeline.ingest_urls = AsyncMock(return_value=_ingestion_result([
            _success_result("https://example.com", chunks=3),
        ]))
        pipeline.__aenter__ = AsyncMock(return_value=pipeline)
        pipeline.__aexit__ = AsyncMock(return_value=None)
        mock_pipeline_cls.return_value = pipeline

        # qdrant present, but NO db_engine
        app = _make_app(qdrant=_mock_qdrant())
        with TestClient(app) as client:
            resp = client.post("/ingest", json={"user_id": 1, "urls": ["https://example.com"]})

        data = resp.json()
        assert data["success_count"] == 1
        assert data["chunks_count"] == 3

        # pipeline should have been called with db_session=None
        mock_pipeline_cls.assert_called_once()
        call_kwargs = mock_pipeline_cls.call_args
        assert call_kwargs.kwargs.get("db_session") is None or call_kwargs[1].get("db_session") is None


class TestIngestEndpointWithDB:
    """Test ingestion with a database engine present."""

    @patch("reviewmind.api.endpoints.ingest.IngestionPipeline")
    def test_db_session_provided(self, mock_pipeline_cls):
        pipeline = AsyncMock()
        pipeline.ingest_urls = AsyncMock(return_value=_ingestion_result([
            _success_result("https://example.com", chunks=2),
        ]))
        pipeline.__aenter__ = AsyncMock(return_value=pipeline)
        pipeline.__aexit__ = AsyncMock(return_value=None)
        mock_pipeline_cls.return_value = pipeline

        mock_engine = MagicMock(name="db_engine")
        mock_session = AsyncMock()
        mock_session.commit = AsyncMock()
        mock_session_factory = MagicMock()
        mock_session_factory.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_factory.return_value.__aexit__ = AsyncMock(return_value=None)

        app = _make_app(qdrant=_mock_qdrant(), db_engine=mock_engine)
        with patch("reviewmind.api.endpoints.ingest.async_sessionmaker", return_value=mock_session_factory):
            with TestClient(app) as client:
                resp = client.post("/ingest", json={"user_id": 1, "urls": ["https://example.com"]})

        assert resp.status_code == 200
        data = resp.json()
        assert data["success_count"] == 1


class TestIngestEndpointPipelineError:
    """Test endpoint catches unexpected pipeline errors."""

    @patch("reviewmind.api.endpoints.ingest.IngestionPipeline")
    def test_unexpected_exception_returns_graceful_response(self, mock_pipeline_cls):
        mock_pipeline_cls.side_effect = RuntimeError("Something exploded")

        app = _make_app(qdrant=_mock_qdrant())
        with TestClient(app) as client:
            resp = client.post("/ingest", json={"user_id": 1, "urls": ["https://a.com"]})

        assert resp.status_code == 200
        data = resp.json()
        assert data["failed_count"] == 1
        assert data["results"][0]["status"] == "failed"
        assert "Internal error" in data["results"][0]["error"]


class TestIngestEndpointValidation:
    """Test request validation."""

    def test_missing_urls(self):
        app = _make_app(qdrant=_mock_qdrant())
        with TestClient(app) as client:
            resp = client.post("/ingest", json={"user_id": 1})
        assert resp.status_code == 422

    def test_empty_urls(self):
        app = _make_app(qdrant=_mock_qdrant())
        with TestClient(app) as client:
            resp = client.post("/ingest", json={"user_id": 1, "urls": []})
        assert resp.status_code == 422

    def test_missing_user_id(self):
        app = _make_app(qdrant=_mock_qdrant())
        with TestClient(app) as client:
            resp = client.post("/ingest", json={"urls": ["https://example.com"]})
        assert resp.status_code == 422

    def test_invalid_json(self):
        app = _make_app(qdrant=_mock_qdrant())
        with TestClient(app) as client:
            resp = client.post("/ingest", content="not json", headers={"content-type": "application/json"})
        assert resp.status_code == 422


class TestIngestEndpointProductQuery:
    """Test product_query handling."""

    @patch("reviewmind.api.endpoints.ingest.IngestionPipeline")
    def test_product_query_passed_to_pipeline(self, mock_pipeline_cls):
        pipeline = AsyncMock()
        pipeline.ingest_urls = AsyncMock(return_value=_ingestion_result([
            _success_result("https://example.com"),
        ]))
        pipeline.__aenter__ = AsyncMock(return_value=pipeline)
        pipeline.__aexit__ = AsyncMock(return_value=None)
        mock_pipeline_cls.return_value = pipeline

        app = _make_app(qdrant=_mock_qdrant())
        with TestClient(app) as client:
            client.post("/ingest", json={
                "user_id": 1,
                "urls": ["https://example.com"],
                "product_query": "Sony WH-1000XM5",
            })

        pipeline.ingest_urls.assert_called_once()
        call_kwargs = pipeline.ingest_urls.call_args
        assert call_kwargs.kwargs["product_query"] == "Sony WH-1000XM5"

    @patch("reviewmind.api.endpoints.ingest.IngestionPipeline")
    def test_default_product_query_empty(self, mock_pipeline_cls):
        pipeline = AsyncMock()
        pipeline.ingest_urls = AsyncMock(return_value=_ingestion_result([
            _success_result("https://example.com"),
        ]))
        pipeline.__aenter__ = AsyncMock(return_value=pipeline)
        pipeline.__aexit__ = AsyncMock(return_value=None)
        mock_pipeline_cls.return_value = pipeline

        app = _make_app(qdrant=_mock_qdrant())
        with TestClient(app) as client:
            client.post("/ingest", json={
                "user_id": 1,
                "urls": ["https://example.com"],
            })

        call_kwargs = pipeline.ingest_urls.call_args
        assert call_kwargs.kwargs["product_query"] == ""


class TestIngestEndpointSessionId:
    """Test session_id propagation."""

    @patch("reviewmind.api.endpoints.ingest.IngestionPipeline")
    def test_session_id_passed_through(self, mock_pipeline_cls):
        pipeline = AsyncMock()
        pipeline.ingest_urls = AsyncMock(return_value=_ingestion_result([
            _success_result("https://example.com"),
        ]))
        pipeline.__aenter__ = AsyncMock(return_value=pipeline)
        pipeline.__aexit__ = AsyncMock(return_value=None)
        mock_pipeline_cls.return_value = pipeline

        app = _make_app(qdrant=_mock_qdrant())
        with TestClient(app) as client:
            client.post("/ingest", json={
                "user_id": 1,
                "session_id": "sess-xyz",
                "urls": ["https://example.com"],
            })

        call_kwargs = pipeline.ingest_urls.call_args
        assert call_kwargs.kwargs["session_id"] == "sess-xyz"


# ══════════════════════════════════════════════════════════════
# Tests — Router wiring
# ══════════════════════════════════════════════════════════════


class TestIngestRouterWiring:
    """Test that ingest router is properly wired to the main api_router."""

    def test_ingest_router_in_api_router(self):
        from reviewmind.api.router import api_router

        routes = [r.path for r in api_router.routes]
        assert "/ingest" in routes

    def test_ingest_endpoint_post_method(self):
        from reviewmind.api.router import api_router

        for route in api_router.routes:
            if getattr(route, "path", None) == "/ingest":
                assert "POST" in route.methods
                break
        else:
            pytest.fail("/ingest route not found")


# ══════════════════════════════════════════════════════════════
# Tests — Full app integration
# ══════════════════════════════════════════════════════════════


class TestIngestFullApp:
    """Test /ingest is reachable via the full create_app factory."""

    @patch("reviewmind.api.endpoints.ingest.IngestionPipeline")
    def test_ingest_reachable(self, mock_pipeline_cls):
        pipeline = AsyncMock()
        pipeline.ingest_urls = AsyncMock(return_value=_ingestion_result([
            _success_result("https://example.com", chunks=1),
        ]))
        pipeline.__aenter__ = AsyncMock(return_value=pipeline)
        pipeline.__aexit__ = AsyncMock(return_value=None)
        mock_pipeline_cls.return_value = pipeline

        from reviewmind.main import create_app

        app = create_app()
        app.state.qdrant = _mock_qdrant()
        with TestClient(app, raise_server_exceptions=False) as client:
            resp = client.post("/ingest", json={"user_id": 1, "urls": ["https://example.com"]})

        assert resp.status_code == 200
        assert resp.json()["success_count"] == 1


# ══════════════════════════════════════════════════════════════
# Tests — Schema exports from reviewmind.api.schemas
# ══════════════════════════════════════════════════════════════


class TestSchemaExports:
    """Test that new schema classes are importable."""

    def test_import_ingest_request(self):
        from reviewmind.api.schemas import IngestRequest
        assert IngestRequest is not None

    def test_import_ingest_response(self):
        from reviewmind.api.schemas import IngestResponse
        assert IngestResponse is not None

    def test_import_ingest_url_result(self):
        from reviewmind.api.schemas import IngestURLResult
        assert IngestURLResult is not None
