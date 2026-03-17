"""Unit tests for TASK-032 — GET /status/{job_id} endpoint.

Tests cover:
- JobStatusResponse schema validation
- GET /status/{job_id}: success, job not found, invalid UUID, DB unavailable
- Celery result fetching
- Router wiring: status_router registered in api_router
- Full app integration: /status available via create_app()
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from reviewmind.api.endpoints.status import _get_celery_result, router
from reviewmind.api.schemas import JobStatusResponse

# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════


def _make_app(**state_attrs) -> FastAPI:
    """Create a minimal FastAPI app with the status router and optional state."""
    app = FastAPI()
    app.include_router(router)
    for k, v in state_attrs.items():
        setattr(app.state, k, v)
    return app


def _mock_engine() -> MagicMock:
    return MagicMock(name="db_engine")


def _make_job(
    *,
    job_id: uuid.UUID | None = None,
    user_id: int = 123,
    job_type: str = "manual_links",
    status: str = "pending",
    product_query: str = "Sony WH-1000XM5",
    celery_task_id: str | None = None,
    created_at: datetime | None = None,
    completed_at: datetime | None = None,
) -> MagicMock:
    job = MagicMock()
    job.id = job_id or uuid.uuid4()
    job.user_id = user_id
    job.job_type = job_type
    job.status = status
    job.product_query = product_query
    job.celery_task_id = celery_task_id
    job.created_at = created_at or datetime.now(timezone.utc)
    job.completed_at = completed_at
    return job


# ══════════════════════════════════════════════════════════════
# Tests — Schema model
# ══════════════════════════════════════════════════════════════


class TestJobStatusResponse:
    """Test JobStatusResponse pydantic model."""

    def test_minimal_response(self):
        resp = JobStatusResponse(job_id="abc-123", status="pending")
        assert resp.job_id == "abc-123"
        assert resp.status == "pending"
        assert resp.progress is None
        assert resp.completed_at is None

    def test_full_response(self):
        resp = JobStatusResponse(
            job_id="abc-123",
            status="done",
            job_type="manual_links",
            product_query="headphones",
            celery_task_id="celery-task-1",
            created_at="2026-03-17T12:00:00+00:00",
            completed_at="2026-03-17T12:05:00+00:00",
            progress={"success_count": 3, "chunks_count": 25},
        )
        assert resp.job_type == "manual_links"
        assert resp.product_query == "headphones"
        assert resp.celery_task_id == "celery-task-1"
        assert resp.completed_at == "2026-03-17T12:05:00+00:00"
        assert resp.progress["success_count"] == 3

    def test_defaults(self):
        resp = JobStatusResponse(job_id="x", status="running")
        assert resp.job_type == ""
        assert resp.product_query is None
        assert resp.celery_task_id is None


# ══════════════════════════════════════════════════════════════
# Tests — GET /status/{job_id} endpoint
# ══════════════════════════════════════════════════════════════


class TestGetJobStatusSuccess:
    """Test successful job status retrieval."""

    def test_pending_job(self):
        job_id = uuid.uuid4()
        job = _make_job(job_id=job_id, status="pending")

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("reviewmind.api.endpoints.status.async_sessionmaker") as mock_factory:
            mock_factory.return_value = MagicMock(return_value=mock_session)
            with patch("reviewmind.api.endpoints.status.JobRepository") as MockRepo:
                MockRepo.return_value.get_by_id = AsyncMock(return_value=job)

                app = _make_app(db_engine=_mock_engine())
                client = TestClient(app)
                resp = client.get(f"/status/{job_id}")

        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == str(job_id)
        assert data["status"] == "pending"

    def test_done_job_with_celery_result(self):
        job_id = uuid.uuid4()
        job = _make_job(
            job_id=job_id,
            status="done",
            celery_task_id="celery-abc",
            completed_at=datetime.now(timezone.utc),
        )

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        progress = {"success_count": 2, "chunks_count": 10}

        with patch("reviewmind.api.endpoints.status.async_sessionmaker") as mock_factory:
            mock_factory.return_value = MagicMock(return_value=mock_session)
            with patch("reviewmind.api.endpoints.status.JobRepository") as MockRepo:
                MockRepo.return_value.get_by_id = AsyncMock(return_value=job)
                with patch("reviewmind.api.endpoints.status._get_celery_result", return_value=progress):
                    app = _make_app(db_engine=_mock_engine())
                    client = TestClient(app)
                    resp = client.get(f"/status/{job_id}")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "done"
        assert data["progress"]["success_count"] == 2

    def test_running_job_no_celery_result(self):
        job_id = uuid.uuid4()
        job = _make_job(job_id=job_id, status="running", celery_task_id="celery-xyz")

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("reviewmind.api.endpoints.status.async_sessionmaker") as mock_factory:
            mock_factory.return_value = MagicMock(return_value=mock_session)
            with patch("reviewmind.api.endpoints.status.JobRepository") as MockRepo:
                MockRepo.return_value.get_by_id = AsyncMock(return_value=job)

                app = _make_app(db_engine=_mock_engine())
                client = TestClient(app)
                resp = client.get(f"/status/{job_id}")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "running"
        assert data["progress"] is None

    def test_failed_job(self):
        job_id = uuid.uuid4()
        job = _make_job(
            job_id=job_id,
            status="failed",
            celery_task_id="celery-fail",
            completed_at=datetime.now(timezone.utc),
        )

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("reviewmind.api.endpoints.status.async_sessionmaker") as mock_factory:
            mock_factory.return_value = MagicMock(return_value=mock_session)
            with patch("reviewmind.api.endpoints.status.JobRepository") as MockRepo:
                MockRepo.return_value.get_by_id = AsyncMock(return_value=job)
                with patch("reviewmind.api.endpoints.status._get_celery_result", return_value={"error": "timeout"}):
                    app = _make_app(db_engine=_mock_engine())
                    client = TestClient(app)
                    resp = client.get(f"/status/{job_id}")

        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "failed"
        assert data["completed_at"] is not None


class TestGetJobStatusErrors:
    """Test error scenarios for GET /status/{job_id}."""

    def test_invalid_uuid_returns_400(self):
        app = _make_app(db_engine=_mock_engine())
        client = TestClient(app)
        resp = client.get("/status/not-a-uuid")
        assert resp.status_code == 400
        assert "UUID" in resp.json()["detail"]

    def test_job_not_found_returns_404(self):
        job_id = uuid.uuid4()

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with patch("reviewmind.api.endpoints.status.async_sessionmaker") as mock_factory:
            mock_factory.return_value = MagicMock(return_value=mock_session)
            with patch("reviewmind.api.endpoints.status.JobRepository") as MockRepo:
                MockRepo.return_value.get_by_id = AsyncMock(return_value=None)

                app = _make_app(db_engine=_mock_engine())
                client = TestClient(app)
                resp = client.get(f"/status/{job_id}")

        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"]

    def test_db_unavailable_returns_503(self):
        job_id = uuid.uuid4()
        app = _make_app()  # no db_engine
        client = TestClient(app)
        resp = client.get(f"/status/{job_id}")
        assert resp.status_code == 503
        assert "Database unavailable" in resp.json()["detail"]


# ══════════════════════════════════════════════════════════════
# Tests — _get_celery_result helper
# ══════════════════════════════════════════════════════════════


class TestGetCeleryResult:
    """Tests for the _get_celery_result helper."""

    def test_returns_dict_result_when_ready(self):
        mock_result = MagicMock()
        mock_result.ready.return_value = True
        mock_result.result = {"success_count": 5}

        with patch("reviewmind.workers.celery_app.celery_app") as mock_app:
            mock_app.AsyncResult.return_value = mock_result
            result = _get_celery_result("task-123")

        assert result == {"success_count": 5}

    def test_returns_none_when_not_ready(self):
        mock_result = MagicMock()
        mock_result.ready.return_value = False

        with patch("reviewmind.workers.celery_app.celery_app") as mock_app:
            mock_app.AsyncResult.return_value = mock_result
            result = _get_celery_result("task-123")

        assert result is None

    def test_returns_none_when_result_not_dict(self):
        mock_result = MagicMock()
        mock_result.ready.return_value = True
        mock_result.result = "just a string"

        with patch("reviewmind.workers.celery_app.celery_app") as mock_app:
            mock_app.AsyncResult.return_value = mock_result
            result = _get_celery_result("task-123")

        assert result is None

    def test_returns_none_on_exception(self):
        with patch("reviewmind.workers.celery_app.celery_app") as mock_app:
            mock_app.AsyncResult.side_effect = Exception("backend down")
            result = _get_celery_result("task-123")

        assert result is None


# ══════════════════════════════════════════════════════════════
# Tests — Router wiring
# ══════════════════════════════════════════════════════════════


class TestStatusRouterWiring:
    """Verify the status router is part of the api_router."""

    def test_status_route_in_api_router(self):
        from reviewmind.api.router import api_router

        paths = [r.path for r in api_router.routes]
        assert "/status/{job_id}" in paths

    def test_status_route_has_tag(self):
        from reviewmind.api.router import api_router

        for r in api_router.routes:
            if getattr(r, "path", "") == "/status/{job_id}":
                assert "status" in r.tags
                break
        else:
            pytest.fail("/status/{job_id} not found in api_router routes")


class TestStatusFullApp:
    """Verify the endpoint is available through the full app."""

    def test_status_route_registered_in_app(self):
        from reviewmind.main import create_app

        app = create_app()
        paths = [r.path for r in app.routes]
        assert "/status/{job_id}" in paths


# ══════════════════════════════════════════════════════════════
# Tests — Schema exports
# ══════════════════════════════════════════════════════════════


class TestSchemaExports:
    """Verify that new schemas are importable."""

    def test_job_status_response_importable(self):
        from reviewmind.api.schemas import JobStatusResponse

        assert JobStatusResponse is not None

    def test_job_status_response_has_required_fields(self):
        fields = set(JobStatusResponse.model_fields.keys())
        assert "job_id" in fields
        assert "status" in fields
        assert "progress" in fields
        assert "completed_at" in fields
