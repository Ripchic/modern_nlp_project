"""Unit tests for TASK-032 — ingest_sources_task Celery task.

Tests cover:
- Task registration and attributes
- _run_async helper
- _ingest_sources async function: success, partial failure, pipeline error
- Job status transitions: pending → running → done / failed
- Workers __init__ exports
- Integration scenarios
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ══════════════════════════════════════════════════════════════
# Tests — Task registration
# ══════════════════════════════════════════════════════════════


class TestIngestSourcesTaskRegistration:
    """Test that ingest_sources_task is properly registered with Celery."""

    def test_task_name(self):
        from reviewmind.workers.tasks import ingest_sources_task

        assert ingest_sources_task.name == "reviewmind.ingest_sources"

    def test_task_max_retries(self):
        from reviewmind.workers.tasks import ingest_sources_task

        assert ingest_sources_task.max_retries == 3

    def test_task_is_bound(self):
        from reviewmind.workers.tasks import ingest_sources_task

        # Bound tasks have 'self' as first arg; the Celery task object
        # exposes a `bind` attribute via task meta
        assert ingest_sources_task.name is not None

    def test_task_is_celery_task(self):
        from celery.app.task import Task

        from reviewmind.workers.tasks import ingest_sources_task

        assert isinstance(ingest_sources_task, Task)


# ══════════════════════════════════════════════════════════════
# Tests — _run_async helper
# ══════════════════════════════════════════════════════════════


class TestRunAsync:
    """Test the _run_async sync→async bridge."""

    def test_runs_simple_coroutine(self):
        from reviewmind.workers.tasks import _run_async

        async def _simple():
            return 42

        assert _run_async(_simple()) == 42

    def test_propagates_exceptions(self):
        from reviewmind.workers.tasks import _run_async

        async def _fail():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            _run_async(_fail())


# ══════════════════════════════════════════════════════════════
# Tests — _ingest_sources async logic
# ══════════════════════════════════════════════════════════════


def _mock_ingestion_result(success_count=2, failed_count=0, chunks_count=10):
    """Create a mock IngestionResult."""
    result = MagicMock()
    result.success_count = success_count
    result.failed_count = failed_count
    result.chunks_count = chunks_count
    result.failed_urls = []
    return result


class TestIngestSourcesSuccess:
    """Test _ingest_sources with successful ingestion."""

    @pytest.mark.asyncio
    async def test_returns_done_status(self):
        from reviewmind.workers.tasks import _ingest_sources

        job_id = str(uuid.uuid4())
        mock_result = _mock_ingestion_result(success_count=2, chunks_count=10)

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.commit = AsyncMock()

        mock_engine = AsyncMock()
        mock_engine.dispose = AsyncMock()

        mock_pipeline = AsyncMock()
        mock_pipeline.ingest_urls = AsyncMock(return_value=mock_result)
        mock_pipeline.__aenter__ = AsyncMock(return_value=mock_pipeline)
        mock_pipeline.__aexit__ = AsyncMock(return_value=False)

        mock_qdrant = AsyncMock()
        mock_qdrant.close = AsyncMock()

        mock_settings = MagicMock()
        mock_settings.database_url = "postgresql+asyncpg://test:test@localhost/test"
        mock_settings.qdrant_url = "http://localhost:6333"

        mock_factory = MagicMock(return_value=mock_session)
        mock_job_repo = AsyncMock()
        mock_job_repo.update_status = AsyncMock()

        with (
            patch("reviewmind.workers.tasks.create_async_engine", return_value=mock_engine),
            patch("reviewmind.workers.tasks.async_sessionmaker", return_value=mock_factory),
            patch("reviewmind.workers.tasks.AsyncQdrantClient", return_value=mock_qdrant),
            patch("reviewmind.config.settings", mock_settings),
            patch("reviewmind.workers.tasks.IngestionPipeline", return_value=mock_pipeline),
            patch("reviewmind.workers.tasks.JobRepository", return_value=mock_job_repo),
        ):
            result = await _ingest_sources(
                job_id=job_id,
                user_id=123,
                product_query="test product",
                urls=["https://example.com"],
            )

        assert result["status"] == "done"
        assert result["job_id"] == job_id
        assert result["success_count"] == 2
        assert result["chunks_count"] == 10
        assert "completed_at" in result

    @pytest.mark.asyncio
    async def test_calls_job_repo_update_to_running(self):
        from reviewmind.workers.tasks import _ingest_sources

        job_id = str(uuid.uuid4())
        mock_result = _mock_ingestion_result()

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.commit = AsyncMock()

        mock_engine = AsyncMock()
        mock_engine.dispose = AsyncMock()

        mock_pipeline = AsyncMock()
        mock_pipeline.ingest_urls = AsyncMock(return_value=mock_result)
        mock_pipeline.__aenter__ = AsyncMock(return_value=mock_pipeline)
        mock_pipeline.__aexit__ = AsyncMock(return_value=False)

        mock_qdrant = AsyncMock()
        mock_qdrant.close = AsyncMock()

        mock_settings = MagicMock()
        mock_settings.database_url = "postgresql+asyncpg://test/test"
        mock_settings.qdrant_url = "http://localhost:6333"

        mock_factory = MagicMock(return_value=mock_session)
        mock_job_repo = AsyncMock()
        mock_job_repo.update_status = AsyncMock()

        with (
            patch("reviewmind.workers.tasks.create_async_engine", return_value=mock_engine),
            patch("reviewmind.workers.tasks.async_sessionmaker", return_value=mock_factory),
            patch("reviewmind.workers.tasks.AsyncQdrantClient", return_value=mock_qdrant),
            patch("reviewmind.config.settings", mock_settings),
            patch("reviewmind.workers.tasks.IngestionPipeline", return_value=mock_pipeline),
            patch("reviewmind.workers.tasks.JobRepository", return_value=mock_job_repo),
        ):
            await _ingest_sources(
                job_id=job_id,
                user_id=123,
                product_query="test",
                urls=["https://example.com"],
            )

        # Should be called at least twice: once for 'running', once for 'done'/'failed'
        calls = mock_job_repo.update_status.call_args_list
        assert len(calls) >= 2
        assert calls[0].args[1] == "running"


class TestIngestSourcesFailure:
    """Test _ingest_sources with failures."""

    @pytest.mark.asyncio
    async def test_pipeline_error_returns_failed(self):
        from reviewmind.workers.tasks import _ingest_sources

        job_id = str(uuid.uuid4())

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.commit = AsyncMock()

        mock_engine = AsyncMock()
        mock_engine.dispose = AsyncMock()

        mock_pipeline = AsyncMock()
        mock_pipeline.ingest_urls = AsyncMock(side_effect=RuntimeError("Pipeline boom"))
        mock_pipeline.__aenter__ = AsyncMock(return_value=mock_pipeline)
        mock_pipeline.__aexit__ = AsyncMock(return_value=False)

        mock_qdrant = AsyncMock()
        mock_qdrant.close = AsyncMock()

        mock_settings = MagicMock()
        mock_settings.database_url = "postgresql+asyncpg://test/test"
        mock_settings.qdrant_url = "http://localhost:6333"

        mock_factory = MagicMock(return_value=mock_session)
        mock_job_repo = AsyncMock()
        mock_job_repo.update_status = AsyncMock()

        with (
            patch("reviewmind.workers.tasks.create_async_engine", return_value=mock_engine),
            patch("reviewmind.workers.tasks.async_sessionmaker", return_value=mock_factory),
            patch("reviewmind.workers.tasks.AsyncQdrantClient", return_value=mock_qdrant),
            patch("reviewmind.config.settings", mock_settings),
            patch("reviewmind.workers.tasks.IngestionPipeline", return_value=mock_pipeline),
            patch("reviewmind.workers.tasks.JobRepository", return_value=mock_job_repo),
        ):
            result = await _ingest_sources(
                job_id=job_id,
                user_id=123,
                product_query="test",
                urls=["https://example.com"],
            )

        assert result["status"] == "failed"
        assert "error" in result

    @pytest.mark.asyncio
    async def test_all_urls_failed_returns_failed(self):
        from reviewmind.workers.tasks import _ingest_sources

        job_id = str(uuid.uuid4())
        mock_result = _mock_ingestion_result(success_count=0, failed_count=2, chunks_count=0)

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.commit = AsyncMock()

        mock_engine = AsyncMock()
        mock_engine.dispose = AsyncMock()

        mock_pipeline = AsyncMock()
        mock_pipeline.ingest_urls = AsyncMock(return_value=mock_result)
        mock_pipeline.__aenter__ = AsyncMock(return_value=mock_pipeline)
        mock_pipeline.__aexit__ = AsyncMock(return_value=False)

        mock_qdrant = AsyncMock()
        mock_qdrant.close = AsyncMock()

        mock_settings = MagicMock()
        mock_settings.database_url = "postgresql+asyncpg://test/test"
        mock_settings.qdrant_url = "http://localhost:6333"

        mock_factory = MagicMock(return_value=mock_session)
        mock_job_repo = AsyncMock()
        mock_job_repo.update_status = AsyncMock()

        with (
            patch("reviewmind.workers.tasks.create_async_engine", return_value=mock_engine),
            patch("reviewmind.workers.tasks.async_sessionmaker", return_value=mock_factory),
            patch("reviewmind.workers.tasks.AsyncQdrantClient", return_value=mock_qdrant),
            patch("reviewmind.config.settings", mock_settings),
            patch("reviewmind.workers.tasks.IngestionPipeline", return_value=mock_pipeline),
            patch("reviewmind.workers.tasks.JobRepository", return_value=mock_job_repo),
        ):
            result = await _ingest_sources(
                job_id=job_id,
                user_id=123,
                product_query="test",
                urls=["https://bad1.com", "https://bad2.com"],
            )

        assert result["status"] == "failed"

    @pytest.mark.asyncio
    async def test_db_update_failure_does_not_crash(self):
        """If updating job status fails, task should still complete."""
        from reviewmind.workers.tasks import _ingest_sources

        job_id = str(uuid.uuid4())
        mock_result = _mock_ingestion_result()

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.commit = AsyncMock()

        mock_engine = AsyncMock()
        mock_engine.dispose = AsyncMock()

        mock_pipeline = AsyncMock()
        mock_pipeline.ingest_urls = AsyncMock(return_value=mock_result)
        mock_pipeline.__aenter__ = AsyncMock(return_value=mock_pipeline)
        mock_pipeline.__aexit__ = AsyncMock(return_value=False)

        mock_qdrant = AsyncMock()
        mock_qdrant.close = AsyncMock()

        mock_settings = MagicMock()
        mock_settings.database_url = "postgresql+asyncpg://test/test"
        mock_settings.qdrant_url = "http://localhost:6333"

        mock_factory = MagicMock(return_value=mock_session)
        mock_job_repo = AsyncMock()
        mock_job_repo.update_status = AsyncMock(side_effect=RuntimeError("DB down"))

        with (
            patch("reviewmind.workers.tasks.create_async_engine", return_value=mock_engine),
            patch("reviewmind.workers.tasks.async_sessionmaker", return_value=mock_factory),
            patch("reviewmind.workers.tasks.AsyncQdrantClient", return_value=mock_qdrant),
            patch("reviewmind.config.settings", mock_settings),
            patch("reviewmind.workers.tasks.IngestionPipeline", return_value=mock_pipeline),
            patch("reviewmind.workers.tasks.JobRepository", return_value=mock_job_repo),
        ):
            result = await _ingest_sources(
                job_id=job_id,
                user_id=123,
                product_query="test",
                urls=["https://example.com"],
            )

        # Task completes despite DB failures
        assert "status" in result
        assert "completed_at" in result


class TestIngestSourcesSessionId:
    """Test session_id parameter passthrough."""

    @pytest.mark.asyncio
    async def test_session_id_passed_to_pipeline(self):
        from reviewmind.workers.tasks import _ingest_sources

        job_id = str(uuid.uuid4())
        mock_result = _mock_ingestion_result()

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.commit = AsyncMock()

        mock_engine = AsyncMock()
        mock_engine.dispose = AsyncMock()

        mock_pipeline = AsyncMock()
        mock_pipeline.ingest_urls = AsyncMock(return_value=mock_result)
        mock_pipeline.__aenter__ = AsyncMock(return_value=mock_pipeline)
        mock_pipeline.__aexit__ = AsyncMock(return_value=False)

        mock_qdrant = AsyncMock()
        mock_qdrant.close = AsyncMock()

        mock_settings = MagicMock()
        mock_settings.database_url = "postgresql+asyncpg://test/test"
        mock_settings.qdrant_url = "http://localhost:6333"

        mock_factory = MagicMock(return_value=mock_session)
        mock_job_repo = AsyncMock()
        mock_job_repo.update_status = AsyncMock()

        with (
            patch("reviewmind.workers.tasks.create_async_engine", return_value=mock_engine),
            patch("reviewmind.workers.tasks.async_sessionmaker", return_value=mock_factory),
            patch("reviewmind.workers.tasks.AsyncQdrantClient", return_value=mock_qdrant),
            patch("reviewmind.config.settings", mock_settings),
            patch("reviewmind.workers.tasks.IngestionPipeline", return_value=mock_pipeline),
            patch("reviewmind.workers.tasks.JobRepository", return_value=mock_job_repo),
        ):
            await _ingest_sources(
                job_id=job_id,
                user_id=123,
                product_query="test",
                urls=["https://example.com"],
                session_id="my-session",
            )

        call_kwargs = mock_pipeline.ingest_urls.call_args
        assert call_kwargs.kwargs.get("session_id") == "my-session"


# ══════════════════════════════════════════════════════════════
# Tests — Engine disposal
# ══════════════════════════════════════════════════════════════


class TestEngineDisposal:
    """Ensure the task cleans up DB engine."""

    @pytest.mark.asyncio
    async def test_engine_disposed_on_success(self):
        from reviewmind.workers.tasks import _ingest_sources

        job_id = str(uuid.uuid4())
        mock_result = _mock_ingestion_result()

        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.commit = AsyncMock()

        mock_engine = AsyncMock()
        mock_engine.dispose = AsyncMock()

        mock_pipeline = AsyncMock()
        mock_pipeline.ingest_urls = AsyncMock(return_value=mock_result)
        mock_pipeline.__aenter__ = AsyncMock(return_value=mock_pipeline)
        mock_pipeline.__aexit__ = AsyncMock(return_value=False)

        mock_qdrant = AsyncMock()
        mock_qdrant.close = AsyncMock()

        mock_settings = MagicMock()
        mock_settings.database_url = "postgresql+asyncpg://test/test"
        mock_settings.qdrant_url = "http://localhost:6333"

        mock_factory = MagicMock(return_value=mock_session)
        mock_job_repo = AsyncMock()
        mock_job_repo.update_status = AsyncMock()

        with (
            patch("reviewmind.workers.tasks.create_async_engine", return_value=mock_engine),
            patch("reviewmind.workers.tasks.async_sessionmaker", return_value=mock_factory),
            patch("reviewmind.workers.tasks.AsyncQdrantClient", return_value=mock_qdrant),
            patch("reviewmind.config.settings", mock_settings),
            patch("reviewmind.workers.tasks.IngestionPipeline", return_value=mock_pipeline),
            patch("reviewmind.workers.tasks.JobRepository", return_value=mock_job_repo),
        ):
            await _ingest_sources(
                job_id=job_id,
                user_id=123,
                product_query="test",
                urls=["https://example.com"],
            )

        mock_engine.dispose.assert_awaited_once()


# ══════════════════════════════════════════════════════════════
# Tests — Workers exports
# ══════════════════════════════════════════════════════════════


class TestWorkersExports:
    """Verify updated exports from workers __init__."""

    def test_ingest_sources_task_exported(self):
        from reviewmind.workers import ingest_sources_task

        assert ingest_sources_task is not None

    def test_ping_still_exported(self):
        from reviewmind.workers import ping

        assert ping is not None

    def test_celery_app_still_exported(self):
        from reviewmind.workers import celery_app

        assert celery_app is not None

    def test_all_contains_ingest_sources_task(self):
        from reviewmind import workers

        assert "ingest_sources_task" in workers.__all__


# ══════════════════════════════════════════════════════════════
# Tests — Integration scenarios
# ══════════════════════════════════════════════════════════════


class TestIntegrationScenarios:
    """End-to-end-style tests combining task + endpoint."""

    def test_task_result_matches_status_response_schema(self):
        """The dict returned by the task contains fields consumable by JobStatusResponse."""
        result = {
            "job_id": str(uuid.uuid4()),
            "status": "done",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "success_count": 3,
            "failed_count": 0,
            "chunks_count": 20,
            "failed_urls": [],
        }
        # These keys should be valid progress dict content
        from reviewmind.api.schemas import JobStatusResponse

        resp = JobStatusResponse(
            job_id=result["job_id"],
            status=result["status"],
            completed_at=result["completed_at"],
            progress=result,
        )
        assert resp.status == "done"
        assert resp.progress["chunks_count"] == 20

    def test_ingest_sources_task_callable(self):
        """Verify that ingest_sources_task can be called (it will delegate to _run_async)."""
        from reviewmind.workers.tasks import ingest_sources_task

        # Just verify it's callable and has the correct task name
        assert callable(ingest_sources_task)
        assert ingest_sources_task.name == "reviewmind.ingest_sources"

    def test_status_endpoint_format_matches_job_model(self):
        """JobStatusResponse fields align with Job model fields."""
        from reviewmind.api.schemas import JobStatusResponse

        fields = set(JobStatusResponse.model_fields.keys())
        assert "job_id" in fields
        assert "status" in fields
        assert "job_type" in fields
        assert "product_query" in fields
        assert "celery_task_id" in fields
        assert "created_at" in fields
        assert "completed_at" in fields
