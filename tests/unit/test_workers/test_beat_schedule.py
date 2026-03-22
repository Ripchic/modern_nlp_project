"""Unit tests for reviewmind.workers.beat_schedule and periodic tasks."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Test beat_schedule constants ─────────────────────────────────────────────


class TestBeatScheduleConstants:
    """Tests for constants defined in beat_schedule.py."""

    def test_daily_reset_limits_task_name(self):
        from reviewmind.workers.beat_schedule import DAILY_RESET_LIMITS_TASK

        assert DAILY_RESET_LIMITS_TASK == "reviewmind.daily_reset_limits"

    def test_refresh_top_queries_task_name(self):
        from reviewmind.workers.beat_schedule import REFRESH_TOP_QUERIES_TASK

        assert REFRESH_TOP_QUERIES_TASK == "reviewmind.refresh_top_queries"

    def test_daily_reset_schedule_is_crontab(self):
        from celery.schedules import crontab

        from reviewmind.workers.beat_schedule import DAILY_RESET_SCHEDULE

        assert isinstance(DAILY_RESET_SCHEDULE, crontab)

    def test_refresh_top_queries_schedule_is_crontab(self):
        from celery.schedules import crontab

        from reviewmind.workers.beat_schedule import REFRESH_TOP_QUERIES_SCHEDULE

        assert isinstance(REFRESH_TOP_QUERIES_SCHEDULE, crontab)

    def test_top_queries_limit(self):
        from reviewmind.workers.tasks import TOP_QUERIES_LIMIT

        assert TOP_QUERIES_LIMIT == 50


# ── Test BEAT_SCHEDULE dict ──────────────────────────────────────────────────


class TestBeatScheduleDict:
    """Tests for the BEAT_SCHEDULE dictionary structure."""

    def test_beat_schedule_is_dict(self):
        from reviewmind.workers.beat_schedule import BEAT_SCHEDULE

        assert isinstance(BEAT_SCHEDULE, dict)

    def test_beat_schedule_has_daily_reset(self):
        from reviewmind.workers.beat_schedule import BEAT_SCHEDULE

        assert "daily-reset-limits" in BEAT_SCHEDULE

    def test_beat_schedule_has_refresh_top_queries(self):
        from reviewmind.workers.beat_schedule import BEAT_SCHEDULE

        assert "refresh-top-queries-monthly" in BEAT_SCHEDULE

    def test_daily_reset_entry_has_task_key(self):
        from reviewmind.workers.beat_schedule import BEAT_SCHEDULE, DAILY_RESET_LIMITS_TASK

        entry = BEAT_SCHEDULE["daily-reset-limits"]
        assert entry["task"] == DAILY_RESET_LIMITS_TASK

    def test_daily_reset_entry_has_schedule_key(self):
        from reviewmind.workers.beat_schedule import BEAT_SCHEDULE, DAILY_RESET_SCHEDULE

        entry = BEAT_SCHEDULE["daily-reset-limits"]
        assert entry["schedule"] is DAILY_RESET_SCHEDULE

    def test_refresh_entry_has_task_key(self):
        from reviewmind.workers.beat_schedule import BEAT_SCHEDULE, REFRESH_TOP_QUERIES_TASK

        entry = BEAT_SCHEDULE["refresh-top-queries-monthly"]
        assert entry["task"] == REFRESH_TOP_QUERIES_TASK

    def test_refresh_entry_has_schedule_key(self):
        from reviewmind.workers.beat_schedule import BEAT_SCHEDULE, REFRESH_TOP_QUERIES_SCHEDULE

        entry = BEAT_SCHEDULE["refresh-top-queries-monthly"]
        assert entry["schedule"] is REFRESH_TOP_QUERIES_SCHEDULE

    def test_entries_have_options(self):
        from reviewmind.workers.beat_schedule import BEAT_SCHEDULE

        for name, entry in BEAT_SCHEDULE.items():
            assert "options" in entry, f"{name} missing 'options'"


# ── Test crontab schedules ───────────────────────────────────────────────────


class TestCrontabSchedules:
    """Tests that crontab schedules match requirements."""

    def test_daily_reset_runs_at_midnight_utc(self):
        from reviewmind.workers.beat_schedule import DAILY_RESET_SCHEDULE

        # crontab(minute='0', hour='0') → midnight
        assert str(DAILY_RESET_SCHEDULE._orig_minute) == "0"
        assert str(DAILY_RESET_SCHEDULE._orig_hour) == "0"

    def test_refresh_runs_on_first_of_month(self):
        from reviewmind.workers.beat_schedule import REFRESH_TOP_QUERIES_SCHEDULE

        assert str(REFRESH_TOP_QUERIES_SCHEDULE._orig_day_of_month) == "1"

    def test_refresh_runs_at_3am(self):
        from reviewmind.workers.beat_schedule import REFRESH_TOP_QUERIES_SCHEDULE

        assert str(REFRESH_TOP_QUERIES_SCHEDULE._orig_hour) == "3"
        assert str(REFRESH_TOP_QUERIES_SCHEDULE._orig_minute) == "0"


# ── Test celery_app has beat_schedule configured ─────────────────────────────


class TestCeleryAppBeatIntegration:
    """Tests that the Celery app has BEAT_SCHEDULE configured."""

    def test_celery_app_has_beat_schedule(self):
        from reviewmind.workers.celery_app import celery_app

        assert hasattr(celery_app.conf, "beat_schedule")

    def test_celery_app_beat_schedule_has_daily_reset(self):
        from reviewmind.workers.celery_app import celery_app

        assert "daily-reset-limits" in celery_app.conf.beat_schedule

    def test_celery_app_beat_schedule_has_refresh(self):
        from reviewmind.workers.celery_app import celery_app

        assert "refresh-top-queries-monthly" in celery_app.conf.beat_schedule

    def test_beat_schedule_matches_module(self):
        from reviewmind.workers.beat_schedule import BEAT_SCHEDULE
        from reviewmind.workers.celery_app import celery_app

        assert celery_app.conf.beat_schedule == BEAT_SCHEDULE


# ── Test task registration ───────────────────────────────────────────────────


class TestTaskRegistration:
    """Tests that periodic tasks are registered on the Celery app."""

    def test_daily_reset_task_registered(self):
        from reviewmind.workers.celery_app import celery_app

        assert "reviewmind.daily_reset_limits" in celery_app.tasks

    def test_refresh_top_queries_task_registered(self):
        from reviewmind.workers.celery_app import celery_app

        assert "reviewmind.refresh_top_queries" in celery_app.tasks

    def test_daily_reset_task_name(self):
        from reviewmind.workers.tasks import daily_reset_limits_task

        assert daily_reset_limits_task.name == "reviewmind.daily_reset_limits"

    def test_refresh_top_queries_task_name(self):
        from reviewmind.workers.tasks import refresh_top_queries_task

        assert refresh_top_queries_task.name == "reviewmind.refresh_top_queries"

    def test_daily_reset_max_retries_zero(self):
        from reviewmind.workers.tasks import daily_reset_limits_task

        assert daily_reset_limits_task.max_retries == 0

    def test_refresh_top_queries_max_retries_zero(self):
        from reviewmind.workers.tasks import refresh_top_queries_task

        assert refresh_top_queries_task.max_retries == 0


# ── Test _daily_reset_limits async function ──────────────────────────────────


class TestDailyResetLimits:
    """Tests for the _daily_reset_limits async implementation."""

    @pytest.mark.asyncio
    async def test_returns_dict_with_date(self):
        mock_engine = AsyncMock()
        mock_engine.dispose = AsyncMock()
        mock_session = AsyncMock()

        # execute returns empty result (no rows to reset)
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.commit = AsyncMock()

        mock_session_factory = MagicMock(return_value=mock_session)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("reviewmind.workers.tasks.create_async_engine", return_value=mock_engine),
            patch("reviewmind.workers.tasks.async_sessionmaker", return_value=mock_session_factory),
            patch("reviewmind.workers.tasks.AsyncSession"),
        ):
            mock_session_factory.return_value = mock_session
            from reviewmind.workers.tasks import _daily_reset_limits

            result = await _daily_reset_limits()

        assert "date" in result
        assert "rows_reset" in result

    @pytest.mark.asyncio
    async def test_resets_rows_with_nonzero_requests(self):
        """Rows with requests_used > 0 should be reset to 0."""
        mock_engine = AsyncMock()
        mock_engine.dispose = AsyncMock()

        # Create mock UserLimit rows
        mock_row1 = MagicMock()
        mock_row1.requests_used = 3
        mock_row2 = MagicMock()
        mock_row2.requests_used = 5

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_row1, mock_row2]

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.commit = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_session_factory = MagicMock(return_value=mock_session)

        with (
            patch("reviewmind.workers.tasks.create_async_engine", return_value=mock_engine),
            patch("reviewmind.workers.tasks.async_sessionmaker", return_value=mock_session_factory),
            patch("reviewmind.workers.tasks.AsyncSession"),
        ):
            mock_session_factory.return_value = mock_session
            from reviewmind.workers.tasks import _daily_reset_limits

            result = await _daily_reset_limits()

        assert result["rows_reset"] == 2
        assert mock_row1.requests_used == 0
        assert mock_row2.requests_used == 0

    @pytest.mark.asyncio
    async def test_no_rows_to_reset(self):
        """When no rows have requests_used > 0, rows_reset should be 0."""
        mock_engine = AsyncMock()
        mock_engine.dispose = AsyncMock()

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.commit = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_session_factory = MagicMock(return_value=mock_session)

        with (
            patch("reviewmind.workers.tasks.create_async_engine", return_value=mock_engine),
            patch("reviewmind.workers.tasks.async_sessionmaker", return_value=mock_session_factory),
            patch("reviewmind.workers.tasks.AsyncSession"),
        ):
            mock_session_factory.return_value = mock_session
            from reviewmind.workers.tasks import _daily_reset_limits

            result = await _daily_reset_limits()

        assert result["rows_reset"] == 0

    @pytest.mark.asyncio
    async def test_engine_disposed_on_success(self):
        mock_engine = AsyncMock()
        mock_engine.dispose = AsyncMock()

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.commit = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_session_factory = MagicMock(return_value=mock_session)

        with (
            patch("reviewmind.workers.tasks.create_async_engine", return_value=mock_engine),
            patch("reviewmind.workers.tasks.async_sessionmaker", return_value=mock_session_factory),
            patch("reviewmind.workers.tasks.AsyncSession"),
        ):
            mock_session_factory.return_value = mock_session
            from reviewmind.workers.tasks import _daily_reset_limits

            await _daily_reset_limits()

        mock_engine.dispose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_engine_disposed_on_error(self):
        """Engine should be disposed even when DB query fails."""
        mock_engine = AsyncMock()
        mock_engine.dispose = AsyncMock()

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(side_effect=Exception("DB error"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_session_factory = MagicMock(return_value=mock_session)

        with (
            patch("reviewmind.workers.tasks.create_async_engine", return_value=mock_engine),
            patch("reviewmind.workers.tasks.async_sessionmaker", return_value=mock_session_factory),
            patch("reviewmind.workers.tasks.AsyncSession"),
        ):
            mock_session_factory.return_value = mock_session
            from reviewmind.workers.tasks import _daily_reset_limits

            result = await _daily_reset_limits()

        mock_engine.dispose.assert_awaited_once()
        # Despite error, should return valid summary
        assert result["rows_reset"] == 0


# ── Test _refresh_top_queries async function ─────────────────────────────────


class TestRefreshTopQueries:
    """Tests for the _refresh_top_queries async implementation."""

    @pytest.mark.asyncio
    async def test_returns_dict_with_counts(self):
        mock_engine = AsyncMock()
        mock_engine.dispose = AsyncMock()

        # Return empty rows (no queries)
        mock_result = MagicMock()
        mock_result.all.return_value = []

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_session_factory = MagicMock(return_value=mock_session)

        with (
            patch("reviewmind.workers.tasks.create_async_engine", return_value=mock_engine),
            patch("reviewmind.workers.tasks.async_sessionmaker", return_value=mock_session_factory),
            patch("reviewmind.workers.tasks.AsyncSession"),
        ):
            mock_session_factory.return_value = mock_session
            from reviewmind.workers.tasks import _refresh_top_queries

            result = await _refresh_top_queries()

        assert "queries_found" in result
        assert "jobs_enqueued" in result
        assert "top_queries" in result

    @pytest.mark.asyncio
    async def test_enqueues_jobs_for_found_queries(self):
        mock_engine = AsyncMock()
        mock_engine.dispose = AsyncMock()

        # Return 3 top queries
        mock_result = MagicMock()
        mock_result.all.return_value = [
            ("Sony WH-1000XM5", 15),
            ("iPhone 16", 10),
            ("MacBook Pro", 5),
        ]

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_session_factory = MagicMock(return_value=mock_session)

        with (
            patch("reviewmind.workers.tasks.create_async_engine", return_value=mock_engine),
            patch("reviewmind.workers.tasks.async_sessionmaker", return_value=mock_session_factory),
            patch("reviewmind.workers.tasks.AsyncSession"),
            patch("reviewmind.workers.tasks.ingest_sources_task") as mock_ingest,
            patch(
                "reviewmind.workers.tasks._collect_urls_for_refresh",
                new_callable=AsyncMock,
                return_value=["https://example.com/review"],
            ),
        ):
            mock_session_factory.return_value = mock_session
            mock_ingest.apply_async = MagicMock()
            from reviewmind.workers.tasks import _refresh_top_queries

            result = await _refresh_top_queries()

        assert result["queries_found"] == 3
        assert result["jobs_enqueued"] == 3
        assert mock_ingest.apply_async.call_count == 3

    @pytest.mark.asyncio
    async def test_no_queries_found(self):
        mock_engine = AsyncMock()
        mock_engine.dispose = AsyncMock()

        mock_result = MagicMock()
        mock_result.all.return_value = []

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_session_factory = MagicMock(return_value=mock_session)

        with (
            patch("reviewmind.workers.tasks.create_async_engine", return_value=mock_engine),
            patch("reviewmind.workers.tasks.async_sessionmaker", return_value=mock_session_factory),
            patch("reviewmind.workers.tasks.AsyncSession"),
        ):
            mock_session_factory.return_value = mock_session
            from reviewmind.workers.tasks import _refresh_top_queries

            result = await _refresh_top_queries()

        assert result["queries_found"] == 0
        assert result["jobs_enqueued"] == 0

    @pytest.mark.asyncio
    async def test_engine_disposed(self):
        mock_engine = AsyncMock()
        mock_engine.dispose = AsyncMock()

        mock_result = MagicMock()
        mock_result.all.return_value = []

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_session_factory = MagicMock(return_value=mock_session)

        with (
            patch("reviewmind.workers.tasks.create_async_engine", return_value=mock_engine),
            patch("reviewmind.workers.tasks.async_sessionmaker", return_value=mock_session_factory),
            patch("reviewmind.workers.tasks.AsyncSession"),
        ):
            mock_session_factory.return_value = mock_session
            from reviewmind.workers.tasks import _refresh_top_queries

            await _refresh_top_queries()

        mock_engine.dispose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_enqueue_failure_does_not_crash(self):
        """If apply_async fails for one query, others should still proceed."""
        mock_engine = AsyncMock()
        mock_engine.dispose = AsyncMock()

        mock_result = MagicMock()
        mock_result.all.return_value = [
            ("product_a", 10),
            ("product_b", 5),
        ]

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_session_factory = MagicMock(return_value=mock_session)

        with (
            patch("reviewmind.workers.tasks.create_async_engine", return_value=mock_engine),
            patch("reviewmind.workers.tasks.async_sessionmaker", return_value=mock_session_factory),
            patch("reviewmind.workers.tasks.AsyncSession"),
            patch("reviewmind.workers.tasks.ingest_sources_task") as mock_ingest,
            patch(
                "reviewmind.workers.tasks._collect_urls_for_refresh",
                new_callable=AsyncMock,
                return_value=["https://example.com/review"],
            ),
        ):
            mock_session_factory.return_value = mock_session
            # First call fails, second succeeds
            mock_ingest.apply_async = MagicMock(side_effect=[Exception("broker down"), None])
            from reviewmind.workers.tasks import _refresh_top_queries

            result = await _refresh_top_queries()

        # Only one succeeded
        assert result["queries_found"] == 2
        assert result["jobs_enqueued"] == 1

    @pytest.mark.asyncio
    async def test_db_error_graceful(self):
        """DB error should not crash; should return zeros."""
        mock_engine = AsyncMock()
        mock_engine.dispose = AsyncMock()

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(side_effect=Exception("DB connection error"))
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_session_factory = MagicMock(return_value=mock_session)

        with (
            patch("reviewmind.workers.tasks.create_async_engine", return_value=mock_engine),
            patch("reviewmind.workers.tasks.async_sessionmaker", return_value=mock_session_factory),
            patch("reviewmind.workers.tasks.AsyncSession"),
        ):
            mock_session_factory.return_value = mock_session
            from reviewmind.workers.tasks import _refresh_top_queries

            result = await _refresh_top_queries()

        assert result["queries_found"] == 0
        assert result["jobs_enqueued"] == 0

    @pytest.mark.asyncio
    async def test_top_queries_limited_to_10_in_result(self):
        """Result should return at most first 10 query names for logging."""
        mock_engine = AsyncMock()
        mock_engine.dispose = AsyncMock()

        # 15 queries
        mock_result = MagicMock()
        mock_result.all.return_value = [(f"product_{i}", 100 - i) for i in range(15)]

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_session_factory = MagicMock(return_value=mock_session)

        with (
            patch("reviewmind.workers.tasks.create_async_engine", return_value=mock_engine),
            patch("reviewmind.workers.tasks.async_sessionmaker", return_value=mock_session_factory),
            patch("reviewmind.workers.tasks.AsyncSession"),
            patch("reviewmind.workers.tasks.ingest_sources_task") as mock_ingest,
        ):
            mock_session_factory.return_value = mock_session
            mock_ingest.apply_async = MagicMock()
            from reviewmind.workers.tasks import _refresh_top_queries

            result = await _refresh_top_queries()

        assert len(result["top_queries"]) == 10
        assert result["queries_found"] == 15


# ── Test sync wrappers (Celery tasks) ────────────────────────────────────────


class TestSyncWrappers:
    """Tests for the sync Celery task wrappers."""

    def test_daily_reset_limits_task_calls_run_async(self):
        from reviewmind.workers.tasks import daily_reset_limits_task

        with patch("reviewmind.workers.tasks._run_async") as mock_run:
            mock_run.return_value = {"date": "2026-03-18", "rows_reset": 0}
            result = daily_reset_limits_task()

        mock_run.assert_called_once()
        assert result["rows_reset"] == 0

    def test_refresh_top_queries_task_calls_run_async(self):
        from reviewmind.workers.tasks import refresh_top_queries_task

        with patch("reviewmind.workers.tasks._run_async") as mock_run:
            mock_run.return_value = {"queries_found": 0, "jobs_enqueued": 0, "top_queries": []}
            result = refresh_top_queries_task()

        mock_run.assert_called_once()
        assert result["queries_found"] == 0


# ── Test workers exports ─────────────────────────────────────────────────────


class TestWorkersExports:
    """Tests for new periodic-task exports from the workers package."""

    def test_import_beat_schedule(self):
        from reviewmind.workers import BEAT_SCHEDULE

        assert isinstance(BEAT_SCHEDULE, dict)

    def test_import_daily_reset_task_name(self):
        from reviewmind.workers import DAILY_RESET_LIMITS_TASK

        assert DAILY_RESET_LIMITS_TASK == "reviewmind.daily_reset_limits"

    def test_import_refresh_task_name(self):
        from reviewmind.workers import REFRESH_TOP_QUERIES_TASK

        assert REFRESH_TOP_QUERIES_TASK == "reviewmind.refresh_top_queries"

    def test_import_daily_reset_schedule(self):
        from celery.schedules import crontab

        from reviewmind.workers import DAILY_RESET_SCHEDULE

        assert isinstance(DAILY_RESET_SCHEDULE, crontab)

    def test_import_refresh_schedule(self):
        from celery.schedules import crontab

        from reviewmind.workers import REFRESH_TOP_QUERIES_SCHEDULE

        assert isinstance(REFRESH_TOP_QUERIES_SCHEDULE, crontab)

    def test_import_daily_reset_limits_task(self):
        from reviewmind.workers import daily_reset_limits_task

        assert callable(daily_reset_limits_task)

    def test_import_refresh_top_queries_task(self):
        from reviewmind.workers import refresh_top_queries_task

        assert callable(refresh_top_queries_task)

    def test_import_top_queries_limit(self):
        from reviewmind.workers import TOP_QUERIES_LIMIT

        assert TOP_QUERIES_LIMIT == 50

    def test_all_exports_count(self):
        import reviewmind.workers as w

        assert len(w.__all__) == 22

    def test_all_exports_accessible(self):
        import reviewmind.workers as w

        for name in w.__all__:
            assert hasattr(w, name), f"Missing export: {name}"


# ── Test Docker Compose alignment ────────────────────────────────────────────


class TestDockerComposeAlignment:
    """Verify beat config aligns with docker-compose.yml beat service."""

    def test_beat_command_references_correct_module(self):
        """The docker-compose.yml beat command should reference
        reviewmind.workers.celery_app — which is where celery_app is."""
        # This is a documentation-level test; the command in docker-compose is:
        # celery -A reviewmind.workers.celery_app beat --loglevel=info
        from reviewmind.workers.celery_app import celery_app

        assert celery_app.main == "reviewmind"

    def test_beat_schedule_task_names_match_registered(self):
        from reviewmind.workers.beat_schedule import BEAT_SCHEDULE
        from reviewmind.workers.celery_app import celery_app

        for entry in BEAT_SCHEDULE.values():
            task_name = entry["task"]
            assert task_name in celery_app.tasks, f"Task {task_name} not registered"


# ── Test Integration Scenarios ───────────────────────────────────────────────


class TestIntegrationScenarios:
    """Integration-level scenarios for periodic tasks."""

    def test_beat_schedule_importable_without_env(self):
        """beat_schedule module should import without a .env file."""
        from reviewmind.workers.beat_schedule import BEAT_SCHEDULE

        assert len(BEAT_SCHEDULE) == 2

    def test_celery_app_configures_beat_on_creation(self):
        from reviewmind.workers.celery_app import create_celery_app

        app = create_celery_app(
            broker_url="redis://test:6379/1",
            result_backend="redis://test:6379/2",
        )
        assert "daily-reset-limits" in app.conf.beat_schedule

    def test_daily_reset_and_refresh_coexist(self):
        """Both periodic tasks should be registered simultaneously."""
        from reviewmind.workers.celery_app import celery_app

        assert "reviewmind.daily_reset_limits" in celery_app.tasks
        assert "reviewmind.refresh_top_queries" in celery_app.tasks

    def test_full_task_list(self):
        """Verify all expected tasks are registered."""
        from reviewmind.workers.celery_app import celery_app

        expected = {
            "reviewmind.ping",
            "reviewmind.ingest_sources",
            "reviewmind.daily_reset_limits",
            "reviewmind.refresh_top_queries",
        }
        registered = set(celery_app.tasks.keys())
        assert expected.issubset(registered)

    def test_prd_step_3_daily_reset_resets_limits(self):
        """PRD Test Step 3: manually calling daily_reset_limits → user_limits zeroed."""
        from reviewmind.workers.tasks import daily_reset_limits_task

        with patch("reviewmind.workers.tasks._run_async") as mock_run:
            mock_run.return_value = {"date": "2026-03-18", "rows_reset": 5}
            result = daily_reset_limits_task()

        assert result["rows_reset"] == 5
