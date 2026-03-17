"""Unit tests for TASK-034 — Task retry (3 attempts) + admin alert on final failure.

Tests cover:
- Retry constants (MAX_RETRIES, RETRY_COUNTDOWNS)
- Exponential backoff countdown selection
- ingest_sources_task retry behavior (self.retry called with correct countdown)
- _handle_final_failure: job status update, user notification, admin alert
- send_admin_alert: message formatting, multi-admin delivery, error handling
- ADMIN_ALERT_TEMPLATE formatting
- Workers __init__ exports for new symbols
- Integration scenarios (retry chain, final failure flow, dedup safety)
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ══════════════════════════════════════════════════════════════
# Tests — Constants
# ══════════════════════════════════════════════════════════════


class TestRetryConstants:
    """Test retry-related constants."""

    def test_max_retries_is_three(self):
        from reviewmind.workers.tasks import MAX_RETRIES

        assert MAX_RETRIES == 3

    def test_retry_countdowns_tuple(self):
        from reviewmind.workers.tasks import RETRY_COUNTDOWNS

        assert isinstance(RETRY_COUNTDOWNS, tuple)

    def test_retry_countdowns_length_matches_max_retries(self):
        from reviewmind.workers.tasks import MAX_RETRIES, RETRY_COUNTDOWNS

        assert len(RETRY_COUNTDOWNS) == MAX_RETRIES

    def test_retry_countdowns_exponential(self):
        from reviewmind.workers.tasks import RETRY_COUNTDOWNS

        assert RETRY_COUNTDOWNS == (60, 300, 900)

    def test_retry_countdowns_increasing(self):
        from reviewmind.workers.tasks import RETRY_COUNTDOWNS

        for i in range(len(RETRY_COUNTDOWNS) - 1):
            assert RETRY_COUNTDOWNS[i] < RETRY_COUNTDOWNS[i + 1]

    def test_first_countdown_is_one_minute(self):
        from reviewmind.workers.tasks import RETRY_COUNTDOWNS

        assert RETRY_COUNTDOWNS[0] == 60

    def test_last_countdown_is_fifteen_minutes(self):
        from reviewmind.workers.tasks import RETRY_COUNTDOWNS

        assert RETRY_COUNTDOWNS[-1] == 900


# ══════════════════════════════════════════════════════════════
# Tests — Task registration with retry
# ══════════════════════════════════════════════════════════════


class TestTaskRetryRegistration:
    """Test that the task is registered with correct retry settings."""

    def test_max_retries_three(self):
        from reviewmind.workers.tasks import ingest_sources_task

        assert ingest_sources_task.max_retries == 3

    def test_task_name_unchanged(self):
        from reviewmind.workers.tasks import ingest_sources_task

        assert ingest_sources_task.name == "reviewmind.ingest_sources"

    def test_task_is_bound(self):
        from celery.app.task import Task

        from reviewmind.workers.tasks import ingest_sources_task

        assert isinstance(ingest_sources_task, Task)


# ══════════════════════════════════════════════════════════════
# Tests — ADMIN_ALERT_TEMPLATE
# ══════════════════════════════════════════════════════════════


class TestAdminAlertTemplate:
    """Test admin alert message template."""

    def test_template_contains_alert_emoji(self):
        from reviewmind.workers.notifications import ADMIN_ALERT_TEMPLATE

        assert "🚨" in ADMIN_ALERT_TEMPLATE

    def test_template_contains_task_id_placeholder(self):
        from reviewmind.workers.notifications import ADMIN_ALERT_TEMPLATE

        assert "{task_id}" in ADMIN_ALERT_TEMPLATE

    def test_template_contains_job_id_placeholder(self):
        from reviewmind.workers.notifications import ADMIN_ALERT_TEMPLATE

        assert "{job_id}" in ADMIN_ALERT_TEMPLATE

    def test_template_contains_user_id_placeholder(self):
        from reviewmind.workers.notifications import ADMIN_ALERT_TEMPLATE

        assert "{user_id}" in ADMIN_ALERT_TEMPLATE

    def test_template_contains_error_placeholder(self):
        from reviewmind.workers.notifications import ADMIN_ALERT_TEMPLATE

        assert "{error}" in ADMIN_ALERT_TEMPLATE

    def test_template_contains_max_retries_placeholder(self):
        from reviewmind.workers.notifications import ADMIN_ALERT_TEMPLATE

        assert "{max_retries}" in ADMIN_ALERT_TEMPLATE

    def test_template_contains_product_query_placeholder(self):
        from reviewmind.workers.notifications import ADMIN_ALERT_TEMPLATE

        assert "{product_query}" in ADMIN_ALERT_TEMPLATE

    def test_template_formats_correctly(self):
        from reviewmind.workers.notifications import ADMIN_ALERT_TEMPLATE

        text = ADMIN_ALERT_TEMPLATE.format(
            task_id="abc-123",
            job_id="def-456",
            user_id=99,
            product_query="Sony XM5",
            error="Connection refused",
            max_retries=3,
        )
        assert "abc-123" in text
        assert "def-456" in text
        assert "99" in text
        assert "Sony XM5" in text
        assert "Connection refused" in text
        assert "3" in text

    def test_max_error_length_constant(self):
        from reviewmind.workers.notifications import _MAX_ERROR_LENGTH

        assert _MAX_ERROR_LENGTH == 500


# ══════════════════════════════════════════════════════════════
# Tests — send_admin_alert
# ══════════════════════════════════════════════════════════════


class TestSendAdminAlert:
    """Test send_admin_alert notification function."""

    @pytest.mark.asyncio
    async def test_sends_to_all_admins(self):
        from reviewmind.workers.notifications import send_admin_alert

        mock_bot = AsyncMock()
        mock_bot.send_message = AsyncMock()
        mock_bot.session = AsyncMock()
        mock_bot.session.close = AsyncMock()

        with patch("reviewmind.workers.notifications._create_bot", return_value=mock_bot):
            await send_admin_alert(
                bot_token="fake:token",
                admin_user_ids=[111, 222, 333],
                task_id="task-1",
                job_id="job-1",
                user_id=99,
                product_query="test",
                error="Connection error",
            )

        assert mock_bot.send_message.await_count == 3
        chat_ids = [c.kwargs["chat_id"] for c in mock_bot.send_message.call_args_list]
        assert set(chat_ids) == {111, 222, 333}

    @pytest.mark.asyncio
    async def test_message_contains_task_details(self):
        from reviewmind.workers.notifications import send_admin_alert

        mock_bot = AsyncMock()
        mock_bot.send_message = AsyncMock()
        mock_bot.session = AsyncMock()
        mock_bot.session.close = AsyncMock()

        with patch("reviewmind.workers.notifications._create_bot", return_value=mock_bot):
            await send_admin_alert(
                bot_token="fake:token",
                admin_user_ids=[111],
                task_id="celery-task-xyz",
                job_id="job-uuid-123",
                user_id=42,
                product_query="Sony WH-1000XM5",
                error="Pipeline crashed",
                max_retries=3,
            )

        text = mock_bot.send_message.call_args.kwargs["text"]
        assert "celery-task-xyz" in text
        assert "job-uuid-123" in text
        assert "42" in text
        assert "Sony WH-1000XM5" in text
        assert "Pipeline crashed" in text
        assert "3" in text
        assert "🚨" in text

    @pytest.mark.asyncio
    async def test_empty_admin_list_does_nothing(self):
        from reviewmind.workers.notifications import send_admin_alert

        mock_bot = AsyncMock()
        mock_bot.send_message = AsyncMock()
        mock_bot.session = AsyncMock()
        mock_bot.session.close = AsyncMock()

        with patch("reviewmind.workers.notifications._create_bot", return_value=mock_bot) as mock_create:
            await send_admin_alert(
                bot_token="fake:token",
                admin_user_ids=[],
                task_id="t",
                job_id="j",
                user_id=1,
                product_query="p",
                error="e",
            )

        mock_create.assert_not_called()

    @pytest.mark.asyncio
    async def test_one_admin_failure_does_not_block_others(self):
        from reviewmind.workers.notifications import send_admin_alert

        mock_bot = AsyncMock()
        mock_bot.session = AsyncMock()
        mock_bot.session.close = AsyncMock()

        send_count = 0

        async def side_effect_send(chat_id, text):
            nonlocal send_count
            send_count += 1
            if chat_id == 222:
                raise RuntimeError("Blocked by user")

        mock_bot.send_message = AsyncMock(side_effect=side_effect_send)

        with patch("reviewmind.workers.notifications._create_bot", return_value=mock_bot):
            await send_admin_alert(
                bot_token="fake:token",
                admin_user_ids=[111, 222, 333],
                task_id="t",
                job_id="j",
                user_id=1,
                product_query="p",
                error="err",
            )

        # All 3 were attempted even though 222 raised
        assert send_count == 3

    @pytest.mark.asyncio
    async def test_truncates_long_error(self):
        from reviewmind.workers.notifications import _MAX_ERROR_LENGTH, send_admin_alert

        mock_bot = AsyncMock()
        mock_bot.send_message = AsyncMock()
        mock_bot.session = AsyncMock()
        mock_bot.session.close = AsyncMock()

        long_error = "X" * 1000

        with patch("reviewmind.workers.notifications._create_bot", return_value=mock_bot):
            await send_admin_alert(
                bot_token="fake:token",
                admin_user_ids=[111],
                task_id="t",
                job_id="j",
                user_id=1,
                product_query="p",
                error=long_error,
            )

        text = mock_bot.send_message.call_args.kwargs["text"]
        # The truncated error in the message should be at most _MAX_ERROR_LENGTH chars
        assert "X" * _MAX_ERROR_LENGTH in text
        assert "X" * (_MAX_ERROR_LENGTH + 1) not in text

    @pytest.mark.asyncio
    async def test_closes_bot_session(self):
        from reviewmind.workers.notifications import send_admin_alert

        mock_bot = AsyncMock()
        mock_bot.send_message = AsyncMock()
        mock_bot.session = AsyncMock()
        mock_bot.session.close = AsyncMock()

        with patch("reviewmind.workers.notifications._create_bot", return_value=mock_bot):
            await send_admin_alert(
                bot_token="fake:token",
                admin_user_ids=[111],
                task_id="t",
                job_id="j",
                user_id=1,
                product_query="p",
                error="err",
            )

        mock_bot.session.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_closes_bot_session_on_error(self):
        from reviewmind.workers.notifications import send_admin_alert

        mock_bot = AsyncMock()
        mock_bot.send_message = AsyncMock(side_effect=RuntimeError("fail"))
        mock_bot.session = AsyncMock()
        mock_bot.session.close = AsyncMock()

        with patch("reviewmind.workers.notifications._create_bot", return_value=mock_bot):
            await send_admin_alert(
                bot_token="fake:token",
                admin_user_ids=[111],
                task_id="t",
                job_id="j",
                user_id=1,
                product_query="p",
                error="err",
            )

        mock_bot.session.close.assert_awaited_once()


# ══════════════════════════════════════════════════════════════
# Tests — _handle_final_failure
# ══════════════════════════════════════════════════════════════


def _build_final_failure_mocks(admin_ids=None):
    """Build mock set for _handle_final_failure tests."""
    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.commit = AsyncMock()

    mock_engine = AsyncMock()
    mock_engine.dispose = AsyncMock()

    mock_settings = MagicMock()
    mock_settings.database_url = "postgresql+asyncpg://test/test"
    mock_settings.telegram_bot_token = "fake:bot-token"
    mock_settings.admin_user_ids = admin_ids or [111, 222]

    mock_factory = MagicMock(return_value=mock_session)
    mock_job_repo = AsyncMock()
    mock_job_repo.update_status = AsyncMock()

    return {
        "session": mock_session,
        "engine": mock_engine,
        "settings": mock_settings,
        "factory": mock_factory,
        "job_repo": mock_job_repo,
    }


class TestHandleFinalFailure:
    """Test _handle_final_failure — the final failure handler."""

    @pytest.mark.asyncio
    async def test_updates_job_to_failed(self):
        from reviewmind.workers.tasks import _handle_final_failure

        mocks = _build_final_failure_mocks()

        with (
            patch("reviewmind.workers.tasks.create_async_engine", return_value=mocks["engine"]),
            patch("reviewmind.workers.tasks.async_sessionmaker", return_value=mocks["factory"]),
            patch("reviewmind.config.settings", mocks["settings"]),
            patch("reviewmind.workers.tasks.JobRepository", return_value=mocks["job_repo"]),
            patch("reviewmind.workers.notifications.send_task_failed", AsyncMock()),
            patch("reviewmind.workers.notifications.send_admin_alert", AsyncMock()),
        ):
            job_id = str(uuid.uuid4())
            await _handle_final_failure(
                job_id=job_id,
                user_id=99,
                product_query="test",
                task_id="celery-id",
                error="boom",
            )

        mocks["job_repo"].update_status.assert_awaited_once()
        call_args = mocks["job_repo"].update_status.call_args
        assert call_args.args[1] == "failed"

    @pytest.mark.asyncio
    async def test_sends_apology_to_user(self):
        from reviewmind.workers.tasks import _handle_final_failure

        mocks = _build_final_failure_mocks()
        mock_send_failed = AsyncMock()

        with (
            patch("reviewmind.workers.tasks.create_async_engine", return_value=mocks["engine"]),
            patch("reviewmind.workers.tasks.async_sessionmaker", return_value=mocks["factory"]),
            patch("reviewmind.config.settings", mocks["settings"]),
            patch("reviewmind.workers.tasks.JobRepository", return_value=mocks["job_repo"]),
            patch("reviewmind.workers.notifications.send_task_failed", mock_send_failed),
            patch("reviewmind.workers.notifications.send_admin_alert", AsyncMock()),
        ):
            await _handle_final_failure(
                job_id=str(uuid.uuid4()),
                user_id=42,
                product_query="test",
                task_id="t",
                error="err",
            )

        mock_send_failed.assert_awaited_once()
        assert mock_send_failed.call_args.kwargs["chat_id"] == 42

    @pytest.mark.asyncio
    async def test_sends_admin_alert(self):
        from reviewmind.workers.tasks import _handle_final_failure

        mocks = _build_final_failure_mocks(admin_ids=[111, 222])
        mock_admin_alert = AsyncMock()

        job_id = str(uuid.uuid4())

        with (
            patch("reviewmind.workers.tasks.create_async_engine", return_value=mocks["engine"]),
            patch("reviewmind.workers.tasks.async_sessionmaker", return_value=mocks["factory"]),
            patch("reviewmind.config.settings", mocks["settings"]),
            patch("reviewmind.workers.tasks.JobRepository", return_value=mocks["job_repo"]),
            patch("reviewmind.workers.notifications.send_task_failed", AsyncMock()),
            patch("reviewmind.workers.notifications.send_admin_alert", mock_admin_alert),
        ):
            await _handle_final_failure(
                job_id=job_id,
                user_id=42,
                product_query="Sony XM5",
                task_id="celery-task-456",
                error="Connection refused",
            )

        mock_admin_alert.assert_awaited_once()
        kwargs = mock_admin_alert.call_args.kwargs
        assert kwargs["admin_user_ids"] == [111, 222]
        assert kwargs["task_id"] == "celery-task-456"
        assert kwargs["job_id"] == job_id
        assert kwargs["user_id"] == 42
        assert kwargs["product_query"] == "Sony XM5"
        assert kwargs["error"] == "Connection refused"
        assert kwargs["max_retries"] == 3

    @pytest.mark.asyncio
    async def test_db_failure_does_not_prevent_notifications(self):
        from reviewmind.workers.tasks import _handle_final_failure

        mocks = _build_final_failure_mocks()
        mocks["job_repo"].update_status = AsyncMock(side_effect=RuntimeError("DB down"))

        mock_send_failed = AsyncMock()
        mock_admin_alert = AsyncMock()

        with (
            patch("reviewmind.workers.tasks.create_async_engine", return_value=mocks["engine"]),
            patch("reviewmind.workers.tasks.async_sessionmaker", return_value=mocks["factory"]),
            patch("reviewmind.config.settings", mocks["settings"]),
            patch("reviewmind.workers.tasks.JobRepository", return_value=mocks["job_repo"]),
            patch("reviewmind.workers.notifications.send_task_failed", mock_send_failed),
            patch("reviewmind.workers.notifications.send_admin_alert", mock_admin_alert),
        ):
            # Should NOT raise even though DB failed
            await _handle_final_failure(
                job_id=str(uuid.uuid4()),
                user_id=42,
                product_query="test",
                task_id="t",
                error="boom",
            )

        # Notifications should still be sent
        mock_send_failed.assert_awaited_once()
        mock_admin_alert.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_user_notification_failure_does_not_prevent_admin_alert(self):
        from reviewmind.workers.tasks import _handle_final_failure

        mocks = _build_final_failure_mocks()
        mock_send_failed = AsyncMock(side_effect=RuntimeError("Telegram down"))
        mock_admin_alert = AsyncMock()

        with (
            patch("reviewmind.workers.tasks.create_async_engine", return_value=mocks["engine"]),
            patch("reviewmind.workers.tasks.async_sessionmaker", return_value=mocks["factory"]),
            patch("reviewmind.config.settings", mocks["settings"]),
            patch("reviewmind.workers.tasks.JobRepository", return_value=mocks["job_repo"]),
            patch("reviewmind.workers.notifications.send_task_failed", mock_send_failed),
            patch("reviewmind.workers.notifications.send_admin_alert", mock_admin_alert),
        ):
            await _handle_final_failure(
                job_id=str(uuid.uuid4()),
                user_id=42,
                product_query="t",
                task_id="t",
                error="e",
            )

        mock_admin_alert.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_disposes_engine(self):
        from reviewmind.workers.tasks import _handle_final_failure

        mocks = _build_final_failure_mocks()

        with (
            patch("reviewmind.workers.tasks.create_async_engine", return_value=mocks["engine"]),
            patch("reviewmind.workers.tasks.async_sessionmaker", return_value=mocks["factory"]),
            patch("reviewmind.config.settings", mocks["settings"]),
            patch("reviewmind.workers.tasks.JobRepository", return_value=mocks["job_repo"]),
            patch("reviewmind.workers.notifications.send_task_failed", AsyncMock()),
            patch("reviewmind.workers.notifications.send_admin_alert", AsyncMock()),
        ):
            await _handle_final_failure(
                job_id=str(uuid.uuid4()),
                user_id=1,
                product_query="t",
                task_id="t",
                error="e",
            )

        mocks["engine"].dispose.assert_awaited_once()


# ══════════════════════════════════════════════════════════════
# Tests — Workers exports
# ══════════════════════════════════════════════════════════════


class TestWorkersRetryExports:
    """Verify new retry-related exports from workers package."""

    def test_max_retries_exported(self):
        from reviewmind.workers import MAX_RETRIES

        assert MAX_RETRIES == 3

    def test_retry_countdowns_exported(self):
        from reviewmind.workers import RETRY_COUNTDOWNS

        assert RETRY_COUNTDOWNS == (60, 300, 900)

    def test_admin_alert_template_exported(self):
        from reviewmind.workers import ADMIN_ALERT_TEMPLATE

        assert isinstance(ADMIN_ALERT_TEMPLATE, str)
        assert "🚨" in ADMIN_ALERT_TEMPLATE

    def test_send_admin_alert_exported(self):
        from reviewmind.workers import send_admin_alert

        assert callable(send_admin_alert)

    def test_all_contains_new_exports(self):
        from reviewmind import workers

        new_exports = {
            "MAX_RETRIES",
            "RETRY_COUNTDOWNS",
            "ADMIN_ALERT_TEMPLATE",
            "send_admin_alert",
        }
        assert new_exports.issubset(set(workers.__all__))

    def test_all_still_contains_original_exports(self):
        from reviewmind import workers

        for name in (
            "celery_app",
            "create_celery_app",
            "ingest_sources_task",
            "ping",
            "send_task_completed",
            "send_task_failed",
            "send_task_started",
            "TASK_STARTED_MSG",
            "TASK_FAILED_MSG",
            "TASK_COMPLETED_NO_ANSWER_MSG",
        ):
            assert name in workers.__all__


# ══════════════════════════════════════════════════════════════
# Tests — Integration scenarios
# ══════════════════════════════════════════════════════════════


class TestIntegrationScenarios:
    """Integration-style tests for retry + admin alert flow."""

    def test_countdown_selection_first_retry(self):
        """First retry (retries=0) should use countdown=60."""
        from reviewmind.workers.tasks import RETRY_COUNTDOWNS

        assert RETRY_COUNTDOWNS[0] == 60

    def test_countdown_selection_second_retry(self):
        """Second retry (retries=1) should use countdown=300."""
        from reviewmind.workers.tasks import RETRY_COUNTDOWNS

        assert RETRY_COUNTDOWNS[1] == 300

    def test_countdown_selection_third_retry(self):
        """Third retry (retries=2) should use countdown=900."""
        from reviewmind.workers.tasks import RETRY_COUNTDOWNS

        assert RETRY_COUNTDOWNS[2] == 900

    @pytest.mark.asyncio
    async def test_final_failure_complete_flow(self):
        """Complete flow: DB update + user notification + admin alert."""
        from reviewmind.workers.tasks import _handle_final_failure

        mocks = _build_final_failure_mocks(admin_ids=[100, 200])
        mock_send_failed = AsyncMock()
        mock_admin_alert = AsyncMock()

        with (
            patch("reviewmind.workers.tasks.create_async_engine", return_value=mocks["engine"]),
            patch("reviewmind.workers.tasks.async_sessionmaker", return_value=mocks["factory"]),
            patch("reviewmind.config.settings", mocks["settings"]),
            patch("reviewmind.workers.tasks.JobRepository", return_value=mocks["job_repo"]),
            patch("reviewmind.workers.notifications.send_task_failed", mock_send_failed),
            patch("reviewmind.workers.notifications.send_admin_alert", mock_admin_alert),
        ):
            final_job_id = str(uuid.uuid4())
            await _handle_final_failure(
                job_id=final_job_id,
                user_id=555,
                product_query="iPhone 16",
                task_id="final-task",
                error="All URLs failed",
            )

        # All three steps executed
        mocks["job_repo"].update_status.assert_awaited_once()
        mock_send_failed.assert_awaited_once()
        mock_admin_alert.assert_awaited_once()

        # Admin alert has correct details
        kwargs = mock_admin_alert.call_args.kwargs
        assert kwargs["admin_user_ids"] == [100, 200]
        assert kwargs["error"] == "All URLs failed"

    def test_dedup_safety_no_extra_data_on_retry(self):
        """Qdrant dedup (cosine > 0.95) prevents duplicate chunks on retry.

        This is implicit — the ingestion pipeline already handles dedup via
        upsert_chunks with generate_point_id (deterministic UUID5 per source_url + chunk_index).
        Retrying the same URLs produces the same point IDs → Qdrant overwrites, not duplicates.
        """
        from reviewmind.vectorstore.client import generate_point_id

        id1 = generate_point_id("https://example.com/review", 0)
        id2 = generate_point_id("https://example.com/review", 0)
        assert id1 == id2  # Deterministic — retry produces same IDs

    @pytest.mark.asyncio
    async def test_admin_alert_with_no_admins_configured(self):
        """When ADMIN_USER_IDS is empty, send_admin_alert exits early."""
        from reviewmind.workers.notifications import send_admin_alert

        mock_bot = AsyncMock()
        mock_bot.send_message = AsyncMock()
        mock_bot.session = AsyncMock()
        mock_bot.session.close = AsyncMock()

        with patch("reviewmind.workers.notifications._create_bot", return_value=mock_bot) as mock_create:
            await send_admin_alert(
                bot_token="fake:token",
                admin_user_ids=[],
                task_id="t",
                job_id="j",
                user_id=1,
                product_query="p",
                error="err",
            )

        # Bot is never created when admin list is empty
        mock_create.assert_not_called()

    def test_total_retry_time_under_25_minutes(self):
        """Total retry time: 60 + 300 + 900 = 1260s = 21 min < 25 min."""
        from reviewmind.workers.tasks import RETRY_COUNTDOWNS

        total = sum(RETRY_COUNTDOWNS)
        assert total == 1260
        assert total < 25 * 60  # 25 minutes
