"""Unit tests for TASK-033 — Push notifications from Celery tasks.

Tests cover:
- Message constants and templates
- _create_bot helper
- _run_rag_query helper
- send_task_started
- send_task_completed (success / empty answer / RAG error / send error)
- send_task_failed
- Integration with _ingest_sources (notification calls after task completes)
- Workers __init__ exports for notification symbols
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ══════════════════════════════════════════════════════════════
# Tests — Constants
# ══════════════════════════════════════════════════════════════


class TestConstants:
    """Test notification message constants."""

    def test_task_started_msg_contains_emoji(self):
        from reviewmind.workers.notifications import TASK_STARTED_MSG

        assert "⏳" in TASK_STARTED_MSG

    def test_task_started_msg_mentions_time(self):
        from reviewmind.workers.notifications import TASK_STARTED_MSG

        assert "3 мин" in TASK_STARTED_MSG

    def test_task_failed_msg_contains_emoji(self):
        from reviewmind.workers.notifications import TASK_FAILED_MSG

        assert "😔" in TASK_FAILED_MSG

    def test_task_completed_no_answer_msg_nonempty(self):
        from reviewmind.workers.notifications import TASK_COMPLETED_NO_ANSWER_MSG

        assert len(TASK_COMPLETED_NO_ANSWER_MSG) > 10

    def test_max_answer_length(self):
        from reviewmind.workers.notifications import _MAX_ANSWER_LENGTH

        assert _MAX_ANSWER_LENGTH == 4096

    def test_task_started_msg_mentions_result(self):
        from reviewmind.workers.notifications import TASK_STARTED_MSG

        assert "результат" in TASK_STARTED_MSG.lower()


# ══════════════════════════════════════════════════════════════
# Tests — _create_bot
# ══════════════════════════════════════════════════════════════


class TestCreateBot:
    """Test the _create_bot helper."""

    # aiogram validates tokens: left part must be digits, right part non-empty
    _VALID_TOKEN = "123456789:ABCdefGHIjklMNOpqrSTUvwxYZ"

    def test_returns_bot_instance(self):
        from reviewmind.workers.notifications import _create_bot

        bot = _create_bot(self._VALID_TOKEN)
        assert bot.token == self._VALID_TOKEN

    def test_html_parse_mode(self):
        from aiogram.enums import ParseMode

        from reviewmind.workers.notifications import _create_bot

        bot = _create_bot(self._VALID_TOKEN)
        assert bot.default.parse_mode == ParseMode.HTML


# ══════════════════════════════════════════════════════════════
# Tests — _run_rag_query
# ══════════════════════════════════════════════════════════════


class TestRunRagQuery:
    """Test the _run_rag_query helper."""

    @pytest.mark.asyncio
    async def test_returns_answer_on_success(self):
        from reviewmind.workers.notifications import _run_rag_query

        mock_response = MagicMock()
        mock_response.answer = "Analysis result"
        mock_response.error = None

        mock_rag = AsyncMock()
        mock_rag.query = AsyncMock(return_value=mock_response)
        mock_rag.__aenter__ = AsyncMock(return_value=mock_rag)
        mock_rag.__aexit__ = AsyncMock(return_value=False)

        with patch("reviewmind.workers.notifications.RAGPipeline", return_value=mock_rag):
            result = await _run_rag_query(AsyncMock(), "test product")

        assert result == "Analysis result"

    @pytest.mark.asyncio
    async def test_returns_none_on_empty_answer(self):
        from reviewmind.workers.notifications import _run_rag_query

        mock_response = MagicMock()
        mock_response.answer = ""
        mock_response.error = None

        mock_rag = AsyncMock()
        mock_rag.query = AsyncMock(return_value=mock_response)
        mock_rag.__aenter__ = AsyncMock(return_value=mock_rag)
        mock_rag.__aexit__ = AsyncMock(return_value=False)

        with patch("reviewmind.workers.notifications.RAGPipeline", return_value=mock_rag):
            result = await _run_rag_query(AsyncMock(), "test product")

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_none_on_error_with_empty_answer(self):
        from reviewmind.workers.notifications import _run_rag_query

        mock_response = MagicMock()
        mock_response.answer = ""
        mock_response.error = "Embedding error"

        mock_rag = AsyncMock()
        mock_rag.query = AsyncMock(return_value=mock_response)
        mock_rag.__aenter__ = AsyncMock(return_value=mock_rag)
        mock_rag.__aexit__ = AsyncMock(return_value=False)

        with patch("reviewmind.workers.notifications.RAGPipeline", return_value=mock_rag):
            result = await _run_rag_query(AsyncMock(), "test product")

        assert result is None

    @pytest.mark.asyncio
    async def test_passes_product_query_to_rag(self):
        from reviewmind.workers.notifications import _run_rag_query

        mock_response = MagicMock()
        mock_response.answer = "ok"
        mock_response.error = None

        mock_rag = AsyncMock()
        mock_rag.query = AsyncMock(return_value=mock_response)
        mock_rag.__aenter__ = AsyncMock(return_value=mock_rag)
        mock_rag.__aexit__ = AsyncMock(return_value=False)

        with patch("reviewmind.workers.notifications.RAGPipeline", return_value=mock_rag):
            await _run_rag_query(AsyncMock(), "Sony WH-1000XM5")

        call_kwargs = mock_rag.query.call_args
        assert call_kwargs.kwargs["product_query"] == "Sony WH-1000XM5"
        assert call_kwargs.kwargs["user_query"] == "Sony WH-1000XM5"


# ══════════════════════════════════════════════════════════════
# Tests — send_task_started
# ══════════════════════════════════════════════════════════════


class TestSendTaskStarted:
    """Test send_task_started notification."""

    @pytest.mark.asyncio
    async def test_sends_message_to_chat(self):
        from reviewmind.workers.notifications import TASK_STARTED_MSG, send_task_started

        mock_bot = AsyncMock()
        mock_bot.send_message = AsyncMock()
        mock_bot.session = AsyncMock()
        mock_bot.session.close = AsyncMock()

        with patch("reviewmind.workers.notifications._create_bot", return_value=mock_bot):
            await send_task_started(bot_token="fake:token", chat_id=12345)

        mock_bot.send_message.assert_awaited_once_with(
            chat_id=12345, text=TASK_STARTED_MSG
        )

    @pytest.mark.asyncio
    async def test_closes_bot_session(self):
        from reviewmind.workers.notifications import send_task_started

        mock_bot = AsyncMock()
        mock_bot.send_message = AsyncMock()
        mock_bot.session = AsyncMock()
        mock_bot.session.close = AsyncMock()

        with patch("reviewmind.workers.notifications._create_bot", return_value=mock_bot):
            await send_task_started(bot_token="fake:token", chat_id=12345)

        mock_bot.session.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_does_not_raise_on_send_error(self):
        from reviewmind.workers.notifications import send_task_started

        mock_bot = AsyncMock()
        mock_bot.send_message = AsyncMock(side_effect=RuntimeError("Network error"))
        mock_bot.session = AsyncMock()
        mock_bot.session.close = AsyncMock()

        with patch("reviewmind.workers.notifications._create_bot", return_value=mock_bot):
            # Should NOT raise
            await send_task_started(bot_token="fake:token", chat_id=12345)

    @pytest.mark.asyncio
    async def test_closes_session_even_on_error(self):
        from reviewmind.workers.notifications import send_task_started

        mock_bot = AsyncMock()
        mock_bot.send_message = AsyncMock(side_effect=RuntimeError("err"))
        mock_bot.session = AsyncMock()
        mock_bot.session.close = AsyncMock()

        with patch("reviewmind.workers.notifications._create_bot", return_value=mock_bot):
            await send_task_started(bot_token="fake:token", chat_id=99)

        mock_bot.session.close.assert_awaited_once()


# ══════════════════════════════════════════════════════════════
# Tests — send_task_completed
# ══════════════════════════════════════════════════════════════


class TestSendTaskCompleted:
    """Test send_task_completed notification."""

    @pytest.mark.asyncio
    async def test_sends_rag_analysis(self):
        from reviewmind.workers.notifications import send_task_completed

        mock_bot = AsyncMock()
        mock_bot.send_message = AsyncMock()
        mock_bot.session = AsyncMock()
        mock_bot.session.close = AsyncMock()

        mock_qdrant = AsyncMock()
        mock_qdrant.close = AsyncMock()

        with (
            patch("reviewmind.workers.notifications._create_bot", return_value=mock_bot),
            patch("reviewmind.workers.notifications.AsyncQdrantClient", return_value=mock_qdrant),
            patch(
                "reviewmind.workers.notifications._run_rag_query",
                return_value="Great headphones analysis",
            ),
        ):
            await send_task_completed(
                bot_token="fake:token",
                chat_id=12345,
                product_query="Sony WH-1000XM5",
                qdrant_url="http://localhost:6333",
            )

        mock_bot.send_message.assert_awaited_once()
        call_kwargs = mock_bot.send_message.call_args.kwargs
        assert call_kwargs["text"] == "Great headphones analysis"
        assert call_kwargs["reply_markup"] is not None  # feedback_keyboard

    @pytest.mark.asyncio
    async def test_sends_feedback_keyboard(self):
        from reviewmind.workers.notifications import send_task_completed

        mock_bot = AsyncMock()
        mock_bot.send_message = AsyncMock()
        mock_bot.session = AsyncMock()
        mock_bot.session.close = AsyncMock()

        mock_qdrant = AsyncMock()
        mock_qdrant.close = AsyncMock()

        with (
            patch("reviewmind.workers.notifications._create_bot", return_value=mock_bot),
            patch("reviewmind.workers.notifications.AsyncQdrantClient", return_value=mock_qdrant),
            patch("reviewmind.workers.notifications._run_rag_query", return_value="Analysis"),
        ):
            await send_task_completed(
                bot_token="fake:token",
                chat_id=12345,
                product_query="test",
                qdrant_url="http://localhost:6333",
            )

        from aiogram.types import InlineKeyboardMarkup

        call_kwargs = mock_bot.send_message.call_args.kwargs
        assert isinstance(call_kwargs["reply_markup"], InlineKeyboardMarkup)

    @pytest.mark.asyncio
    async def test_sends_fallback_on_empty_answer(self):
        from reviewmind.workers.notifications import (
            TASK_COMPLETED_NO_ANSWER_MSG,
            send_task_completed,
        )

        mock_bot = AsyncMock()
        mock_bot.send_message = AsyncMock()
        mock_bot.session = AsyncMock()
        mock_bot.session.close = AsyncMock()

        mock_qdrant = AsyncMock()
        mock_qdrant.close = AsyncMock()

        with (
            patch("reviewmind.workers.notifications._create_bot", return_value=mock_bot),
            patch("reviewmind.workers.notifications.AsyncQdrantClient", return_value=mock_qdrant),
            patch("reviewmind.workers.notifications._run_rag_query", return_value=None),
        ):
            await send_task_completed(
                bot_token="fake:token",
                chat_id=12345,
                product_query="test",
                qdrant_url="http://localhost:6333",
            )

        call_kwargs = mock_bot.send_message.call_args.kwargs
        assert call_kwargs["text"] == TASK_COMPLETED_NO_ANSWER_MSG

    @pytest.mark.asyncio
    async def test_truncates_long_answer(self):
        from reviewmind.workers.notifications import send_task_completed

        mock_bot = AsyncMock()
        mock_bot.send_message = AsyncMock()
        mock_bot.session = AsyncMock()
        mock_bot.session.close = AsyncMock()

        mock_qdrant = AsyncMock()
        mock_qdrant.close = AsyncMock()

        long_answer = "A" * 5000

        with (
            patch("reviewmind.workers.notifications._create_bot", return_value=mock_bot),
            patch("reviewmind.workers.notifications.AsyncQdrantClient", return_value=mock_qdrant),
            patch(
                "reviewmind.workers.notifications._run_rag_query",
                return_value=long_answer,
            ),
        ):
            await send_task_completed(
                bot_token="fake:token",
                chat_id=12345,
                product_query="test",
                qdrant_url="http://localhost:6333",
            )

        call_kwargs = mock_bot.send_message.call_args.kwargs
        assert len(call_kwargs["text"]) <= 4096
        assert call_kwargs["text"].endswith("...")

    @pytest.mark.asyncio
    async def test_closes_qdrant_and_bot_on_success(self):
        from reviewmind.workers.notifications import send_task_completed

        mock_bot = AsyncMock()
        mock_bot.send_message = AsyncMock()
        mock_bot.session = AsyncMock()
        mock_bot.session.close = AsyncMock()

        mock_qdrant = AsyncMock()
        mock_qdrant.close = AsyncMock()

        with (
            patch("reviewmind.workers.notifications._create_bot", return_value=mock_bot),
            patch("reviewmind.workers.notifications.AsyncQdrantClient", return_value=mock_qdrant),
            patch("reviewmind.workers.notifications._run_rag_query", return_value="ok"),
        ):
            await send_task_completed(
                bot_token="fake:token",
                chat_id=12345,
                product_query="test",
                qdrant_url="http://localhost:6333",
            )

        mock_qdrant.close.assert_awaited_once()
        mock_bot.session.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_handles_rag_exception_gracefully(self):
        from reviewmind.workers.notifications import (
            TASK_COMPLETED_NO_ANSWER_MSG,
            send_task_completed,
        )

        mock_bot = AsyncMock()
        mock_bot.send_message = AsyncMock()
        mock_bot.session = AsyncMock()
        mock_bot.session.close = AsyncMock()

        mock_qdrant = AsyncMock()
        mock_qdrant.close = AsyncMock()

        with (
            patch("reviewmind.workers.notifications._create_bot", return_value=mock_bot),
            patch("reviewmind.workers.notifications.AsyncQdrantClient", return_value=mock_qdrant),
            patch(
                "reviewmind.workers.notifications._run_rag_query",
                side_effect=RuntimeError("RAG crashed"),
            ),
        ):
            # Should NOT raise
            await send_task_completed(
                bot_token="fake:token",
                chat_id=12345,
                product_query="test",
                qdrant_url="http://localhost:6333",
            )

        # Should try to send fallback
        assert mock_bot.send_message.await_count >= 1
        last_call = mock_bot.send_message.call_args
        assert last_call.kwargs["text"] == TASK_COMPLETED_NO_ANSWER_MSG

    @pytest.mark.asyncio
    async def test_closes_resources_on_rag_error(self):
        from reviewmind.workers.notifications import send_task_completed

        mock_bot = AsyncMock()
        mock_bot.send_message = AsyncMock()
        mock_bot.session = AsyncMock()
        mock_bot.session.close = AsyncMock()

        mock_qdrant = AsyncMock()
        mock_qdrant.close = AsyncMock()

        with (
            patch("reviewmind.workers.notifications._create_bot", return_value=mock_bot),
            patch("reviewmind.workers.notifications.AsyncQdrantClient", return_value=mock_qdrant),
            patch(
                "reviewmind.workers.notifications._run_rag_query",
                side_effect=RuntimeError("boom"),
            ),
        ):
            await send_task_completed(
                bot_token="fake:token",
                chat_id=12345,
                product_query="test",
                qdrant_url="http://localhost:6333",
            )

        mock_qdrant.close.assert_awaited_once()
        mock_bot.session.close.assert_awaited_once()


# ══════════════════════════════════════════════════════════════
# Tests — send_task_failed
# ══════════════════════════════════════════════════════════════


class TestSendTaskFailed:
    """Test send_task_failed notification."""

    @pytest.mark.asyncio
    async def test_sends_apology_message(self):
        from reviewmind.workers.notifications import TASK_FAILED_MSG, send_task_failed

        mock_bot = AsyncMock()
        mock_bot.send_message = AsyncMock()
        mock_bot.session = AsyncMock()
        mock_bot.session.close = AsyncMock()

        with patch("reviewmind.workers.notifications._create_bot", return_value=mock_bot):
            await send_task_failed(bot_token="fake:token", chat_id=12345)

        mock_bot.send_message.assert_awaited_once_with(
            chat_id=12345, text=TASK_FAILED_MSG
        )

    @pytest.mark.asyncio
    async def test_closes_bot_session(self):
        from reviewmind.workers.notifications import send_task_failed

        mock_bot = AsyncMock()
        mock_bot.send_message = AsyncMock()
        mock_bot.session = AsyncMock()
        mock_bot.session.close = AsyncMock()

        with patch("reviewmind.workers.notifications._create_bot", return_value=mock_bot):
            await send_task_failed(bot_token="fake:token", chat_id=12345)

        mock_bot.session.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_does_not_raise_on_send_error(self):
        from reviewmind.workers.notifications import send_task_failed

        mock_bot = AsyncMock()
        mock_bot.send_message = AsyncMock(side_effect=RuntimeError("blocked"))
        mock_bot.session = AsyncMock()
        mock_bot.session.close = AsyncMock()

        with patch("reviewmind.workers.notifications._create_bot", return_value=mock_bot):
            # Should NOT raise
            await send_task_failed(bot_token="fake:token", chat_id=12345)


# ══════════════════════════════════════════════════════════════
# Tests — Integration: _ingest_sources calls notifications
# ══════════════════════════════════════════════════════════════


def _mock_ingestion_result(success_count=2, failed_count=0, chunks_count=10):
    """Create a mock IngestionResult."""
    result = MagicMock()
    result.success_count = success_count
    result.failed_count = failed_count
    result.chunks_count = chunks_count
    result.failed_urls = []
    return result


def _build_ingest_mocks(ingestion_result=None, settings_extras=None):
    """Build the common mock set for _ingest_sources tests."""
    if ingestion_result is None:
        ingestion_result = _mock_ingestion_result()

    mock_session = AsyncMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=False)
    mock_session.commit = AsyncMock()

    mock_engine = AsyncMock()
    mock_engine.dispose = AsyncMock()

    mock_pipeline = AsyncMock()
    mock_pipeline.ingest_urls = AsyncMock(return_value=ingestion_result)
    mock_pipeline.__aenter__ = AsyncMock(return_value=mock_pipeline)
    mock_pipeline.__aexit__ = AsyncMock(return_value=False)

    mock_qdrant = AsyncMock()
    mock_qdrant.close = AsyncMock()

    mock_settings = MagicMock()
    mock_settings.database_url = "postgresql+asyncpg://test/test"
    mock_settings.qdrant_url = "http://localhost:6333"
    mock_settings.telegram_bot_token = "fake:bot-token"
    if settings_extras:
        for k, v in settings_extras.items():
            setattr(mock_settings, k, v)

    mock_factory = MagicMock(return_value=mock_session)
    mock_job_repo = AsyncMock()
    mock_job_repo.update_status = AsyncMock()

    return {
        "session": mock_session,
        "engine": mock_engine,
        "pipeline": mock_pipeline,
        "qdrant": mock_qdrant,
        "settings": mock_settings,
        "factory": mock_factory,
        "job_repo": mock_job_repo,
    }


class TestIngestSourcesNotifications:
    """Test that _ingest_sources calls push notifications."""

    @pytest.mark.asyncio
    async def test_calls_send_task_completed_on_success(self):
        import uuid

        from reviewmind.workers.tasks import _ingest_sources

        mocks = _build_ingest_mocks()
        mock_send_completed = AsyncMock()
        mock_send_failed = AsyncMock()

        with (
            patch("reviewmind.workers.tasks.create_async_engine", return_value=mocks["engine"]),
            patch("reviewmind.workers.tasks.async_sessionmaker", return_value=mocks["factory"]),
            patch("reviewmind.workers.tasks.AsyncQdrantClient", return_value=mocks["qdrant"]),
            patch("reviewmind.config.settings", mocks["settings"]),
            patch("reviewmind.workers.tasks.IngestionPipeline", return_value=mocks["pipeline"]),
            patch("reviewmind.workers.tasks.JobRepository", return_value=mocks["job_repo"]),
            patch(
                "reviewmind.workers.notifications.send_task_completed",
                mock_send_completed,
            ),
            patch(
                "reviewmind.workers.notifications.send_task_failed",
                mock_send_failed,
            ),
        ):
            await _ingest_sources(
                job_id=str(uuid.uuid4()),
                user_id=12345,
                product_query="test product",
                urls=["https://example.com"],
            )

        mock_send_completed.assert_awaited_once()
        call_kwargs = mock_send_completed.call_args.kwargs
        assert call_kwargs["chat_id"] == 12345
        assert call_kwargs["product_query"] == "test product"
        assert call_kwargs["bot_token"] == "fake:bot-token"
        assert call_kwargs["qdrant_url"] == "http://localhost:6333"

        mock_send_failed.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_calls_send_task_failed_on_failure(self):
        import uuid

        from reviewmind.workers.tasks import _ingest_sources

        mocks = _build_ingest_mocks(
            ingestion_result=_mock_ingestion_result(success_count=0, failed_count=2, chunks_count=0)
        )
        mock_send_completed = AsyncMock()
        mock_send_failed = AsyncMock()

        with (
            patch("reviewmind.workers.tasks.create_async_engine", return_value=mocks["engine"]),
            patch("reviewmind.workers.tasks.async_sessionmaker", return_value=mocks["factory"]),
            patch("reviewmind.workers.tasks.AsyncQdrantClient", return_value=mocks["qdrant"]),
            patch("reviewmind.config.settings", mocks["settings"]),
            patch("reviewmind.workers.tasks.IngestionPipeline", return_value=mocks["pipeline"]),
            patch("reviewmind.workers.tasks.JobRepository", return_value=mocks["job_repo"]),
            patch(
                "reviewmind.workers.notifications.send_task_completed",
                mock_send_completed,
            ),
            patch(
                "reviewmind.workers.notifications.send_task_failed",
                mock_send_failed,
            ),
        ):
            await _ingest_sources(
                job_id=str(uuid.uuid4()),
                user_id=12345,
                product_query="test",
                urls=["https://bad.com"],
            )

        mock_send_failed.assert_awaited_once()
        call_kwargs = mock_send_failed.call_args.kwargs
        assert call_kwargs["chat_id"] == 12345
        assert call_kwargs["bot_token"] == "fake:bot-token"

        mock_send_completed.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_calls_send_task_failed_on_pipeline_error(self):
        import uuid

        from reviewmind.workers.tasks import _ingest_sources

        mocks = _build_ingest_mocks()
        mocks["pipeline"].ingest_urls = AsyncMock(side_effect=RuntimeError("Pipeline boom"))

        mock_send_completed = AsyncMock()
        mock_send_failed = AsyncMock()

        with (
            patch("reviewmind.workers.tasks.create_async_engine", return_value=mocks["engine"]),
            patch("reviewmind.workers.tasks.async_sessionmaker", return_value=mocks["factory"]),
            patch("reviewmind.workers.tasks.AsyncQdrantClient", return_value=mocks["qdrant"]),
            patch("reviewmind.config.settings", mocks["settings"]),
            patch("reviewmind.workers.tasks.IngestionPipeline", return_value=mocks["pipeline"]),
            patch("reviewmind.workers.tasks.JobRepository", return_value=mocks["job_repo"]),
            patch(
                "reviewmind.workers.notifications.send_task_completed",
                mock_send_completed,
            ),
            patch(
                "reviewmind.workers.notifications.send_task_failed",
                mock_send_failed,
            ),
        ):
            await _ingest_sources(
                job_id=str(uuid.uuid4()),
                user_id=12345,
                product_query="test",
                urls=["https://example.com"],
            )

        mock_send_failed.assert_awaited_once()
        mock_send_completed.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_notification_failure_does_not_crash_task(self):
        """If notification sending fails, the task still completes normally."""
        import uuid

        from reviewmind.workers.tasks import _ingest_sources

        mocks = _build_ingest_mocks()
        mock_send_completed = AsyncMock(side_effect=RuntimeError("Telegram down"))

        with (
            patch("reviewmind.workers.tasks.create_async_engine", return_value=mocks["engine"]),
            patch("reviewmind.workers.tasks.async_sessionmaker", return_value=mocks["factory"]),
            patch("reviewmind.workers.tasks.AsyncQdrantClient", return_value=mocks["qdrant"]),
            patch("reviewmind.config.settings", mocks["settings"]),
            patch("reviewmind.workers.tasks.IngestionPipeline", return_value=mocks["pipeline"]),
            patch("reviewmind.workers.tasks.JobRepository", return_value=mocks["job_repo"]),
            patch(
                "reviewmind.workers.notifications.send_task_completed",
                mock_send_completed,
            ),
        ):
            result = await _ingest_sources(
                job_id=str(uuid.uuid4()),
                user_id=12345,
                product_query="test",
                urls=["https://example.com"],
            )

        # Task should still return normally
        assert result["status"] == "done"

    @pytest.mark.asyncio
    async def test_notification_after_engine_dispose(self):
        """Notifications are sent after engine.dispose(), so they don't hold DB connections."""
        import uuid

        from reviewmind.workers.tasks import _ingest_sources

        mocks = _build_ingest_mocks()
        call_order: list[str] = []

        original_dispose = mocks["engine"].dispose

        async def track_dispose():
            call_order.append("dispose")
            return await original_dispose()

        mocks["engine"].dispose = track_dispose

        async def track_send_completed(**kwargs):
            call_order.append("send_completed")

        with (
            patch("reviewmind.workers.tasks.create_async_engine", return_value=mocks["engine"]),
            patch("reviewmind.workers.tasks.async_sessionmaker", return_value=mocks["factory"]),
            patch("reviewmind.workers.tasks.AsyncQdrantClient", return_value=mocks["qdrant"]),
            patch("reviewmind.config.settings", mocks["settings"]),
            patch("reviewmind.workers.tasks.IngestionPipeline", return_value=mocks["pipeline"]),
            patch("reviewmind.workers.tasks.JobRepository", return_value=mocks["job_repo"]),
            patch(
                "reviewmind.workers.notifications.send_task_completed",
                track_send_completed,
            ),
        ):
            await _ingest_sources(
                job_id=str(uuid.uuid4()),
                user_id=12345,
                product_query="test",
                urls=["https://example.com"],
            )

        assert "dispose" in call_order
        assert "send_completed" in call_order
        assert call_order.index("dispose") < call_order.index("send_completed")


# ══════════════════════════════════════════════════════════════
# Tests — Workers __init__ exports
# ══════════════════════════════════════════════════════════════


class TestWorkersNotificationExports:
    """Verify that notification symbols are exported from workers package."""

    def test_send_task_started_exported(self):
        from reviewmind.workers import send_task_started

        assert callable(send_task_started)

    def test_send_task_completed_exported(self):
        from reviewmind.workers import send_task_completed

        assert callable(send_task_completed)

    def test_send_task_failed_exported(self):
        from reviewmind.workers import send_task_failed

        assert callable(send_task_failed)

    def test_task_started_msg_exported(self):
        from reviewmind.workers import TASK_STARTED_MSG

        assert isinstance(TASK_STARTED_MSG, str)

    def test_task_failed_msg_exported(self):
        from reviewmind.workers import TASK_FAILED_MSG

        assert isinstance(TASK_FAILED_MSG, str)

    def test_task_completed_no_answer_msg_exported(self):
        from reviewmind.workers import TASK_COMPLETED_NO_ANSWER_MSG

        assert isinstance(TASK_COMPLETED_NO_ANSWER_MSG, str)

    def test_all_contains_notification_names(self):
        from reviewmind import workers

        expected = {
            "send_task_started",
            "send_task_completed",
            "send_task_failed",
            "TASK_STARTED_MSG",
            "TASK_FAILED_MSG",
            "TASK_COMPLETED_NO_ANSWER_MSG",
        }
        assert expected.issubset(set(workers.__all__))

    def test_all_still_contains_original_exports(self):
        from reviewmind import workers

        for name in ("celery_app", "create_celery_app", "ingest_sources_task", "ping"):
            assert name in workers.__all__


# ══════════════════════════════════════════════════════════════
# Tests — Integration scenarios
# ══════════════════════════════════════════════════════════════


class TestIntegrationScenarios:
    """Integration-style tests for the complete notification workflow."""

    @pytest.mark.asyncio
    async def test_full_success_flow(self):
        """Complete flow: task starts → ingestion succeeds → push analysis → user receives."""
        from reviewmind.workers.notifications import send_task_completed, send_task_started

        mock_bot = AsyncMock()
        mock_bot.send_message = AsyncMock()
        mock_bot.session = AsyncMock()
        mock_bot.session.close = AsyncMock()

        mock_qdrant = AsyncMock()
        mock_qdrant.close = AsyncMock()

        with (
            patch("reviewmind.workers.notifications._create_bot", return_value=mock_bot),
            patch("reviewmind.workers.notifications.AsyncQdrantClient", return_value=mock_qdrant),
            patch(
                "reviewmind.workers.notifications._run_rag_query",
                return_value="✅ Плюсы: отличного качества",
            ),
        ):
            await send_task_started(bot_token="t", chat_id=100)
            await send_task_completed(
                bot_token="t",
                chat_id=100,
                product_query="Наушники",
                qdrant_url="http://localhost:6333",
            )

        # Two messages: started + analysis
        assert mock_bot.send_message.await_count == 2

    @pytest.mark.asyncio
    async def test_full_failure_flow(self):
        """Complete flow: task starts → ingestion fails → apology sent."""
        from reviewmind.workers.notifications import (
            TASK_FAILED_MSG,
            TASK_STARTED_MSG,
            send_task_failed,
            send_task_started,
        )

        mock_bot = AsyncMock()
        mock_bot.send_message = AsyncMock()
        mock_bot.session = AsyncMock()
        mock_bot.session.close = AsyncMock()

        with patch("reviewmind.workers.notifications._create_bot", return_value=mock_bot):
            await send_task_started(bot_token="t", chat_id=200)
            await send_task_failed(bot_token="t", chat_id=200)

        calls = mock_bot.send_message.call_args_list
        assert len(calls) == 2
        assert calls[0].kwargs["text"] == TASK_STARTED_MSG
        assert calls[1].kwargs["text"] == TASK_FAILED_MSG

    @pytest.mark.asyncio
    async def test_push_within_5_min_acceptance(self):
        """Acceptance: push notification is sent synchronously — no artificial delay."""
        import time

        from reviewmind.workers.notifications import send_task_completed

        mock_bot = AsyncMock()
        mock_bot.send_message = AsyncMock()
        mock_bot.session = AsyncMock()
        mock_bot.session.close = AsyncMock()

        mock_qdrant = AsyncMock()
        mock_qdrant.close = AsyncMock()

        start = time.monotonic()

        with (
            patch("reviewmind.workers.notifications._create_bot", return_value=mock_bot),
            patch("reviewmind.workers.notifications.AsyncQdrantClient", return_value=mock_qdrant),
            patch("reviewmind.workers.notifications._run_rag_query", return_value="ok"),
        ):
            await send_task_completed(
                bot_token="t",
                chat_id=100,
                product_query="test",
                qdrant_url="http://localhost:6333",
            )

        elapsed = time.monotonic() - start
        # With mocks it should be near-instant (well under 5 minutes)
        assert elapsed < 5.0

    def test_notification_module_importable(self):
        """Verify the module can be imported without side effects."""
        import reviewmind.workers.notifications

        assert hasattr(reviewmind.workers.notifications, "send_task_started")
        assert hasattr(reviewmind.workers.notifications, "send_task_completed")
        assert hasattr(reviewmind.workers.notifications, "send_task_failed")

    @pytest.mark.asyncio
    async def test_feedback_keyboard_buttons_present(self):
        """The feedback keyboard sent with analysis has the expected buttons."""
        from reviewmind.workers.notifications import send_task_completed

        mock_bot = AsyncMock()
        mock_bot.send_message = AsyncMock()
        mock_bot.session = AsyncMock()
        mock_bot.session.close = AsyncMock()

        mock_qdrant = AsyncMock()
        mock_qdrant.close = AsyncMock()

        with (
            patch("reviewmind.workers.notifications._create_bot", return_value=mock_bot),
            patch("reviewmind.workers.notifications.AsyncQdrantClient", return_value=mock_qdrant),
            patch("reviewmind.workers.notifications._run_rag_query", return_value="Analysis"),
        ):
            await send_task_completed(
                bot_token="t",
                chat_id=100,
                product_query="test",
                qdrant_url="http://localhost:6333",
            )

        markup = mock_bot.send_message.call_args.kwargs["reply_markup"]
        button_texts = [btn.text for row in markup.inline_keyboard for btn in row]
        assert "👍 Полезно" in button_texts
        assert "👎 Не то" in button_texts
        assert "📎 Источники" in button_texts
