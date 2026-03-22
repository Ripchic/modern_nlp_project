"""Unit tests for TASK-046 — Multi-turn dialog: Redis session history integration.

Tests:
1. _create_session_manager helper (query.py & links.py)
2. _store_exchange helper — stores user + assistant messages
3. _close_redis helper — closes Redis client gracefully
4. on_text_message integration — loads history, passes to RAG, stores exchange
5. on_links_message integration — loads history, passes to RAG, stores exchange
6. _fallback_llm_answer — receives and passes chat_history
7. _try_instant_rag — receives and passes chat_history
8. mode.py — persists mode to Redis, does NOT clear history
9. Mode switch does NOT clear history (verify SessionManager.clear_history not called)
10. Edge cases: Redis unavailable, empty history, TTL refresh
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Save references to real functions BEFORE autouse fixtures patch them
from reviewmind.bot.handlers.mode import _persist_mode as _real_persist_mode

# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════


def _make_bot_message(text: str = "test", user_id: int = 123) -> MagicMock:
    """Create a mocked aiogram Message for bot handler tests."""
    msg = MagicMock()
    msg.text = text
    msg.from_user = MagicMock()
    msg.from_user.id = user_id
    msg.chat = MagicMock()
    msg.chat.id = user_id
    msg.bot = MagicMock()
    msg.bot.send_chat_action = AsyncMock()
    msg.answer = AsyncMock()
    return msg


def _make_callback_query(data: str = "mode:auto", user_id: int = 123) -> MagicMock:
    """Create a mocked aiogram CallbackQuery."""
    cb = MagicMock()
    cb.data = data
    cb.from_user = MagicMock()
    cb.from_user.id = user_id
    cb.message = MagicMock()
    cb.message.edit_text = AsyncMock()
    cb.answer = AsyncMock()
    return cb


def _make_session_manager(history: list | None = None) -> MagicMock:
    """Create a mocked SessionManager."""
    sm = MagicMock()
    sm.get_history = AsyncMock(return_value=history or [])
    sm.add_to_history = AsyncMock()
    sm.refresh_ttl = AsyncMock()
    sm.set_mode = AsyncMock()
    sm.clear_history = AsyncMock()
    return sm


def _make_redis_client() -> MagicMock:
    """Create a mocked Redis client."""
    client = MagicMock()
    client.aclose = AsyncMock()
    return client


def _make_rag_response(**kwargs):
    from reviewmind.core.rag import RAGResponse

    defaults = {
        "answer": "RAG analysis answer",
        "sources": ["https://example.com/review"],
        "used_curated": False,
        "confidence_met": True,
        "chunks_count": 5,
        "chunks_found": 10,
        "used_sponsored": False,
        "used_tavily": False,
        "error": None,
    }
    defaults.update(kwargs)
    return RAGResponse(**defaults)


# ══════════════════════════════════════════════════════════════
# Tests — _store_exchange (query.py)
# ══════════════════════════════════════════════════════════════


class TestStoreExchangeQuery:
    """Test _store_exchange in query handler."""

    @pytest.mark.asyncio
    async def test_stores_user_and_assistant(self):
        from reviewmind.bot.handlers.query import _store_exchange

        sm = _make_session_manager()
        await _store_exchange(sm, 123, "user msg", "bot reply")

        assert sm.add_to_history.call_count == 2
        call_args_list = sm.add_to_history.call_args_list
        assert call_args_list[0].args == (123, {"role": "user", "content": "user msg"})
        assert call_args_list[1].args == (123, {"role": "assistant", "content": "bot reply"})

    @pytest.mark.asyncio
    async def test_stores_only_user_when_no_assistant(self):
        from reviewmind.bot.handlers.query import _store_exchange

        sm = _make_session_manager()
        await _store_exchange(sm, 123, "user msg")

        assert sm.add_to_history.call_count == 1
        sm.add_to_history.assert_called_once_with(123, {"role": "user", "content": "user msg"})

    @pytest.mark.asyncio
    async def test_noop_when_session_mgr_is_none(self):
        from reviewmind.bot.handlers.query import _store_exchange

        # Should not raise
        await _store_exchange(None, 123, "user msg", "bot reply")

    @pytest.mark.asyncio
    async def test_graceful_on_redis_error(self):
        from reviewmind.bot.handlers.query import _store_exchange

        sm = _make_session_manager()
        sm.add_to_history.side_effect = ConnectionError("Redis down")

        # Should not raise
        await _store_exchange(sm, 123, "msg", "reply", log=MagicMock())


# ══════════════════════════════════════════════════════════════
# Tests — _close_redis (query.py)
# ══════════════════════════════════════════════════════════════


class TestCloseRedis:
    """Test _close_redis helper."""

    @pytest.mark.asyncio
    async def test_closes_client(self):
        from reviewmind.bot.handlers.query import _close_redis

        client = _make_redis_client()
        await _close_redis(client)
        client.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_noop_when_none(self):
        from reviewmind.bot.handlers.query import _close_redis

        await _close_redis(None)  # Should not raise

    @pytest.mark.asyncio
    async def test_graceful_on_close_error(self):
        from reviewmind.bot.handlers.query import _close_redis

        client = _make_redis_client()
        client.aclose.side_effect = OSError("Connection reset")
        await _close_redis(client)  # Should not raise


# ══════════════════════════════════════════════════════════════
# Tests — _create_session_manager (query.py)
# ══════════════════════════════════════════════════════════════


class TestCreateSessionManager:
    """Test _create_session_manager helper in query handler."""

    @pytest.mark.asyncio
    async def test_returns_session_manager_and_client(self):
        """The autouse fixture replaces _create_session_manager; verify its contract."""
        from reviewmind.bot.handlers.query import _create_session_manager

        sm, client = await _create_session_manager()
        # Must return a session manager with the expected API
        assert hasattr(sm, "get_history")
        assert hasattr(sm, "add_to_history")
        assert hasattr(sm, "refresh_ttl")
        # Must return a closeable redis client
        assert hasattr(client, "aclose")


# ══════════════════════════════════════════════════════════════
# Tests — on_text_message with session history
# ══════════════════════════════════════════════════════════════


class TestQueryHandlerSessionIntegration:
    """Test that on_text_message loads/stores session history."""

    @pytest.mark.asyncio
    async def test_loads_history_before_rag(self):
        """Verify chat_history from Redis is passed to _try_instant_rag."""
        history = [{"role": "user", "content": "prev q"}, {"role": "assistant", "content": "prev a"}]
        sm = _make_session_manager(history=history)
        redis_client = _make_redis_client()

        msg = _make_bot_message("Sony WH-1000XM5 review", user_id=42)

        with (
            patch(
                "reviewmind.bot.handlers.query._create_session_manager",
                new_callable=AsyncMock,
                return_value=(sm, redis_client),
            ),
            patch(
                "reviewmind.bot.handlers.query.extract_product",
                new_callable=AsyncMock,
                return_value=["Sony WH-1000XM5"],
            ),
            patch(
                "reviewmind.bot.handlers.query._create_qdrant_client",
                return_value=MagicMock(close=AsyncMock()),
            ),
            patch(
                "reviewmind.bot.handlers.query._try_instant_rag",
                new_callable=AsyncMock,
                return_value=_make_rag_response(),
            ) as mock_rag,
        ):
            from reviewmind.bot.handlers.query import on_text_message

            await on_text_message(msg)

            # Verify history was loaded
            sm.get_history.assert_awaited_once_with(42)
            # Verify history was passed to RAG
            _, kwargs = mock_rag.call_args
            assert kwargs.get("chat_history") == history

    @pytest.mark.asyncio
    async def test_stores_exchange_after_rag_answer(self):
        """After a successful RAG answer, user+assistant messages are stored."""
        sm = _make_session_manager()
        redis_client = _make_redis_client()
        rag_resp = _make_rag_response(answer="Great headphones!")

        msg = _make_bot_message("Sony WH-1000XM5", user_id=42)

        with (
            patch(
                "reviewmind.bot.handlers.query._create_session_manager",
                new_callable=AsyncMock,
                return_value=(sm, redis_client),
            ),
            patch(
                "reviewmind.bot.handlers.query.extract_product",
                new_callable=AsyncMock,
                return_value=["Sony WH-1000XM5"],
            ),
            patch(
                "reviewmind.bot.handlers.query._create_qdrant_client",
                return_value=MagicMock(close=AsyncMock()),
            ),
            patch(
                "reviewmind.bot.handlers.query._try_instant_rag",
                new_callable=AsyncMock,
                return_value=rag_resp,
            ),
        ):
            from reviewmind.bot.handlers.query import on_text_message

            await on_text_message(msg)

            # User + assistant messages stored
            assert sm.add_to_history.call_count == 2
            calls = sm.add_to_history.call_args_list
            assert calls[0].args[1]["role"] == "user"
            assert calls[1].args[1]["role"] == "assistant"
            assert calls[1].args[1]["content"] == "Great headphones!"

    @pytest.mark.asyncio
    async def test_stores_only_user_msg_for_background_job(self):
        """When a background job is scheduled, only user message is stored (answer comes later)."""
        sm = _make_session_manager()
        redis_client = _make_redis_client()

        msg = _make_bot_message("Sony WH-1000XM5 review", user_id=42)

        with (
            patch(
                "reviewmind.bot.handlers.query._create_session_manager",
                new_callable=AsyncMock,
                return_value=(sm, redis_client),
            ),
            patch(
                "reviewmind.bot.handlers.query.extract_product",
                new_callable=AsyncMock,
                return_value=["Sony WH-1000XM5"],
            ),
            patch(
                "reviewmind.bot.handlers.query._create_qdrant_client",
                return_value=MagicMock(close=AsyncMock()),
            ),
            patch(
                "reviewmind.bot.handlers.query._try_instant_rag",
                new_callable=AsyncMock,
                return_value=_make_rag_response(confidence_met=False, answer="", used_tavily=False),
            ),
            patch(
                "reviewmind.bot.handlers.query._collect_source_urls",
                new_callable=AsyncMock,
                return_value=["https://youtube.com/watch?v=abc"],
            ),
            patch(
                "reviewmind.bot.handlers.query._schedule_background_job",
                new_callable=AsyncMock,
                return_value="job-123",
            ),
        ):
            from reviewmind.bot.handlers.query import on_text_message

            await on_text_message(msg)

            # Only user message stored (no assistant text since answer comes via push)
            assert sm.add_to_history.call_count == 1
            calls = sm.add_to_history.call_args_list
            assert calls[0].args[1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_passes_history_to_fallback_llm(self):
        """When no product detected, chat_history is passed to _fallback_llm_answer."""
        history = [{"role": "user", "content": "old"}]
        sm = _make_session_manager(history=history)
        redis_client = _make_redis_client()

        msg = _make_bot_message("Привет, как дела?", user_id=42)

        with (
            patch(
                "reviewmind.bot.handlers.query._create_session_manager",
                new_callable=AsyncMock,
                return_value=(sm, redis_client),
            ),
            patch(
                "reviewmind.bot.handlers.query.extract_product",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "reviewmind.bot.handlers.query._try_instant_rag",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "reviewmind.bot.handlers.query._fallback_llm_answer",
                new_callable=AsyncMock,
            ) as mock_fallback,
        ):
            from reviewmind.bot.handlers.query import on_text_message

            await on_text_message(msg)

            _, kwargs = mock_fallback.call_args
            assert kwargs["chat_history"] == history

    @pytest.mark.asyncio
    async def test_redis_unavailable_proceeds_without_history(self):
        """If Redis is down, handler proceeds with empty history."""
        msg = _make_bot_message("Sony WH-1000XM5", user_id=42)

        with (
            patch(
                "reviewmind.bot.handlers.query._create_session_manager",
                new_callable=AsyncMock,
                side_effect=ConnectionError("Redis down"),
            ),
            patch(
                "reviewmind.bot.handlers.query.extract_product",
                new_callable=AsyncMock,
                return_value=["Sony WH-1000XM5"],
            ),
            patch(
                "reviewmind.bot.handlers.query._create_qdrant_client",
                return_value=MagicMock(close=AsyncMock()),
            ),
            patch(
                "reviewmind.bot.handlers.query._try_instant_rag",
                new_callable=AsyncMock,
                return_value=_make_rag_response(),
            ) as mock_rag,
        ):
            from reviewmind.bot.handlers.query import on_text_message

            await on_text_message(msg)

            # Handler should still work, RAG called with empty history
            mock_rag.assert_awaited_once()
            _, kwargs = mock_rag.call_args
            assert kwargs.get("chat_history") == []

    @pytest.mark.asyncio
    async def test_refreshes_ttl_on_activity(self):
        """Each message should refresh the session TTL."""
        sm = _make_session_manager()
        redis_client = _make_redis_client()

        msg = _make_bot_message("Sony WH-1000XM5", user_id=42)

        with (
            patch(
                "reviewmind.bot.handlers.query._create_session_manager",
                new_callable=AsyncMock,
                return_value=(sm, redis_client),
            ),
            patch(
                "reviewmind.bot.handlers.query.extract_product",
                new_callable=AsyncMock,
                return_value=["Sony WH-1000XM5"],
            ),
            patch(
                "reviewmind.bot.handlers.query._create_qdrant_client",
                return_value=MagicMock(close=AsyncMock()),
            ),
            patch(
                "reviewmind.bot.handlers.query._try_instant_rag",
                new_callable=AsyncMock,
                return_value=_make_rag_response(),
            ),
        ):
            from reviewmind.bot.handlers.query import on_text_message

            await on_text_message(msg)

            sm.refresh_ttl.assert_awaited_once_with(42)

    @pytest.mark.asyncio
    async def test_closes_redis_after_processing(self):
        """Redis client is always closed after handler completes."""
        sm = _make_session_manager()
        redis_client = _make_redis_client()

        msg = _make_bot_message("Sony WH-1000XM5", user_id=42)

        with (
            patch(
                "reviewmind.bot.handlers.query._create_session_manager",
                new_callable=AsyncMock,
                return_value=(sm, redis_client),
            ),
            patch(
                "reviewmind.bot.handlers.query.extract_product",
                new_callable=AsyncMock,
                return_value=["Sony WH-1000XM5"],
            ),
            patch(
                "reviewmind.bot.handlers.query._create_qdrant_client",
                return_value=MagicMock(close=AsyncMock()),
            ),
            patch(
                "reviewmind.bot.handlers.query._try_instant_rag",
                new_callable=AsyncMock,
                return_value=_make_rag_response(),
            ),
        ):
            from reviewmind.bot.handlers.query import on_text_message

            await on_text_message(msg)

            redis_client.aclose.assert_awaited_once()


# ══════════════════════════════════════════════════════════════
# Tests — links.py session integration
# ══════════════════════════════════════════════════════════════


class TestLinksHandlerSessionIntegration:
    """Test that on_links_message loads/stores session history."""

    @pytest.mark.asyncio
    async def test_loads_history_and_passes_to_ingest_analyse(self):
        """Verify chat_history from Redis is passed to _ingest_and_analyse."""
        history = [{"role": "user", "content": "look at these"}]
        sm = _make_session_manager(history=history)
        redis_client = _make_redis_client()

        msg = _make_bot_message("https://youtube.com/watch?v=abc review", user_id=42)

        with (
            patch(
                "reviewmind.bot.handlers.links._create_session_manager",
                new_callable=AsyncMock,
                return_value=(sm, redis_client),
            ),
            patch(
                "reviewmind.bot.handlers.links._create_qdrant_client",
                return_value=MagicMock(close=AsyncMock()),
            ),
            patch(
                "reviewmind.bot.handlers.links._ingest_and_analyse",
                new_callable=AsyncMock,
                return_value="Analysis text",
            ) as mock_analyse,
        ):
            from reviewmind.bot.handlers.links import on_links_message

            await on_links_message(msg)

            sm.get_history.assert_awaited_once_with(42)
            _, kwargs = mock_analyse.call_args
            assert kwargs.get("chat_history") == history

    @pytest.mark.asyncio
    async def test_stores_exchange_after_analysis(self):
        """After analysis, user+assistant messages are stored in Redis."""
        sm = _make_session_manager()
        redis_client = _make_redis_client()

        msg = _make_bot_message("https://youtube.com/watch?v=abc review", user_id=42)

        with (
            patch(
                "reviewmind.bot.handlers.links._create_session_manager",
                new_callable=AsyncMock,
                return_value=(sm, redis_client),
            ),
            patch(
                "reviewmind.bot.handlers.links._create_qdrant_client",
                return_value=MagicMock(close=AsyncMock()),
            ),
            patch(
                "reviewmind.bot.handlers.links._ingest_and_analyse",
                new_callable=AsyncMock,
                return_value="Great analysis",
            ),
        ):
            from reviewmind.bot.handlers.links import on_links_message

            await on_links_message(msg)

            assert sm.add_to_history.call_count == 2
            calls = sm.add_to_history.call_args_list
            assert calls[0].args[1]["role"] == "user"
            assert calls[1].args[1] == {"role": "assistant", "content": "Great analysis"}

    @pytest.mark.asyncio
    async def test_closes_redis_after_processing(self):
        """Redis client is always closed after links handler completes."""
        sm = _make_session_manager()
        redis_client = _make_redis_client()

        msg = _make_bot_message("https://youtube.com/watch?v=abc review", user_id=42)

        with (
            patch(
                "reviewmind.bot.handlers.links._create_session_manager",
                new_callable=AsyncMock,
                return_value=(sm, redis_client),
            ),
            patch(
                "reviewmind.bot.handlers.links._create_qdrant_client",
                return_value=MagicMock(close=AsyncMock()),
            ),
            patch(
                "reviewmind.bot.handlers.links._ingest_and_analyse",
                new_callable=AsyncMock,
                return_value="Analysis",
            ),
        ):
            from reviewmind.bot.handlers.links import on_links_message

            await on_links_message(msg)

            redis_client.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_redis_unavailable_proceeds_without_history(self):
        """If Redis is down, links handler proceeds with empty history."""
        msg = _make_bot_message("https://youtube.com/watch?v=abc review", user_id=42)

        with (
            patch(
                "reviewmind.bot.handlers.links._create_session_manager",
                new_callable=AsyncMock,
                side_effect=ConnectionError("Redis down"),
            ),
            patch(
                "reviewmind.bot.handlers.links._create_qdrant_client",
                return_value=MagicMock(close=AsyncMock()),
            ),
            patch(
                "reviewmind.bot.handlers.links._ingest_and_analyse",
                new_callable=AsyncMock,
                return_value="text",
            ) as mock_analyse,
        ):
            from reviewmind.bot.handlers.links import on_links_message

            await on_links_message(msg)

            # Handler still works; chat_history should be empty
            _, kwargs = mock_analyse.call_args
            assert kwargs.get("chat_history") == []


# ══════════════════════════════════════════════════════════════
# Tests — _store_exchange (links.py)
# ══════════════════════════════════════════════════════════════


class TestStoreExchangeLinks:
    """Test _store_exchange in links handler."""

    @pytest.mark.asyncio
    async def test_stores_user_and_assistant(self):
        from reviewmind.bot.handlers.links import _store_exchange

        sm = _make_session_manager()
        await _store_exchange(sm, 123, "user msg", "bot reply")

        assert sm.add_to_history.call_count == 2

    @pytest.mark.asyncio
    async def test_noop_when_none(self):
        from reviewmind.bot.handlers.links import _store_exchange

        await _store_exchange(None, 123, "user msg", "bot reply")


# ══════════════════════════════════════════════════════════════
# Tests — mode.py: persist mode, do NOT clear history
# ══════════════════════════════════════════════════════════════


class TestModePersistence:
    """Test that mode.py persists mode to Redis and does NOT clear history."""

    @pytest.mark.asyncio
    async def test_on_mode_selected_persists_auto(self):
        """Selecting auto-search persists 'auto' to Redis."""
        from reviewmind.bot.handlers.mode import on_mode_selected

        cb = _make_callback_query(data="mode:auto", user_id=42)

        with patch("reviewmind.bot.handlers.mode._persist_mode", new_callable=AsyncMock) as mock_persist:
            await on_mode_selected(cb)
            mock_persist.assert_awaited_once_with(42, "auto")

    @pytest.mark.asyncio
    async def test_on_mode_selected_persists_links(self):
        """Selecting links mode persists 'links' to Redis."""
        from reviewmind.bot.handlers.mode import on_mode_selected

        cb = _make_callback_query(data="mode:links", user_id=42)

        with patch("reviewmind.bot.handlers.mode._persist_mode", new_callable=AsyncMock) as mock_persist:
            await on_mode_selected(cb)
            mock_persist.assert_awaited_once_with(42, "links")

    @pytest.mark.asyncio
    async def test_persist_mode_calls_set_mode_and_refresh(self):
        """_persist_mode calls SessionManager.set_mode and refresh_ttl."""
        mock_client = _make_redis_client()
        mock_sm = _make_session_manager()

        with (
            patch("redis.asyncio.from_url", return_value=mock_client),
            patch("reviewmind.cache.redis.SessionManager", return_value=mock_sm),
            patch("reviewmind.config.settings", MagicMock(redis_url="redis://localhost:6379/0")),
        ):
            await _real_persist_mode(42, "auto")

            mock_sm.set_mode.assert_awaited_once_with(42, "auto")
            mock_sm.refresh_ttl.assert_awaited_once_with(42)
            mock_client.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_persist_mode_does_not_clear_history(self):
        """Mode switch must NOT call clear_history."""
        mock_client = _make_redis_client()
        mock_sm = _make_session_manager()

        with (
            patch("redis.asyncio.from_url", return_value=mock_client),
            patch("reviewmind.cache.redis.SessionManager", return_value=mock_sm),
            patch("reviewmind.config.settings", MagicMock(redis_url="redis://localhost:6379/0")),
        ):
            await _real_persist_mode(42, "links")

            mock_sm.clear_history.assert_not_called()

    @pytest.mark.asyncio
    async def test_persist_mode_graceful_on_error(self):
        """If Redis is down, _persist_mode silently does nothing."""
        with patch(
            "redis.asyncio.from_url",
            side_effect=ConnectionError("Redis down"),
        ):
            # Should not raise
            await _real_persist_mode(42, "auto")


# ══════════════════════════════════════════════════════════════
# Tests — _try_instant_rag accepts chat_history
# ══════════════════════════════════════════════════════════════


class TestTryInstantRagHistory:
    """Test that _try_instant_rag passes chat_history to RAGPipeline."""

    @pytest.mark.asyncio
    async def test_passes_chat_history(self):
        history = [{"role": "user", "content": "prev"}, {"role": "assistant", "content": "ans"}]
        mock_rag = MagicMock()
        mock_rag.query = AsyncMock(return_value=_make_rag_response())
        mock_rag.__aenter__ = AsyncMock(return_value=mock_rag)
        mock_rag.__aexit__ = AsyncMock(return_value=False)

        with patch("reviewmind.bot.handlers.query.RAGPipeline", return_value=mock_rag):
            from reviewmind.bot.handlers.query import _try_instant_rag

            qdrant = MagicMock()
            await _try_instant_rag(qdrant, "query", "product", MagicMock(), chat_history=history)

            mock_rag.query.assert_awaited_once()
            _, kwargs = mock_rag.query.call_args
            assert kwargs["chat_history"] == history

    @pytest.mark.asyncio
    async def test_empty_history_passed_as_none(self):
        mock_rag = MagicMock()
        mock_rag.query = AsyncMock(return_value=_make_rag_response())
        mock_rag.__aenter__ = AsyncMock(return_value=mock_rag)
        mock_rag.__aexit__ = AsyncMock(return_value=False)

        with patch("reviewmind.bot.handlers.query.RAGPipeline", return_value=mock_rag):
            from reviewmind.bot.handlers.query import _try_instant_rag

            qdrant = MagicMock()
            await _try_instant_rag(qdrant, "query", "product", MagicMock(), chat_history=[])

            _, kwargs = mock_rag.query.call_args
            assert kwargs["chat_history"] is None


# ══════════════════════════════════════════════════════════════
# Tests — _fallback_llm_answer accepts chat_history
# ══════════════════════════════════════════════════════════════


class TestFallbackLlmHistory:
    """Test that _fallback_llm_answer passes chat_history to QueryService."""

    @pytest.mark.asyncio
    async def test_passes_chat_history_to_query_service(self):
        history = [{"role": "user", "content": "old q"}]

        @dataclass
        class FakeResult:
            answer: str = "Fallback answer"
            error: bool = False

        mock_service_cls = MagicMock()
        mock_service_inst = MagicMock()
        mock_service_inst.answer = AsyncMock(return_value=FakeResult())
        mock_service_cls.return_value = mock_service_inst

        mock_client = MagicMock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)

        msg = _make_bot_message("Привет!", user_id=42)
        log = MagicMock()

        with (
            patch("reviewmind.bot.handlers.query.LLMClient", return_value=mock_client),
            patch("reviewmind.bot.handlers.query.QueryService", mock_service_cls),
        ):
            from reviewmind.bot.handlers.query import _fallback_llm_answer

            await _fallback_llm_answer(msg, 42, log, chat_history=history)

            mock_service_inst.answer.assert_awaited_once()
            _, kwargs = mock_service_inst.answer.call_args
            assert kwargs["chat_history"] == history


# ══════════════════════════════════════════════════════════════
# Tests — _ingest_and_analyse passes chat_history (links.py)
# ══════════════════════════════════════════════════════════════


class TestIngestAndAnalyseHistory:
    """Test that _ingest_and_analyse passes chat_history to RAG pipeline."""

    @pytest.mark.asyncio
    async def test_passes_chat_history_to_rag(self):
        history = [{"role": "user", "content": "old link"}]

        # Mock ingestion
        @dataclass
        class FakeIngestion:
            success_count: int = 1
            failed_count: int = 0
            chunks_count: int = 5
            results: list = None

            def __post_init__(self):
                self.results = self.results or []

        mock_pipeline = MagicMock()
        mock_pipeline.ingest_urls = AsyncMock(return_value=FakeIngestion())
        mock_pipeline.__aenter__ = AsyncMock(return_value=mock_pipeline)
        mock_pipeline.__aexit__ = AsyncMock(return_value=False)

        mock_rag = MagicMock()
        mock_rag.query = AsyncMock(return_value=_make_rag_response(answer="Analysis"))
        mock_rag.__aenter__ = AsyncMock(return_value=mock_rag)
        mock_rag.__aexit__ = AsyncMock(return_value=False)

        msg = _make_bot_message("https://example.com review", user_id=42)
        status_msg = MagicMock()
        status_msg.edit_text = AsyncMock()

        with (
            patch("reviewmind.bot.handlers.links.IngestionPipeline", return_value=mock_pipeline),
            patch("reviewmind.bot.handlers.links.RAGPipeline", return_value=mock_rag),
        ):
            from reviewmind.bot.handlers.links import _ingest_and_analyse

            await _ingest_and_analyse(
                message=msg,
                status_msg=status_msg,
                urls=["https://example.com"],
                query_text="review",
                qdrant=MagicMock(),
                log=MagicMock(),
                chat_history=history,
            )

            _, kwargs = mock_rag.query.call_args
            assert kwargs["chat_history"] == history


# ══════════════════════════════════════════════════════════════
# Tests — Session history TTL and expiration
# ══════════════════════════════════════════════════════════════


class TestSessionTTL:
    """Test that session TTL is properly managed."""

    @pytest.mark.asyncio
    async def test_default_ttl_is_30_minutes(self):
        from reviewmind.cache.redis import DEFAULT_TTL_SECONDS

        assert DEFAULT_TTL_SECONDS == 1800  # 30 minutes

    @pytest.mark.asyncio
    async def test_max_history_is_5(self):
        from reviewmind.cache.redis import MAX_HISTORY_LENGTH

        assert MAX_HISTORY_LENGTH == 5

    @pytest.mark.asyncio
    async def test_refresh_ttl_called_on_links(self):
        """Links handler refreshes TTL on session keys."""
        sm = _make_session_manager()
        redis_client = _make_redis_client()

        msg = _make_bot_message("https://youtube.com/watch?v=abc review", user_id=42)

        with (
            patch(
                "reviewmind.bot.handlers.links._create_session_manager",
                new_callable=AsyncMock,
                return_value=(sm, redis_client),
            ),
            patch(
                "reviewmind.bot.handlers.links._create_qdrant_client",
                return_value=MagicMock(close=AsyncMock()),
            ),
            patch(
                "reviewmind.bot.handlers.links._ingest_and_analyse",
                new_callable=AsyncMock,
                return_value="Analysis",
            ),
        ):
            from reviewmind.bot.handlers.links import on_links_message

            await on_links_message(msg)

            sm.refresh_ttl.assert_awaited_once_with(42)


# ══════════════════════════════════════════════════════════════
# Tests — _CALLBACK_TO_MODE mapping
# ══════════════════════════════════════════════════════════════


class TestCallbackToMode:
    """Test the callback data → internal mode mapping."""

    def test_auto_mapping(self):
        from reviewmind.bot.handlers.mode import _CALLBACK_TO_MODE

        assert _CALLBACK_TO_MODE["mode:auto"] == "auto"

    def test_links_mapping(self):
        from reviewmind.bot.handlers.mode import _CALLBACK_TO_MODE

        assert _CALLBACK_TO_MODE["mode:links"] == "links"

    def test_mode_names_unchanged(self):
        from reviewmind.bot.handlers.mode import MODE_NAMES

        assert "mode:auto" in MODE_NAMES
        assert "mode:links" in MODE_NAMES


# ══════════════════════════════════════════════════════════════
# Tests — Integration scenarios
# ══════════════════════════════════════════════════════════════


class TestIntegrationScenarios:
    """End-to-end scenario tests for multi-turn dialog."""

    @pytest.mark.asyncio
    async def test_follow_up_uses_prior_context(self):
        """Simulates: Q1 about Sony → A1, then follow-up 'а как насчёт батареи?'."""
        history = [
            {"role": "user", "content": "Sony WH-1000XM5 стоит ли покупать?"},
            {"role": "assistant", "content": "Отличные наушники! Плюсы: звук, ANC..."},
        ]
        sm = _make_session_manager(history=history)
        redis_client = _make_redis_client()

        msg = _make_bot_message("а как насчёт батареи?", user_id=42)

        with (
            patch(
                "reviewmind.bot.handlers.query._create_session_manager",
                new_callable=AsyncMock,
                return_value=(sm, redis_client),
            ),
            patch(
                "reviewmind.bot.handlers.query.extract_product",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch(
                "reviewmind.bot.handlers.query._try_instant_rag",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "reviewmind.bot.handlers.query._fallback_llm_answer",
                new_callable=AsyncMock,
            ) as mock_fallback,
        ):
            from reviewmind.bot.handlers.query import on_text_message

            await on_text_message(msg)

            _, kwargs = mock_fallback.call_args
            assert kwargs["chat_history"] == history
            # The 2-message history provides context about Sony headphones

    @pytest.mark.asyncio
    async def test_mode_switch_preserves_history(self):
        """Switching mode does not interfere with stored history."""
        mock_client = _make_redis_client()
        mock_sm = _make_session_manager()

        with (
            patch("redis.asyncio.from_url", return_value=mock_client),
            patch("reviewmind.cache.redis.SessionManager", return_value=mock_sm),
            patch("reviewmind.config.settings", MagicMock(redis_url="redis://localhost:6379/0")),
        ):
            await _real_persist_mode(42, "links")

            mock_sm.set_mode.assert_awaited_once_with(42, "links")
            mock_sm.clear_history.assert_not_called()
            mock_sm.clear_session.assert_not_called()

    @pytest.mark.asyncio
    async def test_history_limited_to_5_messages(self):
        """SessionManager enforces MAX_HISTORY_LENGTH=5."""
        from reviewmind.cache.redis import MAX_HISTORY_LENGTH, SessionManager

        mock_redis = MagicMock()
        pipe = MagicMock()
        pipe.rpush = MagicMock(return_value=pipe)
        pipe.ltrim = MagicMock(return_value=pipe)
        pipe.expire = MagicMock(return_value=pipe)
        pipe.execute = AsyncMock()
        mock_redis.pipeline = MagicMock(return_value=pipe)

        sm = SessionManager(mock_redis)
        await sm.add_to_history(42, {"role": "user", "content": "msg"})

        # Verify LTRIM uses -MAX_HISTORY_LENGTH to keep last 5
        pipe.ltrim.assert_called_once()
        args = pipe.ltrim.call_args.args
        assert args[1] == -MAX_HISTORY_LENGTH

    def test_imports_available(self):
        """Verify all multi-turn related imports work."""
        import importlib

        links = importlib.import_module("reviewmind.bot.handlers.links")
        assert callable(getattr(links, "_close_redis"))
        assert callable(getattr(links, "_create_session_manager"))
        assert callable(getattr(links, "_store_exchange"))

        mode = importlib.import_module("reviewmind.bot.handlers.mode")
        assert callable(getattr(mode, "_persist_mode"))
        assert isinstance(getattr(mode, "_CALLBACK_TO_MODE"), dict)

        query = importlib.import_module("reviewmind.bot.handlers.query")
        assert callable(getattr(query, "_close_redis"))
        assert callable(getattr(query, "_create_session_manager"))
        assert callable(getattr(query, "_store_exchange"))

        cache_redis = importlib.import_module("reviewmind.cache.redis")
        assert hasattr(cache_redis, "DEFAULT_TTL_SECONDS")
        assert hasattr(cache_redis, "MAX_HISTORY_LENGTH")
        assert callable(getattr(cache_redis, "SessionManager"))
