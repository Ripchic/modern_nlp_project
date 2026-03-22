"""Unit tests for reviewmind.cache.redis — SessionManager."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from reviewmind.cache.redis import (
    DEFAULT_TTL_SECONDS,
    MAX_HISTORY_LENGTH,
    VALID_MODES,
    SessionManager,
    _chunks_key,
    _history_key,
    _mode_key,
)

# ── Helpers ──────────────────────────────────────────────────


def _make_redis_mock() -> AsyncMock:
    """Create a mock redis.asyncio.Redis with decode_responses=True behavior."""
    redis = AsyncMock()
    # pipeline() is synchronous in redis.asyncio — returns Pipeline object.
    # Pipeline methods (rpush, ltrim, expire) are sync (queue commands),
    # only execute() is async.
    pipe = MagicMock()
    pipe.execute = AsyncMock(return_value=[])
    redis.pipeline = MagicMock(return_value=pipe)
    return redis


def _make_session(ttl: int | None = None) -> tuple[SessionManager, AsyncMock]:
    """Create a SessionManager with a mocked Redis client."""
    redis = _make_redis_mock()
    sm = SessionManager(redis, ttl=ttl)
    return sm, redis


# ── Constants ────────────────────────────────────────────────


class TestConstants:
    """Test module-level constants."""

    def test_default_ttl(self):
        assert DEFAULT_TTL_SECONDS == 1800

    def test_max_history_length(self):
        assert MAX_HISTORY_LENGTH == 5

    def test_valid_modes(self):
        assert VALID_MODES == frozenset({"auto", "links"})

    def test_valid_modes_is_frozenset(self):
        assert isinstance(VALID_MODES, frozenset)


# ── Key helpers ──────────────────────────────────────────────


class TestKeyHelpers:
    """Test _mode_key, _history_key, _chunks_key."""

    def test_mode_key(self):
        assert _mode_key(123) == "session:123:mode"

    def test_history_key(self):
        assert _history_key(456) == "session:456:history"

    def test_chunks_key(self):
        assert _chunks_key(789) == "session:789:chunks"

    def test_mode_key_large_id(self):
        assert _mode_key(9999999999) == "session:9999999999:mode"

    def test_history_key_zero(self):
        assert _history_key(0) == "session:0:history"


# ── SessionManager Init ─────────────────────────────────────


class TestSessionManagerInit:
    """Test SessionManager construction."""

    def test_creates_with_defaults(self):
        redis = AsyncMock()
        sm = SessionManager(redis)
        assert sm.redis is redis
        assert sm.ttl == DEFAULT_TTL_SECONDS

    def test_custom_ttl(self):
        redis = AsyncMock()
        sm = SessionManager(redis, ttl=60)
        assert sm.ttl == 60

    def test_ttl_none_uses_default(self):
        redis = AsyncMock()
        sm = SessionManager(redis, ttl=None)
        assert sm.ttl == DEFAULT_TTL_SECONDS

    def test_redis_property(self):
        redis = AsyncMock()
        sm = SessionManager(redis)
        assert sm.redis is redis


# ── Mode operations ──────────────────────────────────────────


class TestSetMode:
    """Test SessionManager.set_mode."""

    @pytest.mark.asyncio
    async def test_set_mode_auto(self):
        sm, redis = _make_session()
        await sm.set_mode(123, "auto")
        redis.set.assert_awaited_once_with("session:123:mode", "auto", ex=1800)

    @pytest.mark.asyncio
    async def test_set_mode_links(self):
        sm, redis = _make_session()
        await sm.set_mode(123, "links")
        redis.set.assert_awaited_once_with("session:123:mode", "links", ex=1800)

    @pytest.mark.asyncio
    async def test_set_mode_custom_ttl(self):
        sm, redis = _make_session(ttl=60)
        await sm.set_mode(123, "auto")
        redis.set.assert_awaited_once_with("session:123:mode", "auto", ex=60)

    @pytest.mark.asyncio
    async def test_set_mode_invalid_raises(self):
        sm, _ = _make_session()
        with pytest.raises(ValueError, match="Invalid mode"):
            await sm.set_mode(123, "invalid")

    @pytest.mark.asyncio
    async def test_set_mode_empty_raises(self):
        sm, _ = _make_session()
        with pytest.raises(ValueError, match="Invalid mode"):
            await sm.set_mode(123, "")


class TestGetMode:
    """Test SessionManager.get_mode."""

    @pytest.mark.asyncio
    async def test_get_mode_returns_value(self):
        sm, redis = _make_session()
        redis.get.return_value = "auto"
        result = await sm.get_mode(123)
        assert result == "auto"
        redis.get.assert_awaited_once_with("session:123:mode")

    @pytest.mark.asyncio
    async def test_get_mode_returns_none_if_missing(self):
        sm, redis = _make_session()
        redis.get.return_value = None
        result = await sm.get_mode(123)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_mode_returns_links(self):
        sm, redis = _make_session()
        redis.get.return_value = "links"
        result = await sm.get_mode(123)
        assert result == "links"


# ── History operations ───────────────────────────────────────


class TestAddToHistory:
    """Test SessionManager.add_to_history."""

    @pytest.mark.asyncio
    async def test_add_message(self):
        sm, redis = _make_session()
        pipe = redis.pipeline.return_value
        msg = {"role": "user", "content": "Hello"}
        await sm.add_to_history(123, msg)

        redis.pipeline.assert_called_once()
        pipe.rpush.assert_called_once_with(
            "session:123:history",
            json.dumps(msg, ensure_ascii=False),
        )
        pipe.ltrim.assert_called_once_with("session:123:history", -MAX_HISTORY_LENGTH, -1)
        pipe.expire.assert_called_once_with("session:123:history", 1800)
        pipe.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_add_message_custom_ttl(self):
        sm, redis = _make_session(ttl=300)
        pipe = redis.pipeline.return_value
        msg = {"role": "assistant", "content": "Hi!"}
        await sm.add_to_history(123, msg)
        pipe.expire.assert_called_once_with("session:123:history", 300)

    @pytest.mark.asyncio
    async def test_add_message_unicode(self):
        sm, redis = _make_session()
        pipe = redis.pipeline.return_value
        msg = {"role": "user", "content": "Привет, как дела?"}
        await sm.add_to_history(123, msg)
        expected_json = json.dumps(msg, ensure_ascii=False)
        pipe.rpush.assert_called_once_with("session:123:history", expected_json)
        # Verify Cyrillic is not escaped
        assert "Привет" in expected_json

    @pytest.mark.asyncio
    async def test_add_message_trims_to_max(self):
        sm, redis = _make_session()
        pipe = redis.pipeline.return_value
        msg = {"role": "user", "content": "Test"}
        await sm.add_to_history(123, msg)
        pipe.ltrim.assert_called_once_with("session:123:history", -5, -1)


class TestGetHistory:
    """Test SessionManager.get_history."""

    @pytest.mark.asyncio
    async def test_get_empty_history(self):
        sm, redis = _make_session()
        redis.lrange.return_value = []
        result = await sm.get_history(123)
        assert result == []
        redis.lrange.assert_awaited_once_with("session:123:history", 0, -1)

    @pytest.mark.asyncio
    async def test_get_single_message(self):
        sm, redis = _make_session()
        msg = {"role": "user", "content": "Hello"}
        redis.lrange.return_value = [json.dumps(msg)]
        result = await sm.get_history(123)
        assert result == [msg]

    @pytest.mark.asyncio
    async def test_get_multiple_messages(self):
        sm, redis = _make_session()
        msgs = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1"},
            {"role": "user", "content": "Q2"},
        ]
        redis.lrange.return_value = [json.dumps(m) for m in msgs]
        result = await sm.get_history(123)
        assert result == msgs

    @pytest.mark.asyncio
    async def test_get_history_preserves_order(self):
        sm, redis = _make_session()
        msgs = [{"role": "user", "content": f"msg{i}"} for i in range(5)]
        redis.lrange.return_value = [json.dumps(m) for m in msgs]
        result = await sm.get_history(123)
        assert len(result) == 5
        assert result[0]["content"] == "msg0"
        assert result[4]["content"] == "msg4"

    @pytest.mark.asyncio
    async def test_get_history_skips_corrupted_json(self):
        sm, redis = _make_session()
        redis.lrange.return_value = [
            json.dumps({"role": "user", "content": "OK"}),
            "not-valid-json{{",
            json.dumps({"role": "assistant", "content": "Hi"}),
        ]
        result = await sm.get_history(123)
        assert len(result) == 2
        assert result[0]["content"] == "OK"
        assert result[1]["content"] == "Hi"

    @pytest.mark.asyncio
    async def test_get_history_unicode(self):
        sm, redis = _make_session()
        msg = {"role": "user", "content": "Стоит ли покупать iPhone?"}
        redis.lrange.return_value = [json.dumps(msg, ensure_ascii=False)]
        result = await sm.get_history(123)
        assert result[0]["content"] == "Стоит ли покупать iPhone?"


class TestClearHistory:
    """Test SessionManager.clear_history."""

    @pytest.mark.asyncio
    async def test_clear_history(self):
        sm, redis = _make_session()
        await sm.clear_history(123)
        redis.delete.assert_awaited_once_with("session:123:history")


# ── Chunks operations ────────────────────────────────────────


class TestStoreChunkIds:
    """Test SessionManager.store_chunk_ids."""

    @pytest.mark.asyncio
    async def test_store_chunk_ids(self):
        sm, redis = _make_session()
        ids = ["abc-123", "def-456", "ghi-789"]
        await sm.store_chunk_ids(123, ids)
        redis.set.assert_awaited_once_with(
            "session:123:chunks",
            json.dumps(ids, ensure_ascii=False),
            ex=1800,
        )

    @pytest.mark.asyncio
    async def test_store_empty_chunk_ids(self):
        sm, redis = _make_session()
        await sm.store_chunk_ids(123, [])
        redis.set.assert_awaited_once_with(
            "session:123:chunks",
            "[]",
            ex=1800,
        )

    @pytest.mark.asyncio
    async def test_store_chunk_ids_custom_ttl(self):
        sm, redis = _make_session(ttl=120)
        await sm.store_chunk_ids(123, ["id1"])
        redis.set.assert_awaited_once_with(
            "session:123:chunks",
            json.dumps(["id1"]),
            ex=120,
        )


class TestGetChunkIds:
    """Test SessionManager.get_chunk_ids."""

    @pytest.mark.asyncio
    async def test_get_chunk_ids(self):
        sm, redis = _make_session()
        ids = ["abc-123", "def-456"]
        redis.get.return_value = json.dumps(ids)
        result = await sm.get_chunk_ids(123)
        assert result == ids
        redis.get.assert_awaited_once_with("session:123:chunks")

    @pytest.mark.asyncio
    async def test_get_chunk_ids_none(self):
        sm, redis = _make_session()
        redis.get.return_value = None
        result = await sm.get_chunk_ids(123)
        assert result == []

    @pytest.mark.asyncio
    async def test_get_chunk_ids_empty_list(self):
        sm, redis = _make_session()
        redis.get.return_value = "[]"
        result = await sm.get_chunk_ids(123)
        assert result == []

    @pytest.mark.asyncio
    async def test_get_chunk_ids_corrupted_json(self):
        sm, redis = _make_session()
        redis.get.return_value = "not-json{{"
        result = await sm.get_chunk_ids(123)
        assert result == []

    @pytest.mark.asyncio
    async def test_get_chunk_ids_non_list_json(self):
        sm, redis = _make_session()
        redis.get.return_value = '{"key": "value"}'
        result = await sm.get_chunk_ids(123)
        assert result == []

    @pytest.mark.asyncio
    async def test_get_chunk_ids_coerces_to_str(self):
        sm, redis = _make_session()
        # Integers stored — should be coerced to strings
        redis.get.return_value = json.dumps([1, 2, 3])
        result = await sm.get_chunk_ids(123)
        assert result == ["1", "2", "3"]


class TestClearChunks:
    """Test SessionManager.clear_chunks."""

    @pytest.mark.asyncio
    async def test_clear_chunks(self):
        sm, redis = _make_session()
        await sm.clear_chunks(123)
        redis.delete.assert_awaited_once_with("session:123:chunks")


# ── Session lifecycle ────────────────────────────────────────


class TestRefreshTTL:
    """Test SessionManager.refresh_ttl."""

    @pytest.mark.asyncio
    async def test_refresh_ttl(self):
        sm, redis = _make_session()
        pipe = redis.pipeline.return_value
        await sm.refresh_ttl(123)

        redis.pipeline.assert_called_once()
        assert pipe.expire.call_count == 3
        pipe.expire.assert_any_call("session:123:mode", 1800)
        pipe.expire.assert_any_call("session:123:history", 1800)
        pipe.expire.assert_any_call("session:123:chunks", 1800)
        pipe.execute.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_refresh_ttl_custom(self):
        sm, redis = _make_session(ttl=90)
        pipe = redis.pipeline.return_value
        await sm.refresh_ttl(456)

        pipe.expire.assert_any_call("session:456:mode", 90)
        pipe.expire.assert_any_call("session:456:history", 90)
        pipe.expire.assert_any_call("session:456:chunks", 90)


class TestClearSession:
    """Test SessionManager.clear_session."""

    @pytest.mark.asyncio
    async def test_clear_session(self):
        sm, redis = _make_session()
        await sm.clear_session(123)
        redis.delete.assert_awaited_once_with(
            "session:123:mode",
            "session:123:history",
            "session:123:chunks",
        )


class TestSessionExists:
    """Test SessionManager.session_exists."""

    @pytest.mark.asyncio
    async def test_session_exists_true(self):
        sm, redis = _make_session()
        redis.exists.return_value = 2  # 2 keys exist
        result = await sm.session_exists(123)
        assert result is True
        redis.exists.assert_awaited_once_with(
            "session:123:mode",
            "session:123:history",
            "session:123:chunks",
        )

    @pytest.mark.asyncio
    async def test_session_exists_false(self):
        sm, redis = _make_session()
        redis.exists.return_value = 0
        result = await sm.session_exists(123)
        assert result is False

    @pytest.mark.asyncio
    async def test_session_exists_partial(self):
        sm, redis = _make_session()
        redis.exists.return_value = 1  # only 1 key exists
        result = await sm.session_exists(123)
        assert result is True


class TestGetSessionTTL:
    """Test SessionManager.get_session_ttl."""

    @pytest.mark.asyncio
    async def test_get_session_ttl_active(self):
        sm, redis = _make_session()
        redis.ttl.return_value = 1500
        result = await sm.get_session_ttl(123)
        assert result == 1500
        redis.ttl.assert_awaited_once_with("session:123:mode")

    @pytest.mark.asyncio
    async def test_get_session_ttl_expired(self):
        sm, redis = _make_session()
        redis.ttl.return_value = -2
        result = await sm.get_session_ttl(123)
        assert result == -2

    @pytest.mark.asyncio
    async def test_get_session_ttl_no_expiry(self):
        sm, redis = _make_session()
        redis.ttl.return_value = -1  # Key exists but no TTL
        result = await sm.get_session_ttl(123)
        assert result == -1


# ── Cache __init__.py exports ────────────────────────────────


class TestCacheExports:
    """Test reviewmind.cache package exports."""

    def test_session_manager_exported(self):
        from reviewmind.cache import SessionManager as SM

        assert SM is SessionManager

    def test_constants_exported(self):
        from reviewmind.cache import DEFAULT_TTL_SECONDS as TTL
        from reviewmind.cache import MAX_HISTORY_LENGTH as MHL
        from reviewmind.cache import VALID_MODES as VM

        assert TTL == 1800
        assert MHL == 5
        assert VM == frozenset({"auto", "links"})


# ── Integration-style tests (still mocked) ───────────────────


class TestIntegrationScenarios:
    """Test realistic usage patterns with mocked Redis."""

    @pytest.mark.asyncio
    async def test_full_session_lifecycle(self):
        """Simulate: set mode → add history → store chunks → clear session."""
        sm, redis = _make_session()

        # 1. Set mode
        await sm.set_mode(100, "auto")
        redis.set.assert_awaited_with("session:100:mode", "auto", ex=1800)

        # 2. Add history messages
        await sm.add_to_history(100, {"role": "user", "content": "Q1"})
        await sm.add_to_history(100, {"role": "assistant", "content": "A1"})

        # 3. Store chunks
        await sm.store_chunk_ids(100, ["chunk-1", "chunk-2"])

        # 4. Clear session
        await sm.clear_session(100)
        redis.delete.assert_awaited_with(
            "session:100:mode",
            "session:100:history",
            "session:100:chunks",
        )

    @pytest.mark.asyncio
    async def test_mode_switch_preserves_nothing_in_redis_call(self):
        """Switching mode only touches mode key, not history or chunks."""
        sm, redis = _make_session()

        await sm.set_mode(200, "auto")
        first_call = redis.set.call_args
        assert first_call[0] == ("session:200:mode", "auto")

        redis.set.reset_mock()
        await sm.set_mode(200, "links")
        second_call = redis.set.call_args
        assert second_call[0] == ("session:200:mode", "links")
        # delete was NOT called — history is preserved
        redis.delete.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_history_fifo_max_5(self):
        """Adding 7 messages, ltrim ensures only last 5 are kept."""
        sm, redis = _make_session()
        pipe = redis.pipeline.return_value

        for i in range(7):
            await sm.add_to_history(300, {"role": "user", "content": f"msg{i}"})

        # Each call trims to last 5
        assert pipe.ltrim.call_count == 7
        for c in pipe.ltrim.call_args_list:
            assert c[0] == ("session:300:history", -5, -1)

    @pytest.mark.asyncio
    async def test_different_users_independent_keys(self):
        """Two different user IDs produce different Redis keys."""
        sm, redis = _make_session()

        await sm.set_mode(100, "auto")
        await sm.set_mode(200, "links")

        calls = redis.set.call_args_list
        assert calls[0][0][0] == "session:100:mode"
        assert calls[1][0][0] == "session:200:mode"

    @pytest.mark.asyncio
    async def test_get_mode_after_expiry_returns_none(self):
        """After TTL expiry, get_mode returns None."""
        sm, redis = _make_session()
        redis.get.return_value = None  # simulates expired key
        result = await sm.get_mode(999)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_chunk_ids_after_store(self):
        """Store then retrieve chunk IDs."""
        sm, redis = _make_session()
        ids = ["a", "b", "c"]

        # Store
        await sm.store_chunk_ids(400, ids)

        # Mock get to return what was stored
        redis.get.return_value = json.dumps(ids)
        result = await sm.get_chunk_ids(400)
        assert result == ids
