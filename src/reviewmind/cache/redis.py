"""reviewmind/cache/redis.py — Redis session manager (mode, history, chunks, TTL).

Key format:
    session:{user_id}:mode    — user's current mode ("auto" | "links")
    session:{user_id}:history — last 5 messages as JSON list (FIFO)
    session:{user_id}:chunks  — stored chunk IDs for manual link mode

All keys expire after SESSION_TTL_SECONDS (default 1800 = 30 minutes).
Every write to a user's session refreshes all session keys' TTL.
"""

from __future__ import annotations

import json
from typing import Any

import structlog

logger = structlog.get_logger("reviewmind.cache.redis")

# ── Constants ─────────────────────────────────────────────────────────────────

DEFAULT_TTL_SECONDS: int = 1800  # 30 minutes
MAX_HISTORY_LENGTH: int = 5
VALID_MODES: frozenset[str] = frozenset({"auto", "links"})

# Key templates
_KEY_MODE = "session:{user_id}:mode"
_KEY_HISTORY = "session:{user_id}:history"
_KEY_CHUNKS = "session:{user_id}:chunks"


def _mode_key(user_id: int) -> str:
    return _KEY_MODE.format(user_id=user_id)


def _history_key(user_id: int) -> str:
    return _KEY_HISTORY.format(user_id=user_id)


def _chunks_key(user_id: int) -> str:
    return _KEY_CHUNKS.format(user_id=user_id)


# ── SessionManager ────────────────────────────────────────────────────────────


class SessionManager:
    """Async Redis session manager for per-user mode, history and chunk storage.

    Parameters
    ----------
    redis_client:
        An ``redis.asyncio.Redis`` (or compatible) client instance.
        Must have ``decode_responses=True`` enabled.
    ttl:
        Time-to-live for all session keys, in seconds.  Defaults to
        ``DEFAULT_TTL_SECONDS`` (1800 = 30 minutes).
    """

    def __init__(
        self,
        redis_client: Any,
        ttl: int | None = None,
    ) -> None:
        self._redis = redis_client
        self._ttl = ttl if ttl is not None else DEFAULT_TTL_SECONDS

    # ── Properties ────────────────────────────────────────────

    @property
    def redis(self) -> Any:
        """Underlying Redis client."""
        return self._redis

    @property
    def ttl(self) -> int:
        """Session TTL in seconds."""
        return self._ttl

    # ── Mode ──────────────────────────────────────────────────

    async def set_mode(self, user_id: int, mode: str) -> None:
        """Set the user's current mode (``"auto"`` or ``"links"``)."""
        if mode not in VALID_MODES:
            raise ValueError(f"Invalid mode {mode!r}. Expected one of {sorted(VALID_MODES)}.")
        key = _mode_key(user_id)
        await self._redis.set(key, mode, ex=self._ttl)
        logger.debug("session_mode_set", user_id=user_id, mode=mode)

    async def get_mode(self, user_id: int) -> str | None:
        """Return the user's current mode, or ``None`` if not set / expired."""
        key = _mode_key(user_id)
        value: str | None = await self._redis.get(key)
        return value

    # ── History ───────────────────────────────────────────────

    async def add_to_history(self, user_id: int, message: dict[str, Any]) -> None:
        """Append a message to the user's chat history (FIFO, max 5).

        Parameters
        ----------
        user_id:
            Telegram user ID.
        message:
            A dict with at least ``{"role": ..., "content": ...}``.
        """
        key = _history_key(user_id)
        serialized = json.dumps(message, ensure_ascii=False)

        pipe = self._redis.pipeline()
        pipe.rpush(key, serialized)
        # Trim to keep only the last MAX_HISTORY_LENGTH messages
        pipe.ltrim(key, -MAX_HISTORY_LENGTH, -1)
        pipe.expire(key, self._ttl)
        await pipe.execute()

        logger.debug("session_history_added", user_id=user_id, role=message.get("role"))

    async def get_history(self, user_id: int) -> list[dict[str, Any]]:
        """Return the last ``MAX_HISTORY_LENGTH`` messages for the user.

        Returns an empty list if no history exists or TTL has expired.
        """
        key = _history_key(user_id)
        raw_items: list[str] = await self._redis.lrange(key, 0, -1)
        result: list[dict[str, Any]] = []
        for item in raw_items:
            try:
                result.append(json.loads(item))
            except (json.JSONDecodeError, TypeError):
                logger.warning("session_history_parse_error", user_id=user_id, raw=item)
        return result

    async def clear_history(self, user_id: int) -> None:
        """Delete the user's chat history."""
        key = _history_key(user_id)
        await self._redis.delete(key)
        logger.debug("session_history_cleared", user_id=user_id)

    # ── Chunks ────────────────────────────────────────────────

    async def store_chunk_ids(self, user_id: int, chunk_ids: list[str]) -> None:
        """Store chunk IDs for the user's manual-link session.

        Replaces any previously stored chunk IDs.

        Parameters
        ----------
        user_id:
            Telegram user ID.
        chunk_ids:
            List of Qdrant point IDs (as strings) ingested for this session.
        """
        key = _chunks_key(user_id)
        serialized = json.dumps(chunk_ids, ensure_ascii=False)
        await self._redis.set(key, serialized, ex=self._ttl)
        logger.debug("session_chunks_stored", user_id=user_id, count=len(chunk_ids))

    async def get_chunk_ids(self, user_id: int) -> list[str]:
        """Return stored chunk IDs, or an empty list if none / expired."""
        key = _chunks_key(user_id)
        raw: str | None = await self._redis.get(key)
        if raw is None:
            return []
        try:
            ids = json.loads(raw)
            if isinstance(ids, list):
                return [str(i) for i in ids]
            return []
        except (json.JSONDecodeError, TypeError):
            logger.warning("session_chunks_parse_error", user_id=user_id, raw=raw)
            return []

    async def clear_chunks(self, user_id: int) -> None:
        """Delete the user's stored chunk IDs."""
        key = _chunks_key(user_id)
        await self._redis.delete(key)
        logger.debug("session_chunks_cleared", user_id=user_id)

    # ── Session lifecycle ─────────────────────────────────────

    async def refresh_ttl(self, user_id: int) -> None:
        """Refresh TTL for all session keys of the given user."""
        keys = [_mode_key(user_id), _history_key(user_id), _chunks_key(user_id)]
        pipe = self._redis.pipeline()
        for k in keys:
            pipe.expire(k, self._ttl)
        await pipe.execute()

    async def clear_session(self, user_id: int) -> None:
        """Delete all session data for the given user."""
        keys = [_mode_key(user_id), _history_key(user_id), _chunks_key(user_id)]
        await self._redis.delete(*keys)
        logger.info("session_cleared", user_id=user_id)

    async def session_exists(self, user_id: int) -> bool:
        """Check if any session key exists for the user."""
        keys = [_mode_key(user_id), _history_key(user_id), _chunks_key(user_id)]
        result = await self._redis.exists(*keys)
        return result > 0

    async def get_session_ttl(self, user_id: int) -> int:
        """Return remaining TTL (seconds) for the mode key, or -2 if expired/missing."""
        key = _mode_key(user_id)
        ttl_val: int = await self._redis.ttl(key)
        return ttl_val
