"""Shared fixtures for bot handler tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _make_noop_session_manager():
    """Return a (session_manager, redis_client) pair that does nothing."""
    sm = MagicMock()
    sm.get_history = AsyncMock(return_value=[])
    sm.add_to_history = AsyncMock()
    sm.refresh_ttl = AsyncMock()
    sm.set_mode = AsyncMock()
    sm.clear_history = AsyncMock()

    client = MagicMock()
    client.aclose = AsyncMock()
    return sm, client


@pytest.fixture(autouse=True)
def _bypass_user_limits():
    """Auto-patch limit check/increment in bot handlers so existing tests aren't affected.

    The limit check returns ``None`` (DB unavailable) so the handler proceeds normally.
    """
    with (
        patch("reviewmind.bot.handlers.query._check_user_limit", new_callable=AsyncMock, return_value=None),
        patch("reviewmind.bot.handlers.query._increment_user_limit", new_callable=AsyncMock),
        patch("reviewmind.bot.handlers.links._check_user_limit", new_callable=AsyncMock, return_value=None),
        patch("reviewmind.bot.handlers.links._increment_user_limit", new_callable=AsyncMock),
    ):
        yield


@pytest.fixture(autouse=True)
def _bypass_session_manager():
    """Auto-patch session manager creation so existing tests don't need Redis."""
    noop_sm_query = _make_noop_session_manager()
    noop_sm_links = _make_noop_session_manager()
    with (
        patch(
            "reviewmind.bot.handlers.query._create_session_manager",
            new_callable=AsyncMock,
            return_value=noop_sm_query,
        ),
        patch(
            "reviewmind.bot.handlers.links._create_session_manager",
            new_callable=AsyncMock,
            return_value=noop_sm_links,
        ),
        patch("reviewmind.bot.handlers.mode._persist_mode", new_callable=AsyncMock),
        patch("reviewmind.bot.handlers.mode._get_current_mode", new_callable=AsyncMock, return_value=None),
    ):
        yield
