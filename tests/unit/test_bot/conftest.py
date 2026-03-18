"""Shared fixtures for bot handler tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


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
