"""Unit tests for TASK-036 — Request limits: 3 req/day free, premium bypass.

Tests cover:
- Constants (FREE_DAILY_LIMIT, PREMIUM_SUBSCRIPTION, LIMIT_REACHED_MSG)
- LimitCheckResult data class
- LimitService: admin bypass, premium bypass, normal limit check, increment
- Integration with API POST /query (limit check + increment)
- Integration with bot handlers (query + links)
- Services __init__.py exports
"""

from __future__ import annotations

from datetime import date, timezone
from datetime import datetime as dt
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from reviewmind.services.limit_service import (
    FREE_DAILY_LIMIT,
    LIMIT_REACHED_MSG,
    PREMIUM_SUBSCRIPTION,
    LimitCheckResult,
    LimitService,
)

# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════


def _mock_session() -> MagicMock:
    """Create a mocked AsyncSession."""
    session = MagicMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.flush = AsyncMock()
    session.close = AsyncMock()
    return session


def _free_user(user_id: int = 100) -> MagicMock:
    """Create a mock User with subscription='free'."""
    user = MagicMock()
    user.user_id = user_id
    user.subscription = "free"
    user.is_admin = False
    return user


def _premium_user(user_id: int = 200) -> MagicMock:
    """Create a mock User with subscription='premium'."""
    user = MagicMock()
    user.user_id = user_id
    user.subscription = "premium"
    user.is_admin = False
    return user


def _limit_row(user_id: int, used: int = 0) -> MagicMock:
    """Create a mock UserLimit row."""
    row = MagicMock()
    row.user_id = user_id
    row.date = dt.now(tz=timezone.utc).date()
    row.requests_used = used
    return row


# ══════════════════════════════════════════════════════════════
# TestConstants
# ══════════════════════════════════════════════════════════════


class TestConstants:
    def test_free_daily_limit_value(self):
        assert FREE_DAILY_LIMIT == 10

    def test_premium_subscription_value(self):
        assert PREMIUM_SUBSCRIPTION == "premium"

    def test_limit_reached_msg_has_placeholders(self):
        assert "{used}" in LIMIT_REACHED_MSG
        assert "{limit}" in LIMIT_REACHED_MSG

    def test_limit_reached_msg_format(self):
        msg = LIMIT_REACHED_MSG.format(used=10, limit=10)
        assert "10/10" in msg


# ══════════════════════════════════════════════════════════════
# TestLimitCheckResult
# ══════════════════════════════════════════════════════════════


class TestLimitCheckResult:
    def test_allowed_result(self):
        r = LimitCheckResult(allowed=True, requests_used=1, requests_limit=10, reason="ok")
        assert r.allowed is True
        assert r.requests_used == 1
        assert r.requests_limit == 10
        assert r.reason == "ok"

    def test_denied_result(self):
        r = LimitCheckResult(allowed=False, requests_used=10, requests_limit=10, reason="limit_reached")
        assert r.allowed is False
        assert r.requests_used == 10

    def test_message_when_allowed(self):
        r = LimitCheckResult(allowed=True, requests_used=0, requests_limit=10)
        assert r.message == ""

    def test_message_when_denied(self):
        r = LimitCheckResult(allowed=False, requests_used=10, requests_limit=10)
        assert "10/10" in r.message

    def test_repr(self):
        r = LimitCheckResult(allowed=True, requests_used=0, requests_limit=10, reason="admin")
        assert "admin" in repr(r)
        assert "allowed=True" in repr(r)


# ══════════════════════════════════════════════════════════════
# TestLimitServiceInit
# ══════════════════════════════════════════════════════════════


class TestLimitServiceInit:
    def test_default_daily_limit(self):
        service = LimitService(_mock_session(), admin_user_ids=[])
        assert service.daily_limit == FREE_DAILY_LIMIT

    def test_custom_daily_limit(self):
        service = LimitService(_mock_session(), admin_user_ids=[], daily_limit=10)
        assert service.daily_limit == 10

    def test_admin_ids_from_param(self):
        service = LimitService(_mock_session(), admin_user_ids=[1, 2, 3])
        assert service.admin_user_ids == {1, 2, 3}

    def test_admin_ids_lazy_from_config(self):
        """When admin_user_ids is not passed, it loads from config lazily."""
        service = LimitService(_mock_session())
        service._admin_ids = None
        # Calling admin_user_ids will try to load from config.settings
        # In test env it may succeed (real .env) or fail (no .env) — both should return a set
        ids = service.admin_user_ids
        assert isinstance(ids, set)

    def test_admin_ids_fallback_empty(self):
        # Just ensure no crash on direct creation with None
        service2 = LimitService(_mock_session())
        # Force the lazy load path to fail
        with patch("builtins.__import__", side_effect=ImportError):
            ids = service2.admin_user_ids
            assert ids == set()


# ══════════════════════════════════════════════════════════════
# TestCheckLimitAdmin
# ══════════════════════════════════════════════════════════════


class TestCheckLimitAdmin:
    @pytest.mark.asyncio
    async def test_admin_bypass(self):
        session = _mock_session()
        service = LimitService(session, admin_user_ids=[42])
        result = await service.check_limit(42)
        assert result.allowed is True
        assert result.reason == "admin"

    @pytest.mark.asyncio
    async def test_non_admin_not_bypassed(self):
        session = _mock_session()
        service = LimitService(session, admin_user_ids=[42])
        # Mock user repo to return free user
        service._user_repo.get_by_id = AsyncMock(return_value=_free_user(99))
        # Mock limit repo to return None (no usage today)
        service._limit_repo.get = AsyncMock(return_value=None)
        result = await service.check_limit(99)
        assert result.allowed is True
        assert result.reason == "ok"


# ══════════════════════════════════════════════════════════════
# TestCheckLimitPremium
# ══════════════════════════════════════════════════════════════


class TestCheckLimitPremium:
    @pytest.mark.asyncio
    async def test_premium_bypass(self):
        session = _mock_session()
        service = LimitService(session, admin_user_ids=[])
        service._user_repo.get_by_id = AsyncMock(return_value=_premium_user(200))
        result = await service.check_limit(200)
        assert result.allowed is True
        assert result.reason == "premium"

    @pytest.mark.asyncio
    async def test_free_user_no_premium_bypass(self):
        session = _mock_session()
        service = LimitService(session, admin_user_ids=[])
        service._user_repo.get_by_id = AsyncMock(return_value=_free_user(100))
        service._limit_repo.get = AsyncMock(return_value=None)
        result = await service.check_limit(100)
        assert result.allowed is True
        assert result.reason == "ok"

    @pytest.mark.asyncio
    async def test_user_not_found_no_premium(self):
        session = _mock_session()
        service = LimitService(session, admin_user_ids=[])
        service._user_repo.get_by_id = AsyncMock(return_value=None)
        service._limit_repo.get = AsyncMock(return_value=None)
        result = await service.check_limit(999)
        assert result.allowed is True  # no usage yet


# ══════════════════════════════════════════════════════════════
# TestCheckLimitFree
# ══════════════════════════════════════════════════════════════


class TestCheckLimitFree:
    @pytest.mark.asyncio
    async def test_under_limit(self):
        session = _mock_session()
        service = LimitService(session, admin_user_ids=[])
        service._user_repo.get_by_id = AsyncMock(return_value=_free_user(100))
        service._limit_repo.get = AsyncMock(return_value=_limit_row(100, used=2))
        result = await service.check_limit(100)
        assert result.allowed is True
        assert result.requests_used == 2

    @pytest.mark.asyncio
    async def test_at_limit(self):
        session = _mock_session()
        service = LimitService(session, admin_user_ids=[])
        service._user_repo.get_by_id = AsyncMock(return_value=_free_user(100))
        service._limit_repo.get = AsyncMock(return_value=_limit_row(100, used=10))
        result = await service.check_limit(100)
        assert result.allowed is False
        assert result.reason == "limit_reached"

    @pytest.mark.asyncio
    async def test_over_limit(self):
        session = _mock_session()
        service = LimitService(session, admin_user_ids=[])
        service._user_repo.get_by_id = AsyncMock(return_value=_free_user(100))
        service._limit_repo.get = AsyncMock(return_value=_limit_row(100, used=11))
        result = await service.check_limit(100)
        assert result.allowed is False
        assert result.requests_used == 11

    @pytest.mark.asyncio
    async def test_no_usage_today(self):
        session = _mock_session()
        service = LimitService(session, admin_user_ids=[])
        service._user_repo.get_by_id = AsyncMock(return_value=_free_user(100))
        service._limit_repo.get = AsyncMock(return_value=None)
        result = await service.check_limit(100)
        assert result.allowed is True
        assert result.requests_used == 0

    @pytest.mark.asyncio
    async def test_custom_limit(self):
        session = _mock_session()
        service = LimitService(session, admin_user_ids=[], daily_limit=1)
        service._user_repo.get_by_id = AsyncMock(return_value=_free_user(100))
        service._limit_repo.get = AsyncMock(return_value=_limit_row(100, used=1))
        result = await service.check_limit(100)
        assert result.allowed is False

    @pytest.mark.asyncio
    async def test_new_day_resets(self):
        """New date key means 0 usage even if yesterday was maxed."""
        session = _mock_session()
        service = LimitService(session, admin_user_ids=[])
        service._user_repo.get_by_id = AsyncMock(return_value=_free_user(100))
        # No row for today → 0 usage
        service._limit_repo.get = AsyncMock(return_value=None)
        result = await service.check_limit(100)
        assert result.allowed is True
        assert result.requests_used == 0


# ══════════════════════════════════════════════════════════════
# TestIncrement
# ══════════════════════════════════════════════════════════════


class TestIncrement:
    @pytest.mark.asyncio
    async def test_increment_creates_user(self):
        session = _mock_session()
        service = LimitService(session, admin_user_ids=[])
        service._user_repo.get_or_create = AsyncMock(return_value=(_free_user(100), True))
        incremented_row = _limit_row(100, used=1)
        service._limit_repo.increment = AsyncMock(return_value=incremented_row)
        used = await service.increment(100)
        assert used == 1
        service._user_repo.get_or_create.assert_called_once_with(100)

    @pytest.mark.asyncio
    async def test_increment_returns_new_count(self):
        session = _mock_session()
        service = LimitService(session, admin_user_ids=[])
        service._user_repo.get_or_create = AsyncMock(return_value=(_free_user(100), False))
        incremented_row = _limit_row(100, used=3)
        service._limit_repo.increment = AsyncMock(return_value=incremented_row)
        used = await service.increment(100)
        assert used == 3


# ══════════════════════════════════════════════════════════════
# TestGetUsage
# ══════════════════════════════════════════════════════════════


class TestGetUsage:
    @pytest.mark.asyncio
    async def test_get_usage_delegates_to_check(self):
        session = _mock_session()
        service = LimitService(session, admin_user_ids=[42])
        result = await service.get_usage(42)
        assert result.allowed is True
        assert result.reason == "admin"


# ══════════════════════════════════════════════════════════════
# TestTodayHelper
# ══════════════════════════════════════════════════════════════


class TestTodayHelper:
    def test_returns_date(self):
        service = LimitService(_mock_session(), admin_user_ids=[])
        today = service._today()
        assert isinstance(today, date)

    def test_returns_utc_date(self):
        service = LimitService(_mock_session(), admin_user_ids=[])
        expected = dt.now(tz=timezone.utc).date()
        assert service._today() == expected


# ══════════════════════════════════════════════════════════════
# TestApiQueryLimitIntegration
# ══════════════════════════════════════════════════════════════


class TestApiQueryLimitIntegration:
    """Test that POST /query checks and increments limits."""

    def _make_app(self, *, limit_allowed: bool = True, has_db: bool = True) -> FastAPI:
        from reviewmind.api.endpoints.query import router

        app = FastAPI()
        app.include_router(router)

        if has_db:
            engine = MagicMock()
            app.state.db_engine = engine
        else:
            app.state.db_engine = None

        app.state.qdrant = None  # force LLM fallback
        return app

    @patch("reviewmind.api.endpoints.query._increment_limit", new_callable=AsyncMock)
    @patch("reviewmind.api.endpoints.query._check_limit", new_callable=AsyncMock)
    @patch("reviewmind.api.endpoints.query.LLMClient")
    def test_limit_check_blocks_request(self, mock_llm_cls, mock_check, mock_inc):
        mock_check.return_value = LimitCheckResult(
            allowed=False, requests_used=3, requests_limit=3, reason="limit_reached"
        )
        app = self._make_app()
        client = TestClient(app)
        resp = client.post("/query", json={"user_id": 100, "query": "test"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["error"] is True
        assert "3/3" in data["answer"]
        mock_inc.assert_not_called()

    @patch("reviewmind.api.endpoints.query._increment_limit", new_callable=AsyncMock)
    @patch("reviewmind.api.endpoints.query._check_limit", new_callable=AsyncMock)
    @patch("reviewmind.api.endpoints.query.LLMClient")
    def test_limit_allows_and_increments(self, mock_llm_cls, mock_check, mock_inc):
        mock_check.return_value = LimitCheckResult(
            allowed=True, requests_used=1, requests_limit=3, reason="ok"
        )
        mock_client_instance = MagicMock()
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        mock_llm_cls.return_value = mock_client_instance

        mock_service = MagicMock()
        mock_service.answer = AsyncMock(return_value=MagicMock(answer="Test answer", error=None))

        with patch("reviewmind.api.endpoints.query.QueryService", return_value=mock_service):
            with patch("reviewmind.api.endpoints.query._log_query", new_callable=AsyncMock, return_value=1):
                app = self._make_app()
                client = TestClient(app)
                resp = client.post("/query", json={"user_id": 100, "query": "test"})

        assert resp.status_code == 200
        mock_inc.assert_called_once()

    @patch("reviewmind.api.endpoints.query._increment_limit", new_callable=AsyncMock)
    @patch("reviewmind.api.endpoints.query._check_limit", new_callable=AsyncMock)
    @patch("reviewmind.api.endpoints.query.LLMClient")
    def test_limit_check_none_allows(self, mock_llm_cls, mock_check, mock_inc):
        """When DB is unavailable (check returns None), request proceeds."""
        mock_check.return_value = None
        mock_client_instance = MagicMock()
        mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
        mock_client_instance.__aexit__ = AsyncMock(return_value=False)
        mock_llm_cls.return_value = mock_client_instance

        mock_service = MagicMock()
        mock_service.answer = AsyncMock(return_value=MagicMock(answer="Test", error=None))

        with patch("reviewmind.api.endpoints.query.QueryService", return_value=mock_service):
            with patch("reviewmind.api.endpoints.query._log_query", new_callable=AsyncMock, return_value=1):
                app = self._make_app()
                client = TestClient(app)
                resp = client.post("/query", json={"user_id": 100, "query": "test"})

        assert resp.status_code == 200
        # Request proceeded (not blocked by limits)
        assert "3/3" not in resp.json()["answer"]


# ══════════════════════════════════════════════════════════════
# TestBotQueryLimitIntegration
# ══════════════════════════════════════════════════════════════


class TestBotQueryLimitIntegration:
    """Test limit integration in bot query handler."""

    @pytest.mark.asyncio
    @patch("reviewmind.bot.handlers.query._increment_user_limit", new_callable=AsyncMock)
    @patch("reviewmind.bot.handlers.query._check_user_limit", new_callable=AsyncMock)
    async def test_limit_blocks_query(self, mock_check, mock_inc):
        from reviewmind.bot.handlers.query import on_text_message

        mock_check.return_value = LimitCheckResult(
            allowed=False, requests_used=3, requests_limit=3, reason="limit_reached"
        )
        msg = MagicMock()
        msg.text = "Sony WH-1000XM5"
        msg.from_user = MagicMock(id=100)
        msg.bot = MagicMock()
        msg.bot.send_chat_action = AsyncMock()
        msg.answer = AsyncMock()
        msg.chat = MagicMock(id=1)

        await on_text_message(msg)

        msg.answer.assert_called_once()
        call_args = msg.answer.call_args
        assert "3/3" in call_args[0][0]
        mock_inc.assert_not_called()

    @pytest.mark.asyncio
    @patch("reviewmind.bot.handlers.query._increment_user_limit", new_callable=AsyncMock)
    @patch("reviewmind.bot.handlers.query._check_user_limit", new_callable=AsyncMock)
    async def test_limit_none_allows_query(self, mock_check, mock_inc):
        """DB unavailable → limit check returns None → query proceeds."""
        from reviewmind.bot.handlers.query import on_text_message

        mock_check.return_value = None
        msg = MagicMock()
        msg.text = "Hello"
        msg.from_user = MagicMock(id=100)
        msg.bot = MagicMock()
        msg.bot.send_chat_action = AsyncMock()
        msg.answer = AsyncMock()
        msg.chat = MagicMock(id=1)

        # Will proceed to product extraction, which will return []
        # and then fallback LLM
        with patch("reviewmind.bot.handlers.query.extract_product", new_callable=AsyncMock, return_value=[]):
            with patch("reviewmind.bot.handlers.query._fallback_llm_answer", new_callable=AsyncMock):
                await on_text_message(msg)

        # Should have proceeded past the limit check
        mock_check.assert_called_once()


# ══════════════════════════════════════════════════════════════
# TestBotLinksLimitIntegration
# ══════════════════════════════════════════════════════════════


class TestBotLinksLimitIntegration:
    """Test limit integration in bot links handler."""

    @pytest.mark.asyncio
    @patch("reviewmind.bot.handlers.links._increment_user_limit", new_callable=AsyncMock)
    @patch("reviewmind.bot.handlers.links._check_user_limit", new_callable=AsyncMock)
    async def test_limit_blocks_links(self, mock_check, mock_inc):
        from reviewmind.bot.handlers.links import on_links_message

        mock_check.return_value = LimitCheckResult(
            allowed=False, requests_used=3, requests_limit=3, reason="limit_reached"
        )
        msg = MagicMock()
        msg.text = "https://youtube.com/watch?v=abc123"
        msg.from_user = MagicMock(id=100)
        msg.bot = MagicMock()
        msg.bot.send_chat_action = AsyncMock()
        msg.answer = AsyncMock()
        msg.chat = MagicMock(id=1)

        await on_links_message(msg)

        msg.answer.assert_called_once()
        call_args = msg.answer.call_args
        assert "3/3" in call_args[0][0]
        mock_inc.assert_not_called()


# ══════════════════════════════════════════════════════════════
# TestSubscribeKeyboard
# ══════════════════════════════════════════════════════════════


class TestSubscribeKeyboard:
    def test_subscribe_keyboard_has_buttons(self):
        from reviewmind.bot.keyboards import subscribe_keyboard

        kb = subscribe_keyboard()
        assert kb.inline_keyboard
        texts = [btn.text for row in kb.inline_keyboard for btn in row]
        assert any("Безлимит" in t for t in texts)

    def test_subscribe_keyboard_callback_data(self):
        from reviewmind.bot.keyboards import SUBSCRIBE_ACTION, subscribe_keyboard

        kb = subscribe_keyboard()
        callbacks = [btn.callback_data for row in kb.inline_keyboard for btn in row]
        assert SUBSCRIBE_ACTION in callbacks


# ══════════════════════════════════════════════════════════════
# TestServicesExports
# ══════════════════════════════════════════════════════════════


class TestServicesExports:
    def test_limit_service_importable(self):
        from reviewmind.services import LimitService  # noqa: F401

    def test_limit_check_result_importable(self):
        from reviewmind.services import LimitCheckResult  # noqa: F401

    def test_constants_importable(self):
        from reviewmind.services import FREE_DAILY_LIMIT, LIMIT_REACHED_MSG, PREMIUM_SUBSCRIPTION

        assert FREE_DAILY_LIMIT == 10
        assert PREMIUM_SUBSCRIPTION == "premium"
        assert "{used}" in LIMIT_REACHED_MSG

    def test_all_exports(self):
        import reviewmind.services as svc

        assert "LimitService" in svc.__all__
        assert "LimitCheckResult" in svc.__all__
        assert "FREE_DAILY_LIMIT" in svc.__all__


# ══════════════════════════════════════════════════════════════
# TestIntegrationScenarios
# ══════════════════════════════════════════════════════════════


class TestIntegrationScenarios:
    """End-to-end scenario tests matching task acceptance criteria."""

    @pytest.mark.asyncio
    async def test_10_requests_allowed_11th_blocked(self):
        """Free user: 10 queries allowed, 11th rejected."""
        session = _mock_session()
        service = LimitService(session, admin_user_ids=[])
        service._user_repo.get_by_id = AsyncMock(return_value=_free_user(100))

        # Requests 1-10: allowed
        for i in range(10):
            service._limit_repo.get = AsyncMock(return_value=_limit_row(100, used=i))
            result = await service.check_limit(100)
            assert result.allowed is True, f"Request {i + 1} should be allowed"

        # Request 11: blocked
        service._limit_repo.get = AsyncMock(return_value=_limit_row(100, used=10))
        result = await service.check_limit(100)
        assert result.allowed is False

    @pytest.mark.asyncio
    async def test_premium_unlimited(self):
        """Premium user: unlimited queries."""
        session = _mock_session()
        service = LimitService(session, admin_user_ids=[])
        service._user_repo.get_by_id = AsyncMock(return_value=_premium_user(200))

        for _ in range(10):
            result = await service.check_limit(200)
            assert result.allowed is True

    @pytest.mark.asyncio
    async def test_admin_unlimited(self):
        """Admin user: unlimited queries."""
        session = _mock_session()
        service = LimitService(session, admin_user_ids=[42])

        for _ in range(10):
            result = await service.check_limit(42)
            assert result.allowed is True
            assert result.reason == "admin"

    @pytest.mark.asyncio
    async def test_limit_message_shows_subscribe_hint(self):
        """Limit message should mention subscription."""
        session = _mock_session()
        service = LimitService(session, admin_user_ids=[])
        service._user_repo.get_by_id = AsyncMock(return_value=_free_user(100))
        service._limit_repo.get = AsyncMock(return_value=_limit_row(100, used=10))
        result = await service.check_limit(100)
        assert "10/10" in result.message
        assert "безлимитный" in result.message.lower() or "подпиш" in result.message.lower()

    @pytest.mark.asyncio
    async def test_new_day_resets_counter(self):
        """Next UTC day → fresh counter."""
        session = _mock_session()
        service = LimitService(session, admin_user_ids=[])
        service._user_repo.get_by_id = AsyncMock(return_value=_free_user(100))
        service._limit_repo.get = AsyncMock(return_value=None)  # no row for today
        result = await service.check_limit(100)
        assert result.allowed is True
        assert result.requests_used == 0
