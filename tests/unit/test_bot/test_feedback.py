"""Unit tests for TASK-048 — Feedback system (👍/👎, POST /feedback, sources).

Tests cover:
- FeedbackRequest / FeedbackResponse schema validation
- POST /feedback endpoint: success, 404, 503, 422 validation
- Bot handler: on_feedback_useful, on_feedback_bad, on_feedback_sources
- parse_query_log_id helper
- format_sources_list helper
- feedback_keyboard with query_log_id encoding
- Router/dispatcher wiring
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiogram.types import CallbackQuery, Chat, InlineKeyboardMarkup, Message, User
from fastapi import FastAPI
from fastapi.testclient import TestClient

from reviewmind.api.endpoints.feedback import router as api_feedback_router
from reviewmind.api.schemas import FeedbackRequest, FeedbackResponse
from reviewmind.bot.handlers.feedback import (
    FEEDBACK_ERROR_MSG,
    FEEDBACK_THANKS_MSG,
    NO_SOURCES_MSG,
    RATING_BAD,
    RATING_USEFUL,
    format_sources_list,
    on_feedback_bad,
    on_feedback_sources,
    on_feedback_useful,
    parse_query_log_id,
)
from reviewmind.bot.keyboards import (
    FEEDBACK_BAD,
    FEEDBACK_PREFIX,
    FEEDBACK_SOURCES,
    FEEDBACK_USEFUL,
    feedback_keyboard,
)

# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════


def _make_user(user_id: int = 12345) -> User:
    return User(id=user_id, is_bot=False, first_name="Test")


def _make_chat(chat_id: int = 12345) -> Chat:
    return Chat(id=chat_id, type="private")


def _make_callback(data: str, user_id: int = 12345) -> MagicMock:
    cb = MagicMock(spec=CallbackQuery)
    cb.data = data
    cb.from_user = _make_user(user_id)
    cb.message = MagicMock(spec=Message)
    cb.message.answer = AsyncMock()
    cb.answer = AsyncMock()
    return cb


def _make_api_app(**state_attrs) -> FastAPI:
    app = FastAPI()
    app.include_router(api_feedback_router)
    for k, v in state_attrs.items():
        setattr(app.state, k, v)
    return app


def _mock_engine() -> MagicMock:
    return MagicMock(name="db_engine")


# ══════════════════════════════════════════════════════════════
# Tests — Constants
# ══════════════════════════════════════════════════════════════


class TestConstants:
    def test_rating_useful_value(self):
        assert RATING_USEFUL == 1

    def test_rating_bad_value(self):
        assert RATING_BAD == -1

    def test_feedback_thanks_msg(self):
        assert "Спасибо" in FEEDBACK_THANKS_MSG

    def test_no_sources_msg(self):
        assert NO_SOURCES_MSG == "Источники не найдены."

    def test_feedback_prefix(self):
        assert FEEDBACK_PREFIX == "feedback:"


# ══════════════════════════════════════════════════════════════
# Tests — parse_query_log_id
# ══════════════════════════════════════════════════════════════


class TestParseQueryLogId:
    def test_no_id(self):
        assert parse_query_log_id("feedback:useful") is None

    def test_with_id(self):
        assert parse_query_log_id("feedback:useful:42") == 42

    def test_with_bad_id(self):
        assert parse_query_log_id("feedback:bad:7") == 7

    def test_sources_with_id(self):
        assert parse_query_log_id("feedback:sources:123") == 123

    def test_non_numeric_id(self):
        assert parse_query_log_id("feedback:useful:abc") is None

    def test_empty_string(self):
        assert parse_query_log_id("") is None

    def test_single_colon(self):
        assert parse_query_log_id("feedback") is None


# ══════════════════════════════════════════════════════════════
# Tests — format_sources_list
# ══════════════════════════════════════════════════════════════


class TestFormatSourcesList:
    def test_empty_list(self):
        assert format_sources_list([]) == NO_SOURCES_MSG

    def test_string_urls(self):
        result = format_sources_list(["https://example.com", "https://test.com"])
        assert "1. https://example.com" in result
        assert "2. https://test.com" in result

    def test_dict_with_url(self):
        result = format_sources_list([{"url": "https://example.com"}])
        assert "1. https://example.com" in result

    def test_dict_with_source_url_key(self):
        result = format_sources_list([{"source_url": "https://example.com"}])
        assert "1. https://example.com" in result

    def test_curated_badge(self):
        result = format_sources_list([{"url": "https://wirecutter.com", "is_curated": True}])
        assert "📚" in result
        assert "https://wirecutter.com" in result

    def test_sponsored_badge(self):
        result = format_sources_list([{"url": "https://sponsor.com", "is_sponsored": True}])
        assert "⚠️ [sponsored]" in result

    def test_curated_and_sponsored(self):
        result = format_sources_list([{"url": "https://both.com", "is_curated": True, "is_sponsored": True}])
        assert "📚" in result
        assert "⚠️ [sponsored]" in result

    def test_mixed_types(self):
        sources = [
            "https://plain.com",
            {"url": "https://curated.com", "is_curated": True},
            {"url": "https://sponsored.com", "is_sponsored": True},
        ]
        result = format_sources_list(sources)
        assert "1. https://plain.com" in result
        assert "2. 📚 https://curated.com" in result
        assert "3. ⚠️ [sponsored] https://sponsored.com" in result

    def test_numbering(self):
        result = format_sources_list(["a", "b", "c"])
        lines = result.strip().split("\n")
        assert len(lines) == 3
        assert lines[0].startswith("1.")
        assert lines[2].startswith("3.")


# ══════════════════════════════════════════════════════════════
# Tests — feedback_keyboard
# ══════════════════════════════════════════════════════════════


class TestFeedbackKeyboard:
    def test_returns_inline_markup(self):
        kb = feedback_keyboard()
        assert isinstance(kb, InlineKeyboardMarkup)

    def test_three_buttons(self):
        kb = feedback_keyboard()
        assert len(kb.inline_keyboard[0]) == 3

    def test_no_id_callback_data(self):
        kb = feedback_keyboard()
        buttons = kb.inline_keyboard[0]
        assert buttons[0].callback_data == FEEDBACK_USEFUL
        assert buttons[1].callback_data == FEEDBACK_BAD
        assert buttons[2].callback_data == FEEDBACK_SOURCES

    def test_with_id_callback_data(self):
        kb = feedback_keyboard(query_log_id=42)
        buttons = kb.inline_keyboard[0]
        assert buttons[0].callback_data == "feedback:useful:42"
        assert buttons[1].callback_data == "feedback:bad:42"
        assert buttons[2].callback_data == "feedback:sources:42"

    def test_button_labels(self):
        kb = feedback_keyboard()
        buttons = kb.inline_keyboard[0]
        assert "👍" in buttons[0].text
        assert "👎" in buttons[1].text
        assert "📎" in buttons[2].text


# ══════════════════════════════════════════════════════════════
# Tests — Bot callback: on_feedback_useful
# ══════════════════════════════════════════════════════════════


class TestOnFeedbackUseful:
    @pytest.mark.asyncio
    async def test_with_id_saves_rating(self):
        cb = _make_callback("feedback:useful:10")
        with patch("reviewmind.bot.handlers.feedback._save_rating", new_callable=AsyncMock, return_value=True) as mock:
            await on_feedback_useful(cb)
            mock.assert_awaited_once_with(10, RATING_USEFUL, 12345)
            cb.answer.assert_awaited_once_with(FEEDBACK_THANKS_MSG)

    @pytest.mark.asyncio
    async def test_with_id_save_fails(self):
        cb = _make_callback("feedback:useful:10")
        with patch("reviewmind.bot.handlers.feedback._save_rating", new_callable=AsyncMock, return_value=False):
            await on_feedback_useful(cb)
            cb.answer.assert_awaited_once_with(FEEDBACK_ERROR_MSG)

    @pytest.mark.asyncio
    async def test_no_id_fallback_to_latest(self):
        cb = _make_callback("feedback:useful")
        with (
            patch("reviewmind.bot.handlers.feedback._get_latest_query_log_id", new_callable=AsyncMock, return_value=99),
            patch("reviewmind.bot.handlers.feedback._save_rating", new_callable=AsyncMock, return_value=True) as mock,
        ):
            await on_feedback_useful(cb)
            mock.assert_awaited_once_with(99, RATING_USEFUL, 12345)

    @pytest.mark.asyncio
    async def test_no_id_no_latest_still_thanks(self):
        cb = _make_callback("feedback:useful")
        with patch(
            "reviewmind.bot.handlers.feedback._get_latest_query_log_id", new_callable=AsyncMock, return_value=None
        ):
            await on_feedback_useful(cb)
            cb.answer.assert_awaited_once_with(FEEDBACK_THANKS_MSG)


# ══════════════════════════════════════════════════════════════
# Tests — Bot callback: on_feedback_bad
# ══════════════════════════════════════════════════════════════


class TestOnFeedbackBad:
    @pytest.mark.asyncio
    async def test_saves_negative_rating(self):
        cb = _make_callback("feedback:bad:20")
        with patch("reviewmind.bot.handlers.feedback._save_rating", new_callable=AsyncMock, return_value=True) as mock:
            await on_feedback_bad(cb)
            mock.assert_awaited_once_with(20, RATING_BAD, 12345)

    @pytest.mark.asyncio
    async def test_save_fails(self):
        cb = _make_callback("feedback:bad:20")
        with patch("reviewmind.bot.handlers.feedback._save_rating", new_callable=AsyncMock, return_value=False):
            await on_feedback_bad(cb)
            cb.answer.assert_awaited_once_with(FEEDBACK_ERROR_MSG)

    @pytest.mark.asyncio
    async def test_no_id_fallback(self):
        cb = _make_callback("feedback:bad")
        with (
            patch("reviewmind.bot.handlers.feedback._get_latest_query_log_id", new_callable=AsyncMock, return_value=5),
            patch("reviewmind.bot.handlers.feedback._save_rating", new_callable=AsyncMock, return_value=True) as mock,
        ):
            await on_feedback_bad(cb)
            mock.assert_awaited_once_with(5, RATING_BAD, 12345)


# ══════════════════════════════════════════════════════════════
# Tests — Bot callback: on_feedback_sources
# ══════════════════════════════════════════════════════════════


class TestOnFeedbackSources:
    @pytest.mark.asyncio
    async def test_shows_sources(self):
        cb = _make_callback("feedback:sources:30")
        sources = ["https://example.com", "https://test.org"]
        with patch(
            "reviewmind.bot.handlers.feedback._get_sources_for_log", new_callable=AsyncMock, return_value=sources
        ):
            await on_feedback_sources(cb)
            # Short text → show_alert
            cb.answer.assert_awaited_once()
            call_text = cb.answer.call_args[0][0]
            assert "1. https://example.com" in call_text

    @pytest.mark.asyncio
    async def test_long_sources_sent_as_message(self):
        cb = _make_callback("feedback:sources:30")
        sources = [f"https://example.com/very-long-path/article/{i}" for i in range(20)]
        with patch(
            "reviewmind.bot.handlers.feedback._get_sources_for_log", new_callable=AsyncMock, return_value=sources
        ):
            await on_feedback_sources(cb)
            # Long → message.answer
            cb.message.answer.assert_awaited_once()
            call_text = cb.message.answer.call_args[0][0]
            assert "📎" in call_text
            assert "Источники" in call_text

    @pytest.mark.asyncio
    async def test_no_sources(self):
        cb = _make_callback("feedback:sources:30")
        with patch("reviewmind.bot.handlers.feedback._get_sources_for_log", new_callable=AsyncMock, return_value=[]):
            await on_feedback_sources(cb)
            cb.answer.assert_awaited_once_with(NO_SOURCES_MSG, show_alert=True)

    @pytest.mark.asyncio
    async def test_sources_none(self):
        cb = _make_callback("feedback:sources:30")
        with patch("reviewmind.bot.handlers.feedback._get_sources_for_log", new_callable=AsyncMock, return_value=None):
            await on_feedback_sources(cb)
            cb.answer.assert_awaited_once_with(NO_SOURCES_MSG, show_alert=True)

    @pytest.mark.asyncio
    async def test_no_query_log_id(self):
        cb = _make_callback("feedback:sources")
        with patch(
            "reviewmind.bot.handlers.feedback._get_latest_query_log_id", new_callable=AsyncMock, return_value=None
        ):
            await on_feedback_sources(cb)
            cb.answer.assert_awaited_once_with(NO_SOURCES_MSG, show_alert=True)


# ══════════════════════════════════════════════════════════════
# Tests — API: FeedbackRequest / FeedbackResponse schemas
# ══════════════════════════════════════════════════════════════


class TestFeedbackSchemas:
    def test_valid_request_thumbs_up(self):
        req = FeedbackRequest(query_log_id=1, rating=1)
        assert req.query_log_id == 1
        assert req.rating == 1

    def test_valid_request_thumbs_down(self):
        req = FeedbackRequest(query_log_id=99, rating=-1)
        assert req.rating == -1

    def test_invalid_rating_too_high(self):
        with pytest.raises(Exception):
            FeedbackRequest(query_log_id=1, rating=2)

    def test_invalid_rating_too_low(self):
        with pytest.raises(Exception):
            FeedbackRequest(query_log_id=1, rating=-2)

    def test_missing_query_log_id(self):
        with pytest.raises(Exception):
            FeedbackRequest(rating=1)  # type: ignore[call-arg]

    def test_response_model(self):
        resp = FeedbackResponse(query_log_id=1, rating=1, message="OK")
        assert resp.query_log_id == 1
        assert resp.message == "OK"

    def test_response_default_message(self):
        resp = FeedbackResponse(query_log_id=1, rating=1)
        assert "Спасибо" in resp.message


# ══════════════════════════════════════════════════════════════
# Tests — API: POST /feedback endpoint
# ══════════════════════════════════════════════════════════════


class TestApiFeedbackEndpoint:
    def _mock_update_rating(self, return_value):
        """Return a mock QueryLogRepository whose update_rating returns *return_value*."""
        mock_repo = MagicMock()
        mock_repo.update_rating = AsyncMock(return_value=return_value)
        return mock_repo

    def test_success(self):
        mock_log = MagicMock()
        mock_log.id = 1
        mock_log.rating = 1

        app = _make_api_app(db_engine=_mock_engine())
        with (
            patch("reviewmind.api.endpoints.feedback.async_sessionmaker") as mock_sm,
            patch("reviewmind.api.endpoints.feedback.QueryLogRepository") as mock_repo_cls,
        ):
            mock_session = AsyncMock()
            mock_sm.return_value = MagicMock(__aenter__=AsyncMock(return_value=mock_session), __aexit__=AsyncMock())
            # Make the context manager work correctly
            mock_sm.return_value.__call__ = MagicMock(return_value=mock_session)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)
            mock_repo_cls.return_value.update_rating = AsyncMock(return_value=mock_log)

            client = TestClient(app)
            resp = client.post("/feedback", json={"query_log_id": 1, "rating": 1})
            assert resp.status_code == 200
            data = resp.json()
            assert data["query_log_id"] == 1
            assert data["rating"] == 1
            assert "Спасибо" in data["message"]

    def test_not_found(self):
        app = _make_api_app(db_engine=_mock_engine())
        with (
            patch("reviewmind.api.endpoints.feedback.async_sessionmaker") as mock_sm,
            patch("reviewmind.api.endpoints.feedback.QueryLogRepository") as mock_repo_cls,
        ):
            mock_session = AsyncMock()
            mock_sm.return_value = MagicMock(__aenter__=AsyncMock(return_value=mock_session), __aexit__=AsyncMock())
            mock_sm.return_value.__call__ = MagicMock(return_value=mock_session)
            mock_session.__aenter__ = AsyncMock(return_value=mock_session)
            mock_session.__aexit__ = AsyncMock(return_value=False)
            mock_repo_cls.return_value.update_rating = AsyncMock(return_value=None)

            client = TestClient(app)
            resp = client.post("/feedback", json={"query_log_id": 999, "rating": -1})
            assert resp.status_code == 404

    def test_no_db_engine(self):
        app = _make_api_app()  # no db_engine
        client = TestClient(app)
        resp = client.post("/feedback", json={"query_log_id": 1, "rating": 1})
        assert resp.status_code == 503

    def test_invalid_body(self):
        app = _make_api_app(db_engine=_mock_engine())
        client = TestClient(app)
        resp = client.post("/feedback", json={"query_log_id": 1})
        assert resp.status_code == 422

    def test_rating_zero_accepted(self):
        # rating=0 is within [-1, 1] range, should be accepted
        req = FeedbackRequest(query_log_id=1, rating=0)
        assert req.rating == 0


# ══════════════════════════════════════════════════════════════
# Tests — Router wiring
# ══════════════════════════════════════════════════════════════


class TestRouterWiring:
    def test_api_router_includes_feedback(self):
        from reviewmind.api.router import api_router

        routes = [r.path for r in api_router.routes]
        assert "/feedback" in routes

    def test_bot_dispatcher_includes_feedback(self):
        """Verify feedback_router is referenced in create_dispatcher source."""
        import inspect

        from reviewmind.bot import main as bot_main

        source = inspect.getsource(bot_main.create_dispatcher)
        assert "feedback_router" in source

    def test_feedback_router_before_query(self):
        """Feedback callback router must be registered before the catch-all query router.

        We verify by inspecting the source code ordering rather than calling
        create_dispatcher() again (aiogram routers can only be attached once).
        """
        import inspect

        from reviewmind.bot import main as bot_main

        source = inspect.getsource(bot_main.create_dispatcher)
        feedback_pos = source.index("feedback_router")
        query_pos = source.index("query_router")
        assert feedback_pos < query_pos


# ══════════════════════════════════════════════════════════════
# Tests — Schema exports
# ══════════════════════════════════════════════════════════════


class TestSchemaExports:
    def test_feedback_request_importable(self):
        from reviewmind.api.schemas import FeedbackRequest as FR

        assert FR is not None

    def test_feedback_response_importable(self):
        from reviewmind.api.schemas import FeedbackResponse as FR

        assert FR is not None

    def test_feedback_keyboard_from_keyboards(self):
        from reviewmind.bot.keyboards import feedback_keyboard as fk

        assert callable(fk)


# ══════════════════════════════════════════════════════════════
# Tests — Full app integration
# ══════════════════════════════════════════════════════════════


class TestFullApp:
    def test_feedback_endpoint_accessible_via_main_app(self):
        from reviewmind.main import create_app

        app = create_app()
        routes = [r.path for r in app.routes]
        assert "/feedback" in routes


# ══════════════════════════════════════════════════════════════
# Tests — Integration scenarios
# ══════════════════════════════════════════════════════════════


class TestIntegrationScenarios:
    """End-to-end-like tests that verify the full feedback flow."""

    def test_feedback_keyboard_roundtrip(self):
        """Keyboard encodes ID → parse_query_log_id decodes it."""
        kb = feedback_keyboard(query_log_id=42)
        useful_data = kb.inline_keyboard[0][0].callback_data
        bad_data = kb.inline_keyboard[0][1].callback_data
        sources_data = kb.inline_keyboard[0][2].callback_data

        assert parse_query_log_id(useful_data) == 42
        assert parse_query_log_id(bad_data) == 42
        assert parse_query_log_id(sources_data) == 42

    def test_feedback_keyboard_no_id_roundtrip(self):
        kb = feedback_keyboard()
        useful_data = kb.inline_keyboard[0][0].callback_data
        assert parse_query_log_id(useful_data) is None

    @pytest.mark.asyncio
    async def test_useful_then_bad_updates_rating(self):
        """Simulate pressing 👍 then 👎 — both calls go through."""
        cb_useful = _make_callback("feedback:useful:50")
        cb_bad = _make_callback("feedback:bad:50")
        with patch("reviewmind.bot.handlers.feedback._save_rating", new_callable=AsyncMock, return_value=True) as mock:
            await on_feedback_useful(cb_useful)
            await on_feedback_bad(cb_bad)
            assert mock.await_count == 2
            calls = mock.call_args_list
            assert calls[0].args == (50, RATING_USEFUL, 12345)
            assert calls[1].args == (50, RATING_BAD, 12345)

    def test_format_sources_with_real_data(self):
        """Format a realistic sources_used payload."""
        sources = [
            {"url": "https://www.youtube.com/watch?v=abc", "is_curated": False, "is_sponsored": False},
            {"url": "https://www.wirecutter.com/best-headphones", "is_curated": True},
            {"url": "https://sponsor.tech/review", "is_sponsored": True},
        ]
        result = format_sources_list(sources)
        lines = result.strip().split("\n")
        assert len(lines) == 3
        assert "📚" in lines[1]
        assert "⚠️" in lines[2]
        assert "youtube.com" in lines[0]
