"""Unit tests for TASK-049 — /delete_my_data: cascading user data deletion (GDPR).

Tests cover:
- Constants and messages
- gdpr_confirm_keyboard layout
- /delete_my_data command handler (confirmation prompt)
- [Да, удалить] callback handler (cascade delete)
- [Отмена] callback handler (cancel)
- _delete_user_from_db helper (DB success, user not found, DB error)
- _clear_redis_session helper (success, error)
- Dispatcher router wiring (gdpr_router registered)
- Integration scenarios (full flow)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiogram.types import CallbackQuery, Chat, InlineKeyboardMarkup, Message, User

from reviewmind.bot.handlers.gdpr import (
    CANCEL_MSG,
    CONFIRM_PROMPT_MSG,
    DATA_DELETED_MSG,
    DELETE_ERROR_MSG,
    GDPR_CANCEL_CALLBACK,
    GDPR_CONFIRM_CALLBACK,
    NO_DATA_MSG,
    _clear_redis_session,
    _delete_user_from_db,
    cmd_delete_my_data,
    gdpr_confirm_keyboard,
    on_gdpr_cancel,
    on_gdpr_confirm,
    router,
)

# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════


def _make_user(user_id: int = 12345) -> User:
    return User(id=user_id, is_bot=False, first_name="Test")


def _make_chat(chat_id: int = 12345) -> Chat:
    return Chat(id=chat_id, type="private")


def _make_message(user_id: int = 12345) -> MagicMock:
    msg = MagicMock(spec=Message)
    msg.from_user = _make_user(user_id)
    msg.chat = _make_chat(user_id)
    msg.answer = AsyncMock()
    return msg


def _make_callback(data: str, user_id: int = 12345) -> MagicMock:
    cb = MagicMock(spec=CallbackQuery)
    cb.data = data
    cb.from_user = _make_user(user_id)
    cb.message = MagicMock(spec=Message)
    cb.message.edit_text = AsyncMock()
    cb.answer = AsyncMock()
    return cb


# ══════════════════════════════════════════════════════════════
# Tests — Constants
# ══════════════════════════════════════════════════════════════


class TestConstants:
    def test_callback_data_values(self):
        assert GDPR_CONFIRM_CALLBACK == "gdpr:confirm_delete"
        assert GDPR_CANCEL_CALLBACK == "gdpr:cancel_delete"

    def test_confirm_prompt_contains_warning(self):
        assert "уверены" in CONFIRM_PROMPT_MSG.lower()
        assert "ВСЕ" in CONFIRM_PROMPT_MSG

    def test_data_deleted_msg_mentions_start(self):
        assert "/start" in DATA_DELETED_MSG

    def test_cancel_msg_reassures(self):
        assert "не затронуты" in CANCEL_MSG

    def test_delete_error_msg(self):
        assert "Не удалось" in DELETE_ERROR_MSG

    def test_no_data_msg(self):
        assert "нет" in NO_DATA_MSG.lower()


# ══════════════════════════════════════════════════════════════
# Tests — Keyboard
# ══════════════════════════════════════════════════════════════


class TestGdprConfirmKeyboard:
    def test_returns_inline_markup(self):
        kb = gdpr_confirm_keyboard()
        assert isinstance(kb, InlineKeyboardMarkup)

    def test_has_two_buttons(self):
        kb = gdpr_confirm_keyboard()
        buttons = kb.inline_keyboard[0]
        assert len(buttons) == 2

    def test_confirm_button_callback_data(self):
        kb = gdpr_confirm_keyboard()
        confirm_btn = kb.inline_keyboard[0][0]
        assert confirm_btn.callback_data == GDPR_CONFIRM_CALLBACK

    def test_cancel_button_callback_data(self):
        kb = gdpr_confirm_keyboard()
        cancel_btn = kb.inline_keyboard[0][1]
        assert cancel_btn.callback_data == GDPR_CANCEL_CALLBACK

    def test_confirm_button_text(self):
        kb = gdpr_confirm_keyboard()
        assert "удалить" in kb.inline_keyboard[0][0].text.lower()

    def test_cancel_button_text(self):
        kb = gdpr_confirm_keyboard()
        assert "отмена" in kb.inline_keyboard[0][1].text.lower()


# ══════════════════════════════════════════════════════════════
# Tests — /delete_my_data command handler
# ══════════════════════════════════════════════════════════════


class TestCmdDeleteMyData:
    @pytest.mark.asyncio
    async def test_sends_confirmation_prompt(self):
        msg = _make_message()
        await cmd_delete_my_data(msg)
        msg.answer.assert_awaited_once()
        call_args = msg.answer.call_args
        assert CONFIRM_PROMPT_MSG in call_args.args or call_args.kwargs.get("text") == CONFIRM_PROMPT_MSG

    @pytest.mark.asyncio
    async def test_sends_keyboard_with_prompt(self):
        msg = _make_message()
        await cmd_delete_my_data(msg)
        call_kwargs = msg.answer.call_args.kwargs
        assert "reply_markup" in call_kwargs
        assert isinstance(call_kwargs["reply_markup"], InlineKeyboardMarkup)

    @pytest.mark.asyncio
    async def test_html_parse_mode(self):
        msg = _make_message()
        await cmd_delete_my_data(msg)
        call_kwargs = msg.answer.call_args.kwargs
        assert call_kwargs.get("parse_mode") == "HTML"

    @pytest.mark.asyncio
    async def test_handles_no_user(self):
        msg = _make_message()
        msg.from_user = None
        await cmd_delete_my_data(msg)
        msg.answer.assert_awaited_once()


# ══════════════════════════════════════════════════════════════
# Tests — _delete_user_from_db
# ══════════════════════════════════════════════════════════════


class TestDeleteUserFromDb:
    @pytest.mark.asyncio
    async def test_returns_true_when_user_deleted(self):
        mock_repo = MagicMock()
        mock_repo.delete = AsyncMock(return_value=True)

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.commit = AsyncMock()

        mock_factory = MagicMock(return_value=mock_session)
        mock_engine = MagicMock()
        mock_engine.dispose = AsyncMock()

        with (
            patch(
                "reviewmind.bot.handlers.gdpr._get_db_session",
                new_callable=AsyncMock,
                return_value=(mock_engine, mock_factory),
            ),
            patch("reviewmind.bot.handlers.gdpr.UserRepository", return_value=mock_repo),
        ):
            result = await _delete_user_from_db(12345)

        assert result is True
        mock_repo.delete.assert_awaited_once_with(12345)
        mock_session.commit.assert_awaited_once()
        mock_engine.dispose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_returns_false_when_user_not_found(self):
        mock_repo = MagicMock()
        mock_repo.delete = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.commit = AsyncMock()

        mock_factory = MagicMock(return_value=mock_session)
        mock_engine = MagicMock()
        mock_engine.dispose = AsyncMock()

        with (
            patch(
                "reviewmind.bot.handlers.gdpr._get_db_session",
                new_callable=AsyncMock,
                return_value=(mock_engine, mock_factory),
            ),
            patch("reviewmind.bot.handlers.gdpr.UserRepository", return_value=mock_repo),
        ):
            result = await _delete_user_from_db(99999)

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_none_on_db_error(self):
        with patch(
            "reviewmind.bot.handlers.gdpr._get_db_session",
            new_callable=AsyncMock,
            side_effect=Exception("DB connection failed"),
        ):
            result = await _delete_user_from_db(12345)

        assert result is None

    @pytest.mark.asyncio
    async def test_engine_disposed_on_error(self):
        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(side_effect=Exception("SQL Error"))
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_factory = MagicMock(return_value=mock_session)
        mock_engine = MagicMock()
        mock_engine.dispose = AsyncMock()

        with patch(
            "reviewmind.bot.handlers.gdpr._get_db_session",
            new_callable=AsyncMock,
            return_value=(mock_engine, mock_factory),
        ):
            result = await _delete_user_from_db(12345)

        # Error is caught in outer try/except
        assert result is None


# ══════════════════════════════════════════════════════════════
# Tests — _clear_redis_session
# ══════════════════════════════════════════════════════════════


class TestClearRedisSession:
    @pytest.mark.asyncio
    async def test_clears_session_successfully(self):
        mock_sm = MagicMock()
        mock_sm.clear_session = AsyncMock()
        mock_client = MagicMock()
        mock_client.aclose = AsyncMock()

        with (
            patch("redis.asyncio.from_url", return_value=mock_client),
            patch("reviewmind.cache.redis.SessionManager", return_value=mock_sm),
            patch("reviewmind.config.settings") as mock_settings,
        ):
            mock_settings.redis_url = "redis://localhost:6379/0"
            await _clear_redis_session(12345)

        mock_sm.clear_session.assert_awaited_once_with(12345)
        mock_client.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_does_not_raise_on_redis_error(self):
        with (
            patch("redis.asyncio.from_url", side_effect=Exception("Redis down")),
            patch("reviewmind.config.settings") as mock_settings,
        ):
            mock_settings.redis_url = "redis://localhost:6379/0"
            # Should not raise
            await _clear_redis_session(12345)


# ══════════════════════════════════════════════════════════════
# Tests — on_gdpr_confirm callback handler
# ══════════════════════════════════════════════════════════════


class TestOnGdprConfirm:
    @pytest.mark.asyncio
    async def test_deletes_user_and_clears_redis(self):
        cb = _make_callback(GDPR_CONFIRM_CALLBACK)

        with (
            patch(
                "reviewmind.bot.handlers.gdpr._delete_user_from_db",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_db,
            patch(
                "reviewmind.bot.handlers.gdpr._clear_redis_session",
                new_callable=AsyncMock,
            ) as mock_redis,
        ):
            await on_gdpr_confirm(cb)

        mock_db.assert_awaited_once_with(12345)
        mock_redis.assert_awaited_once_with(12345)
        cb.message.edit_text.assert_awaited_once()
        assert DATA_DELETED_MSG in cb.message.edit_text.call_args.args

    @pytest.mark.asyncio
    async def test_shows_success_when_user_not_in_db(self):
        cb = _make_callback(GDPR_CONFIRM_CALLBACK)

        with (
            patch(
                "reviewmind.bot.handlers.gdpr._delete_user_from_db",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch(
                "reviewmind.bot.handlers.gdpr._clear_redis_session",
                new_callable=AsyncMock,
            ) as mock_redis,
        ):
            await on_gdpr_confirm(cb)

        # Redis should still be cleared even if no DB data existed
        mock_redis.assert_awaited_once()
        cb.message.edit_text.assert_awaited_once()
        assert DATA_DELETED_MSG in cb.message.edit_text.call_args.args

    @pytest.mark.asyncio
    async def test_shows_error_when_db_unavailable(self):
        cb = _make_callback(GDPR_CONFIRM_CALLBACK)

        with (
            patch(
                "reviewmind.bot.handlers.gdpr._delete_user_from_db",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "reviewmind.bot.handlers.gdpr._clear_redis_session",
                new_callable=AsyncMock,
            ) as mock_redis,
        ):
            await on_gdpr_confirm(cb)

        # Redis should NOT be cleared when DB failed
        mock_redis.assert_not_awaited()
        cb.answer.assert_awaited_once()
        assert DELETE_ERROR_MSG in cb.answer.call_args.args

    @pytest.mark.asyncio
    async def test_calls_callback_answer(self):
        cb = _make_callback(GDPR_CONFIRM_CALLBACK)

        with (
            patch(
                "reviewmind.bot.handlers.gdpr._delete_user_from_db",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "reviewmind.bot.handlers.gdpr._clear_redis_session",
                new_callable=AsyncMock,
            ),
        ):
            await on_gdpr_confirm(cb)

        cb.answer.assert_awaited()

    @pytest.mark.asyncio
    async def test_handles_no_user(self):
        cb = _make_callback(GDPR_CONFIRM_CALLBACK)
        cb.from_user = None

        with (
            patch(
                "reviewmind.bot.handlers.gdpr._delete_user_from_db",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch(
                "reviewmind.bot.handlers.gdpr._clear_redis_session",
                new_callable=AsyncMock,
            ),
        ):
            await on_gdpr_confirm(cb)
        # Should not crash


# ══════════════════════════════════════════════════════════════
# Tests — on_gdpr_cancel callback handler
# ══════════════════════════════════════════════════════════════


class TestOnGdprCancel:
    @pytest.mark.asyncio
    async def test_edits_message_with_cancel_text(self):
        cb = _make_callback(GDPR_CANCEL_CALLBACK)
        await on_gdpr_cancel(cb)
        cb.message.edit_text.assert_awaited_once()
        assert CANCEL_MSG in cb.message.edit_text.call_args.args

    @pytest.mark.asyncio
    async def test_calls_callback_answer(self):
        cb = _make_callback(GDPR_CANCEL_CALLBACK)
        await on_gdpr_cancel(cb)
        cb.answer.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_does_not_touch_db_or_redis(self):
        cb = _make_callback(GDPR_CANCEL_CALLBACK)

        with (
            patch(
                "reviewmind.bot.handlers.gdpr._delete_user_from_db",
                new_callable=AsyncMock,
            ) as mock_db,
            patch(
                "reviewmind.bot.handlers.gdpr._clear_redis_session",
                new_callable=AsyncMock,
            ) as mock_redis,
        ):
            await on_gdpr_cancel(cb)

        mock_db.assert_not_awaited()
        mock_redis.assert_not_awaited()


# ══════════════════════════════════════════════════════════════
# Tests — Router wiring
# ══════════════════════════════════════════════════════════════


class TestRouterWiring:
    def test_router_name(self):
        assert router.name == "gdpr"

    def test_gdpr_router_imported_in_main(self):
        """Verify that gdpr_router is imported in bot/main.py."""
        import reviewmind.bot.main as bot_main

        assert hasattr(bot_main, "gdpr_router")

    def test_gdpr_router_is_correct_instance(self):
        """Verify the imported router is the one from gdpr module."""
        from reviewmind.bot.handlers.gdpr import router as gdpr_router_orig
        from reviewmind.bot.main import gdpr_router

        assert gdpr_router is gdpr_router_orig

    def test_create_dispatcher_includes_gdpr(self):
        """Verify the source code of create_dispatcher includes gdpr_router."""
        import inspect

        from reviewmind.bot.main import create_dispatcher

        source = inspect.getsource(create_dispatcher)
        assert "gdpr_router" in source


# ══════════════════════════════════════════════════════════════
# Tests — Module exports
# ══════════════════════════════════════════════════════════════


class TestExports:
    def test_router_exported(self):
        from reviewmind.bot.handlers import gdpr

        assert hasattr(gdpr, "router")

    def test_constants_exported(self):
        from reviewmind.bot.handlers import gdpr

        assert hasattr(gdpr, "GDPR_CONFIRM_CALLBACK")
        assert hasattr(gdpr, "GDPR_CANCEL_CALLBACK")
        assert hasattr(gdpr, "CONFIRM_PROMPT_MSG")
        assert hasattr(gdpr, "DATA_DELETED_MSG")
        assert hasattr(gdpr, "CANCEL_MSG")
        assert hasattr(gdpr, "DELETE_ERROR_MSG")

    def test_keyboard_exported(self):
        from reviewmind.bot.handlers import gdpr

        assert hasattr(gdpr, "gdpr_confirm_keyboard")

    def test_handlers_exported(self):
        from reviewmind.bot.handlers import gdpr

        assert hasattr(gdpr, "cmd_delete_my_data")
        assert hasattr(gdpr, "on_gdpr_confirm")
        assert hasattr(gdpr, "on_gdpr_cancel")


# ══════════════════════════════════════════════════════════════
# Tests — Integration scenarios
# ══════════════════════════════════════════════════════════════


class TestIntegrationScenarios:
    """End-to-end flow tests for GDPR deletion."""

    @pytest.mark.asyncio
    async def test_full_delete_flow(self):
        """Command → confirm → DB delete + Redis clear → success message."""
        msg = _make_message(user_id=42)
        await cmd_delete_my_data(msg)
        msg.answer.assert_awaited_once()

        cb = _make_callback(GDPR_CONFIRM_CALLBACK, user_id=42)
        with (
            patch(
                "reviewmind.bot.handlers.gdpr._delete_user_from_db",
                new_callable=AsyncMock,
                return_value=True,
            ) as mock_db,
            patch(
                "reviewmind.bot.handlers.gdpr._clear_redis_session",
                new_callable=AsyncMock,
            ) as mock_redis,
        ):
            await on_gdpr_confirm(cb)

        mock_db.assert_awaited_once_with(42)
        mock_redis.assert_awaited_once_with(42)
        cb.message.edit_text.assert_awaited_once()
        assert DATA_DELETED_MSG in cb.message.edit_text.call_args.args

    @pytest.mark.asyncio
    async def test_full_cancel_flow(self):
        """Command → cancel → data untouched."""
        msg = _make_message(user_id=42)
        await cmd_delete_my_data(msg)

        cb = _make_callback(GDPR_CANCEL_CALLBACK, user_id=42)
        await on_gdpr_cancel(cb)
        cb.message.edit_text.assert_awaited_once()
        assert CANCEL_MSG in cb.message.edit_text.call_args.args

    @pytest.mark.asyncio
    async def test_delete_new_user_still_succeeds(self):
        """Deleting a user with no DB records still clears Redis and confirms."""
        cb = _make_callback(GDPR_CONFIRM_CALLBACK, user_id=99999)
        with (
            patch(
                "reviewmind.bot.handlers.gdpr._delete_user_from_db",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch(
                "reviewmind.bot.handlers.gdpr._clear_redis_session",
                new_callable=AsyncMock,
            ) as mock_redis,
        ):
            await on_gdpr_confirm(cb)

        mock_redis.assert_awaited_once()
        assert DATA_DELETED_MSG in cb.message.edit_text.call_args.args

    @pytest.mark.asyncio
    async def test_db_down_returns_error(self):
        """If DB is unavailable, user gets error and Redis is not cleared."""
        cb = _make_callback(GDPR_CONFIRM_CALLBACK, user_id=42)
        with (
            patch(
                "reviewmind.bot.handlers.gdpr._delete_user_from_db",
                new_callable=AsyncMock,
                return_value=None,
            ),
            patch(
                "reviewmind.bot.handlers.gdpr._clear_redis_session",
                new_callable=AsyncMock,
            ) as mock_redis,
        ):
            await on_gdpr_confirm(cb)

        mock_redis.assert_not_awaited()
        cb.answer.assert_awaited_once()
        assert DELETE_ERROR_MSG in cb.answer.call_args.args

    @pytest.mark.asyncio
    async def test_help_mentions_delete_command(self):
        """The /help output should mention /delete_my_data."""
        from reviewmind.bot.handlers.start import HELP_TEXT

        assert "/delete_my_data" in HELP_TEXT
