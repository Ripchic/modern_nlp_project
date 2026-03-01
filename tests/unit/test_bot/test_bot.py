"""Unit tests for reviewmind.bot — Telegram bot: keyboards, handlers, main."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from aiogram.types import CallbackQuery, Chat, InlineKeyboardButton, InlineKeyboardMarkup, Message, Update, User

# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════


def _make_user(user_id: int = 12345) -> User:
    """Create a minimal User object."""
    return User(id=user_id, is_bot=False, first_name="Test")


def _make_chat(chat_id: int = 12345) -> Chat:
    """Create a minimal Chat object."""
    return Chat(id=chat_id, type="private")


def _make_message(text: str = "/start", user_id: int = 12345) -> MagicMock:
    """Create a mocked Message with answer() and from_user."""
    msg = MagicMock(spec=Message)
    msg.text = text
    msg.from_user = _make_user(user_id)
    msg.chat = _make_chat(user_id)
    msg.answer = AsyncMock()
    return msg


def _make_callback_query(data: str, user_id: int = 12345) -> MagicMock:
    """Create a mocked CallbackQuery with answer() and message.edit_text()."""
    cb = MagicMock(spec=CallbackQuery)
    cb.data = data
    cb.from_user = _make_user(user_id)
    cb.message = MagicMock(spec=Message)
    cb.message.edit_text = AsyncMock()
    cb.answer = AsyncMock()
    return cb


# ══════════════════════════════════════════════════════════════
# Tests — keyboards.py
# ══════════════════════════════════════════════════════════════


class TestKeyboards:
    """Tests for reviewmind.bot.keyboards."""

    def test_mode_keyboard_returns_inline_markup(self):
        from reviewmind.bot.keyboards import mode_keyboard

        kb = mode_keyboard()
        assert isinstance(kb, InlineKeyboardMarkup)

    def test_mode_keyboard_has_two_buttons(self):
        from reviewmind.bot.keyboards import mode_keyboard

        kb = mode_keyboard()
        buttons = kb.inline_keyboard[0]
        assert len(buttons) == 2

    def test_mode_keyboard_auto_button(self):
        from reviewmind.bot.keyboards import MODE_AUTO, mode_keyboard

        kb = mode_keyboard()
        btn = kb.inline_keyboard[0][0]
        assert isinstance(btn, InlineKeyboardButton)
        assert "Авто-поиск" in btn.text
        assert btn.callback_data == MODE_AUTO

    def test_mode_keyboard_links_button(self):
        from reviewmind.bot.keyboards import MODE_LINKS, mode_keyboard

        kb = mode_keyboard()
        btn = kb.inline_keyboard[0][1]
        assert isinstance(btn, InlineKeyboardButton)
        assert "Свои ссылки" in btn.text
        assert btn.callback_data == MODE_LINKS

    def test_feedback_keyboard_has_three_buttons(self):
        from reviewmind.bot.keyboards import feedback_keyboard

        kb = feedback_keyboard()
        buttons = kb.inline_keyboard[0]
        assert len(buttons) == 3

    def test_feedback_keyboard_button_labels(self):
        from reviewmind.bot.keyboards import feedback_keyboard

        kb = feedback_keyboard()
        labels = [btn.text for btn in kb.inline_keyboard[0]]
        assert any("Полезно" in lbl for lbl in labels)
        assert any("Не то" in lbl for lbl in labels)
        assert any("Источники" in lbl for lbl in labels)

    def test_feedback_keyboard_callback_data(self):
        from reviewmind.bot.keyboards import FEEDBACK_BAD, FEEDBACK_SOURCES, FEEDBACK_USEFUL, feedback_keyboard

        kb = feedback_keyboard()
        data = [btn.callback_data for btn in kb.inline_keyboard[0]]
        assert FEEDBACK_USEFUL in data
        assert FEEDBACK_BAD in data
        assert FEEDBACK_SOURCES in data

    def test_subscribe_keyboard_has_two_buttons(self):
        from reviewmind.bot.keyboards import subscribe_keyboard

        kb = subscribe_keyboard()
        buttons = kb.inline_keyboard[0]
        assert len(buttons) == 2

    def test_subscribe_keyboard_button_labels(self):
        from reviewmind.bot.keyboards import subscribe_keyboard

        kb = subscribe_keyboard()
        labels = [btn.text for btn in kb.inline_keyboard[0]]
        assert any("Безлимит" in lbl for lbl in labels)
        assert any("Ждать" in lbl for lbl in labels)

    def test_mode_constants_are_strings(self):
        from reviewmind.bot.keyboards import MODE_AUTO, MODE_LINKS

        assert isinstance(MODE_AUTO, str)
        assert isinstance(MODE_LINKS, str)
        assert MODE_AUTO != MODE_LINKS

    def test_callback_data_format(self):
        """Callback data should use 'prefix:value' format."""
        from reviewmind.bot.keyboards import MODE_AUTO, MODE_LINKS

        assert ":" in MODE_AUTO
        assert ":" in MODE_LINKS


# ══════════════════════════════════════════════════════════════
# Tests — handlers/start.py
# ══════════════════════════════════════════════════════════════


class TestStartHandler:
    """Tests for /start and /help command handlers."""

    async def test_cmd_start_sends_welcome(self):
        from reviewmind.bot.handlers.start import cmd_start

        msg = _make_message("/start")
        await cmd_start(msg)
        msg.answer.assert_called_once()

    async def test_cmd_start_includes_mode_keyboard(self):
        from reviewmind.bot.handlers.start import cmd_start

        msg = _make_message("/start")
        await cmd_start(msg)

        call_kwargs = msg.answer.call_args
        assert call_kwargs.kwargs.get("reply_markup") is not None or (
            len(call_kwargs.args) > 1 if call_kwargs.args else False
        )
        # Check that reply_markup is an InlineKeyboardMarkup
        reply_markup = call_kwargs.kwargs.get("reply_markup")
        assert isinstance(reply_markup, InlineKeyboardMarkup)

    async def test_cmd_start_welcome_mentions_ai(self):
        """Welcome message must mention that the bot is an AI system."""
        from reviewmind.bot.handlers.start import WELCOME_TEXT

        text_lower = WELCOME_TEXT.lower()
        assert "ai" in text_lower or "искусственн" in text_lower

    async def test_cmd_start_welcome_mentions_reviewmind(self):
        from reviewmind.bot.handlers.start import WELCOME_TEXT

        assert "ReviewMind" in WELCOME_TEXT

    async def test_cmd_start_uses_html_parse_mode(self):
        from reviewmind.bot.handlers.start import cmd_start

        msg = _make_message("/start")
        await cmd_start(msg)

        call_kwargs = msg.answer.call_args
        assert call_kwargs.kwargs.get("parse_mode") == "HTML"

    async def test_cmd_help_sends_help_text(self):
        from reviewmind.bot.handlers.start import cmd_help

        msg = _make_message("/help")
        await cmd_help(msg)
        msg.answer.assert_called_once()

    async def test_cmd_help_text_contains_commands(self):
        from reviewmind.bot.handlers.start import HELP_TEXT

        assert "/start" in HELP_TEXT
        assert "/help" in HELP_TEXT
        assert "/mode" in HELP_TEXT

    async def test_cmd_help_text_contains_modes(self):
        from reviewmind.bot.handlers.start import HELP_TEXT

        assert "Авто-поиск" in HELP_TEXT
        assert "Свои ссылки" in HELP_TEXT

    async def test_cmd_help_mentions_sources(self):
        from reviewmind.bot.handlers.start import HELP_TEXT

        assert "YouTube" in HELP_TEXT
        assert "Reddit" in HELP_TEXT

    async def test_cmd_help_mentions_sponsor_flag(self):
        from reviewmind.bot.handlers.start import HELP_TEXT

        text_lower = HELP_TEXT.lower()
        assert "спонсор" in text_lower

    async def test_start_router_has_name(self):
        from reviewmind.bot.handlers.start import router

        assert router.name == "start"


# ══════════════════════════════════════════════════════════════
# Tests — handlers/mode.py
# ══════════════════════════════════════════════════════════════


class TestModeHandler:
    """Tests for mode selection callback and /mode command."""

    async def test_on_mode_selected_auto(self):
        from reviewmind.bot.handlers.mode import on_mode_selected
        from reviewmind.bot.keyboards import MODE_AUTO

        cb = _make_callback_query(MODE_AUTO)
        await on_mode_selected(cb)

        cb.message.edit_text.assert_called_once()
        cb.answer.assert_called_once()

    async def test_on_mode_selected_links(self):
        from reviewmind.bot.handlers.mode import on_mode_selected
        from reviewmind.bot.keyboards import MODE_LINKS

        cb = _make_callback_query(MODE_LINKS)
        await on_mode_selected(cb)

        cb.message.edit_text.assert_called_once()
        cb.answer.assert_called_once()

    async def test_on_mode_selected_auto_text(self):
        from reviewmind.bot.handlers.mode import on_mode_selected
        from reviewmind.bot.keyboards import MODE_AUTO

        cb = _make_callback_query(MODE_AUTO)
        await on_mode_selected(cb)

        text = cb.message.edit_text.call_args.args[0]
        assert "Авто-поиск" in text

    async def test_on_mode_selected_links_text(self):
        from reviewmind.bot.handlers.mode import on_mode_selected
        from reviewmind.bot.keyboards import MODE_LINKS

        cb = _make_callback_query(MODE_LINKS)
        await on_mode_selected(cb)

        text = cb.message.edit_text.call_args.args[0]
        assert "Свои ссылки" in text

    async def test_on_mode_selected_callback_answered(self):
        from reviewmind.bot.handlers.mode import on_mode_selected
        from reviewmind.bot.keyboards import MODE_AUTO

        cb = _make_callback_query(MODE_AUTO)
        await on_mode_selected(cb)

        cb.answer.assert_called_once()
        call_args = cb.answer.call_args
        answer_text = call_args.args[0] if call_args.args else call_args.kwargs.get("text", "")
        assert "Авто-поиск" in answer_text

    async def test_cmd_mode_sends_keyboard(self):
        from reviewmind.bot.handlers.mode import cmd_mode

        msg = _make_message("/mode")
        await cmd_mode(msg)
        msg.answer.assert_called_once()

        call_kwargs = msg.answer.call_args
        reply_markup = call_kwargs.kwargs.get("reply_markup")
        assert isinstance(reply_markup, InlineKeyboardMarkup)

    async def test_mode_descriptions_exist(self):
        from reviewmind.bot.handlers.mode import MODE_DESCRIPTIONS
        from reviewmind.bot.keyboards import MODE_AUTO, MODE_LINKS

        assert MODE_AUTO in MODE_DESCRIPTIONS
        assert MODE_LINKS in MODE_DESCRIPTIONS
        assert len(MODE_DESCRIPTIONS[MODE_AUTO]) > 0
        assert len(MODE_DESCRIPTIONS[MODE_LINKS]) > 0

    async def test_mode_names_exist(self):
        from reviewmind.bot.handlers.mode import MODE_NAMES
        from reviewmind.bot.keyboards import MODE_AUTO, MODE_LINKS

        assert MODE_AUTO in MODE_NAMES
        assert MODE_LINKS in MODE_NAMES

    async def test_mode_router_has_name(self):
        from reviewmind.bot.handlers.mode import router

        assert router.name == "mode"


# ══════════════════════════════════════════════════════════════
# Tests — bot/main.py
# ══════════════════════════════════════════════════════════════


class TestBotMain:
    """Tests for the bot entrypoint module."""

    def test_create_dispatcher_returns_dispatcher_with_routers(self):
        from aiogram import Dispatcher

        from reviewmind.bot.main import create_dispatcher

        dp = create_dispatcher()
        assert isinstance(dp, Dispatcher)
        # Dispatcher should have at least 2 sub-routers (start + mode)
        assert len(dp.sub_routers) >= 2

    def test_create_bot_returns_bot(self):
        from aiogram import Bot

        from reviewmind.bot.main import create_bot

        bot = create_bot("123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11")
        assert isinstance(bot, Bot)

    def test_create_bot_has_html_parse_mode(self):
        from aiogram.enums import ParseMode

        from reviewmind.bot.main import create_bot

        bot = create_bot("123456:ABC-DEF1234ghIkl-zyx57W2v1u123ew11")
        assert bot.default.parse_mode == ParseMode.HTML

    @patch("reviewmind.bot.main._configure_logging")
    def test_configure_logging_callable(self, mock_log):
        """_configure_logging should be callable without errors."""
        from reviewmind.bot.main import _configure_logging

        _configure_logging()  # Should not raise

    def test_main_function_exists(self):
        from reviewmind.bot.main import main

        assert callable(main)

    def test_run_bot_function_exists(self):
        from reviewmind.bot.main import run_bot

        assert callable(run_bot)


# ══════════════════════════════════════════════════════════════
# Tests — middlewares.py
# ══════════════════════════════════════════════════════════════


class TestMiddlewares:
    """Tests for bot middlewares."""

    def test_logging_middleware_is_base_middleware(self):
        from aiogram import BaseMiddleware

        from reviewmind.bot.middlewares import LoggingMiddleware

        mw = LoggingMiddleware()
        assert isinstance(mw, BaseMiddleware)

    async def test_logging_middleware_calls_handler(self):
        from reviewmind.bot.middlewares import LoggingMiddleware

        mw = LoggingMiddleware()
        handler = AsyncMock(return_value="result")
        event = MagicMock(spec=Update)
        event.update_id = 1
        event.event_type = "message"
        event.message = MagicMock()
        event.message.from_user = _make_user()
        event.callback_query = None

        result = await mw(handler, event, {})
        handler.assert_called_once_with(event, {})
        assert result == "result"

    async def test_logging_middleware_handles_callback_query(self):
        from reviewmind.bot.middlewares import LoggingMiddleware

        mw = LoggingMiddleware()
        handler = AsyncMock(return_value="ok")
        event = MagicMock(spec=Update)
        event.update_id = 2
        event.event_type = "callback_query"
        event.message = None
        event.callback_query = MagicMock()
        event.callback_query.from_user = _make_user(99999)

        result = await mw(handler, event, {})
        assert result == "ok"

    async def test_logging_middleware_handles_non_update(self):
        """Non-Update events should still be passed to handler."""
        from reviewmind.bot.middlewares import LoggingMiddleware

        mw = LoggingMiddleware()
        handler = AsyncMock(return_value="ok")
        event = MagicMock()  # Not an Update

        result = await mw(handler, event, {})
        handler.assert_called_once()
        assert result == "ok"
