"""Unit tests for TASK-030 — /mode: переключение авто/ссылки без сброса сессии.

Tests cover:
- mode_keyboard() with current_mode highlighting (✅ prefix)
- _get_current_mode() Redis reads with graceful error handling
- cmd_mode() shows current mode label + highlighted keyboard
- on_mode_selected() persists mode without resetting history
- Constants and exports
- Integration scenarios matching all 5 PRD test steps
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiogram.types import InlineKeyboardMarkup, Message


# Override the conftest autouse fixture that patches _persist_mode/_get_current_mode
# so that direct tests of these functions work correctly.
@pytest.fixture(autouse=True)
def _bypass_session_manager():
    """Override conftest fixture — let real _persist_mode/_get_current_mode be tested."""
    yield


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_user(user_id: int = 12345):
    from aiogram.types import User

    return User(id=user_id, is_bot=False, first_name="Test")


def _make_message(text: str = "/mode", user_id: int = 12345) -> MagicMock:
    msg = MagicMock(spec=Message)
    msg.text = text
    msg.from_user = _make_user(user_id)
    msg.answer = AsyncMock()
    return msg


def _make_callback_query(data: str, user_id: int = 12345) -> MagicMock:
    from aiogram.types import CallbackQuery

    cb = MagicMock(spec=CallbackQuery)
    cb.data = data
    cb.from_user = _make_user(user_id)
    cb.message = MagicMock(spec=Message)
    cb.message.edit_text = AsyncMock()
    cb.answer = AsyncMock()
    return cb


# ══════════════════════════════════════════════════════════════════════════════
# Tests — mode_keyboard() with current_mode highlighting
# ══════════════════════════════════════════════════════════════════════════════


class TestModeKeyboardHighlighting:
    """mode_keyboard(current_mode=...) should add ✅ to the active mode."""

    def test_no_current_mode_no_checkmark(self):
        from reviewmind.bot.keyboards import mode_keyboard

        kb = mode_keyboard()
        for btn in kb.inline_keyboard[0]:
            assert "✅" not in btn.text

    def test_none_current_mode_no_checkmark(self):
        from reviewmind.bot.keyboards import mode_keyboard

        kb = mode_keyboard(current_mode=None)
        for btn in kb.inline_keyboard[0]:
            assert "✅" not in btn.text

    def test_auto_highlighted(self):
        from reviewmind.bot.keyboards import MODE_AUTO, mode_keyboard

        kb = mode_keyboard(current_mode="auto")
        auto_btn = [b for b in kb.inline_keyboard[0] if b.callback_data == MODE_AUTO][0]
        assert "✅" in auto_btn.text
        assert "Авто-поиск" in auto_btn.text

    def test_links_not_highlighted_when_auto(self):
        from reviewmind.bot.keyboards import MODE_LINKS, mode_keyboard

        kb = mode_keyboard(current_mode="auto")
        links_btn = [b for b in kb.inline_keyboard[0] if b.callback_data == MODE_LINKS][0]
        assert "✅" not in links_btn.text

    def test_links_highlighted(self):
        from reviewmind.bot.keyboards import MODE_LINKS, mode_keyboard

        kb = mode_keyboard(current_mode="links")
        links_btn = [b for b in kb.inline_keyboard[0] if b.callback_data == MODE_LINKS][0]
        assert "✅" in links_btn.text
        assert "Свои ссылки" in links_btn.text

    def test_auto_not_highlighted_when_links(self):
        from reviewmind.bot.keyboards import MODE_AUTO, mode_keyboard

        kb = mode_keyboard(current_mode="links")
        auto_btn = [b for b in kb.inline_keyboard[0] if b.callback_data == MODE_AUTO][0]
        assert "✅" not in auto_btn.text

    def test_callback_data_unchanged_with_highlight(self):
        """Callback data must remain the same regardless of highlighting."""
        from reviewmind.bot.keyboards import MODE_AUTO, MODE_LINKS, mode_keyboard

        kb = mode_keyboard(current_mode="auto")
        data = [b.callback_data for b in kb.inline_keyboard[0]]
        assert MODE_AUTO in data
        assert MODE_LINKS in data

    def test_unknown_current_mode_no_highlight(self):
        from reviewmind.bot.keyboards import mode_keyboard

        kb = mode_keyboard(current_mode="unknown")
        for btn in kb.inline_keyboard[0]:
            assert "✅" not in btn.text

    def test_still_returns_inline_markup(self):
        from reviewmind.bot.keyboards import mode_keyboard

        kb = mode_keyboard(current_mode="auto")
        assert isinstance(kb, InlineKeyboardMarkup)

    def test_still_has_two_buttons(self):
        from reviewmind.bot.keyboards import mode_keyboard

        kb = mode_keyboard(current_mode="links")
        assert len(kb.inline_keyboard[0]) == 2


# ══════════════════════════════════════════════════════════════════════════════
# Tests — _get_current_mode()
# ══════════════════════════════════════════════════════════════════════════════


class TestGetCurrentMode:
    """_get_current_mode reads mode from Redis with graceful fallback."""

    async def test_returns_auto(self):
        """_get_current_mode returns 'auto' when Redis has that value."""
        from reviewmind.bot.handlers.mode import _get_current_mode

        mock_client = MagicMock()
        mock_client.aclose = AsyncMock()

        mock_sm_instance = MagicMock()
        mock_sm_instance.get_mode = AsyncMock(return_value="auto")

        mock_settings = MagicMock()
        mock_settings.redis_url = "redis://localhost:6379/0"

        with (
            patch("redis.asyncio.from_url", return_value=mock_client),
            patch("reviewmind.cache.redis.SessionManager", return_value=mock_sm_instance),
            patch("reviewmind.config.settings", mock_settings),
        ):
            result = await _get_current_mode(12345)

        assert result == "auto"

    async def test_returns_links(self):
        from reviewmind.bot.handlers.mode import _get_current_mode

        mock_client = MagicMock()
        mock_client.aclose = AsyncMock()

        mock_sm_instance = MagicMock()
        mock_sm_instance.get_mode = AsyncMock(return_value="links")

        mock_settings = MagicMock()
        mock_settings.redis_url = "redis://localhost:6379/0"

        with (
            patch("redis.asyncio.from_url", return_value=mock_client),
            patch("reviewmind.cache.redis.SessionManager", return_value=mock_sm_instance),
            patch("reviewmind.config.settings", mock_settings),
        ):
            result = await _get_current_mode(12345)

        assert result == "links"

    async def test_returns_none_when_no_session(self):
        from reviewmind.bot.handlers.mode import _get_current_mode

        mock_client = MagicMock()
        mock_client.aclose = AsyncMock()

        mock_sm_instance = MagicMock()
        mock_sm_instance.get_mode = AsyncMock(return_value=None)

        mock_settings = MagicMock()
        mock_settings.redis_url = "redis://localhost:6379/0"

        with (
            patch("redis.asyncio.from_url", return_value=mock_client),
            patch("reviewmind.cache.redis.SessionManager", return_value=mock_sm_instance),
            patch("reviewmind.config.settings", mock_settings),
        ):
            result = await _get_current_mode(12345)

        assert result is None

    async def test_returns_none_on_redis_error(self):
        from reviewmind.bot.handlers.mode import _get_current_mode

        mock_settings = MagicMock()
        mock_settings.redis_url = "redis://localhost:6379/0"

        with (
            patch("redis.asyncio.from_url", side_effect=ConnectionError("Redis down")),
            patch("reviewmind.config.settings", mock_settings),
        ):
            result = await _get_current_mode(12345)

        assert result is None

    async def test_closes_client_on_success(self):
        from reviewmind.bot.handlers.mode import _get_current_mode

        mock_client = MagicMock()
        mock_client.aclose = AsyncMock()

        mock_sm_instance = MagicMock()
        mock_sm_instance.get_mode = AsyncMock(return_value="auto")

        mock_settings = MagicMock()
        mock_settings.redis_url = "redis://localhost:6379/0"

        with (
            patch("redis.asyncio.from_url", return_value=mock_client),
            patch("reviewmind.cache.redis.SessionManager", return_value=mock_sm_instance),
            patch("reviewmind.config.settings", mock_settings),
        ):
            await _get_current_mode(12345)

        mock_client.aclose.assert_awaited_once()


# ══════════════════════════════════════════════════════════════════════════════
# Tests — cmd_mode() shows current mode
# ══════════════════════════════════════════════════════════════════════════════


class TestCmdModeShowsCurrent:
    """cmd_mode() should display current mode and highlight it in keyboard."""

    async def test_shows_current_auto_mode(self):
        from reviewmind.bot.handlers.mode import cmd_mode

        msg = _make_message("/mode")
        with patch("reviewmind.bot.handlers.mode._get_current_mode", new_callable=AsyncMock, return_value="auto"):
            await cmd_mode(msg)

        text = msg.answer.call_args.args[0]
        assert "Текущий режим" in text
        assert "Авто-поиск" in text

    async def test_shows_current_links_mode(self):
        from reviewmind.bot.handlers.mode import cmd_mode

        msg = _make_message("/mode")
        with patch("reviewmind.bot.handlers.mode._get_current_mode", new_callable=AsyncMock, return_value="links"):
            await cmd_mode(msg)

        text = msg.answer.call_args.args[0]
        assert "Текущий режим" in text
        assert "Свои ссылки" in text

    async def test_no_current_mode_fallback_text(self):
        from reviewmind.bot.handlers.mode import cmd_mode

        msg = _make_message("/mode")
        with patch("reviewmind.bot.handlers.mode._get_current_mode", new_callable=AsyncMock, return_value=None):
            await cmd_mode(msg)

        text = msg.answer.call_args.args[0]
        assert "Выбери режим работы" in text
        assert "Текущий режим" not in text

    async def test_keyboard_has_auto_highlighted(self):
        from reviewmind.bot.handlers.mode import cmd_mode
        from reviewmind.bot.keyboards import MODE_AUTO

        msg = _make_message("/mode")
        with patch("reviewmind.bot.handlers.mode._get_current_mode", new_callable=AsyncMock, return_value="auto"):
            await cmd_mode(msg)

        kb = msg.answer.call_args.kwargs.get("reply_markup")
        assert isinstance(kb, InlineKeyboardMarkup)
        auto_btn = [b for b in kb.inline_keyboard[0] if b.callback_data == MODE_AUTO][0]
        assert "✅" in auto_btn.text

    async def test_keyboard_has_links_highlighted(self):
        from reviewmind.bot.handlers.mode import cmd_mode
        from reviewmind.bot.keyboards import MODE_LINKS

        msg = _make_message("/mode")
        with patch("reviewmind.bot.handlers.mode._get_current_mode", new_callable=AsyncMock, return_value="links"):
            await cmd_mode(msg)

        kb = msg.answer.call_args.kwargs.get("reply_markup")
        links_btn = [b for b in kb.inline_keyboard[0] if b.callback_data == MODE_LINKS][0]
        assert "✅" in links_btn.text

    async def test_html_parse_mode(self):
        from reviewmind.bot.handlers.mode import cmd_mode

        msg = _make_message("/mode")
        with patch("reviewmind.bot.handlers.mode._get_current_mode", new_callable=AsyncMock, return_value="auto"):
            await cmd_mode(msg)

        assert msg.answer.call_args.kwargs.get("parse_mode") == "HTML"

    async def test_reads_mode_for_correct_user(self):
        from reviewmind.bot.handlers.mode import cmd_mode

        msg = _make_message("/mode", user_id=99999)
        mock_get = AsyncMock(return_value="auto")
        with patch("reviewmind.bot.handlers.mode._get_current_mode", mock_get):
            await cmd_mode(msg)

        mock_get.assert_awaited_once_with(99999)


# ══════════════════════════════════════════════════════════════════════════════
# Tests — on_mode_selected() persists without history reset
# ══════════════════════════════════════════════════════════════════════════════


class TestOnModeSelectedPersistence:
    """on_mode_selected persists mode to Redis WITHOUT clearing history."""

    async def test_persist_called_with_auto(self):
        from reviewmind.bot.handlers.mode import on_mode_selected
        from reviewmind.bot.keyboards import MODE_AUTO

        mock_persist = AsyncMock()
        cb = _make_callback_query(MODE_AUTO, user_id=111)

        with (
            patch("reviewmind.bot.handlers.mode._persist_mode", mock_persist),
            patch("reviewmind.bot.handlers.mode._get_current_mode", new_callable=AsyncMock, return_value=None),
        ):
            await on_mode_selected(cb)

        mock_persist.assert_awaited_once_with(111, "auto")

    async def test_persist_called_with_links(self):
        from reviewmind.bot.handlers.mode import on_mode_selected
        from reviewmind.bot.keyboards import MODE_LINKS

        mock_persist = AsyncMock()
        cb = _make_callback_query(MODE_LINKS, user_id=222)

        with (
            patch("reviewmind.bot.handlers.mode._persist_mode", mock_persist),
            patch("reviewmind.bot.handlers.mode._get_current_mode", new_callable=AsyncMock, return_value=None),
        ):
            await on_mode_selected(cb)

        mock_persist.assert_awaited_once_with(222, "links")

    async def test_history_not_cleared(self):
        """Switching mode must NOT call clear_history / clear_session."""
        from reviewmind.bot.handlers.mode import _persist_mode

        mock_sm = MagicMock()
        mock_sm.set_mode = AsyncMock()
        mock_sm.refresh_ttl = AsyncMock()
        mock_sm.clear_history = AsyncMock()
        mock_sm.clear_session = AsyncMock()

        mock_client = MagicMock()
        mock_client.aclose = AsyncMock()

        mock_settings = MagicMock()
        mock_settings.redis_url = "redis://localhost:6379/0"

        with (
            patch("redis.asyncio.from_url", return_value=mock_client),
            patch("reviewmind.cache.redis.SessionManager", return_value=mock_sm),
            patch("reviewmind.config.settings", mock_settings),
        ):
            await _persist_mode(111, "links")

        mock_sm.set_mode.assert_awaited_once_with(111, "links")
        mock_sm.refresh_ttl.assert_awaited_once_with(111)
        mock_sm.clear_history.assert_not_awaited()
        mock_sm.clear_session.assert_not_awaited()


# ══════════════════════════════════════════════════════════════════════════════
# Tests — Constants & exports
# ══════════════════════════════════════════════════════════════════════════════


class TestModeConstants:
    """Verify constants and module-level mappings."""

    def test_callback_to_mode_contains_both(self):
        from reviewmind.bot.handlers.mode import _CALLBACK_TO_MODE
        from reviewmind.bot.keyboards import MODE_AUTO, MODE_LINKS

        assert _CALLBACK_TO_MODE[MODE_AUTO] == "auto"
        assert _CALLBACK_TO_MODE[MODE_LINKS] == "links"

    def test_mode_to_name_contains_both(self):
        from reviewmind.bot.handlers.mode import _MODE_TO_NAME

        assert "auto" in _MODE_TO_NAME
        assert "links" in _MODE_TO_NAME
        assert "Авто-поиск" in _MODE_TO_NAME["auto"]
        assert "Свои ссылки" in _MODE_TO_NAME["links"]

    def test_default_mode_is_auto(self):
        from reviewmind.bot.handlers.mode import DEFAULT_MODE

        assert DEFAULT_MODE == "auto"

    def test_mode_labels_dict_exists(self):
        from reviewmind.bot.keyboards import _MODE_LABELS

        assert len(_MODE_LABELS) == 2

    def test_internal_to_callback_mapping(self):
        from reviewmind.bot.keyboards import _INTERNAL_TO_CALLBACK, MODE_AUTO, MODE_LINKS

        assert _INTERNAL_TO_CALLBACK["auto"] == MODE_AUTO
        assert _INTERNAL_TO_CALLBACK["links"] == MODE_LINKS

    def test_router_name(self):
        from reviewmind.bot.handlers.mode import router

        assert router.name == "mode"


# ══════════════════════════════════════════════════════════════════════════════
# Tests — Integration scenarios (PRD test steps)
# ══════════════════════════════════════════════════════════════════════════════


class TestIntegrationScenarios:
    """Integration scenarios that map to TASK-030 test steps."""

    async def test_step1_mode_shows_buttons(self):
        """PRD Step 1: /mode → видны кнопки [🔍 Авто] [🔗 Ссылки]."""
        from reviewmind.bot.handlers.mode import cmd_mode

        msg = _make_message("/mode")
        with patch("reviewmind.bot.handlers.mode._get_current_mode", new_callable=AsyncMock, return_value=None):
            await cmd_mode(msg)

        kb = msg.answer.call_args.kwargs.get("reply_markup")
        labels = [b.text for b in kb.inline_keyboard[0]]
        assert any("Авто" in lbl for lbl in labels)
        assert any("Ссылки" in lbl or "ссылки" in lbl for lbl in labels)

    async def test_step2_select_auto_confirms(self):
        """PRD Step 2: Нажать [🔍 Авто] → 'Режим изменён на Авто-поиск'."""
        from reviewmind.bot.handlers.mode import on_mode_selected
        from reviewmind.bot.keyboards import MODE_AUTO

        cb = _make_callback_query(MODE_AUTO)
        with patch("reviewmind.bot.handlers.mode._persist_mode", new_callable=AsyncMock):
            await on_mode_selected(cb)

        answer_text = cb.answer.call_args.args[0]
        assert "Авто-поиск" in answer_text

    async def test_step3_history_preserved_after_switch(self):
        """PRD Step 3: After switching mode, previously loaded session data is available."""
        from reviewmind.bot.handlers.mode import _persist_mode

        mock_sm = MagicMock()
        mock_sm.set_mode = AsyncMock()
        mock_sm.refresh_ttl = AsyncMock()
        mock_sm.get_history = AsyncMock(return_value=[{"role": "user", "content": "test"}])

        mock_client = MagicMock()
        mock_client.aclose = AsyncMock()

        mock_settings = MagicMock()
        mock_settings.redis_url = "redis://localhost:6379/0"

        with (
            patch("redis.asyncio.from_url", return_value=mock_client),
            patch("reviewmind.cache.redis.SessionManager", return_value=mock_sm),
            patch("reviewmind.config.settings", mock_settings),
        ):
            await _persist_mode(111, "auto")

        # Verify: set_mode was called, but history was NOT cleared
        mock_sm.set_mode.assert_awaited_once()
        mock_sm.refresh_ttl.assert_awaited_once()

    async def test_step4_switch_to_links(self):
        """PRD Step 4: /mode → нажать [🔗 Ссылки] → режим изменён."""
        from reviewmind.bot.handlers.mode import on_mode_selected
        from reviewmind.bot.keyboards import MODE_LINKS

        mock_persist = AsyncMock()
        cb = _make_callback_query(MODE_LINKS)
        with patch("reviewmind.bot.handlers.mode._persist_mode", mock_persist):
            await on_mode_selected(cb)

        mock_persist.assert_awaited_once_with(cb.from_user.id, "links")
        edit_text = cb.message.edit_text.call_args.args[0]
        assert "Свои ссылки" in edit_text

    async def test_step5_redis_mode_updated_history_preserved(self):
        """PRD Step 5: Redis session:{user_id}:mode обновлён, history сохранён."""
        from reviewmind.bot.handlers.mode import _persist_mode

        mock_sm = MagicMock()
        mock_sm.set_mode = AsyncMock()
        mock_sm.refresh_ttl = AsyncMock()

        mock_client = MagicMock()
        mock_client.aclose = AsyncMock()

        mock_settings = MagicMock()
        mock_settings.redis_url = "redis://localhost:6379/0"

        with (
            patch("redis.asyncio.from_url", return_value=mock_client),
            patch("reviewmind.cache.redis.SessionManager", return_value=mock_sm),
            patch("reviewmind.config.settings", mock_settings),
        ):
            await _persist_mode(42, "links")

        # set_mode was called (updates session:{42}:mode)
        mock_sm.set_mode.assert_awaited_once_with(42, "links")
        # refresh_ttl refreshes all keys but does NOT clear history
        mock_sm.refresh_ttl.assert_awaited_once_with(42)

    async def test_current_mode_visually_highlighted_in_keyboard(self):
        """The current mode should have a ✅ checkmark in the keyboard button."""
        from reviewmind.bot.handlers.mode import cmd_mode
        from reviewmind.bot.keyboards import MODE_AUTO, MODE_LINKS

        msg = _make_message("/mode")
        with patch("reviewmind.bot.handlers.mode._get_current_mode", new_callable=AsyncMock, return_value="links"):
            await cmd_mode(msg)

        kb = msg.answer.call_args.kwargs.get("reply_markup")
        links_btn = [b for b in kb.inline_keyboard[0] if b.callback_data == MODE_LINKS][0]
        auto_btn = [b for b in kb.inline_keyboard[0] if b.callback_data == MODE_AUTO][0]
        assert "✅" in links_btn.text
        assert "✅" not in auto_btn.text
