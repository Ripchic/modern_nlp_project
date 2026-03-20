"""reviewmind/bot/keyboards.py — Inline-клавиатуры для Telegram-бота."""

from __future__ import annotations

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

# ── Callback data constants ──────────────────────────────────
MODE_AUTO = "mode:auto"
MODE_LINKS = "mode:links"
FEEDBACK_USEFUL = "feedback:useful"
FEEDBACK_BAD = "feedback:bad"
FEEDBACK_SOURCES = "feedback:sources"
SUBSCRIBE_ACTION = "subscribe:start"
SUBSCRIBE_WAIT = "subscribe:wait"

# Callback data prefixes used by the feedback handler to parse query_log_id
FEEDBACK_PREFIX = "feedback:"


_MODE_LABELS: dict[str, str] = {
    MODE_AUTO: "🔍 Авто-поиск",
    MODE_LINKS: "🔗 Свои ссылки",
}

# Internal mode string → callback constant
_INTERNAL_TO_CALLBACK: dict[str, str] = {
    "auto": MODE_AUTO,
    "links": MODE_LINKS,
}


def mode_keyboard(current_mode: str | None = None) -> InlineKeyboardMarkup:
    """Inline keyboard for choosing analysis mode (auto-search or manual links).

    Parameters
    ----------
    current_mode:
        Internal mode string (``"auto"`` or ``"links"``).  When provided, the
        active mode button gets a ``✅`` prefix so the user can see which mode
        is currently selected.
    """
    active_callback = _INTERNAL_TO_CALLBACK.get(current_mode or "")
    buttons: list[InlineKeyboardButton] = []
    for cb_data, label in _MODE_LABELS.items():
        text = f"✅ {label}" if cb_data == active_callback else label
        buttons.append(InlineKeyboardButton(text=text, callback_data=cb_data))
    return InlineKeyboardMarkup(inline_keyboard=[buttons])


def feedback_keyboard(query_log_id: int | None = None) -> InlineKeyboardMarkup:
    """Inline keyboard for feedback on bot responses.

    When *query_log_id* is provided, the ID is encoded into each callback
    value so the feedback handler can update the exact ``query_logs`` row.
    Format: ``feedback:useful:123`` / ``feedback:bad:123`` / ``feedback:sources:123``.
    """
    suffix = f":{query_log_id}" if query_log_id is not None else ""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="👍 Полезно", callback_data=f"{FEEDBACK_USEFUL}{suffix}"),
                InlineKeyboardButton(text="👎 Не то", callback_data=f"{FEEDBACK_BAD}{suffix}"),
                InlineKeyboardButton(text="📎 Источники", callback_data=f"{FEEDBACK_SOURCES}{suffix}"),
            ],
        ]
    )


def subscribe_keyboard() -> InlineKeyboardMarkup:
    """Inline keyboard shown when the free query limit is exhausted."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="⭐ Безлимит за 299₽/мес", callback_data=SUBSCRIBE_ACTION),
                InlineKeyboardButton(text="Ждать до завтра", callback_data=SUBSCRIBE_WAIT),
            ],
        ]
    )
