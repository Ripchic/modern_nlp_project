"""reviewmind/bot/keyboards.py — Inline-клавиатуры для Telegram-бота."""

from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

# ── Callback data constants ──────────────────────────────────
MODE_AUTO = "mode:auto"
MODE_LINKS = "mode:links"
FEEDBACK_USEFUL = "feedback:useful"
FEEDBACK_BAD = "feedback:bad"
FEEDBACK_SOURCES = "feedback:sources"
SUBSCRIBE_ACTION = "subscribe:start"
SUBSCRIBE_WAIT = "subscribe:wait"


def mode_keyboard() -> InlineKeyboardMarkup:
    """Inline keyboard for choosing analysis mode (auto-search or manual links)."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="🔍 Авто-поиск", callback_data=MODE_AUTO),
                InlineKeyboardButton(text="🔗 Свои ссылки", callback_data=MODE_LINKS),
            ],
        ]
    )


def feedback_keyboard() -> InlineKeyboardMarkup:
    """Inline keyboard for feedback on bot responses."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="👍 Полезно", callback_data=FEEDBACK_USEFUL),
                InlineKeyboardButton(text="👎 Не то", callback_data=FEEDBACK_BAD),
                InlineKeyboardButton(text="📎 Источники", callback_data=FEEDBACK_SOURCES),
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
