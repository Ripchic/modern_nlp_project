"""reviewmind/bot/handlers/mode.py — Обработка выбора и переключения режима.

Mode is persisted to Redis via :class:`~reviewmind.cache.redis.SessionManager`.
Switching mode does **not** clear the chat history.
"""

import structlog
from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import CallbackQuery, Message

from reviewmind.bot.keyboards import MODE_AUTO, MODE_LINKS, mode_keyboard

logger = structlog.get_logger(__name__)

router = Router(name="mode")

# Map callback data → internal mode string used by SessionManager
_CALLBACK_TO_MODE = {
    MODE_AUTO: "auto",
    MODE_LINKS: "links",
}

# Reverse: internal mode → human-readable label
_MODE_TO_NAME = {
    "auto": "🔍 Авто-поиск",
    "links": "🔗 Свои ссылки",
}

MODE_NAMES = {
    MODE_AUTO: "🔍 Авто-поиск",
    MODE_LINKS: "🔗 Свои ссылки",
}

MODE_DESCRIPTIONS = {
    MODE_AUTO: (
        "🔍 Режим <b>Авто-поиск</b> активирован!\n\n"
        "Просто напиши название товара — я сам найду обзоры "
        "на YouTube, Reddit и экспертных сайтах и сформирую анализ."
    ),
    MODE_LINKS: (
        "🔗 Режим <b>Свои ссылки</b> активирован!\n\n"
        "Отправь мне ссылки на обзоры (YouTube, Reddit или веб-страницы), "
        "и я проанализирую их содержимое."
    ),
}

# Default mode when Redis has no record
DEFAULT_MODE = "auto"


async def _get_current_mode(user_id: int) -> str | None:
    """Read the user's current mode from Redis.  Returns ``None`` on failure."""
    try:
        from redis.asyncio import from_url as redis_from_url  # noqa: PLC0415

        from reviewmind.cache.redis import SessionManager  # noqa: PLC0415
        from reviewmind.config import settings  # noqa: PLC0415

        client = redis_from_url(settings.redis_url, decode_responses=True)
        try:
            sm = SessionManager(client)
            return await sm.get_mode(user_id)
        finally:
            await client.aclose()
    except Exception as exc:
        logger.warning("mode_read_failed", user_id=user_id, error=str(exc))
        return None


async def _persist_mode(user_id: int, mode: str) -> None:
    """Persist the selected mode to Redis.  Best-effort — never raises."""
    try:
        from redis.asyncio import from_url as redis_from_url  # noqa: PLC0415

        from reviewmind.cache.redis import SessionManager  # noqa: PLC0415
        from reviewmind.config import settings  # noqa: PLC0415

        client = redis_from_url(settings.redis_url, decode_responses=True)
        sm = SessionManager(client)
        await sm.set_mode(user_id, mode)
        # Refresh TTL on all session keys (history is NOT cleared)
        await sm.refresh_ttl(user_id)
        await client.aclose()
    except Exception as exc:
        logger.warning("mode_persist_failed", user_id=user_id, mode=mode, error=str(exc))


@router.callback_query(F.data.in_({MODE_AUTO, MODE_LINKS}))
async def on_mode_selected(callback: CallbackQuery) -> None:
    """Handle inline button press for mode selection."""
    mode = callback.data
    description = MODE_DESCRIPTIONS.get(mode, "Режим выбран.")
    internal_mode = _CALLBACK_TO_MODE.get(mode, "auto")

    await callback.message.edit_text(description, parse_mode="HTML")  # type: ignore[union-attr]
    await callback.answer(f"Выбран режим: {MODE_NAMES.get(mode, mode)}")

    # Persist mode to Redis (does NOT clear chat history)
    user_id = callback.from_user.id if callback.from_user else 0
    await _persist_mode(user_id, internal_mode)


@router.message(Command("mode"))
async def cmd_mode(message: Message) -> None:
    """Handle /mode command — show mode selection keyboard with current mode highlighted."""
    user_id = message.from_user.id if message.from_user else 0
    current_mode = await _get_current_mode(user_id)

    current_label = _MODE_TO_NAME.get(current_mode or "", None)
    if current_label:
        text = f"Текущий режим: <b>{current_label}</b>\n\nВыбери режим работы:"
    else:
        text = "Выбери режим работы:"

    await message.answer(
        text,
        reply_markup=mode_keyboard(current_mode=current_mode),
        parse_mode="HTML",
    )
