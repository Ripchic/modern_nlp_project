"""reviewmind/bot/handlers/gdpr.py — /delete_my_data: каскадное удаление данных пользователя.

Flow:
1. User sends /delete_my_data → confirmation prompt with [Да, удалить] [Отмена].
2. User presses [Да, удалить] → cascade delete from PostgreSQL + Redis session clear.
3. User presses [Отмена] → "Данные не затронуты".

PostgreSQL deletion relies on ``ON DELETE CASCADE`` FK constraints:
deleting the ``users`` row cascades to ``user_limits``, ``subscriptions``,
``jobs``, and ``query_logs``.
"""

from __future__ import annotations

import structlog
from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from reviewmind.db.repositories.users import UserRepository

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

router = Router(name="gdpr")

# ── Constants ────────────────────────────────────────────────────────────────

GDPR_CONFIRM_CALLBACK = "gdpr:confirm_delete"
GDPR_CANCEL_CALLBACK = "gdpr:cancel_delete"

CONFIRM_PROMPT_MSG = (
    "⚠️ <b>Вы уверены?</b>\n\n"
    "Это действие удалит <b>ВСЕ</b> ваши данные:\n"
    "• История запросов\n"
    "• Лимиты и подписки\n"
    "• Задачи и сессия\n\n"
    "Отменить удаление будет невозможно."
)

DATA_DELETED_MSG = "✅ Все ваши данные удалены. Вы можете начать заново командой /start."
CANCEL_MSG = "Хорошо, ваши данные не затронуты. 👍"
DELETE_ERROR_MSG = "⚠️ Не удалось удалить данные. Попробуйте позже."
NO_DATA_MSG = "ℹ️ У вас нет сохранённых данных."


# ── Keyboard ─────────────────────────────────────────────────────────────────


def gdpr_confirm_keyboard() -> InlineKeyboardMarkup:
    """Inline keyboard with [Да, удалить] and [Отмена] buttons."""
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="🗑 Да, удалить", callback_data=GDPR_CONFIRM_CALLBACK),
                InlineKeyboardButton(text="Отмена", callback_data=GDPR_CANCEL_CALLBACK),
            ],
        ]
    )


# ── Helpers ──────────────────────────────────────────────────────────────────


async def _get_db_session():
    """Create a short-lived async DB session from config."""
    from reviewmind.config import settings  # noqa: PLC0415

    engine = create_async_engine(settings.database_url, pool_pre_ping=True, echo=False)
    session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    return engine, session_factory


async def _delete_user_from_db(user_id: int) -> bool | None:
    """Delete the user row from PostgreSQL (cascades to all related tables).

    Returns:
        True  — user found and deleted.
        False — user not found (no data to delete).
        None  — DB unavailable or error.
    """
    try:
        engine, session_factory = await _get_db_session()
        try:
            async with session_factory() as session:
                repo = UserRepository(session)
                deleted = await repo.delete(user_id)
                await session.commit()
                logger.info("gdpr_db_deleted", user_id=user_id, deleted=deleted)
                return deleted
        finally:
            await engine.dispose()
    except Exception as exc:
        logger.error("gdpr_db_delete_error", user_id=user_id, error=str(exc))
        return None


async def _clear_redis_session(user_id: int) -> None:
    """Clear all Redis session keys for the user. Best-effort — never raises."""
    try:
        from redis.asyncio import from_url as redis_from_url  # noqa: PLC0415

        from reviewmind.cache.redis import SessionManager  # noqa: PLC0415
        from reviewmind.config import settings  # noqa: PLC0415

        client = redis_from_url(settings.redis_url, decode_responses=True)
        sm = SessionManager(client)
        await sm.clear_session(user_id)
        await client.aclose()
        logger.info("gdpr_redis_cleared", user_id=user_id)
    except Exception as exc:
        logger.warning("gdpr_redis_clear_error", user_id=user_id, error=str(exc))


# ── Handlers ─────────────────────────────────────────────────────────────────


@router.message(Command("delete_my_data"))
async def cmd_delete_my_data(message: Message) -> None:
    """Handle /delete_my_data — show confirmation prompt."""
    user_id = message.from_user.id if message.from_user else 0
    logger.info("gdpr_delete_requested", user_id=user_id)
    await message.answer(CONFIRM_PROMPT_MSG, reply_markup=gdpr_confirm_keyboard(), parse_mode="HTML")


@router.callback_query(F.data == GDPR_CONFIRM_CALLBACK)
async def on_gdpr_confirm(callback: CallbackQuery) -> None:
    """Handle [Да, удалить] — cascade delete all user data."""
    user_id = callback.from_user.id if callback.from_user else 0
    log = logger.bind(user_id=user_id)
    log.info("gdpr_confirm_pressed")

    # 1. Delete from PostgreSQL (cascades to user_limits, subscriptions, jobs, query_logs)
    db_result = await _delete_user_from_db(user_id)

    if db_result is None:
        # DB unavailable
        await callback.answer(DELETE_ERROR_MSG, show_alert=True)
        return

    # 2. Clear Redis session (best-effort)
    await _clear_redis_session(user_id)

    # 3. Inform user
    if db_result:
        await callback.message.edit_text(DATA_DELETED_MSG, parse_mode="HTML")
    else:
        # User not found in DB — still clear Redis and confirm
        await callback.message.edit_text(DATA_DELETED_MSG, parse_mode="HTML")

    await callback.answer()
    log.info("gdpr_delete_completed", had_db_data=db_result)


@router.callback_query(F.data == GDPR_CANCEL_CALLBACK)
async def on_gdpr_cancel(callback: CallbackQuery) -> None:
    """Handle [Отмена] — inform user that data is untouched."""
    user_id = callback.from_user.id if callback.from_user else 0
    logger.info("gdpr_cancel_pressed", user_id=user_id)
    await callback.message.edit_text(CANCEL_MSG, parse_mode="HTML")
    await callback.answer()
