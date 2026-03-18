"""reviewmind/bot/handlers/feedback.py — 👍/👎 обработка + 📎 Источники.

Handles inline-keyboard callbacks from the feedback buttons placed under
every RAG-generated answer.  Updates the ``query_logs.rating`` column in
PostgreSQL and shows the user a confirmation message.
"""

from __future__ import annotations

import structlog
from aiogram import F, Router
from aiogram.types import CallbackQuery
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from reviewmind.db.repositories.query_logs import QueryLogRepository

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

router = Router(name="feedback")

# ── Constants ────────────────────────────────────────────────────────────────

RATING_USEFUL = 1
RATING_BAD = -1

FEEDBACK_THANKS_MSG = "Спасибо за отзыв! 🙏"
NO_SOURCES_MSG = "Источники не найдены."
FEEDBACK_ERROR_MSG = "⚠️ Не удалось сохранить отзыв. Попробуйте позже."
SOURCES_ERROR_MSG = "⚠️ Не удалось получить источники. Попробуйте позже."

# Badge mapping used when formatting the source list
_CURATED_BADGE = "📚"
_SPONSORED_BADGE = "⚠️ [sponsored]"


# ── Helpers ──────────────────────────────────────────────────────────────────


def parse_query_log_id(callback_data: str) -> int | None:
    """Extract ``query_log_id`` from callback data if present.

    Expected formats:
    - ``feedback:useful``        → None
    - ``feedback:useful:42``     → 42
    - ``feedback:bad:7``         → 7
    - ``feedback:sources:123``   → 123
    """
    parts = callback_data.split(":")
    if len(parts) >= 3:
        try:
            return int(parts[2])
        except (ValueError, IndexError):
            return None
    return None


def format_sources_list(sources: list) -> str:
    """Format a ``query_logs.sources_used`` JSON list for display.

    Each element can be either:
    - a plain string (URL) — displayed as-is.
    - a dict with keys ``url``, optional ``is_curated``, ``is_sponsored`` —
      formatted with badges.
    """
    if not sources:
        return NO_SOURCES_MSG

    lines: list[str] = []
    for idx, item in enumerate(sources, 1):
        if isinstance(item, dict):
            url = item.get("url", item.get("source_url", ""))
            badges: list[str] = []
            if item.get("is_curated"):
                badges.append(_CURATED_BADGE)
            if item.get("is_sponsored"):
                badges.append(_SPONSORED_BADGE)
            prefix = " ".join(badges) + " " if badges else ""
            lines.append(f"{idx}. {prefix}{url}")
        else:
            lines.append(f"{idx}. {item}")

    return "\n".join(lines)


async def _get_db_session():
    """Create a short-lived async DB session from config."""
    from reviewmind.config import settings  # noqa: PLC0415

    engine = create_async_engine(settings.database_url, pool_pre_ping=True, echo=False)
    session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    return engine, session_factory


async def _save_rating(query_log_id: int, rating: int, user_id: int) -> bool:
    """Persist *rating* to the ``query_logs`` row.  Returns True on success."""
    try:
        engine, session_factory = await _get_db_session()
        try:
            async with session_factory() as session:
                repo = QueryLogRepository(session)
                result = await repo.update_rating(query_log_id, rating)
                if result is None:
                    logger.warning("feedback_query_log_not_found", query_log_id=query_log_id, user_id=user_id)
                    return False
                await session.commit()
                logger.info("feedback_saved", query_log_id=query_log_id, rating=rating, user_id=user_id)
                return True
        finally:
            await engine.dispose()
    except Exception as exc:
        logger.error("feedback_save_error", error=str(exc), query_log_id=query_log_id, user_id=user_id)
        return False


async def _get_latest_query_log_id(user_id: int) -> int | None:
    """Return the ID of the most recent query_log for *user_id*."""
    try:
        engine, session_factory = await _get_db_session()
        try:
            async with session_factory() as session:
                repo = QueryLogRepository(session)
                logs = await repo.list_by_user(user_id, limit=1)
                if logs:
                    return logs[0].id
                return None
        finally:
            await engine.dispose()
    except Exception as exc:
        logger.warning("feedback_latest_log_lookup_failed", error=str(exc), user_id=user_id)
        return None


async def _get_sources_for_log(query_log_id: int) -> list | None:
    """Fetch ``sources_used`` from a query_log row.  Returns None on error."""
    try:
        engine, session_factory = await _get_db_session()
        try:
            async with session_factory() as session:
                repo = QueryLogRepository(session)
                log_entry = await repo.get_by_id(query_log_id)
                if log_entry is None:
                    return None
                return log_entry.sources_used
        finally:
            await engine.dispose()
    except Exception as exc:
        logger.warning("feedback_sources_lookup_failed", error=str(exc), query_log_id=query_log_id)
        return None


async def _resolve_query_log_id(callback_data: str, user_id: int) -> int | None:
    """Parse query_log_id from callback data or fall back to the latest log for the user."""
    log_id = parse_query_log_id(callback_data)
    if log_id is not None:
        return log_id
    return await _get_latest_query_log_id(user_id)


# ── Callback handlers ───────────────────────────────────────────────────────


@router.callback_query(F.data.startswith("feedback:useful"))
async def on_feedback_useful(callback: CallbackQuery) -> None:
    """Handle 👍 Полезно button press."""
    user_id = callback.from_user.id if callback.from_user else 0
    log = logger.bind(user_id=user_id, action="useful")
    log.info("feedback_callback_received")

    query_log_id = await _resolve_query_log_id(callback.data or "", user_id)

    if query_log_id is not None:
        saved = await _save_rating(query_log_id, RATING_USEFUL, user_id)
        if saved:
            await callback.answer(FEEDBACK_THANKS_MSG)
        else:
            await callback.answer(FEEDBACK_ERROR_MSG)
    else:
        log.warning("feedback_no_query_log_id")
        await callback.answer(FEEDBACK_THANKS_MSG)


@router.callback_query(F.data.startswith("feedback:bad"))
async def on_feedback_bad(callback: CallbackQuery) -> None:
    """Handle 👎 Не то button press."""
    user_id = callback.from_user.id if callback.from_user else 0
    log = logger.bind(user_id=user_id, action="bad")
    log.info("feedback_callback_received")

    query_log_id = await _resolve_query_log_id(callback.data or "", user_id)

    if query_log_id is not None:
        saved = await _save_rating(query_log_id, RATING_BAD, user_id)
        if saved:
            await callback.answer(FEEDBACK_THANKS_MSG)
        else:
            await callback.answer(FEEDBACK_ERROR_MSG)
    else:
        log.warning("feedback_no_query_log_id")
        await callback.answer(FEEDBACK_THANKS_MSG)


@router.callback_query(F.data.startswith("feedback:sources"))
async def on_feedback_sources(callback: CallbackQuery) -> None:
    """Handle 📎 Источники button press — display the source list."""
    user_id = callback.from_user.id if callback.from_user else 0
    log = logger.bind(user_id=user_id, action="sources")
    log.info("sources_callback_received")

    query_log_id = await _resolve_query_log_id(callback.data or "", user_id)

    if query_log_id is None:
        await callback.answer(NO_SOURCES_MSG, show_alert=True)
        return

    sources = await _get_sources_for_log(query_log_id)

    if sources is None or not sources:
        await callback.answer(NO_SOURCES_MSG, show_alert=True)
        return

    sources_text = format_sources_list(sources)

    # Telegram callback.answer is limited; use message.answer for long lists
    if callback.message and len(sources_text) > 200:
        await callback.message.answer(f"📎 <b>Источники:</b>\n\n{sources_text}")
        await callback.answer()
    else:
        await callback.answer(sources_text, show_alert=True)
