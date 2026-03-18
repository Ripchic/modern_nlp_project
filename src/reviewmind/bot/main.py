"""reviewmind/bot/main.py — Entrypoint Telegram-бота (long polling)."""

from __future__ import annotations

import asyncio
import logging
import sys

import structlog
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

from reviewmind.bot.handlers.feedback import router as feedback_router
from reviewmind.bot.handlers.links import router as links_router
from reviewmind.bot.handlers.mode import router as mode_router
from reviewmind.bot.handlers.query import router as query_router
from reviewmind.bot.handlers.start import router as start_router
from reviewmind.bot.middlewares import LoggingMiddleware

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


def _configure_logging() -> None:
    """Configure structlog + stdlib logging for the bot process."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        stream=sys.stdout,
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        logger_factory=structlog.PrintLoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def create_dispatcher() -> Dispatcher:
    """Create and configure the aiogram Dispatcher with all routers."""
    dp = Dispatcher()

    # Register outer middleware for structured logging
    dp.update.outer_middleware(LoggingMiddleware())

    # Register handlers — order matters: start/help first, then mode callbacks,
    # then feedback callbacks, then links (URL messages),
    # then query (catch-all for text messages) last.
    dp.include_router(start_router)
    dp.include_router(mode_router)
    dp.include_router(feedback_router)
    dp.include_router(links_router)
    dp.include_router(query_router)

    return dp


def create_bot(token: str) -> Bot:
    """Create the aiogram Bot instance with default properties."""
    return Bot(
        token=token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )


async def run_bot() -> None:
    """Run the bot with long polling."""
    from reviewmind.config import settings

    _configure_logging()

    bot = create_bot(settings.telegram_bot_token)
    dp = create_dispatcher()

    logger.info("bot_starting", mode="long_polling")

    try:
        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
    finally:
        await bot.session.close()
        logger.info("bot_stopped")


def main() -> None:
    """Synchronous entrypoint for ``python -m reviewmind.bot.main``."""
    asyncio.run(run_bot())


if __name__ == "__main__":
    main()
