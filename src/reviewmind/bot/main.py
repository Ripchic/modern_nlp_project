"""reviewmind/bot/main.py — Entrypoint Telegram-бота (long polling)."""

from __future__ import annotations

import asyncio
import logging
import os
import sys

import structlog
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.client.session.aiohttp import AiohttpSession
from aiogram.enums import ParseMode
from aiogram.types import BotCommand

from reviewmind.bot.handlers.feedback import router as feedback_router
from reviewmind.bot.handlers.gdpr import router as gdpr_router
from reviewmind.bot.handlers.links import router as links_router
from reviewmind.bot.handlers.mode import router as mode_router
from reviewmind.bot.handlers.payment import router as payment_router
from reviewmind.bot.handlers.query import router as query_router
from reviewmind.bot.handlers.start import router as start_router
from reviewmind.bot.middlewares import AdminMiddleware, LoggingMiddleware

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

    # Register admin auto-registration middleware
    dp.update.outer_middleware(AdminMiddleware())

    # Register handlers — order matters: start/help first, then mode callbacks,
    # then payment (pre_checkout + successful_payment), then feedback callbacks,
    # then links (URL messages), then query (catch-all for text messages) last.
    dp.include_router(start_router)
    dp.include_router(mode_router)
    dp.include_router(gdpr_router)
    dp.include_router(payment_router)
    dp.include_router(feedback_router)
    dp.include_router(links_router)
    dp.include_router(query_router)

    return dp


def create_bot(token: str) -> Bot:
    """Create the aiogram Bot instance with default properties."""
    proxy_url = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
    session = AiohttpSession(proxy=proxy_url) if proxy_url else None
    return Bot(
        token=token,
        session=session,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )


async def _set_commands(bot: Bot) -> None:
    """Register the bot command menu shown in the Telegram UI."""
    commands = [
        BotCommand(command="start", description="Запустить бота / главное меню"),
        BotCommand(command="help", description="Помощь и список команд"),
        BotCommand(command="subscribe", description="Оформить Premium-подписку"),
        BotCommand(command="myid", description="Узнать свой Telegram ID"),
        BotCommand(command="delete_my_data", description="Удалить мои данные (GDPR)"),
    ]
    await bot.set_my_commands(commands)


async def run_bot() -> None:
    """Run the bot with long polling."""
    from reviewmind.config import settings

    _configure_logging()

    bot = create_bot(settings.telegram_bot_token)
    dp = create_dispatcher()

    logger.info("bot_starting", mode="long_polling")

    await _set_commands(bot)

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
