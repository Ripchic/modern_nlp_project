"""reviewmind/bot/middlewares.py — Bot middlewares (logging, rate limit, auth).

Rate-limit and auth middlewares will be added in later tasks.
This module provides a logging middleware for structured request tracking.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

import structlog
from aiogram import BaseMiddleware
from aiogram.types import TelegramObject, Update

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)


class LoggingMiddleware(BaseMiddleware):
    """Middleware that logs every incoming update with user info."""

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        """Log the incoming update and delegate to the next handler."""
        if isinstance(event, Update):
            user_id = None
            update_type = event.event_type
            if event.message and event.message.from_user:
                user_id = event.message.from_user.id
            elif event.callback_query and event.callback_query.from_user:
                user_id = event.callback_query.from_user.id

            logger.info(
                "update_received",
                update_id=event.update_id,
                update_type=update_type,
                user_id=user_id,
            )

        return await handler(event, data)
