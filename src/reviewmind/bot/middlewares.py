"""reviewmind/bot/middlewares.py — Bot middlewares (logging, admin auto-registration).

Provides:
- ``LoggingMiddleware`` — structured request tracking for every update.
- ``AdminMiddleware`` — auto-creates admin user records with ``is_admin=True``
  on first interaction.  Admin IDs are read from ``config.admin_user_ids``.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

import structlog
from aiogram import BaseMiddleware
from aiogram.types import TelegramObject, Update

from reviewmind.db.repositories.users import UserRepository
from reviewmind.db.session import build_engine, build_session_factory

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


def _extract_user_id(event: TelegramObject) -> int | None:
    """Extract the Telegram user_id from an Update."""
    if not isinstance(event, Update):
        return None
    if event.message and event.message.from_user:
        return event.message.from_user.id
    if event.callback_query and event.callback_query.from_user:
        return event.callback_query.from_user.id
    return None


class AdminMiddleware(BaseMiddleware):
    """Ensure admin users are registered in the DB with ``is_admin=True``.

    On every update from a user whose ID is in ``admin_user_ids``:
    - If the user doesn't exist in the ``users`` table → create with ``is_admin=True``.
    - If the user exists but ``is_admin`` is ``False`` → set ``is_admin=True``.

    DB failures are non-fatal; the handler always proceeds.

    Parameters
    ----------
    admin_user_ids:
        Explicit set of admin IDs.  When *None*, loaded lazily from config.
    database_url:
        Explicit DB URL.  When *None*, loaded lazily from config.
    """

    def __init__(
        self,
        *,
        admin_user_ids: set[int] | list[int] | None = None,
        database_url: str | None = None,
    ) -> None:
        super().__init__()
        self._admin_ids: set[int] | None = set(admin_user_ids) if admin_user_ids is not None else None
        self._database_url: str | None = database_url
        self._seen: set[int] = set()  # skip repeat DB calls within one process lifetime

    # ── Properties ───────────────────────────────────────────

    @property
    def admin_user_ids(self) -> set[int]:
        if self._admin_ids is None:
            try:
                from reviewmind.config import settings  # noqa: PLC0415

                self._admin_ids = set(settings.admin_user_ids)
            except Exception:
                self._admin_ids = set()
        return self._admin_ids

    @property
    def database_url(self) -> str | None:
        if self._database_url is None:
            try:
                from reviewmind.config import settings  # noqa: PLC0415

                self._database_url = settings.database_url
            except Exception:
                pass
        return self._database_url

    # ── Middleware ────────────────────────────────────────────

    async def __call__(
        self,
        handler: Callable[[TelegramObject, dict[str, Any]], Awaitable[Any]],
        event: TelegramObject,
        data: dict[str, Any],
    ) -> Any:
        user_id = _extract_user_id(event)
        if user_id is not None and user_id in self.admin_user_ids and user_id not in self._seen:
            await self._ensure_admin_in_db(user_id)
            self._seen.add(user_id)

        return await handler(event, data)

    # ── Internal ─────────────────────────────────────────────

    async def _ensure_admin_in_db(self, user_id: int) -> None:
        """Create or update the admin user record.  Never raises."""
        db_url = self.database_url
        if not db_url:
            logger.warning("admin_middleware_no_db", user_id=user_id)
            return

        try:
            engine = build_engine(db_url)
            session_factory = build_session_factory(engine)

            async with session_factory() as session:
                repo = UserRepository(session)
                user = await repo.get_by_id(user_id)
                if user is None:
                    await repo.create(user_id, is_admin=True)
                    logger.info("admin_user_created", user_id=user_id)
                elif not user.is_admin:
                    await repo.update(user_id, is_admin=True)
                    logger.info("admin_flag_set", user_id=user_id)
                else:
                    logger.debug("admin_already_registered", user_id=user_id)
                await session.commit()

            await engine.dispose()

        except Exception:
            logger.exception("admin_middleware_error", user_id=user_id)
