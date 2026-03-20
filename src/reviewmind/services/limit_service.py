"""reviewmind/services/limit_service.py — Управление лимитами пользователей.

Free tier: 3 запроса в сутки (UTC).
Premium/admin: без лимита.
Counter stored in ``user_limits`` (user_id + date composite PK).
"""

from __future__ import annotations

from datetime import date, timezone
from datetime import datetime as dt

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from reviewmind.db.repositories.limits import UserLimitRepository
from reviewmind.db.repositories.users import UserRepository

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

FREE_DAILY_LIMIT: int = 10
"""Maximum number of queries per day for free-tier users."""

PREMIUM_SUBSCRIPTION: str = "premium"
"""Subscription value that bypasses limits."""

LIMIT_REACHED_MSG: str = (
    "⚠️ Вы использовали {used}/{limit} бесплатных запросов на сегодня.\n"
    "Подпишитесь на безлимитный тариф или попробуйте завтра!"
)
"""Message template shown when the daily limit is exhausted."""


# ── Data class ───────────────────────────────────────────────────────────────


class LimitCheckResult:
    """Result of a limit check."""

    __slots__ = ("allowed", "requests_used", "requests_limit", "reason")

    def __init__(
        self,
        *,
        allowed: bool,
        requests_used: int,
        requests_limit: int,
        reason: str = "",
    ) -> None:
        self.allowed = allowed
        self.requests_used = requests_used
        self.requests_limit = requests_limit
        self.reason = reason

    def __repr__(self) -> str:
        return (
            f"LimitCheckResult(allowed={self.allowed}, "
            f"used={self.requests_used}/{self.requests_limit}, "
            f"reason={self.reason!r})"
        )

    @property
    def message(self) -> str:
        """Human-readable limit-reached message (empty when allowed)."""
        if self.allowed:
            return ""
        return LIMIT_REACHED_MSG.format(used=self.requests_used, limit=self.requests_limit)


# ── Service ──────────────────────────────────────────────────────────────────


class LimitService:
    """Manages per-user daily request limits.

    Parameters
    ----------
    session:
        SQLAlchemy async session (caller is responsible for commit/rollback).
    admin_user_ids:
        Set of Telegram user IDs that bypass limits.
        When *None*, loaded lazily from ``config.settings.admin_user_ids``.
    daily_limit:
        Maximum daily requests for free-tier users.
    """

    def __init__(
        self,
        session: AsyncSession,
        *,
        admin_user_ids: set[int] | list[int] | None = None,
        daily_limit: int = FREE_DAILY_LIMIT,
    ) -> None:
        self._session = session
        self._user_repo = UserRepository(session)
        self._limit_repo = UserLimitRepository(session)
        self._daily_limit = daily_limit
        self._admin_ids: set[int] | None = set(admin_user_ids) if admin_user_ids is not None else None

    # ── Properties ───────────────────────────────────────────

    @property
    def daily_limit(self) -> int:
        return self._daily_limit

    @property
    def admin_user_ids(self) -> set[int]:
        if self._admin_ids is None:
            try:
                from reviewmind.config import settings  # noqa: PLC0415

                self._admin_ids = set(settings.admin_user_ids)
            except Exception:
                self._admin_ids = set()
        return self._admin_ids

    # ── Public API ───────────────────────────────────────────

    def _today(self) -> date:
        """Return the current UTC date.  Extracted for easy mocking in tests."""
        return dt.now(tz=timezone.utc).date()

    def _is_admin(self, user_id: int) -> bool:
        return user_id in self.admin_user_ids

    async def _is_premium(self, user_id: int) -> bool:
        """Check if the user holds a premium subscription."""
        user = await self._user_repo.get_by_id(user_id)
        if user is None:
            return False
        return user.subscription == PREMIUM_SUBSCRIPTION

    async def check_limit(self, user_id: int) -> LimitCheckResult:
        """Check whether *user_id* may make another query today.

        Returns a :class:`LimitCheckResult` describing the outcome.
        """
        log = logger.bind(user_id=user_id)

        # Admin bypass
        if self._is_admin(user_id):
            log.debug("limit_bypass", reason="admin")
            return LimitCheckResult(
                allowed=True,
                requests_used=0,
                requests_limit=self._daily_limit,
                reason="admin",
            )

        # Premium bypass
        if await self._is_premium(user_id):
            log.debug("limit_bypass", reason="premium")
            return LimitCheckResult(
                allowed=True,
                requests_used=0,
                requests_limit=self._daily_limit,
                reason="premium",
            )

        # Fetch today's counter
        today = self._today()
        row = await self._limit_repo.get(user_id, today)
        used = row.requests_used if row else 0

        if used >= self._daily_limit:
            log.info("limit_reached", used=used, limit=self._daily_limit)
            return LimitCheckResult(
                allowed=False,
                requests_used=used,
                requests_limit=self._daily_limit,
                reason="limit_reached",
            )

        log.debug("limit_ok", used=used, limit=self._daily_limit)
        return LimitCheckResult(
            allowed=True,
            requests_used=used,
            requests_limit=self._daily_limit,
            reason="ok",
        )

    async def increment(self, user_id: int) -> int:
        """Increment today's request counter for *user_id*.

        Ensures the ``users`` row exists (get_or_create).
        Returns the new ``requests_used`` value.
        """
        await self._user_repo.get_or_create(user_id)
        today = self._today()
        row = await self._limit_repo.increment(user_id, today)
        logger.debug("limit_incremented", user_id=user_id, used=row.requests_used)
        return row.requests_used

    async def get_usage(self, user_id: int) -> LimitCheckResult:
        """Return the current usage without incrementing.

        Equivalent to :meth:`check_limit` but named for clarity in
        read-only contexts.
        """
        return await self.check_limit(user_id)
