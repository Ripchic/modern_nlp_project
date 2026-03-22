# reviewmind/db/repositories/limits.py — CRUD для user_limits
from __future__ import annotations

from datetime import date

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from reviewmind.db.models import UserLimit


class UserLimitRepository:
    """CRUD operations for the `user_limits` table."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get(self, user_id: int, date_: date) -> UserLimit | None:
        result = await self._session.execute(
            select(UserLimit).where(UserLimit.user_id == user_id, UserLimit.date == date_)
        )
        return result.scalar_one_or_none()

    async def get_or_create(self, user_id: int, date_: date) -> tuple[UserLimit, bool]:
        """Return (limit_row, created)."""
        row = await self.get(user_id, date_)
        if row is not None:
            return row, False
        row = UserLimit(user_id=user_id, date=date_, requests_used=0)
        self._session.add(row)
        await self._session.flush()
        return row, True

    async def increment(self, user_id: int, date_: date, *, by: int = 1) -> UserLimit:
        """Increment requests_used by `by`, creating the row if needed."""
        row, _ = await self.get_or_create(user_id, date_)
        row.requests_used += by
        await self._session.flush()
        return row

    async def reset(self, user_id: int, date_: date) -> UserLimit | None:
        row = await self.get(user_id, date_)
        if row is None:
            return None
        row.requests_used = 0
        await self._session.flush()
        return row
