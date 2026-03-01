# reviewmind/db/repositories/users.py — CRUD для users
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from reviewmind.db.models import User


class UserRepository:
    """CRUD operations for the `users` table."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_id(self, user_id: int) -> User | None:
        result = await self._session.execute(select(User).where(User.user_id == user_id))
        return result.scalar_one_or_none()

    async def create(
        self,
        user_id: int,
        *,
        is_admin: bool = False,
        subscription: str = "free",
    ) -> User:
        user = User(user_id=user_id, is_admin=is_admin, subscription=subscription)
        self._session.add(user)
        await self._session.flush()
        return user

    async def get_or_create(self, user_id: int) -> tuple[User, bool]:
        """Return (user, created). `created` is True when a new row was inserted."""
        user = await self.get_by_id(user_id)
        if user is not None:
            return user, False
        user = await self.create(user_id)
        return user, True

    async def update(self, user_id: int, **kwargs) -> User | None:
        user = await self.get_by_id(user_id)
        if user is None:
            return None
        for key, value in kwargs.items():
            setattr(user, key, value)
        await self._session.flush()
        return user

    async def delete(self, user_id: int) -> bool:
        user = await self.get_by_id(user_id)
        if user is None:
            return False
        await self._session.delete(user)
        await self._session.flush()
        return True
