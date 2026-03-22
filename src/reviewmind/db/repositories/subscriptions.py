# reviewmind/db/repositories/subscriptions.py — CRUD для subscriptions
from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from reviewmind.db.models import Subscription


class SubscriptionRepository:
    """CRUD operations for the `subscriptions` table."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_id(self, sub_id: int) -> Subscription | None:
        result = await self._session.execute(select(Subscription).where(Subscription.id == sub_id))
        return result.scalar_one_or_none()

    async def get_by_charge_id(self, charge_id: str) -> Subscription | None:
        result = await self._session.execute(
            select(Subscription).where(Subscription.telegram_payment_charge_id == charge_id)
        )
        return result.scalar_one_or_none()

    async def create(
        self,
        user_id: int,
        telegram_payment_charge_id: str,
        amount_stars: int,
        activated_at: datetime,
        expires_at: datetime,
        status: str = "active",
    ) -> Subscription:
        sub = Subscription(
            user_id=user_id,
            telegram_payment_charge_id=telegram_payment_charge_id,
            amount_stars=amount_stars,
            activated_at=activated_at,
            expires_at=expires_at,
            status=status,
        )
        self._session.add(sub)
        await self._session.flush()
        return sub

    async def update(self, sub_id: int, **kwargs) -> Subscription | None:
        sub = await self.get_by_id(sub_id)
        if sub is None:
            return None
        for key, value in kwargs.items():
            setattr(sub, key, value)
        await self._session.flush()
        return sub

    async def list_by_user(self, user_id: int) -> list[Subscription]:
        result = await self._session.execute(
            select(Subscription).where(Subscription.user_id == user_id).order_by(Subscription.activated_at.desc())
        )
        return list(result.scalars().all())
