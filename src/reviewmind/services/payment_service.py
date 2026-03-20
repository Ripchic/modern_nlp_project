"""reviewmind/services/payment_service.py — Telegram Stars payment logic.

Handles subscription activation after successful Telegram Stars payment:
1. Creates a ``subscriptions`` row (status='active', expires_at=+30 days).
2. Updates ``users.subscription`` to 'premium' and ``users.sub_expires_at``.
3. Deduplicates by ``telegram_payment_charge_id`` (UNIQUE constraint).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import structlog
from sqlalchemy.ext.asyncio import AsyncSession

from reviewmind.db.repositories.subscriptions import SubscriptionRepository
from reviewmind.db.repositories.users import UserRepository

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

SUBSCRIPTION_DAYS: int = 30
"""Duration of a single subscription period in days."""

SUBSCRIPTION_PRICE_STARS: int = 75
"""Price in Telegram Stars (XTR). ~299 RUB at current rate."""

SUBSCRIPTION_PRICE_LABEL: str = "ReviewMind Premium (30 дней)"
"""Label displayed in the Telegram payment invoice."""

SUBSCRIPTION_DESCRIPTION: str = (
    "Безлимитные запросы к AI-анализатору обзоров на 30 дней. "
    "Без ограничений по количеству запросов в сутки."
)

SUBSCRIPTION_ACTIVATED_MSG: str = (
    "🎉 <b>Подписка активирована!</b>\n\n"
    "Ваш тариф: <b>Premium</b>\n"
    "Действует до: <b>{expires_at}</b>\n\n"
    "Лимиты на запросы сняты. Приятного использования!"
)

SUBSCRIPTION_ALREADY_ACTIVE_MSG: str = (
    "ℹ️ У вас уже есть активная подписка до <b>{expires_at}</b>.\n"
    "Оплата не была проведена повторно."
)

SUBSCRIPTION_ERROR_MSG: str = "⚠️ Не удалось активировать подписку. Попробуйте позже."


# ── Result dataclass ─────────────────────────────────────────────────────────


class ActivationResult:
    """Result of a subscription activation attempt."""

    __slots__ = ("success", "already_active", "expires_at", "subscription_id", "error")

    def __init__(
        self,
        *,
        success: bool,
        already_active: bool = False,
        expires_at: datetime | None = None,
        subscription_id: int | None = None,
        error: str | None = None,
    ) -> None:
        self.success = success
        self.already_active = already_active
        self.expires_at = expires_at
        self.subscription_id = subscription_id
        self.error = error

    def __repr__(self) -> str:
        return (
            f"ActivationResult(success={self.success}, already_active={self.already_active}, "
            f"expires_at={self.expires_at}, sub_id={self.subscription_id})"
        )

    @property
    def message(self) -> str:
        """Human-readable message for the Telegram user."""
        if self.already_active and self.expires_at:
            return SUBSCRIPTION_ALREADY_ACTIVE_MSG.format(
                expires_at=self.expires_at.strftime("%d.%m.%Y %H:%M UTC")
            )
        if self.success and self.expires_at:
            return SUBSCRIPTION_ACTIVATED_MSG.format(
                expires_at=self.expires_at.strftime("%d.%m.%Y %H:%M UTC")
            )
        return SUBSCRIPTION_ERROR_MSG


# ── Service ──────────────────────────────────────────────────────────────────


class PaymentService:
    """Manages Telegram Stars subscription payments.

    Parameters
    ----------
    session:
        SQLAlchemy async session (caller handles commit/rollback).
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session
        self._user_repo = UserRepository(session)
        self._sub_repo = SubscriptionRepository(session)

    async def activate_subscription(
        self,
        user_id: int,
        telegram_payment_charge_id: str,
        amount_stars: int = SUBSCRIPTION_PRICE_STARS,
    ) -> ActivationResult:
        """Activate a premium subscription after successful payment.

        1. Check for duplicate ``charge_id`` → idempotent.
        2. Ensure ``users`` row exists.
        3. Create ``subscriptions`` row.
        4. Update ``users.subscription = 'premium'``, ``users.sub_expires_at``.

        Returns :class:`ActivationResult`.
        """
        log = logger.bind(user_id=user_id, charge_id=telegram_payment_charge_id)

        # 1. Deduplication — already processed this charge?
        existing = await self._sub_repo.get_by_charge_id(telegram_payment_charge_id)
        if existing is not None:
            log.info("payment_duplicate", sub_id=existing.id)
            return ActivationResult(
                success=True,
                already_active=True,
                expires_at=existing.expires_at,
                subscription_id=existing.id,
            )

        # 2. Ensure user exists
        user, _created = await self._user_repo.get_or_create(user_id)

        # 3. Calculate subscription period
        now = datetime.now(tz=timezone.utc)
        # If user already has an active sub, extend from current expiry
        if user.sub_expires_at and user.sub_expires_at > now:
            base = user.sub_expires_at
        else:
            base = now
        expires_at = base + timedelta(days=SUBSCRIPTION_DAYS)

        # 4. Create subscription record
        sub = await self._sub_repo.create(
            user_id=user_id,
            telegram_payment_charge_id=telegram_payment_charge_id,
            amount_stars=amount_stars,
            activated_at=now,
            expires_at=expires_at,
            status="active",
        )

        # 5. Update user record
        await self._user_repo.update(
            user_id,
            subscription="premium",
            sub_expires_at=expires_at,
        )

        log.info("subscription_activated", sub_id=sub.id, expires_at=expires_at.isoformat())
        return ActivationResult(
            success=True,
            expires_at=expires_at,
            subscription_id=sub.id,
        )
