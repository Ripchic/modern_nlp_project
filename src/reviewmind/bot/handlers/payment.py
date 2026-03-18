"""reviewmind/bot/handlers/payment.py — Telegram Stars payment handlers.

Flow:
1. User sends /subscribe or presses [⭐ Безлимит] → bot sends Invoice (XTR).
2. Telegram sends pre_checkout_query → handler answers ``ok=True``.
3. Telegram sends successful_payment → handler activates subscription in DB.

Uses Telegram Payments API with currency ``XTR`` (Telegram Stars).
"""

from __future__ import annotations

import structlog
from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import (
    CallbackQuery,
    LabeledPrice,
    Message,
    PreCheckoutQuery,
    SuccessfulPayment,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from reviewmind.bot.keyboards import SUBSCRIBE_ACTION
from reviewmind.services.payment_service import (
    SUBSCRIPTION_DESCRIPTION,
    SUBSCRIPTION_ERROR_MSG,
    SUBSCRIPTION_PRICE_LABEL,
    SUBSCRIPTION_PRICE_STARS,
    PaymentService,
)

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

router = Router(name="payment")

# ── Constants ────────────────────────────────────────────────────────────────

INVOICE_TITLE: str = "ReviewMind Premium"
INVOICE_CURRENCY: str = "XTR"
INVOICE_PAYLOAD: str = "reviewmind_premium_30d"

PRE_CHECKOUT_OK: str = ""
PRE_CHECKOUT_ERROR: str = "Не удалось обработать платёж. Попробуйте позже."

SUBSCRIBE_PROMPT_MSG: str = (
    "⭐ <b>ReviewMind Premium</b>\n\n"
    "Безлимитные запросы на 30 дней.\n"
    "Стоимость: <b>{price} Telegram Star(s)</b>\n\n"
    "Нажмите кнопку ниже для оплаты."
)


# ── Helpers ──────────────────────────────────────────────────────────────────


async def _get_db_session():
    """Create a short-lived async DB session from config."""
    from reviewmind.config import settings  # noqa: PLC0415

    engine = create_async_engine(settings.database_url, pool_pre_ping=True, echo=False)
    session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    return engine, session_factory


async def _send_invoice(message: Message) -> None:
    """Send a Telegram Stars invoice to the user."""
    await message.answer_invoice(
        title=INVOICE_TITLE,
        description=SUBSCRIPTION_DESCRIPTION,
        payload=INVOICE_PAYLOAD,
        currency=INVOICE_CURRENCY,
        prices=[LabeledPrice(label=SUBSCRIPTION_PRICE_LABEL, amount=SUBSCRIPTION_PRICE_STARS)],
    )


# ── Handlers ─────────────────────────────────────────────────────────────────


@router.message(Command("subscribe"))
async def cmd_subscribe(message: Message) -> None:
    """Handle /subscribe — send payment invoice."""
    user_id = message.from_user.id if message.from_user else 0
    logger.info("subscribe_command", user_id=user_id)
    await _send_invoice(message)


@router.callback_query(F.data == SUBSCRIBE_ACTION)
async def on_subscribe_button(callback: CallbackQuery) -> None:
    """Handle [⭐ Безлимит] inline button — send payment invoice."""
    user_id = callback.from_user.id if callback.from_user else 0
    logger.info("subscribe_button_pressed", user_id=user_id)
    await callback.answer()
    if callback.message:
        await _send_invoice(callback.message)


@router.pre_checkout_query()
async def on_pre_checkout(pre_checkout_query: PreCheckoutQuery) -> None:
    """Handle pre-checkout query — always approve for Telegram Stars."""
    user_id = pre_checkout_query.from_user.id if pre_checkout_query.from_user else 0
    logger.info(
        "pre_checkout",
        user_id=user_id,
        payload=pre_checkout_query.invoice_payload,
        currency=pre_checkout_query.currency,
        amount=pre_checkout_query.total_amount,
    )
    await pre_checkout_query.answer(ok=True)


@router.message(F.successful_payment)
async def on_successful_payment(message: Message) -> None:
    """Handle successful payment — activate premium subscription."""
    payment: SuccessfulPayment = message.successful_payment  # type: ignore[assignment]
    user_id = message.from_user.id if message.from_user else 0
    charge_id = payment.telegram_payment_charge_id
    amount = payment.total_amount

    log = logger.bind(user_id=user_id, charge_id=charge_id, amount=amount)
    log.info("payment_received")

    try:
        engine, session_factory = await _get_db_session()
        try:
            async with session_factory() as session:
                service = PaymentService(session)
                result = await service.activate_subscription(
                    user_id=user_id,
                    telegram_payment_charge_id=charge_id,
                    amount_stars=amount,
                )
                await session.commit()

            await message.answer(result.message, parse_mode="HTML")
            log.info(
                "payment_processed",
                success=result.success,
                already_active=result.already_active,
                sub_id=result.subscription_id,
            )
        finally:
            await engine.dispose()
    except Exception as exc:
        log.error("payment_activation_error", error=str(exc))
        await message.answer(SUBSCRIPTION_ERROR_MSG, parse_mode="HTML")
