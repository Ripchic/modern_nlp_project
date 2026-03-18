"""Unit tests for TASK-037 — Telegram Stars: payment integration (XTR).

Tests cover:
- PaymentService constants
- ActivationResult data class and message property
- PaymentService.activate_subscription (new, duplicate, extension)
- Bot handler: /subscribe command
- Bot handler: subscribe button callback
- Bot handler: pre_checkout_query
- Bot handler: successful_payment
- Dispatcher wiring (payment_router registered)
- Services __init__.py exports
- /help text includes /subscribe
- Integration scenarios
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiogram.types import CallbackQuery, Chat, Message, PreCheckoutQuery, SuccessfulPayment, User

from reviewmind.services.payment_service import (
    SUBSCRIPTION_ACTIVATED_MSG,
    SUBSCRIPTION_ALREADY_ACTIVE_MSG,
    SUBSCRIPTION_DAYS,
    SUBSCRIPTION_DESCRIPTION,
    SUBSCRIPTION_ERROR_MSG,
    SUBSCRIPTION_PRICE_LABEL,
    SUBSCRIPTION_PRICE_STARS,
    ActivationResult,
    PaymentService,
)

# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════


def _mock_session() -> MagicMock:
    """Create a mocked AsyncSession."""
    session = MagicMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.flush = AsyncMock()
    session.close = AsyncMock()
    return session


def _make_user_model(user_id: int = 100, subscription: str = "free", sub_expires_at=None) -> MagicMock:
    """Create a mock DB User model."""
    user = MagicMock()
    user.user_id = user_id
    user.subscription = subscription
    user.sub_expires_at = sub_expires_at
    user.is_admin = False
    return user


def _make_subscription(sub_id: int = 1, charge_id: str = "charge_abc", expires_at=None) -> MagicMock:
    """Create a mock DB Subscription model."""
    sub = MagicMock()
    sub.id = sub_id
    sub.telegram_payment_charge_id = charge_id
    sub.expires_at = expires_at or datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc)
    sub.amount_stars = SUBSCRIPTION_PRICE_STARS
    sub.status = "active"
    return sub


def _make_tg_user(user_id: int = 12345) -> User:
    return User(id=user_id, is_bot=False, first_name="Test")


def _make_chat(chat_id: int = 12345) -> Chat:
    return Chat(id=chat_id, type="private")


def _make_message(user_id: int = 12345) -> MagicMock:
    msg = MagicMock(spec=Message)
    msg.from_user = _make_tg_user(user_id)
    msg.chat = _make_chat(user_id)
    msg.answer = AsyncMock()
    msg.answer_invoice = AsyncMock()
    msg.successful_payment = None
    return msg


def _make_callback(data: str, user_id: int = 12345) -> MagicMock:
    cb = MagicMock(spec=CallbackQuery)
    cb.data = data
    cb.from_user = _make_tg_user(user_id)
    cb.message = _make_message(user_id)
    cb.answer = AsyncMock()
    return cb


def _make_pre_checkout(user_id: int = 12345) -> MagicMock:
    pcq = MagicMock(spec=PreCheckoutQuery)
    pcq.from_user = _make_tg_user(user_id)
    pcq.invoice_payload = "reviewmind_premium_30d"
    pcq.currency = "XTR"
    pcq.total_amount = SUBSCRIPTION_PRICE_STARS
    pcq.answer = AsyncMock()
    return pcq


def _make_successful_payment(charge_id: str = "charge_xyz", amount: int = 1) -> MagicMock:
    payment = MagicMock(spec=SuccessfulPayment)
    payment.telegram_payment_charge_id = charge_id
    payment.total_amount = amount
    payment.currency = "XTR"
    payment.invoice_payload = "reviewmind_premium_30d"
    return payment


# ══════════════════════════════════════════════════════════════
# Tests — Constants
# ══════════════════════════════════════════════════════════════


class TestConstants:
    def test_subscription_days(self):
        assert SUBSCRIPTION_DAYS == 30

    def test_subscription_price_stars(self):
        assert SUBSCRIPTION_PRICE_STARS == 1

    def test_subscription_price_label_not_empty(self):
        assert len(SUBSCRIPTION_PRICE_LABEL) > 0

    def test_subscription_description_not_empty(self):
        assert len(SUBSCRIPTION_DESCRIPTION) > 0

    def test_activated_msg_contains_placeholder(self):
        assert "{expires_at}" in SUBSCRIPTION_ACTIVATED_MSG

    def test_already_active_msg_contains_placeholder(self):
        assert "{expires_at}" in SUBSCRIPTION_ALREADY_ACTIVE_MSG

    def test_error_msg_not_empty(self):
        assert len(SUBSCRIPTION_ERROR_MSG) > 0


# ══════════════════════════════════════════════════════════════
# Tests — ActivationResult
# ══════════════════════════════════════════════════════════════


class TestActivationResult:
    def test_success_result(self):
        expires = datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc)
        result = ActivationResult(success=True, expires_at=expires, subscription_id=42)
        assert result.success is True
        assert result.already_active is False
        assert result.expires_at == expires
        assert result.subscription_id == 42

    def test_already_active_result(self):
        expires = datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc)
        result = ActivationResult(success=True, already_active=True, expires_at=expires, subscription_id=1)
        assert result.already_active is True

    def test_error_result(self):
        result = ActivationResult(success=False, error="DB unavailable")
        assert result.success is False
        assert result.error == "DB unavailable"

    def test_message_success(self):
        expires = datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc)
        result = ActivationResult(success=True, expires_at=expires)
        msg = result.message
        assert "Подписка активирована" in msg
        assert "18.04.2026" in msg

    def test_message_already_active(self):
        expires = datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc)
        result = ActivationResult(success=True, already_active=True, expires_at=expires)
        msg = result.message
        assert "уже есть активная" in msg

    def test_message_error(self):
        result = ActivationResult(success=False)
        assert result.message == SUBSCRIPTION_ERROR_MSG

    def test_repr(self):
        result = ActivationResult(success=True, subscription_id=5)
        repr_str = repr(result)
        assert "success=True" in repr_str
        assert "sub_id=5" in repr_str


# ══════════════════════════════════════════════════════════════
# Tests — PaymentService.activate_subscription
# ══════════════════════════════════════════════════════════════


class TestActivateSubscription:
    @pytest.mark.asyncio
    async def test_new_subscription(self):
        """First-time payment creates subscription and upgrades user to premium."""
        session = _mock_session()
        created_sub = _make_subscription(sub_id=10, charge_id="charge_new")
        user = _make_user_model(user_id=100, subscription="free")

        with (
            patch.object(PaymentService, "__init__", lambda self, s: None),
        ):
            service = PaymentService(session)
            service._session = session
            service._sub_repo = MagicMock()
            service._sub_repo.get_by_charge_id = AsyncMock(return_value=None)
            service._sub_repo.create = AsyncMock(return_value=created_sub)
            service._user_repo = MagicMock()
            service._user_repo.get_or_create = AsyncMock(return_value=(user, False))
            service._user_repo.update = AsyncMock(return_value=user)

            result = await service.activate_subscription(
                user_id=100, telegram_payment_charge_id="charge_new",
            )

        assert result.success is True
        assert result.already_active is False
        assert result.subscription_id == 10
        service._sub_repo.create.assert_awaited_once()
        service._user_repo.update.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_duplicate_charge_id(self):
        """Duplicate charge_id returns already_active without creating new sub."""
        session = _mock_session()
        existing_sub = _make_subscription(sub_id=5, charge_id="charge_dup")

        with patch.object(PaymentService, "__init__", lambda self, s: None):
            service = PaymentService(session)
            service._session = session
            service._sub_repo = MagicMock()
            service._sub_repo.get_by_charge_id = AsyncMock(return_value=existing_sub)
            service._user_repo = MagicMock()

            result = await service.activate_subscription(
                user_id=100, telegram_payment_charge_id="charge_dup",
            )

        assert result.success is True
        assert result.already_active is True
        assert result.subscription_id == 5
        service._sub_repo.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_extend_existing_subscription(self):
        """If user has active sub, new payment extends from current expiry."""
        session = _mock_session()
        now = datetime.now(tz=timezone.utc)
        existing_expiry = now + timedelta(days=10)
        user = _make_user_model(user_id=100, subscription="premium", sub_expires_at=existing_expiry)
        created_sub = _make_subscription(sub_id=11)

        with patch.object(PaymentService, "__init__", lambda self, s: None):
            service = PaymentService(session)
            service._session = session
            service._sub_repo = MagicMock()
            service._sub_repo.get_by_charge_id = AsyncMock(return_value=None)
            service._sub_repo.create = AsyncMock(return_value=created_sub)
            service._user_repo = MagicMock()
            service._user_repo.get_or_create = AsyncMock(return_value=(user, False))
            service._user_repo.update = AsyncMock(return_value=user)

            result = await service.activate_subscription(
                user_id=100, telegram_payment_charge_id="charge_ext",
            )

        assert result.success is True
        # Verify the expires_at passed to create includes the extension
        call_kwargs = service._sub_repo.create.call_args[1]
        expected_min = existing_expiry + timedelta(days=SUBSCRIPTION_DAYS - 1)
        assert call_kwargs["expires_at"] > expected_min

    @pytest.mark.asyncio
    async def test_expired_subscription_starts_from_now(self):
        """If previous sub expired, new sub starts from now (not from old expiry)."""
        session = _mock_session()
        past_expiry = datetime(2025, 1, 1, tzinfo=timezone.utc)
        user = _make_user_model(user_id=100, subscription="free", sub_expires_at=past_expiry)
        created_sub = _make_subscription(sub_id=12)

        with patch.object(PaymentService, "__init__", lambda self, s: None):
            service = PaymentService(session)
            service._session = session
            service._sub_repo = MagicMock()
            service._sub_repo.get_by_charge_id = AsyncMock(return_value=None)
            service._sub_repo.create = AsyncMock(return_value=created_sub)
            service._user_repo = MagicMock()
            service._user_repo.get_or_create = AsyncMock(return_value=(user, False))
            service._user_repo.update = AsyncMock(return_value=user)

            result = await service.activate_subscription(
                user_id=100, telegram_payment_charge_id="charge_new2",
            )

        assert result.success is True
        call_kwargs = service._sub_repo.create.call_args[1]
        # Should start from ~now, not from past_expiry
        assert call_kwargs["expires_at"] > datetime.now(tz=timezone.utc) + timedelta(days=29)

    @pytest.mark.asyncio
    async def test_new_user_created(self):
        """If user doesn't exist in DB, get_or_create creates a new one."""
        session = _mock_session()
        user = _make_user_model(user_id=999)
        created_sub = _make_subscription(sub_id=13)

        with patch.object(PaymentService, "__init__", lambda self, s: None):
            service = PaymentService(session)
            service._session = session
            service._sub_repo = MagicMock()
            service._sub_repo.get_by_charge_id = AsyncMock(return_value=None)
            service._sub_repo.create = AsyncMock(return_value=created_sub)
            service._user_repo = MagicMock()
            service._user_repo.get_or_create = AsyncMock(return_value=(user, True))
            service._user_repo.update = AsyncMock(return_value=user)

            result = await service.activate_subscription(
                user_id=999, telegram_payment_charge_id="charge_newuser",
            )

        assert result.success is True
        service._user_repo.get_or_create.assert_awaited_once_with(999)

    @pytest.mark.asyncio
    async def test_custom_amount_stars(self):
        """amount_stars parameter is forwarded to sub_repo.create."""
        session = _mock_session()
        user = _make_user_model()
        created_sub = _make_subscription(sub_id=14)

        with patch.object(PaymentService, "__init__", lambda self, s: None):
            service = PaymentService(session)
            service._session = session
            service._sub_repo = MagicMock()
            service._sub_repo.get_by_charge_id = AsyncMock(return_value=None)
            service._sub_repo.create = AsyncMock(return_value=created_sub)
            service._user_repo = MagicMock()
            service._user_repo.get_or_create = AsyncMock(return_value=(user, False))
            service._user_repo.update = AsyncMock(return_value=user)

            await service.activate_subscription(
                user_id=100, telegram_payment_charge_id="charge_amt", amount_stars=5,
            )

        call_kwargs = service._sub_repo.create.call_args[1]
        assert call_kwargs["amount_stars"] == 5


# ══════════════════════════════════════════════════════════════
# Tests — Bot Handlers
# ══════════════════════════════════════════════════════════════


class TestCmdSubscribe:
    @pytest.mark.asyncio
    async def test_sends_invoice(self):
        from reviewmind.bot.handlers.payment import cmd_subscribe

        msg = _make_message()
        await cmd_subscribe(msg)
        msg.answer_invoice.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_invoice_currency_is_xtr(self):
        from reviewmind.bot.handlers.payment import cmd_subscribe

        msg = _make_message()
        await cmd_subscribe(msg)
        call_kwargs = msg.answer_invoice.call_args[1]
        assert call_kwargs["currency"] == "XTR"

    @pytest.mark.asyncio
    async def test_invoice_contains_price(self):
        from reviewmind.bot.handlers.payment import cmd_subscribe

        msg = _make_message()
        await cmd_subscribe(msg)
        call_kwargs = msg.answer_invoice.call_args[1]
        prices = call_kwargs["prices"]
        assert len(prices) == 1
        assert prices[0].amount == SUBSCRIPTION_PRICE_STARS

    @pytest.mark.asyncio
    async def test_invoice_payload(self):
        from reviewmind.bot.handlers.payment import INVOICE_PAYLOAD, cmd_subscribe

        msg = _make_message()
        await cmd_subscribe(msg)
        call_kwargs = msg.answer_invoice.call_args[1]
        assert call_kwargs["payload"] == INVOICE_PAYLOAD


class TestSubscribeButton:
    @pytest.mark.asyncio
    async def test_callback_sends_invoice(self):
        from reviewmind.bot.handlers.payment import on_subscribe_button

        cb = _make_callback("subscribe:start")
        await on_subscribe_button(cb)
        cb.answer.assert_awaited_once()
        cb.message.answer_invoice.assert_awaited_once()


class TestPreCheckout:
    @pytest.mark.asyncio
    async def test_always_approves(self):
        from reviewmind.bot.handlers.payment import on_pre_checkout

        pcq = _make_pre_checkout()
        await on_pre_checkout(pcq)
        pcq.answer.assert_awaited_once_with(ok=True)


class TestSuccessfulPayment:
    @pytest.mark.asyncio
    async def test_activates_subscription(self):
        from reviewmind.bot.handlers.payment import on_successful_payment

        msg = _make_message()
        msg.successful_payment = _make_successful_payment(charge_id="charge_success")

        mock_engine = MagicMock()
        mock_engine.dispose = AsyncMock()
        mock_session_factory = MagicMock()

        mock_result = ActivationResult(
            success=True,
            expires_at=datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc),
            subscription_id=1,
        )
        mock_service = MagicMock()
        mock_service.activate_subscription = AsyncMock(return_value=mock_result)

        mock_db_session = MagicMock()
        mock_db_session.commit = AsyncMock()
        mock_db_session.rollback = AsyncMock()
        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_db_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_session_factory.return_value = mock_session_ctx

        with (
            patch(
                "reviewmind.bot.handlers.payment._get_db_session",
                new_callable=AsyncMock,
                return_value=(mock_engine, mock_session_factory),
            ),
            patch(
                "reviewmind.bot.handlers.payment.PaymentService",
                return_value=mock_service,
            ),
        ):
            await on_successful_payment(msg)

        msg.answer.assert_awaited_once()
        answer_text = msg.answer.call_args[0][0]
        assert "Подписка активирована" in answer_text

    @pytest.mark.asyncio
    async def test_db_error_sends_error_msg(self):
        from reviewmind.bot.handlers.payment import on_successful_payment

        msg = _make_message()
        msg.successful_payment = _make_successful_payment()

        with patch(
            "reviewmind.bot.handlers.payment._get_db_session",
            new_callable=AsyncMock,
            side_effect=Exception("DB down"),
        ):
            await on_successful_payment(msg)

        msg.answer.assert_awaited_once()
        answer_text = msg.answer.call_args[0][0]
        assert "Не удалось" in answer_text

    @pytest.mark.asyncio
    async def test_engine_disposed_on_success(self):
        from reviewmind.bot.handlers.payment import on_successful_payment

        msg = _make_message()
        msg.successful_payment = _make_successful_payment()

        mock_engine = MagicMock()
        mock_engine.dispose = AsyncMock()
        mock_session_factory = MagicMock()

        mock_result = ActivationResult(success=True, expires_at=datetime.now(tz=timezone.utc))
        mock_service = MagicMock()
        mock_service.activate_subscription = AsyncMock(return_value=mock_result)

        mock_db_session = MagicMock()
        mock_db_session.commit = AsyncMock()
        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_db_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_session_factory.return_value = mock_session_ctx

        with (
            patch(
                "reviewmind.bot.handlers.payment._get_db_session",
                new_callable=AsyncMock,
                return_value=(mock_engine, mock_session_factory),
            ),
            patch(
                "reviewmind.bot.handlers.payment.PaymentService",
                return_value=mock_service,
            ),
        ):
            await on_successful_payment(msg)

        mock_engine.dispose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_duplicate_payment_shows_already_active(self):
        from reviewmind.bot.handlers.payment import on_successful_payment

        msg = _make_message()
        msg.successful_payment = _make_successful_payment(charge_id="charge_dup2")

        mock_engine = MagicMock()
        mock_engine.dispose = AsyncMock()
        mock_session_factory = MagicMock()

        expires = datetime(2026, 5, 1, 0, 0, tzinfo=timezone.utc)
        mock_result = ActivationResult(
            success=True, already_active=True, expires_at=expires, subscription_id=5,
        )
        mock_service = MagicMock()
        mock_service.activate_subscription = AsyncMock(return_value=mock_result)

        mock_db_session = MagicMock()
        mock_db_session.commit = AsyncMock()
        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_db_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_session_factory.return_value = mock_session_ctx

        with (
            patch(
                "reviewmind.bot.handlers.payment._get_db_session",
                new_callable=AsyncMock,
                return_value=(mock_engine, mock_session_factory),
            ),
            patch(
                "reviewmind.bot.handlers.payment.PaymentService",
                return_value=mock_service,
            ),
        ):
            await on_successful_payment(msg)

        answer_text = msg.answer.call_args[0][0]
        assert "уже есть активная" in answer_text


# ══════════════════════════════════════════════════════════════
# Tests — Dispatcher Wiring
# ══════════════════════════════════════════════════════════════


class TestRouterWiring:
    def test_payment_router_exists(self):
        from reviewmind.bot.handlers.payment import router

        assert router.name == "payment"

    def test_payment_router_registered_in_dispatcher(self):
        """Check that bot/main.py includes payment_router import."""
        import inspect

        import reviewmind.bot.main as main_mod

        source = inspect.getsource(main_mod.create_dispatcher)
        assert "payment_router" in source

    def test_payment_router_before_query_in_source(self):
        """Payment router must be listed before query (catch-all) in create_dispatcher."""
        import inspect

        import reviewmind.bot.main as main_mod

        source = inspect.getsource(main_mod.create_dispatcher)
        payment_pos = source.index("payment_router")
        query_pos = source.index("query_router")
        assert payment_pos < query_pos


# ══════════════════════════════════════════════════════════════
# Tests — Services Exports
# ══════════════════════════════════════════════════════════════


class TestServicesExports:
    def test_payment_service_exported(self):
        from reviewmind.services import PaymentService

        assert PaymentService is not None

    def test_activation_result_exported(self):
        from reviewmind.services import ActivationResult

        assert ActivationResult is not None

    def test_subscription_days_exported(self):
        from reviewmind.services import SUBSCRIPTION_DAYS

        assert SUBSCRIPTION_DAYS == 30

    def test_subscription_price_stars_exported(self):
        from reviewmind.services import SUBSCRIPTION_PRICE_STARS

        assert SUBSCRIPTION_PRICE_STARS == 1

    def test_subscription_msgs_exported(self):
        from reviewmind.services import SUBSCRIPTION_ACTIVATED_MSG, SUBSCRIPTION_ERROR_MSG

        assert len(SUBSCRIPTION_ACTIVATED_MSG) > 0
        assert len(SUBSCRIPTION_ERROR_MSG) > 0


# ══════════════════════════════════════════════════════════════
# Tests — /help mentions /subscribe
# ══════════════════════════════════════════════════════════════


class TestHelpText:
    def test_help_mentions_subscribe(self):
        from reviewmind.bot.handlers.start import HELP_TEXT

        assert "/subscribe" in HELP_TEXT


# ══════════════════════════════════════════════════════════════
# Tests — Bot Handler Constants
# ══════════════════════════════════════════════════════════════


class TestHandlerConstants:
    def test_invoice_currency(self):
        from reviewmind.bot.handlers.payment import INVOICE_CURRENCY

        assert INVOICE_CURRENCY == "XTR"

    def test_invoice_payload(self):
        from reviewmind.bot.handlers.payment import INVOICE_PAYLOAD

        assert INVOICE_PAYLOAD == "reviewmind_premium_30d"

    def test_subscribe_prompt_msg(self):
        from reviewmind.bot.handlers.payment import SUBSCRIBE_PROMPT_MSG

        assert "{price}" in SUBSCRIBE_PROMPT_MSG


# ══════════════════════════════════════════════════════════════
# Tests — Integration Scenarios
# ══════════════════════════════════════════════════════════════


class TestIntegrationScenarios:
    """End-to-end-style tests for the payment flow."""

    @pytest.mark.asyncio
    async def test_full_subscribe_flow(self):
        """Simulate: /subscribe → invoice → pre_checkout → successful_payment → premium."""
        from reviewmind.bot.handlers.payment import cmd_subscribe, on_pre_checkout, on_successful_payment

        # Step 1: /subscribe sends invoice
        msg = _make_message(user_id=555)
        await cmd_subscribe(msg)
        msg.answer_invoice.assert_awaited_once()

        # Step 2: pre_checkout approved
        pcq = _make_pre_checkout(user_id=555)
        await on_pre_checkout(pcq)
        pcq.answer.assert_awaited_once_with(ok=True)

        # Step 3: successful payment activates sub
        pay_msg = _make_message(user_id=555)
        pay_msg.successful_payment = _make_successful_payment(charge_id="flow_charge")

        mock_engine = MagicMock()
        mock_engine.dispose = AsyncMock()
        mock_session_factory = MagicMock()

        mock_result = ActivationResult(
            success=True,
            expires_at=datetime(2026, 4, 18, 12, 0, tzinfo=timezone.utc),
            subscription_id=99,
        )
        mock_service = MagicMock()
        mock_service.activate_subscription = AsyncMock(return_value=mock_result)

        mock_db_session = MagicMock()
        mock_db_session.commit = AsyncMock()
        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_db_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_session_factory.return_value = mock_session_ctx

        with (
            patch(
                "reviewmind.bot.handlers.payment._get_db_session",
                new_callable=AsyncMock,
                return_value=(mock_engine, mock_session_factory),
            ),
            patch(
                "reviewmind.bot.handlers.payment.PaymentService",
                return_value=mock_service,
            ),
        ):
            await on_successful_payment(pay_msg)

        answer_text = pay_msg.answer.call_args[0][0]
        assert "Подписка активирована" in answer_text

    @pytest.mark.asyncio
    async def test_subscribe_keyboard_callback_data(self):
        """The subscribe_keyboard uses the same callback_data as the handler."""
        from reviewmind.bot.keyboards import SUBSCRIBE_ACTION, subscribe_keyboard

        kb = subscribe_keyboard()
        buttons = [btn for row in kb.inline_keyboard for btn in row]
        callback_data_values = [btn.callback_data for btn in buttons]
        assert SUBSCRIBE_ACTION in callback_data_values

    @pytest.mark.asyncio
    async def test_payment_service_with_real_repos(self):
        """PaymentService instantiates with a session (no crash)."""
        session = _mock_session()
        service = PaymentService(session)
        assert service._user_repo is not None
        assert service._sub_repo is not None

    def test_subscription_model_has_charge_id_unique(self):
        """The Subscription model's telegram_payment_charge_id is unique."""
        from reviewmind.db.models import Subscription

        col = Subscription.__table__.c.telegram_payment_charge_id
        assert col.unique is True

    @pytest.mark.asyncio
    async def test_multiple_payments_extend_subscription_cumulatively(self):
        """Paying N times extends subscription to ~N*30 days from now."""
        num_payments = 3
        now = datetime.now(tz=timezone.utc)

        # Track the running expiry across payments
        current_expiry: datetime | None = None

        for i in range(num_payments):
            # Build user mock reflecting current subscription state
            if current_expiry and current_expiry > now:
                user = _make_user_model(user_id=100, subscription="premium", sub_expires_at=current_expiry)
            else:
                user = _make_user_model(user_id=100, subscription="free", sub_expires_at=current_expiry)

            session = _mock_session()
            # We need the created sub to carry the correct expires_at; capture it from create() args
            async def _fake_create(**kwargs):
                sub = MagicMock()
                sub.id = 20 + i
                sub.expires_at = kwargs["expires_at"]
                sub.amount_stars = kwargs.get("amount_stars", SUBSCRIPTION_PRICE_STARS)
                sub.status = "active"
                return sub

            with patch.object(PaymentService, "__init__", lambda self, s: None):
                service = PaymentService(session)
                service._session = session
                service._sub_repo = MagicMock()
                service._sub_repo.get_by_charge_id = AsyncMock(return_value=None)
                service._sub_repo.create = AsyncMock(side_effect=_fake_create)
                service._user_repo = MagicMock()
                service._user_repo.get_or_create = AsyncMock(return_value=(user, False))
                service._user_repo.update = AsyncMock(return_value=user)

                result = await service.activate_subscription(
                    user_id=100, telegram_payment_charge_id=f"charge_multi_{i}",
                )

            assert result.success is True
            assert result.already_active is False
            current_expiry = result.expires_at

        # After 3 payments the expiry should be ~90 days from now
        expected_min = now + timedelta(days=SUBSCRIPTION_DAYS * num_payments - 1)
        expected_max = now + timedelta(days=SUBSCRIPTION_DAYS * num_payments + 1)
        assert expected_min <= current_expiry <= expected_max, (
            f"After {num_payments} payments, expected expiry near "
            f"{expected_min.isoformat()} – {expected_max.isoformat()}, "
            f"got {current_expiry.isoformat()}"
        )

    @pytest.mark.asyncio
    async def test_payment_after_limit_reached(self):
        """Scenario: user hits limit, subscribes, then can query again."""
        from reviewmind.services.limit_service import LimitService

        session = _mock_session()
        premium_user = _make_user_model(user_id=300, subscription="premium")

        service = LimitService(session, admin_user_ids=[])
        with patch.object(service, "_user_repo") as mock_repo:
            mock_repo.get_by_id = AsyncMock(return_value=premium_user)
            result = await service.check_limit(300)

        assert result.allowed is True
        assert result.reason == "premium"
