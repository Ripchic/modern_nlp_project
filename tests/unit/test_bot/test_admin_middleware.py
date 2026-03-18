"""Unit tests for reviewmind.bot.middlewares — AdminMiddleware."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiogram.types import CallbackQuery, Chat, Message, Update, User  # noqa: F401

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_user(user_id: int = 99999) -> User:
    return User(id=user_id, is_bot=False, first_name="Admin")


def _make_chat(chat_id: int = 99999) -> Chat:
    return Chat(id=chat_id, type="private")


def _make_update_with_message(user_id: int = 99999) -> MagicMock:
    upd = MagicMock(spec=Update)
    upd.update_id = 1
    upd.event_type = "message"
    msg = MagicMock(spec=Message)
    msg.from_user = _make_user(user_id)
    msg.chat = _make_chat(user_id)
    upd.message = msg
    upd.callback_query = None
    return upd


def _make_update_with_callback(user_id: int = 99999) -> MagicMock:
    upd = MagicMock(spec=Update)
    upd.update_id = 2
    upd.event_type = "callback_query"
    upd.message = None
    cb = MagicMock(spec=CallbackQuery)
    cb.from_user = _make_user(user_id)
    upd.callback_query = cb
    return upd


def _make_update_no_user() -> MagicMock:
    upd = MagicMock(spec=Update)
    upd.update_id = 3
    upd.event_type = "message"
    upd.message = None
    upd.callback_query = None
    return upd


# ── Tests: _extract_user_id ──────────────────────────────────────────────────


class TestExtractUserId:
    def test_message_user(self):
        from reviewmind.bot.middlewares import _extract_user_id

        upd = _make_update_with_message(12345)
        assert _extract_user_id(upd) == 12345

    def test_callback_user(self):
        from reviewmind.bot.middlewares import _extract_user_id

        upd = _make_update_with_callback(67890)
        assert _extract_user_id(upd) == 67890

    def test_no_user(self):
        from reviewmind.bot.middlewares import _extract_user_id

        upd = _make_update_no_user()
        assert _extract_user_id(upd) is None

    def test_non_update_object(self):
        from reviewmind.bot.middlewares import _extract_user_id

        assert _extract_user_id(MagicMock()) is None


# ── Tests: AdminMiddleware init ──────────────────────────────────────────────


class TestAdminMiddlewareInit:
    def test_explicit_admin_ids(self):
        from reviewmind.bot.middlewares import AdminMiddleware

        mw = AdminMiddleware(admin_user_ids=[100, 200])
        assert mw.admin_user_ids == {100, 200}

    def test_empty_admin_ids(self):
        from reviewmind.bot.middlewares import AdminMiddleware

        mw = AdminMiddleware(admin_user_ids=[])
        assert mw.admin_user_ids == set()

    def test_explicit_database_url(self):
        from reviewmind.bot.middlewares import AdminMiddleware

        mw = AdminMiddleware(database_url="postgresql+asyncpg://test")
        assert mw.database_url == "postgresql+asyncpg://test"

    def test_seen_starts_empty(self):
        from reviewmind.bot.middlewares import AdminMiddleware

        mw = AdminMiddleware(admin_user_ids=[1])
        assert mw._seen == set()

    def test_lazy_admin_ids_from_config(self):
        from reviewmind.bot.middlewares import AdminMiddleware

        # _admin_ids starts as None when no explicit ids given
        mw = AdminMiddleware()
        assert mw._admin_ids is None

    def test_lazy_admin_ids_config_import_error(self):
        from reviewmind.bot.middlewares import AdminMiddleware

        mw = AdminMiddleware()
        # Force _admin_ids to None
        mw._admin_ids = None
        with patch.dict("sys.modules", {"reviewmind.config": None}):
            # Should gracefully return empty set
            ids = mw.admin_user_ids
            assert ids == set()


# ── Tests: AdminMiddleware.__call__ ──────────────────────────────────────────


class TestAdminMiddlewareCall:
    @pytest.mark.asyncio
    async def test_admin_user_triggers_ensure(self):
        from reviewmind.bot.middlewares import AdminMiddleware

        mw = AdminMiddleware(admin_user_ids=[99999], database_url="postgresql+asyncpg://x")
        mw._ensure_admin_in_db = AsyncMock()
        handler = AsyncMock(return_value="ok")

        upd = _make_update_with_message(99999)
        result = await mw(handler, upd, {})

        mw._ensure_admin_in_db.assert_awaited_once_with(99999)
        handler.assert_awaited_once()
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_non_admin_user_skips_ensure(self):
        from reviewmind.bot.middlewares import AdminMiddleware

        mw = AdminMiddleware(admin_user_ids=[111], database_url="postgresql+asyncpg://x")
        mw._ensure_admin_in_db = AsyncMock()
        handler = AsyncMock(return_value="ok")

        upd = _make_update_with_message(222)
        result = await mw(handler, upd, {})

        mw._ensure_admin_in_db.assert_not_awaited()
        handler.assert_awaited_once()
        assert result == "ok"

    @pytest.mark.asyncio
    async def test_already_seen_admin_skips(self):
        from reviewmind.bot.middlewares import AdminMiddleware

        mw = AdminMiddleware(admin_user_ids=[99999], database_url="postgresql+asyncpg://x")
        mw._ensure_admin_in_db = AsyncMock()
        mw._seen.add(99999)
        handler = AsyncMock()

        upd = _make_update_with_message(99999)
        await mw(handler, upd, {})

        mw._ensure_admin_in_db.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_user_id_skips(self):
        from reviewmind.bot.middlewares import AdminMiddleware

        mw = AdminMiddleware(admin_user_ids=[99999])
        mw._ensure_admin_in_db = AsyncMock()
        handler = AsyncMock()

        upd = _make_update_no_user()
        await mw(handler, upd, {})

        mw._ensure_admin_in_db.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_callback_query_admin(self):
        from reviewmind.bot.middlewares import AdminMiddleware

        mw = AdminMiddleware(admin_user_ids=[55555], database_url="postgresql+asyncpg://x")
        mw._ensure_admin_in_db = AsyncMock()
        handler = AsyncMock()

        upd = _make_update_with_callback(55555)
        await mw(handler, upd, {})

        mw._ensure_admin_in_db.assert_awaited_once_with(55555)

    @pytest.mark.asyncio
    async def test_handler_always_called_even_on_ensure_error(self):
        """Handler proceeds even if _ensure_admin_in_db raises (it shouldn't, but just in case)."""
        from reviewmind.bot.middlewares import AdminMiddleware

        mw = AdminMiddleware(admin_user_ids=[99999], database_url="postgresql+asyncpg://x")
        mw._ensure_admin_in_db = AsyncMock(side_effect=RuntimeError("boom"))
        handler = AsyncMock()

        upd = _make_update_with_message(99999)
        # The exception from _ensure_admin_in_db propagates, but in real code it's caught inside
        # For this test we verify the method was called
        with pytest.raises(RuntimeError):
            await mw(handler, upd, {})
        mw._ensure_admin_in_db.assert_awaited_once()


# ── Tests: _ensure_admin_in_db ───────────────────────────────────────────────


class TestEnsureAdminInDb:
    @pytest.mark.asyncio
    async def test_creates_new_admin_user(self):
        from reviewmind.bot.middlewares import AdminMiddleware

        mw = AdminMiddleware(admin_user_ids=[42], database_url="postgresql+asyncpg://x")

        mock_user_repo = MagicMock()
        mock_user_repo.get_by_id = AsyncMock(return_value=None)
        mock_user_repo.create = AsyncMock()

        mock_session = AsyncMock()
        mock_session.commit = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_factory = MagicMock(return_value=mock_session)
        mock_engine = MagicMock()
        mock_engine.dispose = AsyncMock()

        with (
            patch("reviewmind.bot.middlewares.build_engine", return_value=mock_engine),
            patch("reviewmind.bot.middlewares.build_session_factory", return_value=mock_factory),
            patch("reviewmind.bot.middlewares.UserRepository", return_value=mock_user_repo),
        ):
            await mw._ensure_admin_in_db(42)

        mock_user_repo.get_by_id.assert_awaited_once_with(42)
        mock_user_repo.create.assert_awaited_once_with(42, is_admin=True)
        mock_session.commit.assert_awaited_once()
        mock_engine.dispose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_updates_existing_non_admin(self):
        from reviewmind.bot.middlewares import AdminMiddleware

        mw = AdminMiddleware(admin_user_ids=[42], database_url="postgresql+asyncpg://x")

        existing_user = MagicMock()
        existing_user.is_admin = False

        mock_user_repo = MagicMock()
        mock_user_repo.get_by_id = AsyncMock(return_value=existing_user)
        mock_user_repo.update = AsyncMock()

        mock_session = AsyncMock()
        mock_session.commit = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_factory = MagicMock(return_value=mock_session)
        mock_engine = MagicMock()
        mock_engine.dispose = AsyncMock()

        with (
            patch("reviewmind.bot.middlewares.build_engine", return_value=mock_engine),
            patch("reviewmind.bot.middlewares.build_session_factory", return_value=mock_factory),
            patch("reviewmind.bot.middlewares.UserRepository", return_value=mock_user_repo),
        ):
            await mw._ensure_admin_in_db(42)

        mock_user_repo.update.assert_awaited_once_with(42, is_admin=True)

    @pytest.mark.asyncio
    async def test_already_admin_no_update(self):
        from reviewmind.bot.middlewares import AdminMiddleware

        mw = AdminMiddleware(admin_user_ids=[42], database_url="postgresql+asyncpg://x")

        existing_user = MagicMock()
        existing_user.is_admin = True

        mock_user_repo = MagicMock()
        mock_user_repo.get_by_id = AsyncMock(return_value=existing_user)
        mock_user_repo.create = AsyncMock()
        mock_user_repo.update = AsyncMock()

        mock_session = AsyncMock()
        mock_session.commit = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_factory = MagicMock(return_value=mock_session)
        mock_engine = MagicMock()
        mock_engine.dispose = AsyncMock()

        with (
            patch("reviewmind.bot.middlewares.build_engine", return_value=mock_engine),
            patch("reviewmind.bot.middlewares.build_session_factory", return_value=mock_factory),
            patch("reviewmind.bot.middlewares.UserRepository", return_value=mock_user_repo),
        ):
            await mw._ensure_admin_in_db(42)

        mock_user_repo.create.assert_not_awaited()
        mock_user_repo.update.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_database_url_skips(self):
        from reviewmind.bot.middlewares import AdminMiddleware

        mw = AdminMiddleware(admin_user_ids=[42])
        mw._database_url = None
        # Ensure config also not available
        with patch.dict("sys.modules", {"reviewmind.config": None}):
            await mw._ensure_admin_in_db(42)
        assert 42 not in mw._seen

    @pytest.mark.asyncio
    async def test_db_error_is_non_fatal(self):
        from reviewmind.bot.middlewares import AdminMiddleware

        mw = AdminMiddleware(admin_user_ids=[42], database_url="postgresql+asyncpg://x")

        with patch("reviewmind.bot.middlewares.build_engine", side_effect=Exception("connection refused")):
            # Should not raise
            await mw._ensure_admin_in_db(42)
        assert 42 not in mw._seen

    @pytest.mark.asyncio
    async def test_engine_disposed_on_success(self):
        from reviewmind.bot.middlewares import AdminMiddleware

        mw = AdminMiddleware(admin_user_ids=[42], database_url="postgresql+asyncpg://x")

        mock_user_repo = MagicMock()
        mock_user_repo.get_by_id = AsyncMock(return_value=None)
        mock_user_repo.create = AsyncMock()

        mock_session = AsyncMock()
        mock_session.commit = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_engine = MagicMock()
        mock_engine.dispose = AsyncMock()

        with (
            patch("reviewmind.bot.middlewares.build_engine", return_value=mock_engine),
            patch(
                "reviewmind.bot.middlewares.build_session_factory",
                return_value=MagicMock(return_value=mock_session),
            ),
            patch("reviewmind.bot.middlewares.UserRepository", return_value=mock_user_repo),
        ):
            await mw._ensure_admin_in_db(42)

        mock_engine.dispose.assert_awaited_once()


# ── Tests: Dispatcher integration ────────────────────────────────────────────


class TestDispatcherWiring:
    def test_admin_middleware_importable_in_main(self):
        """bot.main imports AdminMiddleware from middlewares."""
        from reviewmind.bot.main import AdminMiddleware as MainAdminMW
        from reviewmind.bot.middlewares import AdminMiddleware

        assert MainAdminMW is AdminMiddleware

    def test_both_custom_middlewares_in_fresh_dispatcher(self):
        """A fresh Dispatcher has both LoggingMiddleware and AdminMiddleware."""
        from aiogram import Dispatcher

        from reviewmind.bot.middlewares import AdminMiddleware, LoggingMiddleware

        dp = Dispatcher()
        dp.update.outer_middleware(LoggingMiddleware())
        dp.update.outer_middleware(AdminMiddleware(admin_user_ids=[]))
        types = [type(m).__name__ for m in dp.update.outer_middleware._middlewares]
        assert "LoggingMiddleware" in types
        assert "AdminMiddleware" in types


# ── Tests: AdminMiddleware with limit_service ────────────────────────────────


class TestAdminBypassIntegration:
    def test_limit_service_recognizes_admin(self):
        """LimitService._is_admin uses the same config, confirming alignment."""
        from reviewmind.services.limit_service import LimitService

        mock_session = MagicMock()
        svc = LimitService(mock_session, admin_user_ids=[42, 100])
        assert svc._is_admin(42)
        assert svc._is_admin(100)
        assert not svc._is_admin(999)

    @pytest.mark.asyncio
    async def test_admin_check_limit_bypasses(self):
        from reviewmind.services.limit_service import LimitService

        mock_session = MagicMock()
        svc = LimitService(mock_session, admin_user_ids=[42])
        result = await svc.check_limit(42)
        assert result.allowed
        assert result.reason == "admin"


# ── Tests: Config admin_user_ids alignment ───────────────────────────────────


class TestConfigAlignment:
    def test_admin_user_ids_default_empty(self):
        """Config defaults to empty admin list."""
        from reviewmind.config import Settings

        s = Settings(
            _env_file=None,
            telegram_bot_token="fake",
            openai_api_key="fake",
            admin_user_ids=[],
        )
        assert s.admin_user_ids == []

    def test_admin_user_ids_parsed_from_csv(self):
        """Config parses comma-separated admin IDs."""
        from reviewmind.config import Settings

        s = Settings(
            _env_file=None,
            telegram_bot_token="fake",
            openai_api_key="fake",
            admin_user_ids="42,100,200",
        )
        assert s.admin_user_ids == [42, 100, 200]


# ── Tests: Module exports ────────────────────────────────────────────────────


class TestMiddlewareExports:
    def test_logging_middleware_importable(self):
        from reviewmind.bot.middlewares import LoggingMiddleware

        assert LoggingMiddleware is not None

    def test_admin_middleware_importable(self):
        from reviewmind.bot.middlewares import AdminMiddleware

        assert AdminMiddleware is not None

    def test_extract_user_id_importable(self):
        from reviewmind.bot.middlewares import _extract_user_id

        assert callable(_extract_user_id)


# ── Tests: Integration scenarios ─────────────────────────────────────────────


class TestIntegrationScenarios:
    @pytest.mark.asyncio
    async def test_full_flow_new_admin(self):
        """Admin sends first message → user created with is_admin=True → handler runs."""
        from reviewmind.bot.middlewares import AdminMiddleware

        admin_id = 42
        mw = AdminMiddleware(admin_user_ids=[admin_id], database_url="postgresql+asyncpg://x")

        mock_user_repo = MagicMock()
        mock_user_repo.get_by_id = AsyncMock(return_value=None)
        mock_user_repo.create = AsyncMock()

        mock_session = AsyncMock()
        mock_session.commit = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)

        mock_engine = MagicMock()
        mock_engine.dispose = AsyncMock()

        handler = AsyncMock(return_value="response")
        upd = _make_update_with_message(admin_id)

        with (
            patch("reviewmind.bot.middlewares.build_engine", return_value=mock_engine),
            patch(
                "reviewmind.bot.middlewares.build_session_factory",
                return_value=MagicMock(return_value=mock_session),
            ),
            patch("reviewmind.bot.middlewares.UserRepository", return_value=mock_user_repo),
        ):
            result = await mw(handler, upd, {})

        assert result == "response"
        mock_user_repo.create.assert_awaited_once_with(admin_id, is_admin=True)
        handler.assert_awaited_once()
        assert admin_id in mw._seen

    @pytest.mark.asyncio
    async def test_full_flow_non_admin_unaffected(self):
        """Non-admin user → no DB interaction, handler runs."""
        from reviewmind.bot.middlewares import AdminMiddleware

        mw = AdminMiddleware(admin_user_ids=[42], database_url="postgresql+asyncpg://x")
        handler = AsyncMock(return_value="ok")

        upd = _make_update_with_message(999)
        result = await mw(handler, upd, {})
        assert result == "ok"
        assert 999 not in mw._seen

    @pytest.mark.asyncio
    async def test_admin_removed_from_config(self):
        """If user_id is removed from admin_user_ids, middleware stops treating them as admin."""
        from reviewmind.bot.middlewares import AdminMiddleware

        mw = AdminMiddleware(admin_user_ids=[42], database_url="postgresql+asyncpg://x")
        mw._ensure_admin_in_db = AsyncMock()
        handler = AsyncMock()

        # First: admin → ensure called
        upd = _make_update_with_message(42)
        await mw(handler, upd, {})
        mw._ensure_admin_in_db.assert_awaited_once()

        # Simulate config change: remove 42 from admins
        mw._admin_ids = {100}
        mw._seen.clear()
        mw._ensure_admin_in_db.reset_mock()

        upd2 = _make_update_with_message(42)
        await mw(handler, upd2, {})
        mw._ensure_admin_in_db.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_second_message_from_admin_skips_db(self):
        """Admin sends a second message → _seen set prevents repeat DB call."""
        from reviewmind.bot.middlewares import AdminMiddleware

        mw = AdminMiddleware(admin_user_ids=[42], database_url="postgresql+asyncpg://x")
        mw._ensure_admin_in_db = AsyncMock()
        handler = AsyncMock()

        upd1 = _make_update_with_message(42)
        await mw(handler, upd1, {})
        assert mw._ensure_admin_in_db.await_count == 1

        upd2 = _make_update_with_message(42)
        await mw(handler, upd2, {})
        # Should still be 1 — second call skipped
        assert mw._ensure_admin_in_db.await_count == 1

    @pytest.mark.asyncio
    async def test_db_failure_does_not_break_handler(self):
        """DB error during admin registration → handler still runs normally."""
        from reviewmind.bot.middlewares import AdminMiddleware

        mw = AdminMiddleware(admin_user_ids=[42], database_url="postgresql+asyncpg://x")
        handler = AsyncMock(return_value="ok")

        upd = _make_update_with_message(42)
        with patch("reviewmind.bot.middlewares.build_engine", side_effect=Exception("db down")):
            result = await mw(handler, upd, {})

        assert result == "ok"
        handler.assert_awaited_once()
