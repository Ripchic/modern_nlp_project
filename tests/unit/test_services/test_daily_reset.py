"""Unit tests for TASK-039 — Daily limit reset (date-based keying + Celery Beat safety net).

Tests verify:
- Date-based keying: new UTC day → fresh counter automatically (no row for new date)
- Old user_limits rows preserved for analytics (not deleted)
- Midnight UTC transition handled correctly
- Premium users unaffected by limits on any day
- _daily_reset_limits() uses UTC (not local time)
- Beat task is safety net; primary mechanism is date-keyed counters
"""

from __future__ import annotations

from datetime import date, timedelta, timezone
from datetime import datetime as dt
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reviewmind.services.limit_service import (
    LimitService,
)

# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════


def _mock_session() -> MagicMock:
    session = MagicMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.flush = AsyncMock()
    session.close = AsyncMock()
    return session


def _free_user(user_id: int = 100) -> MagicMock:
    user = MagicMock()
    user.user_id = user_id
    user.subscription = "free"
    user.is_admin = False
    return user


def _premium_user(user_id: int = 200) -> MagicMock:
    user = MagicMock()
    user.user_id = user_id
    user.subscription = "premium"
    user.is_admin = False
    return user


def _limit_row(user_id: int, used: int, row_date: date | None = None) -> MagicMock:
    row = MagicMock()
    row.user_id = user_id
    row.date = row_date or dt.now(tz=timezone.utc).date()
    row.requests_used = used
    return row


# ══════════════════════════════════════════════════════════════
# TestDateBasedKeying
# ══════════════════════════════════════════════════════════════


class TestDateBasedKeying:
    """New UTC day → no row for today → requests_used == 0 → allowed."""

    @pytest.mark.asyncio
    async def test_new_day_no_row_means_zero_usage(self):
        """When there's no user_limits row for today, usage is 0."""
        session = _mock_session()
        service = LimitService(session, admin_user_ids=[])
        service._user_repo.get_by_id = AsyncMock(return_value=_free_user(100))
        service._limit_repo.get = AsyncMock(return_value=None)

        result = await service.check_limit(100)
        assert result.allowed is True
        assert result.requests_used == 0

    @pytest.mark.asyncio
    async def test_exhausted_yesterday_fresh_today(self):
        """User hit limit yesterday; today is a new date → allowed."""
        session = _mock_session()
        service = LimitService(session, admin_user_ids=[])
        service._user_repo.get_by_id = AsyncMock(return_value=_free_user(100))

        today = dt.now(tz=timezone.utc).date()

        # check_limit asks for *today*, which returns None (no row)
        service._limit_repo.get = AsyncMock(return_value=None)

        result = await service.check_limit(100)
        assert result.allowed is True
        assert result.requests_used == 0

        # Confirm it queried with today's date (not yesterday)
        service._limit_repo.get.assert_called_once_with(100, today)

    @pytest.mark.asyncio
    async def test_today_method_returns_utc_date(self):
        """LimitService._today() always returns UTC date."""
        service = LimitService(_mock_session(), admin_user_ids=[])
        expected = dt.now(tz=timezone.utc).date()
        assert service._today() == expected

    @pytest.mark.asyncio
    async def test_date_keying_is_per_user(self):
        """Different users have independent daily counters."""
        session = _mock_session()
        service = LimitService(session, admin_user_ids=[])
        service._user_repo.get_by_id = AsyncMock(return_value=_free_user())

        today = service._today()

        # User 100: 10 used → blocked
        service._limit_repo.get = AsyncMock(return_value=_limit_row(100, used=10, row_date=today))
        r1 = await service.check_limit(100)
        assert r1.allowed is False

        # User 200: 0 used → allowed
        service._limit_repo.get = AsyncMock(return_value=None)
        r2 = await service.check_limit(200)
        assert r2.allowed is True


# ══════════════════════════════════════════════════════════════
# TestMidnightTransition
# ══════════════════════════════════════════════════════════════


class TestMidnightTransition:
    """Verify correct behavior around UTC midnight."""

    @pytest.mark.asyncio
    async def test_mock_day_change(self):
        """Simulated day-change: _today() returns different dates → fresh counter."""
        session = _mock_session()
        service = LimitService(session, admin_user_ids=[])
        service._user_repo.get_by_id = AsyncMock(return_value=_free_user(100))

        day1 = date(2026, 3, 18)
        day2 = date(2026, 3, 19)

        # Day 1: 10 requests used → blocked
        service._today = lambda: day1  # type: ignore[assignment]
        service._limit_repo.get = AsyncMock(return_value=_limit_row(100, used=10, row_date=day1))
        r1 = await service.check_limit(100)
        assert r1.allowed is False

        # Day 2: no row for new date → fresh counter → allowed
        service._today = lambda: day2  # type: ignore[assignment]
        service._limit_repo.get = AsyncMock(return_value=None)
        r2 = await service.check_limit(100)
        assert r2.allowed is True
        assert r2.requests_used == 0

    @pytest.mark.asyncio
    async def test_limit_repo_queried_with_correct_date(self):
        """check_limit passes _today() to repo.get."""
        session = _mock_session()
        service = LimitService(session, admin_user_ids=[])
        service._user_repo.get_by_id = AsyncMock(return_value=_free_user(100))
        service._limit_repo.get = AsyncMock(return_value=None)

        forced_date = date(2026, 6, 15)
        service._today = lambda: forced_date  # type: ignore[assignment]

        await service.check_limit(100)
        service._limit_repo.get.assert_called_once_with(100, forced_date)

    @pytest.mark.asyncio
    async def test_increment_uses_today_date(self):
        """increment passes _today() to repo.increment."""
        session = _mock_session()
        service = LimitService(session, admin_user_ids=[])
        service._user_repo.get_or_create = AsyncMock(return_value=(_free_user(100), False))
        service._limit_repo.increment = AsyncMock(return_value=_limit_row(100, used=1))

        forced_date = date(2026, 12, 31)
        service._today = lambda: forced_date  # type: ignore[assignment]

        await service.increment(100)
        service._limit_repo.increment.assert_called_once_with(100, forced_date)


# ══════════════════════════════════════════════════════════════
# TestOldRowsPreserved
# ══════════════════════════════════════════════════════════════


class TestOldRowsPreserved:
    """Old user_limits rows are NOT deleted (analytics)."""

    @pytest.mark.asyncio
    async def test_two_days_two_rows(self):
        """Simulating two separate days produces two independent lookups."""
        session = _mock_session()
        service = LimitService(session, admin_user_ids=[])
        service._user_repo.get_by_id = AsyncMock(return_value=_free_user(100))

        day1 = date(2026, 3, 18)
        day2 = date(2026, 3, 19)

        # Day 1 query
        service._today = lambda: day1  # type: ignore[assignment]
        service._limit_repo.get = AsyncMock(return_value=_limit_row(100, used=10, row_date=day1))
        r1 = await service.check_limit(100)
        assert r1.allowed is False

        # Day 2 query — repo returns None for new date (old row untouched)
        service._today = lambda: day2  # type: ignore[assignment]
        service._limit_repo.get = AsyncMock(return_value=None)
        r2 = await service.check_limit(100)
        assert r2.allowed is True

    @pytest.mark.asyncio
    async def test_old_rows_not_deleted_by_service(self):
        """LimitService never calls delete on old user_limits rows."""
        session = _mock_session()
        service = LimitService(session, admin_user_ids=[])
        service._user_repo.get_by_id = AsyncMock(return_value=_free_user(100))
        service._limit_repo.get = AsyncMock(return_value=None)

        for delta in range(5):
            forced = date(2026, 3, 15) + timedelta(days=delta)
            service._today = lambda d=forced: d  # type: ignore[assignment]
            await service.check_limit(100)

        # The repo has no delete method called
        assert not hasattr(service._limit_repo, "delete") or not service._limit_repo.delete.called


# ══════════════════════════════════════════════════════════════
# TestPremiumNotAffected
# ══════════════════════════════════════════════════════════════


class TestPremiumNotAffected:
    """Premium users bypass limits regardless of day."""

    @pytest.mark.asyncio
    async def test_premium_bypass_every_day(self):
        session = _mock_session()
        service = LimitService(session, admin_user_ids=[])
        service._user_repo.get_by_id = AsyncMock(return_value=_premium_user(200))

        for delta in range(7):
            forced = date(2026, 3, 15) + timedelta(days=delta)
            service._today = lambda d=forced: d  # type: ignore[assignment]
            result = await service.check_limit(200)
            assert result.allowed is True
            assert result.reason == "premium"

    @pytest.mark.asyncio
    async def test_premium_does_not_check_repo(self):
        """Premium users short-circuit before touching user_limits repo."""
        session = _mock_session()
        service = LimitService(session, admin_user_ids=[])
        service._user_repo.get_by_id = AsyncMock(return_value=_premium_user(200))
        service._limit_repo.get = AsyncMock()

        await service.check_limit(200)
        service._limit_repo.get.assert_not_called()


# ══════════════════════════════════════════════════════════════
# TestDailyResetLimitsTask
# ══════════════════════════════════════════════════════════════


class TestDailyResetLimitsTask:
    """Test the _daily_reset_limits async function (Celery Beat safety net)."""

    @pytest.mark.asyncio
    async def test_uses_utc_date(self):
        """Ensure _daily_reset_limits uses UTC date, not local."""
        from reviewmind.workers.tasks import _daily_reset_limits

        mock_engine = MagicMock()
        mock_engine.dispose = AsyncMock()

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.commit = AsyncMock()

        # Return empty result (no rows to reset)
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute = AsyncMock(return_value=mock_result)

        mock_factory = MagicMock(return_value=mock_session)

        with (
            patch("reviewmind.workers.tasks.create_async_engine", return_value=mock_engine),
            patch("reviewmind.workers.tasks.async_sessionmaker", return_value=mock_factory),
            patch("reviewmind.workers.tasks.datetime") as mock_datetime,
            patch("reviewmind.config.settings"),
        ):
            mock_now = MagicMock()
            mock_now.date.return_value = date(2026, 3, 19)
            mock_datetime.now.return_value = mock_now
            mock_datetime.side_effect = lambda *a, **kw: dt(*a, **kw)

            result = await _daily_reset_limits()

        assert result["date"] == "2026-03-19"
        mock_datetime.now.assert_called_once_with(timezone.utc)

    @pytest.mark.asyncio
    async def test_resets_todays_rows_only(self):
        """Safety net zeros today's pre-created rows, not old ones."""
        from reviewmind.workers.tasks import _daily_reset_limits

        mock_engine = MagicMock()
        mock_engine.dispose = AsyncMock()

        # Simulate two rows with requests_used > 0 for today
        row1 = MagicMock()
        row1.requests_used = 5
        row2 = MagicMock()
        row2.requests_used = 2

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.commit = AsyncMock()

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [row1, row2]
        mock_session.execute = AsyncMock(return_value=mock_result)

        mock_factory = MagicMock(return_value=mock_session)

        with (
            patch("reviewmind.workers.tasks.create_async_engine", return_value=mock_engine),
            patch("reviewmind.workers.tasks.async_sessionmaker", return_value=mock_factory),
            patch("reviewmind.config.settings"),
        ):
            result = await _daily_reset_limits()

        assert result["rows_reset"] == 2
        assert row1.requests_used == 0
        assert row2.requests_used == 0
        mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_rows_to_reset(self):
        """Empty result → rows_reset == 0, no error."""
        from reviewmind.workers.tasks import _daily_reset_limits

        mock_engine = MagicMock()
        mock_engine.dispose = AsyncMock()

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.commit = AsyncMock()

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute = AsyncMock(return_value=mock_result)

        mock_factory = MagicMock(return_value=mock_session)

        with (
            patch("reviewmind.workers.tasks.create_async_engine", return_value=mock_engine),
            patch("reviewmind.workers.tasks.async_sessionmaker", return_value=mock_factory),
            patch("reviewmind.config.settings"),
        ):
            result = await _daily_reset_limits()

        assert result["rows_reset"] == 0

    @pytest.mark.asyncio
    async def test_db_error_graceful(self):
        """DB error → caught, rows_reset == 0, engine disposed."""
        from reviewmind.workers.tasks import _daily_reset_limits

        mock_engine = MagicMock()
        mock_engine.dispose = AsyncMock()

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.execute = AsyncMock(side_effect=RuntimeError("DB offline"))

        mock_factory = MagicMock(return_value=mock_session)

        with (
            patch("reviewmind.workers.tasks.create_async_engine", return_value=mock_engine),
            patch("reviewmind.workers.tasks.async_sessionmaker", return_value=mock_factory),
            patch("reviewmind.config.settings"),
        ):
            result = await _daily_reset_limits()

        assert result["rows_reset"] == 0
        mock_engine.dispose.assert_called_once()

    @pytest.mark.asyncio
    async def test_engine_disposed(self):
        """Engine is always disposed, even on success."""
        from reviewmind.workers.tasks import _daily_reset_limits

        mock_engine = MagicMock()
        mock_engine.dispose = AsyncMock()

        mock_session = MagicMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.commit = AsyncMock()

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute = AsyncMock(return_value=mock_result)

        mock_factory = MagicMock(return_value=mock_session)

        with (
            patch("reviewmind.workers.tasks.create_async_engine", return_value=mock_engine),
            patch("reviewmind.workers.tasks.async_sessionmaker", return_value=mock_factory),
            patch("reviewmind.config.settings"),
        ):
            await _daily_reset_limits()

        mock_engine.dispose.assert_called_once()


# ══════════════════════════════════════════════════════════════
# TestTaskRegistration
# ══════════════════════════════════════════════════════════════


class TestTaskRegistration:
    def test_daily_reset_task_registered(self):
        from reviewmind.workers.tasks import daily_reset_limits_task

        assert daily_reset_limits_task.name == "reviewmind.daily_reset_limits"

    def test_daily_reset_task_max_retries_zero(self):
        from reviewmind.workers.tasks import daily_reset_limits_task

        assert daily_reset_limits_task.max_retries == 0

    def test_beat_schedule_includes_task(self):
        from reviewmind.workers.beat_schedule import BEAT_SCHEDULE, DAILY_RESET_LIMITS_TASK

        assert "daily-reset-limits" in BEAT_SCHEDULE
        assert BEAT_SCHEDULE["daily-reset-limits"]["task"] == DAILY_RESET_LIMITS_TASK

    def test_beat_schedule_midnight_utc(self):
        from reviewmind.workers.beat_schedule import DAILY_RESET_SCHEDULE

        assert DAILY_RESET_SCHEDULE._orig_minute == "0"
        assert DAILY_RESET_SCHEDULE._orig_hour == "0"


# ══════════════════════════════════════════════════════════════
# TestIntegrationScenarios
# ══════════════════════════════════════════════════════════════


class TestIntegrationScenarios:
    """End-to-end scenario tests matching TASK-039 test_steps."""

    @pytest.mark.asyncio
    async def test_step1_exhaust_limit_11th_blocked(self):
        """Step 1: 10 requests → all pass; 11th → blocked."""
        session = _mock_session()
        service = LimitService(session, admin_user_ids=[])
        service._user_repo.get_by_id = AsyncMock(return_value=_free_user(100))

        for used in range(10):
            service._limit_repo.get = AsyncMock(return_value=_limit_row(100, used=used))
            r = await service.check_limit(100)
            assert r.allowed is True, f"Request #{used + 1} should be allowed"

        # 11th → blocked
        service._limit_repo.get = AsyncMock(return_value=_limit_row(100, used=10))
        r = await service.check_limit(100)
        assert r.allowed is False
        assert r.reason == "limit_reached"

    @pytest.mark.asyncio
    async def test_step2_3_next_day_allowed(self):
        """Steps 2-3: Switch to next day → request allowed (new counter)."""
        session = _mock_session()
        service = LimitService(session, admin_user_ids=[])
        service._user_repo.get_by_id = AsyncMock(return_value=_free_user(100))

        day1 = date(2026, 3, 18)
        day2 = date(2026, 3, 19)

        # Day 1: blocked
        service._today = lambda: day1  # type: ignore[assignment]
        service._limit_repo.get = AsyncMock(return_value=_limit_row(100, used=10, row_date=day1))
        r = await service.check_limit(100)
        assert r.allowed is False

        # Day 2: fresh counter
        service._today = lambda: day2  # type: ignore[assignment]
        service._limit_repo.get = AsyncMock(return_value=None)
        r = await service.check_limit(100)
        assert r.allowed is True
        assert r.requests_used == 0

    @pytest.mark.asyncio
    async def test_step4_two_rows_for_two_dates(self):
        """Step 4: Two separate date-keyed rows exist independently."""
        session = _mock_session()
        service = LimitService(session, admin_user_ids=[])
        service._user_repo.get_by_id = AsyncMock(return_value=_free_user(100))
        service._user_repo.get_or_create = AsyncMock(return_value=(_free_user(100), False))

        day1 = date(2026, 3, 18)
        day2 = date(2026, 3, 19)

        # Day 1: increment 3 times
        service._today = lambda: day1  # type: ignore[assignment]
        incremented = _limit_row(100, used=3, row_date=day1)
        service._limit_repo.increment = AsyncMock(return_value=incremented)
        await service.increment(100)
        service._limit_repo.increment.assert_called_with(100, day1)

        # Day 2: increment
        service._today = lambda: day2  # type: ignore[assignment]
        incremented2 = _limit_row(100, used=1, row_date=day2)
        service._limit_repo.increment = AsyncMock(return_value=incremented2)
        await service.increment(100)
        service._limit_repo.increment.assert_called_with(100, day2)

    @pytest.mark.asyncio
    async def test_step5_premium_no_limit_any_day(self):
        """Step 5: Premium user is never limited."""
        session = _mock_session()
        service = LimitService(session, admin_user_ids=[])
        service._user_repo.get_by_id = AsyncMock(return_value=_premium_user(200))

        for delta in range(3):
            forced = date(2026, 3, 18) + timedelta(days=delta)
            service._today = lambda d=forced: d  # type: ignore[assignment]
            r = await service.check_limit(200)
            assert r.allowed is True
            assert r.reason == "premium"

    @pytest.mark.asyncio
    async def test_admin_no_limit_any_day(self):
        """Admin user bypasses limits on all days too."""
        session = _mock_session()
        service = LimitService(session, admin_user_ids=[42])

        for delta in range(5):
            forced = date(2026, 3, 18) + timedelta(days=delta)
            service._today = lambda d=forced: d  # type: ignore[assignment]
            r = await service.check_limit(42)
            assert r.allowed is True
            assert r.reason == "admin"

    def test_limit_service_today_returns_utc(self):
        """_today() always returns UTC date."""
        service = LimitService(_mock_session(), admin_user_ids=[])
        expected = dt.now(tz=timezone.utc).date()
        assert service._today() == expected
