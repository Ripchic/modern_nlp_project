"""Unit tests for repositories with mocked AsyncSession (TASK-009)."""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from reviewmind.db.models import Job, QueryLog, Source, Subscription, User, UserLimit
from reviewmind.db.repositories.jobs import JobRepository
from reviewmind.db.repositories.limits import UserLimitRepository
from reviewmind.db.repositories.query_logs import QueryLogRepository
from reviewmind.db.repositories.sources import SourceRepository
from reviewmind.db.repositories.subscriptions import SubscriptionRepository
from reviewmind.db.repositories.users import UserRepository


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_session(scalar_result=None, scalars_result=None):
    """Build a minimal mock AsyncSession."""
    session = MagicMock()
    session.add = MagicMock()
    session.delete = AsyncMock()
    session.flush = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()

    result_mock = MagicMock()
    result_mock.scalar_one_or_none.return_value = scalar_result
    result_mock.scalars.return_value.all.return_value = scalars_result or []

    session.execute = AsyncMock(return_value=result_mock)
    return session


# ---------------------------------------------------------------------------
# TestUserRepository
# ---------------------------------------------------------------------------
class TestUserRepository:
    def test_init(self):
        session = make_session()
        repo = UserRepository(session)
        assert repo._session is session

    @pytest.mark.asyncio
    async def test_get_by_id_found(self):
        user = User(user_id=42)
        session = make_session(scalar_result=user)
        repo = UserRepository(session)
        result = await repo.get_by_id(42)
        assert result is user
        session.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self):
        session = make_session(scalar_result=None)
        repo = UserRepository(session)
        result = await repo.get_by_id(999)
        assert result is None

    @pytest.mark.asyncio
    async def test_create(self):
        session = make_session()
        repo = UserRepository(session)
        user = await repo.create(100, is_admin=True, subscription="premium")
        assert isinstance(user, User)
        assert user.user_id == 100
        assert user.is_admin is True
        assert user.subscription == "premium"
        session.add.assert_called_once_with(user)
        session.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_or_create_existing(self):
        existing = User(user_id=5)
        session = make_session(scalar_result=existing)
        repo = UserRepository(session)
        user, created = await repo.get_or_create(5)
        assert user is existing
        assert created is False

    @pytest.mark.asyncio
    async def test_get_or_create_new(self):
        session = make_session(scalar_result=None)
        repo = UserRepository(session)
        user, created = await repo.get_or_create(99)
        assert created is True
        assert user.user_id == 99

    @pytest.mark.asyncio
    async def test_update_found(self):
        user = User(user_id=1, subscription="free")
        session = make_session(scalar_result=user)
        repo = UserRepository(session)
        result = await repo.update(1, subscription="premium")
        assert result is user
        assert user.subscription == "premium"
        session.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_not_found(self):
        session = make_session(scalar_result=None)
        repo = UserRepository(session)
        result = await repo.update(999, subscription="premium")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_found(self):
        user = User(user_id=7)
        session = make_session(scalar_result=user)
        repo = UserRepository(session)
        result = await repo.delete(7)
        assert result is True
        session.delete.assert_called_once_with(user)

    @pytest.mark.asyncio
    async def test_delete_not_found(self):
        session = make_session(scalar_result=None)
        repo = UserRepository(session)
        result = await repo.delete(999)
        assert result is False


# ---------------------------------------------------------------------------
# TestUserLimitRepository
# ---------------------------------------------------------------------------
class TestUserLimitRepository:
    @pytest.mark.asyncio
    async def test_get_found(self):
        today = date.today()
        row = UserLimit(user_id=1, date=today, requests_used=3)
        session = make_session(scalar_result=row)
        repo = UserLimitRepository(session)
        result = await repo.get(1, today)
        assert result is row

    @pytest.mark.asyncio
    async def test_get_not_found(self):
        session = make_session(scalar_result=None)
        repo = UserLimitRepository(session)
        result = await repo.get(1, date.today())
        assert result is None

    @pytest.mark.asyncio
    async def test_get_or_create_existing(self):
        today = date.today()
        row = UserLimit(user_id=1, date=today, requests_used=5)
        session = make_session(scalar_result=row)
        repo = UserLimitRepository(session)
        result, created = await repo.get_or_create(1, today)
        assert result is row
        assert created is False

    @pytest.mark.asyncio
    async def test_get_or_create_new(self):
        today = date.today()
        session = make_session(scalar_result=None)
        repo = UserLimitRepository(session)
        result, created = await repo.get_or_create(1, today)
        assert created is True
        assert result.user_id == 1
        assert result.requests_used == 0

    @pytest.mark.asyncio
    async def test_increment_existing(self):
        today = date.today()
        row = UserLimit(user_id=1, date=today, requests_used=2)
        session = make_session(scalar_result=row)
        repo = UserLimitRepository(session)
        result = await repo.increment(1, today, by=3)
        assert result.requests_used == 5

    @pytest.mark.asyncio
    async def test_increment_default_by_one(self):
        today = date.today()
        row = UserLimit(user_id=1, date=today, requests_used=0)
        session = make_session(scalar_result=row)
        repo = UserLimitRepository(session)
        result = await repo.increment(1, today)
        assert result.requests_used == 1

    @pytest.mark.asyncio
    async def test_reset(self):
        today = date.today()
        row = UserLimit(user_id=1, date=today, requests_used=10)
        session = make_session(scalar_result=row)
        repo = UserLimitRepository(session)
        result = await repo.reset(1, today)
        assert result.requests_used == 0

    @pytest.mark.asyncio
    async def test_reset_not_found(self):
        session = make_session(scalar_result=None)
        repo = UserLimitRepository(session)
        result = await repo.reset(999, date.today())
        assert result is None


# ---------------------------------------------------------------------------
# TestSourceRepository
# ---------------------------------------------------------------------------
class TestSourceRepository:
    @pytest.mark.asyncio
    async def test_get_by_id_found(self):
        source = Source(id=1, source_url="https://test.com", source_type="web")
        session = make_session(scalar_result=source)
        repo = SourceRepository(session)
        result = await repo.get_by_id(1)
        assert result is source

    @pytest.mark.asyncio
    async def test_get_by_url_not_found(self):
        session = make_session(scalar_result=None)
        repo = SourceRepository(session)
        result = await repo.get_by_url("https://not-exists.com")
        assert result is None

    @pytest.mark.asyncio
    async def test_create_minimal(self):
        session = make_session()
        repo = SourceRepository(session)
        source = await repo.create("https://example.com", "web")
        assert source.source_url == "https://example.com"
        assert source.source_type == "web"
        assert source.is_sponsored is False
        assert source.is_curated is False
        session.add.assert_called_once()
        session.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_with_all_fields(self):
        session = make_session()
        repo = SourceRepository(session)
        now = datetime.now(tz=timezone.utc)
        source = await repo.create(
            "https://youtube.com/watch?v=abc",
            "youtube",
            product_query="best headphones",
            parsed_at=now,
            is_sponsored=True,
            is_curated=False,
            language="ru",
            author="Reviewer",
        )
        assert source.is_sponsored is True
        assert source.product_query == "best headphones"
        assert source.author == "Reviewer"

    @pytest.mark.asyncio
    async def test_get_or_create_existing(self):
        source = Source(id=1, source_url="https://exists.com", source_type="web")
        session = make_session(scalar_result=source)
        repo = SourceRepository(session)
        result, created = await repo.get_or_create("https://exists.com", "web")
        assert result is source
        assert created is False

    @pytest.mark.asyncio
    async def test_get_or_create_new(self):
        session = make_session(scalar_result=None)
        repo = SourceRepository(session)
        result, created = await repo.get_or_create("https://new.com", "web")
        assert created is True

    @pytest.mark.asyncio
    async def test_update_found(self):
        source = Source(id=1, source_url="https://test.com", source_type="web", is_sponsored=False)
        session = make_session(scalar_result=source)
        repo = SourceRepository(session)
        result = await repo.update(1, is_sponsored=True)
        assert result.is_sponsored is True

    @pytest.mark.asyncio
    async def test_update_not_found(self):
        session = make_session(scalar_result=None)
        repo = SourceRepository(session)
        result = await repo.update(999, is_sponsored=True)
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_found(self):
        source = Source(id=1, source_url="https://test.com", source_type="web")
        session = make_session(scalar_result=source)
        repo = SourceRepository(session)
        result = await repo.delete(1)
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_not_found(self):
        session = make_session(scalar_result=None)
        repo = SourceRepository(session)
        result = await repo.delete(999)
        assert result is False


# ---------------------------------------------------------------------------
# TestJobRepository
# ---------------------------------------------------------------------------
class TestJobRepository:
    @pytest.mark.asyncio
    async def test_get_by_id_found(self):
        job_id = uuid.uuid4()
        job = Job(id=job_id, user_id=1, job_type="auto_search", status="pending")
        session = make_session(scalar_result=job)
        repo = JobRepository(session)
        result = await repo.get_by_id(job_id)
        assert result is job

    @pytest.mark.asyncio
    async def test_get_by_id_not_found(self):
        session = make_session(scalar_result=None)
        repo = JobRepository(session)
        result = await repo.get_by_id(uuid.uuid4())
        assert result is None

    @pytest.mark.asyncio
    async def test_create(self):
        session = make_session()
        repo = JobRepository(session)
        job = await repo.create(1, "manual_links", product_query="iPhone 15")
        assert isinstance(job.id, uuid.UUID)
        assert job.user_id == 1
        assert job.job_type == "manual_links"
        assert job.status == "pending"
        assert job.product_query == "iPhone 15"
        session.add.assert_called_once()
        session.flush.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_status_found(self):
        job_id = uuid.uuid4()
        job = Job(id=job_id, user_id=1, job_type="auto_search", status="pending")
        session = make_session(scalar_result=job)
        repo = JobRepository(session)
        now = datetime.now(tz=timezone.utc)
        result = await repo.update_status(job_id, "done", completed_at=now)
        assert result.status == "done"
        assert result.completed_at == now

    @pytest.mark.asyncio
    async def test_update_status_not_found(self):
        session = make_session(scalar_result=None)
        repo = JobRepository(session)
        result = await repo.update_status(uuid.uuid4(), "done")
        assert result is None

    @pytest.mark.asyncio
    async def test_update_status_sets_celery_task_id(self):
        job_id = uuid.uuid4()
        job = Job(id=job_id, user_id=1, job_type="auto_search", status="pending")
        session = make_session(scalar_result=job)
        repo = JobRepository(session)
        result = await repo.update_status(job_id, "running", celery_task_id="celery-xyz")
        assert result.celery_task_id == "celery-xyz"
        assert result.status == "running"

    @pytest.mark.asyncio
    async def test_list_by_user(self):
        jobs = [
            Job(id=uuid.uuid4(), user_id=1, job_type="auto_search", status="done"),
            Job(id=uuid.uuid4(), user_id=1, job_type="manual_links", status="pending"),
        ]
        session = make_session(scalars_result=jobs)
        repo = JobRepository(session)
        result = await repo.list_by_user(1)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_delete_found(self):
        job_id = uuid.uuid4()
        job = Job(id=job_id, user_id=1, job_type="auto_search", status="done")
        session = make_session(scalar_result=job)
        repo = JobRepository(session)
        result = await repo.delete(job_id)
        assert result is True

    @pytest.mark.asyncio
    async def test_delete_not_found(self):
        session = make_session(scalar_result=None)
        repo = JobRepository(session)
        result = await repo.delete(uuid.uuid4())
        assert result is False


# ---------------------------------------------------------------------------
# TestQueryLogRepository
# ---------------------------------------------------------------------------
class TestQueryLogRepository:
    @pytest.mark.asyncio
    async def test_create_minimal(self):
        session = make_session()
        repo = QueryLogRepository(session)
        log = await repo.create(1)
        assert log.user_id == 1
        assert log.used_tavily is False
        assert log.rating is None
        session.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_full(self):
        session = make_session()
        repo = QueryLogRepository(session)
        log = await repo.create(
            2,
            session_id="sess_abc",
            mode="auto",
            query_text="best TV",
            response_text="Here are the best TVs...",
            sources_used=[{"url": "https://example.com"}],
            response_time_ms=450,
            used_tavily=True,
        )
        assert log.mode == "auto"
        assert log.used_tavily is True
        assert log.response_time_ms == 450

    @pytest.mark.asyncio
    async def test_get_by_id(self):
        log = QueryLog(id=1, user_id=1)
        session = make_session(scalar_result=log)
        repo = QueryLogRepository(session)
        result = await repo.get_by_id(1)
        assert result is log

    @pytest.mark.asyncio
    async def test_update_rating_found(self):
        log = QueryLog(id=1, user_id=1)
        log.rating = None
        session = make_session(scalar_result=log)
        repo = QueryLogRepository(session)
        result = await repo.update_rating(1, 1)
        assert result.rating == 1

    @pytest.mark.asyncio
    async def test_update_rating_not_found(self):
        session = make_session(scalar_result=None)
        repo = QueryLogRepository(session)
        result = await repo.update_rating(999, 1)
        assert result is None

    @pytest.mark.asyncio
    async def test_list_by_user(self):
        logs = [QueryLog(id=i, user_id=5) for i in range(3)]
        session = make_session(scalars_result=logs)
        repo = QueryLogRepository(session)
        result = await repo.list_by_user(5)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# TestSubscriptionRepository
# ---------------------------------------------------------------------------
class TestSubscriptionRepository:
    @pytest.mark.asyncio
    async def test_create(self):
        session = make_session()
        repo = SubscriptionRepository(session)
        now = datetime.now(tz=timezone.utc)
        sub = await repo.create(1, "charge_001", 100, now, now)
        assert sub.user_id == 1
        assert sub.status == "active"
        session.add.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_by_charge_id(self):
        sub = Subscription(
            user_id=1,
            telegram_payment_charge_id="charge_001",
            amount_stars=100,
            activated_at=datetime.now(tz=timezone.utc),
            expires_at=datetime.now(tz=timezone.utc),
            status="active",
        )
        session = make_session(scalar_result=sub)
        repo = SubscriptionRepository(session)
        result = await repo.get_by_charge_id("charge_001")
        assert result is sub

    @pytest.mark.asyncio
    async def test_update_status(self):
        sub = Subscription(
            user_id=1,
            telegram_payment_charge_id="charge_001",
            amount_stars=100,
            activated_at=datetime.now(tz=timezone.utc),
            expires_at=datetime.now(tz=timezone.utc),
            status="active",
        )
        sub.id = 1
        session = make_session(scalar_result=sub)
        repo = SubscriptionRepository(session)
        result = await repo.update(1, status="expired")
        assert result.status == "expired"

    @pytest.mark.asyncio
    async def test_list_by_user(self):
        subs = [
            Subscription(
                user_id=1,
                telegram_payment_charge_id=f"charge_{i}",
                amount_stars=100,
                activated_at=datetime.now(tz=timezone.utc),
                expires_at=datetime.now(tz=timezone.utc),
                status="active",
            )
            for i in range(2)
        ]
        session = make_session(scalars_result=subs)
        repo = SubscriptionRepository(session)
        result = await repo.list_by_user(1)
        assert len(result) == 2
