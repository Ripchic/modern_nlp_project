"""Unit tests for SQLAlchemy ORM models (TASK-009)."""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone

import sqlalchemy as sa

from reviewmind.db.models import Base, Job, QueryLog, Source, Subscription, User, UserLimit


# ---------------------------------------------------------------------------
# TestBase
# ---------------------------------------------------------------------------
class TestBase:
    def test_base_is_declarative(self):
        assert hasattr(Base, "metadata")
        assert hasattr(Base, "registry")

    def test_all_tables_registered(self):
        table_names = set(Base.metadata.tables.keys())
        expected = {"users", "user_limits", "subscriptions", "sources", "jobs", "query_logs"}
        assert expected == table_names


# ---------------------------------------------------------------------------
# TestUserModel
# ---------------------------------------------------------------------------
class TestUserModel:
    def test_tablename(self):
        assert User.__tablename__ == "users"

    def test_primary_key_column(self):
        col = User.__table__.c["user_id"]
        assert col.primary_key
        assert isinstance(col.type, sa.BigInteger)

    def test_is_admin_default(self):
        col = User.__table__.c["is_admin"]
        assert col.type.__class__.__name__ == "Boolean"
        assert col.server_default.arg == "false"

    def test_subscription_default(self):
        col = User.__table__.c["subscription"]
        assert col.server_default.arg == "free"

    def test_sub_expires_at_nullable(self):
        col = User.__table__.c["sub_expires_at"]
        assert col.nullable is True

    def test_created_at_has_server_default(self):
        col = User.__table__.c["created_at"]
        assert col.server_default is not None

    def test_instantiation_defaults(self):
        user = User(user_id=123)
        assert user.user_id == 123
        # is_admin default is applied server-side (server_default='false')
        # Python-level value before INSERT is None; verify column default exists instead
        assert User.__table__.c["is_admin"].server_default.arg == "false"
        assert user.sub_expires_at is None

    def test_relationships_exist(self):
        assert hasattr(User, "limits")
        assert hasattr(User, "subscriptions")
        assert hasattr(User, "jobs")
        assert hasattr(User, "query_logs")


# ---------------------------------------------------------------------------
# TestUserLimitModel
# ---------------------------------------------------------------------------
class TestUserLimitModel:
    def test_tablename(self):
        assert UserLimit.__tablename__ == "user_limits"

    def test_composite_primary_key(self):
        pk_cols = {c.name for c in UserLimit.__table__.primary_key.columns}
        assert pk_cols == {"user_id", "date"}

    def test_requests_used_default(self):
        col = UserLimit.__table__.c["requests_used"]
        assert col.server_default.arg == "0"

    def test_foreign_key_to_users(self):
        fks = list(UserLimit.__table__.foreign_keys)
        assert any(fk.column.table.name == "users" for fk in fks)

    def test_unique_constraint(self):
        constraints = {c.name for c in UserLimit.__table__.constraints if hasattr(c, "name")}
        assert "uq_user_limits_user_date" in constraints

    def test_instantiation(self):
        today = date.today()
        row = UserLimit(user_id=1, date=today, requests_used=5)
        assert row.requests_used == 5


# ---------------------------------------------------------------------------
# TestSubscriptionModel
# ---------------------------------------------------------------------------
class TestSubscriptionModel:
    def test_tablename(self):
        assert Subscription.__tablename__ == "subscriptions"

    def test_primary_key(self):
        col = Subscription.__table__.c["id"]
        assert col.primary_key
        assert col.autoincrement

    def test_charge_id_unique(self):
        col = Subscription.__table__.c["telegram_payment_charge_id"]
        assert any(c.__class__.__name__ == "UniqueConstraint" or col.unique for c in [col])

    def test_foreign_key_to_users(self):
        fks = list(Subscription.__table__.foreign_keys)
        assert any(fk.column.table.name == "users" for fk in fks)

    def test_instantiation(self):
        now = datetime.now(tz=timezone.utc)
        sub = Subscription(
            user_id=1,
            telegram_payment_charge_id="charge_abc",
            amount_stars=100,
            activated_at=now,
            expires_at=now,
            status="active",
        )
        assert sub.status == "active"
        assert sub.amount_stars == 100


# ---------------------------------------------------------------------------
# TestSourceModel
# ---------------------------------------------------------------------------
class TestSourceModel:
    def test_tablename(self):
        assert Source.__tablename__ == "sources"

    def test_source_url_unique(self):
        col = Source.__table__.c["source_url"]
        assert col.unique

    def test_indexes(self):
        index_names = {idx.name for idx in Source.__table__.indexes}
        assert "ix_sources_source_type" in index_names
        assert "ix_sources_is_curated" in index_names
        assert "ix_sources_is_sponsored" in index_names

    def test_is_sponsored_default(self):
        col = Source.__table__.c["is_sponsored"]
        assert col.server_default.arg == "false"

    def test_is_curated_default(self):
        col = Source.__table__.c["is_curated"]
        assert col.server_default.arg == "false"

    def test_nullable_fields(self):
        for field in ("product_query", "parsed_at", "language", "author"):
            assert Source.__table__.c[field].nullable is True

    def test_instantiation(self):
        source = Source(source_url="https://example.com/review", source_type="web")
        assert source.source_url == "https://example.com/review"
        assert source.source_type == "web"
        # is_sponsored / is_curated defaults applied server-side; verify column config
        assert Source.__table__.c["is_sponsored"].server_default.arg == "false"
        assert Source.__table__.c["is_curated"].server_default.arg == "false"


# ---------------------------------------------------------------------------
# TestJobModel
# ---------------------------------------------------------------------------
class TestJobModel:
    def test_tablename(self):
        assert Job.__tablename__ == "jobs"

    def test_primary_key_is_uuid(self):
        col = Job.__table__.c["id"]
        assert col.primary_key
        # UUID type check
        assert "UUID" in type(col.type).__name__

    def test_foreign_key_to_users(self):
        fks = list(Job.__table__.foreign_keys)
        assert any(fk.column.table.name == "users" for fk in fks)

    def test_indexes(self):
        index_names = {idx.name for idx in Job.__table__.indexes}
        assert "ix_jobs_user_id" in index_names
        assert "ix_jobs_status" in index_names

    def test_completed_at_nullable(self):
        col = Job.__table__.c["completed_at"]
        assert col.nullable is True

    def test_instantiation(self):
        job_id = uuid.uuid4()
        job = Job(id=job_id, user_id=1, job_type="auto_search", status="pending")
        assert job.id == job_id
        assert job.status == "pending"
        assert job.job_type == "auto_search"


# ---------------------------------------------------------------------------
# TestQueryLogModel
# ---------------------------------------------------------------------------
class TestQueryLogModel:
    def test_tablename(self):
        assert QueryLog.__tablename__ == "query_logs"

    def test_primary_key(self):
        col = QueryLog.__table__.c["id"]
        assert col.primary_key
        assert col.autoincrement

    def test_foreign_key_to_users(self):
        fks = list(QueryLog.__table__.foreign_keys)
        assert any(fk.column.table.name == "users" for fk in fks)

    def test_indexes(self):
        index_names = {idx.name for idx in QueryLog.__table__.indexes}
        assert "ix_query_logs_user_id" in index_names
        assert "ix_query_logs_created_at" in index_names

    def test_nullable_fields(self):
        nullable = (
            "session_id",
            "mode",
            "query_text",
            "response_text",
            "sources_used",
            "rating",
            "response_time_ms",
        )
        for field in nullable:
            assert QueryLog.__table__.c[field].nullable is True

    def test_used_tavily_default(self):
        col = QueryLog.__table__.c["used_tavily"]
        assert col.server_default.arg == "false"

    def test_instantiation(self):
        log = QueryLog(user_id=1, query_text="best headphones", response_text="Here are...")
        assert log.user_id == 1
        assert log.query_text == "best headphones"
        assert log.rating is None
        # used_tavily default is applied server-side; verify column config
        assert QueryLog.__table__.c["used_tavily"].server_default.arg == "false"
