# reviewmind/db/models.py — SQLAlchemy ORM модели (все 6 таблиц по PRD секция 6.1)
from __future__ import annotations

import uuid
from datetime import date, datetime

from sqlalchemy import (
    JSON,
    TIMESTAMP,
    BigInteger,
    Boolean,
    Date,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass


# ---------------------------------------------------------------------------
# users
# ---------------------------------------------------------------------------
class User(Base):
    __tablename__ = "users"

    user_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)  # telegram_user_id
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false", nullable=False)
    subscription: Mapped[str] = mapped_column(
        String(20), default="free", server_default="free", nullable=False
    )
    sub_expires_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )

    # relationships
    limits: Mapped[list[UserLimit]] = relationship("UserLimit", back_populates="user", cascade="all, delete-orphan")
    subscriptions: Mapped[list[Subscription]] = relationship(
        "Subscription", back_populates="user", cascade="all, delete-orphan"
    )
    jobs: Mapped[list[Job]] = relationship("Job", back_populates="user", cascade="all, delete-orphan")
    query_logs: Mapped[list[QueryLog]] = relationship(
        "QueryLog", back_populates="user", cascade="all, delete-orphan"
    )


# ---------------------------------------------------------------------------
# user_limits
# ---------------------------------------------------------------------------
class UserLimit(Base):
    __tablename__ = "user_limits"
    __table_args__ = (UniqueConstraint("user_id", "date", name="uq_user_limits_user_date"),)

    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.user_id", ondelete="CASCADE"), primary_key=True
    )
    date: Mapped[date] = mapped_column(Date, primary_key=True)
    requests_used: Mapped[int] = mapped_column(Integer, default=0, server_default="0", nullable=False)

    user: Mapped[User] = relationship("User", back_populates="limits")


# ---------------------------------------------------------------------------
# subscriptions
# ---------------------------------------------------------------------------
class Subscription(Base):
    __tablename__ = "subscriptions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False, index=True
    )
    telegram_payment_charge_id: Mapped[str] = mapped_column(String(256), unique=True, nullable=False)
    amount_stars: Mapped[int] = mapped_column(Integer, nullable=False)
    activated_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)  # 'active' | 'expired' | 'cancelled'

    user: Mapped[User] = relationship("User", back_populates="subscriptions")


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------
class Source(Base):
    __tablename__ = "sources"
    __table_args__ = (
        Index("ix_sources_source_type", "source_type"),
        Index("ix_sources_is_curated", "is_curated"),
        Index("ix_sources_is_sponsored", "is_sponsored"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source_url: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    source_type: Mapped[str] = mapped_column(
        String(20), nullable=False
    )  # 'youtube' | 'reddit' | 'web' | 'tavily' | 'curated'
    product_query: Mapped[str | None] = mapped_column(Text, nullable=True)
    parsed_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)
    is_sponsored: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false", nullable=False)
    is_curated: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false", nullable=False)
    language: Mapped[str | None] = mapped_column(String(10), nullable=True)
    author: Mapped[str | None] = mapped_column(Text, nullable=True)


# ---------------------------------------------------------------------------
# jobs
# ---------------------------------------------------------------------------
class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (
        Index("ix_jobs_user_id", "user_id"),
        Index("ix_jobs_status", "status"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False
    )
    job_type: Mapped[str] = mapped_column(String(20), nullable=False)  # 'auto_search' | 'manual_links'
    status: Mapped[str] = mapped_column(
        String(20), default="pending", nullable=False
    )  # 'pending' | 'running' | 'done' | 'failed'
    product_query: Mapped[str | None] = mapped_column(Text, nullable=True)
    celery_task_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(TIMESTAMP(timezone=True), nullable=True)

    user: Mapped[User] = relationship("User", back_populates="jobs")


# ---------------------------------------------------------------------------
# query_logs
# ---------------------------------------------------------------------------
class QueryLog(Base):
    __tablename__ = "query_logs"
    __table_args__ = (
        Index("ix_query_logs_user_id", "user_id"),
        Index("ix_query_logs_created_at", "created_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False
    )
    session_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    mode: Mapped[str | None] = mapped_column(String(20), nullable=True)  # 'auto' | 'manual'
    query_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    response_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    sources_used: Mapped[list | None] = mapped_column(JSON, nullable=True)
    rating: Mapped[int | None] = mapped_column(SmallInteger, nullable=True)  # 1 (👍) | -1 (👎) | NULL
    response_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    used_tavily: Mapped[bool] = mapped_column(Boolean, default=False, server_default="false", nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True), server_default=func.now(), nullable=False
    )

    user: Mapped[User] = relationship("User", back_populates="query_logs")
