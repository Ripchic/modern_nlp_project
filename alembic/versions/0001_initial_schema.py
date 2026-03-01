"""initial schema — all 6 tables

Revision ID: 0001
Revises:
Create Date: 2026-03-01 00:00:00.000000
"""
from __future__ import annotations

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# ---------------------------------------------------------------------------
# Revision metadata
# ---------------------------------------------------------------------------
revision: str = "0001"
down_revision: str | None = None
branch_labels: str | tuple[str, ...] | None = None
depends_on: str | tuple[str, ...] | None = None


def upgrade() -> None:
    # ------------------------------------------------------------------
    # users
    # ------------------------------------------------------------------
    op.create_table(
        "users",
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column("is_admin", sa.Boolean(), server_default="false", nullable=False),
        sa.Column("subscription", sa.String(20), server_default="free", nullable=False),
        sa.Column("sub_expires_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("user_id"),
    )

    # ------------------------------------------------------------------
    # user_limits
    # ------------------------------------------------------------------
    op.create_table(
        "user_limits",
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("requests_used", sa.Integer(), server_default="0", nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("user_id", "date"),
        sa.UniqueConstraint("user_id", "date", name="uq_user_limits_user_date"),
    )

    # ------------------------------------------------------------------
    # subscriptions
    # ------------------------------------------------------------------
    op.create_table(
        "subscriptions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column("telegram_payment_charge_id", sa.String(256), nullable=False),
        sa.Column("amount_stars", sa.Integer(), nullable=False),
        sa.Column("activated_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("expires_at", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("telegram_payment_charge_id"),
    )
    op.create_index("ix_subscriptions_user_id", "subscriptions", ["user_id"])

    # ------------------------------------------------------------------
    # sources
    # ------------------------------------------------------------------
    op.create_table(
        "sources",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("source_url", sa.Text(), nullable=False),
        sa.Column("source_type", sa.String(20), nullable=False),
        sa.Column("product_query", sa.Text(), nullable=True),
        sa.Column("parsed_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.Column("is_sponsored", sa.Boolean(), server_default="false", nullable=False),
        sa.Column("is_curated", sa.Boolean(), server_default="false", nullable=False),
        sa.Column("language", sa.String(10), nullable=True),
        sa.Column("author", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("source_url"),
    )
    op.create_index("ix_sources_source_type", "sources", ["source_type"])
    op.create_index("ix_sources_is_curated", "sources", ["is_curated"])
    op.create_index("ix_sources_is_sponsored", "sources", ["is_sponsored"])

    # ------------------------------------------------------------------
    # jobs
    # ------------------------------------------------------------------
    op.create_table(
        "jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column("job_type", sa.String(20), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("product_query", sa.Text(), nullable=True),
        sa.Column("celery_task_id", sa.String(256), nullable=True),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.Column("completed_at", sa.TIMESTAMP(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_jobs_user_id", "jobs", ["user_id"])
    op.create_index("ix_jobs_status", "jobs", ["status"])

    # ------------------------------------------------------------------
    # query_logs
    # ------------------------------------------------------------------
    op.create_table(
        "query_logs",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.BigInteger(), nullable=False),
        sa.Column("session_id", sa.String(64), nullable=True),
        sa.Column("mode", sa.String(20), nullable=True),
        sa.Column("query_text", sa.Text(), nullable=True),
        sa.Column("response_text", sa.Text(), nullable=True),
        sa.Column("sources_used", sa.JSON(), nullable=True),
        sa.Column("rating", sa.SmallInteger(), nullable=True),
        sa.Column("response_time_ms", sa.Integer(), nullable=True),
        sa.Column("used_tavily", sa.Boolean(), server_default="false", nullable=False),
        sa.Column("created_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.ForeignKeyConstraint(["user_id"], ["users.user_id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_query_logs_user_id", "query_logs", ["user_id"])
    op.create_index("ix_query_logs_created_at", "query_logs", ["created_at"])


def downgrade() -> None:
    op.drop_table("query_logs")
    op.drop_table("jobs")
    op.drop_table("sources")
    op.drop_table("subscriptions")
    op.drop_table("user_limits")
    op.drop_table("users")
