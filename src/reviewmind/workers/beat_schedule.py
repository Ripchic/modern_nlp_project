"""reviewmind/workers/beat_schedule.py — Celery Beat periodic task schedule.

Defines crontab-based schedules for periodic tasks:
- ``daily_reset_limits`` — every day at 00:00 UTC, resets ``user_limits.requests_used``.
- ``refresh_top_queries`` — every 30 days, refreshes top-50 product queries in ``auto_crawled``.
"""

from __future__ import annotations

from celery.schedules import crontab

# ── Task names ───────────────────────────────────────────────────────────────

DAILY_RESET_LIMITS_TASK: str = "reviewmind.daily_reset_limits"
"""Celery task name for the daily limit reset."""

REFRESH_TOP_QUERIES_TASK: str = "reviewmind.refresh_top_queries"
"""Celery task name for periodic top-query refresh."""

# ── Schedules ────────────────────────────────────────────────────────────────

DAILY_RESET_SCHEDULE = crontab(minute="0", hour="0")
"""Run every day at 00:00 UTC."""

REFRESH_TOP_QUERIES_SCHEDULE = crontab(minute="0", hour="3", day_of_month="1")
"""Run on the 1st of every month at 03:00 UTC (~every 30 days)."""

# ── Beat schedule dict ───────────────────────────────────────────────────────

BEAT_SCHEDULE: dict = {
    "daily-reset-limits": {
        "task": DAILY_RESET_LIMITS_TASK,
        "schedule": DAILY_RESET_SCHEDULE,
        "options": {"queue": "default"},
    },
    "refresh-top-queries-monthly": {
        "task": REFRESH_TOP_QUERIES_TASK,
        "schedule": REFRESH_TOP_QUERIES_SCHEDULE,
        "options": {"queue": "default"},
    },
}
