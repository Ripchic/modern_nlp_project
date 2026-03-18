"""reviewmind.workers — Celery app, tasks, and beat schedule."""

from reviewmind.workers.beat_schedule import (
    BEAT_SCHEDULE,
    DAILY_RESET_LIMITS_TASK,
    DAILY_RESET_SCHEDULE,
    REFRESH_TOP_QUERIES_SCHEDULE,
    REFRESH_TOP_QUERIES_TASK,
)
from reviewmind.workers.celery_app import celery_app, create_celery_app
from reviewmind.workers.notifications import (
    ADMIN_ALERT_TEMPLATE,
    TASK_COMPLETED_NO_ANSWER_MSG,
    TASK_FAILED_MSG,
    TASK_STARTED_MSG,
    send_admin_alert,
    send_task_completed,
    send_task_failed,
    send_task_started,
)
from reviewmind.workers.tasks import (
    MAX_RETRIES,
    RETRY_COUNTDOWNS,
    TOP_QUERIES_LIMIT,
    daily_reset_limits_task,
    ingest_sources_task,
    ping,
    refresh_top_queries_task,
)

__all__ = [
    "ADMIN_ALERT_TEMPLATE",
    "BEAT_SCHEDULE",
    "DAILY_RESET_LIMITS_TASK",
    "DAILY_RESET_SCHEDULE",
    "MAX_RETRIES",
    "REFRESH_TOP_QUERIES_SCHEDULE",
    "REFRESH_TOP_QUERIES_TASK",
    "RETRY_COUNTDOWNS",
    "TASK_COMPLETED_NO_ANSWER_MSG",
    "TASK_FAILED_MSG",
    "TASK_STARTED_MSG",
    "TOP_QUERIES_LIMIT",
    "celery_app",
    "create_celery_app",
    "daily_reset_limits_task",
    "ingest_sources_task",
    "ping",
    "refresh_top_queries_task",
    "send_admin_alert",
    "send_task_completed",
    "send_task_failed",
    "send_task_started",
]
