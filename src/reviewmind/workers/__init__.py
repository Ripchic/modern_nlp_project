"""reviewmind.workers — Celery app, tasks, and beat schedule."""

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
    ingest_sources_task,
    ping,
)

__all__ = [
    "ADMIN_ALERT_TEMPLATE",
    "MAX_RETRIES",
    "RETRY_COUNTDOWNS",
    "TASK_COMPLETED_NO_ANSWER_MSG",
    "TASK_FAILED_MSG",
    "TASK_STARTED_MSG",
    "celery_app",
    "create_celery_app",
    "ingest_sources_task",
    "ping",
    "send_admin_alert",
    "send_task_completed",
    "send_task_failed",
    "send_task_started",
]
