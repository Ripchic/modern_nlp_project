"""reviewmind/workers/tasks.py — Celery task definitions."""

from __future__ import annotations

import structlog

from reviewmind.workers.celery_app import celery_app

logger = structlog.get_logger(__name__)


@celery_app.task(name="reviewmind.ping", bind=True, max_retries=0)
def ping(self: object) -> dict:
    """Health-check task that returns a simple status dict."""
    logger.info("ping_task_executed")
    return {"status": "pong"}
