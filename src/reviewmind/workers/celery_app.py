"""reviewmind/workers/celery_app.py — Celery app factory with Redis broker."""

from __future__ import annotations

import structlog
from celery import Celery

logger = structlog.get_logger(__name__)

# ── Defaults ────────────────────────────────────────────────────────────────

DEFAULT_BROKER_URL = "redis://localhost:6379/1"
DEFAULT_RESULT_BACKEND = "redis://localhost:6379/2"
TASK_MODULES = ["reviewmind.workers.tasks"]


def create_celery_app(
    broker_url: str | None = None,
    result_backend: str | None = None,
) -> Celery:
    """Create and configure a Celery application.

    Parameters are read from config (lazy) when not provided explicitly.
    This allows the module to be imported without requiring a .env file
    (Celery discovers the app at import time).
    """
    if broker_url is None or result_backend is None:
        try:
            from reviewmind.config import settings

            broker_url = broker_url or settings.celery_broker_url
            result_backend = result_backend or settings.celery_result_backend
        except Exception:
            broker_url = broker_url or DEFAULT_BROKER_URL
            result_backend = result_backend or DEFAULT_RESULT_BACKEND

    app = Celery(
        "reviewmind",
        broker=broker_url,
        backend=result_backend,
    )

    from reviewmind.workers.beat_schedule import BEAT_SCHEDULE

    app.conf.update(
        # Serialisation
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        # Reliability
        task_acks_late=True,
        worker_prefetch_multiplier=1,
        # Timezone
        timezone="UTC",
        enable_utc=True,
        # Task discovery
        include=TASK_MODULES,
        # Results expiry (24h)
        result_expires=86400,
        # Broker connection retry on startup
        broker_connection_retry_on_startup=True,
        # Beat schedule
        beat_schedule=BEAT_SCHEDULE,
    )

    logger.info(
        "celery_app_created",
        broker=broker_url,
        backend=result_backend,
        task_modules=TASK_MODULES,
    )

    return app


# Module-level app instance used by `celery -A reviewmind.workers.celery_app`
celery_app = create_celery_app()
