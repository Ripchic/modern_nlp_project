"""reviewmind/api/endpoints/status.py — GET /status/{job_id} endpoint.

Returns the current status of a background ingestion job, including optional
progress information fetched from Celery's result backend.
"""

from __future__ import annotations

import uuid

import structlog
from fastapi import APIRouter, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from reviewmind.api.schemas import JobStatusResponse
from reviewmind.db.repositories.jobs import JobRepository

logger = structlog.get_logger(__name__)

router = APIRouter()


@router.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str, request: Request) -> JobStatusResponse:
    """Return the current status of a background job.

    Reads the ``jobs`` table from PostgreSQL.  If a Celery result backend is
    available and the task has finished, ``progress`` will contain the task
    return value.
    """
    log = logger.bind(job_id=job_id)

    # Validate UUID format
    try:
        job_uuid = uuid.UUID(job_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid job_id format — must be a UUID")

    # Require a database engine
    engine = getattr(request.app.state, "db_engine", None)
    if engine is None:
        log.error("db_unavailable")
        raise HTTPException(status_code=503, detail="Database unavailable")

    session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as db_session:
        repo = JobRepository(db_session)
        job = await repo.get_by_id(job_uuid)

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")

    # Optionally fetch Celery task result for extra progress info
    progress: dict | None = None
    if job.celery_task_id and job.status in ("done", "failed"):
        progress = _get_celery_result(job.celery_task_id)

    return JobStatusResponse(
        job_id=str(job.id),
        status=job.status,
        job_type=job.job_type or "",
        product_query=job.product_query,
        celery_task_id=job.celery_task_id,
        created_at=job.created_at.isoformat() if job.created_at else None,
        completed_at=job.completed_at.isoformat() if job.completed_at else None,
        progress=progress,
    )


def _get_celery_result(celery_task_id: str) -> dict | None:
    """Attempt to read the task result from Celery's result backend.

    Returns ``None`` silently if the backend is unreachable or the result
    is not yet stored.
    """
    try:
        from reviewmind.workers.celery_app import celery_app

        result = celery_app.AsyncResult(celery_task_id)
        if result.ready():
            return result.result if isinstance(result.result, dict) else None
    except Exception as exc:
        logger.debug("celery_result_fetch_failed", error=str(exc))
    return None
