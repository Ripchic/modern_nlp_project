"""reviewmind/workers/tasks.py — Celery task definitions."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

import structlog
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from reviewmind.db.repositories.jobs import JobRepository
from reviewmind.ingestion.pipeline import IngestionPipeline
from reviewmind.workers.celery_app import celery_app

logger = structlog.get_logger(__name__)


@celery_app.task(name="reviewmind.ping", bind=True, max_retries=0)
def ping(self: object) -> dict:
    """Health-check task that returns a simple status dict."""
    logger.info("ping_task_executed")
    return {"status": "pong"}


# ---------------------------------------------------------------------------
# Ingestion task
# ---------------------------------------------------------------------------

def _run_async(coro):
    """Run an async coroutine from a sync Celery task."""
    return asyncio.run(coro)


async def _ingest_sources(
    *,
    job_id: str,
    user_id: int,
    product_query: str,
    urls: list[str],
    session_id: str | None = None,
) -> dict:
    """Async implementation of the ingestion task.

    1. Update job status to 'running' in PostgreSQL.
    2. Run the ingestion pipeline for all URLs.
    3. Update job status to 'done' (or 'failed') with completed_at.
    4. Return a summary dict.
    """
    log = logger.bind(job_id=job_id, user_id=user_id, product_query=product_query)
    log.info("ingest_task_start", urls_count=len(urls))

    job_uuid = uuid.UUID(job_id)

    from reviewmind.config import settings

    engine = create_async_engine(settings.database_url, pool_pre_ping=True, echo=False)
    session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

    # -- Mark job as running ---------------------------------------------------
    try:
        async with session_factory() as db_session:
            job_repo = JobRepository(db_session)
            await job_repo.update_status(job_uuid, "running")
            await db_session.commit()
        log.info("job_status_updated", status="running")
    except Exception as exc:
        log.warning("job_update_running_failed", error=str(exc))

    # -- Run ingestion pipeline ------------------------------------------------
    success = False
    summary: dict = {}
    try:
        qdrant = AsyncQdrantClient(url=settings.qdrant_url, timeout=30)

        async with session_factory() as db_session:
            async with IngestionPipeline(
                qdrant_client=qdrant,
                db_session=db_session,
            ) as pipeline:
                result = await pipeline.ingest_urls(
                    urls=urls,
                    product_query=product_query,
                    session_id=session_id,
                )
            await db_session.commit()

        await qdrant.close()

        summary = {
            "success_count": result.success_count,
            "failed_count": result.failed_count,
            "chunks_count": result.chunks_count,
            "failed_urls": result.failed_urls,
        }
        success = result.success_count > 0
        log.info("ingest_pipeline_done", **summary)

    except Exception as exc:
        log.error("ingest_pipeline_error", error=str(exc))
        summary = {"error": str(exc)}

    # -- Update job status -----------------------------------------------------
    final_status = "done" if success else "failed"
    now = datetime.now(timezone.utc)
    try:
        async with session_factory() as db_session:
            job_repo = JobRepository(db_session)
            await job_repo.update_status(job_uuid, final_status, completed_at=now)
            await db_session.commit()
        log.info("job_status_updated", status=final_status, completed_at=now.isoformat())
    except Exception as exc:
        log.warning("job_update_final_failed", error=str(exc))

    await engine.dispose()

    return {
        "job_id": job_id,
        "status": final_status,
        "completed_at": now.isoformat(),
        **summary,
    }


@celery_app.task(name="reviewmind.ingest_sources", bind=True, max_retries=0)
def ingest_sources_task(
    self: object,
    *,
    job_id: str,
    user_id: int,
    product_query: str,
    urls: list[str],
    session_id: str | None = None,
) -> dict:
    """Celery task: run the ingestion pipeline for a set of URLs.

    Creates/updates a ``Job`` record in PostgreSQL through the lifecycle:
    pending → running → done | failed.

    Parameters
    ----------
    job_id:
        UUID of the pre-created Job row (as string).
    user_id:
        Telegram user ID (for logging / ownership).
    product_query:
        Product name used for Qdrant payload tagging.
    urls:
        List of URLs to ingest.
    session_id:
        Optional session identifier.

    Returns
    -------
    dict
        Summary with ``job_id``, ``status``, ``completed_at``, counts.
    """
    return _run_async(
        _ingest_sources(
            job_id=job_id,
            user_id=user_id,
            product_query=product_query,
            urls=urls,
            session_id=session_id,
        )
    )
