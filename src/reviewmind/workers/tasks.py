"""reviewmind/workers/tasks.py — Celery task definitions."""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

import structlog
from qdrant_client import AsyncQdrantClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from reviewmind.db.models import QueryLog, UserLimit
from reviewmind.db.repositories.jobs import JobRepository
from reviewmind.ingestion.pipeline import IngestionPipeline
from reviewmind.workers.celery_app import celery_app

logger = structlog.get_logger(__name__)

# ── Retry constants ──────────────────────────────────────────────────────────

MAX_RETRIES = 3
RETRY_COUNTDOWNS = (60, 300, 900)  # 1 min, 5 min, 15 min — exponential backoff


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

    # -- Send push notification to the user ---------------------------------
    try:
        from reviewmind.workers.notifications import (
            send_task_completed,
            send_task_failed,
        )

        if success:
            await send_task_completed(
                bot_token=settings.telegram_bot_token,
                chat_id=user_id,
                product_query=product_query,
                qdrant_url=settings.qdrant_url,
            )
        else:
            await send_task_failed(
                bot_token=settings.telegram_bot_token,
                chat_id=user_id,
            )
    except Exception as exc:
        log.error("push_notification_failed", error=str(exc))

    return {
        "job_id": job_id,
        "status": final_status,
        "completed_at": now.isoformat(),
        **summary,
    }


async def _handle_final_failure(
    *,
    job_id: str,
    user_id: int,
    product_query: str,
    task_id: str,
    error: str,
) -> None:
    """Handle the final failure after all retries are exhausted.

    1. Update job status to 'failed' in PostgreSQL.
    2. Send apology to the user.
    3. Send alert to all admin users.
    """
    log = logger.bind(job_id=job_id, user_id=user_id, task_id=task_id)
    log.error("handling_final_failure", error=error)

    from reviewmind.config import settings

    job_uuid = uuid.UUID(job_id)
    now = datetime.now(timezone.utc)

    # 1. Update job status in PostgreSQL
    try:
        engine = create_async_engine(settings.database_url, pool_pre_ping=True, echo=False)
        session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
        async with session_factory() as db_session:
            job_repo = JobRepository(db_session)
            await job_repo.update_status(job_uuid, "failed", completed_at=now)
            await db_session.commit()
        await engine.dispose()
        log.info("job_status_updated_final_failure", status="failed")
    except Exception as exc:
        log.warning("job_update_final_failure_db_error", error=str(exc))

    # 2. Send apology to the user
    try:
        from reviewmind.workers.notifications import send_task_failed

        await send_task_failed(
            bot_token=settings.telegram_bot_token,
            chat_id=user_id,
        )
    except Exception as exc:
        log.error("final_failure_user_notification_failed", error=str(exc))

    # 3. Send alert to admins
    try:
        from reviewmind.workers.notifications import send_admin_alert

        await send_admin_alert(
            bot_token=settings.telegram_bot_token,
            admin_user_ids=settings.admin_user_ids,
            task_id=task_id,
            job_id=job_id,
            user_id=user_id,
            product_query=product_query,
            error=error,
            max_retries=MAX_RETRIES,
        )
    except Exception as exc:
        log.error("final_failure_admin_alert_failed", error=str(exc))


@celery_app.task(name="reviewmind.ingest_sources", bind=True, max_retries=MAX_RETRIES)
def ingest_sources_task(
    self: object,
    *,
    job_id: str,
    user_id: int,
    product_query: str,
    urls: list[str],
    session_id: str | None = None,
) -> dict:
    import time as _time

    _task_start = _time.perf_counter()
    """Celery task: run the ingestion pipeline for a set of URLs.

    Creates/updates a ``Job`` record in PostgreSQL through the lifecycle:
    pending → running → done | failed.

    Retries up to 3 times with exponential backoff (60s, 300s, 900s).
    On final failure: updates job status to 'failed', sends apology to user,
    and alerts all admins.

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
    try:
        result = _run_async(
            _ingest_sources(
                job_id=job_id,
                user_id=user_id,
                product_query=product_query,
                urls=urls,
                session_id=session_id,
            )
        )
        try:
            from reviewmind.metrics import CELERY_TASK_DURATION_SECONDS, CELERY_TASKS_TOTAL

            CELERY_TASK_DURATION_SECONDS.labels(task_name="ingest_sources", status="success").observe(
                _time.perf_counter() - _task_start
            )
            CELERY_TASKS_TOTAL.labels(task_name="ingest_sources", status="success").inc()
        except Exception:
            pass
        return result
    except Exception as exc:
        retry_num = self.request.retries
        countdown = RETRY_COUNTDOWNS[retry_num] if retry_num < len(RETRY_COUNTDOWNS) else RETRY_COUNTDOWNS[-1]
        log = logger.bind(
            job_id=job_id,
            user_id=user_id,
            retry=retry_num,
            max_retries=MAX_RETRIES,
        )

        if retry_num < MAX_RETRIES:
            log.warning(
                "ingest_task_retry",
                countdown=countdown,
                error=str(exc),
            )
            try:
                from reviewmind.metrics import CELERY_TASK_DURATION_SECONDS, CELERY_TASKS_TOTAL

                CELERY_TASK_DURATION_SECONDS.labels(task_name="ingest_sources", status="retry").observe(
                    _time.perf_counter() - _task_start
                )
                CELERY_TASKS_TOTAL.labels(task_name="ingest_sources", status="retry").inc()
            except Exception:
                pass
            raise self.retry(exc=exc, countdown=countdown)

        # Final failure — all retries exhausted
        try:
            from reviewmind.metrics import CELERY_TASK_DURATION_SECONDS, CELERY_TASKS_TOTAL

            CELERY_TASK_DURATION_SECONDS.labels(task_name="ingest_sources", status="failure").observe(
                _time.perf_counter() - _task_start
            )
            CELERY_TASKS_TOTAL.labels(task_name="ingest_sources", status="failure").inc()
        except Exception:
            pass
        log.error("ingest_task_final_failure", error=str(exc))
        _run_async(
            _handle_final_failure(
                job_id=job_id,
                user_id=user_id,
                product_query=product_query,
                task_id=self.request.id or "unknown",
                error=str(exc),
            )
        )
        raise


# ---------------------------------------------------------------------------
# Periodic tasks (Celery Beat)
# ---------------------------------------------------------------------------

# ── Constants for periodic tasks ─────────────────────────────────────────────

TOP_QUERIES_LIMIT: int = 50
"""Number of top product queries to refresh during the monthly task."""

MAX_REFRESH_URLS: int = 10
"""Maximum number of URLs to collect per query during refresh."""


async def _collect_urls_for_refresh(query_text: str) -> list[str]:
    """Search YouTube + Reddit for review URLs for a product query.

    Returns up to :data:`MAX_REFRESH_URLS` unique URLs, or an empty list
    if both searches fail.
    """
    urls: list[str] = []
    seen: set[str] = set()
    search_query = query_text + " review"

    try:
        from reviewmind.config import get_settings  # noqa: PLC0415
        from reviewmind.scrapers.youtube import YouTubeScraper  # noqa: PLC0415

        yt = YouTubeScraper(
            cookie_path=get_settings().youtube_cookies_path or None,
        )
        videos = yt.search_videos(search_query, max_results=5)
        for v in videos:
            if v.url and v.url not in seen:
                seen.add(v.url)
                urls.append(v.url)
    except Exception as exc:
        logger.warning("refresh_youtube_search_failed", query=query_text, error=str(exc))

    try:
        from reviewmind.scrapers.reddit import RedditScraper  # noqa: PLC0415

        reddit = RedditScraper()
        posts = reddit.search_posts(search_query, limit=5)
        for p in posts:
            if p.url and p.url not in seen:
                seen.add(p.url)
                urls.append(p.url)
    except Exception as exc:
        logger.warning("refresh_reddit_search_failed", query=query_text, error=str(exc))

    return urls[:MAX_REFRESH_URLS]


async def _daily_reset_limits() -> dict:
    """Safety-net reset for ``user_limits`` at midnight UTC.

    The primary daily-reset mechanism is **date-based keying**: each
    ``user_limits`` row is keyed by ``(user_id, date)``, so a new UTC day
    automatically yields a fresh counter (``requests_used`` starts at 0)
    with no explicit reset needed.

    This Beat task runs at 00:00 UTC as a **safety net** — it zeros out
    any ``requests_used`` values for the *new* day's rows that might have
    been pre-created (e.g. race conditions at midnight, manual inserts, or
    clock-skew).  Old rows for previous dates are **preserved** for
    analytics.

    Returns a summary dict with the number of rows reset.
    """
    from reviewmind.config import settings

    engine = create_async_engine(settings.database_url, pool_pre_ping=True, echo=False)
    session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

    today = datetime.now(timezone.utc).date()
    rows_reset = 0

    try:
        async with session_factory() as session:
            result = await session.execute(
                select(UserLimit).where(
                    UserLimit.date == today,
                    UserLimit.requests_used > 0,
                )
            )
            rows = result.scalars().all()
            for row in rows:
                row.requests_used = 0
                rows_reset += 1
            await session.commit()
        logger.info("daily_reset_limits_done", date=str(today), rows_reset=rows_reset)
    except Exception as exc:
        logger.error("daily_reset_limits_error", error=str(exc))
    finally:
        await engine.dispose()

    return {"date": str(today), "rows_reset": rows_reset}


async def _refresh_top_queries() -> dict:
    """Fetch top-50 product queries from ``query_logs`` and re-ingest them.

    Gathers the most frequent ``product_query`` values from the query_logs
    table (non-NULL, grouped and ordered by count DESC), then for each
    query collects source URLs via YouTube + Reddit search before enqueuing
    an ingestion job.  Queries yielding zero URLs are skipped.

    Returns a summary dict with the queries found and jobs enqueued.
    """
    from reviewmind.config import settings

    engine = create_async_engine(settings.database_url, pool_pre_ping=True, echo=False)
    session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

    queries_found: list[str] = []
    jobs_enqueued = 0
    skipped_no_urls = 0

    try:
        async with session_factory() as session:
            # Find top-N product queries by frequency
            stmt = (
                select(
                    QueryLog.query_text,
                    func.count(QueryLog.id).label("cnt"),
                )
                .where(QueryLog.query_text.isnot(None))
                .group_by(QueryLog.query_text)
                .order_by(func.count(QueryLog.id).desc())
                .limit(TOP_QUERIES_LIMIT)
            )
            result = await session.execute(stmt)
            rows = result.all()
            queries_found = [row[0] for row in rows if row[0]]

        logger.info("refresh_top_queries_found", count=len(queries_found))

        # Enqueue ingestion for each top query (via existing Celery task)
        for query_text in queries_found:
            try:
                # Collect source URLs via YouTube + Reddit search
                urls = await _collect_urls_for_refresh(query_text)
                if not urls:
                    skipped_no_urls += 1
                    logger.info("refresh_top_queries_skip_no_urls", query=query_text)
                    continue

                job_id = str(uuid.uuid4())
                ingest_sources_task.apply_async(
                    kwargs={
                        "job_id": job_id,
                        "user_id": 0,  # system-level job
                        "product_query": query_text,
                        "urls": urls,
                    },
                )
                jobs_enqueued += 1
            except Exception as exc:
                logger.warning(
                    "refresh_top_queries_enqueue_failed",
                    query=query_text,
                    error=str(exc),
                )

        logger.info(
            "refresh_top_queries_done",
            queries=len(queries_found),
            jobs_enqueued=jobs_enqueued,
            skipped_no_urls=skipped_no_urls,
        )
    except Exception as exc:
        logger.error("refresh_top_queries_error", error=str(exc))
    finally:
        await engine.dispose()

    return {
        "queries_found": len(queries_found),
        "jobs_enqueued": jobs_enqueued,
        "skipped_no_urls": skipped_no_urls,
        "top_queries": queries_found[:10],  # return first 10 for logging
    }


@celery_app.task(name="reviewmind.daily_reset_limits", bind=True, max_retries=0)
def daily_reset_limits_task(self: object) -> dict:
    """Celery Beat task: safety-net reset of user_limits at midnight UTC.

    The primary mechanism is **date-based keying** — each new UTC day
    the ``LimitService`` reads the counter for today's date, which starts
    at 0 (no row exists yet).  This task is a safety net that zeros out
    any rows pre-created for today.  Old rows are preserved for analytics.

    Scheduled to run daily at 00:00 UTC.
    """
    logger.info("daily_reset_limits_task_start")
    return _run_async(_daily_reset_limits())


@celery_app.task(name="reviewmind.refresh_top_queries", bind=True, max_retries=0)
def refresh_top_queries_task(self: object) -> dict:
    """Celery Beat task: refresh top-50 product queries.

    Scheduled to run on the 1st of every month at 03:00 UTC.
    """
    logger.info("refresh_top_queries_task_start")
    return _run_async(_refresh_top_queries())
