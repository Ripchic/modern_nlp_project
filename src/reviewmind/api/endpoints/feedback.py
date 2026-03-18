"""reviewmind/api/endpoints/feedback.py — POST /feedback endpoint.

Accepts a ``query_log_id`` and ``rating`` (1 for 👍, -1 for 👎) and
persists the rating to the ``query_logs`` table.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from reviewmind.api.rate_limit import RATE_LIMIT_STRING, _check_exempt, limiter
from reviewmind.api.schemas import FeedbackRequest, FeedbackResponse
from reviewmind.db.repositories.query_logs import QueryLogRepository

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

router = APIRouter()


@router.post("/feedback", response_model=FeedbackResponse)
@limiter.limit(RATE_LIMIT_STRING, exempt_when=_check_exempt)
async def post_feedback(body: FeedbackRequest, request: Request) -> FeedbackResponse:
    """Update the rating on an existing query log entry."""
    log = logger.bind(query_log_id=body.query_log_id, rating=body.rating)
    log.info("api_feedback_received")

    engine = getattr(request.app.state, "db_engine", None)
    if engine is None:
        log.warning("feedback_no_db_engine")
        raise HTTPException(status_code=503, detail="Database unavailable")

    try:
        session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
        async with session_factory() as session:
            repo = QueryLogRepository(session)
            log_entry = await repo.update_rating(body.query_log_id, body.rating)
            if log_entry is None:
                raise HTTPException(status_code=404, detail="Query log not found")
            await session.commit()
    except HTTPException:
        raise
    except Exception as exc:
        log.error("feedback_db_error", error=str(exc))
        raise HTTPException(status_code=500, detail="Failed to save feedback") from exc

    log.info("api_feedback_saved", query_log_id=body.query_log_id, new_rating=body.rating)
    return FeedbackResponse(
        query_log_id=body.query_log_id,
        rating=body.rating,
        message="Спасибо за отзыв!",
    )
