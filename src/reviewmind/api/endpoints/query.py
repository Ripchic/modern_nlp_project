"""reviewmind/api/endpoints/query.py — POST /query endpoint.

Full RAG-query endpoint with response timing and query logging.
"""

from __future__ import annotations

import time

import structlog
from fastapi import APIRouter, Request
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from reviewmind.api.rate_limit import RATE_LIMIT_STRING, _check_exempt, limiter
from reviewmind.api.schemas import QueryRequest, QueryResponse
from reviewmind.core.llm import LLMClient
from reviewmind.core.rag import RAGPipeline
from reviewmind.db.repositories.query_logs import QueryLogRepository
from reviewmind.services.limit_service import LimitService
from reviewmind.services.query_service import QueryService

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

router = APIRouter()

_FALLBACK_ANSWER = (
    "😔 Извините, не удалось обработать ваш запрос. Пожалуйста, попробуйте ещё раз через несколько секунд."
)


async def _log_query(
    request: Request,
    *,
    user_id: int,
    session_id: str | None,
    mode: str | None,
    query_text: str,
    response_text: str,
    sources_used: list[str] | None,
    response_time_ms: int,
    used_tavily: bool,
) -> int | None:
    """Persist a query log entry if a database session is available.

    Returns the query_log id on success, or ``None`` if DB is unavailable.
    """
    engine = getattr(request.app.state, "db_engine", None)
    if engine is None:
        logger.debug("query_log_skipped", reason="no_db_engine")
        return None

    try:
        session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
        async with session_factory() as session:
            repo = QueryLogRepository(session)
            log_entry = await repo.create(
                user_id=user_id,
                session_id=session_id,
                mode=mode,
                query_text=query_text,
                response_text=response_text,
                sources_used=sources_used,
                response_time_ms=response_time_ms,
                used_tavily=used_tavily,
            )
            await session.commit()
            return log_entry.id
    except Exception as exc:
        logger.warning("query_log_failed", error=str(exc))
        return None


async def _check_limit(request: Request, user_id: int) -> object | None:
    """Check daily limit.  Returns *None* when DB is unavailable."""
    engine = getattr(request.app.state, "db_engine", None)
    if engine is None:
        return None
    try:
        session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
        async with session_factory() as session:
            service = LimitService(session)
            result = await service.check_limit(user_id)
            await session.commit()
        return result
    except Exception as exc:
        logger.warning("limit_check_failed", error=str(exc))
        return None


async def _increment_limit(request: Request, user_id: int) -> None:
    """Increment daily counter.  Best-effort — never raises."""
    engine = getattr(request.app.state, "db_engine", None)
    if engine is None:
        return
    try:
        session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
        async with session_factory() as session:
            service = LimitService(session)
            await service.increment(user_id)
            await session.commit()
    except Exception as exc:
        logger.warning("limit_increment_failed", error=str(exc))


@router.post("/query", response_model=QueryResponse)
@limiter.limit(RATE_LIMIT_STRING, exempt_when=_check_exempt)
async def post_query(body: QueryRequest, request: Request) -> QueryResponse:
    """Accept a user query, run RAG pipeline, log result, and return the answer."""
    start_ms = time.monotonic()
    log = logger.bind(user_id=body.user_id, mode=body.mode, session_id=body.session_id)
    log.info("api_query_received", query_len=len(body.query))

    # ── Check daily limit ────────────────────────────────────
    limit_result = await _check_limit(request, body.user_id)
    if limit_result is not None and not limit_result.allowed:
        elapsed_ms = int((time.monotonic() - start_ms) * 1000)
        return QueryResponse(
            answer=limit_result.message,
            response_time_ms=elapsed_ms,
            error=True,
        )

    qdrant = getattr(request.app.state, "qdrant", None)

    # If Qdrant is unavailable, fall back to direct LLM
    if qdrant is None:
        log.warning("qdrant_unavailable_falling_back_to_llm")

        try:
            async with LLMClient() as client:
                service = QueryService(llm_client=client)
                result = await service.answer(body.query, chat_history=body.chat_history)
            elapsed_ms = int((time.monotonic() - start_ms) * 1000)
            query_log_id = await _log_query(
                request,
                user_id=body.user_id,
                session_id=body.session_id,
                mode=body.mode,
                query_text=body.query,
                response_text=result.answer,
                sources_used=None,
                response_time_ms=elapsed_ms,
                used_tavily=False,
            )
            await _increment_limit(request, body.user_id)
            return QueryResponse(
                answer=result.answer,
                response_time_ms=elapsed_ms,
                query_log_id=query_log_id,
                error=result.error,
            )
        except Exception as exc:
            log.error("fallback_llm_error", error=str(exc))
            elapsed_ms = int((time.monotonic() - start_ms) * 1000)
            return QueryResponse(
                answer=_FALLBACK_ANSWER,
                response_time_ms=elapsed_ms,
                error=True,
            )

    # Full RAG pipeline
    try:
        pipeline = RAGPipeline(qdrant_client=qdrant)
        rag_result = await pipeline.query(
            user_query=body.query,
            chat_history=body.chat_history,
            product_query=body.product_query,
            session_id=body.session_id,
        )
        await pipeline.close()
    except Exception as exc:
        log.error("rag_pipeline_error", error=str(exc))
        elapsed_ms = int((time.monotonic() - start_ms) * 1000)
        return QueryResponse(
            answer=_FALLBACK_ANSWER,
            response_time_ms=elapsed_ms,
            error=True,
        )

    elapsed_ms = int((time.monotonic() - start_ms) * 1000)

    answer = rag_result.answer or _FALLBACK_ANSWER
    has_error = bool(rag_result.error) or not rag_result.answer

    log.info(
        "api_query_done",
        response_time_ms=elapsed_ms,
        chunks_count=rag_result.chunks_count,
        sources_count=len(rag_result.sources),
        confidence_met=rag_result.confidence_met,
        used_curated=rag_result.used_curated,
        used_tavily=rag_result.used_tavily,
    )

    query_log_id = await _log_query(
        request,
        user_id=body.user_id,
        session_id=body.session_id,
        mode=body.mode,
        query_text=body.query,
        response_text=answer,
        sources_used=rag_result.sources if rag_result.sources else None,
        response_time_ms=elapsed_ms,
        used_tavily=rag_result.used_tavily,
    )

    await _increment_limit(request, body.user_id)

    return QueryResponse(
        answer=answer,
        sources=rag_result.sources,
        used_curated=rag_result.used_curated,
        used_tavily=rag_result.used_tavily,
        confidence_met=rag_result.confidence_met,
        chunks_count=rag_result.chunks_count,
        response_time_ms=elapsed_ms,
        query_log_id=query_log_id,
        error=has_error,
    )
