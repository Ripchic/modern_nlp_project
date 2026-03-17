"""reviewmind/api/endpoints/query.py — POST /query endpoint.

Full RAG-query endpoint with response timing and query logging.
"""

from __future__ import annotations

import time

import structlog
from fastapi import APIRouter, Request
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from reviewmind.api.schemas import QueryRequest, QueryResponse
from reviewmind.core.llm import LLMClient
from reviewmind.core.rag import RAGPipeline
from reviewmind.db.repositories.query_logs import QueryLogRepository
from reviewmind.services.query_service import QueryService

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

router = APIRouter()

_FALLBACK_ANSWER = (
    "😔 Извините, не удалось обработать ваш запрос. "
    "Пожалуйста, попробуйте ещё раз через несколько секунд."
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


@router.post("/query", response_model=QueryResponse)
async def post_query(body: QueryRequest, request: Request) -> QueryResponse:
    """Accept a user query, run RAG pipeline, log result, and return the answer."""
    start_ms = time.monotonic()
    log = logger.bind(user_id=body.user_id, mode=body.mode, session_id=body.session_id)
    log.info("api_query_received", query_len=len(body.query))

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
        used_tavily=False,
    )

    return QueryResponse(
        answer=answer,
        sources=rag_result.sources,
        used_curated=rag_result.used_curated,
        used_tavily=False,
        confidence_met=rag_result.confidence_met,
        chunks_count=rag_result.chunks_count,
        response_time_ms=elapsed_ms,
        query_log_id=query_log_id,
        error=has_error,
    )
