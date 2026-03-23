"""reviewmind/bot/handlers/query.py — Авто-режим: полный pipeline для текстовых запросов.

Flow:
1. Extract product name from the user query.
2. If product found → check Qdrant cache for existing data.
3. If cached (confidence met) → instant RAG answer (< 3 sec).
4. If no/insufficient data → quick Tavily answer + Celery background
   job (YouTube + Reddit search → ingest → push final answer).
5. If no product detected → fallback to direct LLM.
"""

from __future__ import annotations

import uuid

import structlog
from aiogram import Router
from aiogram.enums import ChatAction
from aiogram.types import Message
from qdrant_client import AsyncQdrantClient

from reviewmind.bot.keyboards import feedback_keyboard, subscribe_keyboard
from reviewmind.cache.redis import SessionManager
from reviewmind.core.llm import LLMClient
from reviewmind.core.rag import RAGPipeline
from reviewmind.services.comparison_service import compare_products, detect_comparison
from reviewmind.services.product_extractor import extract_product
from reviewmind.services.query_service import QueryService

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

router = Router(name="query")

# ── Constants ────────────────────────────────────────────────────────────────

_MAX_ANSWER_LENGTH = 4096  # Telegram message length limit

_SEARCHING_MSG = "⏳ Ищу данные (~3 мин)...\nЯ пришлю результат, когда анализ будет готов."
_NO_PRODUCT_FALLBACK_NOTE = (
    "\n\n<i>💡 Совет: назовите конкретный товар (например, «Sony WH-1000XM5»), "
    "чтобы я нашёл и проанализировал обзоры.</i>"
)
_SERVICE_UNAVAILABLE_MSG = "⚠️ Сервис анализа временно недоступен. Попробуйте позже."
_UNEXPECTED_ERROR_MSG = "⚠️ Произошла непредвиденная ошибка. Попробуйте ещё раз."

MAX_SEARCH_URLS = 10  # cap on URLs collected from YouTube + Reddit search


# ── Helpers ──────────────────────────────────────────────────────────────────


def _truncate(text: str, limit: int = _MAX_ANSWER_LENGTH) -> str:
    """Truncate *text* to fit Telegram's message length limit."""
    if len(text) > limit:
        return text[: limit - 3] + "..."
    return text


async def _save_query_log(
    user_id: int,
    query_text: str,
    response_text: str,
    sources: list[str] | None = None,
    *,
    mode: str = "auto",
    used_tavily: bool = False,
) -> int | None:
    """Persist a QueryLog row and return its id.  Best-effort."""
    try:
        from sqlalchemy.ext.asyncio import (  # noqa: PLC0415
            AsyncSession,
            async_sessionmaker,
            create_async_engine,
        )

        from reviewmind.config import settings  # noqa: PLC0415
        from reviewmind.db.repositories.query_logs import QueryLogRepository  # noqa: PLC0415
        from reviewmind.db.repositories.users import UserRepository  # noqa: PLC0415

        engine = create_async_engine(settings.database_url, pool_pre_ping=True, echo=False)
        session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
        try:
            async with session_factory() as session:
                await UserRepository(session).get_or_create(user_id)
                repo = QueryLogRepository(session)
                log_entry = await repo.create(
                    user_id=user_id,
                    mode=mode,
                    query_text=query_text,
                    response_text=response_text,
                    sources_used=sources,
                    used_tavily=used_tavily,
                )
                await session.commit()
                return log_entry.id
        finally:
            await engine.dispose()
    except Exception as exc:
        logger.warning("query_log_save_failed", error=str(exc), user_id=user_id)
        return None


def _create_qdrant_client() -> AsyncQdrantClient:
    """Create an ``AsyncQdrantClient`` from the application settings."""
    from reviewmind.config import settings  # noqa: PLC0415

    return AsyncQdrantClient(url=settings.qdrant_url)


async def _create_session_manager() -> tuple[SessionManager, object]:
    """Create a Redis connection and :class:`SessionManager`.

    Returns ``(session_manager, redis_client)`` — caller must close the
    client with ``await redis_client.aclose()`` when done.
    """
    from redis.asyncio import from_url as redis_from_url  # noqa: PLC0415

    from reviewmind.config import settings  # noqa: PLC0415

    client = redis_from_url(settings.redis_url, decode_responses=True)
    return SessionManager(client), client


def _build_search_query(product_names: list[str]) -> str:
    """Build a search query string for YouTube/Reddit from product names."""
    return " ".join(product_names) + " review"


async def _collect_source_urls(product_names: list[str]) -> list[str]:
    """Search YouTube + Reddit for review URLs.

    Returns up to :data:`MAX_SEARCH_URLS` unique URLs.
    """
    urls: list[str] = []
    seen: set[str] = set()
    search_query = _build_search_query(product_names)

    # ── YouTube search ───────────────────────────────────────
    try:
        from reviewmind.config import get_settings as _get_settings  # noqa: PLC0415
        from reviewmind.scrapers.youtube import YouTubeScraper  # noqa: PLC0415

        yt = YouTubeScraper(
            cookie_path=_get_settings().youtube_cookies_path or None,
        )
        videos = yt.search_videos(search_query, max_results=5)
        for v in videos:
            if v.url and v.url not in seen:
                seen.add(v.url)
                urls.append(v.url)
    except Exception as exc:
        logger.warning("auto_youtube_search_failed", error=str(exc))

    # ── Reddit search ────────────────────────────────────────
    try:
        from reviewmind.scrapers.reddit import RedditScraper  # noqa: PLC0415

        reddit = RedditScraper()
        posts = reddit.search_posts(search_query, limit=5)
        for p in posts:
            if p.url and p.url not in seen:
                seen.add(p.url)
                urls.append(p.url)
    except Exception as exc:
        logger.warning("auto_reddit_search_failed", error=str(exc))

    return urls[:MAX_SEARCH_URLS]


async def _schedule_background_job(
    *,
    user_id: int,
    product_query: str,
    urls: list[str],
    session_id: str | None = None,
) -> str | None:
    """Create a Job row in PostgreSQL and schedule a Celery ingestion task.

    Returns the ``job_id`` (UUID str) on success, or *None* if scheduling fails.
    """
    from reviewmind.config import settings  # noqa: PLC0415

    job_id = str(uuid.uuid4())
    log = logger.bind(job_id=job_id, user_id=user_id, product_query=product_query)

    # Create Job row in DB (best-effort)
    try:
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine  # noqa: PLC0415

        from reviewmind.db.repositories.jobs import JobRepository  # noqa: PLC0415

        engine = create_async_engine(settings.database_url, pool_pre_ping=True, echo=False)
        session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
        async with session_factory() as session:
            repo = JobRepository(session)
            await repo.create(
                user_id=user_id,
                job_type="auto_search",
                product_query=product_query,
                celery_task_id=None,  # updated below after apply_async
            )
            # Use our generated job_id for the row
            # The create method generates its own UUID, but we need ours for Celery
            # Let's update the session with our ID instead:
            await session.rollback()

        # Re-create with our job_id
        from reviewmind.db.models import Job  # noqa: PLC0415

        async with session_factory() as session:
            job = Job(
                id=uuid.UUID(job_id),
                user_id=user_id,
                job_type="auto_search",
                status="pending",
                product_query=product_query,
            )
            session.add(job)
            await session.flush()
            await session.commit()

        await engine.dispose()
        log.info("auto_job_created_in_db")
    except Exception as exc:
        log.warning("auto_job_db_create_failed", error=str(exc))

    # Schedule Celery task
    try:
        from reviewmind.workers.tasks import ingest_sources_task  # noqa: PLC0415

        result = ingest_sources_task.apply_async(
            kwargs={
                "job_id": job_id,
                "user_id": user_id,
                "product_query": product_query,
                "urls": urls,
                "session_id": session_id,
            },
        )
        log.info("auto_celery_task_scheduled", celery_task_id=result.id)
        return job_id
    except Exception as exc:
        log.error("auto_celery_schedule_failed", error=str(exc))
        return None


# ── Handler ──────────────────────────────────────────────────────────────────


@router.message()
async def on_text_message(message: Message) -> None:
    """Handle any text message in auto-mode.

    Full pipeline:
    1. Extract product name(s) from the query.
    2. If product found → try instant RAG from Qdrant.
    3. If Qdrant has confident data → return immediately.
    4. If insufficient data → send quick Tavily answer +
       schedule background Celery job → push final answer later.
    5. If no product detected → direct LLM fallback.
    """
    if not message.text:
        return

    user_id = message.from_user.id if message.from_user else 0
    log = logger.bind(user_id=user_id, text_len=len(message.text))
    log.info("query_received")

    # Show typing indicator
    if message.bot:
        await message.bot.send_chat_action(chat_id=message.chat.id, action=ChatAction.TYPING)

    # ── Step 0: Check daily limit ────────────────────────────
    limit_result = await _check_user_limit(user_id, log)
    if limit_result is not None and not limit_result.allowed:
        await message.answer(limit_result.message, reply_markup=subscribe_keyboard())
        return

    # ── Retrieve chat history from Redis ─────────────────────
    chat_history: list[dict[str, str]] = []
    session_mgr: SessionManager | None = None
    redis_client = None
    try:
        session_mgr, redis_client = await _create_session_manager()
        chat_history = await session_mgr.get_history(user_id)
        await session_mgr.refresh_ttl(user_id)
        log.debug("session_history_loaded", history_len=len(chat_history))
    except Exception as exc:
        log.warning("session_history_load_failed", error=str(exc))

    # ── Step 1: Extract product name(s) ──────────────────────
    try:
        product_names = await extract_product(message.text)
    except Exception:
        product_names = []

    log.info("auto_product_extracted", products=product_names)

    # No product detected → try RAG+Tavily first, then plain LLM fallback
    if not product_names:
        rag_response = None
        try:
            qdrant = _create_qdrant_client()
            try:
                rag_response = await _try_instant_rag(qdrant, message.text, None, log, chat_history=chat_history)
            finally:
                await qdrant.close()
        except Exception as exc:
            log.warning("no_product_rag_failed", error=str(exc))

        if rag_response and rag_response.answer:
            answer = _truncate(rag_response.answer)
            # Add tip only when Tavily was NOT used (meaning LLM answered from
            # its own knowledge) so user knows to name a specific product next.
            if not rag_response.used_tavily and len(answer) + len(_NO_PRODUCT_FALLBACK_NOTE) <= _MAX_ANSWER_LENGTH:
                answer += _NO_PRODUCT_FALLBACK_NOTE
            log_id = await _save_query_log(
                user_id,
                message.text,
                answer,
                sources=rag_response.sources,
                used_tavily=rag_response.used_tavily,
            )
            await message.answer(answer, reply_markup=feedback_keyboard(query_log_id=log_id))
            await _increment_user_limit(user_id, log)
            await _store_exchange(session_mgr, user_id, message.text, answer, log)

            # Schedule background ingestion for Tavily source URLs so they
            # are persisted in auto_crawled for future queries.
            if rag_response.used_tavily and rag_response.sources:
                tavily_urls = [s for s in rag_response.sources if s.startswith("http")]
                if tavily_urls:
                    job_id = await _schedule_background_job(
                        user_id=user_id,
                        product_query=message.text,
                        urls=tavily_urls,
                    )
                    if job_id:
                        log.info("no_product_ingestion_scheduled", job_id=job_id, url_count=len(tavily_urls))

            log.info(
                "no_product_tavily_answer",
                used_tavily=rag_response.used_tavily,
                answer_len=len(answer),
                sources=len(rag_response.sources),
            )
        else:
            # Tavily unavailable or returned no content — plain LLM as last resort
            await _fallback_llm_answer(message, user_id, log, chat_history=chat_history)
            await _store_exchange(session_mgr, user_id, message.text, log=log)

        await _close_redis(redis_client)
        return

    # ── Step 1b: Comparison detection ────────────────────────
    if len(product_names) >= 2 and detect_comparison(message.text):
        log.info("comparison_query_detected", products=product_names)
        await _handle_comparison(
            message=message,
            product_names=product_names,
            user_id=user_id,
            log=log,
            chat_history=chat_history,
            session_mgr=session_mgr,
            redis_client=redis_client,
        )
        return

    product_query = ", ".join(product_names)

    # ── Step 2: Try instant RAG from Qdrant ──────────────────
    try:
        qdrant = _create_qdrant_client()
    except Exception as exc:
        log.error("qdrant_connect_failed", error=str(exc))
        await message.answer(_SERVICE_UNAVAILABLE_MSG)
        await _close_redis(redis_client)
        return

    try:
        rag_response = await _try_instant_rag(qdrant, message.text, product_query, log, chat_history=chat_history)
    except Exception as exc:
        log.error("instant_rag_failed", error=str(exc))
        rag_response = None
    finally:
        await qdrant.close()

    # ── Step 3: If confidence met → instant answer ───────────
    if rag_response and rag_response.confidence_met and rag_response.answer:
        answer = _truncate(rag_response.answer)
        log_id = await _save_query_log(
            user_id,
            message.text,
            answer,
            sources=rag_response.sources,
            used_tavily=getattr(rag_response, "used_tavily", False),
        )
        await message.answer(answer, reply_markup=feedback_keyboard(query_log_id=log_id))
        await _increment_user_limit(user_id, log)
        # Store user + assistant messages in history
        await _store_exchange(session_mgr, user_id, message.text, answer, log)
        await _close_redis(redis_client)
        log.info(
            "auto_instant_rag_answer",
            answer_len=len(answer),
            chunks=rag_response.chunks_count,
            sources=len(rag_response.sources),
            query_log_id=log_id,
        )
        return

    # ── Step 4: Insufficient data → Tavily quick + background job ────
    log.info("auto_cache_miss", product_query=product_query)

    # 4a: Send quick Tavily-based answer if available
    quick_answer_sent = False
    if rag_response and rag_response.answer and rag_response.used_tavily:
        # RAG already triggered Tavily fallback — use that answer
        answer = _truncate(rag_response.answer)
        log_id = await _save_query_log(
            user_id,
            message.text,
            answer,
            sources=rag_response.sources,
            used_tavily=True,
        )
        await message.answer(answer, reply_markup=feedback_keyboard(query_log_id=log_id))
        await _increment_user_limit(user_id, log)
        quick_answer_sent = True
        log.info("auto_tavily_quick_answer_sent", answer_len=len(answer), query_log_id=log_id)

    # 4b: Collect source URLs from YouTube + Reddit
    source_urls = await _collect_source_urls(product_names)

    # Merge Tavily source URLs (from step 4a) into collected URLs for ingestion
    if rag_response and rag_response.used_tavily and rag_response.sources:
        seen = set(source_urls)
        for url in rag_response.sources:
            if url.startswith("http") and url not in seen:
                seen.add(url)
                source_urls.append(url)

    log.info("auto_source_urls_collected", url_count=len(source_urls))

    if source_urls:
        # Schedule background ingestion + push notification
        job_id = await _schedule_background_job(
            user_id=user_id,
            product_query=product_query,
            urls=source_urls,
        )
        if job_id:
            if not quick_answer_sent:
                await message.answer(_SEARCHING_MSG)
            else:
                await message.answer("🔍 Ищу дополнительные источники для более полного анализа...")
            log.info("auto_background_job_scheduled", job_id=job_id)
        else:
            if not quick_answer_sent:
                # Celery unavailable and no quick answer → report error
                await message.answer(_SERVICE_UNAVAILABLE_MSG)
    else:
        # No source URLs found
        if not quick_answer_sent:
            # Try a final direct LLM answer as last resort
            await _fallback_llm_answer(message, user_id, log, chat_history=chat_history)
        else:
            log.info("auto_no_additional_sources")

    # Store user message in history (answer may come later via push)
    await _store_exchange(session_mgr, user_id, message.text, log=log)
    await _close_redis(redis_client)


async def _handle_comparison(
    *,
    message: Message,
    product_names: list[str],
    user_id: int,
    log,
    chat_history: list[dict[str, str]] | None = None,
    session_mgr: SessionManager | None = None,
    redis_client=None,
) -> None:
    """Handle a comparison query by running parallel RAG and generating a table."""
    try:
        qdrant = _create_qdrant_client()
    except Exception as exc:
        log.error("comparison_qdrant_connect_failed", error=str(exc))
        await message.answer(_SERVICE_UNAVAILABLE_MSG)
        await _close_redis(redis_client)
        return

    try:
        result = await compare_products(
            query=message.text,
            qdrant_client=qdrant,
            chat_history=chat_history,
        )
    except Exception as exc:
        log.error("comparison_failed", error=str(exc))
        await message.answer(_UNEXPECTED_ERROR_MSG)
        await _store_exchange(session_mgr, user_id, message.text, log=log)
        await _close_redis(redis_client)
        await qdrant.close()
        return
    finally:
        try:
            await qdrant.close()
        except Exception:
            pass

    if result.answer:
        answer = _truncate(result.answer)
        log_id = await _save_query_log(
            user_id,
            message.text,
            answer,
            sources=result.sources,
        )
        await message.answer(answer, reply_markup=feedback_keyboard(query_log_id=log_id))
        await _increment_user_limit(user_id, log)
        await _store_exchange(session_mgr, user_id, message.text, answer, log)
        log.info(
            "comparison_answer_sent",
            answer_len=len(answer),
            products=result.products,
            total_chunks=result.total_chunks,
            sources=len(result.sources),
            query_log_id=log_id,
        )
    else:
        # Comparison failed — fall back to direct LLM
        log.warning("comparison_empty_answer", error=result.error)
        await _fallback_llm_answer(message, user_id, log, chat_history=chat_history)
        await _store_exchange(session_mgr, user_id, message.text, log=log)

    await _close_redis(redis_client)


async def _try_instant_rag(
    qdrant: AsyncQdrantClient,
    user_query: str,
    product_query: str | None,
    log,
    *,
    chat_history: list[dict[str, str]] | None = None,
) -> object | None:
    """Try a RAG query against existing Qdrant data.

    Returns the :class:`~reviewmind.core.rag.RAGResponse` or ``None``.
    The RAG pipeline includes Tavily fallback when confidence is not met.
    When *product_query* is ``None`` (no specific product detected), the
    pipeline still runs and Tavily is triggered on low Qdrant confidence.
    """
    async with RAGPipeline(qdrant_client=qdrant) as rag:
        return await rag.query(
            user_query=user_query,
            product_query=product_query,
            chat_history=chat_history or None,
        )


async def _check_user_limit(user_id: int, log) -> object | None:
    """Check daily limit for *user_id*.  Returns *None* if DB is unavailable."""
    try:
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine  # noqa: PLC0415

        from reviewmind.config import settings  # noqa: PLC0415
        from reviewmind.services.limit_service import LimitService  # noqa: PLC0415

        engine = create_async_engine(settings.database_url, pool_pre_ping=True, echo=False)
        session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
        async with session_factory() as session:
            service = LimitService(session, admin_user_ids=settings.admin_user_ids)
            result = await service.check_limit(user_id)
            await session.commit()
        await engine.dispose()
        return result
    except Exception as exc:
        log.warning("limit_check_failed", error=str(exc))
        return None


async def _increment_user_limit(user_id: int, log) -> None:
    """Increment the daily request counter for *user_id*.  Best-effort."""
    try:
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine  # noqa: PLC0415

        from reviewmind.config import settings  # noqa: PLC0415
        from reviewmind.services.limit_service import LimitService  # noqa: PLC0415

        engine = create_async_engine(settings.database_url, pool_pre_ping=True, echo=False)
        session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
        async with session_factory() as session:
            service = LimitService(session, admin_user_ids=settings.admin_user_ids)
            await service.increment(user_id)
            await session.commit()
        await engine.dispose()
    except Exception as exc:
        log.warning("limit_increment_failed", error=str(exc))


async def _store_exchange(
    session_mgr: SessionManager | None,
    user_id: int,
    user_text: str,
    assistant_text: str | None = None,
    log=None,
) -> None:
    """Store user and (optionally) assistant messages in Redis history."""
    if session_mgr is None:
        return
    try:
        await session_mgr.add_to_history(user_id, {"role": "user", "content": user_text})
        if assistant_text:
            await session_mgr.add_to_history(user_id, {"role": "assistant", "content": assistant_text})
    except Exception as exc:
        if log:
            log.warning("session_history_store_failed", error=str(exc))


async def _close_redis(redis_client) -> None:
    """Close the Redis client if not None."""
    if redis_client is not None:
        try:
            await redis_client.aclose()
        except Exception:
            pass


async def _fallback_llm_answer(
    message: Message, user_id: int, log, *, chat_history: list[dict[str, str]] | None = None
) -> None:
    """Send a direct LLM answer (no RAG) for non-product queries."""
    try:
        async with LLMClient() as client:
            service = QueryService(llm_client=client)
            result = await service.answer(message.text, chat_history=chat_history)
    except Exception as exc:
        log.error("fallback_llm_error", error=str(exc))
        await message.answer(_UNEXPECTED_ERROR_MSG)
        return

    answer = _truncate(result.answer)

    # Add tip about naming a specific product
    if len(answer) + len(_NO_PRODUCT_FALLBACK_NOTE) <= _MAX_ANSWER_LENGTH:
        answer += _NO_PRODUCT_FALLBACK_NOTE

    await message.answer(answer)
    await _increment_user_limit(user_id, log)
    log.info(
        "auto_fallback_llm_answer",
        answer_len=len(answer),
        error=result.error,
    )
