"""reviewmind/bot/handlers/links.py — Приём и обработка URL в режиме «Свои ссылки».

When a user sends a message containing one or more HTTP(S) URLs, this handler:

1. Extracts URLs from the message text.
2. Ingests each URL through the IngestionPipeline
   (scrape → clean → chunk → embed → upsert).
3. Runs a RAG query on the ingested content.
4. Sends the structured analysis back with feedback buttons.
"""

from __future__ import annotations

import re

import structlog
from aiogram import Router
from aiogram.enums import ChatAction
from aiogram.types import Message
from qdrant_client import AsyncQdrantClient

from reviewmind.bot.keyboards import feedback_keyboard, subscribe_keyboard
from reviewmind.cache.redis import SessionManager
from reviewmind.core.rag import RAGPipeline
from reviewmind.ingestion.pipeline import IngestionPipeline

logger = structlog.get_logger(__name__)

router = Router(name="links")

# ── Constants ────────────────────────────────────────────────────────────────

_URL_RE = re.compile(r"https?://[^\s<>\"']+")
_MAX_ANSWER_LENGTH = 4096  # Telegram message length limit


async def _message_has_url(message: Message) -> bool:
    """Filter: True when the message text contains at least one HTTP(S) URL.

    Uses re.search so the URL can appear anywhere in the text, not just
    at the start (F.text.regexp uses re.match which only checks the start).
    """
    return bool(_URL_RE.search(message.text or ""))
_DEFAULT_QUERY = "Проанализируй эти обзоры"

_PROCESSING_TEMPLATE = "🔄 Обрабатываю {count} {word}...\nЭто может занять некоторое время."
_NO_SUCCESS_MSG = "❌ Не удалось обработать ни одну ссылку."
_ANALYSIS_ERROR_MSG = (
    "⚠️ Ссылки успешно обработаны, но не удалось сгенерировать анализ. "
    "Попробуйте задать вопрос по этим ссылкам."
)
_SERVICE_UNAVAILABLE_MSG = "⚠️ Сервис анализа временно недоступен. Попробуйте позже."
_UNEXPECTED_ERROR_MSG = "⚠️ Произошла ошибка при обработке ссылок. Попробуйте ещё раз."

# Trailing punctuation that is commonly appended by users or Telegram
_TRAILING_JUNK = ".,;:!?)"


# ── Helpers ──────────────────────────────────────────────────────────────────


def extract_urls(text: str) -> list[str]:
    """Extract unique HTTP(S) URLs from *text*, preserving first-seen order.

    Strips trailing punctuation that is unlikely to be part of the URL.
    """
    seen: set[str] = set()
    urls: list[str] = []
    for match in _URL_RE.finditer(text):
        url = match.group().rstrip(_TRAILING_JUNK)
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


# Common filler phrases users type around a URL (Russian + English).
# If the remaining text after URL removal matches one of these — it carries
# no analytical intent, so we fall back to the default analysis query.
_FILLER_RE = re.compile(
    r"^[\.,:;!?\-–—\s]*"
    r"(ссылка|ссылку|обзор|вот|смотри|видео|линк|link|review|check|here)"
    r"[\s]*"
    r"(на|к|для|по|this|the|a)?"
    r"[\s]*"
    r"(обзор|товар|видео|ссылку|ссылка|review|video|link)?"
    r"[:.;!?\-–—\s]*$",
    re.IGNORECASE,
)


def extract_query_text(text: str, urls: list[str]) -> str:
    """Return the non-URL portion of *text* as a user query.

    If the remaining text is empty or is a common filler phrase around a link
    (e.g. "Ссылка на обзор:"), falls back to :data:`_DEFAULT_QUERY`.
    """
    query = text
    for url in urls:
        query = query.replace(url, "")
    query = re.sub(r"\s+", " ", query).strip()
    if not query or _FILLER_RE.match(query):
        return _DEFAULT_QUERY
    return query


def _pluralize_links(count: int) -> str:
    """Russian plural for «ссылка»."""
    if count % 10 == 1 and count % 100 != 11:
        return "ссылку"
    if 2 <= count % 10 <= 4 and not 12 <= count % 100 <= 14:
        return "ссылки"
    return "ссылок"


def _build_failure_lines(results) -> list[str]:
    """Build per-URL failure messages."""
    return [f"❌ Не удалось загрузить: {r.url}" for r in results if not r.success]


# ── Handler ──────────────────────────────────────────────────────────────────


def _create_qdrant_client() -> AsyncQdrantClient:
    """Create an ``AsyncQdrantClient`` from the application settings."""
    from reviewmind.config import settings

    return AsyncQdrantClient(url=settings.qdrant_url)


async def _create_session_manager() -> tuple[SessionManager, object]:
    """Create a Redis connection and :class:`SessionManager`.

    Returns ``(session_manager, redis_client)``.
    """
    from redis.asyncio import from_url as redis_from_url  # noqa: PLC0415

    from reviewmind.config import settings  # noqa: PLC0415

    client = redis_from_url(settings.redis_url, decode_responses=True)
    return SessionManager(client), client


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


async def _save_query_log(
    user_id: int,
    query_text: str,
    response_text: str,
    sources: list[str] | None = None,
    *,
    mode: str = "links",
    used_tavily: bool = False,
) -> int | None:
    """Persist a QueryLog row and return its id.  Best-effort."""
    try:
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine  # noqa: PLC0415

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


@router.message(_message_has_url)
async def on_links_message(message: Message) -> None:
    """Handle messages containing HTTP(S) URLs — ingest and analyse."""
    text = message.text or ""
    urls = extract_urls(text)

    if not urls:
        return

    user_id = message.from_user.id if message.from_user else 0
    query_text = extract_query_text(text, urls)
    log = logger.bind(user_id=user_id, url_count=len(urls))
    log.info(
        "links_received",
        urls=urls[:5],
        raw_text=text[:200],
        query_text=query_text,
    )

    # Show typing indicator
    if message.bot:
        await message.bot.send_chat_action(chat_id=message.chat.id, action=ChatAction.TYPING)

    # Check daily limit
    limit_result = await _check_user_limit(user_id, log)
    if limit_result is not None and not limit_result.allowed:
        await message.answer(limit_result.message, reply_markup=subscribe_keyboard())
        return

    # Retrieve chat history from Redis
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

    # Send processing status
    status_msg = await message.answer(
        _PROCESSING_TEMPLATE.format(count=len(urls), word=_pluralize_links(len(urls)))
    )

    # Create Qdrant client
    try:
        qdrant = _create_qdrant_client()
    except Exception as exc:
        log.error("qdrant_connect_failed", error=str(exc))
        await status_msg.edit_text(_SERVICE_UNAVAILABLE_MSG)
        await _close_redis(redis_client)
        return

    try:
        answer_text = await _ingest_and_analyse(
            message=message,
            status_msg=status_msg,
            urls=urls,
            query_text=query_text,
            qdrant=qdrant,
            log=log,
            chat_history=chat_history,
        )
        # Store exchange in Redis history
        await _store_exchange(session_mgr, user_id, text, answer_text, log)
    except Exception as exc:
        log.error("links_unexpected_error", error=str(exc))
        await status_msg.edit_text(_UNEXPECTED_ERROR_MSG)
    finally:
        await qdrant.close()
        await _close_redis(redis_client)


async def _ingest_and_analyse(
    *,
    message: Message,
    status_msg: Message,
    urls: list[str],
    query_text: str,
    qdrant,
    log,
    chat_history: list[dict[str, str]] | None = None,
) -> str | None:
    """Run the ingestion pipeline and RAG query, editing *status_msg* with the result.

    Returns the answer text sent to the user (for session history), or ``None``.
    """
    product_query = query_text if query_text != _DEFAULT_QUERY else ""

    # ── Step 1: Ingest URLs ──────────────────────────────────
    async with IngestionPipeline(qdrant_client=qdrant) as pipeline:
        ingestion_result = await pipeline.ingest_urls(
            urls=urls,
            product_query=product_query,
        )

    failed_lines = _build_failure_lines(ingestion_result.results)

    log.info(
        "links_ingestion_done",
        success=ingestion_result.success_count,
        failed=ingestion_result.failed_count,
        chunks=ingestion_result.chunks_count,
    )

    # All URLs failed → short-circuit
    if ingestion_result.success_count == 0:
        parts = [_NO_SUCCESS_MSG]
        if failed_lines:
            parts.append("\n".join(failed_lines))
        await status_msg.edit_text("\n\n".join(parts))
        return None

    # ── Step 2: RAG query ────────────────────────────────────
    # NOTE: Do NOT pass product_query as a Qdrant filter here.
    # In links mode the "product_query" is the user's free-text question
    # (e.g. "Сравни шлемы из видео"), not an exact product name.
    # Using it as an exact-match payload filter would return 0 results
    # whenever the text doesn't match what was stored during ingestion.
    # Vector similarity on user_query is sufficient to find relevant chunks.
    log.info(
        "links_rag_query_start",
        query_text=query_text,
        product_query=product_query or None,
    )
    # Pass ALL user URLs for direct retrieval — chunks from previous
    # ingestion runs are still in Qdrant and should be used even if
    # this scrape attempt failed (e.g. YouTube IP block).
    async with RAGPipeline(qdrant_client=qdrant) as rag:
        rag_response = await rag.query(
            user_query=query_text,
            product_query=None,
            chat_history=chat_history or None,
            source_urls=urls,
        )
    log.info(
        "links_rag_query_done",
        answer_preview=rag_response.answer[:150] if rag_response.answer else "<empty>",
        sources_count=len(rag_response.sources),
        confidence_met=rag_response.confidence_met,
        chunks_count=rag_response.chunks_count,
        chunks_found=rag_response.chunks_found,
        used_tavily=rag_response.used_tavily,
        error=rag_response.error,
    )

    # ── Step 3: Build final answer ───────────────────────────
    parts: list[str] = []

    if failed_lines:
        parts.append("\n".join(failed_lines))

    if rag_response.answer:
        parts.append(rag_response.answer)
    else:
        parts.append(_ANALYSIS_ERROR_MSG)
        log.warning("links_rag_empty", error=rag_response.error)

    # Summary line
    summary = f"📊 Обработано источников: {ingestion_result.success_count}/{len(urls)}"
    if ingestion_result.chunks_count:
        summary += f" | Фрагментов: {ingestion_result.chunks_count}"
    parts.append(summary)

    answer = "\n\n".join(parts)
    if len(answer) > _MAX_ANSWER_LENGTH:
        answer = answer[: _MAX_ANSWER_LENGTH - 3] + "..."

    # ── Step 4: Persist query log so sources button works ─────
    user_id = message.from_user.id if message.from_user else 0
    query_log_id = await _save_query_log(
        user_id=user_id,
        query_text=query_text,
        response_text=answer,
        sources=rag_response.sources,
        mode="links",
        used_tavily=rag_response.used_tavily,
    )
    log.info("links_query_log_saved", query_log_id=query_log_id)

    await status_msg.edit_text(
        answer,
        reply_markup=feedback_keyboard(query_log_id),
    )

    # Increment daily counter (best-effort)
    await _increment_user_limit(user_id, log)

    log.info(
        "links_analysis_sent",
        answer_len=len(answer),
        sources=rag_response.sources,
        confidence_met=rag_response.confidence_met,
        query_log_id=query_log_id,
    )

    return answer
