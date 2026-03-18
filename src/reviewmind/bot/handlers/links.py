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


def extract_query_text(text: str, urls: list[str]) -> str:
    """Return the non-URL portion of *text* as a user query.

    If the remaining text is empty, falls back to :data:`_DEFAULT_QUERY`.
    """
    query = text
    for url in urls:
        query = query.replace(url, "")
    query = re.sub(r"\s+", " ", query).strip()
    return query or _DEFAULT_QUERY


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
    log.info("links_received", urls=urls[:5])

    # Show typing indicator
    if message.bot:
        await message.bot.send_chat_action(chat_id=message.chat.id, action=ChatAction.TYPING)

    # Check daily limit
    limit_result = await _check_user_limit(user_id, log)
    if limit_result is not None and not limit_result.allowed:
        await message.answer(limit_result.message, reply_markup=subscribe_keyboard())
        return

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
        return

    try:
        await _ingest_and_analyse(
            message=message,
            status_msg=status_msg,
            urls=urls,
            query_text=query_text,
            qdrant=qdrant,
            log=log,
        )
    except Exception as exc:
        log.error("links_unexpected_error", error=str(exc))
        await status_msg.edit_text(_UNEXPECTED_ERROR_MSG)
    finally:
        await qdrant.close()


async def _ingest_and_analyse(
    *,
    message: Message,
    status_msg: Message,
    urls: list[str],
    query_text: str,
    qdrant,
    log,
) -> None:
    """Run the ingestion pipeline and RAG query, editing *status_msg* with the result."""
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
        return

    # ── Step 2: RAG query ────────────────────────────────────
    async with RAGPipeline(qdrant_client=qdrant) as rag:
        rag_response = await rag.query(
            user_query=query_text,
            product_query=product_query or None,
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

    await status_msg.edit_text(
        answer,
        reply_markup=feedback_keyboard(),
    )

    # Increment daily counter (best-effort)
    user_id = message.from_user.id if message.from_user else 0
    await _increment_user_limit(user_id, log)

    log.info(
        "links_analysis_sent",
        answer_len=len(answer),
        sources=rag_response.sources,
        confidence_met=rag_response.confidence_met,
    )
