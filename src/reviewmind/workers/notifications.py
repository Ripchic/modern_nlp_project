"""reviewmind/workers/notifications.py — Telegram push notifications from Celery tasks.

After a background ingestion task finishes, these helpers:

1. Run a RAG query on the freshly ingested data.
2. Send the structured analysis to the user via Telegram ``Bot.send_message``.
3. On failure, send a user-friendly apology.

All functions are async — callers should use :func:`asyncio.run` (as Celery
tasks do via ``_run_async``).
"""

from __future__ import annotations

import structlog
from aiogram import Bot
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from qdrant_client import AsyncQdrantClient

from reviewmind.bot.keyboards import feedback_keyboard
from reviewmind.core.rag import RAGPipeline

logger = structlog.get_logger(__name__)

# ── Message templates ────────────────────────────────────────────────────────

TASK_STARTED_MSG = "⏳ Ищу данные (~3 мин)...\nЯ пришлю результат, когда анализ будет готов."

TASK_COMPLETED_NO_ANSWER_MSG = (
    "✅ Данные собраны, но не удалось сгенерировать анализ.\n"
    "Попробуйте задать вопрос по этому товару."
)

TASK_FAILED_MSG = (
    "😔 К сожалению, не удалось собрать и обработать данные.\n"
    "Попробуйте ещё раз или измените запрос."
)

_MAX_ANSWER_LENGTH = 4096  # Telegram message length limit


# ── Helpers ──────────────────────────────────────────────────────────────────


def _create_bot(token: str) -> Bot:
    """Create an aiogram Bot instance for sending a one-off message."""
    return Bot(
        token=token,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML),
    )


async def _run_rag_query(
    qdrant: AsyncQdrantClient,
    product_query: str,
) -> str | None:
    """Run a RAG query on freshly ingested data and return the answer text.

    Returns *None* if the pipeline produces an empty or errored response.
    """
    async with RAGPipeline(qdrant_client=qdrant) as rag:
        response = await rag.query(
            user_query=product_query,
            product_query=product_query,
        )

    if response.error:
        logger.warning("push_rag_error", error=response.error)

    return response.answer if response.answer else None


# ── Public API ───────────────────────────────────────────────────────────────


async def send_task_started(
    *,
    bot_token: str,
    chat_id: int,
) -> None:
    """Send the '⏳ Ищу данные ...' notification when a background job starts.

    Parameters
    ----------
    bot_token:
        Telegram bot token from config.
    chat_id:
        Telegram chat ID (typically ``user_id``) to send the message to.
    """
    bot = _create_bot(bot_token)
    try:
        await bot.send_message(chat_id=chat_id, text=TASK_STARTED_MSG)
        logger.info("push_task_started_sent", chat_id=chat_id)
    except Exception as exc:
        logger.error("push_task_started_failed", chat_id=chat_id, error=str(exc))
    finally:
        await bot.session.close()


async def send_task_completed(
    *,
    bot_token: str,
    chat_id: int,
    product_query: str,
    qdrant_url: str,
) -> None:
    """Send the full RAG analysis after a successful background ingestion.

    Steps:
    1. Connect to Qdrant.
    2. Run a RAG query using the *product_query* from the job.
    3. Send the structured answer with feedback buttons.
    4. If RAG fails or returns empty → send a fallback message.

    Parameters
    ----------
    bot_token:
        Telegram bot token from config.
    chat_id:
        Telegram chat ID to send the analysis to.
    product_query:
        The product name / query used during ingestion.
    qdrant_url:
        Qdrant server URL for vector search.
    """
    log = logger.bind(chat_id=chat_id, product_query=product_query)
    log.info("push_task_completed_start")

    bot = _create_bot(bot_token)
    qdrant: AsyncQdrantClient | None = None

    try:
        qdrant = AsyncQdrantClient(url=qdrant_url, timeout=30)

        answer = await _run_rag_query(qdrant, product_query)

        if answer:
            text = answer
            if len(text) > _MAX_ANSWER_LENGTH:
                text = text[: _MAX_ANSWER_LENGTH - 3] + "..."
            await bot.send_message(
                chat_id=chat_id,
                text=text,
                reply_markup=feedback_keyboard(),
            )
            log.info("push_analysis_sent", answer_len=len(text))
        else:
            await bot.send_message(
                chat_id=chat_id,
                text=TASK_COMPLETED_NO_ANSWER_MSG,
            )
            log.warning("push_analysis_empty")

    except Exception as exc:
        log.error("push_task_completed_failed", error=str(exc))
        try:
            await bot.send_message(
                chat_id=chat_id,
                text=TASK_COMPLETED_NO_ANSWER_MSG,
            )
        except Exception:
            log.error("push_fallback_send_failed")
    finally:
        if qdrant is not None:
            await qdrant.close()
        await bot.session.close()


async def send_task_failed(
    *,
    bot_token: str,
    chat_id: int,
) -> None:
    """Send an apology message when a background job fails.

    Parameters
    ----------
    bot_token:
        Telegram bot token from config.
    chat_id:
        Telegram chat ID to notify.
    """
    bot = _create_bot(bot_token)
    try:
        await bot.send_message(chat_id=chat_id, text=TASK_FAILED_MSG)
        logger.info("push_task_failed_sent", chat_id=chat_id)
    except Exception as exc:
        logger.error("push_task_failed_send_error", chat_id=chat_id, error=str(exc))
    finally:
        await bot.session.close()
