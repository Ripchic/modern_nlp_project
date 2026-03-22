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
    "✅ Данные собраны, но не удалось сгенерировать анализ.\nПопробуйте задать вопрос по этому товару."
)

TASK_FAILED_MSG = "😔 К сожалению, не удалось собрать и обработать данные.\nПопробуйте ещё раз или измените запрос."

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


# ── Admin alerts ─────────────────────────────────────────────────────────────

ADMIN_ALERT_TEMPLATE = (
    "🚨 <b>Task failed after all retries</b>\n\n"
    "<b>Task ID:</b> <code>{task_id}</code>\n"
    "<b>Job ID:</b> <code>{job_id}</code>\n"
    "<b>User ID:</b> <code>{user_id}</code>\n"
    "<b>Product:</b> {product_query}\n"
    "<b>Error:</b> <code>{error}</code>\n"
    "<b>Retries exhausted:</b> {max_retries}"
)

_MAX_ERROR_LENGTH = 500  # truncate error message in admin alerts


async def send_admin_alert(
    *,
    bot_token: str,
    admin_user_ids: list[int],
    task_id: str,
    job_id: str,
    user_id: int,
    product_query: str,
    error: str,
    max_retries: int = 3,
) -> None:
    """Send an alert to all admin users when a task fails after all retries.

    Parameters
    ----------
    bot_token:
        Telegram bot token from config.
    admin_user_ids:
        List of Telegram user IDs for admin notifications.
    task_id:
        Celery task ID.
    job_id:
        UUID of the Job row.
    user_id:
        Telegram user ID of the requesting user.
    product_query:
        Product name from the failed task.
    error:
        Error message from the final failure.
    max_retries:
        Number of retries attempted.
    """
    if not admin_user_ids:
        logger.warning("admin_alert_no_admins", task_id=task_id)
        return

    error_truncated = error[:_MAX_ERROR_LENGTH] if len(error) > _MAX_ERROR_LENGTH else error

    text = ADMIN_ALERT_TEMPLATE.format(
        task_id=task_id,
        job_id=job_id,
        user_id=user_id,
        product_query=product_query,
        error=error_truncated,
        max_retries=max_retries,
    )

    bot = _create_bot(bot_token)
    try:
        for admin_id in admin_user_ids:
            try:
                await bot.send_message(chat_id=admin_id, text=text)
                logger.info("admin_alert_sent", admin_id=admin_id, task_id=task_id)
            except Exception as exc:
                logger.error(
                    "admin_alert_send_failed",
                    admin_id=admin_id,
                    task_id=task_id,
                    error=str(exc),
                )
    finally:
        await bot.session.close()
