"""reviewmind/bot/handlers/query.py — Обработка текстовых запросов пользователей.

Temporary direct-LLM handler (without RAG).  Will be replaced by
RAG-pipeline integration in TASK-024.
"""

from __future__ import annotations

import structlog
from aiogram import Router
from aiogram.enums import ChatAction
from aiogram.types import Message

from reviewmind.core.llm import LLMClient
from reviewmind.services.query_service import QueryService

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

router = Router(name="query")

_MAX_ANSWER_LENGTH = 4096  # Telegram message length limit


@router.message()
async def on_text_message(message: Message) -> None:
    """Handle any text message — send it to LLM and reply with the answer.

    Shows a typing indicator while waiting for the LLM response.
    On error, sends a user-friendly fallback message (no tracebacks).
    """
    if not message.text:
        return

    user_id = message.from_user.id if message.from_user else 0
    logger.info("query_received", user_id=user_id, text_len=len(message.text))

    # Show typing indicator
    await message.bot.send_chat_action(chat_id=message.chat.id, action=ChatAction.TYPING)

    async with LLMClient() as client:
        service = QueryService(llm_client=client)
        result = await service.answer(message.text)

    if result.error:
        logger.warning(
            "query_error_sent",
            user_id=user_id,
            error=result.error_message,
        )

    # Telegram has a 4096-char limit; truncate if needed
    answer = result.answer
    if len(answer) > _MAX_ANSWER_LENGTH:
        answer = answer[: _MAX_ANSWER_LENGTH - 3] + "..."

    await message.answer(answer)

    logger.info(
        "query_answered",
        user_id=user_id,
        answer_len=len(answer),
        error=result.error,
    )
