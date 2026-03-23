"""reviewmind/services/query_service.py — Query orchestration service.

Temporary direct-LLM implementation (no RAG). Will be replaced by
RAGPipeline in TASK-024.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog

from reviewmind.core.llm import LLMClient, LLMError
from reviewmind.core.prompts import FALLBACK_SYSTEM_PROMPT

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

# ── System prompt — reuse the FALLBACK prompt from prompts.py ─
_SYSTEM_PROMPT = FALLBACK_SYSTEM_PROMPT

_ERROR_RESPONSE = (
    "😔 Извините, произошла ошибка при обработке вашего запроса. Пожалуйста, попробуйте ещё раз через несколько секунд."
)

_EMPTY_MESSAGE_RESPONSE = "Пожалуйста, отправьте текстовое сообщение с вашим вопросом."


@dataclass
class QueryResult:
    """Result of a query execution."""

    answer: str
    error: bool = False
    error_message: str | None = None
    model: str | None = None
    chat_history: list[dict[str, str]] = field(default_factory=list)


class QueryService:
    """Orchestrates user queries through the LLM.

    This is a temporary implementation that calls the LLM directly without
    RAG context. It will be replaced by RAGPipeline integration later.

    Parameters
    ----------
    llm_client:
        An ``LLMClient`` instance.  If *None*, one is created internally
        using default config settings.
    system_prompt:
        Override the default system prompt (useful for testing).
    """

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        *,
        system_prompt: str | None = None,
    ) -> None:
        self._llm = llm_client
        self._owns_client = llm_client is None
        self._system_prompt = system_prompt or _SYSTEM_PROMPT

    @property
    def llm(self) -> LLMClient:
        """Lazily create an LLMClient if one was not injected."""
        if self._llm is None:
            self._llm = LLMClient()
            self._owns_client = True
        return self._llm

    async def answer(
        self,
        user_message: str,
        *,
        chat_history: list[dict[str, str]] | None = None,
    ) -> QueryResult:
        """Process a user query and return the LLM response.

        Parameters
        ----------
        user_message:
            The text of the user's question.
        chat_history:
            Optional list of prior ``{"role": ..., "content": ...}`` messages
            for multi-turn context.

        Returns
        -------
        QueryResult
            Contains the answer text and metadata.  If an error occurred,
            ``error`` is ``True`` and ``answer`` contains a user-friendly
            fallback message.
        """
        if not user_message or not user_message.strip():
            return QueryResult(answer=_EMPTY_MESSAGE_RESPONSE)

        try:
            response = await self.llm.generate(
                system_prompt=self._system_prompt,
                user_message=user_message.strip(),
                messages=chat_history,
            )

            logger.info(
                "query_success",
                user_message_len=len(user_message),
                response_len=len(response),
            )

            return QueryResult(
                answer=response,
                model=self.llm._model,
            )

        except LLMError as exc:
            logger.error("query_llm_error", error=str(exc))
            return QueryResult(
                answer=_ERROR_RESPONSE,
                error=True,
                error_message=str(exc),
            )

        except Exception as exc:
            logger.error("query_unexpected_error", error=str(exc))
            return QueryResult(
                answer=_ERROR_RESPONSE,
                error=True,
                error_message=f"Unexpected: {exc}",
            )

    async def close(self) -> None:
        """Close the LLM client if we own it."""
        if self._owns_client and self._llm is not None:
            await self._llm.close()

    async def __aenter__(self) -> QueryService:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()
