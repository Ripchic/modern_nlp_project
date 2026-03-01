"""reviewmind/core/llm.py — Async OpenAI client wrapper (configurable base_url).

Supports both OpenAI API and GitHub Models API via OPENAI_BASE_URL switch.
Implements retry with exponential backoff for transient errors.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import openai
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# Transient error types that warrant a retry
_RETRYABLE_ERRORS = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)

# Defaults
_DEFAULT_MODEL = "gpt-4o-mini"
_DEFAULT_TEMPERATURE = 0.3
_DEFAULT_MAX_TOKENS = 1000
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_RETRY_BASE_DELAY = 1.0  # seconds
_DEFAULT_TIMEOUT = 60.0  # seconds


class LLMError(Exception):
    """Base exception for LLM client errors."""


class LLMClient:
    """Async wrapper around ``openai.AsyncOpenAI``.

    Parameters
    ----------
    api_key:
        OpenAI / GitHub Models API key. If *None*, read from
        ``settings.openai_api_key`` at instantiation time.
    base_url:
        API base URL. If *None*, read from ``settings.openai_base_url``.
    model:
        Chat-completion model name.
    temperature:
        Default sampling temperature (0–2).
    max_tokens:
        Default maximum tokens in the response.
    max_retries:
        Maximum number of retry attempts for transient errors.
    timeout:
        Request timeout in seconds.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        # Lazy-import settings so the module can be imported without a .env
        from reviewmind.config import settings

        self._api_key = api_key or settings.openai_api_key
        self._base_url = base_url or settings.openai_base_url
        self._model = model or settings.openai_model or _DEFAULT_MODEL
        self._temperature = temperature if temperature is not None else settings.llm_temperature
        self._max_tokens = max_tokens if max_tokens is not None else settings.llm_max_tokens
        self._max_retries = max_retries
        self._timeout = timeout

        self._client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=self._timeout,
            max_retries=0,  # we handle retries ourselves for finer control
        )

    # ── Public API ────────────────────────────────────────────

    async def generate(
        self,
        system_prompt: str,
        user_message: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
        messages: list[dict[str, str]] | None = None,
    ) -> str:
        """Send a chat-completion request and return the assistant reply.

        Parameters
        ----------
        system_prompt:
            The system message that sets the assistant behaviour.
        user_message:
            The user's message / query.
        temperature:
            Override default temperature for this call.
        max_tokens:
            Override default max_tokens for this call.
        model:
            Override default model for this call.
        messages:
            Optional list of prior chat messages (role/content dicts) to
            include between the system prompt and the new user message,
            enabling multi-turn context.

        Returns
        -------
        str
            The text content of the assistant's reply.

        Raises
        ------
        LLMError
            When the request fails after all retry attempts or on a
            non-retryable error.
        """
        resolved_model = model or self._model
        resolved_temperature = temperature if temperature is not None else self._temperature
        resolved_max_tokens = max_tokens if max_tokens is not None else self._max_tokens

        chat_messages = self._build_messages(system_prompt, user_message, messages)

        return await self._call_with_retry(
            chat_messages=chat_messages,
            model=resolved_model,
            temperature=resolved_temperature,
            max_tokens=resolved_max_tokens,
        )

    async def close(self) -> None:
        """Gracefully close the underlying HTTP client."""
        await self._client.close()

    # ── Internals ─────────────────────────────────────────────

    @staticmethod
    def _build_messages(
        system_prompt: str,
        user_message: str,
        history: list[dict[str, str]] | None = None,
    ) -> list[dict[str, str]]:
        """Assemble the messages list for the chat-completion API."""
        msgs: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        if history:
            msgs.extend(history)
        msgs.append({"role": "user", "content": user_message})
        return msgs

    async def _call_with_retry(
        self,
        *,
        chat_messages: list[dict[str, str]],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Execute the API call with exponential-backoff retry on transient errors."""
        last_error: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                response = await self._client.chat.completions.create(
                    model=model,
                    messages=chat_messages,  # type: ignore[arg-type]
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                content = self._extract_content(response)
                logger.debug(
                    "LLM response received",
                    extra={
                        "model": model,
                        "attempt": attempt,
                        "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                        "completion_tokens": getattr(response.usage, "completion_tokens", None),
                    },
                )
                return content

            except _RETRYABLE_ERRORS as exc:
                last_error = exc
                delay = _DEFAULT_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "LLM transient error, retrying",
                    extra={
                        "error": str(exc),
                        "attempt": attempt,
                        "max_retries": self._max_retries,
                        "retry_delay": delay,
                    },
                )
                if attempt < self._max_retries:
                    await asyncio.sleep(delay)

            except openai.AuthenticationError as exc:
                raise LLMError(f"Authentication failed: {exc}") from exc

            except openai.BadRequestError as exc:
                raise LLMError(f"Bad request: {exc}") from exc

            except openai.APIError as exc:
                raise LLMError(f"OpenAI API error: {exc}") from exc

            except Exception as exc:
                raise LLMError(f"Unexpected error calling LLM: {exc}") from exc

        # All retries exhausted
        raise LLMError(
            f"LLM request failed after {self._max_retries} attempts: {last_error}"
        )

    @staticmethod
    def _extract_content(response: Any) -> str:
        """Extract text content from the API response."""
        if not response.choices:
            raise LLMError("LLM returned empty choices")
        content = response.choices[0].message.content
        if content is None:
            raise LLMError("LLM returned None content")
        return content.strip()

    # ── Context manager support ───────────────────────────────

    async def __aenter__(self) -> LLMClient:
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()
