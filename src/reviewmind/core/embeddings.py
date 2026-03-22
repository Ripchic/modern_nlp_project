"""reviewmind/core/embeddings.py — Embedding service (text-embedding-3-small).

Provides async embedding generation via OpenAI-compatible APIs.
Supports both OpenAI API and GitHub Models API via configurable base_url.
Implements retry with exponential backoff and batch processing.
"""

from __future__ import annotations

import asyncio
import logging
import math
from typing import Any

import openai
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# Defaults
_DEFAULT_MODEL = "text-embedding-3-small"
_DEFAULT_DIMENSION = 1536
_DEFAULT_BATCH_SIZE = 100
_DEFAULT_MAX_RETRIES = 3
_DEFAULT_RETRY_BASE_DELAY = 1.0  # seconds
_DEFAULT_TIMEOUT = 60.0  # seconds

# Transient error types that warrant a retry
_RETRYABLE_ERRORS = (
    openai.RateLimitError,
    openai.APITimeoutError,
    openai.APIConnectionError,
    openai.InternalServerError,
)


class EmbeddingError(Exception):
    """Base exception for embedding service errors."""


class EmbeddingService:
    """Async embedding service using OpenAI-compatible APIs.

    Parameters
    ----------
    api_key:
        OpenAI / GitHub Models API key.  If *None*, read from
        ``settings.openai_api_key`` at instantiation time.
    base_url:
        API base URL.  If *None*, read from ``settings.openai_base_url``.
    model:
        Embedding model name.  Defaults to ``settings.openai_embedding_model``
        or ``"text-embedding-3-small"``.
    dimension:
        Expected embedding vector dimension.  Used for validation.
    batch_size:
        Maximum number of texts per single API request.
    max_retries:
        Maximum retry attempts for transient errors.
    timeout:
        Request timeout in seconds.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        dimension: int = _DEFAULT_DIMENSION,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        max_retries: int = _DEFAULT_MAX_RETRIES,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        from reviewmind.config import settings

        self._api_key = api_key or settings.openai_api_key
        self._base_url = base_url or settings.openai_base_url
        self._model = model or settings.openai_embedding_model or _DEFAULT_MODEL
        self._dimension = dimension
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._timeout = timeout

        self._client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=self._timeout,
            max_retries=0,  # we handle retries ourselves
        )

    # ── Properties ────────────────────────────────────────────

    @property
    def model(self) -> str:
        """Return the embedding model name."""
        return self._model

    @property
    def dimension(self) -> int:
        """Return the expected embedding dimension."""
        return self._dimension

    @property
    def batch_size(self) -> int:
        """Return the maximum batch size."""
        return self._batch_size

    # ── Public API ────────────────────────────────────────────

    async def embed_text(self, text: str) -> list[float]:
        """Embed a single text string.

        Parameters
        ----------
        text:
            The text to embed.  Empty/whitespace-only strings are
            replaced with a single space to avoid API errors.

        Returns
        -------
        list[float]
            Embedding vector of length :attr:`dimension`.

        Raises
        ------
        EmbeddingError
            When the API call fails after all retry attempts.
        """
        sanitized = self._sanitize_text(text)
        vectors = await self._embed_with_retry([sanitized])
        return vectors[0]

    async def embed_batch(
        self,
        texts: list[str],
        *,
        batch_size: int | None = None,
    ) -> list[list[float]]:
        """Embed a batch of texts, automatically splitting into sub-batches.

        Parameters
        ----------
        texts:
            List of texts to embed.
        batch_size:
            Override the default batch size for this call.

        Returns
        -------
        list[list[float]]
            List of embedding vectors, one per input text,
            preserving input order.

        Raises
        ------
        EmbeddingError
            When any API call fails after all retry attempts.
        """
        if not texts:
            return []

        effective_batch_size = batch_size or self._batch_size
        sanitized = [self._sanitize_text(t) for t in texts]
        all_vectors: list[list[float]] = []

        num_batches = math.ceil(len(sanitized) / effective_batch_size)
        for batch_idx in range(num_batches):
            start = batch_idx * effective_batch_size
            end = start + effective_batch_size
            batch = sanitized[start:end]

            logger.debug(
                "Embedding batch %d/%d (%d texts)",
                batch_idx + 1,
                num_batches,
                len(batch),
            )
            vectors = await self._embed_with_retry(batch)
            all_vectors.extend(vectors)

        return all_vectors

    async def close(self) -> None:
        """Gracefully close the underlying HTTP client."""
        await self._client.close()

    # ── Context manager support ───────────────────────────────

    async def __aenter__(self) -> EmbeddingService:
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    # ── Internals ─────────────────────────────────────────────

    @staticmethod
    def _sanitize_text(text: str) -> str:
        """Sanitize text for the embedding API.

        Strips whitespace.  Returns a single space for empty/whitespace-only
        input to avoid API errors with empty strings.
        """
        cleaned = text.strip() if text else ""
        return cleaned if cleaned else " "

    async def _embed_with_retry(self, texts: list[str]) -> list[list[float]]:
        """Execute the embedding API call with exponential-backoff retry."""
        import time as _time

        _embed_start = _time.perf_counter()
        last_error: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                response = await self._client.embeddings.create(
                    model=self._model,
                    input=texts,
                )

                # Extract vectors, sorted by index to preserve input order
                sorted_data = sorted(response.data, key=lambda d: d.index)
                vectors = [item.embedding for item in sorted_data]

                # Validate dimension
                for i, vec in enumerate(vectors):
                    if len(vec) != self._dimension:
                        raise EmbeddingError(f"Expected dimension {self._dimension}, got {len(vec)} for text index {i}")

                logger.debug(
                    "Embedding response received",
                    extra={
                        "model": self._model,
                        "attempt": attempt,
                        "texts_count": len(texts),
                        "prompt_tokens": getattr(response.usage, "prompt_tokens", None),
                        "total_tokens": getattr(response.usage, "total_tokens", None),
                    },
                )
                try:
                    from reviewmind.metrics import EMBEDDING_DURATION_SECONDS

                    EMBEDDING_DURATION_SECONDS.observe(_time.perf_counter() - _embed_start)
                except Exception:
                    pass
                return vectors

            except _RETRYABLE_ERRORS as exc:
                last_error = exc
                delay = _DEFAULT_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "Embedding transient error, retrying",
                    extra={
                        "error": str(exc),
                        "attempt": attempt,
                        "max_retries": self._max_retries,
                        "retry_delay": delay,
                    },
                )
                if attempt < self._max_retries:
                    await asyncio.sleep(delay)

            except EmbeddingError:
                raise

            except openai.AuthenticationError as exc:
                raise EmbeddingError(f"Authentication failed: {exc}") from exc

            except openai.BadRequestError as exc:
                raise EmbeddingError(f"Bad request: {exc}") from exc

            except openai.APIError as exc:
                raise EmbeddingError(f"OpenAI API error: {exc}") from exc

            except Exception as exc:
                raise EmbeddingError(f"Unexpected error generating embeddings: {exc}") from exc

        # All retries exhausted
        raise EmbeddingError(f"Embedding request failed after {self._max_retries} attempts: {last_error}")
