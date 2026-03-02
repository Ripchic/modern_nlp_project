"""Unit tests for reviewmind.core.embeddings — Embedding service."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest

from reviewmind.core.embeddings import (
    _DEFAULT_BATCH_SIZE,
    _DEFAULT_DIMENSION,
    _DEFAULT_MAX_RETRIES,
    _DEFAULT_MODEL,
    _DEFAULT_RETRY_BASE_DELAY,
    _DEFAULT_TIMEOUT,
    EmbeddingError,
    EmbeddingService,
)

# ── Helpers ──────────────────────────────────────────────────


def _make_embedding(values: list[float], index: int = 0):
    """Build a minimal embedding data item."""
    return SimpleNamespace(embedding=values, index=index)


def _make_response(embeddings: list[list[float]], prompt_tokens: int = 10, total_tokens: int = 10):
    """Build a minimal embedding API response."""
    data = [_make_embedding(vec, idx) for idx, vec in enumerate(embeddings)]
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, total_tokens=total_tokens)
    return SimpleNamespace(data=data, usage=usage)


def _make_vector(dim: int = 1536, base: float = 0.1) -> list[float]:
    """Create a dummy vector of a given dimension."""
    return [base * (i + 1) / dim for i in range(dim)]


def _create_service(**kwargs) -> EmbeddingService:
    """Create an EmbeddingService with test defaults, bypassing settings."""
    defaults = {
        "api_key": "test-key",
        "base_url": "https://test.example.com/v1",
        "model": "text-embedding-3-small",
        "dimension": 1536,
    }
    defaults.update(kwargs)
    return EmbeddingService(**defaults)


# ── Constants tests ──────────────────────────────────────────


class TestConstants:
    """Test module-level constants."""

    def test_default_model(self):
        assert _DEFAULT_MODEL == "text-embedding-3-small"

    def test_default_dimension(self):
        assert _DEFAULT_DIMENSION == 1536

    def test_default_batch_size(self):
        assert _DEFAULT_BATCH_SIZE == 100

    def test_default_max_retries(self):
        assert _DEFAULT_MAX_RETRIES == 3

    def test_default_retry_base_delay(self):
        assert _DEFAULT_RETRY_BASE_DELAY == 1.0

    def test_default_timeout(self):
        assert _DEFAULT_TIMEOUT == 60.0


# ── Construction tests ───────────────────────────────────────


class TestEmbeddingServiceInit:
    """Test EmbeddingService construction."""

    def test_creates_with_explicit_params(self):
        svc = _create_service()
        assert svc._api_key == "test-key"
        assert svc._base_url == "https://test.example.com/v1"
        assert svc._model == "text-embedding-3-small"
        assert svc._dimension == 1536

    def test_default_batch_size(self):
        svc = _create_service()
        assert svc._batch_size == _DEFAULT_BATCH_SIZE

    def test_custom_batch_size(self):
        svc = _create_service(batch_size=50)
        assert svc._batch_size == 50

    def test_default_max_retries(self):
        svc = _create_service()
        assert svc._max_retries == _DEFAULT_MAX_RETRIES

    def test_custom_max_retries(self):
        svc = _create_service(max_retries=5)
        assert svc._max_retries == 5

    def test_default_timeout(self):
        svc = _create_service()
        assert svc._timeout == _DEFAULT_TIMEOUT

    def test_custom_timeout(self):
        svc = _create_service(timeout=120.0)
        assert svc._timeout == 120.0

    def test_custom_dimension(self):
        svc = _create_service(dimension=768)
        assert svc._dimension == 768

    def test_creates_async_openai_client(self):
        svc = _create_service()
        assert svc._client is not None

    @patch("reviewmind.core.embeddings.AsyncOpenAI")
    def test_client_max_retries_zero(self, mock_cls):
        """Own retry logic — SDK retries must be disabled."""
        _create_service()
        _, kwargs = mock_cls.call_args
        assert kwargs["max_retries"] == 0

    def test_fallback_to_settings(self):
        """When no params — reads from settings."""
        mock_settings = MagicMock(
            openai_api_key="settings-key",
            openai_base_url="https://settings.example.com/v1",
            openai_embedding_model="custom-model",
        )
        with patch("reviewmind.config.settings", mock_settings):
            svc = EmbeddingService()
            assert svc._api_key == "settings-key"
            assert svc._base_url == "https://settings.example.com/v1"
            assert svc._model == "custom-model"


# ── Properties tests ─────────────────────────────────────────


class TestProperties:
    """Test EmbeddingService properties."""

    def test_model_property(self):
        svc = _create_service(model="my-model")
        assert svc.model == "my-model"

    def test_dimension_property(self):
        svc = _create_service(dimension=768)
        assert svc.dimension == 768

    def test_batch_size_property(self):
        svc = _create_service(batch_size=50)
        assert svc.batch_size == 50


# ── Sanitize text tests ─────────────────────────────────────


class TestSanitizeText:
    """Test the _sanitize_text helper."""

    def test_normal_text(self):
        assert EmbeddingService._sanitize_text("hello world") == "hello world"

    def test_strips_whitespace(self):
        assert EmbeddingService._sanitize_text("  hello  ") == "hello"

    def test_empty_string(self):
        assert EmbeddingService._sanitize_text("") == " "

    def test_whitespace_only(self):
        assert EmbeddingService._sanitize_text("   ") == " "

    def test_none_equivalent(self):
        """Handles falsy-like empty string."""
        assert EmbeddingService._sanitize_text("") == " "

    def test_newlines_stripped(self):
        assert EmbeddingService._sanitize_text("\n\thello\n") == "hello"


# ── embed_text tests ─────────────────────────────────────────


class TestEmbedText:
    """Test embed_text method."""

    @pytest.mark.asyncio
    async def test_embed_text_returns_vector(self):
        svc = _create_service()
        vec = _make_vector()
        svc._client.embeddings = MagicMock()
        svc._client.embeddings.create = AsyncMock(return_value=_make_response([vec]))

        result = await svc.embed_text("test query")
        assert result == vec
        assert len(result) == 1536

    @pytest.mark.asyncio
    async def test_embed_text_calls_api_with_correct_params(self):
        svc = _create_service(model="text-embedding-3-small")
        vec = _make_vector()
        svc._client.embeddings = MagicMock()
        svc._client.embeddings.create = AsyncMock(return_value=_make_response([vec]))

        await svc.embed_text("test query")

        svc._client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input=["test query"],
        )

    @pytest.mark.asyncio
    async def test_embed_text_empty_string(self):
        """Empty strings are sanitized to a single space."""
        svc = _create_service()
        vec = _make_vector()
        svc._client.embeddings = MagicMock()
        svc._client.embeddings.create = AsyncMock(return_value=_make_response([vec]))

        result = await svc.embed_text("")
        svc._client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input=[" "],
        )
        assert len(result) == 1536

    @pytest.mark.asyncio
    async def test_embed_text_whitespace_only(self):
        """Whitespace-only strings are sanitized to a single space."""
        svc = _create_service()
        vec = _make_vector()
        svc._client.embeddings = MagicMock()
        svc._client.embeddings.create = AsyncMock(return_value=_make_response([vec]))

        await svc.embed_text("   \t\n  ")
        svc._client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input=[" "],
        )

    @pytest.mark.asyncio
    async def test_embed_text_strips_whitespace(self):
        svc = _create_service()
        vec = _make_vector()
        svc._client.embeddings = MagicMock()
        svc._client.embeddings.create = AsyncMock(return_value=_make_response([vec]))

        await svc.embed_text("  hello world  ")
        svc._client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input=["hello world"],
        )


# ── embed_batch tests ────────────────────────────────────────


class TestEmbedBatch:
    """Test embed_batch method."""

    @pytest.mark.asyncio
    async def test_embed_batch_empty_list(self):
        svc = _create_service()
        result = await svc.embed_batch([])
        assert result == []

    @pytest.mark.asyncio
    async def test_embed_batch_single_text(self):
        svc = _create_service()
        vec = _make_vector()
        svc._client.embeddings = MagicMock()
        svc._client.embeddings.create = AsyncMock(return_value=_make_response([vec]))

        result = await svc.embed_batch(["hello"])
        assert len(result) == 1
        assert result[0] == vec

    @pytest.mark.asyncio
    async def test_embed_batch_multiple_texts(self):
        svc = _create_service()
        vecs = [_make_vector(base=0.1), _make_vector(base=0.2), _make_vector(base=0.3)]
        svc._client.embeddings = MagicMock()
        svc._client.embeddings.create = AsyncMock(return_value=_make_response(vecs))

        result = await svc.embed_batch(["a", "b", "c"])
        assert len(result) == 3
        assert result[0] == vecs[0]
        assert result[1] == vecs[1]
        assert result[2] == vecs[2]

    @pytest.mark.asyncio
    async def test_embed_batch_splits_into_sub_batches(self):
        """When texts exceed batch_size, split into sub-batches."""
        svc = _create_service(batch_size=2)

        vec1 = _make_vector(base=0.1)
        vec2 = _make_vector(base=0.2)
        vec3 = _make_vector(base=0.3)

        svc._client.embeddings = MagicMock()
        svc._client.embeddings.create = AsyncMock(
            side_effect=[
                _make_response([vec1, vec2]),
                _make_response([vec3]),
            ]
        )

        result = await svc.embed_batch(["a", "b", "c"])
        assert len(result) == 3
        assert result[0] == vec1
        assert result[2] == vec3
        assert svc._client.embeddings.create.call_count == 2

    @pytest.mark.asyncio
    async def test_embed_batch_custom_batch_size_override(self):
        """batch_size param overrides instance default."""
        svc = _create_service(batch_size=100)  # instance default is 100

        vec1 = _make_vector(base=0.1)
        vec2 = _make_vector(base=0.2)
        vec3 = _make_vector(base=0.3)

        svc._client.embeddings = MagicMock()
        svc._client.embeddings.create = AsyncMock(
            side_effect=[
                _make_response([vec1]),
                _make_response([vec2]),
                _make_response([vec3]),
            ]
        )

        result = await svc.embed_batch(["a", "b", "c"], batch_size=1)
        assert len(result) == 3
        assert svc._client.embeddings.create.call_count == 3

    @pytest.mark.asyncio
    async def test_embed_batch_preserves_order(self):
        """Vectors must match input text order."""
        svc = _create_service()
        vecs = [_make_vector(base=float(i)) for i in range(5)]
        svc._client.embeddings = MagicMock()
        svc._client.embeddings.create = AsyncMock(return_value=_make_response(vecs))

        result = await svc.embed_batch(["t0", "t1", "t2", "t3", "t4"])
        assert len(result) == 5
        for i in range(5):
            assert result[i] == vecs[i]

    @pytest.mark.asyncio
    async def test_embed_batch_sanitizes_texts(self):
        """Empty/whitespace texts are sanitized."""
        svc = _create_service()
        vecs = [_make_vector(base=0.1), _make_vector(base=0.2)]
        svc._client.embeddings = MagicMock()
        svc._client.embeddings.create = AsyncMock(return_value=_make_response(vecs))

        await svc.embed_batch(["", "  hello  "])
        svc._client.embeddings.create.assert_called_once_with(
            model="text-embedding-3-small",
            input=[" ", "hello"],
        )


# ── Order preservation with shuffled indices ─────────────────


class TestOrderPreservation:
    """Test that vectors are reordered by index."""

    @pytest.mark.asyncio
    async def test_shuffled_response_indices(self):
        """API may return embeddings out of order — we sort by index."""
        svc = _create_service()
        vec0 = _make_vector(base=0.0)
        vec1 = _make_vector(base=1.0)
        vec2 = _make_vector(base=2.0)

        # Return in reverse order
        data = [
            _make_embedding(vec2, index=2),
            _make_embedding(vec0, index=0),
            _make_embedding(vec1, index=1),
        ]
        usage = SimpleNamespace(prompt_tokens=10, total_tokens=10)
        response = SimpleNamespace(data=data, usage=usage)

        svc._client.embeddings = MagicMock()
        svc._client.embeddings.create = AsyncMock(return_value=response)

        result = await svc.embed_batch(["a", "b", "c"])
        assert result[0] == vec0
        assert result[1] == vec1
        assert result[2] == vec2


# ── Dimension validation tests ───────────────────────────────


class TestDimensionValidation:
    """Test dimension validation."""

    @pytest.mark.asyncio
    async def test_wrong_dimension_raises_error(self):
        svc = _create_service(dimension=1536)
        wrong_vec = [0.1] * 768  # wrong dimension
        svc._client.embeddings = MagicMock()
        svc._client.embeddings.create = AsyncMock(return_value=_make_response([wrong_vec]))

        with pytest.raises(EmbeddingError, match="Expected dimension 1536.*got 768"):
            await svc.embed_text("test")

    @pytest.mark.asyncio
    async def test_correct_dimension_passes(self):
        svc = _create_service(dimension=1536)
        vec = _make_vector(dim=1536)
        svc._client.embeddings = MagicMock()
        svc._client.embeddings.create = AsyncMock(return_value=_make_response([vec]))

        result = await svc.embed_text("test")
        assert len(result) == 1536

    @pytest.mark.asyncio
    async def test_custom_dimension_768(self):
        svc = _create_service(dimension=768)
        vec = _make_vector(dim=768)
        svc._client.embeddings = MagicMock()
        svc._client.embeddings.create = AsyncMock(return_value=_make_response([vec]))

        result = await svc.embed_text("test")
        assert len(result) == 768


# ── Error handling tests ─────────────────────────────────────


class TestErrorHandling:
    """Test error handling and retries."""

    @pytest.mark.asyncio
    async def test_authentication_error(self):
        svc = _create_service()
        svc._client.embeddings = MagicMock()
        svc._client.embeddings.create = AsyncMock(
            side_effect=openai.AuthenticationError(
                message="Invalid API key",
                response=MagicMock(status_code=401),
                body=None,
            )
        )

        with pytest.raises(EmbeddingError, match="Authentication failed"):
            await svc.embed_text("test")

    @pytest.mark.asyncio
    async def test_bad_request_error(self):
        svc = _create_service()
        svc._client.embeddings = MagicMock()
        svc._client.embeddings.create = AsyncMock(
            side_effect=openai.BadRequestError(
                message="Bad request",
                response=MagicMock(status_code=400),
                body=None,
            )
        )

        with pytest.raises(EmbeddingError, match="Bad request"):
            await svc.embed_text("test")

    @pytest.mark.asyncio
    async def test_generic_api_error(self):
        svc = _create_service()
        svc._client.embeddings = MagicMock()
        svc._client.embeddings.create = AsyncMock(
            side_effect=openai.APIError(
                message="Server error",
                request=MagicMock(),
                body=None,
            )
        )

        with pytest.raises(EmbeddingError, match="OpenAI API error"):
            await svc.embed_text("test")

    @pytest.mark.asyncio
    async def test_unexpected_error(self):
        svc = _create_service()
        svc._client.embeddings = MagicMock()
        svc._client.embeddings.create = AsyncMock(side_effect=RuntimeError("boom"))

        with pytest.raises(EmbeddingError, match="Unexpected error"):
            await svc.embed_text("test")

    @pytest.mark.asyncio
    async def test_auth_error_not_retried(self):
        """Non-retryable errors should raise immediately (no retry)."""
        svc = _create_service(max_retries=3)
        svc._client.embeddings = MagicMock()
        svc._client.embeddings.create = AsyncMock(
            side_effect=openai.AuthenticationError(
                message="Invalid API key",
                response=MagicMock(status_code=401),
                body=None,
            )
        )

        with pytest.raises(EmbeddingError):
            await svc.embed_text("test")

        # Called only once — no retry
        assert svc._client.embeddings.create.call_count == 1

    @pytest.mark.asyncio
    async def test_bad_request_not_retried(self):
        svc = _create_service(max_retries=3)
        svc._client.embeddings = MagicMock()
        svc._client.embeddings.create = AsyncMock(
            side_effect=openai.BadRequestError(
                message="Bad",
                response=MagicMock(status_code=400),
                body=None,
            )
        )

        with pytest.raises(EmbeddingError):
            await svc.embed_text("test")

        assert svc._client.embeddings.create.call_count == 1


# ── Retry tests ──────────────────────────────────────────────


class TestRetry:
    """Test retry logic for transient errors."""

    @pytest.mark.asyncio
    async def test_retries_on_rate_limit(self):
        svc = _create_service(max_retries=3)
        vec = _make_vector()

        svc._client.embeddings = MagicMock()
        svc._client.embeddings.create = AsyncMock(
            side_effect=[
                openai.RateLimitError(
                    message="Rate limit",
                    response=MagicMock(status_code=429, headers={}),
                    body=None,
                ),
                _make_response([vec]),
            ]
        )

        with patch("reviewmind.core.embeddings.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await svc.embed_text("test")

        assert result == vec
        assert svc._client.embeddings.create.call_count == 2
        mock_sleep.assert_called_once_with(1.0)  # first retry delay

    @pytest.mark.asyncio
    async def test_retries_on_timeout(self):
        svc = _create_service(max_retries=3)
        vec = _make_vector()

        svc._client.embeddings = MagicMock()
        svc._client.embeddings.create = AsyncMock(
            side_effect=[
                openai.APITimeoutError(request=MagicMock()),
                _make_response([vec]),
            ]
        )

        with patch("reviewmind.core.embeddings.asyncio.sleep", new_callable=AsyncMock):
            result = await svc.embed_text("test")

        assert result == vec

    @pytest.mark.asyncio
    async def test_retries_on_connection_error(self):
        svc = _create_service(max_retries=3)
        vec = _make_vector()

        svc._client.embeddings = MagicMock()
        svc._client.embeddings.create = AsyncMock(
            side_effect=[
                openai.APIConnectionError(request=MagicMock()),
                _make_response([vec]),
            ]
        )

        with patch("reviewmind.core.embeddings.asyncio.sleep", new_callable=AsyncMock):
            result = await svc.embed_text("test")

        assert result == vec

    @pytest.mark.asyncio
    async def test_retries_on_internal_server_error(self):
        svc = _create_service(max_retries=3)
        vec = _make_vector()

        svc._client.embeddings = MagicMock()
        svc._client.embeddings.create = AsyncMock(
            side_effect=[
                openai.InternalServerError(
                    message="ISE",
                    response=MagicMock(status_code=500),
                    body=None,
                ),
                _make_response([vec]),
            ]
        )

        with patch("reviewmind.core.embeddings.asyncio.sleep", new_callable=AsyncMock):
            result = await svc.embed_text("test")

        assert result == vec

    @pytest.mark.asyncio
    async def test_exhausts_all_retries(self):
        svc = _create_service(max_retries=3)

        svc._client.embeddings = MagicMock()
        svc._client.embeddings.create = AsyncMock(
            side_effect=openai.RateLimitError(
                message="Rate limit",
                response=MagicMock(status_code=429, headers={}),
                body=None,
            )
        )

        with patch("reviewmind.core.embeddings.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(EmbeddingError, match="failed after 3 attempts"):
                await svc.embed_text("test")

        assert svc._client.embeddings.create.call_count == 3

    @pytest.mark.asyncio
    async def test_exponential_backoff_delays(self):
        """Retry delays double: 1s, 2s."""
        svc = _create_service(max_retries=3)
        vec = _make_vector()

        svc._client.embeddings = MagicMock()
        svc._client.embeddings.create = AsyncMock(
            side_effect=[
                openai.RateLimitError(
                    message="Rate limit",
                    response=MagicMock(status_code=429, headers={}),
                    body=None,
                ),
                openai.RateLimitError(
                    message="Rate limit",
                    response=MagicMock(status_code=429, headers={}),
                    body=None,
                ),
                _make_response([vec]),
            ]
        )

        with patch("reviewmind.core.embeddings.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            result = await svc.embed_text("test")

        assert result == vec
        # Delays: attempt 1 → 1s, attempt 2 → 2s
        assert mock_sleep.call_count == 2
        mock_sleep.assert_any_call(1.0)
        mock_sleep.assert_any_call(2.0)

    @pytest.mark.asyncio
    async def test_no_sleep_after_last_attempt(self):
        """Sleep is NOT called after the last failed attempt."""
        svc = _create_service(max_retries=2)

        svc._client.embeddings = MagicMock()
        svc._client.embeddings.create = AsyncMock(
            side_effect=openai.RateLimitError(
                message="Rate limit",
                response=MagicMock(status_code=429, headers={}),
                body=None,
            )
        )

        with patch("reviewmind.core.embeddings.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            with pytest.raises(EmbeddingError):
                await svc.embed_text("test")

        # Only 1 sleep: after attempt 1, not after attempt 2 (last attempt)
        assert mock_sleep.call_count == 1


# ── Context manager tests ────────────────────────────────────


class TestContextManager:
    """Test async context manager support."""

    @pytest.mark.asyncio
    async def test_aenter_returns_self(self):
        svc = _create_service()
        async with svc as s:
            assert s is svc

    @pytest.mark.asyncio
    async def test_aexit_calls_close(self):
        svc = _create_service()
        svc.close = AsyncMock()

        async with svc:
            pass

        svc.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_calls_client_close(self):
        svc = _create_service()
        svc._client.close = AsyncMock()

        await svc.close()
        svc._client.close.assert_called_once()


# ── EmbeddingError tests ─────────────────────────────────────


class TestEmbeddingError:
    """Test the EmbeddingError exception class."""

    def test_is_exception(self):
        assert issubclass(EmbeddingError, Exception)

    def test_message(self):
        err = EmbeddingError("test error")
        assert str(err) == "test error"

    def test_can_be_raised(self):
        with pytest.raises(EmbeddingError, match="boom"):
            raise EmbeddingError("boom")


# ── Integration-style tests ──────────────────────────────────


class TestIntegration:
    """Higher-level integration-style tests."""

    @pytest.mark.asyncio
    async def test_embed_and_compare_same_text(self):
        """Two calls with same text produce same vector."""
        svc = _create_service()
        vec = _make_vector()
        svc._client.embeddings = MagicMock()
        svc._client.embeddings.create = AsyncMock(return_value=_make_response([vec]))

        result1 = await svc.embed_text("test")
        result2 = await svc.embed_text("test")
        assert result1 == result2

    @pytest.mark.asyncio
    async def test_large_batch_processing(self):
        """Test batch of 250 texts with batch_size=100 (3 API calls)."""
        svc = _create_service(batch_size=100)
        texts = [f"text {i}" for i in range(250)]

        batch1_vecs = [_make_vector(base=float(i)) for i in range(100)]
        batch2_vecs = [_make_vector(base=float(i + 100)) for i in range(100)]
        batch3_vecs = [_make_vector(base=float(i + 200)) for i in range(50)]

        svc._client.embeddings = MagicMock()
        svc._client.embeddings.create = AsyncMock(
            side_effect=[
                _make_response(batch1_vecs),
                _make_response(batch2_vecs),
                _make_response(batch3_vecs),
            ]
        )

        result = await svc.embed_batch(texts)
        assert len(result) == 250
        assert svc._client.embeddings.create.call_count == 3

    @pytest.mark.asyncio
    async def test_embed_batch_with_context_manager(self):
        """Using embed_batch within a context manager."""
        svc = _create_service()
        svc._client.close = AsyncMock()
        vec = _make_vector()
        svc._client.embeddings = MagicMock()
        svc._client.embeddings.create = AsyncMock(return_value=_make_response([vec]))

        async with svc:
            result = await svc.embed_batch(["hello"])
            assert len(result) == 1

        svc._client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_retry_then_success_in_batch(self):
        """Retry during batch processing still returns correct results."""
        svc = _create_service(max_retries=3, batch_size=2)
        vec1 = _make_vector(base=0.1)
        vec2 = _make_vector(base=0.2)
        vec3 = _make_vector(base=0.3)

        svc._client.embeddings = MagicMock()
        svc._client.embeddings.create = AsyncMock(
            side_effect=[
                # First batch: first attempt fails, then succeeds
                openai.RateLimitError(
                    message="Rate limit",
                    response=MagicMock(status_code=429, headers={}),
                    body=None,
                ),
                _make_response([vec1, vec2]),
                # Second batch: succeeds immediately
                _make_response([vec3]),
            ]
        )

        with patch("reviewmind.core.embeddings.asyncio.sleep", new_callable=AsyncMock):
            result = await svc.embed_batch(["a", "b", "c"])

        assert len(result) == 3
        assert result[0] == vec1
        assert result[2] == vec3
