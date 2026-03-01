"""Unit tests for reviewmind.core.llm — OpenAI client wrapper."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import openai
import pytest

from reviewmind.core.llm import LLMClient, LLMError

# ── Helpers ──────────────────────────────────────────────────


def _make_response(content: str = "Hello!", prompt_tokens: int = 10, completion_tokens: int = 5):
    """Build a minimal ChatCompletion-like response object."""
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    usage = SimpleNamespace(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    return SimpleNamespace(choices=[choice], usage=usage)


def _make_empty_response():
    """Response with no choices."""
    return SimpleNamespace(choices=[], usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0))


def _make_none_content_response():
    """Response with None content."""
    message = SimpleNamespace(content=None)
    choice = SimpleNamespace(message=message)
    usage = SimpleNamespace(prompt_tokens=5, completion_tokens=0)
    return SimpleNamespace(choices=[choice], usage=usage)


def _create_client(**kwargs) -> LLMClient:
    """Create an LLMClient with test defaults, bypassing settings."""
    defaults = {
        "api_key": "test-key",
        "base_url": "https://test.example.com/v1",
        "model": "gpt-4o-mini",
        "temperature": 0.3,
        "max_tokens": 1000,
    }
    defaults.update(kwargs)
    return LLMClient(**defaults)


# ── Construction tests ───────────────────────────────────────


class TestLLMClientInit:
    """Test LLMClient construction."""

    def test_creates_with_explicit_params(self):
        client = _create_client()
        assert client._api_key == "test-key"
        assert client._base_url == "https://test.example.com/v1"
        assert client._model == "gpt-4o-mini"
        assert client._temperature == 0.3
        assert client._max_tokens == 1000

    def test_defaults_from_settings(self):
        """When no explicit params, values come from settings."""
        mock_settings = MagicMock()
        mock_settings.openai_api_key = "settings-key"
        mock_settings.openai_base_url = "https://settings.example.com/v1"
        mock_settings.openai_model = "gpt-4o"
        mock_settings.llm_temperature = 0.7
        mock_settings.llm_max_tokens = 2000

        with patch("reviewmind.config.get_settings", return_value=mock_settings):
            # Clear the lru_cache so it picks up our mock
            from reviewmind.config import get_settings

            get_settings.cache_clear()
            client = LLMClient()
            assert client._api_key == "settings-key"
            assert client._base_url == "https://settings.example.com/v1"
            assert client._model == "gpt-4o"
            assert client._temperature == 0.7
            assert client._max_tokens == 2000

    def test_custom_model(self):
        client = _create_client(model="gpt-4o")
        assert client._model == "gpt-4o"

    def test_custom_temperature(self):
        client = _create_client(temperature=0.9)
        assert client._temperature == 0.9

    def test_custom_max_tokens(self):
        client = _create_client(max_tokens=500)
        assert client._max_tokens == 500

    def test_custom_max_retries(self):
        client = _create_client(max_retries=5)
        assert client._max_retries == 5

    def test_custom_timeout(self):
        client = _create_client(timeout=30.0)
        assert client._timeout == 30.0

    def test_different_base_url(self):
        """Switching base_url allows GitHub Models ↔ OpenAI."""
        github_client = _create_client(base_url="https://models.inference.ai.azure.com")
        assert github_client._base_url == "https://models.inference.ai.azure.com"

        openai_client = _create_client(base_url="https://api.openai.com/v1")
        assert openai_client._base_url == "https://api.openai.com/v1"


# ── Message building tests ───────────────────────────────────


class TestBuildMessages:
    """Test _build_messages static method."""

    def test_basic_messages(self):
        msgs = LLMClient._build_messages("You are helpful", "Hello")
        assert len(msgs) == 2
        assert msgs[0] == {"role": "system", "content": "You are helpful"}
        assert msgs[1] == {"role": "user", "content": "Hello"}

    def test_with_history(self):
        history = [
            {"role": "user", "content": "Previous question"},
            {"role": "assistant", "content": "Previous answer"},
        ]
        msgs = LLMClient._build_messages("System", "New question", history)
        assert len(msgs) == 4
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == "Previous question"
        assert msgs[2]["role"] == "assistant"
        assert msgs[3]["role"] == "user"
        assert msgs[3]["content"] == "New question"

    def test_empty_history(self):
        msgs = LLMClient._build_messages("System", "Question", [])
        assert len(msgs) == 2

    def test_none_history(self):
        msgs = LLMClient._build_messages("System", "Question", None)
        assert len(msgs) == 2


# ── Generate tests ───────────────────────────────────────────


class TestGenerate:
    """Test generate() method."""

    @pytest.mark.asyncio
    async def test_successful_generate(self):
        client = _create_client()
        mock_response = _make_response("Test response")
        client._client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await client.generate("System prompt", "User message")
        assert result == "Test response"

    @pytest.mark.asyncio
    async def test_generate_strips_whitespace(self):
        client = _create_client()
        mock_response = _make_response("  Padded response  ")
        client._client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await client.generate("System", "User")
        assert result == "Padded response"

    @pytest.mark.asyncio
    async def test_generate_with_custom_temperature(self):
        client = _create_client()
        mock_response = _make_response("Response")
        client._client.chat.completions.create = AsyncMock(return_value=mock_response)

        await client.generate("System", "User", temperature=0.9)
        call_kwargs = client._client.chat.completions.create.call_args.kwargs
        assert call_kwargs["temperature"] == 0.9

    @pytest.mark.asyncio
    async def test_generate_with_custom_max_tokens(self):
        client = _create_client()
        mock_response = _make_response("Response")
        client._client.chat.completions.create = AsyncMock(return_value=mock_response)

        await client.generate("System", "User", max_tokens=500)
        call_kwargs = client._client.chat.completions.create.call_args.kwargs
        assert call_kwargs["max_tokens"] == 500

    @pytest.mark.asyncio
    async def test_generate_with_custom_model(self):
        client = _create_client()
        mock_response = _make_response("Response")
        client._client.chat.completions.create = AsyncMock(return_value=mock_response)

        await client.generate("System", "User", model="gpt-4o")
        call_kwargs = client._client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o"

    @pytest.mark.asyncio
    async def test_generate_with_history(self):
        client = _create_client()
        mock_response = _make_response("Contextual response")
        client._client.chat.completions.create = AsyncMock(return_value=mock_response)

        history = [
            {"role": "user", "content": "What's the best phone?"},
            {"role": "assistant", "content": "iPhone 16 is popular."},
        ]
        result = await client.generate("System", "Tell me more", messages=history)
        assert result == "Contextual response"

        call_kwargs = client._client.chat.completions.create.call_args.kwargs
        messages = call_kwargs["messages"]
        assert len(messages) == 4  # system + 2 history + user

    @pytest.mark.asyncio
    async def test_generate_uses_default_params(self):
        client = _create_client(model="gpt-4o-mini", temperature=0.3, max_tokens=1000)
        mock_response = _make_response("Response")
        client._client.chat.completions.create = AsyncMock(return_value=mock_response)

        await client.generate("System", "User")
        call_kwargs = client._client.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o-mini"
        assert call_kwargs["temperature"] == 0.3
        assert call_kwargs["max_tokens"] == 1000


# ── Error handling tests ─────────────────────────────────────


class TestErrorHandling:
    """Test error handling and retry logic."""

    @pytest.mark.asyncio
    async def test_rate_limit_error_retries(self):
        """RateLimitError should trigger retry."""
        client = _create_client(max_retries=2)
        mock_response = _make_response("Success after retry")

        mock_create = AsyncMock(
            side_effect=[
                openai.RateLimitError(
                    message="Rate limit exceeded",
                    response=MagicMock(status_code=429),
                    body=None,
                ),
                mock_response,
            ]
        )
        client._client.chat.completions.create = mock_create

        with patch("reviewmind.core.llm.asyncio.sleep", new_callable=AsyncMock):
            result = await client.generate("System", "User")

        assert result == "Success after retry"
        assert mock_create.call_count == 2

    @pytest.mark.asyncio
    async def test_timeout_error_retries(self):
        """APITimeoutError should trigger retry."""
        client = _create_client(max_retries=2)
        mock_response = _make_response("Success")

        mock_create = AsyncMock(
            side_effect=[
                openai.APITimeoutError(request=MagicMock()),
                mock_response,
            ]
        )
        client._client.chat.completions.create = mock_create

        with patch("reviewmind.core.llm.asyncio.sleep", new_callable=AsyncMock):
            result = await client.generate("System", "User")

        assert result == "Success"

    @pytest.mark.asyncio
    async def test_connection_error_retries(self):
        """APIConnectionError should trigger retry."""
        client = _create_client(max_retries=2)
        mock_response = _make_response("Success")

        mock_create = AsyncMock(
            side_effect=[
                openai.APIConnectionError(request=MagicMock()),
                mock_response,
            ]
        )
        client._client.chat.completions.create = mock_create

        with patch("reviewmind.core.llm.asyncio.sleep", new_callable=AsyncMock):
            result = await client.generate("System", "User")

        assert result == "Success"

    @pytest.mark.asyncio
    async def test_internal_server_error_retries(self):
        """InternalServerError should trigger retry."""
        client = _create_client(max_retries=2)
        mock_response = _make_response("Success")

        mock_create = AsyncMock(
            side_effect=[
                openai.InternalServerError(
                    message="Server error",
                    response=MagicMock(status_code=500),
                    body=None,
                ),
                mock_response,
            ]
        )
        client._client.chat.completions.create = mock_create

        with patch("reviewmind.core.llm.asyncio.sleep", new_callable=AsyncMock):
            result = await client.generate("System", "User")

        assert result == "Success"

    @pytest.mark.asyncio
    async def test_all_retries_exhausted(self):
        """After max_retries, LLMError is raised."""
        client = _create_client(max_retries=2)

        mock_create = AsyncMock(
            side_effect=openai.RateLimitError(
                message="Rate limit exceeded",
                response=MagicMock(status_code=429),
                body=None,
            )
        )
        client._client.chat.completions.create = mock_create

        with patch("reviewmind.core.llm.asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(LLMError, match="failed after 2 attempts"):
                await client.generate("System", "User")

        assert mock_create.call_count == 2

    @pytest.mark.asyncio
    async def test_authentication_error_no_retry(self):
        """AuthenticationError should NOT retry — immediately raises LLMError."""
        client = _create_client(max_retries=3)

        mock_create = AsyncMock(
            side_effect=openai.AuthenticationError(
                message="Invalid API key",
                response=MagicMock(status_code=401),
                body=None,
            )
        )
        client._client.chat.completions.create = mock_create

        with pytest.raises(LLMError, match="Authentication failed"):
            await client.generate("System", "User")

        assert mock_create.call_count == 1  # no retries

    @pytest.mark.asyncio
    async def test_bad_request_error_no_retry(self):
        """BadRequestError should NOT retry — immediately raises LLMError."""
        client = _create_client(max_retries=3)

        mock_create = AsyncMock(
            side_effect=openai.BadRequestError(
                message="Invalid model",
                response=MagicMock(status_code=400),
                body=None,
            )
        )
        client._client.chat.completions.create = mock_create

        with pytest.raises(LLMError, match="Bad request"):
            await client.generate("System", "User")

        assert mock_create.call_count == 1

    @pytest.mark.asyncio
    async def test_empty_choices_raises_error(self):
        """Response with no choices should raise LLMError."""
        client = _create_client()
        client._client.chat.completions.create = AsyncMock(return_value=_make_empty_response())

        with pytest.raises(LLMError, match="empty choices"):
            await client.generate("System", "User")

    @pytest.mark.asyncio
    async def test_none_content_raises_error(self):
        """Response with None content should raise LLMError."""
        client = _create_client()
        client._client.chat.completions.create = AsyncMock(return_value=_make_none_content_response())

        with pytest.raises(LLMError, match="None content"):
            await client.generate("System", "User")

    @pytest.mark.asyncio
    async def test_unexpected_exception_raises_llm_error(self):
        """Unexpected exceptions are wrapped in LLMError."""
        client = _create_client()
        client._client.chat.completions.create = AsyncMock(side_effect=RuntimeError("Something broke"))

        with pytest.raises(LLMError, match="Unexpected error"):
            await client.generate("System", "User")


# ── Retry timing tests ──────────────────────────────────────


class TestRetryTiming:
    """Test exponential backoff logic."""

    @pytest.mark.asyncio
    async def test_exponential_backoff_delays(self):
        """Retry delays should follow exponential backoff: 1s, 2s, 4s..."""
        client = _create_client(max_retries=3)

        mock_create = AsyncMock(
            side_effect=openai.RateLimitError(
                message="Rate limit",
                response=MagicMock(status_code=429),
                body=None,
            )
        )
        client._client.chat.completions.create = mock_create

        sleep_delays: list[float] = []
        original_sleep = AsyncMock(side_effect=lambda d: sleep_delays.append(d))

        with patch("reviewmind.core.llm.asyncio.sleep", original_sleep):
            with pytest.raises(LLMError):
                await client.generate("System", "User")

        # 3 retries → 2 sleeps (no sleep after last attempt)
        assert len(sleep_delays) == 2
        assert sleep_delays[0] == 1.0  # 1 * 2^0
        assert sleep_delays[1] == 2.0  # 1 * 2^1


# ── Context manager tests ───────────────────────────────────


class TestContextManager:
    """Test async context manager support."""

    @pytest.mark.asyncio
    async def test_async_context_manager(self):
        client = _create_client()
        client._client.close = AsyncMock()

        async with client as c:
            assert c is client

        client._client.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_method(self):
        client = _create_client()
        client._client.close = AsyncMock()

        await client.close()
        client._client.close.assert_awaited_once()


# ── LLMError tests ───────────────────────────────────────────


class TestLLMError:
    """Test LLMError exception."""

    def test_is_exception(self):
        assert issubclass(LLMError, Exception)

    def test_message(self):
        err = LLMError("test error")
        assert str(err) == "test error"

    def test_can_chain_exceptions(self):
        original = ValueError("original")
        err = LLMError("wrapped")
        err.__cause__ = original
        assert err.__cause__ is original
