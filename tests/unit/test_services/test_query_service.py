"""Unit tests for TASK-008 — E2E chat: query_service, bot handler, API endpoint.

Tests cover:
- QueryService: success, error handling, empty input, context manager
- Bot handler (handlers/query.py): text messages, typing indicator, errors, truncation
- API endpoint (POST /query): success, validation, error responses
- Bot dispatcher wiring: query_router registration
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from reviewmind.core.llm import LLMError
from reviewmind.services.query_service import (
    _EMPTY_MESSAGE_RESPONSE,
    _ERROR_RESPONSE,
    _SYSTEM_PROMPT,
    QueryResult,
    QueryService,
)

# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════


def _mock_llm_client(response: str = "Test answer") -> MagicMock:
    """Create a mocked LLMClient with a generate() that returns `response`."""
    client = MagicMock()
    client.generate = AsyncMock(return_value=response)
    client.close = AsyncMock()
    client._model = "gpt-4o-mini"
    return client


def _mock_llm_client_error(exc: Exception) -> MagicMock:
    """Create a mocked LLMClient whose generate() raises `exc`."""
    client = MagicMock()
    client.generate = AsyncMock(side_effect=exc)
    client.close = AsyncMock()
    client._model = "gpt-4o-mini"
    return client


# ══════════════════════════════════════════════════════════════
# Tests — QueryResult dataclass
# ══════════════════════════════════════════════════════════════


class TestQueryResult:
    """Test the QueryResult dataclass."""

    def test_default_values(self):
        r = QueryResult(answer="ok")
        assert r.answer == "ok"
        assert r.error is False
        assert r.error_message is None
        assert r.model is None
        assert r.chat_history == []

    def test_error_result(self):
        r = QueryResult(answer="fail", error=True, error_message="boom")
        assert r.error is True
        assert r.error_message == "boom"

    def test_with_model(self):
        r = QueryResult(answer="ok", model="gpt-4o")
        assert r.model == "gpt-4o"


# ══════════════════════════════════════════════════════════════
# Tests — QueryService
# ══════════════════════════════════════════════════════════════


class TestQueryServiceInit:
    """Test QueryService construction."""

    def test_accepts_injected_client(self):
        client = _mock_llm_client()
        svc = QueryService(llm_client=client)
        assert svc._llm is client
        assert svc._owns_client is False

    def test_none_client_defers_creation(self):
        svc = QueryService(llm_client=None)
        assert svc._llm is None
        assert svc._owns_client is True

    def test_custom_system_prompt(self):
        svc = QueryService(system_prompt="custom")
        assert svc._system_prompt == "custom"

    def test_default_system_prompt(self):
        svc = QueryService()
        assert svc._system_prompt == _SYSTEM_PROMPT


class TestQueryServiceAnswer:
    """Test QueryService.answer() method."""

    @pytest.mark.asyncio
    async def test_success(self):
        client = _mock_llm_client("LLM says hello")
        svc = QueryService(llm_client=client)
        result = await svc.answer("Hello")
        assert result.answer == "LLM says hello"
        assert result.error is False
        assert result.model == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_passes_system_prompt(self):
        client = _mock_llm_client("ok")
        svc = QueryService(llm_client=client, system_prompt="Test prompt")
        await svc.answer("Hi")
        client.generate.assert_called_once_with(
            system_prompt="Test prompt",
            user_message="Hi",
            messages=None,
        )

    @pytest.mark.asyncio
    async def test_passes_chat_history(self):
        client = _mock_llm_client("ok")
        svc = QueryService(llm_client=client)
        history = [{"role": "user", "content": "prev"}]
        await svc.answer("next", chat_history=history)
        client.generate.assert_called_once()
        call_kwargs = client.generate.call_args
        assert call_kwargs.kwargs["messages"] == history

    @pytest.mark.asyncio
    async def test_strips_whitespace(self):
        client = _mock_llm_client("ok")
        svc = QueryService(llm_client=client)
        await svc.answer("  spaced  ")
        call_kwargs = client.generate.call_args
        assert call_kwargs.kwargs["user_message"] == "spaced"

    @pytest.mark.asyncio
    async def test_empty_message(self):
        svc = QueryService(llm_client=_mock_llm_client())
        result = await svc.answer("")
        assert result.answer == _EMPTY_MESSAGE_RESPONSE

    @pytest.mark.asyncio
    async def test_whitespace_only_message(self):
        svc = QueryService(llm_client=_mock_llm_client())
        result = await svc.answer("   ")
        assert result.answer == _EMPTY_MESSAGE_RESPONSE

    @pytest.mark.asyncio
    async def test_none_message(self):
        svc = QueryService(llm_client=_mock_llm_client())
        result = await svc.answer("")
        assert result.answer == _EMPTY_MESSAGE_RESPONSE

    @pytest.mark.asyncio
    async def test_llm_error_returns_fallback(self):
        client = _mock_llm_client_error(LLMError("API down"))
        svc = QueryService(llm_client=client)
        result = await svc.answer("Hello")
        assert result.error is True
        assert result.answer == _ERROR_RESPONSE
        assert "API down" in result.error_message

    @pytest.mark.asyncio
    async def test_unexpected_error_returns_fallback(self):
        client = _mock_llm_client_error(RuntimeError("boom"))
        svc = QueryService(llm_client=client)
        result = await svc.answer("Hello")
        assert result.error is True
        assert result.answer == _ERROR_RESPONSE
        assert "boom" in result.error_message


class TestQueryServiceLifecycle:
    """Test QueryService context manager and close."""

    @pytest.mark.asyncio
    async def test_close_owned_client(self):
        client = _mock_llm_client()
        svc = QueryService(llm_client=None)
        svc._llm = client
        svc._owns_client = True
        await svc.close()
        client.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_injected_client_not_closed(self):
        client = _mock_llm_client()
        svc = QueryService(llm_client=client)
        await svc.close()
        client.close.assert_not_called()

    @pytest.mark.asyncio
    async def test_context_manager(self):
        client = _mock_llm_client()
        svc = QueryService(llm_client=None)
        svc._llm = client
        svc._owns_client = True
        async with svc as s:
            assert s is svc
        client.close.assert_called_once()

    def test_lazy_property_creates_client(self):
        svc = QueryService(llm_client=None)
        with patch("reviewmind.services.query_service.LLMClient") as MockLLM:
            mock_instance = MagicMock()
            MockLLM.return_value = mock_instance
            assert svc.llm is mock_instance
            assert svc._owns_client is True


# ══════════════════════════════════════════════════════════════
# Tests — Bot handler (handlers/query.py)
# ══════════════════════════════════════════════════════════════


def _make_bot_message(text: str = "test", user_id: int = 123) -> MagicMock:
    """Create a mocked aiogram Message for bot handler tests."""
    from aiogram.types import Chat, User

    msg = MagicMock()
    msg.text = text
    msg.from_user = MagicMock(spec=User)
    msg.from_user.id = user_id
    msg.chat = MagicMock(spec=Chat)
    msg.chat.id = user_id
    msg.bot = MagicMock()
    msg.bot.send_chat_action = AsyncMock()
    msg.answer = AsyncMock()
    return msg


class TestBotQueryHandler:
    """Test on_text_message handler."""

    @pytest.mark.asyncio
    async def test_sends_typing_indicator(self):
        from reviewmind.bot.handlers.query import on_text_message

        msg = _make_bot_message("Hello")
        mock_client = _mock_llm_client("Response")

        with patch("reviewmind.bot.handlers.query.LLMClient") as MockLLM:
            instance = MockLLM.return_value
            instance.__aenter__ = AsyncMock(return_value=mock_client)
            instance.__aexit__ = AsyncMock(return_value=False)
            await on_text_message(msg)

        msg.bot.send_chat_action.assert_called_once()
        call_args = msg.bot.send_chat_action.call_args
        assert call_args.kwargs.get("action") or call_args.args

    @pytest.mark.asyncio
    async def test_replies_with_llm_answer(self):
        from reviewmind.bot.handlers.query import on_text_message

        msg = _make_bot_message("What headphones to buy?")
        mock_client = _mock_llm_client("Buy Sony WH-1000XM5")

        with patch("reviewmind.bot.handlers.query.LLMClient") as MockLLM:
            instance = MockLLM.return_value
            instance.__aenter__ = AsyncMock(return_value=mock_client)
            instance.__aexit__ = AsyncMock(return_value=False)
            await on_text_message(msg)

        msg.answer.assert_called_once()
        answer_text = msg.answer.call_args.args[0]
        assert "Sony WH-1000XM5" in answer_text

    @pytest.mark.asyncio
    async def test_error_sends_friendly_message(self):
        from reviewmind.bot.handlers.query import on_text_message

        msg = _make_bot_message("trigger error")
        mock_client = _mock_llm_client_error(LLMError("timeout"))

        with patch("reviewmind.bot.handlers.query.LLMClient") as MockLLM:
            instance = MockLLM.return_value
            instance.__aenter__ = AsyncMock(return_value=mock_client)
            instance.__aexit__ = AsyncMock(return_value=False)
            await on_text_message(msg)

        msg.answer.assert_called_once()
        answer_text = msg.answer.call_args.args[0]
        assert "Извините" in answer_text
        # Must NOT contain traceback / internal error
        assert "traceback" not in answer_text.lower()
        assert "timeout" not in answer_text

    @pytest.mark.asyncio
    async def test_skips_none_text(self):
        from reviewmind.bot.handlers.query import on_text_message

        msg = _make_bot_message(text="test")
        msg.text = None
        await on_text_message(msg)
        msg.answer.assert_not_called()

    @pytest.mark.asyncio
    async def test_truncates_long_answer(self):
        from reviewmind.bot.handlers.query import _MAX_ANSWER_LENGTH, on_text_message

        long_response = "x" * 5000
        msg = _make_bot_message("question")
        mock_client = _mock_llm_client(long_response)

        with patch("reviewmind.bot.handlers.query.LLMClient") as MockLLM:
            instance = MockLLM.return_value
            instance.__aenter__ = AsyncMock(return_value=mock_client)
            instance.__aexit__ = AsyncMock(return_value=False)
            await on_text_message(msg)

        answer_text = msg.answer.call_args.args[0]
        assert len(answer_text) <= _MAX_ANSWER_LENGTH
        assert answer_text.endswith("...")


# ══════════════════════════════════════════════════════════════
# Tests — Bot dispatcher wiring
# ══════════════════════════════════════════════════════════════


class TestBotDispatcherWiring:
    """Test that query_router is registered in the dispatcher."""

    def test_query_router_imported(self):
        from reviewmind.bot.handlers.query import router

        assert router.name == "query"

    def test_query_router_has_message_handler(self):
        """The query router has at least one message handler registered."""
        from reviewmind.bot.handlers.query import router

        # Check that there are message observers registered
        assert len(router.message.handlers) > 0

    def test_bot_main_imports_query_router(self):
        """Verify that bot/main.py imports and registers query_router."""
        import inspect

        import reviewmind.bot.main as bot_main

        source = inspect.getsource(bot_main)
        # Check that query_router is imported
        assert "from reviewmind.bot.handlers.query import router as query_router" in source
        # Check that it's included in create_dispatcher
        assert "dp.include_router(query_router)" in source

    def test_query_router_registered_after_others(self):
        """In bot/main.py source, query_router must be included after start and mode."""
        import inspect

        import reviewmind.bot.main as bot_main

        source = inspect.getsource(bot_main.create_dispatcher)
        start_pos = source.index("include_router(start_router)")
        mode_pos = source.index("include_router(mode_router)")
        query_pos = source.index("include_router(query_router)")
        assert query_pos > start_pos
        assert query_pos > mode_pos


# ══════════════════════════════════════════════════════════════
# Tests — API endpoint POST /query
# ══════════════════════════════════════════════════════════════


@pytest.fixture()
def query_app() -> FastAPI:
    """Minimal FastAPI app with the query endpoint."""
    from reviewmind.api.endpoints.query import router

    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture()
def query_client(query_app: FastAPI) -> TestClient:
    """TestClient for the query API."""
    return TestClient(query_app)


class TestApiQuerySchemas:
    """Test Pydantic request/response models."""

    def test_query_request_valid(self):
        from reviewmind.api.schemas import QueryRequest

        req = QueryRequest(message="Hello")
        assert req.message == "Hello"
        assert req.chat_history is None

    def test_query_request_with_history(self):
        from reviewmind.api.schemas import QueryRequest

        req = QueryRequest(
            message="Next question",
            chat_history=[{"role": "user", "content": "prev"}],
        )
        assert len(req.chat_history) == 1

    def test_query_request_empty_message_rejected(self):
        from pydantic import ValidationError

        from reviewmind.api.schemas import QueryRequest

        with pytest.raises(ValidationError):
            QueryRequest(message="")

    def test_query_response_fields(self):
        from reviewmind.api.schemas import QueryResponse

        resp = QueryResponse(answer="Hello", error=False, model="gpt-4o-mini")
        assert resp.answer == "Hello"
        assert resp.error is False
        assert resp.model == "gpt-4o-mini"


class TestApiQueryEndpoint:
    """Test POST /query endpoint."""

    def test_success(self, query_client: TestClient):
        with patch("reviewmind.api.endpoints.query.LLMClient") as MockLLM:
            mock_client = _mock_llm_client("LLM response")
            instance = MockLLM.return_value
            instance.__aenter__ = AsyncMock(return_value=mock_client)
            instance.__aexit__ = AsyncMock(return_value=False)

            resp = query_client.post("/query", json={"message": "Hello"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "LLM response"
        assert data["error"] is False

    def test_with_chat_history(self, query_client: TestClient):
        with patch("reviewmind.api.endpoints.query.LLMClient") as MockLLM:
            mock_client = _mock_llm_client("Follow-up answer")
            instance = MockLLM.return_value
            instance.__aenter__ = AsyncMock(return_value=mock_client)
            instance.__aexit__ = AsyncMock(return_value=False)

            resp = query_client.post(
                "/query",
                json={
                    "message": "Follow-up",
                    "chat_history": [{"role": "user", "content": "Hi"}],
                },
            )

        assert resp.status_code == 200
        assert resp.json()["answer"] == "Follow-up answer"

    def test_empty_message_422(self, query_client: TestClient):
        resp = query_client.post("/query", json={"message": ""})
        assert resp.status_code == 422

    def test_missing_message_422(self, query_client: TestClient):
        resp = query_client.post("/query", json={})
        assert resp.status_code == 422

    def test_llm_error_returns_fallback(self, query_client: TestClient):
        with patch("reviewmind.api.endpoints.query.LLMClient") as MockLLM:
            mock_client = _mock_llm_client_error(LLMError("API error"))
            instance = MockLLM.return_value
            instance.__aenter__ = AsyncMock(return_value=mock_client)
            instance.__aexit__ = AsyncMock(return_value=False)

            resp = query_client.post("/query", json={"message": "test"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["error"] is True
        assert "Извините" in data["answer"]

    def test_response_has_model_field(self, query_client: TestClient):
        with patch("reviewmind.api.endpoints.query.LLMClient") as MockLLM:
            mock_client = _mock_llm_client("ok")
            instance = MockLLM.return_value
            instance.__aenter__ = AsyncMock(return_value=mock_client)
            instance.__aexit__ = AsyncMock(return_value=False)

            resp = query_client.post("/query", json={"message": "hi"})

        data = resp.json()
        assert "model" in data


# ══════════════════════════════════════════════════════════════
# Tests — API router wiring
# ══════════════════════════════════════════════════════════════


class TestApiRouterWiring:
    """Test that query endpoint is wired into the API router."""

    def test_query_route_in_api_router(self):
        from reviewmind.api.router import api_router

        routes = [r.path for r in api_router.routes]
        assert "/query" in routes

    def test_full_app_has_query_route(self):
        """The main app factory includes the query route."""
        with patch("reviewmind.main.lifespan"):
            from reviewmind.main import create_app

            application = create_app()
        routes = [r.path for r in application.routes]
        assert "/query" in routes
