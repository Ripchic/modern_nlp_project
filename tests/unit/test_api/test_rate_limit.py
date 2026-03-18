"""Tests for reviewmind/api/rate_limit.py — slowapi rate limiting middleware."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from reviewmind.api.rate_limit import (
    RATE_LIMIT_PER_MINUTE,
    RATE_LIMIT_STRING,
    _get_rate_limit_string,
    get_user_id_key,
    is_admin_request,
    limiter,
    rate_limit_exceeded_handler,
    setup_rate_limiting,
)


def _full_reset_limiter() -> None:
    """Reset limiter storage AND accumulated route/decorator state.

    slowapi's ``@limiter.limit()`` appends to ``_route_limits`` every time a
    decorated function is created.  When ``_create_rate_limited_app()`` is
    called more than once (as happens across tests), duplicate ``Limit``
    objects accumulate, causing each request to be counted N times.
    """
    limiter.reset()
    limiter._route_limits.clear()
    limiter._Limiter__marked_for_limiting.clear()

# ── Helpers ───────────────────────────────────────────────────


def _make_request(
    user_id: int | None = None,
    client_host: str = "127.0.0.1",
) -> Request:
    """Create a mock Request with optional pre-parsed body."""
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/query",
        "headers": [],
        "query_string": b"",
    }
    req = Request(scope=scope)
    body = {"user_id": user_id} if user_id is not None else {}
    req.state._parsed_body = body  # noqa: SLF001
    req._client = MagicMock()
    req._client.host = client_host
    # Override the client property
    scope["client"] = (client_host, 0)
    return req


def _create_rate_limited_app() -> FastAPI:
    """Build a minimal FastAPI app with rate limiting enabled."""
    from reviewmind.api.rate_limit import RATE_LIMIT_STRING, _check_exempt, limiter

    app = FastAPI()
    setup_rate_limiting(app)

    @app.post("/query")
    @limiter.limit(RATE_LIMIT_STRING, exempt_when=_check_exempt)
    async def query_endpoint(request: Request):
        body = await request.json()
        return {"answer": "ok", "user_id": body.get("user_id")}

    @app.post("/ingest")
    @limiter.limit(RATE_LIMIT_STRING, exempt_when=_check_exempt)
    async def ingest_endpoint(request: Request):
        return {"status": "ok"}

    @app.get("/health")
    async def health_endpoint():
        return {"status": "ok"}

    return app


# ── TestConstants ─────────────────────────────────────────────


class TestConstants:
    """Verify module constants."""

    def test_rate_limit_per_minute_value(self):
        assert RATE_LIMIT_PER_MINUTE == 10

    def test_rate_limit_string_format(self):
        assert RATE_LIMIT_STRING == "10/minute"

    def test_rate_limit_string_contains_per_minute(self):
        assert "/minute" in RATE_LIMIT_STRING

    def test_get_rate_limit_string_default(self):
        with patch("reviewmind.config.settings") as mock_settings:
            mock_settings.rate_limit_per_minute = 10
            result = _get_rate_limit_string()
            assert result == "10/minute"

    def test_get_rate_limit_string_custom(self):
        with patch("reviewmind.config.settings") as mock_settings:
            mock_settings.rate_limit_per_minute = 20
            result = _get_rate_limit_string()
            assert result == "20/minute"

    def test_get_rate_limit_string_fallback_on_error(self):
        with patch("reviewmind.config.settings") as mock_settings:
            type(mock_settings).rate_limit_per_minute = PropertyMock(side_effect=Exception("no config"))
            result = _get_rate_limit_string()
            assert result == "10/minute"


# ── TestGetUserIdKey ──────────────────────────────────────────


class TestGetUserIdKey:
    """Test the key function that extracts user_id from the request."""

    def test_extracts_user_id_from_body(self):
        req = _make_request(user_id=12345)
        assert get_user_id_key(req) == "12345"

    def test_extracts_user_id_zero(self):
        req = _make_request(user_id=0)
        assert get_user_id_key(req) == "0"

    def test_falls_back_to_ip(self):
        req = _make_request(user_id=None)
        req.state._parsed_body = {}  # noqa: SLF001
        assert get_user_id_key(req) == "127.0.0.1"

    def test_falls_back_when_no_body(self):
        req = _make_request(user_id=None)
        req.state._parsed_body = None  # noqa: SLF001
        assert get_user_id_key(req) == "127.0.0.1"

    def test_falls_back_when_body_is_not_dict(self):
        req = _make_request()
        req.state._parsed_body = "invalid"  # noqa: SLF001
        assert get_user_id_key(req) == "127.0.0.1"

    def test_custom_client_host(self):
        req = _make_request(user_id=None)
        req.state._parsed_body = {}  # noqa: SLF001
        # Override scope client
        req.scope["client"] = ("10.0.0.1", 0)
        assert get_user_id_key(req) == "10.0.0.1"

    def test_no_client(self):
        scope = {
            "type": "http",
            "method": "POST",
            "path": "/query",
            "headers": [],
            "query_string": b"",
        }
        req = Request(scope=scope)
        req.state._parsed_body = {}  # noqa: SLF001
        assert get_user_id_key(req) == "unknown"


# ── TestIsAdminRequest ────────────────────────────────────────


class TestIsAdminRequest:
    """Test the admin bypass check."""

    def test_admin_user_returns_true(self):
        req = _make_request(user_id=99999)
        with patch("reviewmind.config.settings") as mock_settings:
            mock_settings.admin_user_ids = [99999, 88888]
            assert is_admin_request(req) is True

    def test_non_admin_user_returns_false(self):
        req = _make_request(user_id=11111)
        with patch("reviewmind.config.settings") as mock_settings:
            mock_settings.admin_user_ids = [99999]
            assert is_admin_request(req) is False

    def test_empty_admin_list(self):
        req = _make_request(user_id=12345)
        with patch("reviewmind.config.settings") as mock_settings:
            mock_settings.admin_user_ids = []
            assert is_admin_request(req) is False

    def test_ip_fallback_not_admin(self):
        req = _make_request(user_id=None)
        req.state._parsed_body = {}  # noqa: SLF001
        with patch("reviewmind.config.settings") as mock_settings:
            mock_settings.admin_user_ids = [99999]
            assert is_admin_request(req) is False

    def test_config_error_returns_false(self):
        req = _make_request(user_id=12345)
        with patch("reviewmind.config.settings", side_effect=Exception("no config")):
            assert is_admin_request(req) is False


# ── TestRateLimitExceededHandler ──────────────────────────────


def _make_rate_limit_exc(detail: str = "10 per 1 minute") -> MagicMock:
    """Create a mock RateLimitExceeded with the expected attributes."""
    exc = MagicMock()
    exc.detail = detail
    exc.retry_after = 60
    return exc


class TestRateLimitExceededHandler:
    """Test the custom 429 handler."""

    def test_returns_429_status(self):
        req = _make_request(user_id=123)
        exc = _make_rate_limit_exc()
        resp = rate_limit_exceeded_handler(req, exc)
        assert resp.status_code == 429

    def test_response_has_retry_after_header(self):
        req = _make_request(user_id=123)
        exc = _make_rate_limit_exc()
        resp = rate_limit_exceeded_handler(req, exc)
        assert "Retry-After" in resp.headers

    def test_response_body_contains_detail(self):
        req = _make_request(user_id=123)
        exc = _make_rate_limit_exc()
        resp = rate_limit_exceeded_handler(req, exc)
        body = json.loads(resp.body)
        assert "detail" in body
        assert "Слишком много запросов" in body["detail"]

    def test_response_body_has_retry_after_field(self):
        req = _make_request(user_id=123)
        exc = _make_rate_limit_exc()
        resp = rate_limit_exceeded_handler(req, exc)
        body = json.loads(resp.body)
        assert "retry_after" in body


# ── TestLimiterInstance ───────────────────────────────────────


class TestLimiterInstance:
    """Test the limiter singleton."""

    def test_limiter_is_limiter(self):
        from slowapi import Limiter

        assert isinstance(limiter, Limiter)

    def test_limiter_has_no_default_limits(self):
        assert limiter._default_limits == []  # noqa: SLF001


# ── TestSetupRateLimiting ─────────────────────────────────────


class TestSetupRateLimiting:
    """Test the setup_rate_limiting helper."""

    def test_attaches_limiter_to_state(self):
        app = FastAPI()
        setup_rate_limiting(app)
        assert app.state.limiter is limiter

    def test_registers_exception_handler(self):
        app = FastAPI()
        setup_rate_limiting(app)
        from slowapi.errors import RateLimitExceeded

        assert RateLimitExceeded in app.exception_handlers

    def test_registers_middleware(self):
        app = FastAPI()
        initial_middleware_count = len(app.user_middleware)
        setup_rate_limiting(app)
        assert len(app.user_middleware) == initial_middleware_count + 1


# ── TestRateLimitMiddleware ───────────────────────────────────


class TestRateLimitMiddleware:
    """Test the JSON body pre-parsing middleware."""

    @pytest.fixture()
    def app(self) -> FastAPI:
        """Minimal app with rate limit middleware, no slowapi decorators."""
        app = FastAPI()

        @app.post("/echo")
        async def echo(request: Request):
            body = getattr(request.state, "_parsed_body", None)
            return {"parsed": body is not None, "user_id": (body or {}).get("user_id")}

        setup_rate_limiting(app)
        return app

    @pytest.fixture()
    def client(self, app: FastAPI) -> TestClient:
        return TestClient(app)

    def test_parses_json_body(self, client: TestClient):
        resp = client.post("/echo", json={"user_id": 42})
        data = resp.json()
        assert data["parsed"] is True
        assert data["user_id"] == 42

    def test_no_body_sets_none(self, client: TestClient):
        resp = client.get("/echo")
        # GET has no body parsing
        assert resp.status_code in (200, 405)

    def test_non_json_content_type(self, client: TestClient):
        resp = client.post(
            "/echo",
            content=b"hello",
            headers={"content-type": "text/plain"},
        )
        data = resp.json()
        assert data["parsed"] is False


# ── TestRateLimitIntegration ──────────────────────────────────


class TestRateLimitIntegration:
    """Integration tests for rate limiting on actual endpoints."""

    @pytest.fixture(autouse=True)
    def _reset_limiter(self):
        """Reset rate limiter storage and route state between tests."""
        _full_reset_limiter()
        yield
        _full_reset_limiter()

    @pytest.fixture()
    def app(self) -> FastAPI:
        return _create_rate_limited_app()

    @pytest.fixture()
    def client(self, app: FastAPI) -> TestClient:
        return TestClient(app)

    def test_first_request_succeeds(self, client: TestClient):
        resp = client.post("/query", json={"user_id": 1, "query": "test"})
        assert resp.status_code == 200

    def test_tenth_request_succeeds(self, client: TestClient):
        for _ in range(RATE_LIMIT_PER_MINUTE):
            resp = client.post("/query", json={"user_id": 2, "query": "test"})
        assert resp.status_code == 200

    def test_eleventh_request_returns_429(self, client: TestClient):
        for _ in range(RATE_LIMIT_PER_MINUTE):
            client.post("/query", json={"user_id": 3, "query": "test"})
        resp = client.post("/query", json={"user_id": 3, "query": "test"})
        assert resp.status_code == 429

    def test_429_response_has_retry_after(self, client: TestClient):
        for _ in range(RATE_LIMIT_PER_MINUTE):
            client.post("/query", json={"user_id": 4, "query": "test"})
        resp = client.post("/query", json={"user_id": 4, "query": "test"})
        assert "Retry-After" in resp.headers

    def test_429_response_json_body(self, client: TestClient):
        for _ in range(RATE_LIMIT_PER_MINUTE):
            client.post("/query", json={"user_id": 5, "query": "test"})
        resp = client.post("/query", json={"user_id": 5, "query": "test"})
        data = resp.json()
        assert "detail" in data
        assert "retry_after" in data

    def test_different_users_have_separate_limits(self, client: TestClient):
        for _ in range(RATE_LIMIT_PER_MINUTE):
            client.post("/query", json={"user_id": 6, "query": "test"})
        # User 6 is rate limited
        resp6 = client.post("/query", json={"user_id": 6, "query": "test"})
        assert resp6.status_code == 429
        # User 7 still has quota
        resp7 = client.post("/query", json={"user_id": 7, "query": "test"})
        assert resp7.status_code == 200

    def test_get_endpoint_not_rate_limited(self, client: TestClient):
        for _ in range(RATE_LIMIT_PER_MINUTE + 5):
            resp = client.get("/health")
        assert resp.status_code == 200

    def test_rate_limit_applies_across_endpoints(self, client: TestClient):
        """Rate limit is per user — requests to /query and /ingest share the same bucket per endpoint."""
        for _ in range(RATE_LIMIT_PER_MINUTE):
            client.post("/query", json={"user_id": 8, "query": "test"})
        resp = client.post("/query", json={"user_id": 8, "query": "test"})
        assert resp.status_code == 429
        # /ingest endpoint has its own limit counter
        resp_ingest = client.post("/ingest", json={"user_id": 8})
        assert resp_ingest.status_code == 200


# ── TestAdminBypass ───────────────────────────────────────────


class TestAdminBypass:
    """Test that admin users bypass rate limiting."""

    @pytest.fixture(autouse=True)
    def _reset_limiter(self):
        _full_reset_limiter()
        yield
        _full_reset_limiter()

    @pytest.fixture()
    def app(self) -> FastAPI:
        return _create_rate_limited_app()

    @pytest.fixture()
    def client(self, app: FastAPI) -> TestClient:
        return TestClient(app)

    def test_admin_not_rate_limited(self, client: TestClient):
        admin_id = 99999
        with patch("reviewmind.config.settings") as mock_settings:
            mock_settings.admin_user_ids = [admin_id]
            for _ in range(RATE_LIMIT_PER_MINUTE + 5):
                resp = client.post("/query", json={"user_id": admin_id, "query": "test"})
            assert resp.status_code == 200

    def test_non_admin_rate_limited(self, client: TestClient):
        user_id = 11111
        with patch("reviewmind.config.settings") as mock_settings:
            mock_settings.admin_user_ids = [99999]
            for _ in range(RATE_LIMIT_PER_MINUTE):
                client.post("/query", json={"user_id": user_id, "query": "test"})
            resp = client.post("/query", json={"user_id": user_id, "query": "test"})
            assert resp.status_code == 429

    def test_admin_and_non_admin_independent(self, client: TestClient):
        admin_id = 99999
        user_id = 11111
        with patch("reviewmind.config.settings") as mock_settings:
            mock_settings.admin_user_ids = [admin_id]
            # Exhaust non-admin's limit
            for _ in range(RATE_LIMIT_PER_MINUTE):
                client.post("/query", json={"user_id": user_id, "query": "test"})
            # Non-admin blocked
            resp_user = client.post("/query", json={"user_id": user_id, "query": "test"})
            assert resp_user.status_code == 429
            # Admin still passes
            resp_admin = client.post("/query", json={"user_id": admin_id, "query": "test"})
            assert resp_admin.status_code == 200


# ── TestMainAppIntegration ────────────────────────────────────


class TestMainAppIntegration:
    """Test that the real app has rate limiting configured."""

    def test_app_has_limiter_in_state(self):
        from reviewmind.main import app

        assert hasattr(app.state, "limiter")

    def test_app_has_rate_limit_exception_handler(self):
        from slowapi.errors import RateLimitExceeded

        from reviewmind.main import app

        assert RateLimitExceeded in app.exception_handlers


# ── TestModuleExports ─────────────────────────────────────────


class TestModuleExports:
    """Verify public API of the rate_limit module."""

    def test_limiter_exported(self):
        from reviewmind.api.rate_limit import limiter

        assert limiter is not None

    def test_rate_limit_string_exported(self):
        from reviewmind.api.rate_limit import RATE_LIMIT_STRING

        assert isinstance(RATE_LIMIT_STRING, str)

    def test_is_admin_request_exported(self):
        from reviewmind.api.rate_limit import is_admin_request

        assert callable(is_admin_request)

    def test_get_user_id_key_exported(self):
        from reviewmind.api.rate_limit import get_user_id_key

        assert callable(get_user_id_key)

    def test_setup_rate_limiting_exported(self):
        from reviewmind.api.rate_limit import setup_rate_limiting

        assert callable(setup_rate_limiting)

    def test_rate_limit_per_minute_exported(self):
        from reviewmind.api.rate_limit import RATE_LIMIT_PER_MINUTE

        assert isinstance(RATE_LIMIT_PER_MINUTE, int)


# ── TestIntegrationScenarios ─────────────────────────────────


class TestIntegrationScenarios:
    """End-to-end scenarios matching TASK-042 test steps."""

    @pytest.fixture(autouse=True)
    def _reset_limiter(self):
        _full_reset_limiter()
        yield
        _full_reset_limiter()

    @pytest.fixture()
    def app(self) -> FastAPI:
        return _create_rate_limited_app()

    @pytest.fixture()
    def client(self, app: FastAPI) -> TestClient:
        return TestClient(app)

    def test_step1_ten_requests_all_pass(self, client: TestClient):
        """Send 10 requests within the window — all succeed."""
        for i in range(RATE_LIMIT_PER_MINUTE):
            resp = client.post("/query", json={"user_id": 100, "query": f"q{i}"})
            assert resp.status_code == 200

    def test_step2_eleventh_returns_429(self, client: TestClient):
        """Send 11th request → HTTP 429."""
        for _ in range(RATE_LIMIT_PER_MINUTE):
            client.post("/query", json={"user_id": 101, "query": "test"})
        resp = client.post("/query", json={"user_id": 101, "query": "test"})
        assert resp.status_code == 429

    def test_step4_admin_sends_15_all_pass(self, client: TestClient):
        """Admin sends 15 requests → all pass (no rate limit)."""
        admin_id = 55555
        with patch("reviewmind.config.settings") as mock_settings:
            mock_settings.admin_user_ids = [admin_id]
            for i in range(15):
                resp = client.post("/query", json={"user_id": admin_id, "query": f"q{i}"})
                assert resp.status_code == 200

    def test_step5_429_has_retry_after(self, client: TestClient):
        """Check Retry-After header in 429 response."""
        for _ in range(RATE_LIMIT_PER_MINUTE):
            client.post("/query", json={"user_id": 102, "query": "test"})
        resp = client.post("/query", json={"user_id": 102, "query": "test"})
        assert resp.status_code == 429
        assert "Retry-After" in resp.headers
        # Retry-After should be a numeric value (seconds)
        assert int(resp.headers["Retry-After"]) >= 0

    def test_config_rate_limit_field(self):
        """Verify rate_limit_per_minute is in config Settings."""
        from reviewmind.config import Settings

        fields = Settings.model_fields
        assert "rate_limit_per_minute" in fields
