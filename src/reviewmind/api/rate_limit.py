"""reviewmind/api/rate_limit.py — Rate limiting via slowapi (10 req/min per user_id).

Integrates slowapi with FastAPI.  Admin users (from config.ADMIN_USER_IDS)
bypass the rate limit.  Non-admin users are limited to RATE_LIMIT_PER_MINUTE
requests per minute, identified by the ``user_id`` field in the JSON body.
"""

from __future__ import annotations

import json
from contextvars import ContextVar
from typing import TYPE_CHECKING

import structlog
from fastapi import Request, Response
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from starlette.responses import JSONResponse

if TYPE_CHECKING:
    from fastapi import FastAPI

logger = structlog.get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────

RATE_LIMIT_PER_MINUTE: int = 10
RATE_LIMIT_STRING: str = f"{RATE_LIMIT_PER_MINUTE}/minute"

# Context variable set to True by the middleware when the request is from an admin.
_is_exempt: ContextVar[bool] = ContextVar("_is_exempt", default=False)


def _get_rate_limit_string() -> str:
    """Build the rate limit string from config, with a safe fallback."""
    try:
        from reviewmind.config import settings

        return f"{settings.rate_limit_per_minute}/minute"
    except Exception:
        return RATE_LIMIT_STRING


# ── Key function ──────────────────────────────────────────────


def get_user_id_key(request: Request) -> str:
    """Extract the rate-limit key from the request.

    Tries to read ``user_id`` from the cached JSON body (stored by the
    middleware).  Falls back to the client IP address.
    """
    body = getattr(request.state, "_parsed_body", None)
    if body and isinstance(body, dict):
        user_id = body.get("user_id")
        if user_id is not None:
            return str(user_id)

    # Fallback to client IP
    return request.client.host if request.client else "unknown"


def is_admin_request(request: Request) -> bool:
    """Return True if the current request originates from an admin user."""
    key = get_user_id_key(request)
    try:
        from reviewmind.config import settings

        uid = int(key)
        return uid in settings.admin_user_ids
    except (ValueError, TypeError, Exception):
        return False


def _check_exempt() -> bool:
    """Return True if the current request is exempt (admin).

    This is a zero-argument callable used by slowapi's ``exempt_when``.
    The value is set by :func:`rate_limit_middleware` before the request
    reaches the decorated endpoint.
    """
    return _is_exempt.get(False)


# ── Limiter instance ──────────────────────────────────────────

limiter = Limiter(
    key_func=get_user_id_key,
    default_limits=[],
    storage_uri="memory://",
)


# ── 429 handler ───────────────────────────────────────────────


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> Response:
    """Return a JSON 429 response with Retry-After header."""
    retry_after = getattr(exc, "retry_after", 60)
    logger.warning(
        "rate_limit_exceeded",
        key=get_user_id_key(request),
        detail=str(exc.detail),
    )
    return JSONResponse(
        status_code=429,
        content={
            "detail": "Слишком много запросов. Пожалуйста, подождите.",
            "retry_after": retry_after,
        },
        headers={"Retry-After": str(retry_after)},
    )


# ── Middleware to pre-parse body for key extraction ───────────


async def rate_limit_middleware(request: Request, call_next):
    """Pre-parse JSON body so the key function can read ``user_id``.

    Also sets the ``_is_exempt`` context variable to ``True`` for admin
    users so that the ``exempt_when`` callback can skip rate limiting.
    """
    if request.method in ("POST", "PUT", "PATCH"):
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            try:
                body_bytes = await request.body()
                parsed = json.loads(body_bytes)
                request.state._parsed_body = parsed  # noqa: SLF001
            except Exception:
                request.state._parsed_body = None  # noqa: SLF001
        else:
            request.state._parsed_body = None  # noqa: SLF001
    else:
        request.state._parsed_body = None  # noqa: SLF001

    # Set admin exemption flag in contextvar
    token = _is_exempt.set(is_admin_request(request))
    try:
        return await call_next(request)
    finally:
        _is_exempt.reset(token)


# ── Setup helper ──────────────────────────────────────────────


def setup_rate_limiting(app: FastAPI) -> None:
    """Attach slowapi rate limiter and error handler to *app*."""
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)
    app.middleware("http")(rate_limit_middleware)
