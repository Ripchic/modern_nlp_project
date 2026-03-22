"""reviewmind/metrics.py — Prometheus metric definitions and ASGI middleware.

Centralises all application metrics.  Metrics are collected by
``prometheus_client`` and exported via ``/metrics`` (mounted as a sub-app
in :func:`setup_metrics`).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from prometheus_client import Counter, Gauge, Histogram, make_asgi_app
from starlette.requests import Request
from starlette.routing import Match

if TYPE_CHECKING:
    from fastapi import FastAPI

# ── HTTP Metrics ─────────────────────────────────────────────────────────────

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    labelnames=["endpoint", "method", "status_code"],
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total number of HTTP requests",
    labelnames=["endpoint", "method", "status_code"],
)

HTTP_REQUESTS_IN_PROGRESS = Gauge(
    "http_requests_in_progress",
    "Number of HTTP requests currently being processed",
    labelnames=["endpoint"],
)

# ── Celery Metrics ───────────────────────────────────────────────────────────

CELERY_TASK_DURATION_SECONDS = Histogram(
    "celery_task_duration_seconds",
    "Celery task execution duration in seconds",
    labelnames=["task_name", "status"],
    buckets=(1.0, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0),
)

CELERY_TASKS_TOTAL = Counter(
    "celery_tasks_total",
    "Total number of Celery tasks completed",
    labelnames=["task_name", "status"],
)

# ── RAG / Embedding / Scraper Metrics ────────────────────────────────────────

RAG_QUERY_DURATION_SECONDS = Histogram(
    "rag_query_duration_seconds",
    "RAG pipeline query duration in seconds",
    buckets=(0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

EMBEDDING_DURATION_SECONDS = Histogram(
    "embedding_duration_seconds",
    "Embedding API call duration in seconds",
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

SCRAPER_DURATION_SECONDS = Histogram(
    "scraper_duration_seconds",
    "Scraper execution duration in seconds",
    labelnames=["scraper_type"],
    buckets=(0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

# ── Rate Limiting ────────────────────────────────────────────────────────────

RATE_LIMIT_HITS_TOTAL = Counter(
    "rate_limit_hits_total",
    "Total number of rate limit rejections (HTTP 429)",
)

# ── Ingestion ────────────────────────────────────────────────────────────────

INGESTION_CHUNKS_TOTAL = Counter(
    "ingestion_chunks_total",
    "Total number of chunks processed during ingestion",
    labelnames=["status"],
)


# ── Middleware ────────────────────────────────────────────────────────────────


def _get_endpoint_path(request: Request) -> str:
    """Resolve the matched route path template (e.g. ``/status/{job_id}``)."""
    for route in request.app.routes:
        match, _ = route.matches(request.scope)
        if match == Match.FULL:
            return getattr(route, "path", request.url.path)
    return request.url.path


async def prometheus_middleware(request: Request, call_next):
    """ASGI middleware that records HTTP request metrics.

    Skips the ``/metrics`` endpoint itself to avoid self-referencing noise.
    """
    path = request.url.path
    if path == "/metrics":
        return await call_next(request)

    endpoint = _get_endpoint_path(request)
    method = request.method

    HTTP_REQUESTS_IN_PROGRESS.labels(endpoint=endpoint).inc()
    start = time.perf_counter()

    try:
        response = await call_next(request)
    except Exception:
        # Record as 500 when an unhandled exception escapes
        duration = time.perf_counter() - start
        HTTP_REQUEST_DURATION_SECONDS.labels(endpoint=endpoint, method=method, status_code="500").observe(duration)
        HTTP_REQUESTS_TOTAL.labels(endpoint=endpoint, method=method, status_code="500").inc()
        raise
    finally:
        HTTP_REQUESTS_IN_PROGRESS.labels(endpoint=endpoint).dec()

    duration = time.perf_counter() - start
    status = str(response.status_code)
    HTTP_REQUEST_DURATION_SECONDS.labels(endpoint=endpoint, method=method, status_code=status).observe(duration)
    HTTP_REQUESTS_TOTAL.labels(endpoint=endpoint, method=method, status_code=status).inc()

    return response


# ── Setup helper ─────────────────────────────────────────────────────────────


def setup_metrics(app: FastAPI) -> None:
    """Mount the ``/metrics`` Prometheus endpoint and register middleware."""
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)
    app.middleware("http")(prometheus_middleware)
