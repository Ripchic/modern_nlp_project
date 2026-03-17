"""reviewmind/api/schemas.py — Pydantic request/response models for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request body for POST /query."""

    user_id: int = Field(..., description="Telegram user ID")
    session_id: str | None = Field(default=None, description="Session identifier")
    query: str = Field(..., min_length=1, max_length=4000, description="User query text")
    mode: str = Field(default="auto", description="Query mode: 'auto' or 'manual'")
    urls: list[str] | None = Field(default=None, description="URLs for manual mode")
    chat_history: list[dict[str, str]] | None = Field(
        default=None,
        description="Optional list of prior messages [{role, content}, ...]",
    )
    product_query: str | None = Field(
        default=None,
        description="Product name for Qdrant pre-filter (auto-extracted if omitted)",
    )


class QueryResponse(BaseModel):
    """Response body for POST /query."""

    answer: str = Field(..., description="LLM-generated response")
    sources: list[str] = Field(default_factory=list, description="Source URLs used in the answer")
    used_curated: bool = Field(default=False, description="Whether curated sources were used")
    used_tavily: bool = Field(default=False, description="Whether Tavily fallback was used")
    confidence_met: bool = Field(default=False, description="Whether confidence check passed")
    chunks_count: int = Field(default=0, description="Number of chunks used in context")
    response_time_ms: int = Field(default=0, description="Response time in milliseconds")
    query_log_id: int | None = Field(default=None, description="Query log ID for feedback")
    error: bool = Field(default=False, description="Whether an error occurred")


# ── Ingest schemas ────────────────────────────────────────────


class IngestRequest(BaseModel):
    """Request body for POST /ingest."""

    user_id: int = Field(..., description="Telegram user ID")
    session_id: str | None = Field(default=None, description="Session identifier")
    urls: list[str] = Field(..., min_length=1, description="List of URLs to ingest")
    product_query: str = Field(
        default="",
        max_length=500,
        description="Product name / search query to tag ingested chunks",
    )


class IngestURLResult(BaseModel):
    """Per-URL ingestion result."""

    url: str = Field(..., description="The URL that was processed")
    status: str = Field(..., description="'success' or 'failed'")
    source_type: str = Field(default="", description="Detected source type (youtube/reddit/web)")
    chunks_count: int = Field(default=0, description="Number of chunks ingested from this URL")
    error: str | None = Field(default=None, description="Error message if status is 'failed'")


class IngestResponse(BaseModel):
    """Response body for POST /ingest."""

    results: list[IngestURLResult] = Field(default_factory=list, description="Per-URL results")
    success_count: int = Field(default=0, description="Number of successfully ingested URLs")
    failed_count: int = Field(default=0, description="Number of failed URLs")
    chunks_count: int = Field(default=0, description="Total number of chunks ingested")
