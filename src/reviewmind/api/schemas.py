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
