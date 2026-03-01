"""reviewmind/api/schemas.py — Pydantic request/response models for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request body for POST /query."""

    message: str = Field(..., min_length=1, max_length=4000, description="User query text")
    chat_history: list[dict[str, str]] | None = Field(
        default=None,
        description="Optional list of prior messages [{role, content}, ...]",
    )


class QueryResponse(BaseModel):
    """Response body for POST /query."""

    answer: str = Field(..., description="LLM-generated response")
    error: bool = Field(default=False, description="Whether an error occurred")
    model: str | None = Field(default=None, description="Model used for generation")
