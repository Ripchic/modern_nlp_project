"""reviewmind/api/endpoints/query.py — POST /query endpoint.

Temporary direct-LLM implementation.  Will be extended with RAG in later tasks.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter

from reviewmind.api.schemas import QueryRequest, QueryResponse
from reviewmind.core.llm import LLMClient
from reviewmind.services.query_service import QueryService

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

router = APIRouter()


@router.post("/query", response_model=QueryResponse)
async def post_query(body: QueryRequest) -> QueryResponse:
    """Accept a user query, call the LLM, and return the answer."""
    logger.info("api_query_received", message_len=len(body.message))

    async with LLMClient() as client:
        service = QueryService(llm_client=client)
        result = await service.answer(
            body.message,
            chat_history=body.chat_history,
        )

    return QueryResponse(
        answer=result.answer,
        error=result.error,
        model=result.model,
    )
