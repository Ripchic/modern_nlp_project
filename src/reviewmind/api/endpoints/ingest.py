"""reviewmind/api/endpoints/ingest.py — POST /ingest endpoint.

Accepts a list of URLs, runs each through the ingestion pipeline
(parse → clean → sponsor detect → chunk → embed → upsert), and returns
per-URL results plus aggregate counts.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Request
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from reviewmind.api.rate_limit import RATE_LIMIT_STRING, _check_exempt, limiter
from reviewmind.api.schemas import IngestRequest, IngestResponse, IngestURLResult
from reviewmind.ingestion.pipeline import IngestionPipeline

logger = structlog.get_logger(__name__)

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
@limiter.limit(RATE_LIMIT_STRING, exempt_when=_check_exempt)
async def post_ingest(body: IngestRequest, request: Request) -> IngestResponse:
    """Ingest a list of URLs: parse → clean → chunk → embed → upsert into Qdrant."""
    log = logger.bind(user_id=body.user_id, session_id=body.session_id, url_count=len(body.urls))
    log.info("api_ingest_received")

    qdrant = getattr(request.app.state, "qdrant", None)
    if qdrant is None:
        log.error("qdrant_unavailable")
        return IngestResponse(
            results=[IngestURLResult(url=u, status="failed", error="Vector store unavailable") for u in body.urls],
            failed_count=len(body.urls),
        )

    # Build an optional DB session for source metadata persistence
    engine = getattr(request.app.state, "db_engine", None)
    session_factory = None
    if engine is not None:
        session_factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

    try:
        if session_factory is not None:
            async with session_factory() as db_session:
                response = await _run_pipeline(body, qdrant, db_session, log)
                await db_session.commit()
                return response
        else:
            return await _run_pipeline(body, qdrant, None, log)
    except Exception as exc:
        log.error("ingest_unexpected_error", error=str(exc))
        return IngestResponse(
            results=[IngestURLResult(url=u, status="failed", error="Internal error") for u in body.urls],
            failed_count=len(body.urls),
        )


async def _run_pipeline(
    body: IngestRequest,
    qdrant,
    db_session: AsyncSession | None,
    log,
) -> IngestResponse:
    """Execute the ingestion pipeline and build the response."""
    async with IngestionPipeline(
        qdrant_client=qdrant,
        db_session=db_session,
    ) as pipeline:
        ingestion_result = await pipeline.ingest_urls(
            urls=body.urls,
            product_query=body.product_query,
            session_id=body.session_id,
        )

    url_results: list[IngestURLResult] = []
    for r in ingestion_result.results:
        url_results.append(
            IngestURLResult(
                url=r.url,
                status="success" if r.success else "failed",
                source_type=r.source_type,
                chunks_count=r.chunks_count,
                error=r.error,
            )
        )

    log.info(
        "ingest_complete",
        success=ingestion_result.success_count,
        failed=ingestion_result.failed_count,
        chunks=ingestion_result.chunks_count,
    )

    return IngestResponse(
        results=url_results,
        success_count=ingestion_result.success_count,
        failed_count=ingestion_result.failed_count,
        chunks_count=ingestion_result.chunks_count,
    )
