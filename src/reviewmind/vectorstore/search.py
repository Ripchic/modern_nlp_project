"""reviewmind/vectorstore/search.py — Hybrid search across Qdrant collections.

Performs parallel searches across ``curated_kb`` (priority) and ``auto_crawled``
collections, merges results, and returns a unified list of :class:`SearchResult`.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchAny,
    MatchValue,
    ScoredPoint,
)

from reviewmind.vectorstore.collections import (
    COLLECTION_AUTO_CRAWLED,
    COLLECTION_CURATED_KB,
)

logger = structlog.get_logger("reviewmind.vectorstore.search")

# ── Constants ────────────────────────────────────────────────────────────────

#: Default number of results to fetch per collection.
DEFAULT_TOP_K = 5

#: Default score threshold — points below this are discarded.
DEFAULT_SCORE_THRESHOLD: float | None = None


# ── Data types ───────────────────────────────────────────────────────────────


@dataclass
class SearchResult:
    """A single search result with text, score, and metadata.

    Attributes
    ----------
    text:
        The chunk text stored in the Qdrant payload.
    score:
        Cosine similarity score from Qdrant (higher = more relevant).
    source_url:
        URL of the original source.
    source_type:
        Type of source, e.g. ``"youtube"``, ``"reddit"``, ``"web"``.
    is_curated:
        ``True`` if the result comes from the ``curated_kb`` collection.
    is_sponsored:
        ``True`` if the source is flagged as sponsored content.
    collection:
        Name of the Qdrant collection this result came from.
    product_query:
        Product query string stored in the payload (may be empty).
    language:
        Detected language of the chunk (e.g. ``"ru"``, ``"en"``).
    chunk_index:
        Index of the chunk within the source document.
    point_id:
        The Qdrant point ID (useful for deduplication).
    extra:
        Any additional payload fields not captured above.
    """

    text: str
    score: float
    source_url: str = ""
    source_type: str = ""
    is_curated: bool = False
    is_sponsored: bool = False
    collection: str = ""
    product_query: str = ""
    language: str = ""
    chunk_index: int = 0
    point_id: str | int = ""
    extra: dict = field(default_factory=dict)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _build_product_filter(product_query: str) -> Filter:
    """Build a Qdrant payload filter to match a specific *product_query*."""
    return Filter(
        must=[
            FieldCondition(
                key="product_query",
                match=MatchValue(value=product_query),
            ),
        ],
    )


def _scored_point_to_result(
    point: ScoredPoint,
    collection: str,
) -> SearchResult:
    """Convert a Qdrant :class:`ScoredPoint` to a :class:`SearchResult`."""
    payload = point.payload or {}

    known_keys = {
        "text",
        "source_url",
        "source_type",
        "is_curated",
        "is_sponsored",
        "product_query",
        "language",
        "chunk_index",
    }
    extra = {k: v for k, v in payload.items() if k not in known_keys}

    return SearchResult(
        text=payload.get("text", ""),
        score=point.score,
        source_url=payload.get("source_url", ""),
        source_type=payload.get("source_type", ""),
        is_curated=bool(payload.get("is_curated", False)),
        is_sponsored=bool(payload.get("is_sponsored", False)),
        collection=collection,
        product_query=payload.get("product_query", ""),
        language=payload.get("language", ""),
        chunk_index=int(payload.get("chunk_index", 0)),
        point_id=point.id,
        extra=extra,
    )


# ── Single-collection search ────────────────────────────────────────────────


async def search_collection(
    client: AsyncQdrantClient,
    collection_name: str,
    query_vector: list[float],
    top_k: int = DEFAULT_TOP_K,
    product_query: str | None = None,
    score_threshold: float | None = DEFAULT_SCORE_THRESHOLD,
) -> list[SearchResult]:
    """Search a single Qdrant collection.

    Parameters
    ----------
    client:
        An active :class:`AsyncQdrantClient`.
    collection_name:
        Name of the Qdrant collection to search.
    query_vector:
        The embedding vector for the user's query.
    top_k:
        Maximum number of results to return.
    product_query:
        Optional product-query string for payload pre-filtering.
    score_threshold:
        Minimum score; points below this are discarded.  ``None`` = no filter.

    Returns
    -------
    list[SearchResult]
        Results sorted by descending score.
    """
    query_filter = _build_product_filter(product_query) if product_query else None

    try:
        response = await client.query_points(
            collection_name=collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False,
            score_threshold=score_threshold,
        )
    except Exception:
        logger.exception(
            "search_collection_error",
            collection=collection_name,
        )
        return []

    results = [_scored_point_to_result(point, collection_name) for point in response.points]
    logger.info(
        "search_collection_done",
        collection=collection_name,
        results_count=len(results),
        top_score=results[0].score if results else None,
    )
    return results


# ── Hybrid (multi-collection) search ────────────────────────────────────────


def _deduplicate_results(results: list[SearchResult]) -> list[SearchResult]:
    """Remove duplicate results (same ``source_url`` + ``chunk_index``).

    When a duplicate is found the entry with the higher score is kept.
    """
    seen: dict[tuple[str, int], SearchResult] = {}
    for r in results:
        key = (r.source_url, r.chunk_index)
        # Keep result with higher score for same source_url+chunk_index
        if key not in seen or r.score > seen[key].score:
            seen[key] = r
    return list(seen.values())


async def hybrid_search(
    client: AsyncQdrantClient,
    query_vector: list[float],
    product_query: str | None = None,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float | None = DEFAULT_SCORE_THRESHOLD,
    collections: list[str] | None = None,
) -> list[SearchResult]:
    """Run parallel searches on multiple Qdrant collections and merge.

    By default, searches both ``curated_kb`` and ``auto_crawled``.

    Parameters
    ----------
    client:
        Active :class:`AsyncQdrantClient`.
    query_vector:
        Embedding vector for the user query.
    product_query:
        Optional product-query string for payload pre-filtering.
    top_k:
        Number of results to fetch **per collection** (default 5).
        After merging and deduplication the final list may be longer.
    score_threshold:
        Minimum similarity score; points below are discarded.  ``None`` = keep
        all.
    collections:
        Names of collections to search.  Defaults to
        ``[curated_kb, auto_crawled]``.

    Returns
    -------
    list[SearchResult]
        Merged, deduplicated results sorted by descending score.
    """
    if collections is None:
        collections = [COLLECTION_CURATED_KB, COLLECTION_AUTO_CRAWLED]

    logger.info(
        "hybrid_search_start",
        collections=collections,
        top_k=top_k,
        has_product_filter=product_query is not None,
    )

    # Launch parallel searches via asyncio.gather
    tasks = [
        search_collection(
            client=client,
            collection_name=col,
            query_vector=query_vector,
            top_k=top_k,
            product_query=product_query,
            score_threshold=score_threshold,
        )
        for col in collections
    ]
    per_collection_results = await asyncio.gather(*tasks, return_exceptions=True)

    # Flatten, skipping any tasks that raised an exception
    merged: list[SearchResult] = []
    for idx, result in enumerate(per_collection_results):
        if isinstance(result, BaseException):
            logger.error(
                "hybrid_search_collection_error",
                collection=collections[idx],
                error=str(result),
            )
            continue
        merged.extend(result)

    # Deduplicate (same source_url + chunk_index → keep higher score)
    deduplicated = _deduplicate_results(merged)

    # Sort by score descending
    deduplicated.sort(key=lambda r: r.score, reverse=True)

    logger.info(
        "hybrid_search_done",
        total_results=len(deduplicated),
        from_curated=sum(1 for r in deduplicated if r.is_curated),
        from_auto=sum(1 for r in deduplicated if not r.is_curated),
    )

    return deduplicated


# ── Source-URL based retrieval ───────────────────────────────────────────────


async def scroll_by_source_urls(
    client: AsyncQdrantClient,
    source_urls: list[str],
    *,
    collection: str = COLLECTION_AUTO_CRAWLED,
    limit: int = 50,
) -> list[SearchResult]:
    """Retrieve all chunks matching *source_urls* via Qdrant scroll.

    Unlike vector search this is an exact payload filter — it returns every
    chunk whose ``source_url`` field is in the given list, regardless of
    embedding similarity.  Useful in links-mode where user explicitly
    provided the URLs, so relevance is guaranteed.
    """
    url_filter = Filter(
        must=[
            FieldCondition(
                key="source_url",
                match=MatchAny(any=source_urls),
            ),
        ],
    )

    try:
        records, _next_offset = await client.scroll(
            collection_name=collection,
            scroll_filter=url_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
    except Exception:
        logger.exception("scroll_by_source_urls_error", urls=source_urls[:3])
        return []

    results = [
        SearchResult(
            text=(r.payload or {}).get("text", ""),
            score=1.0,  # exact match → maximum relevance
            source_url=(r.payload or {}).get("source_url", ""),
            source_type=(r.payload or {}).get("source_type", ""),
            is_curated=bool((r.payload or {}).get("is_curated", False)),
            is_sponsored=bool((r.payload or {}).get("is_sponsored", False)),
            collection=collection,
            product_query=(r.payload or {}).get("product_query", ""),
            language=(r.payload or {}).get("language", ""),
            chunk_index=int((r.payload or {}).get("chunk_index", 0)),
            point_id=r.id,
        )
        for r in records
        if (r.payload or {}).get("text")
    ]

    logger.info(
        "scroll_by_source_urls_done",
        requested_urls=len(source_urls),
        results_count=len(results),
    )
    return results
