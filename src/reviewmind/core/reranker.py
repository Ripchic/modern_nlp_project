"""reviewmind/core/reranker.py — Reranking logic for search results.

Applies score adjustments to search results:
- Curated chunks receive a **×1.3** boost (trusted editorial sources).
- Sponsored chunks receive a **×0.7** downweight (potential bias).

After adjustment the results are merged, deduplicated, sorted by the new
*adjusted* score, and trimmed to *top_k*.
"""

from __future__ import annotations

from dataclasses import replace

import structlog

from reviewmind.vectorstore.search import SearchResult

logger = structlog.get_logger("reviewmind.core.reranker")

# ── Constants ────────────────────────────────────────────────────────────────

#: Score multiplier applied to curated (editorial) results.
CURATED_BOOST: float = 1.3

#: Score multiplier applied to sponsored results.
SPONSORED_DOWNWEIGHT: float = 0.7

#: Default number of results returned after reranking.
DEFAULT_RERANK_TOP_K: int = 8


# ── Helpers ──────────────────────────────────────────────────────────────────


def _compute_adjusted_score(result: SearchResult) -> float:
    """Return the adjusted score for a single :class:`SearchResult`.

    Rules (applied cumulatively):
    * ``is_curated`` → score × :data:`CURATED_BOOST`
    * ``is_sponsored`` → score × :data:`SPONSORED_DOWNWEIGHT`
    * Both flags set → both multipliers are applied.

    Parameters
    ----------
    result:
        The search result to score.

    Returns
    -------
    float
        Adjusted score.
    """
    score = result.score
    if result.is_curated:
        score *= CURATED_BOOST
    if result.is_sponsored:
        score *= SPONSORED_DOWNWEIGHT
    return score


def _deduplicate_by_source(results: list[SearchResult]) -> list[SearchResult]:
    """Remove duplicate results sharing the same ``source_url`` + ``chunk_index``.

    When a duplicate is found the entry with the **higher adjusted score** is
    kept.
    """
    seen: dict[tuple[str, int], SearchResult] = {}
    for r in results:
        key = (r.source_url, r.chunk_index)
        if key not in seen or r.score > seen[key].score:
            seen[key] = r
    return list(seen.values())


# ── Public API ───────────────────────────────────────────────────────────────


def rerank(
    search_results: list[SearchResult],
    top_k: int = DEFAULT_RERANK_TOP_K,
) -> list[SearchResult]:
    """Rerank search results with curated/sponsored adjustments.

    Workflow:
    1. Compute an *adjusted* score for every result (curated ×1.3, sponsored ×0.7).
    2. Replace ``result.score`` with the adjusted value.
    3. Deduplicate by ``(source_url, chunk_index)`` — keep higher score.
    4. Sort by adjusted score descending.
    5. Return the top *top_k* results.

    Parameters
    ----------
    search_results:
        Raw results from :func:`hybrid_search` (or any list of
        :class:`SearchResult`).
    top_k:
        Maximum number of results to return (default 8).

    Returns
    -------
    list[SearchResult]
        Reranked, deduplicated, trimmed list of results.
    """
    if not search_results:
        logger.debug("rerank_empty_input")
        return []

    # 1. Compute adjusted scores and create new result copies
    adjusted: list[SearchResult] = []
    for result in search_results:
        new_score = _compute_adjusted_score(result)
        adjusted.append(replace(result, score=new_score))

    # 2. Deduplicate (same source_url + chunk_index → keep higher score)
    deduplicated = _deduplicate_by_source(adjusted)

    # 3. Sort by adjusted score descending
    deduplicated.sort(key=lambda r: r.score, reverse=True)

    # 4. Trim to top_k
    trimmed = deduplicated[:top_k]

    curated_count = sum(1 for r in trimmed if r.is_curated)
    sponsored_count = sum(1 for r in trimmed if r.is_sponsored)

    logger.info(
        "rerank_done",
        input_count=len(search_results),
        after_dedup=len(deduplicated),
        output_count=len(trimmed),
        top_k=top_k,
        curated_count=curated_count,
        sponsored_count=sponsored_count,
        top_score=trimmed[0].score if trimmed else None,
    )

    return trimmed
