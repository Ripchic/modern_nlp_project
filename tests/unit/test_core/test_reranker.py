"""Unit tests for reviewmind.core.reranker — reranking logic."""

from __future__ import annotations

import pytest

from reviewmind.core.reranker import (
    CURATED_BOOST,
    DEFAULT_RERANK_TOP_K,
    SPONSORED_DOWNWEIGHT,
    _compute_adjusted_score,
    _deduplicate_by_source,
    rerank,
)
from reviewmind.vectorstore.search import SearchResult

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_result(
    *,
    text: str = "chunk text",
    score: float = 0.8,
    source_url: str = "https://example.com/1",
    source_type: str = "web",
    is_curated: bool = False,
    is_sponsored: bool = False,
    collection: str = "auto_crawled",
    product_query: str = "headphones",
    language: str = "en",
    chunk_index: int = 0,
    point_id: str | int = "p1",
) -> SearchResult:
    """Create a :class:`SearchResult` with sensible defaults."""
    return SearchResult(
        text=text,
        score=score,
        source_url=source_url,
        source_type=source_type,
        is_curated=is_curated,
        is_sponsored=is_sponsored,
        collection=collection,
        product_query=product_query,
        language=language,
        chunk_index=chunk_index,
        point_id=point_id,
    )


# ── TestConstants ────────────────────────────────────────────────────────────


class TestConstants:
    """Verify module-level constants match the PRD requirements."""

    def test_curated_boost_value(self):
        assert CURATED_BOOST == 1.3

    def test_sponsored_downweight_value(self):
        assert SPONSORED_DOWNWEIGHT == 0.7

    def test_default_top_k(self):
        assert DEFAULT_RERANK_TOP_K == 8

    def test_curated_boost_type(self):
        assert isinstance(CURATED_BOOST, float)

    def test_sponsored_downweight_type(self):
        assert isinstance(SPONSORED_DOWNWEIGHT, float)


# ── TestComputeAdjustedScore ─────────────────────────────────────────────────


class TestComputeAdjustedScore:
    """Tests for the _compute_adjusted_score helper."""

    def test_normal_result_no_adjustment(self):
        r = _make_result(score=0.9)
        assert _compute_adjusted_score(r) == pytest.approx(0.9)

    def test_curated_boost_applied(self):
        r = _make_result(score=0.8, is_curated=True)
        assert _compute_adjusted_score(r) == pytest.approx(0.8 * CURATED_BOOST)

    def test_sponsored_downweight_applied(self):
        r = _make_result(score=0.9, is_sponsored=True)
        assert _compute_adjusted_score(r) == pytest.approx(0.9 * SPONSORED_DOWNWEIGHT)

    def test_both_curated_and_sponsored(self):
        r = _make_result(score=1.0, is_curated=True, is_sponsored=True)
        expected = 1.0 * CURATED_BOOST * SPONSORED_DOWNWEIGHT
        assert _compute_adjusted_score(r) == pytest.approx(expected)

    def test_zero_score_stays_zero(self):
        r = _make_result(score=0.0, is_curated=True)
        assert _compute_adjusted_score(r) == pytest.approx(0.0)

    def test_curated_beats_higher_auto(self):
        """Curated (0.8) with boost should beat auto (0.85) without boost."""
        curated = _make_result(score=0.8, is_curated=True)
        auto = _make_result(score=0.85)
        assert _compute_adjusted_score(curated) > _compute_adjusted_score(auto)

    def test_sponsored_drops_below_normal(self):
        """Sponsored (0.9) after downweight < normal (0.7)."""
        sponsored = _make_result(score=0.9, is_sponsored=True)
        normal = _make_result(score=0.7)
        assert _compute_adjusted_score(sponsored) < _compute_adjusted_score(normal)

    def test_negative_score_handled(self):
        """Edge case: negative scores should still apply multipliers."""
        r = _make_result(score=-0.5, is_curated=True)
        assert _compute_adjusted_score(r) == pytest.approx(-0.5 * CURATED_BOOST)


# ── TestDeduplicateBySource ──────────────────────────────────────────────────


class TestDeduplicateBySource:
    """Tests for _deduplicate_by_source helper."""

    def test_no_duplicates_pass_through(self):
        results = [
            _make_result(source_url="u1", chunk_index=0, score=0.9),
            _make_result(source_url="u2", chunk_index=0, score=0.8),
        ]
        deduped = _deduplicate_by_source(results)
        assert len(deduped) == 2

    def test_same_url_different_chunk_kept(self):
        results = [
            _make_result(source_url="u1", chunk_index=0, score=0.9),
            _make_result(source_url="u1", chunk_index=1, score=0.8),
        ]
        deduped = _deduplicate_by_source(results)
        assert len(deduped) == 2

    def test_duplicate_keeps_higher_score(self):
        results = [
            _make_result(source_url="u1", chunk_index=0, score=0.7),
            _make_result(source_url="u1", chunk_index=0, score=0.9),
        ]
        deduped = _deduplicate_by_source(results)
        assert len(deduped) == 1
        assert deduped[0].score == pytest.approx(0.9)

    def test_empty_list(self):
        assert _deduplicate_by_source([]) == []

    def test_single_element(self):
        results = [_make_result(score=0.5)]
        deduped = _deduplicate_by_source(results)
        assert len(deduped) == 1

    def test_multiple_duplicates(self):
        results = [
            _make_result(source_url="u1", chunk_index=0, score=0.5),
            _make_result(source_url="u1", chunk_index=0, score=0.7),
            _make_result(source_url="u1", chunk_index=0, score=0.6),
        ]
        deduped = _deduplicate_by_source(results)
        assert len(deduped) == 1
        assert deduped[0].score == pytest.approx(0.7)


# ── TestRerank ───────────────────────────────────────────────────────────────


class TestRerank:
    """Tests for the main rerank() function."""

    def test_empty_input_returns_empty(self):
        assert rerank([]) == []

    def test_single_result(self):
        results = [_make_result(score=0.8)]
        reranked = rerank(results)
        assert len(reranked) == 1
        # Normal result, no adjustment
        assert reranked[0].score == pytest.approx(0.8)

    def test_curated_promoted_above_auto(self):
        """Curated (0.8) should rank above auto (0.85) after ×1.3 boost."""
        results = [
            _make_result(
                score=0.85,
                source_url="auto_url",
                is_curated=False,
                collection="auto_crawled",
            ),
            _make_result(
                score=0.8,
                source_url="curated_url",
                is_curated=True,
                collection="curated_kb",
            ),
        ]
        reranked = rerank(results)
        assert reranked[0].source_url == "curated_url"
        assert reranked[0].score == pytest.approx(0.8 * CURATED_BOOST)

    def test_sponsored_demoted(self):
        """Sponsored (0.9) should rank below normal (0.75) after ×0.7."""
        results = [
            _make_result(
                score=0.9,
                source_url="sponsored_url",
                is_sponsored=True,
            ),
            _make_result(
                score=0.75,
                source_url="normal_url",
            ),
        ]
        reranked = rerank(results)
        assert reranked[0].source_url == "normal_url"
        # sponsored: 0.9 * 0.7 = 0.63 < 0.75
        assert reranked[1].score == pytest.approx(0.9 * SPONSORED_DOWNWEIGHT)

    def test_top_k_trimming(self):
        """With 15 results and top_k=8, output should be exactly 8."""
        results = [
            _make_result(
                score=0.5 + i * 0.02,
                source_url=f"url_{i}",
                chunk_index=0,
            )
            for i in range(15)
        ]
        reranked = rerank(results, top_k=8)
        assert len(reranked) == 8

    def test_top_k_less_than_input(self):
        """With fewer results than top_k, all results should be returned."""
        results = [_make_result(score=0.8, source_url=f"url_{i}") for i in range(3)]
        reranked = rerank(results, top_k=8)
        assert len(reranked) == 3

    def test_sorted_descending(self):
        results = [
            _make_result(score=0.5, source_url="low"),
            _make_result(score=0.9, source_url="high"),
            _make_result(score=0.7, source_url="mid"),
        ]
        reranked = rerank(results)
        scores = [r.score for r in reranked]
        assert scores == sorted(scores, reverse=True)

    def test_deduplication_in_rerank(self):
        """Duplicates from two collections should be collapsed."""
        results = [
            _make_result(
                score=0.7,
                source_url="same_url",
                chunk_index=0,
                collection="curated_kb",
                is_curated=True,
            ),
            _make_result(
                score=0.8,
                source_url="same_url",
                chunk_index=0,
                collection="auto_crawled",
                is_curated=False,
            ),
        ]
        reranked = rerank(results)
        assert len(reranked) == 1
        # Curated: 0.7 * 1.3 = 0.91 > auto: 0.8 → curated wins
        assert reranked[0].is_curated is True

    def test_scores_are_replaced(self):
        """Original score should be replaced by adjusted score."""
        result = _make_result(score=0.8, is_curated=True)
        reranked = rerank([result])
        assert reranked[0].score == pytest.approx(0.8 * CURATED_BOOST)
        assert reranked[0].score != pytest.approx(0.8)

    def test_original_results_not_mutated(self):
        """rerank should not mutate the input list or its items."""
        original = _make_result(score=0.8, is_curated=True)
        original_score = original.score
        rerank([original])
        assert original.score == pytest.approx(original_score)

    def test_custom_top_k(self):
        results = [_make_result(score=0.5 + i * 0.05, source_url=f"url_{i}") for i in range(10)]
        reranked = rerank(results, top_k=3)
        assert len(reranked) == 3

    def test_all_curated(self):
        """All curated results should all receive the boost."""
        results = [
            _make_result(
                score=0.6 + i * 0.05,
                source_url=f"url_{i}",
                is_curated=True,
            )
            for i in range(5)
        ]
        reranked = rerank(results)
        for r in reranked:
            assert r.is_curated is True

    def test_all_sponsored(self):
        """All sponsored results should all receive the downweight."""
        results = [
            _make_result(
                score=0.8 + i * 0.01,
                source_url=f"url_{i}",
                is_sponsored=True,
            )
            for i in range(5)
        ]
        reranked = rerank(results)
        for r in reranked:
            assert r.is_sponsored is True
            assert r.score < 0.8  # all downweighted

    def test_mixed_results_ordering(self):
        """Verify ordering with a realistic mix of result types."""
        results = [
            _make_result(score=0.85, source_url="auto1"),  # 0.85
            _make_result(score=0.8, source_url="curated1", is_curated=True),  # 1.04
            _make_result(score=0.9, source_url="sponsored1", is_sponsored=True),  # 0.63
            _make_result(score=0.75, source_url="auto2"),  # 0.75
            _make_result(score=0.7, source_url="curated2", is_curated=True),  # 0.91
        ]
        reranked = rerank(results)
        urls = [r.source_url for r in reranked]
        assert urls[0] == "curated1"  # 1.04
        assert urls[1] == "curated2"  # 0.91
        assert urls[2] == "auto1"  # 0.85

    def test_top_k_one(self):
        results = [
            _make_result(score=0.9, source_url="u1"),
            _make_result(score=0.8, source_url="u2"),
        ]
        reranked = rerank(results, top_k=1)
        assert len(reranked) == 1
        assert reranked[0].source_url == "u1"


# ── TestNoAutoResults ────────────────────────────────────────────────────────


class TestNoCuratedResults:
    """When curated_kb is empty, only auto_crawled results should be used."""

    def test_only_auto_results(self):
        results = [
            _make_result(
                score=0.8,
                source_url="auto1",
                collection="auto_crawled",
                is_curated=False,
            ),
            _make_result(
                score=0.75,
                source_url="auto2",
                collection="auto_crawled",
                is_curated=False,
            ),
        ]
        reranked = rerank(results)
        assert len(reranked) == 2
        # Scores unchanged (no curated/sponsored flags)
        assert reranked[0].score == pytest.approx(0.8)
        assert reranked[1].score == pytest.approx(0.75)


class TestNoAutoResults:
    """When auto_crawled is empty, only curated_kb results should be used."""

    def test_only_curated_results(self):
        results = [
            _make_result(
                score=0.7,
                source_url="cur1",
                collection="curated_kb",
                is_curated=True,
            ),
            _make_result(
                score=0.65,
                source_url="cur2",
                collection="curated_kb",
                is_curated=True,
            ),
        ]
        reranked = rerank(results)
        assert len(reranked) == 2
        # Both boosted
        assert reranked[0].score == pytest.approx(0.7 * CURATED_BOOST)
        assert reranked[1].score == pytest.approx(0.65 * CURATED_BOOST)


# ── TestEdgeCases ────────────────────────────────────────────────────────────


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_top_k_zero(self):
        results = [_make_result(score=0.9)]
        reranked = rerank(results, top_k=0)
        assert reranked == []

    def test_very_large_top_k(self):
        results = [_make_result(score=0.5, source_url=f"u{i}") for i in range(3)]
        reranked = rerank(results, top_k=1000)
        assert len(reranked) == 3

    def test_all_same_score(self):
        results = [_make_result(score=0.8, source_url=f"url_{i}") for i in range(5)]
        reranked = rerank(results)
        assert len(reranked) == 5
        for r in reranked:
            assert r.score == pytest.approx(0.8)

    def test_metadata_preserved(self):
        """Non-score fields should be preserved after reranking."""
        r = _make_result(
            text="hello world",
            score=0.8,
            source_url="https://example.com",
            source_type="youtube",
            is_curated=True,
            product_query="test",
            language="ru",
            chunk_index=3,
            point_id="abc",
        )
        reranked = rerank([r])
        result = reranked[0]
        assert result.text == "hello world"
        assert result.source_url == "https://example.com"
        assert result.source_type == "youtube"
        assert result.is_curated is True
        assert result.product_query == "test"
        assert result.language == "ru"
        assert result.chunk_index == 3
        assert result.point_id == "abc"

    def test_duplicate_from_both_collections_curated_wins(self):
        """Same source in both collections — curated with boost should win."""
        results = [
            _make_result(
                score=0.75,
                source_url="shared_url",
                chunk_index=0,
                collection="auto_crawled",
                is_curated=False,
            ),
            _make_result(
                score=0.7,
                source_url="shared_url",
                chunk_index=0,
                collection="curated_kb",
                is_curated=True,
            ),
        ]
        reranked = rerank(results)
        assert len(reranked) == 1
        # curated: 0.7 * 1.3 = 0.91 > auto: 0.75
        assert reranked[0].collection == "curated_kb"
        assert reranked[0].score == pytest.approx(0.7 * CURATED_BOOST)


# ── TestIntegration ──────────────────────────────────────────────────────────


class TestIntegrationScenarios:
    """End-to-end integration scenarios."""

    def test_prd_example_curated_beats_auto(self):
        """PRD test step 1: curated(0.8) vs auto(0.85) → curated > auto."""
        results = [
            _make_result(score=0.85, source_url="auto", is_curated=False),
            _make_result(score=0.8, source_url="curated", is_curated=True),
        ]
        reranked = rerank(results)
        assert reranked[0].source_url == "curated"
        assert reranked[0].score == pytest.approx(0.8 * 1.3)  # 1.04

    def test_prd_example_sponsored_demoted(self):
        """PRD test step 2: sponsored(0.9) → 0.63 after downweight."""
        results = [_make_result(score=0.9, source_url="s1", is_sponsored=True)]
        reranked = rerank(results)
        assert reranked[0].score == pytest.approx(0.9 * 0.7)  # 0.63

    def test_prd_example_top_k_trim(self):
        """PRD test step 3: 15 results with top_k=8 → exactly 8."""
        results = [_make_result(score=0.5 + i * 0.03, source_url=f"url_{i}") for i in range(15)]
        reranked = rerank(results, top_k=8)
        assert len(reranked) == 8

    def test_prd_example_empty_list(self):
        """PRD test step 4: empty list → empty list."""
        assert rerank([]) == []

    def test_realistic_mixed_collection(self):
        """Realistic scenario with mixed curated, sponsored, and normal results."""
        results = [
            # Curated editorial reviews
            _make_result(
                score=0.82,
                source_url="wirecutter.com/headphones",
                source_type="web",
                is_curated=True,
                collection="curated_kb",
                chunk_index=0,
            ),
            _make_result(
                score=0.78,
                source_url="rtings.com/headphones",
                source_type="web",
                is_curated=True,
                collection="curated_kb",
                chunk_index=0,
            ),
            # Normal auto-crawled
            _make_result(
                score=0.88,
                source_url="youtube.com/watch?v=abc",
                source_type="youtube",
                is_curated=False,
                collection="auto_crawled",
                chunk_index=0,
            ),
            _make_result(
                score=0.85,
                source_url="reddit.com/r/headphones/xyz",
                source_type="reddit",
                is_curated=False,
                collection="auto_crawled",
                chunk_index=0,
            ),
            # Sponsored content
            _make_result(
                score=0.92,
                source_url="youtube.com/watch?v=sponsor",
                source_type="youtube",
                is_sponsored=True,
                collection="auto_crawled",
                chunk_index=0,
            ),
        ]
        reranked = rerank(results, top_k=5)
        assert len(reranked) == 5

        # Curated wirecutter: 0.82 * 1.3 = 1.066 → should be first
        assert reranked[0].source_url == "wirecutter.com/headphones"
        # Curated rtings: 0.78 * 1.3 = 1.014 → second
        assert reranked[1].source_url == "rtings.com/headphones"
        # YouTube auto: 0.88 → third
        assert reranked[2].source_url == "youtube.com/watch?v=abc"
        # Reddit auto: 0.85 → fourth
        assert reranked[3].source_url == "reddit.com/r/headphones/xyz"
        # Sponsored: 0.92 * 0.7 = 0.644 → last
        assert reranked[4].source_url == "youtube.com/watch?v=sponsor"
