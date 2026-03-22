"""Unit tests for reviewmind.vectorstore.search — hybrid search across collections."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    ScoredPoint,
)

from reviewmind.vectorstore.collections import (
    COLLECTION_AUTO_CRAWLED,
    COLLECTION_CURATED_KB,
)
from reviewmind.vectorstore.search import (
    DEFAULT_SCORE_THRESHOLD,
    DEFAULT_TOP_K,
    SearchResult,
    _build_product_filter,
    _deduplicate_results,
    _scored_point_to_result,
    hybrid_search,
    search_collection,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _make_scored_point(
    point_id: str | int = "pt-1",
    score: float = 0.9,
    payload: dict | None = None,
) -> ScoredPoint:
    """Create a ScoredPoint for testing."""
    if payload is None:
        payload = {
            "text": "Test chunk text",
            "source_url": "https://example.com/review",
            "source_type": "web",
            "is_curated": False,
            "is_sponsored": False,
            "product_query": "test product",
            "language": "en",
            "chunk_index": 0,
        }
    return ScoredPoint(id=point_id, version=1, score=score, payload=payload)


def _make_query_response(points: list[ScoredPoint] | None = None):
    """Create a mock QueryResponse with a .points attribute."""
    resp = MagicMock()
    resp.points = points or []
    return resp


DUMMY_VECTOR = [0.1] * 1536


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════


class TestConstants:
    """Test module-level constants."""

    def test_default_top_k(self):
        assert DEFAULT_TOP_K == 5

    def test_default_score_threshold(self):
        assert DEFAULT_SCORE_THRESHOLD is None

    def test_default_top_k_is_int(self):
        assert isinstance(DEFAULT_TOP_K, int)


# ═══════════════════════════════════════════════════════════════════════════════
# SearchResult dataclass
# ═══════════════════════════════════════════════════════════════════════════════


class TestSearchResult:
    """Test SearchResult dataclass."""

    def test_create_minimal(self):
        r = SearchResult(text="hello", score=0.85)
        assert r.text == "hello"
        assert r.score == 0.85
        assert r.source_url == ""
        assert r.source_type == ""
        assert r.is_curated is False
        assert r.is_sponsored is False
        assert r.collection == ""
        assert r.product_query == ""
        assert r.language == ""
        assert r.chunk_index == 0
        assert r.point_id == ""
        assert r.extra == {}

    def test_create_full(self):
        r = SearchResult(
            text="Sony review",
            score=0.95,
            source_url="https://example.com",
            source_type="youtube",
            is_curated=True,
            is_sponsored=False,
            collection="curated_kb",
            product_query="Sony WH-1000XM5",
            language="ru",
            chunk_index=3,
            point_id="abc-123",
            extra={"author": "John"},
        )
        assert r.text == "Sony review"
        assert r.score == 0.95
        assert r.source_type == "youtube"
        assert r.is_curated is True
        assert r.collection == "curated_kb"
        assert r.chunk_index == 3
        assert r.extra == {"author": "John"}

    def test_extra_defaults_to_empty_dict(self):
        r1 = SearchResult(text="a", score=0.5)
        r2 = SearchResult(text="b", score=0.5)
        # Each instance should have its own dict (no shared mutable default)
        r1.extra["key"] = "val"
        assert "key" not in r2.extra

    def test_score_is_float(self):
        r = SearchResult(text="t", score=0.77)
        assert isinstance(r.score, float)


# ═══════════════════════════════════════════════════════════════════════════════
# _build_product_filter
# ═══════════════════════════════════════════════════════════════════════════════


class TestBuildProductFilter:
    """Test _build_product_filter helper."""

    def test_returns_filter(self):
        f = _build_product_filter("Sony XM5")
        assert isinstance(f, Filter)

    def test_filter_has_must_condition(self):
        f = _build_product_filter("Sony XM5")
        assert f.must is not None
        assert len(f.must) == 1

    def test_filter_condition_key(self):
        f = _build_product_filter("AirPods")
        cond = f.must[0]
        assert isinstance(cond, FieldCondition)
        assert cond.key == "product_query"

    def test_filter_condition_value(self):
        f = _build_product_filter("iPhone 16 Pro")
        cond = f.must[0]
        assert isinstance(cond.match, MatchValue)
        assert cond.match.value == "iPhone 16 Pro"

    def test_empty_string_product(self):
        f = _build_product_filter("")
        cond = f.must[0]
        assert cond.match.value == ""


# ═══════════════════════════════════════════════════════════════════════════════
# _scored_point_to_result
# ═══════════════════════════════════════════════════════════════════════════════


class TestScoredPointToResult:
    """Test conversion from ScoredPoint to SearchResult."""

    def test_basic_conversion(self):
        pt = _make_scored_point(point_id="p1", score=0.91)
        r = _scored_point_to_result(pt, "auto_crawled")
        assert r.text == "Test chunk text"
        assert r.score == 0.91
        assert r.source_url == "https://example.com/review"
        assert r.source_type == "web"
        assert r.is_curated is False
        assert r.is_sponsored is False
        assert r.collection == "auto_crawled"
        assert r.product_query == "test product"
        assert r.language == "en"
        assert r.chunk_index == 0
        assert r.point_id == "p1"

    def test_curated_flag(self):
        pt = _make_scored_point(
            payload={
                "text": "curated text",
                "source_url": "https://curated.com",
                "source_type": "curated",
                "is_curated": True,
                "is_sponsored": False,
                "product_query": "headphones",
                "language": "ru",
                "chunk_index": 2,
            }
        )
        r = _scored_point_to_result(pt, "curated_kb")
        assert r.is_curated is True
        assert r.collection == "curated_kb"

    def test_sponsored_flag(self):
        pt = _make_scored_point(
            payload={
                "text": "sponsored text",
                "source_url": "https://sponsor.com",
                "source_type": "youtube",
                "is_curated": False,
                "is_sponsored": True,
                "product_query": "gadget",
                "language": "en",
                "chunk_index": 0,
            }
        )
        r = _scored_point_to_result(pt, "auto_crawled")
        assert r.is_sponsored is True

    def test_extra_payload_fields(self):
        pt = _make_scored_point(
            payload={
                "text": "text",
                "source_url": "url",
                "author": "Jane",
                "date": "2026-01-01",
                "custom_field": 42,
            }
        )
        r = _scored_point_to_result(pt, "auto_crawled")
        assert r.extra == {"author": "Jane", "date": "2026-01-01", "custom_field": 42}

    def test_missing_payload_fields_use_defaults(self):
        pt = _make_scored_point(payload={})
        r = _scored_point_to_result(pt, "auto_crawled")
        assert r.text == ""
        assert r.source_url == ""
        assert r.source_type == ""
        assert r.is_curated is False
        assert r.is_sponsored is False
        assert r.product_query == ""
        assert r.language == ""
        assert r.chunk_index == 0

    def test_none_payload(self):
        pt = ScoredPoint(id="p1", version=1, score=0.5, payload=None)
        r = _scored_point_to_result(pt, "auto_crawled")
        assert r.text == ""
        assert r.score == 0.5

    def test_point_id_int(self):
        pt = _make_scored_point(point_id=42)
        r = _scored_point_to_result(pt, "auto_crawled")
        assert r.point_id == 42

    def test_point_id_string(self):
        pt = _make_scored_point(point_id="uuid-abc-123")
        r = _scored_point_to_result(pt, "auto_crawled")
        assert r.point_id == "uuid-abc-123"


# ═══════════════════════════════════════════════════════════════════════════════
# _deduplicate_results
# ═══════════════════════════════════════════════════════════════════════════════


class TestDeduplicateResults:
    """Test deduplication logic."""

    def test_no_duplicates(self):
        results = [
            SearchResult(text="a", score=0.9, source_url="url1", chunk_index=0),
            SearchResult(text="b", score=0.8, source_url="url2", chunk_index=0),
        ]
        deduped = _deduplicate_results(results)
        assert len(deduped) == 2

    def test_duplicate_keeps_higher_score(self):
        results = [
            SearchResult(text="a", score=0.7, source_url="url1", chunk_index=0),
            SearchResult(text="a-better", score=0.95, source_url="url1", chunk_index=0),
        ]
        deduped = _deduplicate_results(results)
        assert len(deduped) == 1
        assert deduped[0].score == 0.95
        assert deduped[0].text == "a-better"

    def test_same_url_different_chunk_index_not_duplicate(self):
        results = [
            SearchResult(text="a", score=0.9, source_url="url1", chunk_index=0),
            SearchResult(text="b", score=0.8, source_url="url1", chunk_index=1),
        ]
        deduped = _deduplicate_results(results)
        assert len(deduped) == 2

    def test_empty_list(self):
        assert _deduplicate_results([]) == []

    def test_single_item(self):
        results = [SearchResult(text="a", score=0.9, source_url="url1", chunk_index=0)]
        deduped = _deduplicate_results(results)
        assert len(deduped) == 1

    def test_multiple_duplicates(self):
        results = [
            SearchResult(text="a", score=0.7, source_url="url1", chunk_index=0),
            SearchResult(text="b", score=0.8, source_url="url1", chunk_index=0),
            SearchResult(text="c", score=0.9, source_url="url1", chunk_index=0),
        ]
        deduped = _deduplicate_results(results)
        assert len(deduped) == 1
        assert deduped[0].score == 0.9

    def test_preserves_non_duplicates_with_duplicates(self):
        results = [
            SearchResult(text="a1", score=0.7, source_url="url1", chunk_index=0),
            SearchResult(text="a2", score=0.9, source_url="url1", chunk_index=0),
            SearchResult(text="b1", score=0.85, source_url="url2", chunk_index=0),
        ]
        deduped = _deduplicate_results(results)
        assert len(deduped) == 2
        scores = {r.source_url: r.score for r in deduped}
        assert scores["url1"] == 0.9
        assert scores["url2"] == 0.85


# ═══════════════════════════════════════════════════════════════════════════════
# search_collection
# ═══════════════════════════════════════════════════════════════════════════════


class TestSearchCollection:
    """Test single-collection search."""

    @pytest.mark.asyncio
    async def test_basic_search(self):
        client = AsyncMock()
        client.query_points.return_value = _make_query_response(
            [
                _make_scored_point("p1", 0.95),
                _make_scored_point("p2", 0.85),
            ]
        )

        results = await search_collection(
            client=client,
            collection_name="auto_crawled",
            query_vector=DUMMY_VECTOR,
        )
        assert len(results) == 2
        assert results[0].score == 0.95
        assert results[1].score == 0.85

    @pytest.mark.asyncio
    async def test_search_calls_query_points(self):
        client = AsyncMock()
        client.query_points.return_value = _make_query_response([])

        await search_collection(
            client=client,
            collection_name="curated_kb",
            query_vector=DUMMY_VECTOR,
            top_k=3,
        )
        client.query_points.assert_called_once()
        call_kwargs = client.query_points.call_args.kwargs
        assert call_kwargs["collection_name"] == "curated_kb"
        assert call_kwargs["query"] == DUMMY_VECTOR
        assert call_kwargs["limit"] == 3
        assert call_kwargs["with_payload"] is True
        assert call_kwargs["with_vectors"] is False

    @pytest.mark.asyncio
    async def test_search_with_product_filter(self):
        client = AsyncMock()
        client.query_points.return_value = _make_query_response([])

        await search_collection(
            client=client,
            collection_name="auto_crawled",
            query_vector=DUMMY_VECTOR,
            product_query="Sony XM5",
        )
        call_kwargs = client.query_points.call_args.kwargs
        query_filter = call_kwargs["query_filter"]
        assert query_filter is not None
        assert isinstance(query_filter, Filter)
        assert query_filter.must[0].key == "product_query"
        assert query_filter.must[0].match.value == "Sony XM5"

    @pytest.mark.asyncio
    async def test_search_without_product_filter(self):
        client = AsyncMock()
        client.query_points.return_value = _make_query_response([])

        await search_collection(
            client=client,
            collection_name="auto_crawled",
            query_vector=DUMMY_VECTOR,
            product_query=None,
        )
        call_kwargs = client.query_points.call_args.kwargs
        assert call_kwargs["query_filter"] is None

    @pytest.mark.asyncio
    async def test_search_with_score_threshold(self):
        client = AsyncMock()
        client.query_points.return_value = _make_query_response([])

        await search_collection(
            client=client,
            collection_name="curated_kb",
            query_vector=DUMMY_VECTOR,
            score_threshold=0.75,
        )
        call_kwargs = client.query_points.call_args.kwargs
        assert call_kwargs["score_threshold"] == 0.75

    @pytest.mark.asyncio
    async def test_search_empty_results(self):
        client = AsyncMock()
        client.query_points.return_value = _make_query_response([])

        results = await search_collection(
            client=client,
            collection_name="auto_crawled",
            query_vector=DUMMY_VECTOR,
        )
        assert results == []

    @pytest.mark.asyncio
    async def test_search_error_returns_empty(self):
        """When Qdrant raises an error, return empty list (don't crash)."""
        client = AsyncMock()
        client.query_points.side_effect = Exception("connection refused")

        results = await search_collection(
            client=client,
            collection_name="auto_crawled",
            query_vector=DUMMY_VECTOR,
        )
        assert results == []

    @pytest.mark.asyncio
    async def test_search_sets_collection_on_results(self):
        client = AsyncMock()
        client.query_points.return_value = _make_query_response(
            [
                _make_scored_point("p1", 0.9),
            ]
        )

        results = await search_collection(
            client=client,
            collection_name="curated_kb",
            query_vector=DUMMY_VECTOR,
        )
        assert results[0].collection == "curated_kb"

    @pytest.mark.asyncio
    async def test_search_custom_top_k(self):
        client = AsyncMock()
        client.query_points.return_value = _make_query_response([])

        await search_collection(
            client=client,
            collection_name="auto_crawled",
            query_vector=DUMMY_VECTOR,
            top_k=10,
        )
        call_kwargs = client.query_points.call_args.kwargs
        assert call_kwargs["limit"] == 10


# ═══════════════════════════════════════════════════════════════════════════════
# hybrid_search
# ═══════════════════════════════════════════════════════════════════════════════


class TestHybridSearch:
    """Test hybrid multi-collection search."""

    @pytest.mark.asyncio
    async def test_default_collections(self):
        """By default, searches curated_kb and auto_crawled."""
        client = AsyncMock()
        client.query_points.return_value = _make_query_response([])

        await hybrid_search(client=client, query_vector=DUMMY_VECTOR)
        assert client.query_points.call_count == 2
        collection_names = [call.kwargs["collection_name"] for call in client.query_points.call_args_list]
        assert COLLECTION_CURATED_KB in collection_names
        assert COLLECTION_AUTO_CRAWLED in collection_names

    @pytest.mark.asyncio
    async def test_custom_collections(self):
        client = AsyncMock()
        client.query_points.return_value = _make_query_response([])

        await hybrid_search(
            client=client,
            query_vector=DUMMY_VECTOR,
            collections=["my_col"],
        )
        assert client.query_points.call_count == 1
        assert client.query_points.call_args.kwargs["collection_name"] == "my_col"

    @pytest.mark.asyncio
    async def test_merges_results_from_both_collections(self):
        """Results from both collections are merged into one list."""
        curated_point = _make_scored_point(
            "c1",
            0.92,
            payload={
                "text": "curated text",
                "source_url": "https://curated.com",
                "source_type": "curated",
                "is_curated": True,
                "is_sponsored": False,
                "product_query": "headphones",
                "language": "ru",
                "chunk_index": 0,
            },
        )
        auto_point = _make_scored_point(
            "a1",
            0.88,
            payload={
                "text": "auto text",
                "source_url": "https://auto.com",
                "source_type": "youtube",
                "is_curated": False,
                "is_sponsored": False,
                "product_query": "headphones",
                "language": "en",
                "chunk_index": 0,
            },
        )

        client = AsyncMock()
        # Return different results for different collections
        call_count = 0

        async def mock_query_points(**kwargs):
            nonlocal call_count
            call_count += 1
            if kwargs["collection_name"] == COLLECTION_CURATED_KB:
                return _make_query_response([curated_point])
            return _make_query_response([auto_point])

        client.query_points = mock_query_points

        results = await hybrid_search(client=client, query_vector=DUMMY_VECTOR)
        assert len(results) == 2
        # Should be sorted by score descending
        assert results[0].score >= results[1].score

    @pytest.mark.asyncio
    async def test_results_sorted_by_score_descending(self):
        points = [
            _make_scored_point("p1", 0.7, payload={"text": "low", "source_url": "u1", "chunk_index": 0}),
            _make_scored_point("p2", 0.95, payload={"text": "high", "source_url": "u2", "chunk_index": 0}),
            _make_scored_point("p3", 0.85, payload={"text": "mid", "source_url": "u3", "chunk_index": 0}),
        ]

        client = AsyncMock()
        client.query_points.return_value = _make_query_response(points)

        results = await hybrid_search(
            client=client,
            query_vector=DUMMY_VECTOR,
            collections=["test_col"],
        )
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_deduplication_across_collections(self):
        """Same source_url + chunk_index from both collections → keep higher score."""
        shared_payload = {
            "text": "shared chunk",
            "source_url": "https://shared.com",
            "source_type": "web",
            "is_curated": False,
            "is_sponsored": False,
            "chunk_index": 0,
        }
        p_curated = _make_scored_point("c1", 0.92, payload={**shared_payload, "is_curated": True})
        p_auto = _make_scored_point("a1", 0.88, payload=shared_payload)

        client = AsyncMock()
        call_idx = 0

        async def mock_query(**kwargs):
            nonlocal call_idx
            call_idx += 1
            if kwargs["collection_name"] == COLLECTION_CURATED_KB:
                return _make_query_response([p_curated])
            return _make_query_response([p_auto])

        client.query_points = mock_query

        results = await hybrid_search(client=client, query_vector=DUMMY_VECTOR)
        assert len(results) == 1
        assert results[0].score == 0.92  # Higher score kept

    @pytest.mark.asyncio
    async def test_one_collection_error_returns_other(self):
        """If one collection search fails, return results from the other."""
        auto_point = _make_scored_point(
            "a1",
            0.88,
            payload={
                "text": "auto text",
                "source_url": "https://auto.com",
                "chunk_index": 0,
            },
        )

        client = AsyncMock()

        async def mock_query(**kwargs):
            if kwargs["collection_name"] == COLLECTION_CURATED_KB:
                raise ConnectionError("curated_kb unavailable")
            return _make_query_response([auto_point])

        client.query_points = mock_query

        results = await hybrid_search(client=client, query_vector=DUMMY_VECTOR)
        assert len(results) == 1
        assert results[0].text == "auto text"

    @pytest.mark.asyncio
    async def test_both_collections_empty(self):
        client = AsyncMock()
        client.query_points.return_value = _make_query_response([])

        results = await hybrid_search(client=client, query_vector=DUMMY_VECTOR)
        assert results == []

    @pytest.mark.asyncio
    async def test_both_collections_error(self):
        client = AsyncMock()
        client.query_points.side_effect = Exception("total failure")

        results = await hybrid_search(client=client, query_vector=DUMMY_VECTOR)
        assert results == []

    @pytest.mark.asyncio
    async def test_product_query_filter_passed(self):
        client = AsyncMock()
        client.query_points.return_value = _make_query_response([])

        await hybrid_search(
            client=client,
            query_vector=DUMMY_VECTOR,
            product_query="Sony XM5",
        )
        for call in client.query_points.call_args_list:
            query_filter = call.kwargs["query_filter"]
            assert query_filter is not None
            assert query_filter.must[0].match.value == "Sony XM5"

    @pytest.mark.asyncio
    async def test_top_k_passed_to_each_collection(self):
        client = AsyncMock()
        client.query_points.return_value = _make_query_response([])

        await hybrid_search(
            client=client,
            query_vector=DUMMY_VECTOR,
            top_k=10,
        )
        for call in client.query_points.call_args_list:
            assert call.kwargs["limit"] == 10

    @pytest.mark.asyncio
    async def test_score_threshold_passed(self):
        client = AsyncMock()
        client.query_points.return_value = _make_query_response([])

        await hybrid_search(
            client=client,
            query_vector=DUMMY_VECTOR,
            score_threshold=0.8,
        )
        for call in client.query_points.call_args_list:
            assert call.kwargs["score_threshold"] == 0.8

    @pytest.mark.asyncio
    async def test_parallel_execution(self):
        """Verify that searches run in parallel (asyncio.gather)."""
        order = []

        async def mock_query(**kwargs):
            col = kwargs["collection_name"]
            order.append(f"start_{col}")
            await asyncio.sleep(0.01)
            order.append(f"end_{col}")
            return _make_query_response([])

        client = AsyncMock()
        client.query_points = mock_query

        await hybrid_search(client=client, query_vector=DUMMY_VECTOR)
        # Both should start before either ends (parallel execution)
        start_indices = [i for i, o in enumerate(order) if o.startswith("start_")]
        end_indices = [i for i, o in enumerate(order) if o.startswith("end_")]
        # All starts should come before any end (parallel)
        assert max(start_indices) < min(end_indices)

    @pytest.mark.asyncio
    async def test_many_results_from_multiple_collections(self):
        """Handling large result sets from multiple collections."""
        curated_points = [
            _make_scored_point(
                f"c{i}",
                0.9 - i * 0.01,
                payload={
                    "text": f"curated_{i}",
                    "source_url": f"https://curated.com/{i}",
                    "chunk_index": i,
                },
            )
            for i in range(5)
        ]
        auto_points = [
            _make_scored_point(
                f"a{i}",
                0.88 - i * 0.01,
                payload={
                    "text": f"auto_{i}",
                    "source_url": f"https://auto.com/{i}",
                    "chunk_index": i,
                },
            )
            for i in range(5)
        ]

        client = AsyncMock()

        async def mock_query(**kwargs):
            if kwargs["collection_name"] == COLLECTION_CURATED_KB:
                return _make_query_response(curated_points)
            return _make_query_response(auto_points)

        client.query_points = mock_query

        results = await hybrid_search(client=client, query_vector=DUMMY_VECTOR)
        assert len(results) == 10
        # Check sorted order
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score


# ═══════════════════════════════════════════════════════════════════════════════
# Vectorstore __init__ exports
# ═══════════════════════════════════════════════════════════════════════════════


class TestSearchExports:
    """Verify search symbols are exported from vectorstore package."""

    def test_search_result_exported(self):
        from reviewmind.vectorstore import SearchResult

        assert SearchResult is not None

    def test_hybrid_search_exported(self):
        from reviewmind.vectorstore import hybrid_search

        assert callable(hybrid_search)

    def test_search_collection_exported(self):
        from reviewmind.vectorstore import search_collection

        assert callable(search_collection)

    def test_default_top_k_exported(self):
        from reviewmind.vectorstore import DEFAULT_TOP_K

        assert DEFAULT_TOP_K == 5

    def test_default_score_threshold_exported(self):
        from reviewmind.vectorstore import DEFAULT_SCORE_THRESHOLD

        assert DEFAULT_SCORE_THRESHOLD is None


# ═══════════════════════════════════════════════════════════════════════════════
# Integration-like scenarios (all mocked, but full workflow)
# ═══════════════════════════════════════════════════════════════════════════════


class TestIntegrationScenarios:
    """Scenario-based tests simulating realistic search flows."""

    @pytest.mark.asyncio
    async def test_curated_prioritized_over_auto_same_score(self):
        """Same score from both collections: both present in results."""
        curated_point = _make_scored_point(
            "c1",
            0.90,
            payload={
                "text": "curated review",
                "source_url": "https://wirecutter.com/review",
                "source_type": "curated",
                "is_curated": True,
                "is_sponsored": False,
                "chunk_index": 0,
            },
        )
        auto_point = _make_scored_point(
            "a1",
            0.90,
            payload={
                "text": "auto review",
                "source_url": "https://youtube.com/watch?v=abc",
                "source_type": "youtube",
                "is_curated": False,
                "is_sponsored": False,
                "chunk_index": 0,
            },
        )

        client = AsyncMock()

        async def mock_query(**kwargs):
            if kwargs["collection_name"] == COLLECTION_CURATED_KB:
                return _make_query_response([curated_point])
            return _make_query_response([auto_point])

        client.query_points = mock_query

        results = await hybrid_search(client=client, query_vector=DUMMY_VECTOR)
        assert len(results) == 2
        # Both have same score, but both should be present
        curated_found = any(r.is_curated for r in results)
        auto_found = any(not r.is_curated for r in results)
        assert curated_found and auto_found

    @pytest.mark.asyncio
    async def test_search_with_sponsored_content(self):
        """Verify sponsored flag is preserved through the pipeline."""
        sponsored_point = _make_scored_point(
            "s1",
            0.88,
            payload={
                "text": "amazing product sponsored review",
                "source_url": "https://youtube.com/watch?v=sponsored",
                "source_type": "youtube",
                "is_curated": False,
                "is_sponsored": True,
                "product_query": "Sony XM5",
                "chunk_index": 0,
            },
        )

        client = AsyncMock()
        client.query_points.return_value = _make_query_response([sponsored_point])

        results = await hybrid_search(
            client=client,
            query_vector=DUMMY_VECTOR,
            collections=["auto_crawled"],
        )
        assert len(results) == 1
        assert results[0].is_sponsored is True

    @pytest.mark.asyncio
    async def test_full_workflow_product_query_filter(self):
        """Full workflow: hybrid search with product filter, results from both collections."""
        curated_point = _make_scored_point(
            "c1",
            0.92,
            payload={
                "text": "Sony XM5 are excellent noise cancelling headphones",
                "source_url": "https://wirecutter.com/best-headphones",
                "source_type": "web",
                "is_curated": True,
                "is_sponsored": False,
                "product_query": "Sony WH-1000XM5",
                "language": "en",
                "chunk_index": 0,
            },
        )
        auto_point = _make_scored_point(
            "a1",
            0.85,
            payload={
                "text": "Bought XM5 last week, amazing ANC",
                "source_url": "https://reddit.com/r/headphones/12345",
                "source_type": "reddit",
                "is_curated": False,
                "is_sponsored": False,
                "product_query": "Sony WH-1000XM5",
                "language": "en",
                "chunk_index": 0,
            },
        )

        client = AsyncMock()

        async def mock_query(**kwargs):
            # Verify filter is applied
            assert kwargs["query_filter"] is not None
            assert kwargs["query_filter"].must[0].match.value == "Sony WH-1000XM5"
            if kwargs["collection_name"] == COLLECTION_CURATED_KB:
                return _make_query_response([curated_point])
            return _make_query_response([auto_point])

        client.query_points = mock_query

        results = await hybrid_search(
            client=client,
            query_vector=DUMMY_VECTOR,
            product_query="Sony WH-1000XM5",
        )
        assert len(results) == 2
        assert results[0].score == 0.92  # Curated has higher score
        assert results[0].is_curated is True
        assert results[1].score == 0.85

    @pytest.mark.asyncio
    async def test_empty_curated_returns_auto_only(self):
        """When curated_kb is empty, results come only from auto_crawled."""
        auto_point = _make_scored_point(
            "a1",
            0.80,
            payload={
                "text": "auto review text",
                "source_url": "https://auto.com/review",
                "chunk_index": 0,
            },
        )

        client = AsyncMock()

        async def mock_query(**kwargs):
            if kwargs["collection_name"] == COLLECTION_CURATED_KB:
                return _make_query_response([])
            return _make_query_response([auto_point])

        client.query_points = mock_query

        results = await hybrid_search(client=client, query_vector=DUMMY_VECTOR)
        assert len(results) == 1
        assert results[0].text == "auto review text"

    @pytest.mark.asyncio
    async def test_empty_auto_returns_curated_only(self):
        """When auto_crawled is empty, results come only from curated_kb."""
        curated_point = _make_scored_point(
            "c1",
            0.85,
            payload={
                "text": "curated review text",
                "source_url": "https://curated.com/review",
                "is_curated": True,
                "chunk_index": 0,
            },
        )

        client = AsyncMock()

        async def mock_query(**kwargs):
            if kwargs["collection_name"] == COLLECTION_CURATED_KB:
                return _make_query_response([curated_point])
            return _make_query_response([])

        client.query_points = mock_query

        results = await hybrid_search(client=client, query_vector=DUMMY_VECTOR)
        assert len(results) == 1
        assert results[0].is_curated is True
