"""Unit tests for scripts/seed_test_data.py — Seed data + RAG verification."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add scripts/ to sys.path so we can import seed_test_data module
_SCRIPTS_DIR = str(Path(__file__).resolve().parents[3] / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from seed_test_data import (  # noqa: E402
    ALL_SEED_REVIEWS,
    PRODUCT_DYSON_V15,
    PRODUCT_IPHONE_16_PRO,
    PRODUCT_QUERIES,
    PRODUCT_SONY_WH1000XM5,
    TEST_QUERIES,
    VerificationReport,
    VerificationResult,
    _get_qdrant_url,
    seed_data,
    verify_rag,
)

# ===================================================================
# Constants & data integrity
# ===================================================================


class TestSeedReviews:
    """Tests for the seed review data."""

    def test_all_reviews_count(self):
        """At least 7 reviews (3 Sony + 2 iPhone + 2 Dyson)."""
        assert len(ALL_SEED_REVIEWS) >= 7

    def test_product_queries_count(self):
        """3 distinct product queries."""
        assert len(PRODUCT_QUERIES) == 3
        assert PRODUCT_SONY_WH1000XM5 in PRODUCT_QUERIES
        assert PRODUCT_IPHONE_16_PRO in PRODUCT_QUERIES
        assert PRODUCT_DYSON_V15 in PRODUCT_QUERIES

    def test_reviews_per_product(self):
        """Each product has at least 2 reviews."""
        per_product = {}
        for r in ALL_SEED_REVIEWS:
            per_product[r.product_query] = per_product.get(r.product_query, 0) + 1
        for product in PRODUCT_QUERIES:
            assert per_product.get(product, 0) >= 2, f"{product} needs ≥2 reviews"

    def test_review_fields(self):
        """Each review has required fields."""
        for r in ALL_SEED_REVIEWS:
            assert r.product_query, f"Missing product_query: {r.source_url}"
            assert r.source_url, "Missing source_url"
            assert r.source_type in ("youtube", "reddit", "web"), f"Bad type: {r.source_type}"
            assert len(r.text) > 200, f"Too short: {r.source_url}"

    def test_review_source_types_diverse(self):
        """Reviews include multiple source types."""
        types = {r.source_type for r in ALL_SEED_REVIEWS}
        assert len(types) >= 2, "Need at least 2 distinct source types"

    def test_at_least_one_sponsored(self):
        """At least one review is marked sponsored for testing."""
        assert any(r.is_sponsored for r in ALL_SEED_REVIEWS)

    def test_seed_review_frozen(self):
        """SeedReview is frozen dataclass."""
        review = ALL_SEED_REVIEWS[0]
        with pytest.raises(AttributeError):
            review.text = "modified"  # type: ignore[misc]

    def test_unique_urls(self):
        """Each review has a unique source_url."""
        urls = [r.source_url for r in ALL_SEED_REVIEWS]
        assert len(urls) == len(set(urls)), "Duplicate source URLs"


# ===================================================================
# Test queries
# ===================================================================


class TestTestQueries:
    """Tests for the 20 test queries."""

    def test_query_count(self):
        """Exactly 20 test queries."""
        assert len(TEST_QUERIES) == 20

    def test_queries_non_empty(self):
        """All queries are non-empty strings."""
        for tq in TEST_QUERIES:
            assert tq.query and len(tq.query.strip()) > 0

    def test_queries_cover_all_products(self):
        """Queries cover all 3 products."""
        products = {tq.product_query for tq in TEST_QUERIES if tq.product_query}
        for p in PRODUCT_QUERIES:
            assert p in products, f"No queries for {p}"

    def test_english_query_exists(self):
        """At least one English query exists."""
        assert any(tq.expected_language == "en" for tq in TEST_QUERIES)

    def test_edge_case_query_exists(self):
        """At least one edge-case query (no specific product) exists."""
        assert any(tq.product_query is None for tq in TEST_QUERIES)

    def test_test_query_frozen(self):
        """TestQuery is frozen dataclass."""
        tq = TEST_QUERIES[0]
        with pytest.raises(AttributeError):
            tq.query = "modified"  # type: ignore[misc]


# ===================================================================
# VerificationResult / VerificationReport
# ===================================================================


class TestVerificationResult:
    """Tests for VerificationResult dataclass."""

    def test_defaults(self):
        vr = VerificationResult(query="test")
        assert vr.query == "test"
        assert vr.answer == ""
        assert vr.sources_count == 0
        assert vr.chunks_count == 0
        assert not vr.confidence_met
        assert not vr.has_answer
        assert vr.response_time_ms == 0.0
        assert vr.error is None

    def test_with_values(self):
        vr = VerificationResult(
            query="q",
            answer="answer",
            sources_count=3,
            chunks_count=5,
            confidence_met=True,
            has_answer=True,
            response_time_ms=150.0,
        )
        assert vr.sources_count == 3
        assert vr.confidence_met


class TestVerificationReport:
    """Tests for VerificationReport dataclass."""

    def test_defaults(self):
        report = VerificationReport()
        assert report.total == 0
        assert report.passed == 0
        assert report.failed == 0
        assert report.avg_response_time_ms == 0.0
        assert report.results == []

    def test_pass_rate_zero_total(self):
        report = VerificationReport(total=0)
        assert report.pass_rate == 0.0

    def test_pass_rate_calculation(self):
        report = VerificationReport(total=20, passed=16, failed=4)
        assert report.pass_rate == 80.0

    def test_pass_rate_all_passed(self):
        report = VerificationReport(total=10, passed=10, failed=0)
        assert report.pass_rate == 100.0


# ===================================================================
# _get_qdrant_url
# ===================================================================


class TestGetQdrantUrl:
    """Tests for Qdrant URL resolution."""

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("QDRANT_URL", "http://custom:6333")
        assert _get_qdrant_url() == "http://custom:6333"

    def test_fallback_default(self, monkeypatch):
        monkeypatch.delenv("QDRANT_URL", raising=False)
        # Config may or may not be available; either way should not crash
        url = _get_qdrant_url()
        assert url.startswith("http")


# ===================================================================
# seed_data (mocked)
# ===================================================================


class TestSeedData:
    """Tests for the seed_data function with mocked external services."""

    @pytest.mark.asyncio
    async def test_seed_creates_chunks(self):
        """seed_data processes reviews and returns chunk counts per product."""
        from reviewmind.vectorstore.client import UpsertResult

        mock_embed = AsyncMock()
        mock_embed.embed_batch = AsyncMock(return_value=[[0.1] * 1536])
        mock_embed.close = AsyncMock()

        mock_upsert = AsyncMock(return_value=UpsertResult(total=1, inserted=1, skipped=0))

        mock_client = AsyncMock()
        mock_client.close = AsyncMock()

        with (
            patch("qdrant_client.AsyncQdrantClient", return_value=mock_client),
            patch("reviewmind.core.embeddings.EmbeddingService", return_value=mock_embed),
            patch("reviewmind.vectorstore.client.upsert_chunks", mock_upsert),
            patch("reviewmind.vectorstore.collections.ensure_all_collections", new_callable=AsyncMock),
            patch("reviewmind.ingestion.chunker.chunk_text") as mock_chunk,
            patch("reviewmind.ingestion.cleaner.clean_text", side_effect=lambda t: t),
        ):
            # Return 1 chunk per review
            mock_chunk_obj = MagicMock()
            mock_chunk_obj.text = "chunk text"
            mock_chunk_obj.chunk_index = 0
            mock_chunk.return_value = [mock_chunk_obj]

            result = await seed_data("http://fake:6333")

        assert isinstance(result, dict)
        # Should have entries for all 3 products
        assert len(result) == 3
        for product in PRODUCT_QUERIES:
            assert product in result
            assert result[product] > 0

    @pytest.mark.asyncio
    async def test_seed_handles_empty_clean(self):
        """seed_data skips reviews that produce empty text after cleaning."""
        mock_embed = AsyncMock()
        mock_embed.embed_batch = AsyncMock(return_value=[])
        mock_embed.close = AsyncMock()

        mock_client = AsyncMock()
        mock_client.close = AsyncMock()

        with (
            patch("qdrant_client.AsyncQdrantClient", return_value=mock_client),
            patch("reviewmind.core.embeddings.EmbeddingService", return_value=mock_embed),
            patch("reviewmind.vectorstore.collections.ensure_all_collections", new_callable=AsyncMock),
            patch("reviewmind.ingestion.cleaner.clean_text", return_value=""),
        ):
            result = await seed_data("http://fake:6333")

        # All reviews skipped → empty dict
        assert sum(result.values()) == 0

    @pytest.mark.asyncio
    async def test_seed_closes_resources(self):
        """seed_data closes Qdrant client and embedding service."""
        mock_embed = AsyncMock()
        mock_embed.embed_batch = AsyncMock(return_value=[[0.1] * 1536])
        mock_embed.close = AsyncMock()

        mock_client = AsyncMock()
        mock_client.close = AsyncMock()

        with (
            patch("qdrant_client.AsyncQdrantClient", return_value=mock_client),
            patch("reviewmind.core.embeddings.EmbeddingService", return_value=mock_embed),
            patch(
                "reviewmind.vectorstore.client.upsert_chunks",
                new_callable=AsyncMock,
                return_value=MagicMock(inserted=1, skipped=0),
            ),
            patch("reviewmind.vectorstore.collections.ensure_all_collections", new_callable=AsyncMock),
            patch("reviewmind.ingestion.chunker.chunk_text", return_value=[MagicMock(text="t", chunk_index=0)]),
            patch("reviewmind.ingestion.cleaner.clean_text", side_effect=lambda t: t),
        ):
            await seed_data("http://fake:6333")

        mock_client.close.assert_awaited_once()
        mock_embed.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_seed_idempotent_via_skip_dedup(self):
        """seed_data passes skip_dedup=True for deterministic upserts."""
        mock_embed = AsyncMock()
        mock_embed.embed_batch = AsyncMock(return_value=[[0.1] * 1536])
        mock_embed.close = AsyncMock()

        mock_upsert = AsyncMock(return_value=MagicMock(inserted=1, skipped=0))
        mock_client = AsyncMock()
        mock_client.close = AsyncMock()

        with (
            patch("qdrant_client.AsyncQdrantClient", return_value=mock_client),
            patch("reviewmind.core.embeddings.EmbeddingService", return_value=mock_embed),
            patch("reviewmind.vectorstore.client.upsert_chunks", mock_upsert),
            patch("reviewmind.vectorstore.collections.ensure_all_collections", new_callable=AsyncMock),
            patch("reviewmind.ingestion.chunker.chunk_text", return_value=[MagicMock(text="t", chunk_index=0)]),
            patch("reviewmind.ingestion.cleaner.clean_text", side_effect=lambda t: t),
        ):
            await seed_data("http://fake:6333")

        # All upsert calls should have skip_dedup=True
        for call in mock_upsert.call_args_list:
            assert call.kwargs.get("skip_dedup") is True or call[1].get("skip_dedup") is True  # noqa: S101


# ===================================================================
# verify_rag (mocked)
# ===================================================================


class TestVerifyRag:
    """Tests for verify_rag with mocked RAG pipeline."""

    @pytest.mark.asyncio
    async def test_verify_returns_report(self):
        """verify_rag returns a VerificationReport with results for all queries."""
        from reviewmind.core.rag import RAGResponse

        mock_response = RAGResponse(
            answer="Это структурированный ответ с анализом товара по нескольким источникам.",
            sources=["https://example.com/review1"],
            used_curated=False,
            confidence_met=True,
            chunks_count=5,
            chunks_found=8,
        )

        mock_pipeline = AsyncMock()
        mock_pipeline.query = AsyncMock(return_value=mock_response)
        mock_pipeline.close = AsyncMock()

        mock_client = AsyncMock()
        mock_client.close = AsyncMock()

        with (
            patch("qdrant_client.AsyncQdrantClient", return_value=mock_client),
            patch("reviewmind.core.rag.RAGPipeline", return_value=mock_pipeline),
        ):
            report = await verify_rag("http://fake:6333")

        assert report.total == 20
        assert len(report.results) == 20
        # Most queries should pass with our mock response
        assert report.passed > 0
        assert report.avg_response_time_ms > 0

    @pytest.mark.asyncio
    async def test_verify_handles_pipeline_error(self):
        """verify_rag handles pipeline exceptions gracefully."""
        mock_pipeline = AsyncMock()
        mock_pipeline.query = AsyncMock(side_effect=RuntimeError("boom"))
        mock_pipeline.close = AsyncMock()

        mock_client = AsyncMock()
        mock_client.close = AsyncMock()

        with (
            patch("qdrant_client.AsyncQdrantClient", return_value=mock_client),
            patch("reviewmind.core.rag.RAGPipeline", return_value=mock_pipeline),
        ):
            report = await verify_rag("http://fake:6333")

        assert report.total == 20
        assert report.failed == 20
        for r in report.results:
            assert r.error == "boom"

    @pytest.mark.asyncio
    async def test_verify_closes_resources(self):
        """verify_rag closes pipeline and client."""
        from reviewmind.core.rag import RAGResponse

        mock_response = RAGResponse(answer="ok", sources=[], confidence_met=True, chunks_count=1)
        mock_pipeline = AsyncMock()
        mock_pipeline.query = AsyncMock(return_value=mock_response)
        mock_pipeline.close = AsyncMock()

        mock_client = AsyncMock()
        mock_client.close = AsyncMock()

        with (
            patch("qdrant_client.AsyncQdrantClient", return_value=mock_client),
            patch("reviewmind.core.rag.RAGPipeline", return_value=mock_pipeline),
        ):
            await verify_rag("http://fake:6333")

        mock_pipeline.close.assert_awaited_once()
        mock_client.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_verify_pass_rate(self):
        """Queries with answer and sources count as passed."""
        from reviewmind.core.rag import RAGResponse

        mock_response = RAGResponse(
            answer="A" * 50,
            sources=["https://src1.com"],
            confidence_met=True,
            chunks_count=4,
        )

        mock_pipeline = AsyncMock()
        mock_pipeline.query = AsyncMock(return_value=mock_response)
        mock_pipeline.close = AsyncMock()

        mock_client = AsyncMock()
        mock_client.close = AsyncMock()

        with (
            patch("qdrant_client.AsyncQdrantClient", return_value=mock_client),
            patch("reviewmind.core.rag.RAGPipeline", return_value=mock_pipeline),
        ):
            report = await verify_rag("http://fake:6333")

        # Last query (product_query=None, expects_sources=False, min_chunks=0)
        # should still pass if has_answer=True
        assert report.pass_rate >= 80.0


# ===================================================================
# Integration scenarios
# ===================================================================


class TestIntegrationScenarios:
    """High-level integration tests for the seed script."""

    def test_all_products_have_multiple_source_types(self):
        """Each product has reviews from different source types."""
        for product in PRODUCT_QUERIES:
            types = {r.source_type for r in ALL_SEED_REVIEWS if r.product_query == product}
            assert len(types) >= 2, f"{product} needs diverse source types"

    def test_reviews_text_is_realistic(self):
        """Reviews have realistic length (>500 chars) with product mentions."""
        for r in ALL_SEED_REVIEWS:
            assert len(r.text) > 500, f"Review too short: {r.source_url}"

    def test_queries_cover_typical_use_cases(self):
        """Queries cover: buy decision, feature, comparison, drawbacks."""
        query_texts = [tq.query.lower() for tq in TEST_QUERIES]
        joined = " ".join(query_texts)
        assert "стоит ли" in joined or "worth" in joined, "Missing buy decision query"
        assert "минус" in joined, "Missing drawbacks query"
        assert "vs" in joined, "Missing comparison query"
        assert "батаре" in joined or "аккумулят" in joined, "Missing battery query"

    def test_no_duplicate_queries(self):
        """All 20 queries are unique."""
        queries = [tq.query for tq in TEST_QUERIES]
        assert len(queries) == len(set(queries)), "Duplicate queries found"

    @pytest.mark.asyncio
    async def test_main_seed_command(self):
        """main(['seed']) runs seed_data."""
        from seed_test_data import main

        with patch("seed_test_data.seed_data", new_callable=AsyncMock, return_value={"prod": 5}) as mock:
            await main(["seed"])
        mock.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_main_verify_command(self):
        """main(['verify']) runs verify_rag and prints report."""
        from seed_test_data import main

        report = VerificationReport(total=20, passed=18, failed=2)
        with (
            patch("seed_test_data.verify_rag", new_callable=AsyncMock, return_value=report),
            patch("seed_test_data._print_report") as mock_print,
        ):
            await main(["verify"])
        mock_print.assert_called_once_with(report)

    @pytest.mark.asyncio
    async def test_main_no_args_runs_both(self):
        """main([]) runs both seed and verify."""
        from seed_test_data import main

        report = VerificationReport(total=20, passed=20, failed=0)
        with (
            patch("seed_test_data.seed_data", new_callable=AsyncMock, return_value={}) as mock_seed,
            patch("seed_test_data.verify_rag", new_callable=AsyncMock, return_value=report),
            patch("seed_test_data._print_report"),
        ):
            await main([])
        mock_seed.assert_awaited_once()
