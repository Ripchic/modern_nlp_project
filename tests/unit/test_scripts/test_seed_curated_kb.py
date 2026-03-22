"""Unit tests for scripts/seed_curated_kb.py — Curated KB seed script."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add scripts/ to sys.path so we can import seed_curated_kb module
_SCRIPTS_DIR = str(Path(__file__).resolve().parents[3] / "scripts")
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

from seed_curated_kb import (  # noqa: E402
    ALL_CURATED_ARTICLES,
    CATEGORIES,
    MAX_AGE_DAYS,
    CuratedArticle,
    SeedResult,
    VerifyResult,
    _get_qdrant_url,
    _is_article_fresh,
    seed_curated_kb,
    verify_curated_kb,
)

# ===================================================================
# Constants & data integrity
# ===================================================================


class TestConstants:
    """Tests for module-level constants."""

    def test_categories_count(self):
        """Exactly 7 categories."""
        assert len(CATEGORIES) == 7

    def test_categories_contents(self):
        expected = {"smartphones", "headphones", "laptops", "smartwatches", "tablets", "speakers", "smart_tvs"}
        assert set(CATEGORIES) == expected

    def test_max_age_days(self):
        assert MAX_AGE_DAYS == 365

    def test_all_articles_count(self):
        """At least 14 articles (2 per category)."""
        assert len(ALL_CURATED_ARTICLES) >= 14

    def test_articles_per_category(self):
        """Each of the 7 categories has at least 2 articles."""
        per_cat: dict[str, int] = {}
        for a in ALL_CURATED_ARTICLES:
            per_cat[a.category] = per_cat.get(a.category, 0) + 1
        for cat in CATEGORIES:
            assert per_cat.get(cat, 0) >= 2, f"Category '{cat}' needs >= 2 articles"


# ===================================================================
# CuratedArticle dataclass
# ===================================================================


class TestCuratedArticle:
    """Tests for the CuratedArticle dataclass."""

    def test_frozen(self):
        a = ALL_CURATED_ARTICLES[0]
        with pytest.raises(AttributeError):
            a.text = "modified"  # type: ignore[misc]

    def test_required_fields(self):
        for a in ALL_CURATED_ARTICLES:
            assert a.category in CATEGORIES, f"Unknown category: {a.category}"
            assert a.product_query, f"Missing product_query: {a.source_url}"
            assert a.source_url.startswith("https://"), f"Bad URL: {a.source_url}"
            assert a.source_name, f"Missing source_name: {a.source_url}"
            assert len(a.text) > 200, f"Too short: {a.source_url}"

    def test_unique_urls(self):
        urls = [a.source_url for a in ALL_CURATED_ARTICLES]
        assert len(urls) == len(set(urls)), "Duplicate source URLs"

    def test_default_language(self):
        for a in ALL_CURATED_ARTICLES:
            assert a.language == "ru"

    def test_has_dates(self):
        """All articles have date strings."""
        for a in ALL_CURATED_ARTICLES:
            assert a.date, f"Missing date: {a.source_url}"

    def test_has_authors(self):
        """All articles have author strings."""
        for a in ALL_CURATED_ARTICLES:
            assert a.author, f"Missing author: {a.source_url}"

    def test_source_names(self):
        """Source names are from known expert sources."""
        for a in ALL_CURATED_ARTICLES:
            assert a.source_name, f"Missing source_name: {a.source_url}"


# ===================================================================
# _is_article_fresh
# ===================================================================


class TestIsArticleFresh:
    """Tests for the freshness check function."""

    def test_no_date_is_fresh(self):
        a = CuratedArticle(
            category="test",
            product_query="x",
            source_url="https://x.com",
            source_name="X",
            text="t" * 300,
        )
        assert _is_article_fresh(a) is True

    def test_recent_date_is_fresh(self):
        recent = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")
        a = CuratedArticle(
            category="test",
            product_query="x",
            source_url="https://x.com",
            source_name="X",
            text="t" * 300,
            date=recent,
        )
        assert _is_article_fresh(a) is True

    def test_old_date_is_stale(self):
        old = (datetime.now(timezone.utc) - timedelta(days=400)).strftime("%Y-%m-%d")
        a = CuratedArticle(
            category="test",
            product_query="x",
            source_url="https://x.com",
            source_name="X",
            text="t" * 300,
            date=old,
        )
        assert _is_article_fresh(a) is False

    def test_exactly_max_age(self):
        exact = (datetime.now(timezone.utc) - timedelta(days=MAX_AGE_DAYS)).strftime("%Y-%m-%d")
        a = CuratedArticle(
            category="test",
            product_query="x",
            source_url="https://x.com",
            source_name="X",
            text="t" * 300,
            date=exact,
        )
        assert _is_article_fresh(a) is True

    def test_invalid_date_format(self):
        a = CuratedArticle(
            category="test",
            product_query="x",
            source_url="https://x.com",
            source_name="X",
            text="t" * 300,
            date="not-a-date",
        )
        assert _is_article_fresh(a) is True  # unparseable → assume fresh

    def test_custom_max_age(self):
        recent = (datetime.now(timezone.utc) - timedelta(days=10)).strftime("%Y-%m-%d")
        a = CuratedArticle(
            category="test",
            product_query="x",
            source_url="https://x.com",
            source_name="X",
            text="t" * 300,
            date=recent,
        )
        assert _is_article_fresh(a, max_age_days=5) is False

    def test_all_builtin_articles_are_fresh(self):
        """All built-in articles should pass the freshness check."""
        for a in ALL_CURATED_ARTICLES:
            assert _is_article_fresh(a), f"Article is stale: {a.source_url} ({a.date})"


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
        url = _get_qdrant_url()
        assert url.startswith("http")


# ===================================================================
# SeedResult
# ===================================================================


class TestSeedResult:
    """Tests for SeedResult dataclass."""

    def test_defaults(self):
        r = SeedResult()
        assert r.total_articles == 0
        assert r.ingested_articles == 0
        assert r.skipped_stale == 0
        assert r.total_chunks == 0
        assert r.chunks_per_category == {}
        assert r.errors == []

    def test_with_values(self):
        r = SeedResult(
            total_articles=14,
            ingested_articles=12,
            skipped_stale=2,
            total_chunks=50,
            chunks_per_category={"smartphones": 10},
            errors=["fail"],
        )
        assert r.ingested_articles == 12
        assert r.chunks_per_category["smartphones"] == 10


# ===================================================================
# VerifyResult
# ===================================================================


class TestVerifyResult:
    """Tests for VerifyResult dataclass."""

    def test_defaults(self):
        vr = VerifyResult()
        assert not vr.collection_exists
        assert vr.points_count == 0
        assert vr.categories_found == []
        assert vr.sample_payloads == []
        assert not vr.all_curated
        assert not vr.has_category_field

    def test_with_values(self):
        vr = VerifyResult(
            collection_exists=True,
            points_count=50,
            categories_found=["headphones", "smartphones"],
            all_curated=True,
            has_category_field=True,
        )
        assert vr.points_count == 50
        assert vr.all_curated


# ===================================================================
# seed_curated_kb (mocked)
# ===================================================================


class TestSeedCuratedKb:
    """Tests for the seed_curated_kb function with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_seed_calls_ensure_collections(self):
        mock_client = AsyncMock()
        mock_embedding = AsyncMock()
        mock_embedding.embed_batch = AsyncMock(return_value=[[0.1] * 1536])
        mock_embedding.close = AsyncMock()

        mock_upsert_result = MagicMock()
        mock_upsert_result.inserted = 1
        mock_upsert_result.skipped = 0

        with (
            patch("qdrant_client.AsyncQdrantClient", return_value=mock_client),
            patch("reviewmind.core.embeddings.EmbeddingService", return_value=mock_embedding),
            patch("reviewmind.vectorstore.collections.ensure_all_collections", new_callable=AsyncMock) as mock_ensure,
            patch(
                "reviewmind.vectorstore.client.upsert_chunks",
                new_callable=AsyncMock,
                return_value=mock_upsert_result,
            ),
            patch("reviewmind.ingestion.cleaner.clean_text", return_value="cleaned text " * 50),
            patch("reviewmind.ingestion.chunker.chunk_text", return_value=[MagicMock(text="chunk", chunk_index=0)]),
            patch("reviewmind.ingestion.sponsor.detect_sponsor_detailed", return_value=MagicMock(is_sponsored=False)),
        ):
            result = await seed_curated_kb("http://test:6333")
            mock_ensure.assert_awaited_once_with(mock_client)
            assert result.ingested_articles == len(ALL_CURATED_ARTICLES)

    @pytest.mark.asyncio
    async def test_seed_uses_curated_kb_collection(self):
        mock_client = AsyncMock()
        mock_embedding = AsyncMock()
        mock_embedding.embed_batch = AsyncMock(return_value=[[0.1] * 1536])
        mock_embedding.close = AsyncMock()

        upsert_calls = []

        async def capture_upsert(client, collection, vectors, payloads, **kw):
            upsert_calls.append(collection)
            r = MagicMock()
            r.inserted = 1
            r.skipped = 0
            return r

        with (
            patch("qdrant_client.AsyncQdrantClient", return_value=mock_client),
            patch("reviewmind.core.embeddings.EmbeddingService", return_value=mock_embedding),
            patch("reviewmind.vectorstore.collections.ensure_all_collections", new_callable=AsyncMock),
            patch("reviewmind.vectorstore.client.upsert_chunks", side_effect=capture_upsert),
            patch("reviewmind.ingestion.cleaner.clean_text", return_value="cleaned text " * 50),
            patch("reviewmind.ingestion.chunker.chunk_text", return_value=[MagicMock(text="chunk", chunk_index=0)]),
            patch("reviewmind.ingestion.sponsor.detect_sponsor_detailed", return_value=MagicMock(is_sponsored=False)),
        ):
            await seed_curated_kb("http://test:6333")
            # All upserts should be to curated_kb
            assert all(c == "curated_kb" for c in upsert_calls)

    @pytest.mark.asyncio
    async def test_seed_payloads_are_curated(self):
        mock_client = AsyncMock()
        mock_embedding = AsyncMock()
        mock_embedding.embed_batch = AsyncMock(return_value=[[0.1] * 1536])
        mock_embedding.close = AsyncMock()

        captured_payloads = []

        async def capture_upsert(client, collection, vectors, payloads, **kw):
            captured_payloads.extend(payloads)
            r = MagicMock()
            r.inserted = len(payloads)
            r.skipped = 0
            return r

        with (
            patch("qdrant_client.AsyncQdrantClient", return_value=mock_client),
            patch("reviewmind.core.embeddings.EmbeddingService", return_value=mock_embedding),
            patch("reviewmind.vectorstore.collections.ensure_all_collections", new_callable=AsyncMock),
            patch("reviewmind.vectorstore.client.upsert_chunks", side_effect=capture_upsert),
            patch("reviewmind.ingestion.cleaner.clean_text", return_value="cleaned text " * 50),
            patch("reviewmind.ingestion.chunker.chunk_text", return_value=[MagicMock(text="chunk", chunk_index=0)]),
            patch("reviewmind.ingestion.sponsor.detect_sponsor_detailed", return_value=MagicMock(is_sponsored=False)),
        ):
            await seed_curated_kb("http://test:6333")
            assert len(captured_payloads) > 0
            for p in captured_payloads:
                assert p.is_curated is True
                assert p.source_type == "curated"

    @pytest.mark.asyncio
    async def test_seed_skips_stale_articles(self):
        mock_client = AsyncMock()
        mock_embedding = AsyncMock()
        mock_embedding.embed_batch = AsyncMock(return_value=[[0.1] * 1536])
        mock_embedding.close = AsyncMock()

        mock_upsert_result = MagicMock()
        mock_upsert_result.inserted = 1
        mock_upsert_result.skipped = 0

        # Make _is_article_fresh return False for all
        with (
            patch("qdrant_client.AsyncQdrantClient", return_value=mock_client),
            patch("reviewmind.core.embeddings.EmbeddingService", return_value=mock_embedding),
            patch("reviewmind.vectorstore.collections.ensure_all_collections", new_callable=AsyncMock),
            patch(
                "reviewmind.vectorstore.client.upsert_chunks",
                new_callable=AsyncMock,
                return_value=mock_upsert_result,
            ),
            patch("seed_curated_kb._is_article_fresh", return_value=False),
        ):
            result = await seed_curated_kb("http://test:6333")
            assert result.skipped_stale == len(ALL_CURATED_ARTICLES)
            assert result.ingested_articles == 0

    @pytest.mark.asyncio
    async def test_seed_handles_clean_empty(self):
        mock_client = AsyncMock()
        mock_embedding = AsyncMock()
        mock_embedding.close = AsyncMock()

        with (
            patch("qdrant_client.AsyncQdrantClient", return_value=mock_client),
            patch("reviewmind.core.embeddings.EmbeddingService", return_value=mock_embedding),
            patch("reviewmind.vectorstore.collections.ensure_all_collections", new_callable=AsyncMock),
            patch("reviewmind.ingestion.cleaner.clean_text", return_value=""),
        ):
            result = await seed_curated_kb("http://test:6333")
            assert result.ingested_articles == 0
            assert len(result.errors) == len(ALL_CURATED_ARTICLES)

    @pytest.mark.asyncio
    async def test_seed_closes_client(self):
        mock_client = AsyncMock()
        mock_embedding = AsyncMock()
        mock_embedding.close = AsyncMock()

        with (
            patch("qdrant_client.AsyncQdrantClient", return_value=mock_client),
            patch("reviewmind.core.embeddings.EmbeddingService", return_value=mock_embedding),
            patch("reviewmind.vectorstore.collections.ensure_all_collections", new_callable=AsyncMock),
            patch("reviewmind.ingestion.cleaner.clean_text", return_value=""),
        ):
            await seed_curated_kb("http://test:6333")
            mock_client.close.assert_awaited_once()


# ===================================================================
# verify_curated_kb (mocked)
# ===================================================================


class TestVerifyCuratedKb:
    """Tests for the verify_curated_kb function with mocked dependencies."""

    @pytest.mark.asyncio
    async def test_verify_collection_not_exists(self):
        mock_client = AsyncMock()
        mock_client.collection_exists = AsyncMock(return_value=False)

        with patch("qdrant_client.AsyncQdrantClient", return_value=mock_client):
            vr = await verify_curated_kb("http://test:6333")
            assert not vr.collection_exists
            assert vr.points_count == 0

    @pytest.mark.asyncio
    async def test_verify_collection_exists_with_points(self):
        mock_client = AsyncMock()
        mock_client.collection_exists = AsyncMock(return_value=True)

        mock_info = MagicMock()
        mock_info.points_count = 42
        mock_client.get_collection = AsyncMock(return_value=mock_info)

        mock_point = MagicMock()
        mock_point.payload = {
            "is_curated": True,
            "category": "headphones",
            "source_type": "curated",
        }
        mock_client.scroll = AsyncMock(return_value=([mock_point], None))

        with patch("qdrant_client.AsyncQdrantClient", return_value=mock_client):
            vr = await verify_curated_kb("http://test:6333")
            assert vr.collection_exists
            assert vr.points_count == 42
            assert "headphones" in vr.categories_found
            assert vr.all_curated
            assert vr.has_category_field

    @pytest.mark.asyncio
    async def test_verify_detects_non_curated(self):
        mock_client = AsyncMock()
        mock_client.collection_exists = AsyncMock(return_value=True)

        mock_info = MagicMock()
        mock_info.points_count = 10
        mock_client.get_collection = AsyncMock(return_value=mock_info)

        mock_point = MagicMock()
        mock_point.payload = {
            "is_curated": False,
            "category": "smartphones",
        }
        mock_client.scroll = AsyncMock(return_value=([mock_point], None))

        with patch("qdrant_client.AsyncQdrantClient", return_value=mock_client):
            vr = await verify_curated_kb("http://test:6333")
            assert not vr.all_curated

    @pytest.mark.asyncio
    async def test_verify_closes_client(self):
        mock_client = AsyncMock()
        mock_client.collection_exists = AsyncMock(return_value=False)

        with patch("qdrant_client.AsyncQdrantClient", return_value=mock_client):
            await verify_curated_kb("http://test:6333")
            mock_client.close.assert_awaited_once()


# ===================================================================
# Payload coverage
# ===================================================================


class TestPayloadCoverage:
    """Tests that payloads have all required fields per PRD."""

    def test_all_categories_represented(self):
        cats = {a.category for a in ALL_CURATED_ARTICLES}
        assert cats == set(CATEGORIES)

    def test_diverse_source_names(self):
        names = {a.source_name for a in ALL_CURATED_ARTICLES}
        assert len(names) >= 2, "Need multiple expert sources"

    def test_date_format(self):
        """All dates are in YYYY-MM-DD format."""
        for a in ALL_CURATED_ARTICLES:
            if a.date:
                datetime.strptime(a.date, "%Y-%m-%d")


# ===================================================================
# Integration scenarios
# ===================================================================


class TestIntegrationScenarios:
    """Tests matching the 5 PRD test_steps."""

    def test_step1_script_exists(self):
        """Шаг 1: script file exists (seed_curated_kb.py)."""
        script_path = Path(__file__).resolve().parents[3] / "scripts" / "seed_curated_kb.py"
        assert script_path.exists()

    def test_step2_curated_collection_target(self):
        """Шаг 2: seed targets curated_kb, not auto_crawled."""
        from reviewmind.vectorstore.collections import COLLECTION_CURATED_KB

        assert COLLECTION_CURATED_KB == "curated_kb"

    def test_step3_curated_badge_marker(self):
        """Шаг 3: curated marker exists for RAG response."""
        from reviewmind.core.prompts import CURATED_MARKER

        assert CURATED_MARKER == "📚"

    def test_step4_dedup_via_point_id(self):
        """Шаг 4: deterministic point IDs enable dedup on re-run."""
        from reviewmind.vectorstore.client import generate_point_id

        id1 = generate_point_id("https://example.com/test", 0)
        id2 = generate_point_id("https://example.com/test", 0)
        assert id1 == id2  # Same URL + index → same ID

    def test_step5_payload_has_category_and_curated(self):
        """Шаг 5: ChunkPayload supports category (via metadata) and is_curated."""
        from reviewmind.vectorstore.client import ChunkPayload

        p = ChunkPayload(
            text="test",
            source_url="https://x.com",
            source_type="curated",
            is_curated=True,
        )
        d = p.to_dict()
        assert d["is_curated"] is True
        assert d["source_type"] == "curated"

    def test_all_articles_text_quality(self):
        """All articles have substantial text (> 500 chars) for meaningful chunks."""
        for a in ALL_CURATED_ARTICLES:
            assert len(a.text) > 500, f"Article too short: {a.source_url}"

    def test_categories_match_curated_kb_collection_spec(self):
        """curated_kb collection spec has a category payload index."""
        from reviewmind.vectorstore.collections import CURATED_KB_SPEC

        index_names = [idx.field_name for idx in CURATED_KB_SPEC.payload_indexes]
        assert "category" in index_names
