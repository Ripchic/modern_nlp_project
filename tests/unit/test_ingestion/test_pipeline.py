"""Unit tests for reviewmind.ingestion.pipeline — IngestionPipeline orchestrator."""

from __future__ import annotations

from dataclasses import fields as dataclass_fields
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reviewmind.ingestion.pipeline import (
    _REDDIT_RE,
    _YOUTUBE_RE,
    IngestionPipeline,
    IngestionResult,
    SourceIngestionResult,
    detect_url_type,
)
from reviewmind.vectorstore.client import ChunkPayload, UpsertResult
from reviewmind.vectorstore.collections import COLLECTION_AUTO_CRAWLED

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _long_text(word_count: int = 600) -> str:
    """Generate a long body of text that survives clean_text filtering."""
    return " ".join(["This is a detailed product review sentence with enough words."] * word_count)


def _make_pipeline(
    *,
    qdrant=None,
    db_session=None,
    embedding_service=None,
    collection: str = COLLECTION_AUTO_CRAWLED,
) -> IngestionPipeline:
    from qdrant_client import AsyncQdrantClient as _QC

    q = qdrant or AsyncMock(spec=_QC)
    return IngestionPipeline(
        qdrant_client=q,
        db_session=db_session,
        embedding_service=embedding_service,
        collection_name=collection,
    )


# ===================================================================
# detect_url_type
# ===================================================================


class TestDetectUrlType:
    """Tests for the lightweight URL type classifier."""

    def test_youtube_watch(self):
        assert detect_url_type("https://www.youtube.com/watch?v=abc123") == "youtube"

    def test_youtube_short(self):
        assert detect_url_type("https://youtu.be/abc123") == "youtube"

    def test_youtube_shorts(self):
        assert detect_url_type("https://youtube.com/shorts/xyz") == "youtube"

    def test_youtube_embed(self):
        assert detect_url_type("https://youtube.com/embed/xyz") == "youtube"

    def test_youtube_live(self):
        assert detect_url_type("https://youtube.com/live/xyz") == "youtube"

    def test_reddit_post(self):
        assert detect_url_type("https://www.reddit.com/r/headphones/comments/abc") == "reddit"

    def test_reddit_short(self):
        assert detect_url_type("https://redd.it/abc123") == "reddit"

    def test_web_url(self):
        assert detect_url_type("https://rtings.com/headphones/reviews/sony/wh-1000xm5") == "web"

    def test_web_generic(self):
        assert detect_url_type("https://example.com/review") == "web"

    def test_case_insensitive(self):
        assert detect_url_type("https://YOUTUBE.COM/watch?v=x") == "youtube"
        assert detect_url_type("https://REDDIT.COM/r/test") == "reddit"


class TestUrlRegex:
    """Sanity checks for the compiled regex patterns."""

    def test_youtube_re_matches(self):
        assert _YOUTUBE_RE.search("youtube.com/watch?v=x")

    def test_reddit_re_matches(self):
        assert _REDDIT_RE.search("reddit.com/r/test")

    def test_youtube_re_no_false_positive(self):
        assert _YOUTUBE_RE.search("example.com/youtube") is None

    def test_reddit_re_no_false_positive(self):
        assert _REDDIT_RE.search("example.com/reddit") is None


# ===================================================================
# SourceIngestionResult
# ===================================================================


class TestSourceIngestionResult:
    def test_defaults(self):
        r = SourceIngestionResult(url="https://x.com", success=True)
        assert r.url == "https://x.com"
        assert r.success is True
        assert r.source_type == ""
        assert r.chunks_count == 0
        assert r.is_sponsored is False
        assert r.error is None
        assert r.source_id is None

    def test_failure(self):
        r = SourceIngestionResult(url="http://bad.url", success=False, error="timeout")
        assert r.success is False
        assert r.error == "timeout"

    def test_field_count(self):
        assert len(dataclass_fields(SourceIngestionResult)) == 7


# ===================================================================
# IngestionResult
# ===================================================================


class TestIngestionResult:
    def test_defaults(self):
        r = IngestionResult()
        assert r.success_count == 0
        assert r.failed_count == 0
        assert r.chunks_count == 0
        assert r.failed_urls == []
        assert r.results == []

    def test_field_count(self):
        assert len(dataclass_fields(IngestionResult)) == 5


# ===================================================================
# IngestionPipeline.__init__
# ===================================================================


class TestPipelineInit:
    def test_default_collection(self):
        p = _make_pipeline()
        assert p._collection == COLLECTION_AUTO_CRAWLED

    def test_custom_collection(self):
        p = _make_pipeline(collection="curated_kb")
        assert p._collection == "curated_kb"

    def test_owns_embedding_when_none(self):
        p = _make_pipeline()
        assert p._owns_embedding is True
        assert p._embedding is None

    def test_uses_provided_embedding(self):
        emb = MagicMock()
        p = _make_pipeline(embedding_service=emb)
        assert p._owns_embedding is False
        assert p._embedding is emb

    def test_scrapers_lazy(self):
        p = _make_pipeline()
        assert p._youtube is None
        assert p._reddit is None
        assert p._web is None


# ===================================================================
# Lazy scraper getters
# ===================================================================


class TestLazyScrapers:
    def test_get_youtube(self):
        p = _make_pipeline()
        yt = p._get_youtube()
        assert yt is not None
        assert p._youtube is yt
        # Same instance on second call
        assert p._get_youtube() is yt

    def test_get_reddit(self):
        p = _make_pipeline()
        rd = p._get_reddit()
        assert rd is not None
        assert p._get_reddit() is rd

    def test_get_web(self):
        p = _make_pipeline()
        wb = p._get_web()
        assert wb is not None
        assert p._get_web() is wb


# ===================================================================
# Embedding service lazy property
# ===================================================================


class TestLazyEmbedding:
    @patch("reviewmind.ingestion.pipeline.EmbeddingService")
    def test_creates_embedding_on_demand(self, mock_cls):
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        p = _make_pipeline()
        svc = p.embedding_service
        mock_cls.assert_called_once()
        assert svc is mock_instance

    def test_returns_provided_embedding(self):
        emb = MagicMock()
        p = _make_pipeline(embedding_service=emb)
        assert p.embedding_service is emb


# ===================================================================
# _scrape helper
# ===================================================================


class TestScrape:
    def test_youtube_success(self):
        p = _make_pipeline()
        mock_result = MagicMock()
        mock_result.text = "Video transcript text"
        mock_result.language_code = "en"
        mock_result.language = "English"
        mock_result.extra_metadata = {"author": "TestChannel", "date": "2026-01-01"}

        with patch.object(p, "_get_youtube") as mock_yt:
            mock_yt.return_value.get_transcript_by_url.return_value = mock_result
            text, meta = p._scrape("https://youtube.com/watch?v=x", "youtube")

        assert text == "Video transcript text"
        assert meta["language"] == "en"
        assert meta["author"] == "TestChannel"

    def test_youtube_none(self):
        p = _make_pipeline()
        with patch.object(p, "_get_youtube") as mock_yt:
            mock_yt.return_value.get_transcript_by_url.return_value = None
            text, meta = p._scrape("https://youtube.com/watch?v=x", "youtube")
        assert text == ""

    def test_reddit_success(self):
        p = _make_pipeline()
        mock_post = MagicMock()
        mock_post.full_text = "Reddit post text"
        mock_post.author = "redditor"

        with patch.object(p, "_get_reddit") as mock_rd:
            mock_rd.return_value.parse_url.return_value = mock_post
            text, meta = p._scrape("https://reddit.com/r/test/comments/abc", "reddit")

        assert text == "Reddit post text"
        assert meta["author"] == "redditor"
        assert meta["language"] is None  # Reddit doesn't report

    def test_reddit_none(self):
        p = _make_pipeline()
        with patch.object(p, "_get_reddit") as mock_rd:
            mock_rd.return_value.parse_url.return_value = None
            text, meta = p._scrape("https://reddit.com/r/test", "reddit")
        assert text == ""

    def test_web_success(self):
        p = _make_pipeline()
        mock_page = MagicMock()
        mock_page.text = "Web page content"
        mock_page.author = "Author"
        mock_page.language = "ru"
        mock_page.date = "2026-02-01"

        with patch.object(p, "_get_web") as mock_wb:
            mock_wb.return_value.parse_url.return_value = mock_page
            text, meta = p._scrape("https://rtings.com/review", "web")

        assert text == "Web page content"
        assert meta["language"] == "ru"
        assert meta["date"] == "2026-02-01"

    def test_web_none(self):
        p = _make_pipeline()
        with patch.object(p, "_get_web") as mock_wb:
            mock_wb.return_value.parse_url.return_value = None
            text, meta = p._scrape("https://example.com", "web")
        assert text == ""

    def test_youtube_language_fallback(self):
        """When language_code is empty, falls back to language field."""
        p = _make_pipeline()
        mock_result = MagicMock()
        mock_result.text = "Some text"
        mock_result.language_code = ""
        mock_result.language = "Russian"
        mock_result.extra_metadata = {}

        with patch.object(p, "_get_youtube") as mock_yt:
            mock_yt.return_value.get_transcript_by_url.return_value = mock_result
            _, meta = p._scrape("https://youtube.com/watch?v=x", "youtube")
        assert meta["language"] == "Russian"


# ===================================================================
# ingest_url — full pipeline mock tests
# ===================================================================


class TestIngestUrl:
    """Test the full ingest_url flow with mocked scraping, embedding, and upsert."""

    @pytest.fixture()
    def pipeline(self):
        """Return pipeline with mocked embedding service."""
        emb = AsyncMock()
        emb.embed_batch = AsyncMock(return_value=[[0.1] * 1536, [0.2] * 1536])
        emb.close = AsyncMock()
        p = _make_pipeline(embedding_service=emb)
        return p

    @pytest.mark.asyncio()
    async def test_success_youtube(self, pipeline):
        mock_result = MagicMock()
        mock_result.text = _long_text()
        mock_result.language_code = "en"
        mock_result.language = "English"
        mock_result.extra_metadata = {"author": "Tech"}

        with (
            patch.object(pipeline, "_get_youtube") as mock_yt,
            patch("reviewmind.ingestion.pipeline.upsert_chunks", new_callable=AsyncMock) as mock_upsert,
        ):
            mock_yt.return_value.get_transcript_by_url.return_value = mock_result
            mock_upsert.return_value = UpsertResult(total=2, inserted=2, skipped=0, skipped_indices=[])

            result = await pipeline.ingest_url(
                "https://youtube.com/watch?v=abc",
                "headphones review",
            )

        assert result.success is True
        assert result.source_type == "youtube"
        assert result.chunks_count == 2
        mock_upsert.assert_called_once()

    @pytest.mark.asyncio()
    async def test_success_reddit(self, pipeline):
        mock_post = MagicMock()
        mock_post.full_text = _long_text()
        mock_post.author = "redditor"

        with (
            patch.object(pipeline, "_get_reddit") as mock_rd,
            patch("reviewmind.ingestion.pipeline.upsert_chunks", new_callable=AsyncMock) as mock_upsert,
        ):
            mock_rd.return_value.parse_url.return_value = mock_post
            mock_upsert.return_value = UpsertResult(total=2, inserted=2, skipped=0, skipped_indices=[])
            result = await pipeline.ingest_url("https://reddit.com/r/tech/comments/x", "gadget")

        assert result.success is True
        assert result.source_type == "reddit"

    @pytest.mark.asyncio()
    async def test_success_web(self, pipeline):
        mock_page = MagicMock()
        mock_page.text = _long_text()
        mock_page.author = "Writer"
        mock_page.language = "en"
        mock_page.date = "2026-01"

        with (
            patch.object(pipeline, "_get_web") as mock_wb,
            patch("reviewmind.ingestion.pipeline.upsert_chunks", new_callable=AsyncMock) as mock_upsert,
        ):
            mock_wb.return_value.parse_url.return_value = mock_page
            mock_upsert.return_value = UpsertResult(total=3, inserted=3, skipped=0, skipped_indices=[])
            result = await pipeline.ingest_url("https://rtings.com/headphones", "headphones")

        assert result.success is True
        assert result.source_type == "web"
        assert result.chunks_count == 3

    @pytest.mark.asyncio()
    async def test_scrape_returns_empty(self, pipeline):
        with patch.object(pipeline, "_get_web") as mock_wb:
            mock_wb.return_value.parse_url.return_value = None
            result = await pipeline.ingest_url("https://bad-site.com", "product")

        assert result.success is False
        assert "No text extracted" in result.error

    @pytest.mark.asyncio()
    async def test_scrape_raises_exception(self, pipeline):
        with patch.object(pipeline, "_get_web") as mock_wb:
            mock_wb.return_value.parse_url.side_effect = RuntimeError("connection error")
            result = await pipeline.ingest_url("https://crash.com", "product")

        assert result.success is False
        assert "Scrape failed" in result.error

    @pytest.mark.asyncio()
    async def test_clean_returns_empty(self, pipeline):
        with (
            patch.object(pipeline, "_get_web") as mock_wb,
            patch("reviewmind.ingestion.pipeline.clean_text", return_value=""),
        ):
            mock_page = MagicMock()
            mock_page.text = "short"
            mock_page.author = None
            mock_page.language = None
            mock_page.date = None
            mock_wb.return_value.parse_url.return_value = mock_page
            result = await pipeline.ingest_url("https://example.com", "product")

        assert result.success is False
        assert "too short after cleaning" in result.error

    @pytest.mark.asyncio()
    async def test_embedding_error(self, pipeline):
        from reviewmind.core.embeddings import EmbeddingError

        pipeline._embedding.embed_batch.side_effect = EmbeddingError("quota exceeded")

        with (
            patch.object(pipeline, "_get_web") as mock_wb,
            patch("reviewmind.ingestion.pipeline.clean_text", return_value=_long_text()),
        ):
            mock_page = MagicMock()
            mock_page.text = _long_text()
            mock_page.author = None
            mock_page.language = None
            mock_page.date = None
            mock_wb.return_value.parse_url.return_value = mock_page
            result = await pipeline.ingest_url("https://example.com", "product")

        assert result.success is False
        assert "Embedding failed" in result.error

    @pytest.mark.asyncio()
    async def test_upsert_error(self, pipeline):
        with (
            patch.object(pipeline, "_get_web") as mock_wb,
            patch("reviewmind.ingestion.pipeline.clean_text", return_value=_long_text()),
            patch("reviewmind.ingestion.pipeline.upsert_chunks", new_callable=AsyncMock) as mock_upsert,
        ):
            mock_page = MagicMock()
            mock_page.text = _long_text()
            mock_page.author = None
            mock_page.language = None
            mock_page.date = None
            mock_wb.return_value.parse_url.return_value = mock_page
            mock_upsert.side_effect = RuntimeError("qdrant down")
            result = await pipeline.ingest_url("https://example.com", "product")

        assert result.success is False
        assert "Upsert failed" in result.error


class TestIngestUrlSponsorDetection:
    """Verify that is_sponsored flag is correctly set."""

    @pytest.mark.asyncio()
    async def test_sponsored_flag_set(self):
        emb = AsyncMock()
        emb.embed_batch = AsyncMock(return_value=[[0.1] * 1536])
        emb.close = AsyncMock()
        p = _make_pipeline(embedding_service=emb)

        sponsored_text = "This video is sponsored by NordVPN. " + _long_text()

        with (
            patch.object(p, "_get_web") as mock_wb,
            patch("reviewmind.ingestion.pipeline.clean_text", return_value=sponsored_text),
            patch("reviewmind.ingestion.pipeline.upsert_chunks", new_callable=AsyncMock) as mock_upsert,
        ):
            mock_page = MagicMock()
            mock_page.text = sponsored_text
            mock_page.author = None
            mock_page.language = None
            mock_page.date = None
            mock_wb.return_value.parse_url.return_value = mock_page
            mock_upsert.return_value = UpsertResult(total=1, inserted=1, skipped=0, skipped_indices=[])

            result = await p.ingest_url("https://example.com/sponsored-review", "product")

        assert result.success is True
        assert result.is_sponsored is True

        # Verify payload has is_sponsored=True
        call_args = mock_upsert.call_args
        payloads = call_args[0][3]  # fourth positional arg (client, collection, vectors, payloads)
        assert all(pl.is_sponsored for pl in payloads)

    @pytest.mark.asyncio()
    async def test_non_sponsored_flag(self):
        emb = AsyncMock()
        emb.embed_batch = AsyncMock(return_value=[[0.1] * 1536])
        emb.close = AsyncMock()
        p = _make_pipeline(embedding_service=emb)

        clean_review = _long_text()

        with (
            patch.object(p, "_get_web") as mock_wb,
            patch("reviewmind.ingestion.pipeline.clean_text", return_value=clean_review),
            patch("reviewmind.ingestion.pipeline.upsert_chunks", new_callable=AsyncMock) as mock_upsert,
        ):
            mock_page = MagicMock()
            mock_page.text = clean_review
            mock_page.author = None
            mock_page.language = None
            mock_page.date = None
            mock_wb.return_value.parse_url.return_value = mock_page
            mock_upsert.return_value = UpsertResult(total=1, inserted=1, skipped=0, skipped_indices=[])

            result = await p.ingest_url("https://example.com/honest-review", "product")

        assert result.success is True
        assert result.is_sponsored is False


# ===================================================================
# ingest_url with PostgreSQL persistence
# ===================================================================


class TestIngestUrlWithDB:
    @pytest.mark.asyncio()
    async def test_persists_source(self):
        emb = AsyncMock()
        emb.embed_batch = AsyncMock(return_value=[[0.1] * 1536])
        emb.close = AsyncMock()

        mock_session = AsyncMock()
        mock_source = MagicMock()
        mock_source.id = 42

        p = _make_pipeline(embedding_service=emb, db_session=mock_session)

        with (
            patch.object(p, "_get_web") as mock_wb,
            patch("reviewmind.ingestion.pipeline.clean_text", return_value=_long_text()),
            patch("reviewmind.ingestion.pipeline.upsert_chunks", new_callable=AsyncMock) as mock_upsert,
            patch("reviewmind.ingestion.pipeline.SourceRepository") as mock_repo_cls,
        ):
            mock_page = MagicMock()
            mock_page.text = _long_text()
            mock_page.author = "Author"
            mock_page.language = "en"
            mock_page.date = "2026-01"
            mock_wb.return_value.parse_url.return_value = mock_page
            mock_upsert.return_value = UpsertResult(total=1, inserted=1, skipped=0, skipped_indices=[])

            repo_instance = AsyncMock()
            repo_instance.get_or_create.return_value = (mock_source, True)
            mock_repo_cls.return_value = repo_instance

            result = await p.ingest_url("https://rtings.com/review", "headphones")

        assert result.success is True
        assert result.source_id == 42
        repo_instance.get_or_create.assert_called_once()

    @pytest.mark.asyncio()
    async def test_updates_existing_source(self):
        emb = AsyncMock()
        emb.embed_batch = AsyncMock(return_value=[[0.1] * 1536])
        emb.close = AsyncMock()

        mock_session = AsyncMock()
        mock_source = MagicMock()
        mock_source.id = 7

        p = _make_pipeline(embedding_service=emb, db_session=mock_session)

        with (
            patch.object(p, "_get_web") as mock_wb,
            patch("reviewmind.ingestion.pipeline.clean_text", return_value=_long_text()),
            patch("reviewmind.ingestion.pipeline.upsert_chunks", new_callable=AsyncMock) as mock_upsert,
            patch("reviewmind.ingestion.pipeline.SourceRepository") as mock_repo_cls,
        ):
            mock_page = MagicMock()
            mock_page.text = _long_text()
            mock_page.author = None
            mock_page.language = None
            mock_page.date = None
            mock_wb.return_value.parse_url.return_value = mock_page
            mock_upsert.return_value = UpsertResult(total=1, inserted=1, skipped=0, skipped_indices=[])

            repo_instance = AsyncMock()
            repo_instance.get_or_create.return_value = (mock_source, False)  # already exists
            mock_repo_cls.return_value = repo_instance

            result = await p.ingest_url("https://rtings.com/review", "headphones")

        assert result.source_id == 7
        repo_instance.update.assert_called_once()

    @pytest.mark.asyncio()
    async def test_db_error_nonfatal(self):
        emb = AsyncMock()
        emb.embed_batch = AsyncMock(return_value=[[0.1] * 1536])
        emb.close = AsyncMock()

        mock_session = AsyncMock()
        p = _make_pipeline(embedding_service=emb, db_session=mock_session)

        with (
            patch.object(p, "_get_web") as mock_wb,
            patch("reviewmind.ingestion.pipeline.clean_text", return_value=_long_text()),
            patch("reviewmind.ingestion.pipeline.upsert_chunks", new_callable=AsyncMock) as mock_upsert,
            patch("reviewmind.ingestion.pipeline.SourceRepository") as mock_repo_cls,
        ):
            mock_page = MagicMock()
            mock_page.text = _long_text()
            mock_page.author = None
            mock_page.language = None
            mock_page.date = None
            mock_wb.return_value.parse_url.return_value = mock_page
            mock_upsert.return_value = UpsertResult(total=1, inserted=1, skipped=0, skipped_indices=[])

            repo_instance = AsyncMock()
            repo_instance.get_or_create.side_effect = RuntimeError("DB down")
            mock_repo_cls.return_value = repo_instance

            result = await p.ingest_url("https://example.com", "product")

        # Pipeline still succeeds — DB failure is non-fatal
        assert result.success is True
        assert result.source_id is None

    @pytest.mark.asyncio()
    async def test_no_db_session_skips_persist(self):
        emb = AsyncMock()
        emb.embed_batch = AsyncMock(return_value=[[0.1] * 1536])
        emb.close = AsyncMock()

        p = _make_pipeline(embedding_service=emb, db_session=None)

        with (
            patch.object(p, "_get_web") as mock_wb,
            patch("reviewmind.ingestion.pipeline.clean_text", return_value=_long_text()),
            patch("reviewmind.ingestion.pipeline.upsert_chunks", new_callable=AsyncMock) as mock_upsert,
        ):
            mock_page = MagicMock()
            mock_page.text = _long_text()
            mock_page.author = None
            mock_page.language = None
            mock_page.date = None
            mock_wb.return_value.parse_url.return_value = mock_page
            mock_upsert.return_value = UpsertResult(total=1, inserted=1, skipped=0, skipped_indices=[])

            result = await p.ingest_url("https://example.com", "product")

        assert result.success is True
        assert result.source_id is None


# ===================================================================
# ingest_urls — multiple URLs
# ===================================================================


class TestIngestUrls:
    @pytest.mark.asyncio()
    async def test_all_success(self):
        p = _make_pipeline()
        s1 = SourceIngestionResult(url="u1", success=True, chunks_count=3)
        s2 = SourceIngestionResult(url="u2", success=True, chunks_count=2)

        with patch.object(p, "ingest_url", new_callable=AsyncMock, side_effect=[s1, s2]):
            result = await p.ingest_urls(["u1", "u2"], "product")

        assert result.success_count == 2
        assert result.failed_count == 0
        assert result.chunks_count == 5
        assert result.failed_urls == []
        assert len(result.results) == 2

    @pytest.mark.asyncio()
    async def test_partial_failure(self):
        p = _make_pipeline()
        s1 = SourceIngestionResult(url="u1", success=True, chunks_count=3)
        s2 = SourceIngestionResult(url="u2", success=False, error="bad url")

        with patch.object(p, "ingest_url", new_callable=AsyncMock, side_effect=[s1, s2]):
            result = await p.ingest_urls(["u1", "u2"], "product")

        assert result.success_count == 1
        assert result.failed_count == 1
        assert result.chunks_count == 3
        assert result.failed_urls == ["u2"]

    @pytest.mark.asyncio()
    async def test_all_failure(self):
        p = _make_pipeline()
        s1 = SourceIngestionResult(url="u1", success=False, error="err1")
        s2 = SourceIngestionResult(url="u2", success=False, error="err2")

        with patch.object(p, "ingest_url", new_callable=AsyncMock, side_effect=[s1, s2]):
            result = await p.ingest_urls(["u1", "u2"], "product")

        assert result.success_count == 0
        assert result.failed_count == 2

    @pytest.mark.asyncio()
    async def test_empty_list(self):
        p = _make_pipeline()
        result = await p.ingest_urls([], "product")
        assert result.success_count == 0
        assert result.chunks_count == 0

    @pytest.mark.asyncio()
    async def test_session_id_passed(self):
        p = _make_pipeline()
        s = SourceIngestionResult(url="u", success=True, chunks_count=1)

        with patch.object(p, "ingest_url", new_callable=AsyncMock, return_value=s) as mock_ingest:
            await p.ingest_urls(["u"], "product", session_id="sess-123")

        mock_ingest.assert_called_once_with("u", "product", session_id="sess-123", is_curated=False)

    @pytest.mark.asyncio()
    async def test_is_curated_passed(self):
        p = _make_pipeline()
        s = SourceIngestionResult(url="u", success=True, chunks_count=1)

        with patch.object(p, "ingest_url", new_callable=AsyncMock, return_value=s) as mock_ingest:
            await p.ingest_urls(["u"], "product", is_curated=True)

        mock_ingest.assert_called_once_with("u", "product", session_id=None, is_curated=True)

    @pytest.mark.asyncio()
    async def test_failure_does_not_block_rest(self):
        """Failure of one URL should not prevent processing of subsequent URLs."""
        p = _make_pipeline()
        s1 = SourceIngestionResult(url="u1", success=False, error="bad")
        s2 = SourceIngestionResult(url="u2", success=True, chunks_count=5)
        s3 = SourceIngestionResult(url="u3", success=True, chunks_count=3)

        with patch.object(p, "ingest_url", new_callable=AsyncMock, side_effect=[s1, s2, s3]):
            result = await p.ingest_urls(["u1", "u2", "u3"], "product")

        assert result.success_count == 2
        assert result.failed_count == 1
        assert result.chunks_count == 8


# ===================================================================
# Payload construction
# ===================================================================


class TestPayloadConstruction:
    """Verify that ChunkPayload fields are correctly populated."""

    @pytest.mark.asyncio()
    async def test_payload_fields(self):
        emb = AsyncMock()
        emb.embed_batch = AsyncMock(return_value=[[0.1] * 1536])
        emb.close = AsyncMock()
        p = _make_pipeline(embedding_service=emb)

        with (
            patch.object(p, "_get_youtube") as mock_yt,
            patch("reviewmind.ingestion.pipeline.clean_text", return_value=_long_text()),
            patch("reviewmind.ingestion.pipeline.chunk_text") as mock_chunk,
            patch("reviewmind.ingestion.pipeline.upsert_chunks", new_callable=AsyncMock) as mock_upsert,
        ):
            mock_result = MagicMock()
            mock_result.text = _long_text()
            mock_result.language_code = "en"
            mock_result.language = "English"
            mock_result.extra_metadata = {"author": "Chan", "date": "2026-03"}
            mock_yt.return_value.get_transcript_by_url.return_value = mock_result

            from reviewmind.ingestion.chunker import Chunk

            mock_chunk.return_value = [Chunk(text="chunk text", chunk_index=0, metadata={})]
            mock_upsert.return_value = UpsertResult(total=1, inserted=1, skipped=0, skipped_indices=[])

            await p.ingest_url("https://youtube.com/watch?v=abc", "headphones", session_id="s1")

        # Inspect payload passed to upsert_chunks
        call_args = mock_upsert.call_args
        payloads = call_args[0][3]  # fourth positional arg: payloads list
        assert len(payloads) == 1
        pl = payloads[0]
        assert isinstance(pl, ChunkPayload)
        assert pl.source_url == "https://youtube.com/watch?v=abc"
        assert pl.source_type == "youtube"
        assert pl.product_query == "headphones"
        assert pl.chunk_index == 0
        assert pl.language == "en"
        assert pl.author == "Chan"
        assert pl.date == "2026-03"
        assert pl.session_id == "s1"
        assert pl.is_curated is False

    @pytest.mark.asyncio()
    async def test_payload_curated(self):
        emb = AsyncMock()
        emb.embed_batch = AsyncMock(return_value=[[0.1] * 1536])
        emb.close = AsyncMock()
        p = _make_pipeline(embedding_service=emb)

        with (
            patch.object(p, "_get_web") as mock_wb,
            patch("reviewmind.ingestion.pipeline.clean_text", return_value=_long_text()),
            patch("reviewmind.ingestion.pipeline.chunk_text") as mock_chunk,
            patch("reviewmind.ingestion.pipeline.upsert_chunks", new_callable=AsyncMock) as mock_upsert,
        ):
            mock_page = MagicMock()
            mock_page.text = _long_text()
            mock_page.author = None
            mock_page.language = None
            mock_page.date = None
            mock_wb.return_value.parse_url.return_value = mock_page

            from reviewmind.ingestion.chunker import Chunk

            mock_chunk.return_value = [Chunk(text="c", chunk_index=0)]
            mock_upsert.return_value = UpsertResult(total=1, inserted=1, skipped=0, skipped_indices=[])

            await p.ingest_url("https://curated.com", "laptops", is_curated=True)

        pl = mock_upsert.call_args[0][3][0]
        assert pl.is_curated is True


# ===================================================================
# Lifecycle
# ===================================================================


class TestLifecycle:
    @pytest.mark.asyncio()
    async def test_close_releases_owned_embedding(self):
        p = _make_pipeline()
        # Force lazy creation
        with patch("reviewmind.ingestion.pipeline.EmbeddingService") as mock_cls:
            emb = AsyncMock()
            mock_cls.return_value = emb
            _ = p.embedding_service
        await p.close()
        emb.close.assert_called_once()
        assert p._embedding is None

    @pytest.mark.asyncio()
    async def test_close_skips_provided_embedding(self):
        emb = AsyncMock()
        p = _make_pipeline(embedding_service=emb)
        await p.close()
        emb.close.assert_not_called()

    @pytest.mark.asyncio()
    async def test_context_manager(self):
        emb = AsyncMock()
        async with _make_pipeline(embedding_service=emb) as p:
            assert p._qdrant is not None
        # After exit, close() was called

    @pytest.mark.asyncio()
    async def test_close_without_embedding(self):
        """close() is safe even if embedding was never accessed."""
        p = _make_pipeline()
        await p.close()  # Should not raise


# ===================================================================
# Exports
# ===================================================================


class TestIngestionExports:
    def test_pipeline_exported(self):
        from reviewmind.ingestion import IngestionPipeline

        assert IngestionPipeline is not None

    def test_result_exported(self):
        from reviewmind.ingestion import IngestionResult

        assert IngestionResult is not None

    def test_source_result_exported(self):
        from reviewmind.ingestion import SourceIngestionResult

        assert SourceIngestionResult is not None

    def test_detect_url_type_exported(self):
        from reviewmind.ingestion import detect_url_type

        assert callable(detect_url_type)

    def test_all_previous_exports_intact(self):
        from reviewmind.ingestion import chunk_text, clean_text, detect_sponsor

        assert all(callable(f) for f in [chunk_text, clean_text, detect_sponsor])


# ===================================================================
# Integration scenarios
# ===================================================================


class TestIntegrationScenarios:
    """Higher-level integration-style tests using mocked externals."""

    @pytest.mark.asyncio()
    async def test_multiple_urls_mixed_types(self):
        """ingest_urls with YouTube + Reddit + Web URLs."""
        emb = AsyncMock()
        emb.embed_batch = AsyncMock(return_value=[[0.1] * 1536])
        emb.close = AsyncMock()
        p = _make_pipeline(embedding_service=emb)

        results = []
        for url_type, url in [
            ("youtube", "https://youtube.com/watch?v=1"),
            ("reddit", "https://reddit.com/r/headphones/comments/x"),
            ("web", "https://rtings.com/review"),
        ]:
            results.append(SourceIngestionResult(url=url, success=True, source_type=url_type, chunks_count=2))

        with patch.object(p, "ingest_url", new_callable=AsyncMock, side_effect=results):
            agg = await p.ingest_urls(
                [
                    "https://youtube.com/watch?v=1",
                    "https://reddit.com/r/headphones/comments/x",
                    "https://rtings.com/review",
                ],
                "headphones",
            )

        assert agg.success_count == 3
        assert agg.chunks_count == 6

    @pytest.mark.asyncio()
    async def test_dedup_in_upsert(self):
        """Verify upsert with dedup (skipped chunks) is reflected in result."""
        emb = AsyncMock()
        emb.embed_batch = AsyncMock(return_value=[[0.1] * 1536])
        emb.close = AsyncMock()
        p = _make_pipeline(embedding_service=emb)

        with (
            patch.object(p, "_get_web") as mock_wb,
            patch("reviewmind.ingestion.pipeline.clean_text", return_value=_long_text()),
            patch("reviewmind.ingestion.pipeline.upsert_chunks", new_callable=AsyncMock) as mock_upsert,
        ):
            mock_page = MagicMock()
            mock_page.text = _long_text()
            mock_page.author = None
            mock_page.language = None
            mock_page.date = None
            mock_wb.return_value.parse_url.return_value = mock_page
            mock_upsert.return_value = UpsertResult(total=5, inserted=2, skipped=3, skipped_indices=[0, 1, 2])

            result = await p.ingest_url("https://example.com", "product")

        assert result.success is True
        assert result.chunks_count == 2  # only inserted, not total

    @pytest.mark.asyncio()
    async def test_end_to_end_pipeline_mock(self):
        """Full pipeline mock: scrape → clean → sponsor → chunk → embed → upsert → DB."""
        emb = AsyncMock()
        emb.embed_batch = AsyncMock(return_value=[[0.1] * 1536, [0.2] * 1536])
        emb.close = AsyncMock()

        mock_session = AsyncMock()
        p = _make_pipeline(embedding_service=emb, db_session=mock_session)

        mock_source = MagicMock()
        mock_source.id = 99

        with (
            patch.object(p, "_get_youtube") as mock_yt,
            patch("reviewmind.ingestion.pipeline.clean_text") as mock_clean,
            patch("reviewmind.ingestion.pipeline.detect_sponsor_detailed") as mock_sponsor,
            patch("reviewmind.ingestion.pipeline.chunk_text") as mock_chunk,
            patch("reviewmind.ingestion.pipeline.upsert_chunks", new_callable=AsyncMock) as mock_upsert,
            patch("reviewmind.ingestion.pipeline.SourceRepository") as mock_repo_cls,
        ):
            # Scrape
            vid = MagicMock()
            vid.text = "video text"
            vid.language_code = "en"
            vid.language = "English"
            vid.extra_metadata = {"author": "Tech"}
            mock_yt.return_value.get_transcript_by_url.return_value = vid

            # Clean
            mock_clean.return_value = "cleaned text"

            # Sponsor
            from reviewmind.ingestion.sponsor import SponsorDetectionResult

            mock_sponsor.return_value = SponsorDetectionResult(is_sponsored=False)

            # Chunk
            from reviewmind.ingestion.chunker import Chunk

            mock_chunk.return_value = [
                Chunk(text="chunk 0", chunk_index=0),
                Chunk(text="chunk 1", chunk_index=1),
            ]

            # Upsert
            mock_upsert.return_value = UpsertResult(total=2, inserted=2, skipped=0, skipped_indices=[])

            # DB
            repo_instance = AsyncMock()
            repo_instance.get_or_create.return_value = (mock_source, True)
            mock_repo_cls.return_value = repo_instance

            result = await p.ingest_url(
                "https://youtube.com/watch?v=abc",
                "headphones review",
                session_id="session-1",
            )

        assert result.success is True
        assert result.source_type == "youtube"
        assert result.chunks_count == 2
        assert result.source_id == 99
        assert result.is_sponsored is False

        # Verify call chain
        mock_clean.assert_called_once()
        mock_sponsor.assert_called_once()
        mock_chunk.assert_called_once()
        emb.embed_batch.assert_called_once_with(["chunk 0", "chunk 1"])
        mock_upsert.assert_called_once()
        repo_instance.get_or_create.assert_called_once()

    @pytest.mark.asyncio()
    async def test_no_chunks_produced(self):
        """When chunking returns empty list."""
        emb = AsyncMock()
        emb.embed_batch = AsyncMock()
        emb.close = AsyncMock()
        p = _make_pipeline(embedding_service=emb)

        with (
            patch.object(p, "_get_web") as mock_wb,
            patch("reviewmind.ingestion.pipeline.clean_text", return_value="valid text"),
            patch("reviewmind.ingestion.pipeline.chunk_text", return_value=[]),
        ):
            mock_page = MagicMock()
            mock_page.text = "valid text"
            mock_page.author = None
            mock_page.language = None
            mock_page.date = None
            mock_wb.return_value.parse_url.return_value = mock_page

            result = await p.ingest_url("https://example.com", "product")

        assert result.success is False
        assert "No chunks" in result.error
        emb.embed_batch.assert_not_called()
