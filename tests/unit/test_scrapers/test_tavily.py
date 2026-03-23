"""Unit tests for reviewmind.scrapers.tavily — TavilyScraper."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reviewmind.scrapers.tavily import (
    DEFAULT_MAX_RESULTS,
    DEFAULT_SEARCH_DEPTH,
    DEFAULT_TIMEOUT,
    MIN_CONTENT_LENGTH,
    TavilyResult,
    TavilyScraper,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_api_response(
    results: list[dict] | None = None,
    answer: str | None = None,
) -> dict:
    """Build a fake Tavily API response dict."""
    resp: dict = {"results": results or []}
    if answer:
        resp["answer"] = answer
    return resp


def _make_result_item(
    url: str = "https://example.com/review",
    title: str = "Test Review",
    content: str = "A" * 100,
    score: float = 0.85,
    **extra: object,
) -> dict:
    """Build a single result item as returned by the Tavily API."""
    item: dict = {
        "url": url,
        "title": title,
        "content": content,
        "score": score,
    }
    item.update(extra)
    return item


# ── TestConstants ────────────────────────────────────────────────────────────


class TestConstants:
    """Verify module-level constants."""

    def test_default_max_results(self):
        assert DEFAULT_MAX_RESULTS == 5

    def test_default_search_depth(self):
        assert DEFAULT_SEARCH_DEPTH == "basic"

    def test_min_content_length(self):
        assert MIN_CONTENT_LENGTH == 50

    def test_default_timeout(self):
        assert DEFAULT_TIMEOUT == 30.0


# ── TestTavilyResult ────────────────────────────────────────────────────────


class TestTavilyResult:
    """Verify the TavilyResult dataclass."""

    def test_minimal_creation(self):
        result = TavilyResult(url="https://example.com")
        assert result.url == "https://example.com"
        assert result.title == ""
        assert result.content == ""
        assert result.score == 0.0
        assert result.raw_content is None
        assert result.extra == {}

    def test_full_creation(self):
        result = TavilyResult(
            url="https://example.com/r",
            title="Great Review",
            content="This is a great product.",
            score=0.92,
            raw_content="<p>This is a great product.</p>",
            extra={"published_date": "2026-01-15"},
        )
        assert result.url == "https://example.com/r"
        assert result.title == "Great Review"
        assert result.content == "This is a great product."
        assert result.score == 0.92
        assert result.raw_content == "<p>This is a great product.</p>"
        assert result.extra["published_date"] == "2026-01-15"

    def test_word_count_with_content(self):
        result = TavilyResult(url="https://a.com", content="one two three four five")
        assert result.word_count == 5

    def test_word_count_empty(self):
        result = TavilyResult(url="https://a.com", content="")
        assert result.word_count == 0

    def test_extra_default_independent(self):
        r1 = TavilyResult(url="a")
        r2 = TavilyResult(url="b")
        r1.extra["k"] = "v"
        assert r2.extra == {}


# ── TestTavilyScraperInit ───────────────────────────────────────────────────


class TestTavilyScraperInit:
    """Verify TavilyScraper initialization."""

    def test_default_params(self):
        scraper = TavilyScraper(api_key="test-key")
        assert scraper._api_key == "test-key"
        assert scraper._max_results == DEFAULT_MAX_RESULTS
        assert scraper._search_depth == DEFAULT_SEARCH_DEPTH
        assert scraper._timeout == DEFAULT_TIMEOUT
        assert scraper._client is None

    def test_custom_params(self):
        scraper = TavilyScraper(
            api_key="k",
            max_results=10,
            search_depth="advanced",
            timeout=60.0,
        )
        assert scraper._max_results == 10
        assert scraper._search_depth == "advanced"
        assert scraper._timeout == 60.0

    def test_no_api_key(self):
        scraper = TavilyScraper()
        assert scraper._api_key is None


# ── TestGetApiKey ────────────────────────────────────────────────────────────


class TestGetApiKey:
    """Verify _get_api_key with explicit and config-based keys."""

    def test_explicit_key_used(self):
        scraper = TavilyScraper(api_key="explicit-key")
        assert scraper._get_api_key() == "explicit-key"

    def test_falls_back_to_config(self):
        scraper = TavilyScraper()
        with patch("reviewmind.config.settings", MagicMock(tavily_api_key="from-config")):
            assert scraper._get_api_key() == "from-config"


# ── TestGetClient ────────────────────────────────────────────────────────────


class TestGetClient:
    """Verify lazy client creation."""

    @patch("reviewmind.scrapers.tavily.AsyncTavilyClient")
    def test_creates_client_lazily(self, mock_cls):
        scraper = TavilyScraper(api_key="key")
        client = scraper._get_client()
        mock_cls.assert_called_once_with(api_key="key")
        assert client is mock_cls.return_value

    @patch("reviewmind.scrapers.tavily.AsyncTavilyClient")
    def test_reuses_client(self, mock_cls):
        scraper = TavilyScraper(api_key="key")
        c1 = scraper._get_client()
        c2 = scraper._get_client()
        assert c1 is c2
        mock_cls.assert_called_once()

    def test_raises_on_empty_key(self):
        scraper = TavilyScraper(api_key="")
        with patch("reviewmind.config.settings", MagicMock(tavily_api_key="")):
            with pytest.raises(ValueError, match="Tavily API key is not configured"):
                scraper._get_client()


# ── TestSearch ───────────────────────────────────────────────────────────────


class TestSearch:
    """Verify TavilyScraper.search() with mocked API calls."""

    @pytest.fixture
    def scraper(self):
        return TavilyScraper(api_key="test-key")

    @pytest.fixture
    def mock_client(self):
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_successful_search(self, scraper, mock_client):
        response = _make_api_response(
            results=[
                _make_result_item(url="https://a.com", content="X" * 100, score=0.9),
                _make_result_item(url="https://b.com", content="Y" * 80, score=0.7),
            ]
        )
        mock_client.search = AsyncMock(return_value=response)
        scraper._client = mock_client

        results = await scraper.search("best headphones 2026")

        assert len(results) == 2
        assert results[0].url == "https://a.com"
        assert results[0].score == 0.9
        assert results[1].url == "https://b.com"

    @pytest.mark.asyncio
    async def test_empty_query_returns_empty(self, scraper):
        results = await scraper.search("")
        assert results == []

    @pytest.mark.asyncio
    async def test_whitespace_query_returns_empty(self, scraper):
        results = await scraper.search("   ")
        assert results == []

    @pytest.mark.asyncio
    async def test_filters_short_content(self, scraper, mock_client):
        response = _make_api_response(
            results=[
                _make_result_item(url="https://a.com", content="Too short"),
                _make_result_item(url="https://b.com", content="A" * 100, score=0.8),
            ]
        )
        mock_client.search = AsyncMock(return_value=response)
        scraper._client = mock_client

        results = await scraper.search("query")
        assert len(results) == 1
        assert results[0].url == "https://b.com"

    @pytest.mark.asyncio
    async def test_results_sorted_by_score(self, scraper, mock_client):
        response = _make_api_response(
            results=[
                _make_result_item(url="https://low.com", content="A" * 100, score=0.3),
                _make_result_item(url="https://high.com", content="B" * 100, score=0.95),
                _make_result_item(url="https://mid.com", content="C" * 100, score=0.6),
            ]
        )
        mock_client.search = AsyncMock(return_value=response)
        scraper._client = mock_client

        results = await scraper.search("query")
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_custom_max_results(self, scraper, mock_client):
        mock_client.search = AsyncMock(return_value=_make_api_response())
        scraper._client = mock_client

        await scraper.search("query", max_results=10)

        call_kwargs = mock_client.search.call_args.kwargs
        assert call_kwargs["max_results"] == 10

    @pytest.mark.asyncio
    async def test_custom_search_depth(self, scraper, mock_client):
        mock_client.search = AsyncMock(return_value=_make_api_response())
        scraper._client = mock_client

        await scraper.search("query", search_depth="advanced")

        call_kwargs = mock_client.search.call_args.kwargs
        assert call_kwargs["search_depth"] == "advanced"

    @pytest.mark.asyncio
    async def test_default_params_passed(self, scraper, mock_client):
        mock_client.search = AsyncMock(return_value=_make_api_response())
        scraper._client = mock_client

        await scraper.search("query")

        call_kwargs = mock_client.search.call_args.kwargs
        assert call_kwargs["max_results"] == DEFAULT_MAX_RESULTS
        assert call_kwargs["search_depth"] == DEFAULT_SEARCH_DEPTH
        assert call_kwargs["timeout"] == DEFAULT_TIMEOUT

    @pytest.mark.asyncio
    async def test_include_raw_content(self, scraper, mock_client):
        response = _make_api_response(
            results=[
                {
                    "url": "https://a.com",
                    "title": "T",
                    "content": "A" * 100,
                    "score": 0.8,
                    "raw_content": "<p>raw html</p>",
                }
            ]
        )
        mock_client.search = AsyncMock(return_value=response)
        scraper._client = mock_client

        results = await scraper.search("query", include_raw_content=True)
        assert len(results) == 1
        assert results[0].raw_content == "<p>raw html</p>"

    @pytest.mark.asyncio
    async def test_no_raw_content_by_default(self, scraper, mock_client):
        response = _make_api_response(
            results=[
                {
                    "url": "https://a.com",
                    "title": "T",
                    "content": "A" * 100,
                    "score": 0.8,
                    "raw_content": "<p>raw</p>",
                }
            ]
        )
        mock_client.search = AsyncMock(return_value=response)
        scraper._client = mock_client

        results = await scraper.search("query")
        assert results[0].raw_content is None

    @pytest.mark.asyncio
    async def test_extra_fields_preserved(self, scraper, mock_client):
        response = _make_api_response(
            results=[
                {
                    "url": "https://a.com",
                    "title": "T",
                    "content": "A" * 100,
                    "score": 0.8,
                    "published_date": "2026-03-01",
                    "domain": "a.com",
                }
            ]
        )
        mock_client.search = AsyncMock(return_value=response)
        scraper._client = mock_client

        results = await scraper.search("query")
        assert results[0].extra["published_date"] == "2026-03-01"
        assert results[0].extra["domain"] == "a.com"


# ── TestSearchErrors ─────────────────────────────────────────────────────────


class TestSearchErrors:
    """Verify graceful error handling during search."""

    @pytest.fixture
    def scraper(self):
        return TavilyScraper(api_key="test-key")

    @pytest.fixture
    def mock_client(self):
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_invalid_api_key_error(self, scraper, mock_client):
        from tavily import InvalidAPIKeyError

        mock_client.search = AsyncMock(side_effect=InvalidAPIKeyError("bad key"))
        scraper._client = mock_client

        results = await scraper.search("query")
        assert results == []

    @pytest.mark.asyncio
    async def test_missing_api_key_error(self, scraper, mock_client):
        from tavily import MissingAPIKeyError

        mock_client.search = AsyncMock(side_effect=MissingAPIKeyError())
        scraper._client = mock_client

        results = await scraper.search("query")
        assert results == []

    @pytest.mark.asyncio
    async def test_usage_limit_error(self, scraper, mock_client):
        from tavily import UsageLimitExceededError

        mock_client.search = AsyncMock(side_effect=UsageLimitExceededError("limit"))
        scraper._client = mock_client

        results = await scraper.search("query")
        assert results == []

    @pytest.mark.asyncio
    async def test_generic_exception(self, scraper, mock_client):
        mock_client.search = AsyncMock(side_effect=RuntimeError("network failure"))
        scraper._client = mock_client

        results = await scraper.search("query")
        assert results == []

    @pytest.mark.asyncio
    async def test_no_api_key_configured(self):
        scraper = TavilyScraper(api_key="")
        with patch("reviewmind.config.settings", MagicMock(tavily_api_key="")):
            results = await scraper.search("query")
            assert results == []


# ── TestParseResponse ────────────────────────────────────────────────────────


class TestParseResponse:
    """Verify _parse_response helper."""

    def test_empty_results(self):
        scraper = TavilyScraper(api_key="k")
        parsed = scraper._parse_response({"results": []})
        assert parsed == []

    def test_missing_results_key(self):
        scraper = TavilyScraper(api_key="k")
        parsed = scraper._parse_response({})
        assert parsed == []

    def test_filters_short_content(self):
        scraper = TavilyScraper(api_key="k")
        response = {
            "results": [
                {"url": "https://a.com", "content": "short", "score": 0.9},
                {"url": "https://b.com", "content": "A" * 100, "score": 0.8},
            ]
        }
        parsed = scraper._parse_response(response)
        assert len(parsed) == 1
        assert parsed[0].url == "https://b.com"

    def test_score_sorting(self):
        scraper = TavilyScraper(api_key="k")
        response = {
            "results": [
                {"url": "https://c.com", "content": "C" * 100, "score": 0.3},
                {"url": "https://a.com", "content": "A" * 100, "score": 0.9},
                {"url": "https://b.com", "content": "B" * 100, "score": 0.6},
            ]
        }
        parsed = scraper._parse_response(response)
        assert [r.score for r in parsed] == [0.9, 0.6, 0.3]

    def test_missing_fields_default(self):
        scraper = TavilyScraper(api_key="k")
        response = {"results": [{"content": "A" * 100}]}
        parsed = scraper._parse_response(response)
        assert len(parsed) == 1
        assert parsed[0].url == ""
        assert parsed[0].title == ""
        assert parsed[0].score == 0.0


# ── TestScrapersExports ──────────────────────────────────────────────────────


class TestScrapersExports:
    """Verify scrapers __init__.py exports for Tavily."""

    def test_tavily_scraper_importable(self):
        from reviewmind.scrapers import TavilyScraper as Cls

        assert Cls is not None

    def test_tavily_result_importable(self):
        from reviewmind.scrapers import TavilyResult as Cls

        assert Cls is not None

    def test_tavily_constants_importable(self):
        from reviewmind.scrapers import (
            TAVILY_DEFAULT_MAX_RESULTS,
            TAVILY_DEFAULT_SEARCH_DEPTH,
            TAVILY_DEFAULT_TIMEOUT,
            TAVILY_MIN_CONTENT_LENGTH,
        )

        assert TAVILY_DEFAULT_MAX_RESULTS == 5
        assert TAVILY_DEFAULT_SEARCH_DEPTH == "basic"
        assert TAVILY_DEFAULT_TIMEOUT == 30.0
        assert TAVILY_MIN_CONTENT_LENGTH == 50


# ── TestRAGTavilyIntegration ─────────────────────────────────────────────────


class TestRAGTavilyIntegration:
    """Verify Tavily fallback integration in the RAG pipeline."""

    @pytest.fixture
    def mock_embedding(self):
        svc = AsyncMock()
        svc.embed_text = AsyncMock(return_value=[0.1] * 1536)
        svc.close = AsyncMock()
        return svc

    @pytest.fixture
    def mock_llm(self):
        llm = AsyncMock()
        llm.generate_analysis = AsyncMock(return_value="Analysis with tavily data")
        llm.close = AsyncMock()
        return llm

    @pytest.fixture
    def mock_qdrant(self):
        return AsyncMock()

    @pytest.fixture
    def low_confidence_results(self):
        from reviewmind.vectorstore.search import SearchResult

        # Use curated chunks so auto_crawled_count == 0 → Tavily fallback fires
        return [
            SearchResult(
                text=f"weak chunk {i}",
                score=0.5 - i * 0.1,
                source_url=f"https://weak{i}.com",
                source_type="curated",
                is_curated=True,
            )
            for i in range(2)
        ]

    @pytest.fixture
    def low_confidence_auto_crawled_results(self):
        from reviewmind.vectorstore.search import SearchResult

        # Auto-crawled chunks → Tavily should be skipped (data already exists)
        return [
            SearchResult(
                text=f"weak chunk {i}",
                score=0.5 - i * 0.1,
                source_url=f"https://weak{i}.com",
                source_type="web",
            )
            for i in range(2)
        ]

    @pytest.fixture
    def tavily_results(self):
        return [
            TavilyResult(
                url="https://tavily1.com/review",
                title="Tavily Review 1",
                content="Excellent product with great features " * 5,
                score=0.88,
            ),
            TavilyResult(
                url="https://tavily2.com/review",
                title="Tavily Review 2",
                content="Good value for money " * 5,
                score=0.75,
            ),
        ]

    @pytest.mark.asyncio
    async def test_tavily_fallback_triggered_on_low_confidence(
        self, mock_qdrant, mock_embedding, mock_llm, low_confidence_results, tavily_results
    ):
        from reviewmind.core.rag import RAGPipeline

        mock_tavily = AsyncMock(return_value=tavily_results)

        with (
            patch(
                "reviewmind.core.rag.hybrid_search",
                new_callable=AsyncMock,
                return_value=low_confidence_results,
            ),
            patch.object(
                RAGPipeline,
                "_tavily_fallback",
                mock_tavily,
            ),
        ):
            pipeline = RAGPipeline(mock_qdrant, mock_embedding, mock_llm)
            resp = await pipeline.query("unknown product review")

        assert resp.used_tavily is True
        mock_tavily.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_tavily_not_triggered_on_high_confidence(self, mock_qdrant, mock_embedding, mock_llm):
        from reviewmind.core.rag import RAGPipeline
        from reviewmind.vectorstore.search import SearchResult

        confident_results = [
            SearchResult(text=f"good chunk {i}", score=0.9 - i * 0.02, source_url=f"https://good{i}.com")
            for i in range(5)
        ]

        mock_tavily = AsyncMock(return_value=[])

        with (
            patch(
                "reviewmind.core.rag.hybrid_search",
                new_callable=AsyncMock,
                return_value=confident_results,
            ),
            patch.object(
                RAGPipeline,
                "_tavily_fallback",
                mock_tavily,
            ),
        ):
            pipeline = RAGPipeline(mock_qdrant, mock_embedding, mock_llm)
            resp = await pipeline.query("well-known product")

        assert resp.used_tavily is False
        mock_tavily.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_tavily_fallback_returns_empty(self, mock_qdrant, mock_embedding, mock_llm, low_confidence_results):
        from reviewmind.core.rag import RAGPipeline

        mock_tavily = AsyncMock(return_value=[])

        with (
            patch(
                "reviewmind.core.rag.hybrid_search",
                new_callable=AsyncMock,
                return_value=low_confidence_results,
            ),
            patch.object(
                RAGPipeline,
                "_tavily_fallback",
                mock_tavily,
            ),
        ):
            pipeline = RAGPipeline(mock_qdrant, mock_embedding, mock_llm)
            resp = await pipeline.query("no tavily data product")

        assert resp.used_tavily is False  # empty results → not used
        assert resp.answer != ""  # LLM still called

    @pytest.mark.asyncio
    async def test_tavily_skipped_when_auto_crawled_has_data(
        self, mock_qdrant, mock_embedding, mock_llm, low_confidence_auto_crawled_results
    ):
        """Tavily should NOT fire when auto_crawled already has data (even low confidence)."""
        from reviewmind.core.rag import RAGPipeline

        mock_tavily = AsyncMock(return_value=[])

        with (
            patch(
                "reviewmind.core.rag.hybrid_search",
                new_callable=AsyncMock,
                return_value=low_confidence_auto_crawled_results,
            ),
            patch.object(
                RAGPipeline,
                "_tavily_fallback",
                mock_tavily,
            ),
        ):
            pipeline = RAGPipeline(mock_qdrant, mock_embedding, mock_llm)
            resp = await pipeline.query("already crawled product")

        assert resp.used_tavily is False
        mock_tavily.assert_not_awaited()
        assert resp.answer != ""

    @pytest.mark.asyncio
    async def test_tavily_results_merged_into_context(
        self, mock_qdrant, mock_embedding, mock_llm, low_confidence_results, tavily_results
    ):
        from reviewmind.core.rag import RAGPipeline

        mock_tavily = AsyncMock(return_value=tavily_results)

        with (
            patch(
                "reviewmind.core.rag.hybrid_search",
                new_callable=AsyncMock,
                return_value=low_confidence_results,
            ),
            patch.object(
                RAGPipeline,
                "_tavily_fallback",
                mock_tavily,
            ),
        ):
            pipeline = RAGPipeline(mock_qdrant, mock_embedding, mock_llm)
            resp = await pipeline.query("test product")

        # Original 2 results + 2 tavily results = 4 total in context
        assert resp.chunks_count == 4
        # Sources should include tavily URLs
        assert any("tavily" in s for s in resp.sources)

    @pytest.mark.asyncio
    async def test_tavily_fallback_method_no_api_key(self, mock_qdrant, mock_embedding, mock_llm):
        """When TAVILY_API_KEY is empty, _tavily_fallback returns []."""
        from reviewmind.core.rag import RAGPipeline

        pipeline = RAGPipeline(mock_qdrant, mock_embedding, mock_llm)
        with patch("reviewmind.config.settings", MagicMock(tavily_api_key="")):
            result = await pipeline._tavily_fallback("query", MagicMock())
        assert result == []

    @pytest.mark.asyncio
    async def test_tavily_fallback_method_with_api_key(self, mock_qdrant, mock_embedding, mock_llm, tavily_results):
        """When API key is available, _tavily_fallback calls TavilyScraper.search."""
        from reviewmind.core.rag import RAGPipeline

        mock_scraper_instance = AsyncMock()
        mock_scraper_instance.search = AsyncMock(return_value=tavily_results)

        pipeline = RAGPipeline(mock_qdrant, mock_embedding, mock_llm)
        with (
            patch("reviewmind.config.settings", MagicMock(tavily_api_key="real-key")),
            patch("reviewmind.scrapers.tavily.TavilyScraper", return_value=mock_scraper_instance),
        ):
            result = await pipeline._tavily_fallback("best headphones", MagicMock())

        assert len(result) == 2
        mock_scraper_instance.search.assert_awaited_once_with("best headphones")

    @pytest.mark.asyncio
    async def test_tavily_fallback_method_error_graceful(self, mock_qdrant, mock_embedding, mock_llm):
        """If TavilyScraper.search raises, _tavily_fallback returns []."""
        from reviewmind.core.rag import RAGPipeline

        mock_scraper_instance = AsyncMock()
        mock_scraper_instance.search = AsyncMock(side_effect=RuntimeError("network error"))

        pipeline = RAGPipeline(mock_qdrant, mock_embedding, mock_llm)
        with (
            patch("reviewmind.config.settings", MagicMock(tavily_api_key="key")),
            patch("reviewmind.scrapers.tavily.TavilyScraper", return_value=mock_scraper_instance),
        ):
            result = await pipeline._tavily_fallback("query", MagicMock())

        assert result == []


# ── TestTavilyResultsToSearchResults ─────────────────────────────────────────


class TestTavilyResultsToSearchResults:
    """Verify _tavily_results_to_search_results conversion helper."""

    def test_conversion(self):
        from reviewmind.core.rag import _tavily_results_to_search_results

        tavily = [
            TavilyResult(
                url="https://t.com/1",
                title="Result 1",
                content="Content of result one " * 5,
                score=0.8,
            ),
        ]
        search_results = _tavily_results_to_search_results(tavily)
        assert len(search_results) == 1
        sr = search_results[0]
        assert sr.text == tavily[0].content
        assert sr.source_url == "https://t.com/1"
        assert sr.source_type == "tavily"
        assert sr.is_curated is False
        assert sr.is_sponsored is False
        assert sr.collection == "tavily_fallback"
        assert sr.score == 0.8
        assert sr.extra["title"] == "Result 1"

    def test_filters_empty_content(self):
        from reviewmind.core.rag import _tavily_results_to_search_results

        tavily = [
            TavilyResult(url="https://t.com/1", content=""),
            TavilyResult(url="https://t.com/2", content="Has content"),
        ]
        search_results = _tavily_results_to_search_results(tavily)
        assert len(search_results) == 1
        assert search_results[0].source_url == "https://t.com/2"

    def test_empty_input(self):
        from reviewmind.core.rag import _tavily_results_to_search_results

        assert _tavily_results_to_search_results([]) == []

    def test_multiple_results(self):
        from reviewmind.core.rag import _tavily_results_to_search_results

        tavily = [TavilyResult(url=f"https://t.com/{i}", content=f"Content {i}", score=0.9 - i * 0.1) for i in range(3)]
        search_results = _tavily_results_to_search_results(tavily)
        assert len(search_results) == 3
        assert all(sr.source_type == "tavily" for sr in search_results)


# ── TestRAGResponseUsedTavily ────────────────────────────────────────────────


class TestRAGResponseUsedTavily:
    """Verify the used_tavily field on RAGResponse."""

    def test_default_false(self):
        from reviewmind.core.rag import RAGResponse

        resp = RAGResponse(answer="test")
        assert resp.used_tavily is False

    def test_set_true(self):
        from reviewmind.core.rag import RAGResponse

        resp = RAGResponse(answer="test", used_tavily=True)
        assert resp.used_tavily is True


# ── TestIntegrationScenarios ─────────────────────────────────────────────────


class TestIntegrationScenarios:
    """End-to-end-like scenarios testing the full Tavily workflow."""

    @pytest.mark.asyncio
    async def test_tavily_search_happy_path(self):
        """Full search → parse → return TavilyResult list."""
        scraper = TavilyScraper(api_key="key")
        response = _make_api_response(
            results=[
                _make_result_item(
                    url="https://rtings.com/headphones/sony-xm5",
                    title="Sony XM5 Review",
                    content="The Sony WH-1000XM5 features excellent noise cancelling. " * 3,
                    score=0.92,
                ),
                _make_result_item(
                    url="https://wirecutter.com/best-headphones",
                    title="Best Headphones 2026",
                    content="Our top pick for noise cancelling headphones. " * 3,
                    score=0.88,
                ),
            ]
        )
        mock_client = AsyncMock()
        mock_client.search = AsyncMock(return_value=response)
        scraper._client = mock_client

        results = await scraper.search("Sony WH-1000XM5 review")

        assert len(results) == 2
        assert results[0].score >= results[1].score
        assert "rtings.com" in results[0].url
        assert results[0].word_count > 0

    @pytest.mark.asyncio
    async def test_graceful_degradation_no_key(self):
        """When API key is missing, search returns empty list without crashing."""
        scraper = TavilyScraper(api_key="")
        with patch("reviewmind.config.settings", MagicMock(tavily_api_key="")):
            results = await scraper.search("query")
        assert results == []

    def test_source_type_tavily_matches_collection_enum(self):
        """Verify 'tavily' is a valid SourceType in collections."""
        from reviewmind.vectorstore.collections import SourceType

        assert SourceType.TAVILY.value == "tavily"

    @pytest.mark.asyncio
    async def test_api_query_endpoint_passes_tavily_flag(self):
        """Verify the API /query endpoint passes used_tavily from RAG result."""
        from reviewmind.api.schemas import QueryResponse

        resp = QueryResponse(
            answer="test",
            used_tavily=True,
            response_time_ms=100,
        )
        assert resp.used_tavily is True

    def test_query_log_accepts_used_tavily(self):
        """Verify DB model accepts used_tavily flag."""
        from reviewmind.db.models import QueryLog

        col = QueryLog.__table__.c["used_tavily"]
        assert col is not None
