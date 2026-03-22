"""Unit tests for TASK-045 — Auto-mode: full pipeline.

Tests the auto-mode handler in bot/handlers/query.py:
1. Product extraction flow
2. Instant RAG from Qdrant cache
3. Tavily quick answer + Celery background job
4. Fallback to direct LLM
5. Source URL collection (YouTube + Reddit)
6. Background job scheduling
7. Edge cases and error handling
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════


def _make_bot_message(text: str = "test", user_id: int = 123) -> MagicMock:
    """Create a mocked aiogram Message for bot handler tests."""
    msg = MagicMock()
    msg.text = text
    msg.from_user = MagicMock()
    msg.from_user.id = user_id
    msg.chat = MagicMock()
    msg.chat.id = user_id
    msg.bot = MagicMock()
    msg.bot.send_chat_action = AsyncMock()
    msg.answer = AsyncMock()
    return msg


def _make_rag_response(**kwargs):
    """Create a RAGResponse with sensible defaults."""
    from reviewmind.core.rag import RAGResponse

    defaults = {
        "answer": "RAG analysis answer",
        "sources": ["https://example.com/review"],
        "used_curated": False,
        "confidence_met": True,
        "chunks_count": 5,
        "chunks_found": 10,
        "used_sponsored": False,
        "used_tavily": False,
        "error": None,
    }
    defaults.update(kwargs)
    return RAGResponse(**defaults)


def _make_mock_llm_client(response: str = "Test answer") -> MagicMock:
    client = MagicMock()
    client.generate = AsyncMock(return_value=response)
    client.close = AsyncMock()
    client._model = "gpt-4o-mini"
    return client


# ══════════════════════════════════════════════════════════════
# Tests — Constants
# ══════════════════════════════════════════════════════════════


class TestConstants:
    """Test module-level constants."""

    def test_max_answer_length(self):
        from reviewmind.bot.handlers.query import _MAX_ANSWER_LENGTH

        assert _MAX_ANSWER_LENGTH == 4096

    def test_searching_msg(self):
        from reviewmind.bot.handlers.query import _SEARCHING_MSG

        assert "⏳" in _SEARCHING_MSG
        assert "3 мин" in _SEARCHING_MSG

    def test_max_search_urls(self):
        from reviewmind.bot.handlers.query import MAX_SEARCH_URLS

        assert MAX_SEARCH_URLS == 10

    def test_service_unavailable_msg(self):
        from reviewmind.bot.handlers.query import _SERVICE_UNAVAILABLE_MSG

        assert "⚠️" in _SERVICE_UNAVAILABLE_MSG

    def test_no_product_fallback_note(self):
        from reviewmind.bot.handlers.query import _NO_PRODUCT_FALLBACK_NOTE

        assert "Совет" in _NO_PRODUCT_FALLBACK_NOTE


# ══════════════════════════════════════════════════════════════
# Tests — _truncate helper
# ══════════════════════════════════════════════════════════════


class TestTruncate:
    """Test the _truncate helper."""

    def test_short_text_unchanged(self):
        from reviewmind.bot.handlers.query import _truncate

        assert _truncate("short") == "short"

    def test_long_text_truncated(self):
        from reviewmind.bot.handlers.query import _MAX_ANSWER_LENGTH, _truncate

        long = "x" * 5000
        result = _truncate(long)
        assert len(result) <= _MAX_ANSWER_LENGTH
        assert result.endswith("...")

    def test_exact_limit_unchanged(self):
        from reviewmind.bot.handlers.query import _MAX_ANSWER_LENGTH, _truncate

        text = "x" * _MAX_ANSWER_LENGTH
        assert _truncate(text) == text

    def test_custom_limit(self):
        from reviewmind.bot.handlers.query import _truncate

        result = _truncate("abcdefgh", limit=5)
        assert len(result) == 5
        assert result.endswith("...")


# ══════════════════════════════════════════════════════════════
# Tests — _build_search_query
# ══════════════════════════════════════════════════════════════


class TestBuildSearchQuery:
    """Test the search query builder."""

    def test_single_product(self):
        from reviewmind.bot.handlers.query import _build_search_query

        result = _build_search_query(["Sony WH-1000XM5"])
        assert "Sony WH-1000XM5" in result
        assert result.endswith("review")

    def test_multiple_products(self):
        from reviewmind.bot.handlers.query import _build_search_query

        result = _build_search_query(["iPhone 16", "Samsung S25"])
        assert "iPhone 16" in result
        assert "Samsung S25" in result
        assert result.endswith("review")


# ══════════════════════════════════════════════════════════════
# Tests — _collect_source_urls
# ══════════════════════════════════════════════════════════════


class TestCollectSourceUrls:
    """Test the YouTube + Reddit URL collection."""

    @pytest.mark.asyncio
    async def test_collects_youtube_urls(self):
        from reviewmind.bot.handlers.query import _collect_source_urls

        @dataclass
        class FakeVideo:
            url: str = "https://youtube.com/watch?v=abc"

        with (
            patch("reviewmind.scrapers.youtube.YouTubeScraper") as MockYT,
            patch("reviewmind.scrapers.reddit.RedditScraper") as MockReddit,
        ):
            scraper = MagicMock()
            scraper.search_videos.return_value = [FakeVideo()]
            MockYT.return_value = scraper
            reddit = MagicMock()
            reddit.search_posts.return_value = []
            MockReddit.return_value = reddit

            urls = await _collect_source_urls(["Sony XM5"])

        assert len(urls) == 1
        assert "youtube.com" in urls[0]

    @pytest.mark.asyncio
    async def test_collects_reddit_urls(self):
        from reviewmind.bot.handlers.query import _collect_source_urls

        @dataclass
        class FakePost:
            url: str = "https://reddit.com/r/headphones/post1"

        with (
            patch("reviewmind.scrapers.youtube.YouTubeScraper") as MockYT,
            patch("reviewmind.scrapers.reddit.RedditScraper") as MockReddit,
        ):
            yt = MagicMock()
            yt.search_videos.return_value = []
            MockYT.return_value = yt
            reddit = MagicMock()
            reddit.search_posts.return_value = [FakePost()]
            MockReddit.return_value = reddit

            urls = await _collect_source_urls(["Sony XM5"])

        assert len(urls) == 1
        assert "reddit.com" in urls[0]

    @pytest.mark.asyncio
    async def test_deduplicates_urls(self):
        from reviewmind.bot.handlers.query import _collect_source_urls

        @dataclass
        class FakeVideo:
            url: str = "https://youtube.com/watch?v=same"

        @dataclass
        class FakePost:
            url: str = "https://youtube.com/watch?v=same"

        with (
            patch("reviewmind.scrapers.youtube.YouTubeScraper") as MockYT,
            patch("reviewmind.scrapers.reddit.RedditScraper") as MockReddit,
        ):
            yt = MagicMock()
            yt.search_videos.return_value = [FakeVideo()]
            MockYT.return_value = yt
            reddit = MagicMock()
            reddit.search_posts.return_value = [FakePost()]
            MockReddit.return_value = reddit

            urls = await _collect_source_urls(["Sony XM5"])

        assert len(urls) == 1

    @pytest.mark.asyncio
    async def test_caps_at_max_search_urls(self):
        from reviewmind.bot.handlers.query import MAX_SEARCH_URLS, _collect_source_urls

        @dataclass
        class FakeVideo:
            url: str

        videos = [FakeVideo(url=f"https://youtube.com/watch?v=vid{i}") for i in range(15)]

        with (
            patch("reviewmind.scrapers.youtube.YouTubeScraper") as MockYT,
            patch("reviewmind.scrapers.reddit.RedditScraper") as MockReddit,
        ):
            yt = MagicMock()
            yt.search_videos.return_value = videos
            MockYT.return_value = yt
            reddit = MagicMock()
            reddit.search_posts.return_value = []
            MockReddit.return_value = reddit

            urls = await _collect_source_urls(["Sony XM5"])

        assert len(urls) <= MAX_SEARCH_URLS

    @pytest.mark.asyncio
    async def test_youtube_error_graceful(self):
        from reviewmind.bot.handlers.query import _collect_source_urls

        @dataclass
        class FakePost:
            url: str = "https://reddit.com/r/headphones/p1"

        with (
            patch(
                "reviewmind.scrapers.youtube.YouTubeScraper",
                side_effect=RuntimeError("yt fail"),
            ),
            patch("reviewmind.scrapers.reddit.RedditScraper") as MockReddit,
        ):
            reddit = MagicMock()
            reddit.search_posts.return_value = [FakePost()]
            MockReddit.return_value = reddit

            urls = await _collect_source_urls(["Sony XM5"])

        assert len(urls) == 1  # reddit URL still collected

    @pytest.mark.asyncio
    async def test_both_fail_returns_empty(self):
        from reviewmind.bot.handlers.query import _collect_source_urls

        with (
            patch(
                "reviewmind.scrapers.youtube.YouTubeScraper",
                side_effect=RuntimeError("yt"),
            ),
            patch(
                "reviewmind.scrapers.reddit.RedditScraper",
                side_effect=RuntimeError("reddit"),
            ),
        ):
            urls = await _collect_source_urls(["Sony XM5"])

        assert urls == []

    @pytest.mark.asyncio
    async def test_empty_product_list(self):
        from reviewmind.bot.handlers.query import _collect_source_urls

        with (
            patch("reviewmind.scrapers.youtube.YouTubeScraper") as MockYT,
            patch("reviewmind.scrapers.reddit.RedditScraper") as MockReddit,
        ):
            yt = MagicMock()
            yt.search_videos.return_value = []
            MockYT.return_value = yt
            reddit = MagicMock()
            reddit.search_posts.return_value = []
            MockReddit.return_value = reddit

            urls = await _collect_source_urls([])

        assert urls == []


# ══════════════════════════════════════════════════════════════
# Tests — _schedule_background_job
# ══════════════════════════════════════════════════════════════


class TestScheduleBackgroundJob:
    """Test the background job scheduler."""

    @pytest.mark.asyncio
    async def test_returns_job_id_on_success(self):
        from reviewmind.bot.handlers.query import _schedule_background_job

        mock_settings = MagicMock()
        mock_settings.database_url = "postgresql+asyncpg://test"

        mock_engine = AsyncMock()
        mock_engine.dispose = AsyncMock()
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=False)
        mock_session.add = MagicMock()
        mock_session.flush = AsyncMock()
        mock_session.commit = AsyncMock()
        mock_session.rollback = AsyncMock()

        with (
            patch(
                "reviewmind.config.settings",
                mock_settings,
            ),
            patch(
                "sqlalchemy.ext.asyncio.create_async_engine",
                return_value=mock_engine,
            ),
            patch(
                "sqlalchemy.ext.asyncio.async_sessionmaker",
                return_value=MagicMock(return_value=mock_session),
            ),
            patch("reviewmind.db.models.Job"),
            patch(
                "reviewmind.workers.tasks.ingest_sources_task",
            ) as mock_task,
        ):
            mock_task.apply_async.return_value = MagicMock(id="celery-123")

            job_id = await _schedule_background_job(
                user_id=123,
                product_query="Sony XM5",
                urls=["https://youtube.com/watch?v=abc"],
            )

        assert job_id is not None
        mock_task.apply_async.assert_called_once()

    @pytest.mark.asyncio
    async def test_returns_none_on_celery_failure(self):
        from reviewmind.bot.handlers.query import _schedule_background_job

        mock_settings = MagicMock()
        mock_settings.database_url = "postgresql+asyncpg://test"

        with (
            patch("reviewmind.config.settings", mock_settings),
            patch("sqlalchemy.ext.asyncio.create_async_engine", side_effect=RuntimeError("db")),
            patch(
                "reviewmind.workers.tasks.ingest_sources_task",
            ) as mock_task,
        ):
            mock_task.apply_async.side_effect = RuntimeError("no broker")

            job_id = await _schedule_background_job(
                user_id=123,
                product_query="Sony XM5",
                urls=["https://youtube.com/watch?v=abc"],
            )

        assert job_id is None

    @pytest.mark.asyncio
    async def test_db_failure_does_not_block_celery(self):
        from reviewmind.bot.handlers.query import _schedule_background_job

        mock_settings = MagicMock()
        mock_settings.database_url = "postgresql+asyncpg://test"

        with (
            patch("reviewmind.config.settings", mock_settings),
            patch(
                "sqlalchemy.ext.asyncio.create_async_engine",
                side_effect=RuntimeError("db down"),
            ),
            patch(
                "reviewmind.workers.tasks.ingest_sources_task",
            ) as mock_task,
        ):
            mock_task.apply_async.return_value = MagicMock(id="celery-456")

            job_id = await _schedule_background_job(
                user_id=123,
                product_query="Sony XM5",
                urls=["https://youtube.com/watch?v=abc"],
            )

        assert job_id is not None
        mock_task.apply_async.assert_called_once()


# ══════════════════════════════════════════════════════════════
# Tests — on_text_message (auto-mode flows)
# ══════════════════════════════════════════════════════════════


class TestAutoModeInstantRAG:
    """Test: product found + Qdrant has confident data → instant answer."""

    @pytest.mark.asyncio
    async def test_instant_rag_confident(self):
        from reviewmind.bot.handlers.query import on_text_message

        msg = _make_bot_message("Sony WH-1000XM5 стоит ли покупать?")
        rag_resp = _make_rag_response(
            answer="Sony XM5 — отличные наушники",
            confidence_met=True,
        )

        mock_qdrant = AsyncMock()

        with (
            patch(
                "reviewmind.bot.handlers.query.extract_product",
                new=AsyncMock(return_value=["Sony WH-1000XM5"]),
            ),
            patch(
                "reviewmind.bot.handlers.query._create_qdrant_client",
                return_value=mock_qdrant,
            ),
            patch(
                "reviewmind.bot.handlers.query._try_instant_rag",
                new=AsyncMock(return_value=rag_resp),
            ),
        ):
            await on_text_message(msg)

        # Should send answer with feedback keyboard
        msg.answer.assert_called_once()
        call_kwargs = msg.answer.call_args
        answer = call_kwargs.args[0]
        assert "Sony XM5" in answer
        assert call_kwargs.kwargs.get("reply_markup") is not None

    @pytest.mark.asyncio
    async def test_instant_rag_with_curated_badge(self):
        from reviewmind.bot.handlers.query import on_text_message

        msg = _make_bot_message("Sony WH-1000XM5?")
        rag_resp = _make_rag_response(
            answer="Curated analysis",
            confidence_met=True,
            used_curated=True,
        )

        with (
            patch(
                "reviewmind.bot.handlers.query.extract_product",
                new=AsyncMock(return_value=["Sony WH-1000XM5"]),
            ),
            patch(
                "reviewmind.bot.handlers.query._create_qdrant_client",
                return_value=AsyncMock(),
            ),
            patch(
                "reviewmind.bot.handlers.query._try_instant_rag",
                new=AsyncMock(return_value=rag_resp),
            ),
        ):
            await on_text_message(msg)

        msg.answer.assert_called_once()


class TestAutoModeCacheMiss:
    """Test: product found but insufficient data → Tavily quick + background job."""

    @pytest.mark.asyncio
    async def test_tavily_quick_answer_plus_background(self):
        from reviewmind.bot.handlers.query import on_text_message

        msg = _make_bot_message("Dyson V15 стоит ли покупать?")
        rag_resp = _make_rag_response(
            answer="Quick Tavily-based answer about Dyson",
            confidence_met=False,
            used_tavily=True,
        )

        with (
            patch(
                "reviewmind.bot.handlers.query.extract_product",
                new=AsyncMock(return_value=["Dyson V15"]),
            ),
            patch(
                "reviewmind.bot.handlers.query._create_qdrant_client",
                return_value=AsyncMock(),
            ),
            patch(
                "reviewmind.bot.handlers.query._try_instant_rag",
                new=AsyncMock(return_value=rag_resp),
            ),
            patch(
                "reviewmind.bot.handlers.query._collect_source_urls",
                new=AsyncMock(return_value=["https://youtube.com/watch?v=abc"]),
            ),
            patch(
                "reviewmind.bot.handlers.query._schedule_background_job",
                new=AsyncMock(return_value="job-uuid-123"),
            ),
        ):
            await on_text_message(msg)

        # Two message.answer calls: quick answer + "searching more"
        assert msg.answer.call_count == 2
        first_call = msg.answer.call_args_list[0]
        assert "Dyson" in first_call.args[0]
        second_call = msg.answer.call_args_list[1]
        assert "дополнительные источники" in second_call.args[0]

    @pytest.mark.asyncio
    async def test_no_cache_no_tavily_searching_msg(self):
        from reviewmind.bot.handlers.query import _SEARCHING_MSG, on_text_message

        msg = _make_bot_message("JBL Charge 6?")
        rag_resp = _make_rag_response(
            answer="",
            confidence_met=False,
            used_tavily=False,
        )

        with (
            patch(
                "reviewmind.bot.handlers.query.extract_product",
                new=AsyncMock(return_value=["JBL Charge 6"]),
            ),
            patch(
                "reviewmind.bot.handlers.query._create_qdrant_client",
                return_value=AsyncMock(),
            ),
            patch(
                "reviewmind.bot.handlers.query._try_instant_rag",
                new=AsyncMock(return_value=rag_resp),
            ),
            patch(
                "reviewmind.bot.handlers.query._collect_source_urls",
                new=AsyncMock(return_value=["https://youtube.com/watch?v=x"]),
            ),
            patch(
                "reviewmind.bot.handlers.query._schedule_background_job",
                new=AsyncMock(return_value="job-uuid-456"),
            ),
        ):
            await on_text_message(msg)

        msg.answer.assert_called_once()
        assert _SEARCHING_MSG == msg.answer.call_args.args[0]

    @pytest.mark.asyncio
    async def test_no_sources_found_fallback_llm(self):
        from reviewmind.bot.handlers.query import on_text_message

        msg = _make_bot_message("Obscure gadget xyz?")
        rag_resp = _make_rag_response(
            answer="",
            confidence_met=False,
            used_tavily=False,
        )

        mock_llm = _make_mock_llm_client("Here's what I know about that gadget")

        with (
            patch(
                "reviewmind.bot.handlers.query.extract_product",
                new=AsyncMock(return_value=["Obscure gadget xyz"]),
            ),
            patch(
                "reviewmind.bot.handlers.query._create_qdrant_client",
                return_value=AsyncMock(),
            ),
            patch(
                "reviewmind.bot.handlers.query._try_instant_rag",
                new=AsyncMock(return_value=rag_resp),
            ),
            patch(
                "reviewmind.bot.handlers.query._collect_source_urls",
                new=AsyncMock(return_value=[]),
            ),
            patch("reviewmind.bot.handlers.query.LLMClient") as MockLLM,
        ):
            instance = MockLLM.return_value
            instance.__aenter__ = AsyncMock(return_value=mock_llm)
            instance.__aexit__ = AsyncMock(return_value=False)
            await on_text_message(msg)

        msg.answer.assert_called_once()
        answer = msg.answer.call_args.args[0]
        assert "gadget" in answer


class TestAutoModeNoProduct:
    """Test: no product extracted → direct LLM fallback."""

    @pytest.mark.asyncio
    async def test_direct_llm_fallback(self):
        from reviewmind.bot.handlers.query import on_text_message

        msg = _make_bot_message("Привет, как дела?")
        mock_llm = _make_mock_llm_client("Привет! Я помогу с обзорами.")

        with (
            patch(
                "reviewmind.bot.handlers.query.extract_product",
                new=AsyncMock(return_value=[]),
            ),
            patch("reviewmind.bot.handlers.query.LLMClient") as MockLLM,
        ):
            instance = MockLLM.return_value
            instance.__aenter__ = AsyncMock(return_value=mock_llm)
            instance.__aexit__ = AsyncMock(return_value=False)
            await on_text_message(msg)

        msg.answer.assert_called_once()
        answer = msg.answer.call_args.args[0]
        assert "Привет" in answer
        assert "Совет" in answer  # includes the tip about naming a product

    @pytest.mark.asyncio
    async def test_extraction_error_fallback(self):
        from reviewmind.bot.handlers.query import on_text_message

        msg = _make_bot_message("Some query")
        mock_llm = _make_mock_llm_client("fallback answer")

        with (
            patch(
                "reviewmind.bot.handlers.query.extract_product",
                new=AsyncMock(side_effect=RuntimeError("extract fail")),
            ),
            patch("reviewmind.bot.handlers.query.LLMClient") as MockLLM,
        ):
            instance = MockLLM.return_value
            instance.__aenter__ = AsyncMock(return_value=mock_llm)
            instance.__aexit__ = AsyncMock(return_value=False)
            await on_text_message(msg)

        msg.answer.assert_called_once()


class TestAutoModeEdgeCases:
    """Test edge cases in the auto-mode handler."""

    @pytest.mark.asyncio
    async def test_qdrant_connect_failure(self):
        from reviewmind.bot.handlers.query import _SERVICE_UNAVAILABLE_MSG, on_text_message

        msg = _make_bot_message("Sony XM5?")

        with (
            patch(
                "reviewmind.bot.handlers.query.extract_product",
                new=AsyncMock(return_value=["Sony XM5"]),
            ),
            patch(
                "reviewmind.bot.handlers.query._create_qdrant_client",
                side_effect=RuntimeError("connect failed"),
            ),
        ):
            await on_text_message(msg)

        msg.answer.assert_called_once()
        assert msg.answer.call_args.args[0] == _SERVICE_UNAVAILABLE_MSG

    @pytest.mark.asyncio
    async def test_instant_rag_exception(self):
        """RAG pipeline crash → schedule background job anyway."""
        from reviewmind.bot.handlers.query import on_text_message

        msg = _make_bot_message("Sony XM5?")

        with (
            patch(
                "reviewmind.bot.handlers.query.extract_product",
                new=AsyncMock(return_value=["Sony XM5"]),
            ),
            patch(
                "reviewmind.bot.handlers.query._create_qdrant_client",
                return_value=AsyncMock(),
            ),
            patch(
                "reviewmind.bot.handlers.query._try_instant_rag",
                new=AsyncMock(side_effect=RuntimeError("rag crash")),
            ),
            patch(
                "reviewmind.bot.handlers.query._collect_source_urls",
                new=AsyncMock(return_value=["https://youtube.com/watch?v=abc"]),
            ),
            patch(
                "reviewmind.bot.handlers.query._schedule_background_job",
                new=AsyncMock(return_value="job-123"),
            ),
        ):
            await on_text_message(msg)

        # Should still send searching message
        assert msg.answer.call_count >= 1

    @pytest.mark.asyncio
    async def test_celery_unavailable_no_quick_answer(self):
        """No Celery, no quick answer → service unavailable."""
        from reviewmind.bot.handlers.query import _SERVICE_UNAVAILABLE_MSG, on_text_message

        msg = _make_bot_message("Sony XM5?")
        rag_resp = _make_rag_response(answer="", confidence_met=False)

        with (
            patch(
                "reviewmind.bot.handlers.query.extract_product",
                new=AsyncMock(return_value=["Sony XM5"]),
            ),
            patch(
                "reviewmind.bot.handlers.query._create_qdrant_client",
                return_value=AsyncMock(),
            ),
            patch(
                "reviewmind.bot.handlers.query._try_instant_rag",
                new=AsyncMock(return_value=rag_resp),
            ),
            patch(
                "reviewmind.bot.handlers.query._collect_source_urls",
                new=AsyncMock(return_value=["https://youtube.com/watch?v=x"]),
            ),
            patch(
                "reviewmind.bot.handlers.query._schedule_background_job",
                new=AsyncMock(return_value=None),
            ),
        ):
            await on_text_message(msg)

        msg.answer.assert_called_once()
        assert msg.answer.call_args.args[0] == _SERVICE_UNAVAILABLE_MSG

    @pytest.mark.asyncio
    async def test_no_bot_attribute_no_crash(self):
        """Message with bot=None should not crash typing indicator."""
        from reviewmind.bot.handlers.query import on_text_message

        msg = _make_bot_message("Sony XM5?")
        msg.bot = None

        with (
            patch(
                "reviewmind.bot.handlers.query.extract_product",
                new=AsyncMock(return_value=["Sony XM5"]),
            ),
            patch(
                "reviewmind.bot.handlers.query._create_qdrant_client",
                return_value=AsyncMock(),
            ),
            patch(
                "reviewmind.bot.handlers.query._try_instant_rag",
                new=AsyncMock(return_value=_make_rag_response(confidence_met=True)),
            ),
        ):
            await on_text_message(msg)

        msg.answer.assert_called_once()


class TestAutoModeAnswerStructure:
    """Test that auto-mode answers have expected structure."""

    @pytest.mark.asyncio
    async def test_instant_answer_has_feedback_keyboard(self):
        from reviewmind.bot.handlers.query import on_text_message
        from reviewmind.bot.keyboards import FEEDBACK_BAD, FEEDBACK_SOURCES, FEEDBACK_USEFUL

        msg = _make_bot_message("Sony WH-1000XM5?")
        rag_resp = _make_rag_response(
            answer="Great headphones",
            confidence_met=True,
        )

        with (
            patch(
                "reviewmind.bot.handlers.query.extract_product",
                new=AsyncMock(return_value=["Sony WH-1000XM5"]),
            ),
            patch(
                "reviewmind.bot.handlers.query._create_qdrant_client",
                return_value=AsyncMock(),
            ),
            patch(
                "reviewmind.bot.handlers.query._try_instant_rag",
                new=AsyncMock(return_value=rag_resp),
            ),
        ):
            await on_text_message(msg)

        call_kwargs = msg.answer.call_args.kwargs
        markup = call_kwargs.get("reply_markup")
        assert markup is not None
        button_data = [btn.callback_data for row in markup.inline_keyboard for btn in row]
        assert any(d.startswith(FEEDBACK_USEFUL) for d in button_data)
        assert any(d.startswith(FEEDBACK_BAD) for d in button_data)
        assert any(d.startswith(FEEDBACK_SOURCES) for d in button_data)

    @pytest.mark.asyncio
    async def test_no_product_answer_includes_tip(self):
        from reviewmind.bot.handlers.query import on_text_message

        msg = _make_bot_message("Hello")
        mock_llm = _make_mock_llm_client("Hi")

        with (
            patch(
                "reviewmind.bot.handlers.query.extract_product",
                new=AsyncMock(return_value=[]),
            ),
            patch("reviewmind.bot.handlers.query.LLMClient") as MockLLM,
        ):
            instance = MockLLM.return_value
            instance.__aenter__ = AsyncMock(return_value=mock_llm)
            instance.__aexit__ = AsyncMock(return_value=False)
            await on_text_message(msg)

        answer = msg.answer.call_args.args[0]
        assert "Совет" in answer


# ══════════════════════════════════════════════════════════════
# Tests — _try_instant_rag
# ══════════════════════════════════════════════════════════════


class TestTryInstantRag:
    """Test the _try_instant_rag helper."""

    @pytest.mark.asyncio
    async def test_calls_rag_pipeline(self):
        from reviewmind.bot.handlers.query import _try_instant_rag

        expected = _make_rag_response()
        mock_rag = MagicMock()
        mock_rag.query = AsyncMock(return_value=expected)
        mock_rag.__aenter__ = AsyncMock(return_value=mock_rag)
        mock_rag.__aexit__ = AsyncMock(return_value=False)
        mock_qdrant = AsyncMock()

        with patch("reviewmind.bot.handlers.query.RAGPipeline", return_value=mock_rag):
            result = await _try_instant_rag(
                mock_qdrant,
                "Sony XM5?",
                "Sony WH-1000XM5",
                MagicMock(),
            )

        assert result is expected
        mock_rag.query.assert_called_once_with(
            user_query="Sony XM5?",
            product_query="Sony WH-1000XM5",
            chat_history=None,
        )


# ══════════════════════════════════════════════════════════════
# Tests — _fallback_llm_answer
# ══════════════════════════════════════════════════════════════


class TestFallbackLlmAnswer:
    """Test the _fallback_llm_answer helper."""

    @pytest.mark.asyncio
    async def test_sends_answer(self):
        from reviewmind.bot.handlers.query import _fallback_llm_answer

        msg = _make_bot_message("any text")
        mock_llm = _make_mock_llm_client("LLM answer")

        with patch("reviewmind.bot.handlers.query.LLMClient") as MockLLM:
            instance = MockLLM.return_value
            instance.__aenter__ = AsyncMock(return_value=mock_llm)
            instance.__aexit__ = AsyncMock(return_value=False)
            await _fallback_llm_answer(msg, 123, MagicMock())

        msg.answer.assert_called_once()
        assert "LLM answer" in msg.answer.call_args.args[0]

    @pytest.mark.asyncio
    async def test_error_sends_error_msg(self):
        from reviewmind.bot.handlers.query import _UNEXPECTED_ERROR_MSG, _fallback_llm_answer

        msg = _make_bot_message("any text")

        with patch("reviewmind.bot.handlers.query.LLMClient") as MockLLM:
            instance = MockLLM.return_value
            instance.__aenter__ = AsyncMock(side_effect=RuntimeError("fail"))
            instance.__aexit__ = AsyncMock(return_value=False)
            await _fallback_llm_answer(msg, 123, MagicMock())

        msg.answer.assert_called_once()
        assert msg.answer.call_args.args[0] == _UNEXPECTED_ERROR_MSG


# ══════════════════════════════════════════════════════════════
# Tests — Dispatcher wiring
# ══════════════════════════════════════════════════════════════


class TestDispatcherWiring:
    """Verify auto-mode handler is registered in the dispatcher."""

    def test_query_router_name(self):
        from reviewmind.bot.handlers.query import router

        assert router.name == "query"

    def test_query_router_has_handler(self):
        from reviewmind.bot.handlers.query import router

        assert len(router.message.handlers) > 0

    def test_dispatcher_includes_query_router(self):
        import inspect

        import reviewmind.bot.main as bot_main

        source = inspect.getsource(bot_main.create_dispatcher)
        assert "include_router(query_router)" in source

    def test_query_router_registered_last(self):
        import inspect

        import reviewmind.bot.main as bot_main

        source = inspect.getsource(bot_main.create_dispatcher)
        start_pos = source.index("include_router(start_router)")
        mode_pos = source.index("include_router(mode_router)")
        links_pos = source.index("include_router(links_router)")
        query_pos = source.index("include_router(query_router)")
        assert query_pos > start_pos
        assert query_pos > mode_pos
        assert query_pos > links_pos


# ══════════════════════════════════════════════════════════════
# Tests — Integration Scenarios (per TASK-045 test_steps)
# ══════════════════════════════════════════════════════════════


class TestIntegrationScenarios:
    """High-level integration scenarios from TASK-045 test_steps."""

    @pytest.mark.asyncio
    async def test_seed_product_instant_answer(self):
        """Step 1: Query by product from seed base → instant answer."""
        from reviewmind.bot.handlers.query import on_text_message

        msg = _make_bot_message("Sony WH-1000XM5 стоит ли покупать?")
        rag_resp = _make_rag_response(
            answer="Sony WH-1000XM5: ✅ Плюсы: ... ❌ Минусы: ...",
            confidence_met=True,
            sources=["https://youtube.com/watch?v=abc", "https://reddit.com/r/x"],
        )

        with (
            patch(
                "reviewmind.bot.handlers.query.extract_product",
                new=AsyncMock(return_value=["Sony WH-1000XM5"]),
            ),
            patch(
                "reviewmind.bot.handlers.query._create_qdrant_client",
                return_value=AsyncMock(),
            ),
            patch(
                "reviewmind.bot.handlers.query._try_instant_rag",
                new=AsyncMock(return_value=rag_resp),
            ),
        ):
            await on_text_message(msg)

        assert msg.answer.call_count == 1
        answer = msg.answer.call_args.args[0]
        assert "Sony WH-1000XM5" in answer

    @pytest.mark.asyncio
    async def test_new_product_searching_then_push(self):
        """Step 2: New product → '⏳ Ищу данные...' → push answer later."""
        from reviewmind.bot.handlers.query import _SEARCHING_MSG, on_text_message

        msg = _make_bot_message("Apple Vision Pro 2?")
        rag_resp = _make_rag_response(answer="", confidence_met=False)

        with (
            patch(
                "reviewmind.bot.handlers.query.extract_product",
                new=AsyncMock(return_value=["Apple Vision Pro 2"]),
            ),
            patch(
                "reviewmind.bot.handlers.query._create_qdrant_client",
                return_value=AsyncMock(),
            ),
            patch(
                "reviewmind.bot.handlers.query._try_instant_rag",
                new=AsyncMock(return_value=rag_resp),
            ),
            patch(
                "reviewmind.bot.handlers.query._collect_source_urls",
                new=AsyncMock(return_value=["https://youtube.com/watch?v=new"]),
            ),
            patch(
                "reviewmind.bot.handlers.query._schedule_background_job",
                new=AsyncMock(return_value="job-uuid"),
            ) as mock_schedule,
        ):
            await on_text_message(msg)

        msg.answer.assert_called_once()
        assert _SEARCHING_MSG == msg.answer.call_args.args[0]
        mock_schedule.assert_called_once()

    @pytest.mark.asyncio
    async def test_repeated_product_instant_cache(self):
        """Step 3: Repeated query for same product → instant answer from cache."""
        from reviewmind.bot.handlers.query import on_text_message

        msg = _make_bot_message("Sony WH-1000XM5?")
        rag_resp = _make_rag_response(
            answer="Cached analysis",
            confidence_met=True,
        )

        with (
            patch(
                "reviewmind.bot.handlers.query.extract_product",
                new=AsyncMock(return_value=["Sony WH-1000XM5"]),
            ),
            patch(
                "reviewmind.bot.handlers.query._create_qdrant_client",
                return_value=AsyncMock(),
            ),
            patch(
                "reviewmind.bot.handlers.query._try_instant_rag",
                new=AsyncMock(return_value=rag_resp),
            ),
        ):
            await on_text_message(msg)

        # Only one call — no background scheduling
        assert msg.answer.call_count == 1
        assert "Cached analysis" in msg.answer.call_args.args[0]

    @pytest.mark.asyncio
    async def test_answer_has_feedback_buttons(self):
        """Step 4: Check answer structure with ✅ ❌ ⚖️ 🏆 (feedback keyboard)."""
        from reviewmind.bot.handlers.query import on_text_message

        msg = _make_bot_message("iPhone 16?")
        rag_resp = _make_rag_response(
            answer="Analysis: ✅ Плюсы ❌ Минусы ⚖️ Спорные 🏆 Вывод",
            confidence_met=True,
        )

        with (
            patch(
                "reviewmind.bot.handlers.query.extract_product",
                new=AsyncMock(return_value=["iPhone 16"]),
            ),
            patch(
                "reviewmind.bot.handlers.query._create_qdrant_client",
                return_value=AsyncMock(),
            ),
            patch(
                "reviewmind.bot.handlers.query._try_instant_rag",
                new=AsyncMock(return_value=rag_resp),
            ),
        ):
            await on_text_message(msg)

        call_kwargs = msg.answer.call_args.kwargs
        assert "reply_markup" in call_kwargs
        # Has 3 buttons
        markup = call_kwargs["reply_markup"]
        all_buttons = [btn for row in markup.inline_keyboard for btn in row]
        assert len(all_buttons) == 3

    @pytest.mark.asyncio
    async def test_imports_available(self):
        """Verify all auto-mode imports work."""
        from reviewmind.bot.handlers.query import (
            MAX_SEARCH_URLS,
            _build_search_query,
            _collect_source_urls,
            _create_qdrant_client,
            _fallback_llm_answer,
            _schedule_background_job,
            _truncate,
            _try_instant_rag,
            on_text_message,
            router,
        )

        assert callable(on_text_message)
        assert callable(_truncate)
        assert callable(_build_search_query)
        assert callable(_collect_source_urls)
        assert callable(_schedule_background_job)
        assert callable(_try_instant_rag)
        assert callable(_fallback_llm_answer)
        assert callable(_create_qdrant_client)
        assert router is not None
        assert MAX_SEARCH_URLS == 10
