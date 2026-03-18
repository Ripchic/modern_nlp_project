"""Unit tests for reviewmind.bot.handlers.links — URL links handler."""

from __future__ import annotations

from dataclasses import dataclass, field
from unittest.mock import AsyncMock, MagicMock, patch

from aiogram.types import Chat, InlineKeyboardMarkup, Message, User

# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_user(user_id: int = 12345) -> User:
    return User(id=user_id, is_bot=False, first_name="Test")


def _make_chat(chat_id: int = 12345) -> Chat:
    return Chat(id=chat_id, type="private")


def _make_message(text: str = "", user_id: int = 12345) -> MagicMock:
    msg = MagicMock(spec=Message)
    msg.text = text
    msg.from_user = _make_user(user_id)
    msg.chat = _make_chat(user_id)
    msg.answer = AsyncMock()
    msg.bot = MagicMock()
    msg.bot.send_chat_action = AsyncMock()
    return msg


@dataclass
class FakeSourceResult:
    url: str
    success: bool
    source_type: str = "web"
    chunks_count: int = 0
    is_sponsored: bool = False
    error: str | None = None
    source_id: int | None = None


@dataclass
class FakeIngestionResult:
    success_count: int = 0
    failed_count: int = 0
    chunks_count: int = 0
    failed_urls: list[str] = field(default_factory=list)
    results: list[FakeSourceResult] = field(default_factory=list)


@dataclass
class FakeRAGResponse:
    answer: str = ""
    sources: list[str] = field(default_factory=list)
    used_curated: bool = False
    confidence_met: bool = False
    chunks_count: int = 0
    chunks_found: int = 0
    used_sponsored: bool = False
    error: str | None = None


# ══════════════════════════════════════════════════════════════
# Tests — extract_urls
# ══════════════════════════════════════════════════════════════


class TestExtractUrls:
    def test_single_url(self):
        from reviewmind.bot.handlers.links import extract_urls

        urls = extract_urls("Check https://youtube.com/watch?v=abc")
        assert urls == ["https://youtube.com/watch?v=abc"]

    def test_multiple_urls(self):
        from reviewmind.bot.handlers.links import extract_urls

        text = "https://example.com\nhttps://youtube.com/watch?v=1"
        urls = extract_urls(text)
        assert len(urls) == 2

    def test_duplicate_urls_deduplicated(self):
        from reviewmind.bot.handlers.links import extract_urls

        text = "https://example.com\nhttps://example.com"
        urls = extract_urls(text)
        assert urls == ["https://example.com"]

    def test_no_urls(self):
        from reviewmind.bot.handlers.links import extract_urls

        assert extract_urls("no urls here") == []

    def test_trailing_punctuation_stripped(self):
        from reviewmind.bot.handlers.links import extract_urls

        urls = extract_urls("Visit https://example.com.")
        assert urls == ["https://example.com"]

    def test_trailing_comma_stripped(self):
        from reviewmind.bot.handlers.links import extract_urls

        urls = extract_urls("https://a.com, https://b.com")
        assert len(urls) == 2
        assert urls[0] == "https://a.com"

    def test_http_and_https(self):
        from reviewmind.bot.handlers.links import extract_urls

        text = "http://insecure.com https://secure.com"
        assert len(extract_urls(text)) == 2

    def test_empty_string(self):
        from reviewmind.bot.handlers.links import extract_urls

        assert extract_urls("") == []

    def test_preserves_order(self):
        from reviewmind.bot.handlers.links import extract_urls

        text = "https://z.com https://a.com https://m.com"
        urls = extract_urls(text)
        assert urls == ["https://z.com", "https://a.com", "https://m.com"]

    def test_url_with_query_params(self):
        from reviewmind.bot.handlers.links import extract_urls

        text = "https://example.com/page?q=test&lang=ru"
        urls = extract_urls(text)
        assert urls == ["https://example.com/page?q=test&lang=ru"]

    def test_url_with_path(self):
        from reviewmind.bot.handlers.links import extract_urls

        urls = extract_urls("https://reddit.com/r/headphones/comments/abc123/title")
        assert len(urls) == 1
        assert "reddit.com" in urls[0]


# ══════════════════════════════════════════════════════════════
# Tests — extract_query_text
# ══════════════════════════════════════════════════════════════


class TestExtractQueryText:
    def test_extracts_text_around_url(self):
        from reviewmind.bot.handlers.links import extract_query_text

        result = extract_query_text("Sony XM5 https://example.com обзор", ["https://example.com"])
        assert "Sony XM5" in result
        assert "обзор" in result

    def test_url_only_returns_default(self):
        from reviewmind.bot.handlers.links import _DEFAULT_QUERY, extract_query_text

        result = extract_query_text("https://example.com", ["https://example.com"])
        assert result == _DEFAULT_QUERY

    def test_multiple_urls_removed(self):
        from reviewmind.bot.handlers.links import extract_query_text

        text = "https://a.com and https://b.com review"
        result = extract_query_text(text, ["https://a.com", "https://b.com"])
        assert "review" in result
        assert "https://" not in result

    def test_whitespace_collapsed(self):
        from reviewmind.bot.handlers.links import extract_query_text

        result = extract_query_text("test  https://a.com  review", ["https://a.com"])
        assert "  " not in result


# ══════════════════════════════════════════════════════════════
# Tests — _pluralize_links
# ══════════════════════════════════════════════════════════════


class TestPluralizeLinks:
    def test_one(self):
        from reviewmind.bot.handlers.links import _pluralize_links

        assert _pluralize_links(1) == "ссылку"

    def test_two(self):
        from reviewmind.bot.handlers.links import _pluralize_links

        assert _pluralize_links(2) == "ссылки"

    def test_five(self):
        from reviewmind.bot.handlers.links import _pluralize_links

        assert _pluralize_links(5) == "ссылок"

    def test_eleven(self):
        from reviewmind.bot.handlers.links import _pluralize_links

        assert _pluralize_links(11) == "ссылок"

    def test_twenty_one(self):
        from reviewmind.bot.handlers.links import _pluralize_links

        assert _pluralize_links(21) == "ссылку"

    def test_twenty_two(self):
        from reviewmind.bot.handlers.links import _pluralize_links

        assert _pluralize_links(22) == "ссылки"


# ══════════════════════════════════════════════════════════════
# Tests — _build_failure_lines
# ══════════════════════════════════════════════════════════════


class TestBuildFailureLines:
    def test_no_failures(self):
        from reviewmind.bot.handlers.links import _build_failure_lines

        results = [FakeSourceResult(url="https://a.com", success=True)]
        assert _build_failure_lines(results) == []

    def test_with_failures(self):
        from reviewmind.bot.handlers.links import _build_failure_lines

        results = [
            FakeSourceResult(url="https://ok.com", success=True),
            FakeSourceResult(url="https://bad.com", success=False, error="timeout"),
        ]
        lines = _build_failure_lines(results)
        assert len(lines) == 1
        assert "https://bad.com" in lines[0]
        assert "❌" in lines[0]

    def test_all_failures(self):
        from reviewmind.bot.handlers.links import _build_failure_lines

        results = [
            FakeSourceResult(url="https://a.com", success=False),
            FakeSourceResult(url="https://b.com", success=False),
        ]
        assert len(_build_failure_lines(results)) == 2


# ══════════════════════════════════════════════════════════════
# Tests — on_links_message handler
# ══════════════════════════════════════════════════════════════


class TestOnLinksMessage:
    """Tests for the main handler function."""

    async def test_returns_early_if_no_urls(self):
        from reviewmind.bot.handlers.links import on_links_message

        msg = _make_message("no urls here")
        await on_links_message(msg)
        msg.answer.assert_not_called()

    async def test_returns_early_if_empty_text(self):
        from reviewmind.bot.handlers.links import on_links_message

        msg = _make_message("")
        await on_links_message(msg)
        msg.answer.assert_not_called()

    async def test_sends_processing_status(self):
        """Handler should send a 'processing' message before starting."""
        from reviewmind.bot.handlers.links import on_links_message

        msg = _make_message("https://example.com")
        status_msg = MagicMock(spec=Message)
        status_msg.edit_text = AsyncMock()
        msg.answer.return_value = status_msg

        mock_qdrant = AsyncMock()
        mock_qdrant.close = AsyncMock()

        with (
            patch("reviewmind.bot.handlers.links._create_qdrant_client", return_value=mock_qdrant),
            patch("reviewmind.bot.handlers.links._ingest_and_analyse", new_callable=AsyncMock),
        ):
            await on_links_message(msg)

        msg.answer.assert_called_once()
        call_text = msg.answer.call_args.args[0]
        assert "🔄" in call_text

    async def test_shows_typing_indicator(self):
        from reviewmind.bot.handlers.links import on_links_message

        msg = _make_message("https://example.com")
        status_msg = MagicMock(spec=Message)
        status_msg.edit_text = AsyncMock()
        msg.answer.return_value = status_msg

        mock_qdrant = AsyncMock()
        mock_qdrant.close = AsyncMock()

        with (
            patch("reviewmind.bot.handlers.links._create_qdrant_client", return_value=mock_qdrant),
            patch("reviewmind.bot.handlers.links._ingest_and_analyse", new_callable=AsyncMock),
        ):
            await on_links_message(msg)

        msg.bot.send_chat_action.assert_called_once()

    async def test_qdrant_unavailable_reports_error(self):
        """If Qdrant client creation fails, a user-friendly message is sent."""
        from reviewmind.bot.handlers.links import _SERVICE_UNAVAILABLE_MSG, on_links_message

        msg = _make_message("https://example.com")
        status_msg = MagicMock(spec=Message)
        status_msg.edit_text = AsyncMock()
        msg.answer.return_value = status_msg

        with patch(
            "reviewmind.bot.handlers.links._create_qdrant_client",
            side_effect=RuntimeError("no qdrant"),
        ):
            await on_links_message(msg)

        status_msg.edit_text.assert_called_once_with(_SERVICE_UNAVAILABLE_MSG)

    async def test_unexpected_error_reports_gracefully(self):
        from reviewmind.bot.handlers.links import _UNEXPECTED_ERROR_MSG, on_links_message

        msg = _make_message("https://example.com")
        status_msg = MagicMock(spec=Message)
        status_msg.edit_text = AsyncMock()
        msg.answer.return_value = status_msg

        mock_qdrant = AsyncMock()
        mock_qdrant.close = AsyncMock()

        with (
            patch("reviewmind.bot.handlers.links._create_qdrant_client", return_value=mock_qdrant),
            patch(
                "reviewmind.bot.handlers.links._ingest_and_analyse",
                new_callable=AsyncMock,
                side_effect=RuntimeError("boom"),
            ),
        ):
            await on_links_message(msg)

        status_msg.edit_text.assert_called_with(_UNEXPECTED_ERROR_MSG)

    async def test_qdrant_closed_after_success(self):
        from reviewmind.bot.handlers.links import on_links_message

        msg = _make_message("https://example.com")
        status_msg = MagicMock(spec=Message)
        status_msg.edit_text = AsyncMock()
        msg.answer.return_value = status_msg

        mock_qdrant = AsyncMock()
        mock_qdrant.close = AsyncMock()

        with (
            patch("reviewmind.bot.handlers.links._create_qdrant_client", return_value=mock_qdrant),
            patch("reviewmind.bot.handlers.links._ingest_and_analyse", new_callable=AsyncMock),
        ):
            await on_links_message(msg)

        mock_qdrant.close.assert_called_once()

    async def test_qdrant_closed_after_error(self):
        from reviewmind.bot.handlers.links import on_links_message

        msg = _make_message("https://example.com")
        status_msg = MagicMock(spec=Message)
        status_msg.edit_text = AsyncMock()
        msg.answer.return_value = status_msg

        mock_qdrant = AsyncMock()
        mock_qdrant.close = AsyncMock()

        with (
            patch("reviewmind.bot.handlers.links._create_qdrant_client", return_value=mock_qdrant),
            patch(
                "reviewmind.bot.handlers.links._ingest_and_analyse",
                new_callable=AsyncMock,
                side_effect=RuntimeError("x"),
            ),
        ):
            await on_links_message(msg)

        mock_qdrant.close.assert_called_once()


# ══════════════════════════════════════════════════════════════
# Tests — _ingest_and_analyse
# ══════════════════════════════════════════════════════════════


class TestIngestAndAnalyse:
    """Tests for the _ingest_and_analyse helper."""

    def _make_status_msg(self) -> MagicMock:
        status = MagicMock(spec=Message)
        status.edit_text = AsyncMock()
        return status

    async def test_all_urls_fail_shows_error(self):
        from reviewmind.bot.handlers.links import _NO_SUCCESS_MSG, _ingest_and_analyse

        msg = _make_message("https://bad.com")
        status_msg = self._make_status_msg()

        ingestion_result = FakeIngestionResult(
            failed_count=1,
            results=[FakeSourceResult(url="https://bad.com", success=False, error="timeout")],
        )

        mock_pipeline = AsyncMock()
        mock_pipeline.ingest_urls.return_value = ingestion_result
        mock_pipeline.__aenter__ = AsyncMock(return_value=mock_pipeline)
        mock_pipeline.__aexit__ = AsyncMock(return_value=False)

        with patch("reviewmind.bot.handlers.links.IngestionPipeline", return_value=mock_pipeline):
            await _ingest_and_analyse(
                message=msg,
                status_msg=status_msg,
                urls=["https://bad.com"],
                query_text="test",
                qdrant=AsyncMock(),
                log=MagicMock(),
            )

        call_text = status_msg.edit_text.call_args.args[0]
        assert _NO_SUCCESS_MSG in call_text
        assert "https://bad.com" in call_text

    async def test_success_with_rag_answer(self):
        from reviewmind.bot.handlers.links import _ingest_and_analyse

        msg = _make_message("https://ok.com")
        status_msg = self._make_status_msg()

        ingestion_result = FakeIngestionResult(
            success_count=1,
            chunks_count=5,
            results=[FakeSourceResult(url="https://ok.com", success=True, chunks_count=5)],
        )
        rag_response = FakeRAGResponse(
            answer="✅ Плюсы\n- Great sound\n❌ Минусы\n- Expensive",
            sources=["https://ok.com"],
            confidence_met=True,
        )

        mock_pipeline = AsyncMock()
        mock_pipeline.ingest_urls.return_value = ingestion_result
        mock_pipeline.__aenter__ = AsyncMock(return_value=mock_pipeline)
        mock_pipeline.__aexit__ = AsyncMock(return_value=False)

        mock_rag = AsyncMock()
        mock_rag.query.return_value = rag_response
        mock_rag.__aenter__ = AsyncMock(return_value=mock_rag)
        mock_rag.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("reviewmind.bot.handlers.links.IngestionPipeline", return_value=mock_pipeline),
            patch("reviewmind.bot.handlers.links.RAGPipeline", return_value=mock_rag),
        ):
            await _ingest_and_analyse(
                message=msg,
                status_msg=status_msg,
                urls=["https://ok.com"],
                query_text="test headphones review",
                qdrant=AsyncMock(),
                log=MagicMock(),
            )

        call_text = status_msg.edit_text.call_args.args[0]
        assert "✅ Плюсы" in call_text
        assert "📊" in call_text

    async def test_success_includes_feedback_keyboard(self):
        from reviewmind.bot.handlers.links import _ingest_and_analyse

        msg = _make_message("https://ok.com")
        status_msg = self._make_status_msg()

        ingestion_result = FakeIngestionResult(
            success_count=1,
            chunks_count=3,
            results=[FakeSourceResult(url="https://ok.com", success=True, chunks_count=3)],
        )
        rag_response = FakeRAGResponse(answer="Analysis here", sources=["https://ok.com"])

        mock_pipeline = AsyncMock()
        mock_pipeline.ingest_urls.return_value = ingestion_result
        mock_pipeline.__aenter__ = AsyncMock(return_value=mock_pipeline)
        mock_pipeline.__aexit__ = AsyncMock(return_value=False)

        mock_rag = AsyncMock()
        mock_rag.query.return_value = rag_response
        mock_rag.__aenter__ = AsyncMock(return_value=mock_rag)
        mock_rag.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("reviewmind.bot.handlers.links.IngestionPipeline", return_value=mock_pipeline),
            patch("reviewmind.bot.handlers.links.RAGPipeline", return_value=mock_rag),
        ):
            await _ingest_and_analyse(
                message=msg,
                status_msg=status_msg,
                urls=["https://ok.com"],
                query_text="review",
                qdrant=AsyncMock(),
                log=MagicMock(),
            )

        call_kwargs = status_msg.edit_text.call_args.kwargs
        assert isinstance(call_kwargs.get("reply_markup"), InlineKeyboardMarkup)

    async def test_partial_failure_shows_both(self):
        from reviewmind.bot.handlers.links import _ingest_and_analyse

        msg = _make_message("https://ok.com https://bad.com")
        status_msg = self._make_status_msg()

        ingestion_result = FakeIngestionResult(
            success_count=1,
            failed_count=1,
            chunks_count=4,
            results=[
                FakeSourceResult(url="https://ok.com", success=True, chunks_count=4),
                FakeSourceResult(url="https://bad.com", success=False, error="timeout"),
            ],
        )
        rag_response = FakeRAGResponse(answer="Analysis", sources=["https://ok.com"])

        mock_pipeline = AsyncMock()
        mock_pipeline.ingest_urls.return_value = ingestion_result
        mock_pipeline.__aenter__ = AsyncMock(return_value=mock_pipeline)
        mock_pipeline.__aexit__ = AsyncMock(return_value=False)

        mock_rag = AsyncMock()
        mock_rag.query.return_value = rag_response
        mock_rag.__aenter__ = AsyncMock(return_value=mock_rag)
        mock_rag.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("reviewmind.bot.handlers.links.IngestionPipeline", return_value=mock_pipeline),
            patch("reviewmind.bot.handlers.links.RAGPipeline", return_value=mock_rag),
        ):
            await _ingest_and_analyse(
                message=msg,
                status_msg=status_msg,
                urls=["https://ok.com", "https://bad.com"],
                query_text="review",
                qdrant=AsyncMock(),
                log=MagicMock(),
            )

        call_text = status_msg.edit_text.call_args.args[0]
        assert "https://bad.com" in call_text  # failure reported
        assert "Analysis" in call_text  # answer included
        assert "1/2" in call_text  # summary

    async def test_rag_empty_answer_shows_error(self):
        from reviewmind.bot.handlers.links import _ANALYSIS_ERROR_MSG, _ingest_and_analyse

        msg = _make_message("https://ok.com")
        status_msg = self._make_status_msg()

        ingestion_result = FakeIngestionResult(
            success_count=1,
            chunks_count=3,
            results=[FakeSourceResult(url="https://ok.com", success=True, chunks_count=3)],
        )
        rag_response = FakeRAGResponse(answer="", error="LLM error: timeout")

        mock_pipeline = AsyncMock()
        mock_pipeline.ingest_urls.return_value = ingestion_result
        mock_pipeline.__aenter__ = AsyncMock(return_value=mock_pipeline)
        mock_pipeline.__aexit__ = AsyncMock(return_value=False)

        mock_rag = AsyncMock()
        mock_rag.query.return_value = rag_response
        mock_rag.__aenter__ = AsyncMock(return_value=mock_rag)
        mock_rag.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("reviewmind.bot.handlers.links.IngestionPipeline", return_value=mock_pipeline),
            patch("reviewmind.bot.handlers.links.RAGPipeline", return_value=mock_rag),
        ):
            await _ingest_and_analyse(
                message=msg,
                status_msg=status_msg,
                urls=["https://ok.com"],
                query_text="review",
                qdrant=AsyncMock(),
                log=MagicMock(),
            )

        call_text = status_msg.edit_text.call_args.args[0]
        assert _ANALYSIS_ERROR_MSG in call_text

    async def test_answer_truncated_at_4096(self):
        from reviewmind.bot.handlers.links import _MAX_ANSWER_LENGTH, _ingest_and_analyse

        msg = _make_message("https://ok.com")
        status_msg = self._make_status_msg()

        ingestion_result = FakeIngestionResult(
            success_count=1,
            chunks_count=1,
            results=[FakeSourceResult(url="https://ok.com", success=True, chunks_count=1)],
        )
        rag_response = FakeRAGResponse(answer="x" * 5000, sources=["https://ok.com"])

        mock_pipeline = AsyncMock()
        mock_pipeline.ingest_urls.return_value = ingestion_result
        mock_pipeline.__aenter__ = AsyncMock(return_value=mock_pipeline)
        mock_pipeline.__aexit__ = AsyncMock(return_value=False)

        mock_rag = AsyncMock()
        mock_rag.query.return_value = rag_response
        mock_rag.__aenter__ = AsyncMock(return_value=mock_rag)
        mock_rag.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("reviewmind.bot.handlers.links.IngestionPipeline", return_value=mock_pipeline),
            patch("reviewmind.bot.handlers.links.RAGPipeline", return_value=mock_rag),
        ):
            await _ingest_and_analyse(
                message=msg,
                status_msg=status_msg,
                urls=["https://ok.com"],
                query_text="review",
                qdrant=AsyncMock(),
                log=MagicMock(),
            )

        call_text = status_msg.edit_text.call_args.args[0]
        assert len(call_text) <= _MAX_ANSWER_LENGTH
        assert call_text.endswith("...")

    async def test_product_query_from_user_text(self):
        """When user provides text alongside URLs, it should be the product_query."""
        from reviewmind.bot.handlers.links import _ingest_and_analyse

        msg = _make_message("Sony XM5 https://ok.com")
        status_msg = self._make_status_msg()

        ingestion_result = FakeIngestionResult(
            success_count=1, chunks_count=1,
            results=[FakeSourceResult(url="https://ok.com", success=True, chunks_count=1)],
        )
        rag_response = FakeRAGResponse(answer="Analysis", sources=[])

        mock_pipeline = AsyncMock()
        mock_pipeline.ingest_urls.return_value = ingestion_result
        mock_pipeline.__aenter__ = AsyncMock(return_value=mock_pipeline)
        mock_pipeline.__aexit__ = AsyncMock(return_value=False)

        mock_rag = AsyncMock()
        mock_rag.query.return_value = rag_response
        mock_rag.__aenter__ = AsyncMock(return_value=mock_rag)
        mock_rag.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("reviewmind.bot.handlers.links.IngestionPipeline", return_value=mock_pipeline),
            patch("reviewmind.bot.handlers.links.RAGPipeline", return_value=mock_rag),
        ):
            await _ingest_and_analyse(
                message=msg,
                status_msg=status_msg,
                urls=["https://ok.com"],
                query_text="Sony XM5",
                qdrant=AsyncMock(),
                log=MagicMock(),
            )

        # pipeline.ingest_urls should be called with product_query="Sony XM5"
        call_kwargs = mock_pipeline.ingest_urls.call_args
        pq = call_kwargs.kwargs.get("product_query") or call_kwargs[1].get("product_query")
        assert pq == "Sony XM5"

    async def test_default_query_uses_empty_product_query(self):
        """When no user text, product_query should be empty (no filter)."""
        from reviewmind.bot.handlers.links import _DEFAULT_QUERY, _ingest_and_analyse

        msg = _make_message("https://ok.com")
        status_msg = self._make_status_msg()

        ingestion_result = FakeIngestionResult(
            success_count=1, chunks_count=1,
            results=[FakeSourceResult(url="https://ok.com", success=True, chunks_count=1)],
        )
        rag_response = FakeRAGResponse(answer="Analysis", sources=[])

        mock_pipeline = AsyncMock()
        mock_pipeline.ingest_urls.return_value = ingestion_result
        mock_pipeline.__aenter__ = AsyncMock(return_value=mock_pipeline)
        mock_pipeline.__aexit__ = AsyncMock(return_value=False)

        mock_rag = AsyncMock()
        mock_rag.query.return_value = rag_response
        mock_rag.__aenter__ = AsyncMock(return_value=mock_rag)
        mock_rag.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("reviewmind.bot.handlers.links.IngestionPipeline", return_value=mock_pipeline),
            patch("reviewmind.bot.handlers.links.RAGPipeline", return_value=mock_rag),
        ):
            await _ingest_and_analyse(
                message=msg,
                status_msg=status_msg,
                urls=["https://ok.com"],
                query_text=_DEFAULT_QUERY,
                qdrant=AsyncMock(),
                log=MagicMock(),
            )

        # product_query should be ""
        call_args = mock_pipeline.ingest_urls.call_args
        assert call_args.kwargs.get("product_query") == "" or call_args[1].get("product_query") == ""

    async def test_summary_includes_chunk_count(self):
        from reviewmind.bot.handlers.links import _ingest_and_analyse

        msg = _make_message("https://ok.com")
        status_msg = self._make_status_msg()

        ingestion_result = FakeIngestionResult(
            success_count=1, chunks_count=7,
            results=[FakeSourceResult(url="https://ok.com", success=True, chunks_count=7)],
        )
        rag_response = FakeRAGResponse(answer="Analysis", sources=[])

        mock_pipeline = AsyncMock()
        mock_pipeline.ingest_urls.return_value = ingestion_result
        mock_pipeline.__aenter__ = AsyncMock(return_value=mock_pipeline)
        mock_pipeline.__aexit__ = AsyncMock(return_value=False)

        mock_rag = AsyncMock()
        mock_rag.query.return_value = rag_response
        mock_rag.__aenter__ = AsyncMock(return_value=mock_rag)
        mock_rag.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("reviewmind.bot.handlers.links.IngestionPipeline", return_value=mock_pipeline),
            patch("reviewmind.bot.handlers.links.RAGPipeline", return_value=mock_rag),
        ):
            await _ingest_and_analyse(
                message=msg,
                status_msg=status_msg,
                urls=["https://ok.com"],
                query_text="review",
                qdrant=AsyncMock(),
                log=MagicMock(),
            )

        call_text = status_msg.edit_text.call_args.args[0]
        assert "Фрагментов: 7" in call_text


# ══════════════════════════════════════════════════════════════
# Tests — Dispatcher wiring
# ══════════════════════════════════════════════════════════════


class TestDispatcherWiring:
    _dp = None

    @classmethod
    def _get_dispatcher(cls):
        """Create dispatcher once per class, resetting parent routers first."""
        if cls._dp is not None:
            return cls._dp
        # Reset parent_router on all routers so they can be re-attached
        from reviewmind.bot.handlers.feedback import router as feedback_router
        from reviewmind.bot.handlers.links import router as links_router
        from reviewmind.bot.handlers.mode import router as mode_router
        from reviewmind.bot.handlers.query import router as query_router
        from reviewmind.bot.handlers.start import router as start_router

        for r in (start_router, mode_router, links_router, query_router, feedback_router):
            r._parent_router = None

        from reviewmind.bot.main import create_dispatcher

        cls._dp = create_dispatcher()
        return cls._dp

    def test_links_router_included(self):
        dp = self._get_dispatcher()
        router_names = [r.name for r in dp.sub_routers]
        assert "links" in router_names

    def test_links_router_before_query(self):
        """Links router must be before query (catch-all) to intercept URL messages."""
        dp = self._get_dispatcher()
        router_names = [r.name for r in dp.sub_routers]
        links_idx = router_names.index("links")
        query_idx = router_names.index("query")
        assert links_idx < query_idx

    def test_links_router_after_mode(self):
        dp = self._get_dispatcher()
        router_names = [r.name for r in dp.sub_routers]
        mode_idx = router_names.index("mode")
        links_idx = router_names.index("links")
        assert mode_idx < links_idx

    def test_links_router_has_name(self):
        from reviewmind.bot.handlers.links import router

        assert router.name == "links"


# ══════════════════════════════════════════════════════════════
# Tests — Constants & module exports
# ══════════════════════════════════════════════════════════════


class TestConstants:
    def test_max_answer_length(self):
        from reviewmind.bot.handlers.links import _MAX_ANSWER_LENGTH

        assert _MAX_ANSWER_LENGTH == 4096

    def test_default_query_is_string(self):
        from reviewmind.bot.handlers.links import _DEFAULT_QUERY

        assert isinstance(_DEFAULT_QUERY, str)
        assert len(_DEFAULT_QUERY) > 0

    def test_url_regex_matches_http(self):
        from reviewmind.bot.handlers.links import _URL_RE

        assert _URL_RE.search("http://example.com")

    def test_url_regex_matches_https(self):
        from reviewmind.bot.handlers.links import _URL_RE

        assert _URL_RE.search("https://example.com")

    def test_url_regex_no_match_ftp(self):
        from reviewmind.bot.handlers.links import _URL_RE

        assert _URL_RE.search("ftp://example.com") is None


# ══════════════════════════════════════════════════════════════
# Tests — Integration scenarios
# ══════════════════════════════════════════════════════════════


class TestIntegrationScenarios:
    """End-to-end-like scenarios using mocked services."""

    async def test_youtube_url_flow(self):
        """A YouTube URL should be ingested and analysed."""
        from reviewmind.bot.handlers.links import _ingest_and_analyse

        msg = _make_message("Sony XM5 review https://youtube.com/watch?v=abc123")
        status_msg = MagicMock(spec=Message)
        status_msg.edit_text = AsyncMock()

        ingestion_result = FakeIngestionResult(
            success_count=1, chunks_count=10,
            results=[
                FakeSourceResult(
                    url="https://youtube.com/watch?v=abc123", success=True,
                    source_type="youtube", chunks_count=10,
                ),
            ],
        )
        rag_response = FakeRAGResponse(
            answer="✅ Плюсы\n- Отличное шумоподавление\n❌ Минусы\n- Высокая цена",
            sources=["https://youtube.com/watch?v=abc123"],
            confidence_met=True,
            used_curated=False,
        )

        mock_pipeline = AsyncMock()
        mock_pipeline.ingest_urls.return_value = ingestion_result
        mock_pipeline.__aenter__ = AsyncMock(return_value=mock_pipeline)
        mock_pipeline.__aexit__ = AsyncMock(return_value=False)

        mock_rag = AsyncMock()
        mock_rag.query.return_value = rag_response
        mock_rag.__aenter__ = AsyncMock(return_value=mock_rag)
        mock_rag.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("reviewmind.bot.handlers.links.IngestionPipeline", return_value=mock_pipeline),
            patch("reviewmind.bot.handlers.links.RAGPipeline", return_value=mock_rag),
        ):
            await _ingest_and_analyse(
                message=msg,
                status_msg=status_msg,
                urls=["https://youtube.com/watch?v=abc123"],
                query_text="Sony XM5 review",
                qdrant=AsyncMock(),
                log=MagicMock(),
            )

        call_text = status_msg.edit_text.call_args.args[0]
        assert "Плюсы" in call_text
        assert "Минусы" in call_text
        assert "1/1" in call_text

    async def test_multiple_source_types(self):
        """Mix of YouTube + Reddit + web URLs."""
        from reviewmind.bot.handlers.links import _ingest_and_analyse

        urls = [
            "https://youtube.com/watch?v=a",
            "https://reddit.com/r/headphones/comments/1/x",
            "https://rtings.com/review",
        ]
        msg = _make_message(" ".join(urls))
        status_msg = MagicMock(spec=Message)
        status_msg.edit_text = AsyncMock()

        ingestion_result = FakeIngestionResult(
            success_count=3, chunks_count=15,
            results=[
                FakeSourceResult(url=urls[0], success=True, source_type="youtube", chunks_count=5),
                FakeSourceResult(url=urls[1], success=True, source_type="reddit", chunks_count=4),
                FakeSourceResult(url=urls[2], success=True, source_type="web", chunks_count=6),
            ],
        )
        rag_response = FakeRAGResponse(answer="Combined analysis", sources=urls)

        mock_pipeline = AsyncMock()
        mock_pipeline.ingest_urls.return_value = ingestion_result
        mock_pipeline.__aenter__ = AsyncMock(return_value=mock_pipeline)
        mock_pipeline.__aexit__ = AsyncMock(return_value=False)

        mock_rag = AsyncMock()
        mock_rag.query.return_value = rag_response
        mock_rag.__aenter__ = AsyncMock(return_value=mock_rag)
        mock_rag.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("reviewmind.bot.handlers.links.IngestionPipeline", return_value=mock_pipeline),
            patch("reviewmind.bot.handlers.links.RAGPipeline", return_value=mock_rag),
        ):
            await _ingest_and_analyse(
                message=msg,
                status_msg=status_msg,
                urls=urls,
                query_text="headphones review",
                qdrant=AsyncMock(),
                log=MagicMock(),
            )

        call_text = status_msg.edit_text.call_args.args[0]
        assert "3/3" in call_text
        assert "15" in call_text

    async def test_one_url_fails_others_succeed(self):
        """One broken URL shouldn't block analysis of the working ones."""
        from reviewmind.bot.handlers.links import _ingest_and_analyse

        urls = ["https://ok.com", "https://broken.invalid"]
        msg = _make_message(" ".join(urls))
        status_msg = MagicMock(spec=Message)
        status_msg.edit_text = AsyncMock()

        ingestion_result = FakeIngestionResult(
            success_count=1, failed_count=1, chunks_count=5,
            failed_urls=["https://broken.invalid"],
            results=[
                FakeSourceResult(url=urls[0], success=True, chunks_count=5),
                FakeSourceResult(url=urls[1], success=False, error="DNS resolution failed"),
            ],
        )
        rag_response = FakeRAGResponse(answer="Partial analysis", sources=["https://ok.com"])

        mock_pipeline = AsyncMock()
        mock_pipeline.ingest_urls.return_value = ingestion_result
        mock_pipeline.__aenter__ = AsyncMock(return_value=mock_pipeline)
        mock_pipeline.__aexit__ = AsyncMock(return_value=False)

        mock_rag = AsyncMock()
        mock_rag.query.return_value = rag_response
        mock_rag.__aenter__ = AsyncMock(return_value=mock_rag)
        mock_rag.__aexit__ = AsyncMock(return_value=False)

        with (
            patch("reviewmind.bot.handlers.links.IngestionPipeline", return_value=mock_pipeline),
            patch("reviewmind.bot.handlers.links.RAGPipeline", return_value=mock_rag),
        ):
            await _ingest_and_analyse(
                message=msg,
                status_msg=status_msg,
                urls=urls,
                query_text="review",
                qdrant=AsyncMock(),
                log=MagicMock(),
            )

        call_text = status_msg.edit_text.call_args.args[0]
        assert "https://broken.invalid" in call_text
        assert "Partial analysis" in call_text
        assert "1/2" in call_text
