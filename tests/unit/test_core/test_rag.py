"""Unit tests for reviewmind.core.rag — RAG pipeline orchestrator."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reviewmind.core.rag import (
    CONFIDENCE_MIN_CHUNKS,
    CONFIDENCE_SCORE_THRESHOLD,
    DEFAULT_SEARCH_TOP_K,
    MAX_CONTEXT_CHUNKS,
    RAGPipeline,
    RAGResponse,
    _check_confidence,
    _extract_sources,
    _search_results_to_chunk_contexts,
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
    extra: dict | None = None,
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
        extra=extra or {},
    )


def _make_results(count: int, *, base_score: float = 0.85) -> list[SearchResult]:
    """Create *count* distinct search results."""
    results = []
    for i in range(count):
        results.append(
            _make_result(
                text=f"chunk {i}",
                score=base_score - i * 0.01,
                source_url=f"https://example.com/{i}",
                chunk_index=i,
                point_id=f"p{i}",
            )
        )
    return results


# ── TestConstants ────────────────────────────────────────────────────────────


class TestConstants:
    """Verify module-level constants match the PRD requirements."""

    def test_confidence_min_chunks(self):
        assert CONFIDENCE_MIN_CHUNKS == 3

    def test_confidence_score_threshold(self):
        assert CONFIDENCE_SCORE_THRESHOLD == 0.75

    def test_max_context_chunks(self):
        assert MAX_CONTEXT_CHUNKS == 8

    def test_default_search_top_k(self):
        assert DEFAULT_SEARCH_TOP_K == 5


# ── TestRAGResponse ──────────────────────────────────────────────────────────


class TestRAGResponse:
    """Verify the RAGResponse dataclass."""

    def test_minimal_creation(self):
        resp = RAGResponse(answer="test")
        assert resp.answer == "test"
        assert resp.sources == []
        assert resp.used_curated is False
        assert resp.confidence_met is False
        assert resp.chunks_count == 0
        assert resp.chunks_found == 0
        assert resp.used_sponsored is False
        assert resp.error is None

    def test_full_creation(self):
        resp = RAGResponse(
            answer="analysis",
            sources=["url1", "url2"],
            used_curated=True,
            confidence_met=True,
            chunks_count=5,
            chunks_found=10,
            used_sponsored=True,
            error=None,
        )
        assert resp.answer == "analysis"
        assert len(resp.sources) == 2
        assert resp.used_curated is True
        assert resp.confidence_met is True
        assert resp.chunks_count == 5
        assert resp.chunks_found == 10
        assert resp.used_sponsored is True

    def test_error_response(self):
        resp = RAGResponse(answer="", error="something failed")
        assert resp.answer == ""
        assert resp.error == "something failed"

    def test_sources_default_independent(self):
        """Check that default list is independent across instances."""
        r1 = RAGResponse(answer="a")
        r2 = RAGResponse(answer="b")
        r1.sources.append("url")
        assert r2.sources == []


# ── TestSearchResultsToChunkContexts ─────────────────────────────────────────


class TestSearchResultsToChunkContexts:
    """Verify conversion from SearchResult to ChunkContext."""

    def test_empty_list(self):
        assert _search_results_to_chunk_contexts([]) == []

    def test_single_result(self):
        result = _make_result(
            text="hello",
            source_url="https://example.com/r",
            source_type="youtube",
            is_sponsored=True,
            is_curated=False,
            score=0.9,
        )
        chunks = _search_results_to_chunk_contexts([result])
        assert len(chunks) == 1
        c = chunks[0]
        assert c.text == "hello"
        assert c.source_url == "https://example.com/r"
        assert c.source_type == "youtube"
        assert c.is_sponsored is True
        assert c.is_curated is False
        assert c.score == 0.9

    def test_author_from_extra(self):
        result = _make_result(extra={"author": "TechGuru"})
        chunks = _search_results_to_chunk_contexts([result])
        assert chunks[0].author == "TechGuru"

    def test_author_none_when_missing(self):
        result = _make_result(extra={})
        chunks = _search_results_to_chunk_contexts([result])
        assert chunks[0].author is None

    def test_extra_metadata_contains_fields(self):
        result = _make_result(
            collection="curated_kb",
            language="ru",
            chunk_index=3,
            point_id="xyz",
            product_query="phone",
        )
        chunks = _search_results_to_chunk_contexts([result])
        meta = chunks[0].extra_metadata
        assert meta["collection"] == "curated_kb"
        assert meta["language"] == "ru"
        assert meta["chunk_index"] == 3
        assert meta["point_id"] == "xyz"
        assert meta["product_query"] == "phone"

    def test_multiple_results(self):
        results = _make_results(3)
        chunks = _search_results_to_chunk_contexts(results)
        assert len(chunks) == 3
        assert chunks[0].text == "chunk 0"
        assert chunks[2].text == "chunk 2"


# ── TestExtractSources ───────────────────────────────────────────────────────


class TestExtractSources:
    """Verify _extract_sources helper."""

    def test_empty_list(self):
        assert _extract_sources([]) == []

    def test_single_source(self):
        results = [_make_result(source_url="https://a.com")]
        assert _extract_sources(results) == ["https://a.com"]

    def test_deduplication(self):
        results = [
            _make_result(source_url="https://a.com", chunk_index=0),
            _make_result(source_url="https://a.com", chunk_index=1),
            _make_result(source_url="https://b.com", chunk_index=0),
        ]
        sources = _extract_sources(results)
        assert sources == ["https://a.com", "https://b.com"]

    def test_order_preserved(self):
        results = [
            _make_result(source_url="https://c.com"),
            _make_result(source_url="https://a.com"),
            _make_result(source_url="https://b.com"),
        ]
        sources = _extract_sources(results)
        assert sources == ["https://c.com", "https://a.com", "https://b.com"]

    def test_empty_url_skipped(self):
        results = [
            _make_result(source_url=""),
            _make_result(source_url="https://a.com"),
        ]
        sources = _extract_sources(results)
        assert sources == ["https://a.com"]


# ── TestCheckConfidence ──────────────────────────────────────────────────────


class TestCheckConfidence:
    """Verify the _check_confidence helper."""

    def test_empty_results(self):
        assert _check_confidence([]) is False

    def test_below_threshold(self):
        results = [_make_result(score=0.5), _make_result(score=0.6)]
        assert _check_confidence(results) is False

    def test_exactly_at_threshold_not_counted(self):
        """Score must be strictly greater than threshold."""
        results = [
            _make_result(score=CONFIDENCE_SCORE_THRESHOLD),
            _make_result(score=CONFIDENCE_SCORE_THRESHOLD),
            _make_result(score=CONFIDENCE_SCORE_THRESHOLD),
        ]
        assert _check_confidence(results) is False

    def test_above_threshold_but_not_enough(self):
        results = [
            _make_result(score=0.9),
            _make_result(score=0.85),
        ]
        assert _check_confidence(results) is False

    def test_exactly_min_chunks_above(self):
        results = _make_results(CONFIDENCE_MIN_CHUNKS, base_score=0.9)
        assert _check_confidence(results) is True

    def test_more_than_min_chunks_above(self):
        results = _make_results(5, base_score=0.9)
        assert _check_confidence(results) is True

    def test_mixed_scores(self):
        """3 above threshold + 2 below → confidence met."""
        results = [
            _make_result(score=0.9),
            _make_result(score=0.85),
            _make_result(score=0.76),
            _make_result(score=0.5),
            _make_result(score=0.3),
        ]
        assert _check_confidence(results) is True


# ── TestRAGPipelineInit ──────────────────────────────────────────────────────


class TestRAGPipelineInit:
    """Verify RAGPipeline initialization."""

    def test_init_with_all_deps(self):
        qdrant = MagicMock()
        embed = MagicMock()
        llm = MagicMock()
        pipeline = RAGPipeline(qdrant, embed, llm)
        assert pipeline._qdrant is qdrant
        assert pipeline._embedding is embed
        assert pipeline._llm is llm
        assert pipeline._owns_embedding is False
        assert pipeline._owns_llm is False

    def test_init_lazy_services(self):
        qdrant = MagicMock()
        pipeline = RAGPipeline(qdrant)
        assert pipeline._embedding is None
        assert pipeline._llm is None
        assert pipeline._owns_embedding is True
        assert pipeline._owns_llm is True

    def test_custom_top_k(self):
        qdrant = MagicMock()
        pipeline = RAGPipeline(qdrant, search_top_k=10, rerank_top_k=15)
        assert pipeline._search_top_k == 10
        assert pipeline._rerank_top_k == 15

    def test_default_top_k(self):
        qdrant = MagicMock()
        pipeline = RAGPipeline(qdrant)
        assert pipeline._search_top_k == DEFAULT_SEARCH_TOP_K
        assert pipeline._rerank_top_k == 8  # DEFAULT_RERANK_TOP_K


# ── TestRAGPipelineLazyProperties ────────────────────────────────────────────


class TestRAGPipelineLazyProperties:
    """Verify lazy creation of embedding and LLM services."""

    @patch("reviewmind.core.rag.EmbeddingService")
    def test_embedding_service_lazy_creation(self, mock_cls):
        mock_cls.return_value = MagicMock()
        qdrant = MagicMock()
        pipeline = RAGPipeline(qdrant)

        # First access creates the service
        svc = pipeline.embedding_service
        mock_cls.assert_called_once()
        assert svc is mock_cls.return_value

        # Second access returns the same instance
        svc2 = pipeline.embedding_service
        assert svc2 is svc
        mock_cls.assert_called_once()

    @patch("reviewmind.core.rag.LLMClient")
    def test_llm_client_lazy_creation(self, mock_cls):
        mock_cls.return_value = MagicMock()
        qdrant = MagicMock()
        pipeline = RAGPipeline(qdrant)

        client = pipeline.llm_client
        mock_cls.assert_called_once()
        assert client is mock_cls.return_value

    def test_provided_services_not_recreated(self):
        qdrant = MagicMock()
        embed = MagicMock()
        llm = MagicMock()
        pipeline = RAGPipeline(qdrant, embed, llm)
        assert pipeline.embedding_service is embed
        assert pipeline.llm_client is llm


# ── TestRAGPipelineQuery ─────────────────────────────────────────────────────


class TestRAGPipelineQuery:
    """Verify the full query() pipeline with mocked dependencies."""

    @pytest.fixture
    def mock_embedding(self):
        svc = AsyncMock()
        svc.embed_text = AsyncMock(return_value=[0.1] * 1536)
        svc.close = AsyncMock()
        return svc

    @pytest.fixture
    def mock_llm(self):
        llm = AsyncMock()
        llm.generate_analysis = AsyncMock(return_value="✅ Плюсы\n- Good sound")
        llm.close = AsyncMock()
        return llm

    @pytest.fixture
    def mock_qdrant(self):
        return AsyncMock()

    @pytest.fixture
    def search_results_confident(self):
        """5 results, all above confidence threshold."""
        return [
            _make_result(
                text=f"chunk {i}",
                score=0.9 - i * 0.02,
                source_url=f"https://source{i}.com/review",
                source_type="youtube" if i % 2 == 0 else "web",
                is_curated=i == 0,
                is_sponsored=i == 3,
                chunk_index=i,
                point_id=f"p{i}",
            )
            for i in range(5)
        ]

    @pytest.fixture
    def search_results_low_confidence(self):
        """3 results, all below confidence threshold."""
        return [
            _make_result(score=0.5, chunk_index=0, source_url="https://a.com"),
            _make_result(score=0.4, chunk_index=1, source_url="https://b.com"),
            _make_result(score=0.3, chunk_index=2, source_url="https://c.com"),
        ]

    @pytest.mark.asyncio
    async def test_full_pipeline_happy_path(self, mock_qdrant, mock_embedding, mock_llm, search_results_confident):
        with patch(
            "reviewmind.core.rag.hybrid_search",
            new_callable=AsyncMock,
            return_value=search_results_confident,
        ):
            pipeline = RAGPipeline(mock_qdrant, mock_embedding, mock_llm)
            resp = await pipeline.query("Sony WH-1000XM5 стоит ли?")

        assert resp.answer == "✅ Плюсы\n- Good sound"
        assert resp.error is None
        assert resp.chunks_count > 0
        assert resp.chunks_found == 5
        assert len(resp.sources) > 0
        assert resp.confidence_met is True
        assert resp.used_curated is True  # i==0 is_curated=True
        mock_embedding.embed_text.assert_awaited_once()
        mock_llm.generate_analysis.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_pipeline_low_confidence(self, mock_qdrant, mock_embedding, mock_llm, search_results_low_confidence):
        with patch(
            "reviewmind.core.rag.hybrid_search",
            new_callable=AsyncMock,
            return_value=search_results_low_confidence,
        ):
            pipeline = RAGPipeline(mock_qdrant, mock_embedding, mock_llm)
            resp = await pipeline.query("unknown product")

        assert resp.confidence_met is False
        assert resp.answer != ""  # still generates an answer
        assert resp.error is None

    @pytest.mark.asyncio
    async def test_pipeline_no_results(self, mock_qdrant, mock_embedding, mock_llm):
        with (
            patch(
                "reviewmind.core.rag.hybrid_search",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch.object(RAGPipeline, "_tavily_fallback", new_callable=AsyncMock, return_value=[]),
        ):
            pipeline = RAGPipeline(mock_qdrant, mock_embedding, mock_llm)
            resp = await pipeline.query("no data product")

        assert resp.confidence_met is False
        assert resp.chunks_count == 0
        assert resp.chunks_found == 0
        assert resp.sources == []
        assert resp.used_curated is False
        # LLM is still called with empty context
        mock_llm.generate_analysis.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_pipeline_passes_product_query(self, mock_qdrant, mock_embedding, mock_llm):
        mock_search = AsyncMock(return_value=[])
        with patch("reviewmind.core.rag.hybrid_search", mock_search):
            pipeline = RAGPipeline(mock_qdrant, mock_embedding, mock_llm)
            await pipeline.query("best headphones?", product_query="Sony WH-1000XM5")

        mock_search.assert_awaited_once()
        call_kwargs = mock_search.call_args
        assert call_kwargs.kwargs.get("product_query") == "Sony WH-1000XM5"

    @pytest.mark.asyncio
    async def test_pipeline_passes_chat_history(self, mock_qdrant, mock_embedding, mock_llm):
        history = [
            {"role": "user", "content": "Tell me about XM5"},
            {"role": "assistant", "content": "Great headphones!"},
        ]
        with patch(
            "reviewmind.core.rag.hybrid_search",
            new_callable=AsyncMock,
            return_value=[],
        ):
            pipeline = RAGPipeline(mock_qdrant, mock_embedding, mock_llm)
            await pipeline.query("what about battery?", chat_history=history)

        call_kwargs = mock_llm.generate_analysis.call_args
        assert call_kwargs.kwargs.get("chat_history") == history

    @pytest.mark.asyncio
    async def test_pipeline_search_top_k(self, mock_qdrant, mock_embedding, mock_llm):
        mock_search = AsyncMock(return_value=[])
        with patch("reviewmind.core.rag.hybrid_search", mock_search):
            pipeline = RAGPipeline(mock_qdrant, mock_embedding, mock_llm, search_top_k=10)
            await pipeline.query("test")

        call_kwargs = mock_search.call_args
        assert call_kwargs.kwargs.get("top_k") == 10

    @pytest.mark.asyncio
    async def test_pipeline_rerank_top_k(self, mock_qdrant, mock_embedding, mock_llm):
        """Verify custom rerank_top_k is passed to rerank()."""
        results = _make_results(15, base_score=0.95)
        mock_rerank = MagicMock(side_effect=lambda res, top_k: res[:top_k])
        with (
            patch(
                "reviewmind.core.rag.hybrid_search",
                new_callable=AsyncMock,
                return_value=results,
            ),
            patch("reviewmind.core.rag.rerank", mock_rerank),
        ):
            pipeline = RAGPipeline(mock_qdrant, mock_embedding, mock_llm, rerank_top_k=4)
            resp = await pipeline.query("test")

        mock_rerank.assert_called_once()
        assert mock_rerank.call_args.kwargs.get("top_k") == 4
        # context limited by MAX_CONTEXT_CHUNKS or rerank result
        assert resp.chunks_count <= 4

    @pytest.mark.asyncio
    async def test_pipeline_max_context_chunks_limit(self, mock_qdrant, mock_embedding, mock_llm):
        """When rerank returns > MAX_CONTEXT_CHUNKS, context is trimmed."""
        results = _make_results(12, base_score=0.95)
        with (
            patch(
                "reviewmind.core.rag.hybrid_search",
                new_callable=AsyncMock,
                return_value=results,
            ),
            patch(
                "reviewmind.core.rag.rerank",
                side_effect=lambda res, top_k: res[:top_k],
            ),
        ):
            pipeline = RAGPipeline(mock_qdrant, mock_embedding, mock_llm, rerank_top_k=12)
            resp = await pipeline.query("test")

        assert resp.chunks_count <= MAX_CONTEXT_CHUNKS

    @pytest.mark.asyncio
    async def test_used_sponsored_flag(self, mock_qdrant, mock_embedding, mock_llm):
        results = [
            _make_result(score=0.9, is_sponsored=True, source_url="https://sp1.com"),
            _make_result(score=0.85, source_url="https://clean.com", chunk_index=1),
        ]
        with patch(
            "reviewmind.core.rag.hybrid_search",
            new_callable=AsyncMock,
            return_value=results,
        ):
            pipeline = RAGPipeline(mock_qdrant, mock_embedding, mock_llm)
            resp = await pipeline.query("test")

        assert resp.used_sponsored is True

    @pytest.mark.asyncio
    async def test_no_sponsored_flag(self, mock_qdrant, mock_embedding, mock_llm):
        results = [_make_result(score=0.9, is_sponsored=False)]
        with patch(
            "reviewmind.core.rag.hybrid_search",
            new_callable=AsyncMock,
            return_value=results,
        ):
            pipeline = RAGPipeline(mock_qdrant, mock_embedding, mock_llm)
            resp = await pipeline.query("test")

        assert resp.used_sponsored is False


# ── TestRAGPipelineErrors ────────────────────────────────────────────────────


class TestRAGPipelineErrors:
    """Verify graceful error handling in the pipeline."""

    @pytest.fixture
    def mock_qdrant(self):
        return AsyncMock()

    @pytest.fixture
    def mock_llm(self):
        llm = AsyncMock()
        llm.generate_analysis = AsyncMock(return_value="answer")
        llm.close = AsyncMock()
        return llm

    @pytest.mark.asyncio
    async def test_embedding_error(self, mock_qdrant, mock_llm):
        from reviewmind.core.embeddings import EmbeddingError

        embed = AsyncMock()
        embed.embed_text = AsyncMock(side_effect=EmbeddingError("API down"))
        embed.close = AsyncMock()

        pipeline = RAGPipeline(mock_qdrant, embed, mock_llm)
        resp = await pipeline.query("test")

        assert resp.answer == ""
        assert "Embedding error" in resp.error
        mock_llm.generate_analysis.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_search_error(self, mock_qdrant, mock_llm):
        embed = AsyncMock()
        embed.embed_text = AsyncMock(return_value=[0.1] * 1536)
        embed.close = AsyncMock()

        with patch(
            "reviewmind.core.rag.hybrid_search",
            new_callable=AsyncMock,
            side_effect=RuntimeError("Qdrant connection lost"),
        ):
            pipeline = RAGPipeline(mock_qdrant, embed, mock_llm)
            resp = await pipeline.query("test")

        assert resp.answer == ""
        assert "Search error" in resp.error

    @pytest.mark.asyncio
    async def test_llm_error(self, mock_qdrant):
        from reviewmind.core.llm import LLMError

        embed = AsyncMock()
        embed.embed_text = AsyncMock(return_value=[0.1] * 1536)
        embed.close = AsyncMock()

        llm = AsyncMock()
        llm.generate_analysis = AsyncMock(side_effect=LLMError("model error"))
        llm.close = AsyncMock()

        results = _make_results(3, base_score=0.9)
        with patch(
            "reviewmind.core.rag.hybrid_search",
            new_callable=AsyncMock,
            return_value=results,
        ):
            pipeline = RAGPipeline(mock_qdrant, embed, llm)
            resp = await pipeline.query("test")

        assert resp.answer == ""
        assert "LLM error" in resp.error
        # Metadata should still be populated even on LLM error
        assert resp.chunks_count > 0
        assert resp.chunks_found == 3
        assert len(resp.sources) > 0


# ── TestRAGPipelineLifecycle ─────────────────────────────────────────────────


class TestRAGPipelineLifecycle:
    """Verify close() and context manager behavior."""

    @pytest.mark.asyncio
    async def test_close_owned_services(self):
        """When embedding/llm were created lazily, close() shuts them down."""
        embed = AsyncMock()
        embed.close = AsyncMock()
        llm = AsyncMock()
        llm.close = AsyncMock()

        qdrant = AsyncMock()

        # Simulate lazy creation
        pipeline = RAGPipeline(qdrant)
        pipeline._embedding = embed
        pipeline._llm = llm
        pipeline._owns_embedding = True
        pipeline._owns_llm = True

        await pipeline.close()
        embed.close.assert_awaited_once()
        llm.close.assert_awaited_once()
        assert pipeline._embedding is None
        assert pipeline._llm is None

    @pytest.mark.asyncio
    async def test_close_does_not_close_external_services(self):
        """When services were provided externally, close() leaves them alone."""
        embed = AsyncMock()
        embed.close = AsyncMock()
        llm = AsyncMock()
        llm.close = AsyncMock()

        qdrant = AsyncMock()
        pipeline = RAGPipeline(qdrant, embed, llm)

        await pipeline.close()
        embed.close.assert_not_awaited()
        llm.close.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_context_manager(self):
        qdrant = AsyncMock()
        embed = AsyncMock()
        embed.close = AsyncMock()
        llm = AsyncMock()
        llm.close = AsyncMock()

        async with RAGPipeline(qdrant, embed, llm) as pipeline:
            assert isinstance(pipeline, RAGPipeline)

        # External services should not be closed
        embed.close.assert_not_awaited()
        llm.close.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_context_manager_closes_owned(self):
        qdrant = AsyncMock()

        embed = AsyncMock()
        embed.close = AsyncMock()
        llm = AsyncMock()
        llm.close = AsyncMock()

        pipeline = RAGPipeline(qdrant)
        pipeline._embedding = embed
        pipeline._llm = llm
        pipeline._owns_embedding = True
        pipeline._owns_llm = True

        async with pipeline:
            pass

        embed.close.assert_awaited_once()
        llm.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_close_without_lazy_init(self):
        """close() should not fail if services were never created."""
        qdrant = AsyncMock()
        pipeline = RAGPipeline(qdrant)
        await pipeline.close()  # Should not raise


# ── TestRAGPipelineIntegration ───────────────────────────────────────────────


class TestRAGPipelineIntegration:
    """Higher-level integration-like tests with full mock wiring."""

    @pytest.mark.asyncio
    async def test_e2e_curated_and_auto_mixed(self):
        """Simulate a query that returns curated + auto results."""
        results = [
            _make_result(
                text="Expert review: great noise cancelling",
                score=0.92,
                source_url="https://wirecutter.com/headphones",
                source_type="web",
                is_curated=True,
                collection="curated_kb",
            ),
            _make_result(
                text="User says: amazing battery life",
                score=0.88,
                source_url="https://reddit.com/r/headphones/1",
                source_type="reddit",
                is_curated=False,
                collection="auto_crawled",
                chunk_index=1,
            ),
            _make_result(
                text="Sponsored content: buy now!",
                score=0.85,
                source_url="https://sponsor.com/review",
                source_type="web",
                is_curated=False,
                is_sponsored=True,
                collection="auto_crawled",
                chunk_index=2,
            ),
            _make_result(
                text="YouTube: detailed sound test",
                score=0.82,
                source_url="https://youtube.com/watch?v=123",
                source_type="youtube",
                is_curated=False,
                collection="auto_crawled",
                chunk_index=3,
            ),
        ]

        embed = AsyncMock()
        embed.embed_text = AsyncMock(return_value=[0.1] * 1536)
        embed.close = AsyncMock()

        llm = AsyncMock()
        llm.generate_analysis = AsyncMock(
            return_value=(
                "✅ Плюсы\n- Отличное шумоподавление\n"
                "❌ Минусы\n- Высокая цена\n"
                "⚖️ Спорные моменты\n- Комфорт\n"
                "🏆 Вывод\n- Рекомендуется"
            )
        )
        llm.close = AsyncMock()

        qdrant = AsyncMock()

        with patch(
            "reviewmind.core.rag.hybrid_search",
            new_callable=AsyncMock,
            return_value=results,
        ):
            pipeline = RAGPipeline(qdrant, embed, llm)
            resp = await pipeline.query(
                "Sony WH-1000XM5 стоит ли покупать?",
                product_query="Sony WH-1000XM5",
            )

        assert "Плюсы" in resp.answer
        assert resp.used_curated is True
        assert resp.used_sponsored is True
        assert resp.confidence_met is True
        assert len(resp.sources) == 4
        assert resp.chunks_count == 4
        assert resp.error is None

    @pytest.mark.asyncio
    async def test_e2e_no_data_product(self):
        """Simulate a query for a product with no data in Qdrant."""
        embed = AsyncMock()
        embed.embed_text = AsyncMock(return_value=[0.1] * 1536)
        embed.close = AsyncMock()

        llm = AsyncMock()
        llm.generate_analysis = AsyncMock(return_value="Контекста недостаточно для полного анализа.")
        llm.close = AsyncMock()

        qdrant = AsyncMock()

        with (
            patch(
                "reviewmind.core.rag.hybrid_search",
                new_callable=AsyncMock,
                return_value=[],
            ),
            patch.object(RAGPipeline, "_tavily_fallback", new_callable=AsyncMock, return_value=[]),
        ):
            pipeline = RAGPipeline(qdrant, embed, llm)
            resp = await pipeline.query("Unknown Gadget X review")

        assert resp.confidence_met is False
        assert resp.chunks_count == 0
        assert resp.sources == []
        assert resp.answer != ""
        assert resp.error is None

    @pytest.mark.asyncio
    async def test_e2e_with_chat_history(self):
        """Verify chat history flows through to generate_analysis."""
        results = _make_results(3, base_score=0.9)
        history = [
            {"role": "user", "content": "Tell me about XM5"},
            {"role": "assistant", "content": "They are great headphones."},
        ]

        embed = AsyncMock()
        embed.embed_text = AsyncMock(return_value=[0.1] * 1536)
        embed.close = AsyncMock()

        llm = AsyncMock()
        llm.generate_analysis = AsyncMock(return_value="Battery is 30h.")
        llm.close = AsyncMock()

        qdrant = AsyncMock()

        with patch(
            "reviewmind.core.rag.hybrid_search",
            new_callable=AsyncMock,
            return_value=results,
        ):
            pipeline = RAGPipeline(qdrant, embed, llm)
            resp = await pipeline.query(
                "А что с батареей?",
                chat_history=history,
            )

        # Verify history was passed to LLM
        call_kwargs = llm.generate_analysis.call_args
        assert call_kwargs.kwargs.get("chat_history") == history
        assert resp.answer == "Battery is 30h."

    @pytest.mark.asyncio
    async def test_e2e_session_id_logging(self):
        """session_id should not affect pipeline logic, just logging."""
        embed = AsyncMock()
        embed.embed_text = AsyncMock(return_value=[0.1] * 1536)
        embed.close = AsyncMock()

        llm = AsyncMock()
        llm.generate_analysis = AsyncMock(return_value="answer")
        llm.close = AsyncMock()

        qdrant = AsyncMock()

        with patch(
            "reviewmind.core.rag.hybrid_search",
            new_callable=AsyncMock,
            return_value=[],
        ):
            pipeline = RAGPipeline(qdrant, embed, llm)
            resp = await pipeline.query("test", session_id="sess-123")

        assert resp.error is None

    @pytest.mark.asyncio
    async def test_chunks_sent_to_llm_as_chunk_contexts(self):
        """Verify that ChunkContext objects are correctly built from SearchResults."""
        from reviewmind.core.prompts import ChunkContext

        results = [
            _make_result(
                text="review text",
                score=0.9,
                source_url="https://source.com",
                source_type="youtube",
                is_curated=True,
                is_sponsored=False,
                extra={"author": "TechChannel"},
            ),
        ]

        embed = AsyncMock()
        embed.embed_text = AsyncMock(return_value=[0.1] * 1536)
        embed.close = AsyncMock()

        llm = AsyncMock()
        llm.generate_analysis = AsyncMock(return_value="response")
        llm.close = AsyncMock()

        qdrant = AsyncMock()

        with (
            patch(
                "reviewmind.core.rag.hybrid_search",
                new_callable=AsyncMock,
                return_value=results,
            ),
            patch.object(RAGPipeline, "_tavily_fallback", new_callable=AsyncMock, return_value=[]),
        ):
            pipeline = RAGPipeline(qdrant, embed, llm)
            await pipeline.query("test")

        call_args = llm.generate_analysis.call_args
        chunks = call_args.kwargs.get("chunks")
        assert chunks is not None
        assert len(chunks) == 1
        assert isinstance(chunks[0], ChunkContext)
        assert chunks[0].text == "review text"
        assert chunks[0].source_url == "https://source.com"
        assert chunks[0].source_type == "youtube"
        assert chunks[0].is_curated is True
        assert chunks[0].author == "TechChannel"
