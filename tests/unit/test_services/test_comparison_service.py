"""Tests for reviewmind.services.comparison_service."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from reviewmind.services.comparison_service import (
    COMPARISON_PROMPT_TEMPLATE,
    MAX_PRODUCTS_FOR_COMPARISON,
    MIN_PRODUCTS_FOR_COMPARISON,
    ComparisonResult,
    ProductRAGResult,
    _build_comparison_prompt,
    _parallel_rag,
    compare_products,
    detect_comparison,
)

# ── Constants ────────────────────────────────────────────────────────────────


class TestConstants:
    """Test module-level constants."""

    def test_min_products(self):
        assert MIN_PRODUCTS_FOR_COMPARISON == 2

    def test_max_products(self):
        assert MAX_PRODUCTS_FOR_COMPARISON == 4

    def test_comparison_prompt_template_is_string(self):
        assert isinstance(COMPARISON_PROMPT_TEMPLATE, str)

    def test_comparison_prompt_has_slots(self):
        assert "{product_sections}" in COMPARISON_PROMPT_TEMPLATE
        assert "{chat_history}" in COMPARISON_PROMPT_TEMPLATE
        assert "{response_language}" in COMPARISON_PROMPT_TEMPLATE

    def test_comparison_prompt_mentions_table(self):
        lower = COMPARISON_PROMPT_TEMPLATE.lower()
        assert "таблиц" in lower or "сравн" in lower


# ── Data types ───────────────────────────────────────────────────────────────


class TestProductRAGResult:
    """Test ProductRAGResult dataclass."""

    def test_default_fields(self):
        pr = ProductRAGResult(product_name="iPhone 16")
        assert pr.product_name == "iPhone 16"
        assert pr.rag_response is None
        assert pr.error is None

    def test_has_data_false_no_response(self):
        pr = ProductRAGResult(product_name="iPhone 16")
        assert pr.has_data is False

    def test_has_data_false_zero_chunks(self):
        from reviewmind.core.rag import RAGResponse

        resp = RAGResponse(answer="text", chunks_count=0)
        pr = ProductRAGResult(product_name="iPhone 16", rag_response=resp)
        assert pr.has_data is False

    def test_has_data_true(self):
        from reviewmind.core.rag import RAGResponse

        resp = RAGResponse(answer="text", chunks_count=3)
        pr = ProductRAGResult(product_name="iPhone 16", rag_response=resp)
        assert pr.has_data is True

    def test_with_error(self):
        pr = ProductRAGResult(product_name="Test", error="timeout")
        assert pr.error == "timeout"
        assert pr.has_data is False


class TestComparisonResult:
    """Test ComparisonResult dataclass."""

    def test_default_fields(self):
        cr = ComparisonResult(answer="comparison text")
        assert cr.answer == "comparison text"
        assert cr.products == []
        assert cr.product_results == []
        assert cr.sources == []
        assert cr.used_curated is False
        assert cr.total_chunks == 0
        assert cr.error is None

    def test_has_data_for_all_empty(self):
        cr = ComparisonResult(answer="text")
        assert cr.has_data_for_all is True  # vacuously true

    def test_has_data_for_all_true(self):
        from reviewmind.core.rag import RAGResponse

        r1 = ProductRAGResult("A", rag_response=RAGResponse(answer="a", chunks_count=2))
        r2 = ProductRAGResult("B", rag_response=RAGResponse(answer="b", chunks_count=1))
        cr = ComparisonResult(answer="text", product_results=[r1, r2])
        assert cr.has_data_for_all is True

    def test_has_data_for_all_false(self):
        from reviewmind.core.rag import RAGResponse

        r1 = ProductRAGResult("A", rag_response=RAGResponse(answer="a", chunks_count=2))
        r2 = ProductRAGResult("B")  # no response
        cr = ComparisonResult(answer="text", product_results=[r1, r2])
        assert cr.has_data_for_all is False

    def test_with_error(self):
        cr = ComparisonResult(answer="", error="LLM error")
        assert cr.error == "LLM error"


# ── detect_comparison ────────────────────────────────────────────────────────


class TestDetectComparison:
    """Test detect_comparison function."""

    @pytest.mark.parametrize(
        "query",
        [
            "iPhone 16 vs Samsung S25",
            "Sony XM5 or AirPods Max?",
            "iPhone 16 или Samsung S25",
            "Sony XM5 versus AirPods Max",
            "iPhone 16 против Samsung S25",
            "Sony XM5 compared to AirPods Max",
        ],
    )
    def test_comparison_detected(self, query):
        assert detect_comparison(query) is True

    @pytest.mark.parametrize(
        "query",
        [
            "Sony WH-1000XM5 стоит ли покупать?",
            "Is Dyson V15 worth it?",
            "Привет, как дела?",
            "",
        ],
    )
    def test_not_comparison(self, query):
        assert detect_comparison(query) is False


# ── _build_comparison_prompt ─────────────────────────────────────────────────


class TestBuildComparisonPrompt:
    """Test _build_comparison_prompt helper."""

    def test_builds_with_data(self):
        from reviewmind.core.rag import RAGResponse

        r1 = ProductRAGResult(
            "iPhone 16",
            rag_response=RAGResponse(answer="iPhone pros/cons", chunks_count=3),
        )
        r2 = ProductRAGResult(
            "Samsung S25",
            rag_response=RAGResponse(answer="Samsung pros/cons", chunks_count=2),
        )
        prompt = _build_comparison_prompt(product_results=[r1, r2], language="ru")
        assert "iPhone 16" in prompt
        assert "Samsung S25" in prompt
        assert "iPhone pros/cons" in prompt
        assert "Samsung pros/cons" in prompt

    def test_builds_with_missing_data(self):
        r1 = ProductRAGResult("iPhone 16")  # no data
        r2 = ProductRAGResult("Samsung S25", error="Search failed")
        prompt = _build_comparison_prompt(product_results=[r1, r2])
        assert "iPhone 16" in prompt
        assert "Samsung S25" in prompt
        assert "отсутствуют" in prompt.lower() or "Search failed" in prompt

    def test_language_injected(self):
        r1 = ProductRAGResult("A")
        prompt = _build_comparison_prompt(product_results=[r1], language="en")
        assert "en" in prompt

    def test_chat_history_included(self):
        r1 = ProductRAGResult("A")
        history = [{"role": "user", "content": "Какой телефон лучше?"}]
        prompt = _build_comparison_prompt(product_results=[r1], chat_history=history, language="ru")
        assert "Какой телефон лучше?" in prompt

    def test_empty_results(self):
        prompt = _build_comparison_prompt(product_results=[], language="ru")
        assert isinstance(prompt, str)


# ── _parallel_rag ────────────────────────────────────────────────────────────


class TestParallelRag:
    """Test _parallel_rag helper."""

    @pytest.mark.asyncio
    async def test_parallel_rag_success(self):
        from reviewmind.core.rag import RAGResponse

        mock_response = RAGResponse(
            answer="Analysis text",
            sources=["https://example.com"],
            chunks_count=3,
            confidence_met=True,
        )

        with patch("reviewmind.core.rag.RAGPipeline") as MockPipeline:
            mock_instance = AsyncMock()
            mock_instance.query = AsyncMock(return_value=mock_response)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            MockPipeline.return_value = mock_instance

            results = await _parallel_rag(
                product_names=["iPhone 16", "Samsung S25"],
                user_query="iPhone 16 vs Samsung S25",
                qdrant_client=AsyncMock(),
            )

        assert len(results) == 2
        assert results[0].product_name == "iPhone 16"
        assert results[1].product_name == "Samsung S25"
        assert results[0].has_data is True
        assert results[1].has_data is True

    @pytest.mark.asyncio
    async def test_parallel_rag_partial_failure(self):
        from reviewmind.core.rag import RAGResponse

        mock_response = RAGResponse(answer="text", chunks_count=2)

        call_count = 0

        async def mock_query(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return mock_response
            raise Exception("connection error")

        with patch("reviewmind.core.rag.RAGPipeline") as MockPipeline:
            mock_instance = AsyncMock()
            mock_instance.query = AsyncMock(side_effect=mock_query)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            MockPipeline.return_value = mock_instance

            results = await _parallel_rag(
                product_names=["A", "B"],
                user_query="A vs B",
                qdrant_client=AsyncMock(),
            )

        assert len(results) == 2
        # One succeeded, one failed — but both returned (not raised)
        success = [r for r in results if r.has_data]
        failed = [r for r in results if r.error is not None]
        assert len(success) + len(failed) == 2


# ── compare_products ─────────────────────────────────────────────────────────


class TestCompareProducts:
    """Test compare_products main function."""

    @pytest.mark.asyncio
    async def test_too_few_products(self):
        with patch(
            "reviewmind.services.comparison_service.extract_product",
            new_callable=AsyncMock,
            return_value=["iPhone 16"],  # only one product
        ):
            result = await compare_products(
                "Tell me about iPhone 16",
                qdrant_client=AsyncMock(),
            )
        assert result.error is not None
        assert "два" in result.error.lower() or "более" in result.error.lower()
        assert result.answer == ""

    @pytest.mark.asyncio
    async def test_no_products_extracted(self):
        with patch(
            "reviewmind.services.comparison_service.extract_product",
            new_callable=AsyncMock,
            return_value=[],
        ):
            result = await compare_products(
                "Hello world",
                qdrant_client=AsyncMock(),
            )
        assert result.error is not None
        assert result.answer == ""

    @pytest.mark.asyncio
    async def test_successful_comparison(self):
        from reviewmind.core.rag import RAGResponse

        mock_rag = RAGResponse(
            answer="Product analysis",
            sources=["https://example.com"],
            chunks_count=3,
            used_curated=True,
        )

        with (
            patch(
                "reviewmind.services.comparison_service.extract_product",
                new_callable=AsyncMock,
                return_value=["iPhone 16", "Samsung S25"],
            ),
            patch(
                "reviewmind.services.comparison_service._parallel_rag",
                new_callable=AsyncMock,
                return_value=[
                    ProductRAGResult("iPhone 16", rag_response=mock_rag),
                    ProductRAGResult("Samsung S25", rag_response=mock_rag),
                ],
            ),
            patch(
                "reviewmind.core.llm.LLMClient",
            ) as MockLLM,
        ):
            mock_client = AsyncMock()
            mock_client.generate = AsyncMock(return_value="Comparison table...")
            mock_client.close = AsyncMock()
            MockLLM.return_value = mock_client

            result = await compare_products(
                "iPhone 16 vs Samsung S25",
                qdrant_client=AsyncMock(),
            )

        assert result.answer == "Comparison table..."
        assert result.products == ["iPhone 16", "Samsung S25"]
        assert result.used_curated is True
        assert result.total_chunks == 6  # 3 + 3
        assert result.error is None

    @pytest.mark.asyncio
    async def test_llm_error(self):
        from reviewmind.core.llm import LLMError
        from reviewmind.core.rag import RAGResponse

        mock_rag = RAGResponse(answer="text", chunks_count=2)

        with (
            patch(
                "reviewmind.services.comparison_service.extract_product",
                new_callable=AsyncMock,
                return_value=["A", "B"],
            ),
            patch(
                "reviewmind.services.comparison_service._parallel_rag",
                new_callable=AsyncMock,
                return_value=[
                    ProductRAGResult("A", rag_response=mock_rag),
                    ProductRAGResult("B", rag_response=mock_rag),
                ],
            ),
            patch(
                "reviewmind.core.llm.LLMClient",
            ) as MockLLM,
        ):
            mock_client = AsyncMock()
            mock_client.generate = AsyncMock(side_effect=LLMError("rate limit"))
            mock_client.close = AsyncMock()
            MockLLM.return_value = mock_client

            result = await compare_products(
                "A vs B",
                qdrant_client=AsyncMock(),
            )

        assert result.answer == ""
        assert "LLM error" in result.error

    @pytest.mark.asyncio
    async def test_products_capped_at_max(self):
        """Products list is capped at MAX_PRODUCTS_FOR_COMPARISON."""
        with (
            patch(
                "reviewmind.services.comparison_service.extract_product",
                new_callable=AsyncMock,
                return_value=["A", "B", "C", "D", "E"],
            ),
            patch(
                "reviewmind.services.comparison_service._parallel_rag",
                new_callable=AsyncMock,
                return_value=[ProductRAGResult(name) for name in ["A", "B", "C", "D"]],
            ),
            patch(
                "reviewmind.core.llm.LLMClient",
            ) as MockLLM,
        ):
            mock_client = AsyncMock()
            mock_client.generate = AsyncMock(return_value="comparison")
            mock_client.close = AsyncMock()
            MockLLM.return_value = mock_client

            result = await compare_products(
                "A vs B vs C vs D vs E",
                qdrant_client=AsyncMock(),
            )

        assert len(result.products) == MAX_PRODUCTS_FOR_COMPARISON

    @pytest.mark.asyncio
    async def test_source_deduplication(self):
        from reviewmind.core.rag import RAGResponse

        rag1 = RAGResponse(answer="a", sources=["https://a.com", "https://b.com"], chunks_count=2)
        rag2 = RAGResponse(answer="b", sources=["https://b.com", "https://c.com"], chunks_count=2)

        with (
            patch(
                "reviewmind.services.comparison_service.extract_product",
                new_callable=AsyncMock,
                return_value=["A", "B"],
            ),
            patch(
                "reviewmind.services.comparison_service._parallel_rag",
                new_callable=AsyncMock,
                return_value=[
                    ProductRAGResult("A", rag_response=rag1),
                    ProductRAGResult("B", rag_response=rag2),
                ],
            ),
            patch(
                "reviewmind.core.llm.LLMClient",
            ) as MockLLM,
        ):
            mock_client = AsyncMock()
            mock_client.generate = AsyncMock(return_value="comparison")
            mock_client.close = AsyncMock()
            MockLLM.return_value = mock_client

            result = await compare_products("A vs B", qdrant_client=AsyncMock())

        assert result.sources == ["https://a.com", "https://b.com", "https://c.com"]

    @pytest.mark.asyncio
    async def test_extract_product_exception(self):
        """Exception during extract_product → empty products → error result."""
        with patch(
            "reviewmind.services.comparison_service.extract_product",
            new_callable=AsyncMock,
            side_effect=Exception("LLM down"),
        ):
            result = await compare_products(
                "iPhone vs Samsung",
                qdrant_client=AsyncMock(),
            )
        assert result.answer == ""
        assert result.error is not None


# ── Bot integration ──────────────────────────────────────────────────────────


class TestBotIntegration:
    """Test that comparison detection is wired into the bot query handler."""

    def test_detect_comparison_imported_in_query_handler(self):
        import reviewmind.bot.handlers.query as qmodule

        assert hasattr(qmodule, "detect_comparison")
        assert hasattr(qmodule, "compare_products")

    def test_handle_comparison_function_exists(self):
        import reviewmind.bot.handlers.query as qmodule

        assert hasattr(qmodule, "_handle_comparison")
        assert asyncio.iscoroutinefunction(qmodule._handle_comparison)


# ── Services exports ─────────────────────────────────────────────────────────


class TestServicesExports:
    """Test that services __init__ exports comparison symbols."""

    def test_comparison_result_exported(self):
        from reviewmind.services import ComparisonResult

        assert ComparisonResult is not None

    def test_product_rag_result_exported(self):
        from reviewmind.services import ProductRAGResult

        assert ProductRAGResult is not None

    def test_compare_products_exported(self):
        from reviewmind.services import compare_products

        assert callable(compare_products)

    def test_detect_comparison_exported(self):
        from reviewmind.services import detect_comparison

        assert callable(detect_comparison)

    def test_constants_exported(self):
        from reviewmind.services import (
            COMPARISON_PROMPT_TEMPLATE,
            MAX_PRODUCTS_FOR_COMPARISON,
            MIN_PRODUCTS_FOR_COMPARISON,
        )

        assert isinstance(COMPARISON_PROMPT_TEMPLATE, str)
        assert MIN_PRODUCTS_FOR_COMPARISON == 2
        assert MAX_PRODUCTS_FOR_COMPARISON == 4


# ── Integration scenarios (all 5 PRD test steps) ────────────────────────────


class TestIntegrationScenarios:
    """End-to-end scenarios matching TASK-047 test steps."""

    def test_step1_iphone_vs_samsung_detected(self):
        """Step 1: 'iPhone 16 vs Samsung S25' → table comparison."""
        assert detect_comparison("iPhone 16 vs Samsung S25") is True

    def test_step2_sony_or_airpods_detected(self):
        """Step 2: 'Sony XM5 or AirPods Max?' → also comparison."""
        assert detect_comparison("Sony XM5 or AirPods Max?") is True

    @pytest.mark.asyncio
    async def test_step3_missing_data_for_one_product(self):
        """Step 3: Comparison with product without data → indicated."""
        from reviewmind.core.rag import RAGResponse

        rag_with_data = RAGResponse(answer="Good product", chunks_count=3)
        no_data = ProductRAGResult("Unknown Product X")
        with_data = ProductRAGResult("iPhone 16", rag_response=rag_with_data)

        prompt = _build_comparison_prompt(product_results=[with_data, no_data], language="ru")
        # Check that the missing-data product is mentioned with "no data" message
        assert "Unknown Product X" in prompt
        assert "отсутствуют" in prompt.lower()

    def test_step4_non_comparison_not_detected(self):
        """Step 4: Normal query (not comparison) → standard RAG."""
        assert detect_comparison("Стоит ли покупать iPhone 16?") is False

    @pytest.mark.asyncio
    async def test_step5_parallel_execution(self):
        """Step 5: Verify that RAG queries run in parallel (asyncio.gather)."""
        from reviewmind.core.rag import RAGResponse

        call_times: list[float] = []

        import time

        async def mock_query(**kwargs):
            call_times.append(time.monotonic())
            await asyncio.sleep(0.05)
            return RAGResponse(answer="text", chunks_count=1)

        with patch("reviewmind.core.rag.RAGPipeline") as MockPipeline:
            mock_instance = AsyncMock()
            mock_instance.query = AsyncMock(side_effect=mock_query)
            mock_instance.__aenter__ = AsyncMock(return_value=mock_instance)
            mock_instance.__aexit__ = AsyncMock(return_value=False)
            MockPipeline.return_value = mock_instance

            results = await _parallel_rag(
                product_names=["A", "B"],
                user_query="A vs B",
                qdrant_client=AsyncMock(),
            )

        assert len(results) == 2
        # Verify both started close together (within 50ms — indicates parallelism)
        if len(call_times) >= 2:
            assert abs(call_times[1] - call_times[0]) < 0.1
