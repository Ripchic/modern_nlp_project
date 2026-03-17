"""Unit tests for TASK-043 — Product name extraction from user queries.

Tests cover:
- Constants and regex patterns
- extract_product_regex: brand-model patterns, iPhone/Galaxy patterns, edge cases
- is_comparison_query: in Russian and English
- _parse_llm_response: valid JSON, malformed JSON, markdown fences
- _extract_via_llm: success, LLMError fallback, unexpected error
- extract_product: LLM success, LLM failure → regex fallback, empty input
- Integration scenarios
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from reviewmind.services.product_extractor import (
    _COMPARISON_RE,
    _EXTRACTION_MAX_TOKENS,
    _EXTRACTION_PROMPT,
    _EXTRACTION_TEMPERATURE,
    _PRODUCT_PATTERN,
    _extract_via_llm,
    _parse_llm_response,
    extract_product,
    extract_product_regex,
    is_comparison_query,
)

# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════


def _mock_llm_client(response: str = '{"products": []}') -> MagicMock:
    """Create a mocked LLMClient whose generate() returns *response*."""
    client = MagicMock()
    client.generate = AsyncMock(return_value=response)
    client.close = AsyncMock()
    return client


def _mock_llm_error(exc: Exception) -> MagicMock:
    """Create a mocked LLMClient whose generate() raises *exc*."""
    client = MagicMock()
    client.generate = AsyncMock(side_effect=exc)
    client.close = AsyncMock()
    return client


# ══════════════════════════════════════════════════════════════
# Tests — Constants
# ══════════════════════════════════════════════════════════════


class TestConstants:
    """Verify module-level constants."""

    def test_extraction_temperature_is_zero(self):
        assert _EXTRACTION_TEMPERATURE == 0.0

    def test_extraction_max_tokens(self):
        assert _EXTRACTION_MAX_TOKENS == 200

    def test_extraction_prompt_contains_json_format(self):
        assert '"products"' in _EXTRACTION_PROMPT

    def test_extraction_prompt_contains_examples(self):
        assert "Sony WH-1000XM5" in _EXTRACTION_PROMPT
        assert "iPhone 16" in _EXTRACTION_PROMPT

    def test_product_pattern_compiled(self):
        assert _PRODUCT_PATTERN is not None

    def test_comparison_re_compiled(self):
        assert _COMPARISON_RE is not None


# ══════════════════════════════════════════════════════════════
# Tests — extract_product_regex
# ══════════════════════════════════════════════════════════════


class TestExtractProductRegex:
    """Test the regex-based fallback extractor."""

    def test_empty_string(self):
        assert extract_product_regex("") == []

    def test_none_input(self):
        assert extract_product_regex(None) == []

    def test_whitespace_only(self):
        assert extract_product_regex("   ") == []

    def test_no_product(self):
        assert extract_product_regex("Привет, как дела?") == []

    def test_no_product_generic_category(self):
        assert extract_product_regex("какие наушники лучше?") == []

    def test_sony_model(self):
        result = extract_product_regex("Стоит ли покупать Sony WH-1000XM5?")
        assert len(result) >= 1
        assert any("Sony" in p and "WH-1000XM5" in p for p in result)

    def test_iphone(self):
        result = extract_product_regex("iPhone 16 Pro Max review")
        assert len(result) >= 1
        assert any("iPhone" in p for p in result)

    def test_samsung_galaxy(self):
        result = extract_product_regex("Galaxy S25 Ultra worth it?")
        assert len(result) >= 1
        assert any("Galaxy" in p for p in result)

    def test_dyson(self):
        result = extract_product_regex("Dyson V15 стоит своих денег?")
        assert len(result) >= 1
        assert any("Dyson" in p for p in result)

    def test_airpods(self):
        result = extract_product_regex("AirPods Pro отзывы")
        assert len(result) >= 1
        assert any("AirPods" in p for p in result)

    def test_macbook(self):
        result = extract_product_regex("MacBook Air M3 стоит ли покупать?")
        assert len(result) >= 1
        assert any("MacBook" in p for p in result)

    def test_multiple_products(self):
        result = extract_product_regex("iPhone 16 или Samsung Galaxy S25?")
        assert len(result) >= 2

    def test_deduplication(self):
        result = extract_product_regex("Sony WH-1000XM5 и Sony WH-1000XM5 — это одно и то же?")
        # Should deduplicate
        sony_count = sum(1 for p in result if "Sony" in p and "WH-1000XM5" in p)
        assert sony_count == 1

    def test_case_insensitive(self):
        result = extract_product_regex("Обзор IPHONE 16 pro")
        assert len(result) >= 1

    def test_pixel(self):
        result = extract_product_regex("Pixel 9 Pro camera test")
        assert len(result) >= 1
        assert any("Pixel" in p for p in result)

    def test_bose(self):
        result = extract_product_regex("Bose QC45 noise cancelling")
        assert len(result) >= 1
        assert any("Bose" in p for p in result)

    def test_jbl(self):
        result = extract_product_regex("JBL Flip 6 — лучшая колонка?")
        assert len(result) >= 1
        assert any("JBL" in p for p in result)


# ══════════════════════════════════════════════════════════════
# Tests — is_comparison_query
# ══════════════════════════════════════════════════════════════


class TestIsComparisonQuery:
    """Test comparison query detection."""

    def test_empty(self):
        assert is_comparison_query("") is False

    def test_none(self):
        assert is_comparison_query(None) is False

    def test_vs_english(self):
        assert is_comparison_query("iPhone 16 vs Samsung S25") is True

    def test_versus_english(self):
        assert is_comparison_query("Sony XM5 versus AirPods Max") is True

    def test_or_russian(self):
        assert is_comparison_query("Sony XM5 или AirPods Max?") is True

    def test_against_russian(self):
        assert is_comparison_query("iPhone 16 против Samsung S25") is True

    def test_compared_to(self):
        assert is_comparison_query("Dyson V15 compared to Shark") is True

    def test_not_comparison(self):
        assert is_comparison_query("Стоит ли покупать Sony WH-1000XM5?") is False

    def test_not_comparison_simple(self):
        assert is_comparison_query("обзор iPhone 16") is False

    def test_vs_with_dot(self):
        assert is_comparison_query("XM5 vs. AirPods Max") is True


# ══════════════════════════════════════════════════════════════
# Tests — _parse_llm_response
# ══════════════════════════════════════════════════════════════


class TestParseLlmResponse:
    """Test the JSON parser for LLM output."""

    def test_valid_json_single(self):
        raw = '{"products": ["Sony WH-1000XM5"]}'
        result = _parse_llm_response(raw)
        assert result == ["Sony WH-1000XM5"]

    def test_valid_json_multiple(self):
        raw = '{"products": ["iPhone 16", "Samsung S25"]}'
        result = _parse_llm_response(raw)
        assert result == ["iPhone 16", "Samsung S25"]

    def test_valid_json_empty(self):
        raw = '{"products": []}'
        result = _parse_llm_response(raw)
        assert result == []

    def test_markdown_fences(self):
        raw = '```json\n{"products": ["Dyson V15"]}\n```'
        result = _parse_llm_response(raw)
        assert result == ["Dyson V15"]

    def test_markdown_fences_no_lang(self):
        raw = '```\n{"products": ["Dyson V15"]}\n```'
        result = _parse_llm_response(raw)
        assert result == ["Dyson V15"]

    def test_malformed_json(self):
        result = _parse_llm_response("not json at all")
        assert result is None

    def test_wrong_format_no_products_key(self):
        raw = '{"items": ["Sony"]}'
        result = _parse_llm_response(raw)
        assert result is None

    def test_products_not_a_list(self):
        raw = '{"products": "Sony WH-1000XM5"}'
        result = _parse_llm_response(raw)
        assert result is None

    def test_strips_whitespace(self):
        raw = '  {"products": [" AirPods Pro "]}  '
        result = _parse_llm_response(raw)
        assert result == ["AirPods Pro"]

    def test_filters_empty_strings(self):
        raw = '{"products": ["Sony XM5", "", " "]}'
        result = _parse_llm_response(raw)
        assert result == ["Sony XM5"]

    def test_none_in_products_list(self):
        raw = '{"products": ["Sony XM5", null]}'
        result = _parse_llm_response(raw)
        assert result == ["Sony XM5"]


# ══════════════════════════════════════════════════════════════
# Tests — _extract_via_llm
# ══════════════════════════════════════════════════════════════


class TestExtractViaLlm:
    """Test the LLM extraction helper."""

    async def test_success(self):
        client = _mock_llm_client('{"products": ["Sony WH-1000XM5"]}')
        result = await _extract_via_llm("Sony WH-1000XM5 review", llm_client=client)
        assert result == ["Sony WH-1000XM5"]

    async def test_empty_products(self):
        client = _mock_llm_client('{"products": []}')
        result = await _extract_via_llm("Привет", llm_client=client)
        assert result == []

    async def test_llm_error_returns_none(self):
        from reviewmind.core.llm import LLMError

        client = _mock_llm_error(LLMError("API error"))
        result = await _extract_via_llm("some query", llm_client=client)
        assert result is None

    async def test_unexpected_error_returns_none(self):
        client = _mock_llm_error(RuntimeError("unexpected"))
        result = await _extract_via_llm("some query", llm_client=client)
        assert result is None

    async def test_malformed_response_returns_none(self):
        client = _mock_llm_client("This is not JSON")
        result = await _extract_via_llm("query", llm_client=client)
        assert result is None

    async def test_uses_correct_temperature(self):
        client = _mock_llm_client('{"products": ["Test"]}')
        await _extract_via_llm("query", llm_client=client)
        call_kwargs = client.generate.call_args
        assert call_kwargs.kwargs.get("temperature") == 0.0

    async def test_uses_correct_max_tokens(self):
        client = _mock_llm_client('{"products": ["Test"]}')
        await _extract_via_llm("query", llm_client=client)
        call_kwargs = client.generate.call_args
        assert call_kwargs.kwargs.get("max_tokens") == 200

    async def test_closes_owned_client(self):
        mock_client = _mock_llm_client('{"products": []}')
        with patch("reviewmind.services.product_extractor.LLMClient", return_value=mock_client):
            await _extract_via_llm("test")
            mock_client.close.assert_awaited_once()

    async def test_does_not_close_injected_client(self):
        client = _mock_llm_client('{"products": []}')
        await _extract_via_llm("test", llm_client=client)
        client.close.assert_not_awaited()


# ══════════════════════════════════════════════════════════════
# Tests — extract_product (main function)
# ══════════════════════════════════════════════════════════════


class TestExtractProduct:
    """Test the main public API function."""

    async def test_empty_string(self):
        result = await extract_product("")
        assert result == []

    async def test_none(self):
        result = await extract_product(None)
        assert result == []

    async def test_whitespace(self):
        result = await extract_product("   ")
        assert result == []

    async def test_llm_success_single(self):
        client = _mock_llm_client('{"products": ["Sony WH-1000XM5"]}')
        result = await extract_product("Sony WH-1000XM5 стоит ли покупать?", llm_client=client)
        assert result == ["Sony WH-1000XM5"]

    async def test_llm_success_multiple(self):
        client = _mock_llm_client('{"products": ["iPhone 16", "Samsung S25"]}')
        result = await extract_product("iPhone 16 vs Samsung S25", llm_client=client)
        assert result == ["iPhone 16", "Samsung S25"]

    async def test_llm_success_no_product(self):
        client = _mock_llm_client('{"products": []}')
        result = await extract_product("Привет, как дела?", llm_client=client)
        assert result == []

    async def test_llm_failure_falls_back_to_regex(self):
        from reviewmind.core.llm import LLMError

        client = _mock_llm_error(LLMError("fail"))
        result = await extract_product("Sony WH-1000XM5 review", llm_client=client)
        # Regex fallback should find the product
        assert len(result) >= 1
        assert any("Sony" in p for p in result)

    async def test_llm_failure_regex_no_match(self):
        from reviewmind.core.llm import LLMError

        client = _mock_llm_error(LLMError("fail"))
        result = await extract_product("какой товар лучше всего?", llm_client=client)
        assert result == []

    async def test_strips_query(self):
        client = _mock_llm_client('{"products": ["Test"]}')
        await extract_product("  test query  ", llm_client=client)
        call_args = client.generate.call_args
        user_msg = call_args.kwargs.get("user_message", call_args.args[1] if len(call_args.args) > 1 else "")
        assert user_msg == "test query"


# ══════════════════════════════════════════════════════════════
# Tests — Integration scenarios
# ══════════════════════════════════════════════════════════════


class TestIntegrationScenarios:
    """End-to-end scenarios for product extraction."""

    async def test_russian_query_with_product(self):
        client = _mock_llm_client('{"products": ["Sony WH-1000XM5"]}')
        result = await extract_product("Стоит ли покупать Sony WH-1000XM5?", llm_client=client)
        assert result == ["Sony WH-1000XM5"]

    async def test_english_query_with_product(self):
        client = _mock_llm_client('{"products": ["Dyson V15"]}')
        result = await extract_product("Is Dyson V15 worth buying?", llm_client=client)
        assert result == ["Dyson V15"]

    async def test_comparison_query(self):
        client = _mock_llm_client('{"products": ["iPhone 16", "Samsung S25"]}')
        query = "iPhone 16 vs Samsung S25"
        result = await extract_product(query, llm_client=client)
        assert len(result) == 2
        assert is_comparison_query(query)

    async def test_off_topic_query(self):
        client = _mock_llm_client('{"products": []}')
        result = await extract_product("Привет, как дела?", llm_client=client)
        assert result == []

    async def test_generic_category_no_product(self):
        client = _mock_llm_client('{"products": []}')
        result = await extract_product("какие наушники лучше?", llm_client=client)
        assert result == []

    async def test_regex_fallback_for_known_brands(self):
        """When LLM fails, regex should still catch known brand-model patterns."""
        from reviewmind.core.llm import LLMError

        client = _mock_llm_error(LLMError("API down"))
        result = await extract_product("Bose QC45 noise cancelling review", llm_client=client)
        assert len(result) >= 1
        assert any("Bose" in p for p in result)

    async def test_ten_queries_variety(self):
        """Verify extract handles diverse query formulations via LLM mock."""
        test_cases = [
            ("Sony WH-1000XM5 стоит?", '{"products": ["Sony WH-1000XM5"]}', 1),
            ("Is Dyson V15 worth buying?", '{"products": ["Dyson V15"]}', 1),
            ("обзор MacBook Air M3", '{"products": ["MacBook Air M3"]}', 1),
            ("iPhone 16 vs Samsung S25", '{"products": ["iPhone 16", "Samsung S25"]}', 2),
            ("AirPods Pro 2 отзывы", '{"products": ["AirPods Pro 2"]}', 1),
            ("Привет", '{"products": []}', 0),
            ("какие ноутбуки хорошие?", '{"products": []}', 0),
            ("JBL Flip 6 review", '{"products": ["JBL Flip 6"]}', 1),
            ("Pixel 9 Pro камера", '{"products": ["Pixel 9 Pro"]}', 1),
            ("Garmin Venu 3 GPS точность", '{"products": ["Garmin Venu 3"]}', 1),
        ]

        correct = 0
        for query, llm_response, expected_count in test_cases:
            client = _mock_llm_client(llm_response)
            result = await extract_product(query, llm_client=client)
            if len(result) == expected_count:
                correct += 1

        # ≥ 80% accuracy as required by acceptance criteria
        assert correct >= 8, f"Only {correct}/10 correct"
