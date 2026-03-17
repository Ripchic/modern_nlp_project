"""Tests for reviewmind.services.language — language detection service."""

from __future__ import annotations

from reviewmind.services.language import (
    FALLBACK_LANGUAGE,
    MIN_TEXT_LENGTH,
    SUPPORTED_LANGUAGES,
    detect_language,
)

# ── Constants ────────────────────────────────────────────────────────────────


class TestConstants:
    """Verify module-level constants."""

    def test_fallback_language_is_ru(self):
        assert FALLBACK_LANGUAGE == "ru"

    def test_min_text_length_positive(self):
        assert MIN_TEXT_LENGTH > 0

    def test_supported_languages_contains_ru_en(self):
        assert "ru" in SUPPORTED_LANGUAGES
        assert "en" in SUPPORTED_LANGUAGES

    def test_supported_languages_is_frozenset(self):
        assert isinstance(SUPPORTED_LANGUAGES, frozenset)


# ── Russian detection ────────────────────────────────────────────────────────


class TestDetectRussian:
    """Russian text must be detected as 'ru'."""

    def test_simple_russian_sentence(self):
        assert detect_language("Стоит ли покупать iPhone?") == "ru"

    def test_russian_product_query(self):
        assert detect_language("Обзор наушников Sony WH-1000XM5") == "ru"

    def test_russian_paragraph(self):
        text = (
            "Эти наушники обладают отличным шумоподавлением. "
            "Батарея держит около 30 часов. Звук сбалансированный."
        )
        assert detect_language(text) == "ru"

    def test_russian_with_numbers(self):
        assert detect_language("Цена 15000 рублей за модель 2026 года") == "ru"

    def test_russian_question(self):
        assert detect_language("Какой смартфон лучше в 2026 году?") == "ru"


# ── English detection ────────────────────────────────────────────────────────


class TestDetectEnglish:
    """English text must be detected as 'en'."""

    def test_simple_english_sentence(self):
        assert detect_language("Is Dyson V15 worth it?") == "en"

    def test_english_review_text(self):
        assert detect_language("The noise cancellation is excellent and battery lasts 30 hours") == "en"

    def test_english_product_comparison(self):
        assert detect_language("iPhone 16 vs Samsung Galaxy S25 comparison") == "en"

    def test_english_paragraph(self):
        text = (
            "These headphones provide exceptional sound quality. "
            "The build quality is premium with comfortable ear cushions. "
            "Battery life exceeds manufacturer claims."
        )
        assert detect_language(text) == "en"


# ── Fallback behaviour ──────────────────────────────────────────────────────


class TestFallback:
    """Edge cases that must return FALLBACK_LANGUAGE without exceptions."""

    def test_empty_string(self):
        assert detect_language("") == FALLBACK_LANGUAGE

    def test_whitespace_only(self):
        assert detect_language("   ") == FALLBACK_LANGUAGE

    def test_single_word_short(self):
        """Very short text (< MIN_TEXT_LENGTH) → fallback."""
        assert detect_language("OK") == FALLBACK_LANGUAGE

    def test_single_character(self):
        assert detect_language("a") == FALLBACK_LANGUAGE

    def test_none_like_empty(self):
        """Empty string should not crash."""
        result = detect_language("")
        assert isinstance(result, str)
        assert result == FALLBACK_LANGUAGE

    def test_only_numbers(self):
        """Pure numbers — langdetect may fail → fallback."""
        result = detect_language("12345")
        assert isinstance(result, str)

    def test_only_punctuation(self):
        """Punctuation-only should not crash."""
        result = detect_language("???!!!")
        assert isinstance(result, str)

    def test_tabs_and_newlines(self):
        assert detect_language("\t\n\r") == FALLBACK_LANGUAGE

    def test_spaces_with_short_text(self):
        """Whitespace-padded short text, after strip < MIN_TEXT_LENGTH."""
        assert detect_language("  ab  ") == FALLBACK_LANGUAGE


# ── Unicode support ──────────────────────────────────────────────────────────


class TestUnicodeSupport:
    """Verify that non-ASCII and mixed scripts don't crash."""

    def test_cyrillic_characters(self):
        result = detect_language("Привет мир, это тестовое сообщение")
        assert result == "ru"

    def test_chinese_characters(self):
        """Chinese text should be detected or fall back gracefully."""
        result = detect_language("这是一个测试消息，关于产品评论")
        assert isinstance(result, str)

    def test_mixed_latin_cyrillic(self):
        """Mixed text should not crash, result is language-dependent."""
        result = detect_language("iPhone 16 Pro отличный смартфон для повседневного использования")
        assert isinstance(result, str)

    def test_emoji_in_text(self):
        result = detect_language("Отличные наушники! 🎧👍 Рекомендую всем")
        assert isinstance(result, str)

    def test_japanese_text(self):
        result = detect_language("このヘッドフォンは素晴らしい音質を持っています")
        assert isinstance(result, str)


# ── Return type & contract ───────────────────────────────────────────────────


class TestReturnContract:
    """The function must always return a non-empty string."""

    def test_returns_string(self):
        assert isinstance(detect_language("test text for language detection"), str)

    def test_returns_non_empty(self):
        assert len(detect_language("")) > 0

    def test_fallback_always_in_supported(self):
        assert FALLBACK_LANGUAGE in SUPPORTED_LANGUAGES

    def test_detected_language_in_supported_or_fallback(self):
        """Any detected result must be in SUPPORTED_LANGUAGES."""
        result = detect_language("This is a test of the language detection system")
        assert result in SUPPORTED_LANGUAGES


# ── Integration with prompts ─────────────────────────────────────────────────


class TestPromptIntegration:
    """Verify language integrates with the prompt system."""

    def test_build_rag_prompt_accepts_language(self):
        from reviewmind.core.prompts import build_rag_system_prompt

        result = build_rag_system_prompt([], language="en")
        assert "en" in result

    def test_build_rag_prompt_default_language(self):
        from reviewmind.core.prompts import DEFAULT_PROMPT_LANGUAGE, build_rag_system_prompt

        result = build_rag_system_prompt([])
        assert DEFAULT_PROMPT_LANGUAGE in result

    def test_build_rag_prompt_with_detected_language(self):
        from reviewmind.core.prompts import build_rag_system_prompt

        lang = detect_language("Is this product worth buying?")
        result = build_rag_system_prompt([], language=lang)
        assert lang in result

    def test_detected_language_in_rag_prompt_template(self):
        """The language slot must appear in the rendered prompt."""
        from reviewmind.core.prompts import build_rag_system_prompt

        result = build_rag_system_prompt([], language="ru")
        assert "определённый язык: ru" in result

    def test_english_language_in_rag_prompt(self):
        from reviewmind.core.prompts import build_rag_system_prompt

        result = build_rag_system_prompt([], language="en")
        assert "определённый язык: en" in result


# ── Exports ──────────────────────────────────────────────────────────────────


class TestExports:
    """Verify public symbols are importable."""

    def test_detect_language_importable(self):
        from reviewmind.services.language import detect_language  # noqa: F811

        assert callable(detect_language)

    def test_constants_importable(self):
        from reviewmind.services.language import (  # noqa: F811
            FALLBACK_LANGUAGE,
            MIN_TEXT_LENGTH,
            SUPPORTED_LANGUAGES,
        )

        assert FALLBACK_LANGUAGE
        assert MIN_TEXT_LENGTH
        assert SUPPORTED_LANGUAGES

    def test_prompt_language_constant_importable(self):
        from reviewmind.core.prompts import DEFAULT_PROMPT_LANGUAGE

        assert DEFAULT_PROMPT_LANGUAGE == "ru"


# ── Integration scenarios ────────────────────────────────────────────────────


class TestIntegrationScenarios:
    """End-to-end scenarios matching acceptance criteria."""

    def test_prd_step1_russian_detection(self):
        """Шаг 1: detect_language('Стоит ли покупать iPhone?') → 'ru'"""
        assert detect_language("Стоит ли покупать iPhone?") == "ru"

    def test_prd_step2_english_detection(self):
        """Шаг 2: detect_language('Is Dyson V15 worth it?') → 'en'"""
        assert detect_language("Is Dyson V15 worth it?") == "en"

    def test_prd_step3_short_text_fallback(self):
        """Шаг 3: detect_language('OK') → fallback 'ru'"""
        assert detect_language("OK") == FALLBACK_LANGUAGE

    def test_prd_step4_empty_text_no_exception(self):
        """Шаг 4: detect_language('') → fallback 'ru' без exception"""
        result = detect_language("")
        assert result == FALLBACK_LANGUAGE

    def test_detect_then_prompt_flow(self):
        """Full flow: detect → build prompt with detected language."""
        from reviewmind.core.prompts import build_rag_system_prompt

        query = "Стоит ли покупать Sony WH-1000XM5?"
        lang = detect_language(query)
        prompt = build_rag_system_prompt([], language=lang)
        assert lang in prompt
        assert "ПРАВИЛА" in prompt
