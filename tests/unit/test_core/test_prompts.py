"""Unit tests for reviewmind.core.prompts — system prompts and context builders."""

from __future__ import annotations

from reviewmind.core.prompts import (
    CURATED_MARKER,
    FALLBACK_SYSTEM_PROMPT,
    NO_CONTEXT_TEXT,
    NO_HISTORY_TEXT,
    RAG_MAX_TOKENS,
    RAG_TEMPERATURE,
    RAG_TOP_P,
    SPONSORED_MARKER,
    SYSTEM_PROMPT_TEMPLATE,
    ChunkContext,
    build_rag_system_prompt,
    format_chat_history,
    format_chunks_for_context,
)

# ── ChunkContext dataclass ────────────────────────────────────


class TestChunkContext:
    """Tests for the ChunkContext dataclass."""

    def test_required_fields(self):
        chunk = ChunkContext(text="Some review text", source_url="https://example.com")
        assert chunk.text == "Some review text"
        assert chunk.source_url == "https://example.com"

    def test_default_source_type(self):
        chunk = ChunkContext(text="text", source_url="https://example.com")
        assert chunk.source_type == "web"

    def test_default_flags(self):
        chunk = ChunkContext(text="text", source_url="https://example.com")
        assert chunk.is_sponsored is False
        assert chunk.is_curated is False

    def test_default_score(self):
        chunk = ChunkContext(text="text", source_url="https://example.com")
        assert chunk.score == 0.0

    def test_default_author_none(self):
        chunk = ChunkContext(text="text", source_url="https://example.com")
        assert chunk.author is None

    def test_default_extra_metadata_empty(self):
        chunk = ChunkContext(text="text", source_url="https://example.com")
        assert chunk.extra_metadata == {}

    def test_extra_metadata_not_shared(self):
        """Mutable default should be independent per instance."""
        a = ChunkContext(text="a", source_url="url")
        b = ChunkContext(text="b", source_url="url")
        a.extra_metadata["key"] = "val"
        assert "key" not in b.extra_metadata

    def test_all_fields_explicit(self):
        chunk = ChunkContext(
            text="Great headphones",
            source_url="https://youtube.com/watch?v=abc",
            source_type="youtube",
            is_sponsored=True,
            is_curated=False,
            score=0.92,
            author="TechChannel",
            extra_metadata={"product": "Sony WH-1000XM5"},
        )
        assert chunk.is_sponsored is True
        assert chunk.author == "TechChannel"
        assert chunk.score == 0.92


# ── format_chunks_for_context ─────────────────────────────────


class TestFormatChunksForContext:
    """Tests for format_chunks_for_context()."""

    def test_empty_list_returns_no_context_text(self):
        result = format_chunks_for_context([])
        assert result == NO_CONTEXT_TEXT

    def test_single_chunk_contains_number(self):
        chunk = ChunkContext(text="Good ANC", source_url="https://example.com")
        result = format_chunks_for_context([chunk])
        assert "[1]" in result

    def test_single_chunk_contains_source_type_uppercase(self):
        chunk = ChunkContext(text="text", source_url="https://r.com", source_type="reddit")
        result = format_chunks_for_context([chunk])
        assert "REDDIT" in result

    def test_single_chunk_contains_url(self):
        chunk = ChunkContext(text="text", source_url="https://rtings.com/headphones")
        result = format_chunks_for_context([chunk])
        assert "https://rtings.com/headphones" in result

    def test_single_chunk_contains_text(self):
        chunk = ChunkContext(text="Excellent noise cancellation", source_url="https://ex.com")
        result = format_chunks_for_context([chunk])
        assert "Excellent noise cancellation" in result

    def test_sponsored_chunk_has_sponsored_marker(self):
        chunk = ChunkContext(text="text", source_url="https://yt.com", is_sponsored=True)
        result = format_chunks_for_context([chunk])
        assert SPONSORED_MARKER in result

    def test_non_sponsored_chunk_no_sponsored_marker(self):
        chunk = ChunkContext(text="text", source_url="https://yt.com", is_sponsored=False)
        result = format_chunks_for_context([chunk])
        assert SPONSORED_MARKER not in result

    def test_curated_chunk_has_curated_marker(self):
        chunk = ChunkContext(text="text", source_url="https://wirecutter.com", is_curated=True)
        result = format_chunks_for_context([chunk])
        assert CURATED_MARKER in result

    def test_non_curated_chunk_no_curated_marker(self):
        chunk = ChunkContext(text="text", source_url="https://yt.com", is_curated=False)
        result = format_chunks_for_context([chunk])
        assert CURATED_MARKER not in result

    def test_author_included_in_source_label(self):
        chunk = ChunkContext(
            text="text",
            source_url="https://youtube.com",
            source_type="youtube",
            author="MKBHD",
        )
        result = format_chunks_for_context([chunk])
        assert "MKBHD" in result

    def test_no_author_uses_type_only(self):
        chunk = ChunkContext(text="text", source_url="https://reddit.com", source_type="reddit")
        result = format_chunks_for_context([chunk])
        assert "REDDIT" in result
        # Should not have parentheses
        assert "(" not in result

    def test_multiple_chunks_all_numbered(self):
        chunks = [ChunkContext(text=f"chunk {i}", source_url=f"https://src{i}.com") for i in range(1, 4)]
        result = format_chunks_for_context(chunks)
        assert "[1]" in result
        assert "[2]" in result
        assert "[3]" in result

    def test_multiple_chunks_all_text_included(self):
        chunks = [
            ChunkContext(text="ABC text", source_url="https://a.com"),
            ChunkContext(text="XYZ text", source_url="https://b.com"),
        ]
        result = format_chunks_for_context(chunks)
        assert "ABC text" in result
        assert "XYZ text" in result

    def test_sponsored_and_curated_together(self):
        """A curated + sponsored chunk (edge case) should show both markers."""
        chunk = ChunkContext(
            text="text",
            source_url="https://ex.com",
            is_sponsored=True,
            is_curated=True,
        )
        result = format_chunks_for_context([chunk])
        assert SPONSORED_MARKER in result
        assert CURATED_MARKER in result

    def test_chunk_text_is_stripped(self):
        chunk = ChunkContext(text="  padded text  ", source_url="https://ex.com")
        result = format_chunks_for_context([chunk])
        # Text should appear stripped in the output
        assert "padded text" in result
        # The chunk line itself should not start with spaces
        assert "\n  padded text" not in result


# ── format_chat_history ───────────────────────────────────────


class TestFormatChatHistory:
    """Tests for format_chat_history()."""

    def test_empty_list_returns_no_history(self):
        result = format_chat_history([])
        assert result == NO_HISTORY_TEXT

    def test_user_role_translated(self):
        history = [{"role": "user", "content": "What about battery life?"}]
        result = format_chat_history(history)
        assert "Пользователь" in result
        assert "What about battery life?" in result

    def test_assistant_role_translated(self):
        history = [{"role": "assistant", "content": "Battery lasts 30 hours."}]
        result = format_chat_history(history)
        assert "ReviewMind" in result
        assert "Battery lasts 30 hours." in result

    def test_system_role_translated(self):
        history = [{"role": "system", "content": "Internal note"}]
        result = format_chat_history(history)
        assert "Система" in result

    def test_unknown_role_capitalized(self):
        history = [{"role": "tool", "content": "Result"}]
        result = format_chat_history(history)
        assert "Tool" in result

    def test_multiple_turns_all_present(self):
        history = [
            {"role": "user", "content": "First question"},
            {"role": "assistant", "content": "First answer"},
            {"role": "user", "content": "Follow-up"},
        ]
        result = format_chat_history(history)
        assert "First question" in result
        assert "First answer" in result
        assert "Follow-up" in result

    def test_order_preserved(self):
        history = [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
        ]
        result = format_chat_history(history)
        q_idx = result.index("q1")
        a_idx = result.index("a1")
        assert q_idx < a_idx


# ── build_rag_system_prompt ───────────────────────────────────


class TestBuildRagSystemPrompt:
    """Tests for build_rag_system_prompt()."""

    def test_contains_rules_section(self):
        result = build_rag_system_prompt([])
        assert "ПРАВИЛА" in result

    def test_contains_context_header(self):
        chunks = [ChunkContext(text="Test text", source_url="https://ex.com")]
        result = build_rag_system_prompt(chunks)
        assert "КОНТЕКСТ" in result

    def test_contains_history_header(self):
        chunks = [ChunkContext(text="Test text", source_url="https://ex.com")]
        result = build_rag_system_prompt(chunks)
        assert "ИСТОРИЯ ДИАЛОГА" in result

    def test_no_chunks_returns_fallback_prompt(self):
        result = build_rag_system_prompt([])
        assert result == FALLBACK_SYSTEM_PROMPT

    def test_no_chunks_fallback_has_rules(self):
        result = build_rag_system_prompt([])
        assert "ПРАВИЛА" in result

    def test_no_history_injects_no_history_text(self):
        chunks = [ChunkContext(text="Test text", source_url="https://ex.com")]
        result = build_rag_system_prompt(chunks)
        assert NO_HISTORY_TEXT in result

    def test_chunks_text_appears_in_prompt(self):
        chunks = [ChunkContext(text="Great sound quality", source_url="https://ex.com")]
        result = build_rag_system_prompt(chunks)
        assert "Great sound quality" in result

    def test_history_appears_in_prompt(self):
        chunks = [ChunkContext(text="Product text", source_url="https://ex.com")]
        history = [{"role": "user", "content": "Tell me about ANC"}]
        result = build_rag_system_prompt(chunks, chat_history=history)
        assert "Tell me about ANC" in result

    def test_sponsored_source_marked_in_prompt(self):
        chunks = [
            ChunkContext(
                text="Sponsored review",
                source_url="https://yt.com",
                is_sponsored=True,
            )
        ]
        result = build_rag_system_prompt(chunks)
        assert SPONSORED_MARKER in result

    def test_curated_source_marked_in_prompt(self):
        chunks = [
            ChunkContext(
                text="Editorial review",
                source_url="https://wirecutter.com",
                is_curated=True,
            )
        ]
        result = build_rag_system_prompt(chunks)
        assert CURATED_MARKER in result

    def test_structured_format_rule_present(self):
        """PRD rule 4: Структурируй ответ must be in the template."""
        chunks = [ChunkContext(text="Test text", source_url="https://ex.com")]
        result = build_rag_system_prompt(chunks)
        assert "✅ Плюсы" in result
        assert "❌ Минусы" in result
        assert "⚖️ Спорные моменты" in result
        assert "🏆 Вывод" in result

    def test_language_rule_present(self):
        """PRD rule 6: Отвечай на языке запроса пользователя."""
        result = build_rag_system_prompt([])
        assert "языке запроса пользователя" in result

    def test_none_history_treated_as_empty(self):
        result_none = build_rag_system_prompt([], chat_history=None)
        result_empty = build_rag_system_prompt([], chat_history=[])
        assert result_none == result_empty

    def test_generation_params_mentioned_in_prompt(self):
        """PRD specifies temperature=0.3, max_tokens=1000, top_p=0.9 in prompt."""
        chunks = [ChunkContext(text="Test text", source_url="https://ex.com")]
        result = build_rag_system_prompt(chunks)
        assert "temperature=0.3" in result
        assert "max_tokens=1000" in result
        assert "top_p=0.9" in result


# ── Prompt constants & RAG parameters ────────────────────────


class TestPromptConstants:
    def test_rag_temperature(self):
        assert RAG_TEMPERATURE == 0.3

    def test_rag_max_tokens(self):
        assert RAG_MAX_TOKENS == 1000

    def test_rag_top_p(self):
        assert RAG_TOP_P == 0.9

    def test_system_prompt_template_has_slots(self):
        """Template must have both required format slots."""
        assert "{retrieved_chunks}" in SYSTEM_PROMPT_TEMPLATE
        assert "{chat_history}" in SYSTEM_PROMPT_TEMPLATE

    def test_fallback_system_prompt_mentions_reviewmind(self):
        assert "ReviewMind" in FALLBACK_SYSTEM_PROMPT

    def test_fallback_mentions_language_rule(self):
        assert "языке запроса пользователя" in FALLBACK_SYSTEM_PROMPT

    def test_sponsored_marker_value(self):
        assert SPONSORED_MARKER == "[sponsored]"

    def test_curated_marker_is_book_emoji(self):
        assert CURATED_MARKER == "📚"
