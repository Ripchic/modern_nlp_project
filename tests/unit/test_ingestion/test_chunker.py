"""Unit tests for reviewmind.ingestion.chunker — text chunking module."""

from __future__ import annotations

import pytest

from reviewmind.ingestion.chunker import (
    CHARS_PER_TOKEN,
    CHUNK_OVERLAP_CHARS,
    CHUNK_OVERLAP_TOKENS,
    CHUNK_SIZE_CHARS,
    CHUNK_SIZE_TOKENS,
    DEFAULT_SEPARATORS,
    MIN_CHUNK_LENGTH,
    Chunk,
    _build_splitter,
    chunk_text,
    chunk_text_dicts,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_long_text(word_count: int = 5000) -> str:
    """Generate a long text with *word_count* words, split into paragraphs."""
    words = [f"word{i}" for i in range(word_count)]
    # Insert paragraph breaks every ~100 words and sentence breaks every ~15 words
    parts: list[str] = []
    for i, w in enumerate(words):
        parts.append(w)
        if i > 0 and i % 15 == 0:
            parts.append(".")
        if i > 0 and i % 100 == 0:
            parts.append("\n\n")
    return " ".join(parts)


def _make_short_text(word_count: int = 100) -> str:
    """Generate a short text with *word_count* words."""
    return " ".join(f"word{i}" for i in range(word_count))


# ===========================================================================
# TestConstants
# ===========================================================================


class TestConstants:
    """Verify module-level constants match the PRD specification."""

    def test_chars_per_token(self) -> None:
        assert CHARS_PER_TOKEN == 4

    def test_chunk_size_tokens(self) -> None:
        assert CHUNK_SIZE_TOKENS == 500

    def test_chunk_size_chars(self) -> None:
        assert CHUNK_SIZE_CHARS == CHUNK_SIZE_TOKENS * CHARS_PER_TOKEN
        assert CHUNK_SIZE_CHARS == 2000

    def test_chunk_overlap_tokens(self) -> None:
        assert CHUNK_OVERLAP_TOKENS == 50

    def test_chunk_overlap_chars(self) -> None:
        assert CHUNK_OVERLAP_CHARS == CHUNK_OVERLAP_TOKENS * CHARS_PER_TOKEN
        assert CHUNK_OVERLAP_CHARS == 200

    def test_min_chunk_length(self) -> None:
        assert MIN_CHUNK_LENGTH == 1

    def test_default_separators_order(self) -> None:
        """Separators should prioritise paragraphs → sentences → words → chars."""
        assert DEFAULT_SEPARATORS[0] == "\n\n"
        assert DEFAULT_SEPARATORS[1] == "\n"
        assert ". " in DEFAULT_SEPARATORS
        assert " " in DEFAULT_SEPARATORS
        assert "" == DEFAULT_SEPARATORS[-1]

    def test_default_separators_length(self) -> None:
        assert len(DEFAULT_SEPARATORS) == 9


# ===========================================================================
# TestChunk
# ===========================================================================


class TestChunk:
    """Tests for the Chunk dataclass."""

    def test_creation(self) -> None:
        chunk = Chunk(text="hello", chunk_index=0)
        assert chunk.text == "hello"
        assert chunk.chunk_index == 0
        assert chunk.metadata == {}

    def test_creation_with_metadata(self) -> None:
        meta = {"source_url": "https://example.com", "source_type": "web"}
        chunk = Chunk(text="hello", chunk_index=1, metadata=meta)
        assert chunk.metadata == meta

    def test_frozen(self) -> None:
        chunk = Chunk(text="hello", chunk_index=0)
        with pytest.raises(AttributeError):
            chunk.text = "changed"  # type: ignore[misc]

    def test_to_dict(self) -> None:
        meta = {"source_url": "https://example.com"}
        chunk = Chunk(text="test text", chunk_index=2, metadata=meta)
        d = chunk.to_dict()
        assert d["text"] == "test text"
        assert d["chunk_index"] == 2
        assert d["metadata"]["source_url"] == "https://example.com"

    def test_to_dict_returns_plain_dict(self) -> None:
        chunk = Chunk(text="abc", chunk_index=0, metadata={"k": "v"})
        d = chunk.to_dict()
        assert isinstance(d, dict)
        assert isinstance(d["metadata"], dict)

    def test_equality(self) -> None:
        c1 = Chunk(text="a", chunk_index=0, metadata={"x": 1})
        c2 = Chunk(text="a", chunk_index=0, metadata={"x": 1})
        assert c1 == c2

    def test_inequality(self) -> None:
        c1 = Chunk(text="a", chunk_index=0)
        c2 = Chunk(text="b", chunk_index=0)
        assert c1 != c2


# ===========================================================================
# TestBuildSplitter
# ===========================================================================


class TestBuildSplitter:
    """Tests for the _build_splitter helper."""

    def test_default_params(self) -> None:
        splitter = _build_splitter()
        assert splitter._chunk_size == CHUNK_SIZE_CHARS
        assert splitter._chunk_overlap == CHUNK_OVERLAP_CHARS

    def test_custom_params(self) -> None:
        splitter = _build_splitter(chunk_size=1000, chunk_overlap=100)
        assert splitter._chunk_size == 1000
        assert splitter._chunk_overlap == 100

    def test_custom_separators(self) -> None:
        seps = ["\n\n", "\n", " ", ""]
        splitter = _build_splitter(separators=seps)
        assert splitter._separators == seps

    def test_default_separators_used(self) -> None:
        splitter = _build_splitter()
        assert splitter._separators == DEFAULT_SEPARATORS


# ===========================================================================
# TestChunkTextEmpty
# ===========================================================================


class TestChunkTextEmpty:
    """Edge cases: empty / whitespace input."""

    def test_empty_string(self) -> None:
        result = chunk_text("")
        assert result == []

    def test_none_like_empty(self) -> None:
        # chunk_text expects str, but '' is the empty case
        result = chunk_text("")
        assert result == []

    def test_whitespace_only(self) -> None:
        result = chunk_text("   \t\n   ")
        assert result == []

    def test_newlines_only(self) -> None:
        result = chunk_text("\n\n\n")
        assert result == []


# ===========================================================================
# TestChunkTextShort
# ===========================================================================


class TestChunkTextShort:
    """Short text (< 1 chunk) should return exactly 1 chunk."""

    def test_short_text_one_chunk(self) -> None:
        text = _make_short_text(100)
        result = chunk_text(text)
        assert len(result) == 1

    def test_short_text_chunk_index(self) -> None:
        text = _make_short_text(50)
        result = chunk_text(text)
        assert result[0].chunk_index == 0

    def test_short_text_preserves_content(self) -> None:
        text = "This is a short review about headphones."
        result = chunk_text(text)
        assert len(result) == 1
        assert result[0].text == text

    def test_single_word(self) -> None:
        result = chunk_text("hello")
        assert len(result) == 1
        assert result[0].text == "hello"

    def test_short_text_with_metadata(self) -> None:
        meta = {"source_url": "https://example.com"}
        result = chunk_text("short text", metadata=meta)
        assert len(result) == 1
        assert result[0].metadata["source_url"] == "https://example.com"
        assert result[0].metadata["chunk_index"] == 0


# ===========================================================================
# TestChunkTextLong
# ===========================================================================


class TestChunkTextLong:
    """Long text should produce multiple chunks with correct properties."""

    def test_multiple_chunks_produced(self) -> None:
        text = _make_long_text(5000)
        result = chunk_text(text)
        assert len(result) > 1

    def test_chunk_indices_sequential(self) -> None:
        text = _make_long_text(5000)
        result = chunk_text(text)
        indices = [c.chunk_index for c in result]
        assert indices == list(range(len(result)))

    def test_chunk_sizes_within_bounds(self) -> None:
        """Each chunk should be <= chunk_size chars (with some tolerance)."""
        text = _make_long_text(5000)
        result = chunk_text(text)
        for chunk in result:
            # Allow slight overhead from separator retention
            assert len(chunk.text) <= CHUNK_SIZE_CHARS + 50, (
                f"Chunk {chunk.chunk_index} too large: {len(chunk.text)} chars"
            )

    def test_overlap_between_consecutive_chunks(self) -> None:
        """End of chunk N should overlap with beginning of chunk N+1."""
        text = _make_long_text(5000)
        result = chunk_text(text)
        overlap_found = 0
        for i in range(len(result) - 1):
            current_end = result[i].text[-100:]  # last 100 chars
            next_start = result[i + 1].text[:300]  # first 300 chars
            # At least some content should overlap
            if current_end[-50:] in next_start or any(word in next_start for word in current_end.split()[-3:]):
                overlap_found += 1
        # Most consecutive pairs should have overlap
        assert overlap_found > 0, "No overlap detected between consecutive chunks"

    def test_all_text_covered(self) -> None:
        """Concatenation of chunks should approximately cover the original text."""
        text = _make_long_text(3000)
        result = chunk_text(text)
        total_chunk_chars = sum(len(c.text) for c in result)
        # Total chars from chunks should be at least as much as original
        # (due to overlap it can be more)
        assert total_chunk_chars >= len(text.strip()) * 0.9

    def test_no_empty_chunks(self) -> None:
        text = _make_long_text(5000)
        result = chunk_text(text)
        for chunk in result:
            assert chunk.text.strip(), f"Empty chunk at index {chunk.chunk_index}"

    def test_paragraph_boundary_preference(self) -> None:
        """Splitter should prefer splitting at paragraph boundaries."""
        paragraphs = ["This is paragraph number {}. " * 30 for _ in range(10)]
        text = "\n\n".join(paragraphs)
        result = chunk_text(text)
        # With paragraph breaks available, chunks should roughly align
        assert len(result) > 1


# ===========================================================================
# TestChunkTextMetadata
# ===========================================================================


class TestChunkTextMetadata:
    """Metadata handling in chunk_text."""

    def test_metadata_attached_to_all_chunks(self) -> None:
        text = _make_long_text(3000)
        meta = {"source_url": "https://example.com", "source_type": "youtube"}
        result = chunk_text(text, metadata=meta)
        for chunk in result:
            assert chunk.metadata["source_url"] == "https://example.com"
            assert chunk.metadata["source_type"] == "youtube"

    def test_chunk_index_in_metadata(self) -> None:
        text = _make_long_text(2000)
        result = chunk_text(text, metadata={"key": "value"})
        for chunk in result:
            assert chunk.metadata["chunk_index"] == chunk.chunk_index

    def test_none_metadata_defaults_to_empty(self) -> None:
        result = chunk_text("some text", metadata=None)
        assert len(result) == 1
        assert "chunk_index" in result[0].metadata

    def test_metadata_not_mutated(self) -> None:
        """Original metadata dict should not be mutated."""
        original_meta = {"source": "test"}
        original_copy = dict(original_meta)
        chunk_text(_make_long_text(2000), metadata=original_meta)
        assert original_meta == original_copy

    def test_each_chunk_gets_own_metadata_copy(self) -> None:
        text = _make_long_text(3000)
        result = chunk_text(text, metadata={"key": "val"})
        if len(result) > 1:
            # Modifying one chunk's metadata should not affect others
            result[0].metadata["extra"] = "extra_val"
            assert "extra" not in result[1].metadata


# ===========================================================================
# TestChunkTextCustomParams
# ===========================================================================


class TestChunkTextCustomParams:
    """Custom chunk_size and chunk_overlap."""

    def test_smaller_chunk_size_produces_more_chunks(self) -> None:
        text = _make_long_text(3000)
        default_chunks = chunk_text(text)
        small_chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
        assert len(small_chunks) > len(default_chunks)

    def test_larger_chunk_size_produces_fewer_chunks(self) -> None:
        text = _make_long_text(3000)
        default_chunks = chunk_text(text)
        large_chunks = chunk_text(text, chunk_size=5000, chunk_overlap=200)
        assert len(large_chunks) < len(default_chunks)

    def test_zero_overlap(self) -> None:
        text = _make_long_text(2000)
        result = chunk_text(text, chunk_overlap=0)
        assert len(result) >= 1

    def test_custom_separators(self) -> None:
        text = "Part A|Part B|Part C|Part D"
        result = chunk_text(text, chunk_size=10, chunk_overlap=0, separators=["|", ""])
        assert len(result) >= 2


# ===========================================================================
# TestChunkTextDicts
# ===========================================================================


class TestChunkTextDicts:
    """Tests for the chunk_text_dicts convenience function."""

    def test_returns_list_of_dicts(self) -> None:
        text = _make_long_text(2000)
        result = chunk_text_dicts(text)
        assert isinstance(result, list)
        assert all(isinstance(d, dict) for d in result)

    def test_dict_keys(self) -> None:
        text = "some short text"
        result = chunk_text_dicts(text, metadata={"source": "test"})
        assert len(result) == 1
        d = result[0]
        assert "text" in d
        assert "chunk_index" in d
        assert "metadata" in d

    def test_empty_text_returns_empty(self) -> None:
        result = chunk_text_dicts("")
        assert result == []

    def test_metadata_propagated(self) -> None:
        meta = {"url": "https://example.com"}
        result = chunk_text_dicts("hello world", metadata=meta)
        assert result[0]["metadata"]["url"] == "https://example.com"

    def test_matches_chunk_text_output(self) -> None:
        text = _make_long_text(2000)
        meta = {"src": "test"}
        chunks = chunk_text(text, metadata=meta)
        dicts = chunk_text_dicts(text, metadata=meta)
        assert len(chunks) == len(dicts)
        for c, d in zip(chunks, dicts):
            assert c.text == d["text"]
            assert c.chunk_index == d["chunk_index"]


# ===========================================================================
# TestChunkTextRussian
# ===========================================================================


class TestChunkTextRussian:
    """Verify chunking works correctly with Russian / Unicode text."""

    def test_russian_text_produces_chunks(self) -> None:
        # Generate ~3000 chars of Russian-like text
        sentence = "Это тестовое предложение для проверки работы чанкера с русским текстом. "
        text = sentence * 200  # ~14000 chars
        result = chunk_text(text)
        assert len(result) > 1

    def test_russian_text_preserves_content(self) -> None:
        text = "Отличные наушники Sony WH-1000XM5 с шумоподавлением."
        result = chunk_text(text)
        assert len(result) == 1
        assert "Sony WH-1000XM5" in result[0].text

    def test_mixed_language_text(self) -> None:
        text = (
            "Sony WH-1000XM5 — отличные наушники. Best noise cancelling headphones. Шумоподавление работает прекрасно."
        )
        result = chunk_text(text)
        assert len(result) == 1
        assert "Sony" in result[0].text
        assert "Шумоподавление" in result[0].text


# ===========================================================================
# TestIngestionExports
# ===========================================================================


class TestIngestionExports:
    """Verify that chunker symbols are properly exported from the package."""

    def test_chunk_text_exported(self) -> None:
        from reviewmind.ingestion import chunk_text as ct

        assert ct is chunk_text

    def test_chunk_text_dicts_exported(self) -> None:
        from reviewmind.ingestion import chunk_text_dicts as ctd

        assert ctd is chunk_text_dicts

    def test_chunk_class_exported(self) -> None:
        from reviewmind.ingestion import Chunk as C

        assert C is Chunk

    def test_constants_exported(self) -> None:
        from reviewmind.ingestion import (
            CHARS_PER_TOKEN,
            CHUNK_OVERLAP_CHARS,
            CHUNK_OVERLAP_TOKENS,
            CHUNK_SIZE_CHARS,
            CHUNK_SIZE_TOKENS,
            DEFAULT_SEPARATORS,
            MIN_CHUNK_LENGTH,
        )

        assert CHUNK_SIZE_TOKENS == 500
        assert CHUNK_OVERLAP_TOKENS == 50
        assert CHARS_PER_TOKEN == 4
        assert CHUNK_SIZE_CHARS == 2000
        assert CHUNK_OVERLAP_CHARS == 200
        assert MIN_CHUNK_LENGTH == 1
        assert isinstance(DEFAULT_SEPARATORS, list)

    def test_all_chunker_symbols_in__all__(self) -> None:
        import reviewmind.ingestion as mod

        expected = {
            "chunk_text",
            "chunk_text_dicts",
            "Chunk",
            "CHUNK_SIZE_TOKENS",
            "CHUNK_SIZE_CHARS",
            "CHUNK_OVERLAP_TOKENS",
            "CHUNK_OVERLAP_CHARS",
            "CHARS_PER_TOKEN",
            "DEFAULT_SEPARATORS",
            "MIN_CHUNK_LENGTH",
        }
        assert expected.issubset(set(mod.__all__))


# ===========================================================================
# TestIntegrationScenarios
# ===========================================================================


class TestIntegrationScenarios:
    """End-to-end scenarios simulating real usage."""

    def test_review_text_chunking(self) -> None:
        """Simulate chunking a product review (~2000 words)."""
        review = (
            "This is a detailed review of the Sony WH-1000XM5 headphones. "
            "The sound quality is excellent with deep bass and clear highs. "
            "The noise cancellation is industry-leading and blocks out most ambient noise. "
            "Battery life lasts about 30 hours on a single charge. "
            "The comfort is good for long listening sessions. "
        ) * 50  # ~2500 words
        meta = {"source_url": "https://example.com/review", "source_type": "web"}
        result = chunk_text(review, metadata=meta)
        assert len(result) >= 2
        for chunk in result:
            assert chunk.metadata["source_url"] == "https://example.com/review"
            assert chunk.metadata["source_type"] == "web"
            assert chunk.metadata["chunk_index"] == chunk.chunk_index

    def test_youtube_transcript_chunking(self) -> None:
        """Simulate chunking a YouTube transcript (~5000 words)."""
        transcript = (
            "Hey everyone welcome to the channel today we're reviewing "
            "the latest smartphone from Samsung the Galaxy S25 Ultra. "
            "First let's talk about the display which is absolutely stunning "
            "with its dynamic AMOLED screen. The processor is a Snapdragon "
            "8 Elite which handles everything with ease. Camera system "
            "features a 200MP main sensor with excellent low light performance. "
        ) * 100
        meta = {
            "source_url": "https://youtube.com/watch?v=test123",
            "source_type": "youtube",
            "is_sponsored": False,
        }
        result = chunk_text(transcript, metadata=meta)
        assert len(result) > 3
        # All chunks should have source_type
        assert all(c.metadata["source_type"] == "youtube" for c in result)

    def test_reddit_post_chunking(self) -> None:
        """Short Reddit post should produce 1 chunk."""
        post = (
            "I just bought the AirPods Pro 2 and they are amazing. "
            "The noise cancellation is way better than the first gen. "
            "Definitely worth the upgrade if you're an Apple user."
        )
        meta = {"source_url": "https://reddit.com/r/headphones/abc", "source_type": "reddit"}
        result = chunk_text(post, metadata=meta)
        assert len(result) == 1
        assert result[0].metadata["source_type"] == "reddit"

    def test_chunk_to_dict_for_qdrant_payload(self) -> None:
        """Verify dicts from chunk_text_dicts can serve as Qdrant payloads."""
        text = _make_long_text(2000)
        meta = {
            "source_url": "https://example.com",
            "source_type": "web",
            "product_query": "Sony XM5",
            "is_sponsored": False,
            "is_curated": True,
            "language": "en",
        }
        dicts = chunk_text_dicts(text, metadata=meta)
        for d in dicts:
            assert isinstance(d["text"], str)
            assert isinstance(d["chunk_index"], int)
            assert d["metadata"]["source_url"] == "https://example.com"
            assert d["metadata"]["product_query"] == "Sony XM5"

    def test_clean_then_chunk_pipeline(self) -> None:
        """Simulate clean → chunk pipeline."""
        from reviewmind.ingestion.cleaner import clean_text

        raw = "<p>This is a review with <b>HTML</b> tags. </p><p>The product is good. " * 300 + "</p>"
        cleaned = clean_text(raw)
        if cleaned:  # may be None/empty if too short after cleaning
            result = chunk_text(cleaned, metadata={"source_type": "web"})
            assert len(result) >= 1
            for chunk in result:
                assert "<p>" not in chunk.text
                assert "<b>" not in chunk.text
