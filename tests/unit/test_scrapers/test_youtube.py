"""Unit tests for reviewmind.scrapers.youtube — YouTubeScraper."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import patch

import pytest

from reviewmind.scrapers.youtube import (
    _YOUTUBE_URL_PATTERNS,
    DEFAULT_LANGUAGES,
    MIN_WORD_COUNT,
    TranscriptResult,
    YouTubeScraper,
)

# ═══════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════


@dataclass
class FakeSnippet:
    """Mimics FetchedTranscriptSnippet for testing."""

    text: str
    start: float = 0.0
    duration: float = 1.0


class FakeFetchedTranscript:
    """Mimics FetchedTranscript for testing."""

    def __init__(
        self,
        snippets: list[FakeSnippet],
        *,
        video_id: str = "dQw4w9WgXcQ",
        language: str = "English",
        language_code: str = "en",
        is_generated: bool = True,
    ):
        self.snippets = snippets
        self.video_id = video_id
        self.language = language
        self.language_code = language_code
        self.is_generated = is_generated


def _make_long_snippets(word_count: int = 600) -> list[FakeSnippet]:
    """Create snippets totalling approximately *word_count* words."""
    words_per_snippet = 10
    num_snippets = word_count // words_per_snippet
    return [
        FakeSnippet(
            text=" ".join(f"word{i}_{j}" for j in range(words_per_snippet)),
            start=float(i),
            duration=1.0,
        )
        for i in range(num_snippets)
    ]


def _make_short_snippets(word_count: int = 50) -> list[FakeSnippet]:
    """Create snippets totalling fewer than MIN_WORD_COUNT words."""
    return [FakeSnippet(text=" ".join(f"w{j}" for j in range(word_count)))]


# ═══════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════


class TestConstants:
    def test_min_word_count(self):
        assert MIN_WORD_COUNT == 500

    def test_default_languages(self):
        assert DEFAULT_LANGUAGES == ("ru", "en")

    def test_url_patterns_count(self):
        assert len(_YOUTUBE_URL_PATTERNS) >= 5

    def test_constants_importable_from_init(self):
        from reviewmind.scrapers import DEFAULT_LANGUAGES as DL
        from reviewmind.scrapers import MIN_WORD_COUNT as MWC

        assert DL == ("ru", "en")
        assert MWC == 500


# ═══════════════════════════════════════════════════════════════
#  TranscriptResult dataclass
# ═══════════════════════════════════════════════════════════════


class TestTranscriptResult:
    def test_basic_creation(self):
        r = TranscriptResult(
            video_id="abc12345678",
            text="hello world",
            language="English",
            language_code="en",
            is_generated=True,
            word_count=2,
            snippet_count=1,
        )
        assert r.video_id == "abc12345678"
        assert r.text == "hello world"
        assert r.word_count == 2

    def test_default_source_url(self):
        r = TranscriptResult(
            video_id="x", text="t", language="en", language_code="en",
            is_generated=False, word_count=1, snippet_count=1,
        )
        assert r.source_url == ""

    def test_default_extra_metadata(self):
        r = TranscriptResult(
            video_id="x", text="t", language="en", language_code="en",
            is_generated=False, word_count=1, snippet_count=1,
        )
        assert r.extra_metadata == {}

    def test_source_url_settable(self):
        r = TranscriptResult(
            video_id="x", text="t", language="en", language_code="en",
            is_generated=False, word_count=1, snippet_count=1,
        )
        r.source_url = "https://youtube.com/watch?v=x"
        assert r.source_url == "https://youtube.com/watch?v=x"

    def test_importable_from_init(self):
        from reviewmind.scrapers import TranscriptResult as TR

        assert TR is TranscriptResult


# ═══════════════════════════════════════════════════════════════
#  extract_video_id
# ═══════════════════════════════════════════════════════════════


class TestExtractVideoId:
    """Test all supported YouTube URL formats."""

    @pytest.mark.parametrize(
        "url, expected_id",
        [
            # Standard watch URL
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("http://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("http://youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            # Watch with extra params
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=120", "dQw4w9WgXcQ"),
            ("https://www.youtube.com/watch?list=PLxx&v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            # Short URL
            ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("http://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            # Shorts
            ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            # Embed
            ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            # Live
            ("https://www.youtube.com/live/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            # With underscores and hyphens in ID
            ("https://youtu.be/Ab_cD-EfGhI", "Ab_cD-EfGhI"),
            # Whitespace around URL
            ("  https://youtu.be/dQw4w9WgXcQ  ", "dQw4w9WgXcQ"),
        ],
    )
    def test_valid_urls(self, url: str, expected_id: str):
        assert YouTubeScraper.extract_video_id(url) == expected_id

    @pytest.mark.parametrize(
        "url",
        [
            "",
            "not a url",
            "https://google.com",
            "https://vimeo.com/123456",
            "https://www.youtube.com/channel/UCxxx",
            "https://www.youtube.com/watch",  # no v= param
            "https://www.youtube.com/watch?v=short",  # not 11 chars
        ],
    )
    def test_invalid_urls_raise_value_error(self, url: str):
        with pytest.raises(ValueError):
            YouTubeScraper.extract_video_id(url)

    def test_none_url_raises(self):
        with pytest.raises(ValueError):
            YouTubeScraper.extract_video_id(None)  # type: ignore[arg-type]

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            YouTubeScraper.extract_video_id("")

    def test_non_string_raises(self):
        with pytest.raises(ValueError):
            YouTubeScraper.extract_video_id(12345)  # type: ignore[arg-type]


# ═══════════════════════════════════════════════════════════════
#  is_youtube_url
# ═══════════════════════════════════════════════════════════════


class TestIsYoutubeUrl:
    @pytest.mark.parametrize(
        "url",
        [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "https://youtube.com/shorts/dQw4w9WgXcQ",
            "https://youtube.com/embed/dQw4w9WgXcQ",
            "https://youtube.com/live/dQw4w9WgXcQ",
        ],
    )
    def test_valid_youtube_urls(self, url: str):
        assert YouTubeScraper.is_youtube_url(url) is True

    @pytest.mark.parametrize(
        "url",
        [
            "",
            None,
            123,
            "https://google.com",
            "https://vimeo.com/123456",
            "not a url",
        ],
    )
    def test_non_youtube_urls(self, url):
        assert YouTubeScraper.is_youtube_url(url) is False


# ═══════════════════════════════════════════════════════════════
#  YouTubeScraper init
# ═══════════════════════════════════════════════════════════════


class TestYouTubeScraperInit:
    def test_default_languages(self):
        scraper = YouTubeScraper()
        assert scraper._languages == ("ru", "en")

    def test_custom_languages(self):
        scraper = YouTubeScraper(languages=["de", "fr"])
        assert scraper._languages == ("de", "fr")

    def test_default_min_word_count(self):
        scraper = YouTubeScraper()
        assert scraper._min_word_count == 500

    def test_custom_min_word_count(self):
        scraper = YouTubeScraper(min_word_count=100)
        assert scraper._min_word_count == 100

    def test_api_instance_created(self):
        scraper = YouTubeScraper()
        assert scraper._api is not None

    def test_importable_from_init(self):
        from reviewmind.scrapers import YouTubeScraper as YTS

        assert YTS is YouTubeScraper


# ═══════════════════════════════════════════════════════════════
#  _build_clean_text
# ═══════════════════════════════════════════════════════════════


class TestBuildCleanText:
    def test_simple_join(self):
        snippets = [FakeSnippet(text="Hello"), FakeSnippet(text="world")]
        assert YouTubeScraper._build_clean_text(snippets) == "Hello world"

    def test_removes_bracketed_markers(self):
        snippets = [
            FakeSnippet(text="[Music] Hello"),
            FakeSnippet(text="world [Applause]"),
        ]
        assert YouTubeScraper._build_clean_text(snippets) == "Hello world"

    def test_collapses_whitespace(self):
        snippets = [FakeSnippet(text="Hello   world  \t foo")]
        assert YouTubeScraper._build_clean_text(snippets) == "Hello world foo"

    def test_skips_empty_snippets(self):
        snippets = [
            FakeSnippet(text="Hello"),
            FakeSnippet(text=""),
            FakeSnippet(text="world"),
        ]
        assert YouTubeScraper._build_clean_text(snippets) == "Hello world"

    def test_skips_only_marker_snippets(self):
        snippets = [
            FakeSnippet(text="Hello"),
            FakeSnippet(text="[Music]"),
            FakeSnippet(text="world"),
        ]
        assert YouTubeScraper._build_clean_text(snippets) == "Hello world"

    def test_empty_list(self):
        assert YouTubeScraper._build_clean_text([]) == ""

    def test_all_empty_snippets(self):
        snippets = [FakeSnippet(text=""), FakeSnippet(text="  ")]
        assert YouTubeScraper._build_clean_text(snippets) == ""

    def test_complex_markers(self):
        snippets = [FakeSnippet(text="[Music playing] Hello [laughs] world [end]")]
        assert YouTubeScraper._build_clean_text(snippets) == "Hello world"


# ═══════════════════════════════════════════════════════════════
#  get_transcript — success
# ═══════════════════════════════════════════════════════════════


class TestGetTranscriptSuccess:
    def test_returns_transcript_result(self):
        scraper = YouTubeScraper(min_word_count=5)
        snippets = _make_long_snippets(600)
        fake_result = FakeFetchedTranscript(snippets)

        with patch.object(scraper._api, "fetch", return_value=fake_result):
            result = scraper.get_transcript("dQw4w9WgXcQ")

        assert result is not None
        assert isinstance(result, TranscriptResult)
        assert result.video_id == "dQw4w9WgXcQ"
        assert result.language == "English"
        assert result.language_code == "en"
        assert result.is_generated is True
        assert result.word_count >= 500
        assert result.snippet_count == len(snippets)

    def test_uses_default_languages(self):
        scraper = YouTubeScraper(min_word_count=5)
        snippets = _make_long_snippets(600)
        fake_result = FakeFetchedTranscript(snippets)

        with patch.object(scraper._api, "fetch", return_value=fake_result) as mock_fetch:
            scraper.get_transcript("abc12345678")
            mock_fetch.assert_called_once_with("abc12345678", languages=("ru", "en"))

    def test_uses_custom_languages(self):
        scraper = YouTubeScraper(min_word_count=5)
        snippets = _make_long_snippets(600)
        fake_result = FakeFetchedTranscript(snippets)

        with patch.object(scraper._api, "fetch", return_value=fake_result) as mock_fetch:
            scraper.get_transcript("abc12345678", languages=("de", "fr"))
            mock_fetch.assert_called_once_with("abc12345678", languages=("de", "fr"))

    def test_strips_video_id_whitespace(self):
        scraper = YouTubeScraper(min_word_count=5)
        snippets = _make_long_snippets(600)
        fake_result = FakeFetchedTranscript(snippets)

        with patch.object(scraper._api, "fetch", return_value=fake_result) as mock_fetch:
            scraper.get_transcript("  dQw4w9WgXcQ  ")
            mock_fetch.assert_called_once_with("dQw4w9WgXcQ", languages=("ru", "en"))

    def test_russian_transcript(self):
        scraper = YouTubeScraper(min_word_count=5)
        snippets = _make_long_snippets(600)
        fake_result = FakeFetchedTranscript(
            snippets, language="Russian", language_code="ru", is_generated=False,
        )

        with patch.object(scraper._api, "fetch", return_value=fake_result):
            result = scraper.get_transcript("abc12345678")

        assert result is not None
        assert result.language == "Russian"
        assert result.language_code == "ru"
        assert result.is_generated is False


# ═══════════════════════════════════════════════════════════════
#  get_transcript — too short
# ═══════════════════════════════════════════════════════════════


class TestGetTranscriptTooShort:
    def test_returns_none_when_below_min_words(self):
        scraper = YouTubeScraper(min_word_count=500)
        snippets = _make_short_snippets(50)
        fake_result = FakeFetchedTranscript(snippets)

        with patch.object(scraper._api, "fetch", return_value=fake_result):
            result = scraper.get_transcript("dQw4w9WgXcQ")

        assert result is None

    def test_custom_min_word_count(self):
        scraper = YouTubeScraper(min_word_count=10)
        snippets = [FakeSnippet(text=" ".join(f"word{i}" for i in range(15)))]
        fake_result = FakeFetchedTranscript(snippets)

        with patch.object(scraper._api, "fetch", return_value=fake_result):
            result = scraper.get_transcript("dQw4w9WgXcQ")

        assert result is not None
        assert result.word_count == 15

    def test_exactly_min_words_passes(self):
        scraper = YouTubeScraper(min_word_count=10)
        snippets = [FakeSnippet(text=" ".join(f"word{i}" for i in range(10)))]
        fake_result = FakeFetchedTranscript(snippets)

        with patch.object(scraper._api, "fetch", return_value=fake_result):
            result = scraper.get_transcript("dQw4w9WgXcQ")

        assert result is not None
        assert result.word_count == 10

    def test_one_below_min_words_fails(self):
        scraper = YouTubeScraper(min_word_count=10)
        snippets = [FakeSnippet(text=" ".join(f"word{i}" for i in range(9)))]
        fake_result = FakeFetchedTranscript(snippets)

        with patch.object(scraper._api, "fetch", return_value=fake_result):
            result = scraper.get_transcript("dQw4w9WgXcQ")

        assert result is None


# ═══════════════════════════════════════════════════════════════
#  get_transcript — error handling
# ═══════════════════════════════════════════════════════════════


class TestGetTranscriptErrors:
    def test_transcripts_disabled_returns_none(self):
        from youtube_transcript_api._errors import TranscriptsDisabled

        scraper = YouTubeScraper()
        with patch.object(
            scraper._api, "fetch", side_effect=TranscriptsDisabled("abc12345678")
        ):
            result = scraper.get_transcript("abc12345678")
        assert result is None

    def test_no_transcript_found_returns_none(self):
        from youtube_transcript_api._errors import NoTranscriptFound

        scraper = YouTubeScraper()
        with patch.object(
            scraper._api,
            "fetch",
            side_effect=NoTranscriptFound("abc12345678", [], []),
        ):
            result = scraper.get_transcript("abc12345678")
        assert result is None

    def test_video_unavailable_returns_none(self):
        from youtube_transcript_api._errors import VideoUnavailable

        scraper = YouTubeScraper()
        with patch.object(
            scraper._api, "fetch", side_effect=VideoUnavailable("abc12345678")
        ):
            result = scraper.get_transcript("abc12345678")
        assert result is None

    def test_unexpected_error_returns_none(self):
        scraper = YouTubeScraper()
        with patch.object(
            scraper._api, "fetch", side_effect=RuntimeError("network error")
        ):
            result = scraper.get_transcript("abc12345678")
        assert result is None

    def test_invalid_video_id_none(self):
        scraper = YouTubeScraper()
        assert scraper.get_transcript(None) is None  # type: ignore[arg-type]

    def test_empty_video_id(self):
        scraper = YouTubeScraper()
        assert scraper.get_transcript("") is None

    def test_non_string_video_id(self):
        scraper = YouTubeScraper()
        assert scraper.get_transcript(12345) is None  # type: ignore[arg-type]


# ═══════════════════════════════════════════════════════════════
#  get_transcript_by_url
# ═══════════════════════════════════════════════════════════════


class TestGetTranscriptByUrl:
    def test_success(self):
        scraper = YouTubeScraper(min_word_count=5)
        snippets = _make_long_snippets(600)
        fake_result = FakeFetchedTranscript(snippets)

        with patch.object(scraper._api, "fetch", return_value=fake_result):
            result = scraper.get_transcript_by_url(
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
            )

        assert result is not None
        assert result.video_id == "dQw4w9WgXcQ"
        assert result.source_url == "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    def test_source_url_is_trimmed(self):
        scraper = YouTubeScraper(min_word_count=5)
        snippets = _make_long_snippets(600)
        fake_result = FakeFetchedTranscript(snippets)

        with patch.object(scraper._api, "fetch", return_value=fake_result):
            result = scraper.get_transcript_by_url(
                "  https://youtu.be/dQw4w9WgXcQ  "
            )

        assert result is not None
        assert result.source_url == "https://youtu.be/dQw4w9WgXcQ"

    def test_invalid_url_raises(self):
        scraper = YouTubeScraper()
        with pytest.raises(ValueError):
            scraper.get_transcript_by_url("https://google.com")

    def test_returns_none_when_transcript_unavailable(self):
        from youtube_transcript_api._errors import TranscriptsDisabled

        scraper = YouTubeScraper()
        with patch.object(
            scraper._api, "fetch", side_effect=TranscriptsDisabled("dQw4w9WgXcQ")
        ):
            result = scraper.get_transcript_by_url(
                "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
            )
        assert result is None

    def test_custom_languages_forwarded(self):
        scraper = YouTubeScraper(min_word_count=5)
        snippets = _make_long_snippets(600)
        fake_result = FakeFetchedTranscript(snippets)

        with patch.object(scraper._api, "fetch", return_value=fake_result) as mock_fetch:
            scraper.get_transcript_by_url(
                "https://youtu.be/dQw4w9WgXcQ",
                languages=("de",),
            )
            mock_fetch.assert_called_once_with("dQw4w9WgXcQ", languages=("de",))

    def test_short_url(self):
        scraper = YouTubeScraper(min_word_count=5)
        snippets = _make_long_snippets(600)
        fake_result = FakeFetchedTranscript(snippets)

        with patch.object(scraper._api, "fetch", return_value=fake_result):
            result = scraper.get_transcript_by_url("https://youtu.be/dQw4w9WgXcQ")

        assert result is not None
        assert result.video_id == "dQw4w9WgXcQ"

    def test_shorts_url(self):
        scraper = YouTubeScraper(min_word_count=5)
        snippets = _make_long_snippets(600)
        fake_result = FakeFetchedTranscript(snippets)

        with patch.object(scraper._api, "fetch", return_value=fake_result):
            result = scraper.get_transcript_by_url(
                "https://youtube.com/shorts/dQw4w9WgXcQ"
            )

        assert result is not None
        assert result.video_id == "dQw4w9WgXcQ"


# ═══════════════════════════════════════════════════════════════
#  Scraper exports
# ═══════════════════════════════════════════════════════════════


class TestScrapersExports:
    def test_all_symbols_exported(self):
        from reviewmind import scrapers

        assert hasattr(scrapers, "YouTubeScraper")
        assert hasattr(scrapers, "TranscriptResult")
        assert hasattr(scrapers, "MIN_WORD_COUNT")
        assert hasattr(scrapers, "DEFAULT_LANGUAGES")

    def test_all_contains_expected_names(self):
        from reviewmind.scrapers import __all__

        expected = {"YouTubeScraper", "TranscriptResult", "MIN_WORD_COUNT", "DEFAULT_LANGUAGES"}
        assert set(__all__) == expected


# ═══════════════════════════════════════════════════════════════
#  Integration-style scenarios (mocked)
# ═══════════════════════════════════════════════════════════════


class TestIntegrationScenarios:
    def test_full_flow_watch_url(self):
        """Full flow: URL → extract_video_id → fetch → clean → TranscriptResult."""
        scraper = YouTubeScraper(min_word_count=5)
        snippets = _make_long_snippets(600)
        fake_result = FakeFetchedTranscript(snippets)

        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42"
        with patch.object(scraper._api, "fetch", return_value=fake_result):
            result = scraper.get_transcript_by_url(url)

        assert result is not None
        assert result.video_id == "dQw4w9WgXcQ"
        assert result.source_url == url
        assert result.word_count >= 500

    def test_is_youtube_url_consistent_with_extract(self):
        """is_youtube_url returns True for all URLs that extract_video_id handles."""
        urls = [
            "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            "https://youtu.be/dQw4w9WgXcQ",
            "https://youtube.com/shorts/dQw4w9WgXcQ",
            "https://youtube.com/embed/dQw4w9WgXcQ",
            "https://youtube.com/live/dQw4w9WgXcQ",
        ]
        for url in urls:
            assert YouTubeScraper.is_youtube_url(url) is True
            video_id = YouTubeScraper.extract_video_id(url)
            assert len(video_id) == 11

    def test_scraper_reusable_for_multiple_videos(self):
        """A single scraper instance can fetch transcripts for multiple videos."""
        scraper = YouTubeScraper(min_word_count=5)
        snippets = _make_long_snippets(600)

        for vid in ("aaaaaaaaaaa", "bbbbbbbbbbb", "ccccccccccc"):
            fake_result = FakeFetchedTranscript(snippets, video_id=vid)
            with patch.object(scraper._api, "fetch", return_value=fake_result):
                result = scraper.get_transcript(vid)
            assert result is not None
            assert result.video_id == vid

    def test_clean_text_removes_all_noise(self):
        """Verify that the cleaned output has no brackets, excess spaces, etc."""
        snippets = [
            FakeSnippet(text="[Music]  Hello"),
            FakeSnippet(text=""),
            FakeSnippet(text="world  [Applause]  test"),
            FakeSnippet(text="[Laughter]"),
        ]
        text = YouTubeScraper._build_clean_text(snippets)
        assert "[" not in text
        assert "]" not in text
        assert "  " not in text
        assert text == "Hello world test"

    def test_error_isolation_per_video(self):
        """Error fetching one video doesn't affect fetching another."""
        from youtube_transcript_api._errors import VideoUnavailable

        scraper = YouTubeScraper(min_word_count=5)
        snippets = _make_long_snippets(600)
        fake_result = FakeFetchedTranscript(snippets, video_id="good_video_id")

        # First call fails
        with patch.object(
            scraper._api, "fetch", side_effect=VideoUnavailable("bad_video_id_")
        ):
            r1 = scraper.get_transcript("bad_video_id_")
        assert r1 is None

        # Second call succeeds
        with patch.object(scraper._api, "fetch", return_value=fake_result):
            r2 = scraper.get_transcript("good_video_id")
        assert r2 is not None
        assert r2.video_id == "good_video_id"
