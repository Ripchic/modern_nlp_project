"""Unit tests for reviewmind.ingestion.url_detector — URL type detection & scraper routing."""

from __future__ import annotations

import pytest

from reviewmind.ingestion.url_detector import (
    _ALLOWED_SCHEMES,
    _REDDIT_RE,
    _YOUTUBE_RE,
    _validate_url,
    detect_url_type,
    route_to_scraper,
)
from reviewmind.scrapers.reddit import RedditScraper
from reviewmind.scrapers.web import WebScraper
from reviewmind.scrapers.youtube import YouTubeScraper

# ===================================================================
# Regex patterns
# ===================================================================


class TestYouTubePattern:
    """_YOUTUBE_RE matches YouTube domains."""

    def test_youtube_com(self) -> None:
        assert _YOUTUBE_RE.search("https://www.youtube.com/watch?v=abc123")

    def test_youtu_be(self) -> None:
        assert _YOUTUBE_RE.search("https://youtu.be/abc123")

    def test_youtube_shorts(self) -> None:
        assert _YOUTUBE_RE.search("https://youtube.com/shorts/abc123")

    def test_youtube_embed(self) -> None:
        assert _YOUTUBE_RE.search("https://www.youtube.com/embed/abc123")

    def test_no_match_reddit(self) -> None:
        assert not _YOUTUBE_RE.search("https://reddit.com/r/test")

    def test_case_insensitive(self) -> None:
        assert _YOUTUBE_RE.search("https://YOUTUBE.COM/watch?v=abc")


class TestRedditPattern:
    """_REDDIT_RE matches Reddit domains."""

    def test_reddit_com(self) -> None:
        assert _REDDIT_RE.search("https://www.reddit.com/r/headphones/comments/abc123/")

    def test_old_reddit(self) -> None:
        assert _REDDIT_RE.search("https://old.reddit.com/r/headphones/comments/abc123/")

    def test_redd_it(self) -> None:
        assert _REDDIT_RE.search("https://redd.it/abc123")

    def test_no_match_youtube(self) -> None:
        assert not _REDDIT_RE.search("https://youtube.com/watch?v=abc")

    def test_case_insensitive(self) -> None:
        assert _REDDIT_RE.search("https://REDDIT.COM/r/test")


# ===================================================================
# _validate_url
# ===================================================================


class TestValidateUrl:
    """URL validation helper."""

    def test_valid_https(self) -> None:
        assert _validate_url("https://example.com") == "https://example.com"

    def test_valid_http(self) -> None:
        assert _validate_url("http://example.com") == "http://example.com"

    def test_strips_whitespace(self) -> None:
        assert _validate_url("  https://example.com  ") == "https://example.com"

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty string"):
            _validate_url("")

    def test_none_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty string"):
            _validate_url(None)  # type: ignore[arg-type]

    def test_missing_scheme_raises(self) -> None:
        with pytest.raises(ValueError, match="missing scheme"):
            _validate_url("example.com")

    def test_ftp_scheme_raises(self) -> None:
        with pytest.raises(ValueError, match="unsupported scheme"):
            _validate_url("ftp://example.com")

    def test_no_hostname_raises(self) -> None:
        with pytest.raises(ValueError, match="missing hostname"):
            _validate_url("https://")

    def test_not_a_url_raises(self) -> None:
        with pytest.raises(ValueError, match="missing scheme"):
            _validate_url("not-a-url")

    def test_allowed_schemes_constant(self) -> None:
        assert "http" in _ALLOWED_SCHEMES
        assert "https" in _ALLOWED_SCHEMES
        assert "ftp" not in _ALLOWED_SCHEMES


# ===================================================================
# detect_url_type
# ===================================================================


class TestDetectUrlType:
    """detect_url_type() correctly classifies URLs."""

    # -- YouTube ---

    def test_youtube_watch(self) -> None:
        assert detect_url_type("https://youtube.com/watch?v=xxx") == "youtube"

    def test_youtube_www(self) -> None:
        assert detect_url_type("https://www.youtube.com/watch?v=abc") == "youtube"

    def test_youtu_be(self) -> None:
        assert detect_url_type("https://youtu.be/abc123def45") == "youtube"

    def test_youtube_shorts(self) -> None:
        assert detect_url_type("https://youtube.com/shorts/abc123def45") == "youtube"

    def test_youtube_embed(self) -> None:
        assert detect_url_type("https://www.youtube.com/embed/abc123def45") == "youtube"

    def test_youtube_live(self) -> None:
        assert detect_url_type("https://www.youtube.com/live/abc123def45") == "youtube"

    # -- Reddit ---

    def test_reddit_post(self) -> None:
        assert detect_url_type("https://reddit.com/r/headphones/comments/abc123/slug/") == "reddit"

    def test_reddit_www(self) -> None:
        assert detect_url_type("https://www.reddit.com/r/headphones/comments/abc/") == "reddit"

    def test_old_reddit(self) -> None:
        assert detect_url_type("https://old.reddit.com/r/headphones/comments/abc/") == "reddit"

    def test_redd_it(self) -> None:
        assert detect_url_type("https://redd.it/abc123") == "reddit"

    # -- Web (generic) ---

    def test_web_rtings(self) -> None:
        assert detect_url_type("https://rtings.com/headphones/reviews/sony/wh-1000xm5") == "web"

    def test_web_generic(self) -> None:
        assert detect_url_type("https://example.com/article") == "web"

    def test_web_4pda(self) -> None:
        assert detect_url_type("https://4pda.to/review/some-device") == "web"

    # -- Invalid URLs → ValueError ---

    def test_invalid_no_scheme(self) -> None:
        with pytest.raises(ValueError):
            detect_url_type("not-a-url")

    def test_invalid_empty(self) -> None:
        with pytest.raises(ValueError):
            detect_url_type("")

    def test_invalid_ftp(self) -> None:
        with pytest.raises(ValueError):
            detect_url_type("ftp://files.example.com/data")


# ===================================================================
# route_to_scraper
# ===================================================================


class TestRouteToScraper:
    """route_to_scraper() returns the correct scraper instance."""

    def test_youtube_returns_youtube_scraper(self) -> None:
        scraper = route_to_scraper("https://www.youtube.com/watch?v=abc123def45")
        assert isinstance(scraper, YouTubeScraper)

    def test_youtu_be_returns_youtube_scraper(self) -> None:
        scraper = route_to_scraper("https://youtu.be/abc123def45")
        assert isinstance(scraper, YouTubeScraper)

    def test_reddit_returns_reddit_scraper(self) -> None:
        scraper = route_to_scraper("https://reddit.com/r/headphones/comments/abc/slug/")
        assert isinstance(scraper, RedditScraper)

    def test_redd_it_returns_reddit_scraper(self) -> None:
        scraper = route_to_scraper("https://redd.it/abc123")
        assert isinstance(scraper, RedditScraper)

    def test_web_returns_web_scraper(self) -> None:
        scraper = route_to_scraper("https://rtings.com/headphones/reviews/sony/wh-1000xm5")
        assert isinstance(scraper, WebScraper)

    def test_generic_web_returns_web_scraper(self) -> None:
        scraper = route_to_scraper("https://example.com/article")
        assert isinstance(scraper, WebScraper)

    def test_invalid_url_raises(self) -> None:
        with pytest.raises(ValueError):
            route_to_scraper("not-a-url")

    def test_each_call_returns_new_instance(self) -> None:
        s1 = route_to_scraper("https://youtube.com/watch?v=abc123def45")
        s2 = route_to_scraper("https://youtube.com/watch?v=abc123def45")
        assert s1 is not s2


# ===================================================================
# Module exports via ingestion __init__
# ===================================================================


class TestModuleExports:
    """url_detector symbols are importable from ingestion package."""

    def test_detect_url_type_from_package(self) -> None:
        from reviewmind.ingestion import detect_url_type as fn

        assert callable(fn)

    def test_route_to_scraper_from_package(self) -> None:
        from reviewmind.ingestion import route_to_scraper as fn

        assert callable(fn)

    def test_scraper_type_alias(self) -> None:
        from reviewmind.ingestion import ScraperType

        assert ScraperType is not None
