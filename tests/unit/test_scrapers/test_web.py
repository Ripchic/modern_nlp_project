"""Unit tests for reviewmind.scrapers.web — WebScraper."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from reviewmind.scrapers.web import (
    DEFAULT_TIMEOUT,
    MIN_TEXT_LENGTH,
    WebPage,
    WebScraper,
    _TimeoutError,
)

# ═══════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════


def _make_html(text: str, title: str = "Test Title", author: str = "Test Author") -> str:
    """Build a minimal HTML document for testing."""
    return f"""
    <html>
    <head>
        <title>{title}</title>
        <meta name="author" content="{author}">
    </head>
    <body>
        <article>
            <h1>{title}</h1>
            <p>{text}</p>
        </article>
    </body>
    </html>
    """


def _long_text(word_count: int = 200) -> str:
    """Generate text with approximately *word_count* words."""
    return " ".join(f"word{i}" for i in range(word_count))


def _short_text(char_count: int = 50) -> str:
    """Generate text shorter than MIN_TEXT_LENGTH."""
    return "a " * (char_count // 2)


class FakeDocument:
    """Mimics trafilatura's Document for bare_extraction results."""

    def __init__(
        self,
        text: str = "",
        title: str | None = None,
        author: str | None = None,
        date: str | None = None,
        sitename: str | None = None,
        description: str | None = None,
        language: str | None = None,
    ):
        self.text = text
        self.title = title
        self.author = author
        self.date = date
        self.sitename = sitename
        self.description = description
        self.language = language


# ═══════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════


class TestConstants:
    def test_default_timeout(self):
        assert DEFAULT_TIMEOUT == 30

    def test_min_text_length(self):
        assert MIN_TEXT_LENGTH == 200

    def test_timeout_is_int(self):
        assert isinstance(DEFAULT_TIMEOUT, int)

    def test_min_text_length_is_int(self):
        assert isinstance(MIN_TEXT_LENGTH, int)


# ═══════════════════════════════════════════════════════════════
#  WebPage dataclass
# ═══════════════════════════════════════════════════════════════


class TestWebPage:
    def test_create_minimal(self):
        page = WebPage(url="https://example.com", text="Hello world")
        assert page.url == "https://example.com"
        assert page.text == "Hello world"
        assert page.title is None
        assert page.author is None
        assert page.date is None
        assert page.sitename is None
        assert page.description is None
        assert page.language is None
        assert page.word_count == 0
        assert page.extra_metadata == {}

    def test_create_full(self):
        page = WebPage(
            url="https://rtings.com/review",
            text="Full review text",
            title="Headphones Review",
            author="John Doe",
            date="2026-01-15",
            sitename="RTINGS",
            description="A review of headphones",
            language="en",
            word_count=150,
            extra_metadata={"category": "headphones"},
        )
        assert page.title == "Headphones Review"
        assert page.author == "John Doe"
        assert page.date == "2026-01-15"
        assert page.sitename == "RTINGS"
        assert page.description == "A review of headphones"
        assert page.language == "en"
        assert page.word_count == 150
        assert page.extra_metadata == {"category": "headphones"}

    def test_extra_metadata_default_not_shared(self):
        p1 = WebPage(url="a", text="a")
        p2 = WebPage(url="b", text="b")
        p1.extra_metadata["key"] = "val"
        assert "key" not in p2.extra_metadata

    def test_word_count_default(self):
        page = WebPage(url="u", text="one two three")
        assert page.word_count == 0  # Computed by scraper, not automatic


# ═══════════════════════════════════════════════════════════════
#  WebScraper init
# ═══════════════════════════════════════════════════════════════


class TestWebScraperInit:
    def test_defaults(self):
        scraper = WebScraper()
        assert scraper._timeout == DEFAULT_TIMEOUT
        assert scraper._min_text_length == MIN_TEXT_LENGTH
        assert scraper._favor_precision is False
        assert scraper._favor_recall is True
        assert scraper._include_comments is False
        assert scraper._include_tables is True

    def test_custom_timeout(self):
        scraper = WebScraper(timeout=10)
        assert scraper._timeout == 10

    def test_custom_min_text_length(self):
        scraper = WebScraper(min_text_length=500)
        assert scraper._min_text_length == 500

    def test_custom_favor_precision(self):
        scraper = WebScraper(favor_precision=True, favor_recall=False)
        assert scraper._favor_precision is True
        assert scraper._favor_recall is False

    def test_custom_include_comments(self):
        scraper = WebScraper(include_comments=True)
        assert scraper._include_comments is True

    def test_custom_include_tables(self):
        scraper = WebScraper(include_tables=False)
        assert scraper._include_tables is False


# ═══════════════════════════════════════════════════════════════
#  is_web_url (static)
# ═══════════════════════════════════════════════════════════════


class TestIsWebUrl:
    @pytest.mark.parametrize("url", [
        "https://www.rtings.com/headphones",
        "http://example.com/page",
        "https://wirecutter.com/reviews/best-headphones",
        "https://4pda.to/review/123",
        "https://example.com",
        "HTTP://EXAMPLE.COM",
    ])
    def test_valid_urls(self, url: str):
        assert WebScraper.is_web_url(url) is True

    @pytest.mark.parametrize("url", [
        "",
        "not-a-url",
        "ftp://files.example.com",
        "just some text",
    ])
    def test_invalid_urls(self, url: str):
        assert WebScraper.is_web_url(url) is False

    def test_none_url(self):
        assert WebScraper.is_web_url(None) is False  # type: ignore[arg-type]

    def test_non_string_url(self):
        assert WebScraper.is_web_url(123) is False  # type: ignore[arg-type]

    def test_whitespace_trimmed(self):
        assert WebScraper.is_web_url("  https://example.com  ") is True


# ═══════════════════════════════════════════════════════════════
#  _validate_url
# ═══════════════════════════════════════════════════════════════


class TestValidateUrl:
    def test_valid_http(self):
        scraper = WebScraper()
        assert scraper._validate_url("http://example.com") is True

    def test_valid_https(self):
        scraper = WebScraper()
        assert scraper._validate_url("https://example.com") is True

    def test_empty_string(self):
        scraper = WebScraper()
        assert scraper._validate_url("") is False

    def test_none(self):
        scraper = WebScraper()
        assert scraper._validate_url(None) is False  # type: ignore[arg-type]

    def test_no_scheme(self):
        scraper = WebScraper()
        assert scraper._validate_url("example.com") is False

    def test_ftp_scheme(self):
        scraper = WebScraper()
        assert scraper._validate_url("ftp://example.com") is False

    def test_whitespace_trimmed(self):
        scraper = WebScraper()
        assert scraper._validate_url("  https://example.com  ") is True


# ═══════════════════════════════════════════════════════════════
#  _clean_text
# ═══════════════════════════════════════════════════════════════


class TestCleanText:
    def test_collapses_spaces(self):
        assert WebScraper._clean_text("hello    world") == "hello world"

    def test_collapses_tabs(self):
        assert WebScraper._clean_text("hello\t\tworld") == "hello world"

    def test_collapses_multiple_newlines(self):
        result = WebScraper._clean_text("hello\n\n\n\nworld")
        assert result == "hello\n\nworld"

    def test_preserves_single_newline(self):
        result = WebScraper._clean_text("hello\nworld")
        assert result == "hello\nworld"

    def test_preserves_double_newline(self):
        result = WebScraper._clean_text("hello\n\nworld")
        assert result == "hello\n\nworld"

    def test_strips_leading_trailing(self):
        assert WebScraper._clean_text("  hello  ") == "hello"

    def test_empty_string(self):
        assert WebScraper._clean_text("") == ""

    def test_mixed_whitespace(self):
        result = WebScraper._clean_text("  hello  \t  world  \n\n\n\n  end  ")
        # Horizontal whitespace collapsed to single space; 4 newlines → 2; outer stripped
        assert "hello world" in result
        assert "\n\n\n" not in result
        assert result.startswith("hello")
        assert result.endswith("end")


# ═══════════════════════════════════════════════════════════════
#  _get_text
# ═══════════════════════════════════════════════════════════════


class TestGetText:
    def test_from_document(self):
        doc = FakeDocument(text="Hello world")
        assert WebScraper._get_text(doc) == "Hello world"

    def test_from_document_none_text(self):
        doc = FakeDocument(text=None)  # type: ignore[arg-type]
        assert WebScraper._get_text(doc) == ""

    def test_from_dict(self):
        d = {"text": "Some text"}
        assert WebScraper._get_text(d) == "Some text"

    def test_from_dict_missing(self):
        d = {"title": "only title"}
        assert WebScraper._get_text(d) == ""

    def test_from_unsupported_type(self):
        assert WebScraper._get_text(42) == ""

    def test_from_document_empty_text(self):
        doc = FakeDocument(text="")
        assert WebScraper._get_text(doc) == ""


# ═══════════════════════════════════════════════════════════════
#  _get_attr
# ═══════════════════════════════════════════════════════════════


class TestGetAttr:
    def test_from_document(self):
        doc = FakeDocument(title="My Title")
        assert WebScraper._get_attr(doc, "title") == "My Title"

    def test_from_document_none(self):
        doc = FakeDocument(title=None)
        assert WebScraper._get_attr(doc, "title") is None

    def test_from_document_empty(self):
        doc = FakeDocument(title="")
        assert WebScraper._get_attr(doc, "title") is None

    def test_from_document_whitespace(self):
        doc = FakeDocument(title="   ")
        assert WebScraper._get_attr(doc, "title") is None

    def test_from_dict(self):
        d = {"author": "Jane Doe"}
        assert WebScraper._get_attr(d, "author") == "Jane Doe"

    def test_from_dict_missing(self):
        d = {"title": "T"}
        assert WebScraper._get_attr(d, "author") is None

    def test_strips_whitespace(self):
        doc = FakeDocument(title="  Trimmed  ")
        assert WebScraper._get_attr(doc, "title") == "Trimmed"

    def test_unknown_attr(self):
        doc = FakeDocument()
        assert WebScraper._get_attr(doc, "nonexistent") is None


# ═══════════════════════════════════════════════════════════════
#  parse_url — success cases
# ═══════════════════════════════════════════════════════════════


class TestParseUrlSuccess:
    @patch("reviewmind.scrapers.web.trafilatura.fetch_url")
    @patch("reviewmind.scrapers.web.trafilatura.bare_extraction")
    def test_basic_extraction(self, mock_extract, mock_fetch):
        long = _long_text(300)
        mock_fetch.return_value = "<html><body>" + long + "</body></html>"
        mock_extract.return_value = FakeDocument(
            text=long,
            title="Review Title",
            author="Author Name",
            date="2026-01-15",
            sitename="RTINGS",
            description="A review",
            language="en",
        )

        scraper = WebScraper()
        result = scraper.parse_url("https://rtings.com/review")

        assert result is not None
        assert result.url == "https://rtings.com/review"
        assert "word0" in result.text
        assert result.title == "Review Title"
        assert result.author == "Author Name"
        assert result.date == "2026-01-15"
        assert result.sitename == "RTINGS"
        assert result.description == "A review"
        assert result.language == "en"
        assert result.word_count > 0

    @patch("reviewmind.scrapers.web.trafilatura.fetch_url")
    @patch("reviewmind.scrapers.web.trafilatura.bare_extraction")
    def test_metadata_optional(self, mock_extract, mock_fetch):
        long = _long_text(300)
        mock_fetch.return_value = "<html>" + long + "</html>"
        mock_extract.return_value = FakeDocument(text=long)

        scraper = WebScraper()
        result = scraper.parse_url("https://example.com/page")

        assert result is not None
        assert result.title is None
        assert result.author is None
        assert result.date is None

    @patch("reviewmind.scrapers.web.trafilatura.fetch_url")
    @patch("reviewmind.scrapers.web.trafilatura.bare_extraction")
    def test_url_whitespace_stripped(self, mock_extract, mock_fetch):
        long = _long_text(300)
        mock_fetch.return_value = long
        mock_extract.return_value = FakeDocument(text=long)

        scraper = WebScraper()
        result = scraper.parse_url("  https://example.com  ")

        assert result is not None
        assert result.url == "https://example.com"

    @patch("reviewmind.scrapers.web.trafilatura.fetch_url")
    @patch("reviewmind.scrapers.web.trafilatura.bare_extraction")
    def test_word_count_computed(self, mock_extract, mock_fetch):
        text = _long_text(50)  # 50 words, but > 200 chars
        mock_fetch.return_value = text
        mock_extract.return_value = FakeDocument(text=text)

        scraper = WebScraper(min_text_length=10)
        result = scraper.parse_url("https://example.com")

        assert result is not None
        assert result.word_count == 50

    @patch("reviewmind.scrapers.web.trafilatura.fetch_url")
    @patch("reviewmind.scrapers.web.trafilatura.bare_extraction")
    def test_dict_result(self, mock_extract, mock_fetch):
        """trafilatura may return a dict in some configurations."""
        long = _long_text(300)
        mock_fetch.return_value = "<html>" + long + "</html>"
        mock_extract.return_value = {"text": long, "title": "Dict Title", "author": "Dict Author"}

        scraper = WebScraper()
        result = scraper.parse_url("https://example.com")

        assert result is not None
        assert result.title == "Dict Title"
        assert result.author == "Dict Author"


# ═══════════════════════════════════════════════════════════════
#  parse_url — failure/edge cases
# ═══════════════════════════════════════════════════════════════


class TestParseUrlFailure:
    def test_invalid_url_none(self):
        scraper = WebScraper()
        assert scraper.parse_url(None) is None  # type: ignore[arg-type]

    def test_invalid_url_empty(self):
        scraper = WebScraper()
        assert scraper.parse_url("") is None

    def test_invalid_url_no_scheme(self):
        scraper = WebScraper()
        assert scraper.parse_url("example.com/page") is None

    def test_invalid_url_ftp(self):
        scraper = WebScraper()
        assert scraper.parse_url("ftp://example.com") is None

    @patch("reviewmind.scrapers.web.trafilatura.fetch_url")
    def test_download_returns_none(self, mock_fetch):
        mock_fetch.return_value = None
        scraper = WebScraper()
        assert scraper.parse_url("https://example.com") is None

    @patch("reviewmind.scrapers.web.trafilatura.fetch_url")
    def test_download_returns_empty(self, mock_fetch):
        mock_fetch.return_value = ""
        scraper = WebScraper()
        assert scraper.parse_url("https://example.com") is None

    @patch("reviewmind.scrapers.web.trafilatura.fetch_url")
    def test_download_exception(self, mock_fetch):
        mock_fetch.side_effect = ConnectionError("Network error")
        scraper = WebScraper()
        assert scraper.parse_url("https://example.com") is None

    @patch("reviewmind.scrapers.web.trafilatura.fetch_url")
    @patch("reviewmind.scrapers.web.trafilatura.bare_extraction")
    def test_extraction_returns_none(self, mock_extract, mock_fetch):
        mock_fetch.return_value = "<html>paywall</html>"
        mock_extract.return_value = None
        scraper = WebScraper()
        assert scraper.parse_url("https://example.com") is None

    @patch("reviewmind.scrapers.web.trafilatura.fetch_url")
    @patch("reviewmind.scrapers.web.trafilatura.bare_extraction")
    def test_extraction_empty_text(self, mock_extract, mock_fetch):
        mock_fetch.return_value = "<html></html>"
        mock_extract.return_value = FakeDocument(text="")
        scraper = WebScraper()
        assert scraper.parse_url("https://example.com") is None

    @patch("reviewmind.scrapers.web.trafilatura.fetch_url")
    @patch("reviewmind.scrapers.web.trafilatura.bare_extraction")
    def test_text_too_short(self, mock_extract, mock_fetch):
        mock_fetch.return_value = "<html>short</html>"
        mock_extract.return_value = FakeDocument(text="Short text")
        scraper = WebScraper()
        assert scraper.parse_url("https://example.com") is None

    @patch("reviewmind.scrapers.web.trafilatura.fetch_url")
    @patch("reviewmind.scrapers.web.trafilatura.bare_extraction")
    def test_extraction_exception(self, mock_extract, mock_fetch):
        mock_fetch.return_value = "<html>content</html>"
        mock_extract.side_effect = RuntimeError("extraction failed")
        scraper = WebScraper()
        assert scraper.parse_url("https://example.com") is None


# ═══════════════════════════════════════════════════════════════
#  parse_html
# ═══════════════════════════════════════════════════════════════


class TestParseHtml:
    @patch("reviewmind.scrapers.web.trafilatura.bare_extraction")
    def test_basic(self, mock_extract):
        long = _long_text(300)
        mock_extract.return_value = FakeDocument(text=long, title="Parsed")

        scraper = WebScraper()
        result = scraper.parse_html("<html>" + long + "</html>", url="https://example.com")

        assert result is not None
        assert result.title == "Parsed"

    def test_empty_html(self):
        scraper = WebScraper()
        assert scraper.parse_html("") is None

    def test_none_html(self):
        scraper = WebScraper()
        assert scraper.parse_html(None) is None  # type: ignore[arg-type]

    def test_non_string_html(self):
        scraper = WebScraper()
        assert scraper.parse_html(123) is None  # type: ignore[arg-type]

    @patch("reviewmind.scrapers.web.trafilatura.bare_extraction")
    def test_without_url(self, mock_extract):
        long = _long_text(300)
        mock_extract.return_value = FakeDocument(text=long)

        scraper = WebScraper()
        result = scraper.parse_html("<html>" + long + "</html>")

        assert result is not None
        assert result.url == ""


# ═══════════════════════════════════════════════════════════════
#  _download with timeout
# ═══════════════════════════════════════════════════════════════


class TestDownload:
    @patch("reviewmind.scrapers.web.trafilatura.fetch_url")
    def test_success(self, mock_fetch):
        mock_fetch.return_value = "<html>content</html>"
        scraper = WebScraper()
        html = scraper._download("https://example.com")
        assert html == "<html>content</html>"

    @patch("reviewmind.scrapers.web.trafilatura.fetch_url")
    def test_returns_none(self, mock_fetch):
        mock_fetch.return_value = None
        scraper = WebScraper()
        assert scraper._download("https://example.com") is None

    @patch("reviewmind.scrapers.web.trafilatura.fetch_url")
    def test_returns_empty(self, mock_fetch):
        mock_fetch.return_value = ""
        scraper = WebScraper()
        assert scraper._download("https://example.com") is None

    @patch("reviewmind.scrapers.web.trafilatura.fetch_url")
    def test_network_error(self, mock_fetch):
        mock_fetch.side_effect = ConnectionError("Host unreachable")
        scraper = WebScraper()
        assert scraper._download("https://example.com") is None

    @patch("reviewmind.scrapers.web.trafilatura.fetch_url")
    def test_timeout_error(self, mock_fetch):
        mock_fetch.side_effect = _TimeoutError("timed out")
        scraper = WebScraper()
        assert scraper._download("https://example.com") is None


# ═══════════════════════════════════════════════════════════════
#  _extract
# ═══════════════════════════════════════════════════════════════


class TestExtract:
    @patch("reviewmind.scrapers.web.trafilatura.bare_extraction")
    def test_success(self, mock_extract):
        long = _long_text(300)
        mock_extract.return_value = FakeDocument(
            text=long, title="T", author="A", date="2026-03-01",
            sitename="S", description="D", language="en",
        )
        scraper = WebScraper()
        result = scraper._extract("<html>X</html>", "https://example.com")

        assert result is not None
        assert result.title == "T"
        assert result.author == "A"
        assert result.date == "2026-03-01"
        assert result.sitename == "S"
        assert result.description == "D"
        assert result.language == "en"
        assert result.word_count == 300

    @patch("reviewmind.scrapers.web.trafilatura.bare_extraction")
    def test_none_result(self, mock_extract):
        mock_extract.return_value = None
        scraper = WebScraper()
        assert scraper._extract("<html></html>", "") is None

    @patch("reviewmind.scrapers.web.trafilatura.bare_extraction")
    def test_exception(self, mock_extract):
        mock_extract.side_effect = ValueError("bad html")
        scraper = WebScraper()
        assert scraper._extract("<html></html>", "") is None

    @patch("reviewmind.scrapers.web.trafilatura.bare_extraction")
    def test_extraction_options_passed(self, mock_extract):
        long = _long_text(300)
        mock_extract.return_value = FakeDocument(text=long)

        scraper = WebScraper(
            favor_precision=True,
            favor_recall=False,
            include_comments=True,
            include_tables=False,
        )
        scraper._extract("<html></html>", "https://example.com")

        _, kwargs = mock_extract.call_args
        assert kwargs["favor_precision"] is True
        assert kwargs["favor_recall"] is False
        assert kwargs["include_comments"] is True
        assert kwargs["include_tables"] is False
        assert kwargs["with_metadata"] is True

    @patch("reviewmind.scrapers.web.trafilatura.bare_extraction")
    def test_text_cleaning_applied(self, mock_extract):
        text = "hello    world\n\n\n\nend " + _long_text(200)
        mock_extract.return_value = FakeDocument(text=text)

        scraper = WebScraper()
        result = scraper._extract("<html></html>", "")

        assert result is not None
        # Multiple spaces should be collapsed
        assert "    " not in result.text
        # Multiple newlines should be collapsed to max 2
        assert "\n\n\n" not in result.text


# ═══════════════════════════════════════════════════════════════
#  Scrapers __init__.py exports
# ═══════════════════════════════════════════════════════════════


class TestScrapersExports:
    def test_web_scraper_importable(self):
        from reviewmind.scrapers import WebScraper as WS
        assert WS is WebScraper

    def test_web_page_importable(self):
        from reviewmind.scrapers import WebPage as WP
        assert WP is WebPage

    def test_default_timeout_importable(self):
        from reviewmind.scrapers import DEFAULT_TIMEOUT as DT
        assert DT == DEFAULT_TIMEOUT

    def test_min_text_length_importable(self):
        from reviewmind.scrapers import MIN_TEXT_LENGTH as MTL
        assert MTL == MIN_TEXT_LENGTH

    def test_all_exports(self):
        import reviewmind.scrapers as scrapers
        assert "WebScraper" in scrapers.__all__
        assert "WebPage" in scrapers.__all__
        assert "DEFAULT_TIMEOUT" in scrapers.__all__
        assert "MIN_TEXT_LENGTH" in scrapers.__all__


# ═══════════════════════════════════════════════════════════════
#  Integration scenarios
# ═══════════════════════════════════════════════════════════════


class TestIntegrationScenarios:
    @patch("reviewmind.scrapers.web.trafilatura.fetch_url")
    @patch("reviewmind.scrapers.web.trafilatura.bare_extraction")
    def test_rtings_style_review(self, mock_extract, mock_fetch):
        """Simulates parsing a product review from rtings.com."""
        review_text = (
            "The Sony WH-1000XM5 wireless headphones feature excellent active noise "
            "cancellation. The overall build quality is premium with a comfortable fit. "
            "Battery life is exceptional at around 30 hours. Sound quality is balanced "
            "with slightly boosted bass. The multi-point connection works seamlessly. "
        ) * 10  # Make it > 200 chars

        mock_fetch.return_value = "<html>" + review_text + "</html>"
        mock_extract.return_value = FakeDocument(
            text=review_text,
            title="Sony WH-1000XM5 Review",
            author="RTINGS Team",
            date="2026-02-01",
            sitename="RTINGS.com",
            language="en",
        )

        scraper = WebScraper()
        result = scraper.parse_url("https://www.rtings.com/headphones/reviews/sony/wh-1000xm5")

        assert result is not None
        assert "noise cancellation" in result.text
        assert result.title == "Sony WH-1000XM5 Review"
        assert result.sitename == "RTINGS.com"
        assert result.word_count > 0

    @patch("reviewmind.scrapers.web.trafilatura.fetch_url")
    def test_paywall_page(self, mock_fetch):
        """Simulates a page behind a paywall (download succeeds but extraction fails)."""
        mock_fetch.return_value = "<html><body>Subscribe to read</body></html>"

        with patch("reviewmind.scrapers.web.trafilatura.bare_extraction", return_value=None):
            scraper = WebScraper()
            result = scraper.parse_url("https://paywall-site.com/article")
            assert result is None

    @patch("reviewmind.scrapers.web.trafilatura.fetch_url")
    def test_unavailable_page(self, mock_fetch):
        """Simulates a page that returns None (404, network error)."""
        mock_fetch.return_value = None
        scraper = WebScraper()
        result = scraper.parse_url("https://example.com/nonexistent")
        assert result is None

    @patch("reviewmind.scrapers.web.trafilatura.fetch_url")
    @patch("reviewmind.scrapers.web.trafilatura.bare_extraction")
    def test_wirecutter_style_review(self, mock_extract, mock_fetch):
        """Simulates parsing a Wirecutter review."""
        text = (
            "After spending 100 hours testing 25 different headphones, we think "
            "the Sony WH-1000XM5 is the best wireless noise-cancelling headphone "
            "for most people. It has the best noise cancellation, great sound, "
            "and excellent battery life. It is comfortable enough to wear all day. "
        ) * 8

        mock_fetch.return_value = "<html>" + text + "</html>"
        mock_extract.return_value = FakeDocument(
            text=text,
            title="The Best Wireless Headphones",
            author="Lauren Dragan",
            date="2026-01-20",
            sitename="Wirecutter",
            description="Expert recommendations for wireless headphones",
            language="en",
        )

        scraper = WebScraper()
        result = scraper.parse_url("https://www.nytimes.com/wirecutter/reviews/best-headphones/")

        assert result is not None
        assert result.author == "Lauren Dragan"
        assert result.sitename == "Wirecutter"
        assert result.word_count > 0

    @patch("reviewmind.scrapers.web.trafilatura.fetch_url")
    @patch("reviewmind.scrapers.web.trafilatura.bare_extraction")
    def test_russian_review(self, mock_extract, mock_fetch):
        """Simulates parsing a Russian-language review (e.g. 4PDA)."""
        text = (
            "Обзор Sony WH-1000XM5. Наушники получили обновлённый дизайн с более "
            "тонким оголовьем и мягкими амбушюрами. Шумоподавление стало ещё лучше "
            "благодаря новым микрофонам. Автономность составляет около 30 часов при "
            "средней громкости. Качество звука порадовало: детальные верхние частоты "
            "и глубокий бас без перебора. Поддержка мультипоинта — подключение к двум "
            "устройствам одновременно. "
        ) * 5

        mock_fetch.return_value = "<html>" + text + "</html>"
        mock_extract.return_value = FakeDocument(
            text=text,
            title="Обзор Sony WH-1000XM5",
            author="Редакция 4PDA",
            sitename="4PDA",
            language="ru",
        )

        scraper = WebScraper()
        result = scraper.parse_url("https://4pda.to/2026/01/15/sony-xm5-review/")

        assert result is not None
        assert "шумоподавление" in result.text.lower()
        assert result.language == "ru"
        assert result.sitename == "4PDA"

    @patch("reviewmind.scrapers.web.trafilatura.fetch_url")
    @patch("reviewmind.scrapers.web.trafilatura.bare_extraction")
    def test_custom_min_text_length(self, mock_extract, mock_fetch):
        """Custom min_text_length filters shorter texts."""
        text = "This is a short article with not much content but enough to test."
        mock_fetch.return_value = "<html>" + text + "</html>"
        mock_extract.return_value = FakeDocument(text=text)

        # Default min_text_length=200 should reject this
        scraper = WebScraper()
        assert scraper.parse_url("https://example.com") is None

        # Lower threshold should accept it
        scraper2 = WebScraper(min_text_length=10)
        result = scraper2.parse_url("https://example.com")
        assert result is not None

    def test_no_html_tags_in_output(self):
        """Verify that _clean_text does not output HTML tags."""
        text_with_noise = "Hello <b>world</b>   some\t\ttext\n\n\n\nend"
        cleaned = WebScraper._clean_text(text_with_noise)
        # Note: HTML tag stripping is trafilatura's job. _clean_text focuses on whitespace.
        # But we verify whitespace cleaning works.
        assert "   " not in cleaned
        assert "\t" not in cleaned
        assert "\n\n\n" not in cleaned
