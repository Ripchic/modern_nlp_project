"""reviewmind/scrapers/web.py — Web page parser (trafilatura).

Extracts the main article text and metadata from arbitrary web pages using
the *trafilatura* library.  Handles timeouts, empty results, paywall pages
and network errors gracefully — returning ``None`` instead of raising.
"""

from __future__ import annotations

import re
import signal
from dataclasses import dataclass, field
from typing import Any

import structlog
import trafilatura

logger = structlog.get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────

DEFAULT_TIMEOUT: int = 30
"""Maximum number of seconds to wait for a page download."""

MIN_TEXT_LENGTH: int = 200
"""Minimum character length of extracted text; shorter pages are skipped."""


# ── Result dataclass ──────────────────────────────────────────


@dataclass
class WebPage:
    """Result of a web page extraction."""

    url: str
    text: str
    title: str | None = None
    author: str | None = None
    date: str | None = None
    sitename: str | None = None
    description: str | None = None
    language: str | None = None
    word_count: int = 0
    extra_metadata: dict[str, Any] = field(default_factory=dict)


# ── Timeout helper ────────────────────────────────────────────


class _TimeoutError(Exception):
    """Raised when a page download exceeds the configured timeout."""


def _timeout_handler(_signum: int, _frame: Any) -> None:  # pragma: no cover
    raise _TimeoutError("Page download timed out")


# ── Scraper class ─────────────────────────────────────────────


class WebScraper:
    """Extracts main article text from web pages using *trafilatura*.

    Usage::

        scraper = WebScraper()
        result = scraper.parse_url("https://www.rtings.com/headphones/reviews/sony/wh-1000xm5")
        if result:
            print(result.text)
    """

    def __init__(
        self,
        *,
        timeout: int = DEFAULT_TIMEOUT,
        min_text_length: int = MIN_TEXT_LENGTH,
        favor_precision: bool = False,
        favor_recall: bool = True,
        include_comments: bool = False,
        include_tables: bool = True,
    ) -> None:
        self._timeout = timeout
        self._min_text_length = min_text_length
        self._favor_precision = favor_precision
        self._favor_recall = favor_recall
        self._include_comments = include_comments
        self._include_tables = include_tables

    # ── Public API ────────────────────────────────────────────

    def parse_url(self, url: str) -> WebPage | None:
        """Download *url* and extract the main text content.

        Args:
            url: An HTTP(S) URL to download and parse.

        Returns:
            A :class:`WebPage` with extracted text and metadata, or ``None``
            when the page cannot be fetched, is behind a paywall, or yields
            too little content (< ``min_text_length`` characters).
        """
        if not self._validate_url(url):
            return None

        url = url.strip()

        html = self._download(url)
        if html is None:
            return None

        return self._extract(html, url)

    def parse_html(self, html: str, *, url: str = "") -> WebPage | None:
        """Extract text from pre-downloaded HTML.

        Useful when the page has already been downloaded by another mechanism
        (e.g. a headless browser).

        Args:
            html: Raw HTML string.
            url: Optional URL hint (used by trafilatura for metadata / link resolution).

        Returns:
            A :class:`WebPage`, or ``None`` on empty/too-short extraction.
        """
        if not html or not isinstance(html, str):
            logger.warning("web.empty_html", url=url)
            return None

        return self._extract(html, url)

    @staticmethod
    def is_web_url(url: str) -> bool:
        """Return ``True`` if *url* looks like a valid HTTP(S) URL.

        Does **not** check YouTube or Reddit URLs — those should be handled
        by their dedicated scrapers.
        """
        if not url or not isinstance(url, str):
            return False
        url = url.strip()
        return bool(re.match(r"https?://[^\s/$.?#].[^\s]*$", url, re.IGNORECASE))

    # ── Internal helpers ──────────────────────────────────────

    def _validate_url(self, url: str) -> bool:
        """Ensure *url* is a non-empty HTTP(S) string."""
        if not url or not isinstance(url, str):
            logger.warning("web.invalid_url", url=url)
            return False
        url = url.strip()
        if not re.match(r"https?://", url, re.IGNORECASE):
            logger.warning("web.invalid_url_scheme", url=url)
            return False
        return True

    def _download(self, url: str) -> str | None:
        """Download *url* with a timeout guard.

        Returns the raw HTML string or ``None`` on failure.
        """
        try:
            # Use signal-based timeout on Unix (macOS/Linux).
            # Falls back to trafilatura defaults on unsupported platforms.
            use_signal_timeout = hasattr(signal, "SIGALRM")
            if use_signal_timeout:
                old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(self._timeout)

            try:
                html = trafilatura.fetch_url(url)
            finally:
                if use_signal_timeout:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)  # type: ignore[arg-type]

        except _TimeoutError:
            logger.warning("web.download_timeout", url=url, timeout=self._timeout)
            return None
        except Exception:
            logger.exception("web.download_error", url=url)
            return None

        if not html:
            logger.info("web.empty_download", url=url)
            return None

        return html

    def _extract(self, html: str, url: str) -> WebPage | None:
        """Run trafilatura extraction on *html* and build a :class:`WebPage`."""
        try:
            result = trafilatura.bare_extraction(
                html,
                url=url or None,
                favor_precision=self._favor_precision,
                favor_recall=self._favor_recall,
                include_comments=self._include_comments,
                include_tables=self._include_tables,
                include_formatting=False,
                include_links=False,
                with_metadata=True,
            )
        except Exception:
            logger.exception("web.extraction_error", url=url)
            return None

        if result is None:
            logger.info("web.no_content_extracted", url=url)
            return None

        # trafilatura.bare_extraction returns a Document object with attributes
        text = self._get_text(result)

        if not text:
            logger.info("web.empty_text", url=url)
            return None

        # Clean the text
        text = self._clean_text(text)

        if len(text) < self._min_text_length:
            logger.info(
                "web.text_too_short",
                url=url,
                text_length=len(text),
                min_required=self._min_text_length,
            )
            return None

        title = self._get_attr(result, "title")
        author = self._get_attr(result, "author")
        date = self._get_attr(result, "date")
        sitename = self._get_attr(result, "sitename")
        description = self._get_attr(result, "description")
        language = self._get_attr(result, "language")

        word_count = len(text.split())

        return WebPage(
            url=url,
            text=text,
            title=title,
            author=author,
            date=date,
            sitename=sitename,
            description=description,
            language=language,
            word_count=word_count,
        )

    # ── Text helpers ──────────────────────────────────────────

    @staticmethod
    def _get_text(result: Any) -> str:
        """Extract text from a trafilatura result (Document or dict)."""
        # bare_extraction returns a Document object with .text attribute
        if hasattr(result, "text"):
            return result.text or ""
        # Fallback for dict-style results
        if isinstance(result, dict):
            return result.get("text", "") or ""
        return ""

    @staticmethod
    def _get_attr(result: Any, attr: str) -> str | None:
        """Safely extract a metadata attribute from a trafilatura result."""
        val = None
        if hasattr(result, attr):
            val = getattr(result, attr, None)
        elif isinstance(result, dict):
            val = result.get(attr)
        if val and isinstance(val, str) and val.strip():
            return val.strip()
        return None

    @staticmethod
    def _clean_text(text: str) -> str:
        """Post-process extracted text.

        * Collapse multiple blank lines to a single blank line.
        * Collapse runs of spaces (not newlines) within a line.
        * Strip leading/trailing whitespace.
        """
        # Collapse runs of horizontal whitespace (spaces/tabs) into one space
        text = re.sub(r"[^\S\n]+", " ", text)
        # Collapse 3+ newlines into 2 (one blank line)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
