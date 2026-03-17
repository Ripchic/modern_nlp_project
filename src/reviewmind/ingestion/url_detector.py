"""reviewmind/ingestion/url_detector.py — URL type detection and scraper routing.

Determines the type of a URL (YouTube, Reddit, or generic web) and routes
it to the appropriate scraper instance.
"""

from __future__ import annotations

import re
from typing import Union
from urllib.parse import urlparse

import structlog

from reviewmind.scrapers.reddit import RedditScraper
from reviewmind.scrapers.web import WebScraper
from reviewmind.scrapers.youtube import YouTubeScraper
from reviewmind.vectorstore.collections import SourceType

logger = structlog.get_logger(__name__)

# ── URL type patterns ─────────────────────────────────────────

_YOUTUBE_RE = re.compile(
    r"(?:youtube\.com|youtu\.be)",
    re.IGNORECASE,
)

_REDDIT_RE = re.compile(
    r"(?:reddit\.com|redd\.it)",
    re.IGNORECASE,
)

# Allowed schemes for URL validation.
_ALLOWED_SCHEMES = frozenset({"http", "https"})

# Type alias for scraper instances returned by route_to_scraper.
ScraperType = Union[YouTubeScraper, RedditScraper, WebScraper]


# ── URL validation ────────────────────────────────────────────


def _validate_url(url: str) -> str:
    """Validate that *url* is a well-formed HTTP(S) URL.

    Returns the normalised URL string.
    Raises :class:`ValueError` for invalid or non-HTTP(S) URLs.
    """
    if not url or not isinstance(url, str):
        raise ValueError(f"Invalid URL: {url!r} — URL must be a non-empty string")

    url = url.strip()
    parsed = urlparse(url)

    if not parsed.scheme:
        raise ValueError(f"Invalid URL: {url!r} — missing scheme (expected http or https)")

    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise ValueError(f"Invalid URL: {url!r} — unsupported scheme '{parsed.scheme}' (expected http or https)")

    if not parsed.netloc:
        raise ValueError(f"Invalid URL: {url!r} — missing hostname")

    return url


# ── Public API ────────────────────────────────────────────────


def detect_url_type(url: str) -> str:
    """Detect the source type for the given URL.

    Parameters
    ----------
    url:
        A valid HTTP(S) URL string.

    Returns
    -------
    str
        One of ``'youtube'``, ``'reddit'``, or ``'web'``.

    Raises
    ------
    ValueError
        If *url* is not a valid HTTP(S) URL.
    """
    validated = _validate_url(url)

    if _YOUTUBE_RE.search(validated):
        return SourceType.YOUTUBE.value
    if _REDDIT_RE.search(validated):
        return SourceType.REDDIT.value
    return SourceType.WEB.value


def route_to_scraper(url: str) -> ScraperType:
    """Create and return the appropriate scraper for *url*.

    Parameters
    ----------
    url:
        A valid HTTP(S) URL string.

    Returns
    -------
    YouTubeScraper | RedditScraper | WebScraper
        A new scraper instance matching the URL type.

    Raises
    ------
    ValueError
        If *url* is not a valid HTTP(S) URL.
    """
    url_type = detect_url_type(url)

    if url_type == SourceType.YOUTUBE.value:
        return YouTubeScraper()
    if url_type == SourceType.REDDIT.value:
        return RedditScraper()
    return WebScraper()
