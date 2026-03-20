"""reviewmind/scrapers/youtube.py — YouTube transcript parser + search.

Extracts transcripts from YouTube videos using youtube-transcript-api.
Supports multiple URL formats: youtube.com/watch?v=, youtu.be/, youtube.com/shorts/.
Prioritises Russian ('ru') and English ('en') transcripts.
Filters out videos with fewer than MIN_WORD_COUNT words.

Also provides search_videos() via YouTube Data API v3 for auto-mode discovery.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from http.cookiejar import MozillaCookieJar
from pathlib import Path
from typing import ClassVar

import httpx
import structlog
from requests import Session
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
)

logger = structlog.get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────

MIN_WORD_COUNT: int = 500
"""Minimum number of words required in a transcript; shorter transcripts are skipped."""

DEFAULT_LANGUAGES: tuple[str, ...] = ("ru", "en")
"""Preferred transcript languages, tried in order."""

DEFAULT_SEARCH_MAX_RESULTS: int = 10
"""Default maximum number of video search results."""

YOUTUBE_API_BASE_URL: str = "https://www.googleapis.com/youtube/v3"
"""Base URL for YouTube Data API v3."""

MAX_AGE_DAYS: int = 365
"""Maximum age of search results in days (1 year)."""

# ── URL patterns ──────────────────────────────────────────────

# Captures the 11-character video ID from various YouTube URL formats.
_YOUTUBE_URL_PATTERNS: list[re.Pattern[str]] = [
    # Standard: https://www.youtube.com/watch?v=VIDEO_ID
    re.compile(
        r"(?:https?://)?(?:www\.)?youtube\.com/watch\?(?:[^&]*&)*v=(?P<id>[A-Za-z0-9_-]{11})"
    ),
    # Short: https://youtu.be/VIDEO_ID
    re.compile(r"(?:https?://)?youtu\.be/(?P<id>[A-Za-z0-9_-]{11})"),
    # Shorts: https://www.youtube.com/shorts/VIDEO_ID
    re.compile(
        r"(?:https?://)?(?:www\.)?youtube\.com/shorts/(?P<id>[A-Za-z0-9_-]{11})"
    ),
    # Embed: https://www.youtube.com/embed/VIDEO_ID
    re.compile(
        r"(?:https?://)?(?:www\.)?youtube\.com/embed/(?P<id>[A-Za-z0-9_-]{11})"
    ),
    # Live: https://www.youtube.com/live/VIDEO_ID
    re.compile(
        r"(?:https?://)?(?:www\.)?youtube\.com/live/(?P<id>[A-Za-z0-9_-]{11})"
    ),
]


# ── Result dataclass ──────────────────────────────────────────


@dataclass
class TranscriptResult:
    """Result of a transcript fetch."""

    video_id: str
    text: str
    language: str
    language_code: str
    is_generated: bool
    word_count: int
    snippet_count: int

    # Optional metadata from URL
    source_url: str = ""
    extra_metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class VideoInfo:
    """Summary of a YouTube video found through search."""

    video_id: str
    title: str
    channel_title: str
    published_at: str
    description: str
    url: str


# ── Scraper class ─────────────────────────────────────────────


class YouTubeScraper:
    """Fetches and cleans YouTube video transcripts.

    Usage::

        scraper = YouTubeScraper()
        result = scraper.get_transcript_by_url("https://youtube.com/watch?v=dQw4w9WgXcQ")
        if result:
            print(result.text)

    Can also be used with raw video IDs::

        result = scraper.get_transcript("dQw4w9WgXcQ")
    """

    # Shared across instances — no instance-level API state modification needed.
    _PATTERNS: ClassVar[list[re.Pattern[str]]] = _YOUTUBE_URL_PATTERNS

    def __init__(
        self,
        *,
        languages: tuple[str, ...] | list[str] = DEFAULT_LANGUAGES,
        min_word_count: int = MIN_WORD_COUNT,
        cookie_path: str | Path | None = None,
    ) -> None:
        self._languages = tuple(languages)
        self._min_word_count = min_word_count
        self._api = self._build_api(cookie_path)

    @staticmethod
    def _build_api(cookie_path: str | Path | None) -> YouTubeTranscriptApi:
        """Create API instance, optionally loading cookies for auth."""
        if cookie_path is not None:
            path = Path(cookie_path)
            if path.is_file():
                jar = MozillaCookieJar(str(path))
                jar.load(ignore_discard=True, ignore_expires=True)
                session = Session()
                session.cookies = jar
                session.headers.update({
                    "User-Agent": (
                        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/134.0.0.0 Safari/537.36"
                    ),
                    "Accept-Language": "ru-RU,ru;q=0.9,en-US;q=0.8,en;q=0.7",
                })
                logger.info("youtube.cookies_loaded", path=str(path), count=len(jar))
                return YouTubeTranscriptApi(http_client=session)
            logger.warning("youtube.cookies_not_found", path=str(path))
        return YouTubeTranscriptApi()

    # ── Public API ────────────────────────────────────────────

    @staticmethod
    def extract_video_id(url: str) -> str:
        """Extract the 11-character video ID from a YouTube URL.

        Supports ``youtube.com/watch?v=``, ``youtu.be/``, ``youtube.com/shorts/``,
        ``youtube.com/embed/`` and ``youtube.com/live/`` formats.

        Args:
            url: A YouTube video URL.

        Returns:
            The 11-character video ID string.

        Raises:
            ValueError: If the URL does not match any known YouTube format.
        """
        if not url or not isinstance(url, str):
            raise ValueError(f"Invalid YouTube URL: {url!r}")

        url = url.strip()
        for pattern in _YOUTUBE_URL_PATTERNS:
            match = pattern.search(url)
            if match:
                return match.group("id")

        raise ValueError(f"Could not extract video ID from URL: {url!r}")

    def get_transcript(
        self,
        video_id: str,
        *,
        languages: tuple[str, ...] | list[str] | None = None,
    ) -> TranscriptResult | None:
        """Fetch transcript for a YouTube video by its ID.

        The transcript text is cleaned: timestamps are removed, segments are
        joined with spaces and normalised.

        Args:
            video_id: The 11-character YouTube video ID.
            languages: Override default language preference (e.g. ``("ru", "en")``).

        Returns:
            A :class:`TranscriptResult` on success, or ``None`` if the transcript
            is unavailable or too short (< ``min_word_count`` words).
        """
        if not video_id or not isinstance(video_id, str):
            logger.warning("youtube.invalid_video_id", video_id=video_id)
            return None

        video_id = video_id.strip()
        langs = languages or self._languages

        try:
            fetched = self._api.fetch(video_id, languages=langs)
        except TranscriptsDisabled:
            logger.info("youtube.transcripts_disabled", video_id=video_id)
            return None
        except NoTranscriptFound:
            logger.info(
                "youtube.no_transcript_found",
                video_id=video_id,
                languages=langs,
            )
            return None
        except VideoUnavailable:
            logger.info("youtube.video_unavailable", video_id=video_id)
            return None
        except RequestBlocked:
            logger.error(
                "youtube.request_blocked",
                video_id=video_id,
                hint="IP blocked by YouTube. Re-export cookies.txt from browser.",
            )
            return None
        except Exception:
            logger.exception("youtube.unexpected_error", video_id=video_id)
            return None

        # Build clean text from snippets (no timestamps).
        text = self._build_clean_text(fetched.snippets)
        word_count = len(text.split())

        if word_count < self._min_word_count:
            logger.info(
                "youtube.transcript_too_short",
                video_id=video_id,
                word_count=word_count,
                min_required=self._min_word_count,
            )
            return None

        return TranscriptResult(
            video_id=video_id,
            text=text,
            language=fetched.language,
            language_code=fetched.language_code,
            is_generated=fetched.is_generated,
            word_count=word_count,
            snippet_count=len(fetched.snippets),
        )

    def get_transcript_by_url(
        self,
        url: str,
        *,
        languages: tuple[str, ...] | list[str] | None = None,
    ) -> TranscriptResult | None:
        """Convenience method: extract video ID from *url* and fetch transcript."""
        logger.info("youtube.get_transcript_by_url_start", url=url)
        video_id = self.extract_video_id(url)
        logger.info("youtube.video_id_extracted", video_id=video_id, url=url)
        result = self.get_transcript(video_id, languages=languages)
        if result is not None:
            result.source_url = url.strip()
            logger.info(
                "youtube.transcript_ok",
                video_id=video_id,
                word_count=result.word_count,
                language=result.language_code,
                is_generated=result.is_generated,
            )
        else:
            logger.warning("youtube.transcript_failed", video_id=video_id, url=url)
        return result

    def search_videos(
        self,
        query: str,
        *,
        max_results: int = DEFAULT_SEARCH_MAX_RESULTS,
        api_key: str | None = None,
    ) -> list[VideoInfo]:
        """Search YouTube for videos matching *query* via Data API v3.

        Only returns videos published within the last :data:`MAX_AGE_DAYS` days.

        Args:
            query: Search query string (e.g. ``"Sony WH-1000XM5 review"``).
            max_results: Maximum number of results (1–50, default 10).
            api_key: YouTube Data API key. Falls back to ``config.youtube_api_key``.

        Returns:
            List of :class:`VideoInfo` objects. May be shorter than
            *max_results* if fewer matching videos exist.
        """
        if not query or not isinstance(query, str):
            logger.warning("youtube.empty_search_query")
            return []

        query = query.strip()
        if not query:
            return []

        key = api_key
        if not key:
            from reviewmind.config import settings

            key = settings.youtube_api_key

        if not key:
            logger.warning("youtube.no_api_key", hint="Set YOUTUBE_API_KEY in .env")
            return []

        max_results = max(1, min(max_results, 50))

        published_after = _published_after_iso()

        try:
            return self._do_search(query, key, max_results, published_after)
        except httpx.HTTPStatusError as exc:
            logger.error(
                "youtube.search_http_error",
                status_code=exc.response.status_code,
                query=query,
            )
            return []
        except httpx.HTTPError:
            logger.exception("youtube.search_network_error", query=query)
            return []
        except Exception:
            logger.exception("youtube.search_unexpected_error", query=query)
            return []

    # ── Helpers ───────────────────────────────────────────────

    @staticmethod
    def _build_clean_text(snippets: list) -> str:
        """Join snippet texts into a single cleaned string.

        * Removes common timestamp artefacts left in auto-generated captions.
        * Collapses runs of whitespace into a single space.
        * Strips leading/trailing whitespace.
        """
        parts: list[str] = []
        for snippet in snippets:
            text = snippet.text
            if not text:
                continue
            # Some auto-captions insert literal timestamps like [Music], [Applause], etc.
            # Strip common non-speech markers.
            text = re.sub(r"\[.*?\]", "", text)
            # Collapse whitespace.
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                parts.append(text)

        return " ".join(parts)

    @staticmethod
    def is_youtube_url(url: str) -> bool:
        """Return ``True`` if *url* looks like a YouTube video URL."""
        if not url or not isinstance(url, str):
            return False
        for pattern in _YOUTUBE_URL_PATTERNS:
            if pattern.search(url.strip()):
                return True
        return False

    def _do_search(
        self,
        query: str,
        api_key: str,
        max_results: int,
        published_after: str,
    ) -> list[VideoInfo]:
        """Execute the YouTube Data API v3 search request."""
        params = {
            "part": "snippet",
            "q": query,
            "type": "video",
            "maxResults": max_results,
            "order": "relevance",
            "publishedAfter": published_after,
            "key": api_key,
        }
        with httpx.Client(timeout=15.0) as client:
            resp = client.get(f"{YOUTUBE_API_BASE_URL}/search", params=params)
            resp.raise_for_status()

        data = resp.json()
        items = data.get("items", [])

        logger.info(
            "youtube.search_completed",
            query=query,
            results_count=len(items),
        )
        return _parse_search_items(items)


# ── Module-level helpers ──────────────────────────────────────


def _published_after_iso(max_age_days: int = MAX_AGE_DAYS) -> str:
    """Return an ISO 8601 timestamp *max_age_days* in the past."""
    from datetime import timedelta

    dt = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_search_items(items: list[dict]) -> list[VideoInfo]:
    """Convert raw YouTube API search items into :class:`VideoInfo` objects."""
    results: list[VideoInfo] = []
    for item in items:
        video_id = item.get("id", {}).get("videoId")
        if not video_id:
            continue
        snippet = item.get("snippet", {})
        results.append(
            VideoInfo(
                video_id=video_id,
                title=snippet.get("title", ""),
                channel_title=snippet.get("channelTitle", ""),
                published_at=snippet.get("publishedAt", ""),
                description=snippet.get("description", ""),
                url=f"https://www.youtube.com/watch?v={video_id}",
            )
        )
    return results
