"""reviewmind/scrapers/youtube.py — YouTube transcript parser.

Extracts transcripts from YouTube videos using youtube-transcript-api.
Supports multiple URL formats: youtube.com/watch?v=, youtu.be/, youtube.com/shorts/.
Prioritises Russian ('ru') and English ('en') transcripts.
Filters out videos with fewer than MIN_WORD_COUNT words.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import ClassVar

import structlog
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

logger = structlog.get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────

MIN_WORD_COUNT: int = 500
"""Minimum number of words required in a transcript; shorter transcripts are skipped."""

DEFAULT_LANGUAGES: tuple[str, ...] = ("ru", "en")
"""Preferred transcript languages, tried in order."""

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
    ) -> None:
        self._languages = tuple(languages)
        self._min_word_count = min_word_count
        self._api = YouTubeTranscriptApi()

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
        """Convenience method: extract video ID from *url* and fetch transcript.

        Args:
            url: A YouTube video URL.
            languages: Override default language preference.

        Returns:
            A :class:`TranscriptResult`, or ``None`` if the URL is invalid
            or the transcript is unavailable/too short.

        Raises:
            ValueError: If *url* is not a recognised YouTube URL format.
        """
        video_id = self.extract_video_id(url)
        result = self.get_transcript(video_id, languages=languages)
        if result is not None:
            result.source_url = url.strip()
        return result

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
