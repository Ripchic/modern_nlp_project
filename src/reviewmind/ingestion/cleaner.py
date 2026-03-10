"""reviewmind.ingestion.cleaner — Text cleaning and normalization.

Provides ``clean_text`` for preparing raw scraped content before chunking
and embedding.  The function removes HTML tags, timestamps, excessive
whitespace, normalises Unicode (NFKC), strips zero-width characters and
other common artefacts found in YouTube transcripts, Reddit posts and web
page extractions.
"""

from __future__ import annotations

import html
import re
import unicodedata

import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------

# HTML tags (including self-closing and comments)
_HTML_TAG_RE = re.compile(r"<[^>]+>", re.DOTALL)
_HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)

# Timestamps — common formats from YouTube / SRT / VTT
# Examples: 00:01:23, 0:12, 12:34:56.789, [00:01], (12:34)
_TIMESTAMP_RE = re.compile(
    r"[\[\(]?\d{0,2}:?\d{1,2}:\d{2}(?:[.,]\d{1,3})?[\]\)]?"
)

# YouTube auto-caption artefacts: [Music], [Applause], [Laughter], etc.
_BRACKET_ARTEFACT_RE = re.compile(
    r"\[(?:Music|Applause|Laughter|Silence|Inaudible|Музыка|Аплодисменты|Смех)\]",
    re.IGNORECASE,
)

# URL patterns (http/https/ftp)
_URL_RE = re.compile(r"https?://\S+|ftp://\S+", re.IGNORECASE)

# Email addresses
_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)

# Zero-width and invisible Unicode characters
_INVISIBLE_CHARS_RE = re.compile(
    r"[\u200b\u200c\u200d\u200e\u200f\ufeff\u00ad\u2060\u2061\u2062\u2063\u2064]"
)

# Runs of whitespace (spaces and tabs only — newlines handled separately)
_HORIZONTAL_WS_RE = re.compile(r"[^\S\n]+")

# Three or more consecutive newlines → collapse to two
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")

# Markdown-style image references: ![alt](url)
_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")

# Markdown-style links: [text](url) → keep text
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")

# Common ad / navigation fragments (very conservative)
_NAV_FRAGMENTS_RE = re.compile(
    r"(?:^|\n)"
    r"(?:Share|Tweet|Pin|Subscribe|Follow us|Подписаться|Поделиться|"
    r"Table of Contents|Содержание|Advertisement|Реклама)"
    r"(?:\n|$)",
    re.IGNORECASE,
)

# Repeated punctuation (e.g. "!!!!!!" → "!")
_REPEATED_PUNCT_RE = re.compile(r"([!?.])\1{2,}")

# Minimum meaningful text length (characters) after cleaning
MIN_CLEAN_LENGTH = 50


def clean_text(raw_text: str, *, remove_urls: bool = True, remove_emails: bool = True) -> str:
    """Clean and normalise *raw_text* for downstream processing.

    Parameters
    ----------
    raw_text:
        The raw text obtained from a scraper.
    remove_urls:
        If ``True`` (default), strip HTTP/FTP URLs.
    remove_emails:
        If ``True`` (default), strip email addresses.

    Returns
    -------
    str
        Cleaned text, or an empty string if the result is below
        ``MIN_CLEAN_LENGTH``.
    """
    if not raw_text or not raw_text.strip():
        return ""

    text = raw_text

    # 1. Unicode normalisation (NFKC — compatibility decomposition + canonical composition)
    text = unicodedata.normalize("NFKC", text)

    # 2. Decode HTML entities (&amp; → &, etc.)
    text = html.unescape(text)

    # 3. Remove HTML comments first, then tags
    text = _HTML_COMMENT_RE.sub("", text)
    text = _HTML_TAG_RE.sub("", text)

    # 4. Remove invisible / zero-width characters
    text = _INVISIBLE_CHARS_RE.sub("", text)

    # 5. Remove bracket artefacts (e.g. [Music])
    text = _BRACKET_ARTEFACT_RE.sub("", text)

    # 6. Remove timestamps
    text = _TIMESTAMP_RE.sub("", text)

    # 7. Remove markdown images
    text = _MD_IMAGE_RE.sub("", text)

    # 8. Convert markdown links to plain text
    text = _MD_LINK_RE.sub(r"\1", text)

    # 9. Optionally strip URLs and emails
    if remove_urls:
        text = _URL_RE.sub("", text)
    if remove_emails:
        text = _EMAIL_RE.sub("", text)

    # 10. Remove common nav/ad fragments
    text = _NAV_FRAGMENTS_RE.sub("\n", text)

    # 11. Collapse repeated punctuation
    text = _REPEATED_PUNCT_RE.sub(r"\1", text)

    # 12. Normalise horizontal whitespace (spaces/tabs) — preserve newlines
    text = _HORIZONTAL_WS_RE.sub(" ", text)

    # 13. Collapse excessive newlines (3+ → 2)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)

    # 14. Strip leading/trailing whitespace per line and globally
    text = "\n".join(line.strip() for line in text.splitlines())
    text = text.strip()

    # 15. Final length check
    if len(text) < MIN_CLEAN_LENGTH:
        logger.warning(
            "cleaned_text_too_short",
            original_len=len(raw_text),
            cleaned_len=len(text),
            min_len=MIN_CLEAN_LENGTH,
        )
        return ""

    return text
