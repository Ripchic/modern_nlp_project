"""reviewmind/services/language.py — Language detection via langdetect.

Provides :func:`detect_language` for automatic detection of the user's
query language.  Falls back to ``'ru'`` (primary market) when detection
is unreliable (empty text, very short input, ambiguous).
"""

from __future__ import annotations

import structlog
from langdetect import DetectorFactory, LangDetectException, detect

logger = structlog.get_logger("reviewmind.services.language")

# ── Constants ────────────────────────────────────────────────────────────────

#: Fallback language when detection is unreliable or impossible.
FALLBACK_LANGUAGE: str = "ru"

#: Minimum number of characters required for reliable detection.
#: Texts shorter than this return :data:`FALLBACK_LANGUAGE`.
MIN_TEXT_LENGTH: int = 4

#: Supported languages that we pass through unchanged.
#: Detected languages outside this set are mapped to :data:`FALLBACK_LANGUAGE`.
SUPPORTED_LANGUAGES: frozenset[str] = frozenset(
    {"ru", "en", "uk", "de", "fr", "es", "it", "pt", "ja", "ko", "zh-cn", "zh-tw"}
)

# Make langdetect deterministic (same input → same output).
DetectorFactory.seed = 0

# ── Public API ───────────────────────────────────────────────────────────────


def detect_language(text: str) -> str:
    """Detect the language of *text*.

    Parameters
    ----------
    text:
        The user's input query.  May be empty, very short, or contain
        mixed languages.

    Returns
    -------
    str
        ISO 639-1 language code (e.g. ``'ru'``, ``'en'``).
        Falls back to :data:`FALLBACK_LANGUAGE` when:
        * *text* is empty or whitespace-only,
        * *text* is shorter than :data:`MIN_TEXT_LENGTH` characters,
        * ``langdetect`` raises an exception,
        * the detected language is not in :data:`SUPPORTED_LANGUAGES`.
    """
    if not text or not text.strip():
        logger.debug("language_detect_empty_input", fallback=FALLBACK_LANGUAGE)
        return FALLBACK_LANGUAGE

    cleaned = text.strip()

    if len(cleaned) < MIN_TEXT_LENGTH:
        logger.debug(
            "language_detect_too_short",
            length=len(cleaned),
            min_required=MIN_TEXT_LENGTH,
            fallback=FALLBACK_LANGUAGE,
        )
        return FALLBACK_LANGUAGE

    try:
        lang = detect(cleaned)
    except LangDetectException:
        logger.debug("language_detect_exception", fallback=FALLBACK_LANGUAGE)
        return FALLBACK_LANGUAGE

    if lang not in SUPPORTED_LANGUAGES:
        logger.debug(
            "language_detect_unsupported",
            detected=lang,
            fallback=FALLBACK_LANGUAGE,
        )
        return FALLBACK_LANGUAGE

    logger.debug("language_detected", language=lang, text_length=len(cleaned))
    return lang
