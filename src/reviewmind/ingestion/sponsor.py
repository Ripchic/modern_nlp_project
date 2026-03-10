"""reviewmind.ingestion.sponsor — Sponsored content detection.

Uses regex heuristics to detect sponsored / promotional markers in both
English and Russian text.  The main entry-point is ``detect_sponsor``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# English sponsor patterns
# ---------------------------------------------------------------------------
_EN_PATTERNS: list[re.Pattern[str]] = [
    # Explicit sponsorship
    re.compile(r"\bsponsored\s+by\b", re.IGNORECASE),
    re.compile(
        r"\bthis\s+(?:video|episode|segment|review|content)\s+is\s+"
        r"(?:brought\s+to\s+you\s+by|sponsored\s+by|paid\s+for\s+by)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bbrought\s+to\s+you\s+by\b", re.IGNORECASE),
    re.compile(r"\bin\s+(?:partnership|collaboration)\s+with\b", re.IGNORECASE),
    re.compile(
        r"\b(?:paid|sponsored)\s+(?:partnership|promotion|collaboration|advertisement|content)\b",
        re.IGNORECASE,
    ),
    # Promo codes / affiliate links
    re.compile(r"\b(?:promo|coupon|discount)\s*code\b", re.IGNORECASE),
    re.compile(r"\buse\s+(?:my\s+)?(?:code|link)\b", re.IGNORECASE),
    re.compile(r"\baffiliate\s+link", re.IGNORECASE),
    re.compile(r"\b(?:special|exclusive)\s+(?:offer|deal|discount)\b", re.IGNORECASE),
    re.compile(r"\bcheck\s+(?:out\s+)?(?:the\s+)?link\s+(?:in\s+(?:the\s+)?description|below)\b", re.IGNORECASE),
    # Generic ad markers
    re.compile(r"(?:^|\s)#(?:ad|sponsored|promo)(?:\s|$)", re.IGNORECASE),
    re.compile(r"\b(?:ad|advertisement)\s*(?:\||:|-)\b", re.IGNORECASE),
    re.compile(r"\bthanks?\s+to\s+\S+\s+for\s+sponsoring\b", re.IGNORECASE),
]

# ---------------------------------------------------------------------------
# Russian sponsor patterns
# ---------------------------------------------------------------------------
_RU_PATTERNS: list[re.Pattern[str]] = [
    # Sponsorship
    re.compile(r"\bспонсор(?:ом|ами|ов|ский|ская|ское|ские)?\b", re.IGNORECASE),
    re.compile(r"\bреклама\b", re.IGNORECASE),
    re.compile(r"\bрекламн(?:ый|ая|ое|ые|ой|ого)\b", re.IGNORECASE),
    re.compile(r"\bрекламная\s+интеграция\b", re.IGNORECASE),
    re.compile(r"\bблагодар(?:им|ю|ит)\s+(?:спонсор|партнёр)", re.IGNORECASE),
    re.compile(r"\bпри\s+(?:поддержке|участии)\b", re.IGNORECASE),
    # Promo codes
    re.compile(r"\bпромокод\b", re.IGNORECASE),
    re.compile(r"\bпромо[- ]?код\b", re.IGNORECASE),
    re.compile(r"\bкупон\b", re.IGNORECASE),
    re.compile(r"\bскидк[аиуе]\s+по\s+(?:ссылке|коду|промокоду)\b", re.IGNORECASE),
    re.compile(r"\bссылк[аиуе]\s+в\s+описании\b", re.IGNORECASE),
    # Partnership
    re.compile(r"\bпартнёрск(?:ий|ая|ое|ие)\b", re.IGNORECASE),
    re.compile(
        r"\b(?:оплачен(?:ный|ная|ное|ные)|заказн(?:ой|ая|ое|ые))\s+(?:обзор|контент|материал|видео|ролик)\b",
        re.IGNORECASE,
    ),
]

ALL_PATTERNS: list[re.Pattern[str]] = _EN_PATTERNS + _RU_PATTERNS


@dataclass(frozen=True)
class SponsorDetectionResult:
    """Result of sponsor detection."""

    is_sponsored: bool
    matched_patterns: list[str] = field(default_factory=list)
    confidence: float = 0.0  # 0.0 – 1.0


def detect_sponsor(text: str, *, threshold: int = 1) -> bool:
    """Detect whether *text* contains sponsored/promotional content.

    Parameters
    ----------
    text:
        The text to analyse.
    threshold:
        Minimum number of distinct pattern matches required to consider
        the text as sponsored.  Default is **1** (any single match is
        sufficient).

    Returns
    -------
    bool
        ``True`` if the text is considered sponsored.
    """
    if not text or not text.strip():
        return False

    matches = _find_matches(text)
    is_sponsored = len(matches) >= threshold

    if is_sponsored:
        logger.info(
            "sponsor_detected",
            match_count=len(matches),
            patterns=matches[:5],  # first 5 for brevity
        )

    return is_sponsored


def detect_sponsor_detailed(text: str, *, threshold: int = 1) -> SponsorDetectionResult:
    """Like :func:`detect_sponsor` but returns full detection details.

    Returns
    -------
    SponsorDetectionResult
        Contains ``is_sponsored``, ``matched_patterns`` and a rough
        ``confidence`` score (0–1).
    """
    if not text or not text.strip():
        return SponsorDetectionResult(is_sponsored=False)

    matches = _find_matches(text)
    is_sponsored = len(matches) >= threshold

    # Rough confidence: saturates at 4 distinct matches → 1.0
    confidence = min(len(matches) / 4, 1.0) if matches else 0.0

    return SponsorDetectionResult(
        is_sponsored=is_sponsored,
        matched_patterns=matches,
        confidence=confidence,
    )


def _find_matches(text: str) -> list[str]:
    """Return a list of unique pattern labels that matched *text*."""
    seen: set[str] = set()
    matches: list[str] = []

    for pattern in ALL_PATTERNS:
        m = pattern.search(text)
        if m:
            matched_text = m.group(0).strip()
            if matched_text.lower() not in seen:
                seen.add(matched_text.lower())
                matches.append(matched_text)

    return matches
