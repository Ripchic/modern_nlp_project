"""reviewmind.ingestion — Text cleaning, sponsor detection, chunking and ingestion pipeline."""

from reviewmind.ingestion.cleaner import MIN_CLEAN_LENGTH, clean_text
from reviewmind.ingestion.sponsor import (
    ALL_PATTERNS,
    SponsorDetectionResult,
    detect_sponsor,
    detect_sponsor_detailed,
)

__all__ = [
    "ALL_PATTERNS",
    "MIN_CLEAN_LENGTH",
    "SponsorDetectionResult",
    "clean_text",
    "detect_sponsor",
    "detect_sponsor_detailed",
]
