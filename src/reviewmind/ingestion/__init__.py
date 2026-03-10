"""reviewmind.ingestion — Text cleaning, sponsor detection, chunking and ingestion pipeline."""

from reviewmind.ingestion.chunker import (
    CHARS_PER_TOKEN,
    CHUNK_OVERLAP_CHARS,
    CHUNK_OVERLAP_TOKENS,
    CHUNK_SIZE_CHARS,
    CHUNK_SIZE_TOKENS,
    DEFAULT_SEPARATORS,
    MIN_CHUNK_LENGTH,
    Chunk,
    chunk_text,
    chunk_text_dicts,
)
from reviewmind.ingestion.cleaner import MIN_CLEAN_LENGTH, clean_text
from reviewmind.ingestion.sponsor import (
    ALL_PATTERNS,
    SponsorDetectionResult,
    detect_sponsor,
    detect_sponsor_detailed,
)

__all__ = [
    "ALL_PATTERNS",
    "CHARS_PER_TOKEN",
    "CHUNK_OVERLAP_CHARS",
    "CHUNK_OVERLAP_TOKENS",
    "CHUNK_SIZE_CHARS",
    "CHUNK_SIZE_TOKENS",
    "Chunk",
    "DEFAULT_SEPARATORS",
    "MIN_CHUNK_LENGTH",
    "MIN_CLEAN_LENGTH",
    "SponsorDetectionResult",
    "chunk_text",
    "chunk_text_dicts",
    "clean_text",
    "detect_sponsor",
    "detect_sponsor_detailed",
]
