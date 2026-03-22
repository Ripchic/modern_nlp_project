"""reviewmind.ingestion.chunker — Text chunking with RecursiveCharacterTextSplitter.

Splits cleaned text into overlapping chunks suitable for embedding and vector storage.
Configuration: ~500 tokens per chunk (~2000 chars), 50 token overlap (~200 chars).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import structlog
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Approximate chars-per-token ratio for multilingual text (EN ~4, RU ~2.5).
# We use a conservative ~4 chars/token to stay within token budgets.
CHARS_PER_TOKEN: int = 4

# Target chunk size in tokens and characters
CHUNK_SIZE_TOKENS: int = 500
CHUNK_SIZE_CHARS: int = CHUNK_SIZE_TOKENS * CHARS_PER_TOKEN  # 2000

# Overlap in tokens and characters
CHUNK_OVERLAP_TOKENS: int = 50
CHUNK_OVERLAP_CHARS: int = CHUNK_OVERLAP_TOKENS * CHARS_PER_TOKEN  # 200

# Minimum text length (chars) to attempt chunking — shorter text is
# returned as a single chunk rather than being silently discarded.
MIN_CHUNK_LENGTH: int = 1

# Default separators for RecursiveCharacterTextSplitter.
# Prioritises paragraph breaks → sentence boundaries → word boundaries.
DEFAULT_SEPARATORS: list[str] = [
    "\n\n",  # paragraph break
    "\n",  # line break
    ". ",  # sentence boundary (period + space)
    "! ",  # exclamation sentence boundary
    "? ",  # question sentence boundary
    "; ",  # semicolon boundary
    ", ",  # clause boundary
    " ",  # word boundary
    "",  # character-level fallback
]


# ---------------------------------------------------------------------------
# Chunk dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Chunk:
    """A single chunk of text with its index and associated metadata."""

    text: str
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise the chunk to a plain dict."""
        return asdict(self)


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------


def _build_splitter(
    chunk_size: int = CHUNK_SIZE_CHARS,
    chunk_overlap: int = CHUNK_OVERLAP_CHARS,
    separators: list[str] | None = None,
) -> RecursiveCharacterTextSplitter:
    """Create a configured ``RecursiveCharacterTextSplitter`` instance."""
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators or DEFAULT_SEPARATORS,
        length_function=len,
        strip_whitespace=True,
        keep_separator=True,
    )


def chunk_text(
    text: str,
    metadata: dict[str, Any] | None = None,
    *,
    chunk_size: int = CHUNK_SIZE_CHARS,
    chunk_overlap: int = CHUNK_OVERLAP_CHARS,
    separators: list[str] | None = None,
) -> list[Chunk]:
    """Split *text* into overlapping chunks with sequential ``chunk_index``.

    Parameters
    ----------
    text:
        Raw or pre-cleaned text to chunk.
    metadata:
        Optional dict of metadata to attach to every chunk (e.g. source_url,
        source_type).  Each :class:`Chunk` will carry a **copy** of *metadata*
        augmented with ``chunk_index``.
    chunk_size:
        Maximum characters per chunk (default ``CHUNK_SIZE_CHARS``).
    chunk_overlap:
        Character overlap between consecutive chunks (default ``CHUNK_OVERLAP_CHARS``).
    separators:
        Custom separator list for ``RecursiveCharacterTextSplitter``.

    Returns
    -------
    list[Chunk]
        Ordered list of chunks.  Empty input → empty list.
    """
    if metadata is None:
        metadata = {}

    # --- guard: empty / whitespace-only text → empty list ---
    if not text or not text.strip():
        logger.debug("chunk_text received empty text, returning empty list")
        return []

    stripped = text.strip()

    splitter = _build_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
    )

    raw_chunks: list[str] = splitter.split_text(stripped)

    # Filter out empty fragments that may result from splitting artefacts
    raw_chunks = [c for c in raw_chunks if c.strip()]

    if not raw_chunks:
        logger.debug("chunk_text: splitter produced no chunks")
        return []

    chunks: list[Chunk] = []
    for idx, chunk_text_item in enumerate(raw_chunks):
        chunk_meta = {**metadata, "chunk_index": idx}
        chunks.append(Chunk(text=chunk_text_item, chunk_index=idx, metadata=chunk_meta))

    logger.info(
        "text_chunked",
        total_chars=len(stripped),
        num_chunks=len(chunks),
        avg_chunk_chars=sum(len(c.text) for c in chunks) // max(len(chunks), 1),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return chunks


def chunk_text_dicts(
    text: str,
    metadata: dict[str, Any] | None = None,
    *,
    chunk_size: int = CHUNK_SIZE_CHARS,
    chunk_overlap: int = CHUNK_OVERLAP_CHARS,
    separators: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Convenience wrapper that returns chunks as plain dicts.

    Same signature as :func:`chunk_text` but serialises each :class:`Chunk`
    via :meth:`Chunk.to_dict`.
    """
    chunks = chunk_text(
        text,
        metadata=metadata,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
    )
    return [c.to_dict() for c in chunks]
