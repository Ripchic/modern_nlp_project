"""reviewmind/vectorstore/client.py — Async Qdrant client wrapper + upsert with deduplication.

Provides :class:`QdrantClientWrapper` for lifecycle management and
:func:`upsert_chunks` to insert embedding vectors into Qdrant with
cosine-similarity deduplication (threshold > 0.95 for same ``source_url`` → skip).
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
)

logger = structlog.get_logger("reviewmind.vectorstore.client")

# ── Constants ────────────────────────────────────────────────────────────────

#: Cosine similarity threshold for deduplication.  Chunks with existing
#: matches above this value **for the same source_url** are skipped.
DEDUP_SIMILARITY_THRESHOLD: float = 0.95

#: Default batch size for upsert operations.
DEFAULT_UPSERT_BATCH_SIZE: int = 64

#: Namespace UUID used to generate deterministic point IDs via ``uuid5``.
_NAMESPACE_UUID = uuid.UUID("a3f1b2c4-d5e6-4a7b-8c9d-0e1f2a3b4c5d")


# ── Data types ───────────────────────────────────────────────────────────────


@dataclass
class ChunkPayload:
    """Payload fields for a single chunk to be stored in Qdrant.

    All fields correspond to the Qdrant payload schema defined in PRD §6.2.

    Attributes
    ----------
    text:
        The actual chunk text.
    source_url:
        URL of the original source document.
    source_type:
        Source type string (``"youtube"``, ``"reddit"``, ``"web"``, ``"tavily"``,
        ``"curated"``).
    product_query:
        The product query string this chunk is associated with.
    chunk_index:
        Zero-based index of the chunk within the source document.
    language:
        Detected language code (e.g. ``"ru"``, ``"en"``).
    is_sponsored:
        Whether the source was detected as sponsored content.
    is_curated:
        Whether the chunk comes from the curated knowledge base.
    source_id:
        Identifier linking to the PostgreSQL ``sources`` table.
    author:
        Author / channel / username of the source (if available).
    date:
        Publication date string (if available).
    session_id:
        Session identifier for per-session grouping.
    """

    text: str
    source_url: str
    source_type: str = ""
    product_query: str = ""
    chunk_index: int = 0
    language: str = ""
    is_sponsored: bool = False
    is_curated: bool = False
    source_id: str | int = ""
    author: str = ""
    date: str = ""
    session_id: str = ""

    def to_dict(self) -> dict:
        """Convert to a plain dict suitable for Qdrant payload storage."""
        return {k: v for k, v in asdict(self).items()}


@dataclass
class UpsertResult:
    """Summary returned by :func:`upsert_chunks`.

    Attributes
    ----------
    total:
        Total number of chunks submitted for upsert.
    inserted:
        Number of chunks actually written to Qdrant.
    skipped:
        Number of chunks skipped due to deduplication.
    skipped_indices:
        Chunk indices (within the input list) that were skipped.
    """

    total: int = 0
    inserted: int = 0
    skipped: int = 0
    skipped_indices: list[int] = field(default_factory=list)


# ── QdrantClientWrapper ─────────────────────────────────────────────────────


class QdrantClientWrapper:
    """Async wrapper around Qdrant client with lifecycle management.

    Usage::

        async with QdrantClientWrapper() as wrapper:
            exists = await wrapper.client.collection_exists("my_col")

    Or create explicitly::

        wrapper = QdrantClientWrapper(url="http://localhost:6333")
        await wrapper.close()
    """

    def __init__(
        self,
        url: str | None = None,
        timeout: int = 10,
    ) -> None:
        if url is None:
            from reviewmind.config import settings

            url = settings.qdrant_url

        self._url = url
        self._timeout = timeout
        self._client: AsyncQdrantClient | None = None

    @property
    def client(self) -> AsyncQdrantClient:
        """Return the underlying async Qdrant client, creating it lazily."""
        if self._client is None:
            self._client = AsyncQdrantClient(url=self._url, timeout=self._timeout)
            logger.info("qdrant_client_created", url=self._url)
        return self._client

    async def close(self) -> None:
        """Close the underlying Qdrant client connection."""
        if self._client is not None:
            await self._client.close()
            logger.info("qdrant_client_closed")
            self._client = None

    async def __aenter__(self) -> QdrantClientWrapper:
        # Ensure the client is created on enter
        _ = self.client
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def health_check(self) -> bool:
        """Return True if Qdrant is reachable, False otherwise."""
        try:
            await self.client.get_collections()
            return True
        except Exception:
            return False


# ── Point ID generation ──────────────────────────────────────────────────────


def generate_point_id(source_url: str, chunk_index: int) -> str:
    """Generate a deterministic UUID for a chunk based on source URL and index.

    Using ``uuid5`` ensures that re-processing the same (source_url, chunk_index)
    yields the same point ID, which makes ``upsert`` naturally idempotent.
    """
    name = f"{source_url}::{chunk_index}"
    return str(uuid.uuid5(_NAMESPACE_UUID, name))


# ── Deduplication helpers ────────────────────────────────────────────────────


async def _check_duplicates(
    client: AsyncQdrantClient,
    collection_name: str,
    vector: list[float],
    source_url: str,
    threshold: float = DEDUP_SIMILARITY_THRESHOLD,
) -> bool:
    """Return ``True`` if a near-duplicate already exists in *collection_name*.

    Searches the collection with *vector*, filtered to the same ``source_url``.
    If the top result has cosine similarity ≥ *threshold*, the chunk is
    considered a duplicate.
    """
    query_filter = Filter(
        must=[
            FieldCondition(
                key="source_url",
                match=MatchValue(value=source_url),
            ),
        ],
    )

    try:
        response = await client.query_points(
            collection_name=collection_name,
            query=vector,
            query_filter=query_filter,
            limit=1,
            with_payload=False,
            with_vectors=False,
            score_threshold=threshold,
        )
        if response.points:
            logger.debug(
                "duplicate_found",
                collection=collection_name,
                source_url=source_url,
                score=response.points[0].score,
            )
            return True
    except Exception:
        logger.exception(
            "dedup_check_error",
            collection=collection_name,
            source_url=source_url,
        )
        # On error, proceed with upsert (don't block on dedup failure)

    return False


# ── Upsert ───────────────────────────────────────────────────────────────────


async def upsert_chunks(
    client: AsyncQdrantClient,
    collection_name: str,
    vectors: list[list[float]],
    payloads: list[ChunkPayload],
    *,
    dedup_threshold: float = DEDUP_SIMILARITY_THRESHOLD,
    batch_size: int = DEFAULT_UPSERT_BATCH_SIZE,
    skip_dedup: bool = False,
) -> UpsertResult:
    """Insert embedding vectors with payloads into a Qdrant collection.

    Before inserting, each chunk is checked for near-duplicates **with the same
    ``source_url``**.  If cosine similarity ≥ *dedup_threshold* (default 0.95),
    the chunk is skipped.

    Parameters
    ----------
    client:
        An active :class:`AsyncQdrantClient`.
    collection_name:
        Target Qdrant collection name.
    vectors:
        List of embedding vectors (one per chunk).  Must be the same length as
        *payloads*.
    payloads:
        List of :class:`ChunkPayload` objects describing each chunk.
    dedup_threshold:
        Cosine similarity threshold for deduplication (default 0.95).
    batch_size:
        Maximum number of points per ``upsert`` call (default 64).
    skip_dedup:
        If ``True``, skip deduplication checks entirely (useful for fresh
        collections or curated seed imports where speed matters).

    Returns
    -------
    UpsertResult
        Summary containing counts of inserted / skipped chunks.

    Raises
    ------
    ValueError
        If ``vectors`` and ``payloads`` have different lengths or are empty.
    """
    if len(vectors) != len(payloads):
        msg = f"vectors and payloads must have the same length, got {len(vectors)} vectors and {len(payloads)} payloads"
        raise ValueError(msg)

    total = len(vectors)
    if total == 0:
        logger.info("upsert_chunks_empty", collection=collection_name)
        return UpsertResult(total=0, inserted=0, skipped=0)

    logger.info(
        "upsert_chunks_start",
        collection=collection_name,
        total=total,
        skip_dedup=skip_dedup,
    )

    # ── Step 1: Deduplication ────────────────────────────────────────────
    points_to_upsert: list[PointStruct] = []
    skipped_indices: list[int] = []

    for idx, (vector, payload) in enumerate(zip(vectors, payloads)):
        # Check for near-duplicate if dedup is enabled
        if not skip_dedup:
            is_dup = await _check_duplicates(
                client=client,
                collection_name=collection_name,
                vector=vector,
                source_url=payload.source_url,
                threshold=dedup_threshold,
            )
            if is_dup:
                skipped_indices.append(idx)
                continue

        point_id = generate_point_id(payload.source_url, payload.chunk_index)
        point = PointStruct(
            id=point_id,
            vector=vector,
            payload=payload.to_dict(),
        )
        points_to_upsert.append(point)

    skipped = len(skipped_indices)

    # ── Step 2: Batch upsert ─────────────────────────────────────────────
    inserted = 0
    for batch_start in range(0, len(points_to_upsert), batch_size):
        batch = points_to_upsert[batch_start : batch_start + batch_size]
        try:
            await client.upsert(
                collection_name=collection_name,
                points=batch,
            )
            inserted += len(batch)
            logger.debug(
                "upsert_batch_done",
                collection=collection_name,
                batch_size=len(batch),
                batch_start=batch_start,
            )
        except Exception:
            logger.exception(
                "upsert_batch_error",
                collection=collection_name,
                batch_start=batch_start,
                batch_size=len(batch),
            )
            raise

    result = UpsertResult(
        total=total,
        inserted=inserted,
        skipped=skipped,
        skipped_indices=skipped_indices,
    )

    logger.info(
        "upsert_chunks_done",
        collection=collection_name,
        total=total,
        inserted=inserted,
        skipped=skipped,
    )

    return result
