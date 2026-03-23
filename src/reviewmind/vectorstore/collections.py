"""reviewmind/vectorstore/collections.py — Qdrant collection definitions & initialization."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

logger = structlog.get_logger("reviewmind.vectorstore.collections")

# ── Constants ────────────────────────────────────────────────────────────────

EMBEDDING_DIMENSION = 1536
DISTANCE_METRIC = Distance.COSINE

COLLECTION_AUTO_CRAWLED = "auto_crawled"
COLLECTION_CURATED_KB = "curated_kb"


class SourceType(str, Enum):
    """Enumeration of supported source types for payload filtering."""

    YOUTUBE = "youtube"
    REDDIT = "reddit"
    WEB = "web"
    TAVILY = "tavily"
    CURATED = "curated"
    FORUM = "forum"  # 4PDA and other IP.Board forums


@dataclass(frozen=True)
class PayloadIndex:
    """Definition of a payload field index to create on a Qdrant collection."""

    field_name: str
    field_schema: PayloadSchemaType


# Payload indexes shared by both collections.
_SHARED_INDEXES: list[PayloadIndex] = [
    PayloadIndex("product_query", PayloadSchemaType.KEYWORD),
    PayloadIndex("source_type", PayloadSchemaType.KEYWORD),
    PayloadIndex("is_sponsored", PayloadSchemaType.BOOL),
    PayloadIndex("is_curated", PayloadSchemaType.BOOL),
    PayloadIndex("source_url", PayloadSchemaType.KEYWORD),
    PayloadIndex("language", PayloadSchemaType.KEYWORD),
]

# Extra index for curated_kb: category field.
_CURATED_EXTRA_INDEXES: list[PayloadIndex] = [
    PayloadIndex("category", PayloadSchemaType.KEYWORD),
]


@dataclass(frozen=True)
class CollectionSpec:
    """Full specification for a Qdrant collection."""

    name: str
    dimension: int = EMBEDDING_DIMENSION
    distance: Distance = DISTANCE_METRIC
    payload_indexes: list[PayloadIndex] = field(default_factory=list)

    @property
    def vector_params(self) -> VectorParams:
        return VectorParams(size=self.dimension, distance=self.distance)


# ── Pre-defined collection specs ─────────────────────────────────────────────

AUTO_CRAWLED_SPEC = CollectionSpec(
    name=COLLECTION_AUTO_CRAWLED,
    payload_indexes=list(_SHARED_INDEXES),
)

CURATED_KB_SPEC = CollectionSpec(
    name=COLLECTION_CURATED_KB,
    payload_indexes=list(_SHARED_INDEXES) + list(_CURATED_EXTRA_INDEXES),
)

ALL_COLLECTIONS: list[CollectionSpec] = [AUTO_CRAWLED_SPEC, CURATED_KB_SPEC]


# ── Initialization helpers ───────────────────────────────────────────────────


async def ensure_collection(
    client: AsyncQdrantClient,
    spec: CollectionSpec,
) -> bool:
    """Create a collection if it does not already exist.

    Returns True if the collection was created, False if it already existed.
    """
    exists = await client.collection_exists(spec.name)
    if exists:
        logger.info("collection_already_exists", collection=spec.name)
        return False

    await client.create_collection(
        collection_name=spec.name,
        vectors_config=spec.vector_params,
    )
    logger.info(
        "collection_created",
        collection=spec.name,
        dimension=spec.dimension,
        distance=spec.distance.value,
    )

    # Create payload indexes
    for idx in spec.payload_indexes:
        await client.create_payload_index(
            collection_name=spec.name,
            field_name=idx.field_name,
            field_schema=idx.field_schema,
        )
        logger.debug(
            "payload_index_created",
            collection=spec.name,
            field=idx.field_name,
            schema=idx.field_schema.value,
        )

    return True


async def ensure_all_collections(client: AsyncQdrantClient) -> dict[str, bool]:
    """Ensure all required collections exist.  Idempotent.

    Returns a dict mapping collection name → was_created (True) / already_existed (False).
    """
    results: dict[str, bool] = {}
    for spec in ALL_COLLECTIONS:
        created = await ensure_collection(client, spec)
        results[spec.name] = created
    return results


async def delete_collection(client: AsyncQdrantClient, name: str) -> bool:
    """Delete a collection if it exists.  Returns True if deleted, False otherwise."""
    exists = await client.collection_exists(name)
    if not exists:
        logger.info("collection_not_found_for_delete", collection=name)
        return False
    await client.delete_collection(name)
    logger.info("collection_deleted", collection=name)
    return True
