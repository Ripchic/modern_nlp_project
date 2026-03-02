"""reviewmind.vectorstore — Qdrant client wrapper and collection management."""

from reviewmind.vectorstore.client import QdrantClientWrapper
from reviewmind.vectorstore.collections import (
    ALL_COLLECTIONS,
    AUTO_CRAWLED_SPEC,
    COLLECTION_AUTO_CRAWLED,
    COLLECTION_CURATED_KB,
    CURATED_KB_SPEC,
    EMBEDDING_DIMENSION,
    CollectionSpec,
    PayloadIndex,
    SourceType,
    ensure_all_collections,
    ensure_collection,
)

__all__ = [
    "ALL_COLLECTIONS",
    "AUTO_CRAWLED_SPEC",
    "COLLECTION_AUTO_CRAWLED",
    "COLLECTION_CURATED_KB",
    "CURATED_KB_SPEC",
    "CollectionSpec",
    "EMBEDDING_DIMENSION",
    "PayloadIndex",
    "QdrantClientWrapper",
    "SourceType",
    "ensure_all_collections",
    "ensure_collection",
]
