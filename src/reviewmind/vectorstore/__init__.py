"""reviewmind.vectorstore — Qdrant client wrapper, collection management, and search."""

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
from reviewmind.vectorstore.search import (
    DEFAULT_SCORE_THRESHOLD,
    DEFAULT_TOP_K,
    SearchResult,
    hybrid_search,
    search_collection,
)

__all__ = [
    "ALL_COLLECTIONS",
    "AUTO_CRAWLED_SPEC",
    "COLLECTION_AUTO_CRAWLED",
    "COLLECTION_CURATED_KB",
    "CURATED_KB_SPEC",
    "CollectionSpec",
    "DEFAULT_SCORE_THRESHOLD",
    "DEFAULT_TOP_K",
    "EMBEDDING_DIMENSION",
    "PayloadIndex",
    "QdrantClientWrapper",
    "SearchResult",
    "SourceType",
    "ensure_all_collections",
    "ensure_collection",
    "hybrid_search",
    "search_collection",
]
