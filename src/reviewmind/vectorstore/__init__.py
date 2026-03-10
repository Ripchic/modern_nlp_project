"""reviewmind.vectorstore — Qdrant client wrapper, collection management, search, and upsert."""

from reviewmind.vectorstore.client import (
    DEDUP_SIMILARITY_THRESHOLD,
    DEFAULT_UPSERT_BATCH_SIZE,
    ChunkPayload,
    QdrantClientWrapper,
    UpsertResult,
    generate_point_id,
    upsert_chunks,
)
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
    "ChunkPayload",
    "CollectionSpec",
    "DEFAULT_SCORE_THRESHOLD",
    "DEFAULT_TOP_K",
    "DEFAULT_UPSERT_BATCH_SIZE",
    "DEDUP_SIMILARITY_THRESHOLD",
    "EMBEDDING_DIMENSION",
    "PayloadIndex",
    "QdrantClientWrapper",
    "SearchResult",
    "SourceType",
    "UpsertResult",
    "ensure_all_collections",
    "ensure_collection",
    "generate_point_id",
    "hybrid_search",
    "search_collection",
    "upsert_chunks",
]
