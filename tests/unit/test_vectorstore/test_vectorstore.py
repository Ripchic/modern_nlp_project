"""Unit tests for reviewmind.vectorstore — client, collections, init_qdrant."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from qdrant_client.models import Distance, PayloadSchemaType, VectorParams

from reviewmind.vectorstore.client import QdrantClientWrapper
from reviewmind.vectorstore.collections import (
    ALL_COLLECTIONS,
    AUTO_CRAWLED_SPEC,
    COLLECTION_AUTO_CRAWLED,
    COLLECTION_CURATED_KB,
    CURATED_KB_SPEC,
    DISTANCE_METRIC,
    EMBEDDING_DIMENSION,
    CollectionSpec,
    PayloadIndex,
    SourceType,
    delete_collection,
    ensure_all_collections,
    ensure_collection,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Constants & enums
# ═══════════════════════════════════════════════════════════════════════════════


class TestConstants:
    """Test module-level constants."""

    def test_embedding_dimension(self):
        assert EMBEDDING_DIMENSION == 1536

    def test_distance_metric(self):
        assert DISTANCE_METRIC == Distance.COSINE

    def test_collection_auto_crawled_name(self):
        assert COLLECTION_AUTO_CRAWLED == "auto_crawled"

    def test_collection_curated_kb_name(self):
        assert COLLECTION_CURATED_KB == "curated_kb"


class TestSourceType:
    """Test SourceType enum."""

    def test_youtube(self):
        assert SourceType.YOUTUBE == "youtube"

    def test_reddit(self):
        assert SourceType.REDDIT == "reddit"

    def test_web(self):
        assert SourceType.WEB == "web"

    def test_tavily(self):
        assert SourceType.TAVILY == "tavily"

    def test_curated(self):
        assert SourceType.CURATED == "curated"

    def test_all_values(self):
        values = {st.value for st in SourceType}
        assert values == {"youtube", "reddit", "web", "tavily", "curated"}


# ═══════════════════════════════════════════════════════════════════════════════
# PayloadIndex dataclass
# ═══════════════════════════════════════════════════════════════════════════════


class TestPayloadIndex:
    """Test PayloadIndex frozen dataclass."""

    def test_creation(self):
        idx = PayloadIndex("product_query", PayloadSchemaType.KEYWORD)
        assert idx.field_name == "product_query"
        assert idx.field_schema == PayloadSchemaType.KEYWORD

    def test_frozen(self):
        idx = PayloadIndex("test", PayloadSchemaType.BOOL)
        with pytest.raises(AttributeError):
            idx.field_name = "other"  # type: ignore[misc]

    def test_equality(self):
        a = PayloadIndex("x", PayloadSchemaType.KEYWORD)
        b = PayloadIndex("x", PayloadSchemaType.KEYWORD)
        assert a == b


# ═══════════════════════════════════════════════════════════════════════════════
# CollectionSpec dataclass
# ═══════════════════════════════════════════════════════════════════════════════


class TestCollectionSpec:
    """Test CollectionSpec dataclass and vector_params property."""

    def test_defaults(self):
        spec = CollectionSpec(name="test_col")
        assert spec.name == "test_col"
        assert spec.dimension == EMBEDDING_DIMENSION
        assert spec.distance == DISTANCE_METRIC
        assert spec.payload_indexes == []

    def test_custom_dimension(self):
        spec = CollectionSpec(name="x", dimension=768)
        assert spec.dimension == 768

    def test_custom_distance(self):
        spec = CollectionSpec(name="x", distance=Distance.EUCLID)
        assert spec.distance == Distance.EUCLID

    def test_vector_params(self):
        spec = CollectionSpec(name="x")
        vp = spec.vector_params
        assert isinstance(vp, VectorParams)
        assert vp.size == EMBEDDING_DIMENSION
        assert vp.distance == Distance.COSINE

    def test_frozen(self):
        spec = CollectionSpec(name="x")
        with pytest.raises(AttributeError):
            spec.name = "y"  # type: ignore[misc]

    def test_with_indexes(self):
        indexes = [PayloadIndex("a", PayloadSchemaType.KEYWORD)]
        spec = CollectionSpec(name="x", payload_indexes=indexes)
        assert len(spec.payload_indexes) == 1


# ═══════════════════════════════════════════════════════════════════════════════
# Pre-defined collection specs
# ═══════════════════════════════════════════════════════════════════════════════


class TestAutoCollectionSpec:
    """Test AUTO_CRAWLED_SPEC properties."""

    def test_name(self):
        assert AUTO_CRAWLED_SPEC.name == "auto_crawled"

    def test_dimension(self):
        assert AUTO_CRAWLED_SPEC.dimension == 1536

    def test_distance(self):
        assert AUTO_CRAWLED_SPEC.distance == Distance.COSINE

    def test_has_product_query_index(self):
        names = {idx.field_name for idx in AUTO_CRAWLED_SPEC.payload_indexes}
        assert "product_query" in names

    def test_has_source_type_index(self):
        names = {idx.field_name for idx in AUTO_CRAWLED_SPEC.payload_indexes}
        assert "source_type" in names

    def test_has_is_sponsored_index(self):
        names = {idx.field_name for idx in AUTO_CRAWLED_SPEC.payload_indexes}
        assert "is_sponsored" in names

    def test_has_is_curated_index(self):
        names = {idx.field_name for idx in AUTO_CRAWLED_SPEC.payload_indexes}
        assert "is_curated" in names

    def test_has_source_url_index(self):
        names = {idx.field_name for idx in AUTO_CRAWLED_SPEC.payload_indexes}
        assert "source_url" in names

    def test_has_language_index(self):
        names = {idx.field_name for idx in AUTO_CRAWLED_SPEC.payload_indexes}
        assert "language" in names

    def test_does_not_have_category_index(self):
        names = {idx.field_name for idx in AUTO_CRAWLED_SPEC.payload_indexes}
        assert "category" not in names

    def test_shared_index_count(self):
        # 6 shared indexes, no extra for auto_crawled
        assert len(AUTO_CRAWLED_SPEC.payload_indexes) == 6


class TestCuratedCollectionSpec:
    """Test CURATED_KB_SPEC properties."""

    def test_name(self):
        assert CURATED_KB_SPEC.name == "curated_kb"

    def test_dimension(self):
        assert CURATED_KB_SPEC.dimension == 1536

    def test_distance(self):
        assert CURATED_KB_SPEC.distance == Distance.COSINE

    def test_has_category_index(self):
        names = {idx.field_name for idx in CURATED_KB_SPEC.payload_indexes}
        assert "category" in names

    def test_has_all_shared_indexes(self):
        names = {idx.field_name for idx in CURATED_KB_SPEC.payload_indexes}
        for shared in ("product_query", "source_type", "is_sponsored", "is_curated", "source_url", "language"):
            assert shared in names

    def test_index_count(self):
        # 6 shared + 1 category
        assert len(CURATED_KB_SPEC.payload_indexes) == 7


class TestAllCollections:
    """Test the ALL_COLLECTIONS list."""

    def test_length(self):
        assert len(ALL_COLLECTIONS) == 2

    def test_contains_auto_crawled(self):
        names = [c.name for c in ALL_COLLECTIONS]
        assert "auto_crawled" in names

    def test_contains_curated_kb(self):
        names = [c.name for c in ALL_COLLECTIONS]
        assert "curated_kb" in names


# ═══════════════════════════════════════════════════════════════════════════════
# QdrantClientWrapper
# ═══════════════════════════════════════════════════════════════════════════════


class TestQdrantClientWrapper:
    """Test QdrantClientWrapper lifecycle and properties."""

    @patch("reviewmind.vectorstore.client.AsyncQdrantClient")
    def test_client_lazy_creation(self, mock_cls):
        wrapper = QdrantClientWrapper(url="http://test:6333")
        assert wrapper._client is None
        _ = wrapper.client
        mock_cls.assert_called_once_with(url="http://test:6333", timeout=10)
        assert wrapper._client is not None

    @patch("reviewmind.vectorstore.client.AsyncQdrantClient")
    def test_client_reuses_instance(self, mock_cls):
        wrapper = QdrantClientWrapper(url="http://test:6333")
        c1 = wrapper.client
        c2 = wrapper.client
        assert c1 is c2
        mock_cls.assert_called_once()

    @patch("reviewmind.vectorstore.client.AsyncQdrantClient")
    def test_custom_timeout(self, mock_cls):
        wrapper = QdrantClientWrapper(url="http://test:6333", timeout=30)
        _ = wrapper.client
        mock_cls.assert_called_once_with(url="http://test:6333", timeout=30)

    def test_default_url_from_config(self):
        mock_settings = MagicMock()
        mock_settings.qdrant_url = "http://config-url:6333"
        with patch("reviewmind.config.settings", mock_settings):
            wrapper = QdrantClientWrapper()
            assert wrapper._url == "http://config-url:6333"

    @pytest.mark.asyncio
    @patch("reviewmind.vectorstore.client.AsyncQdrantClient")
    async def test_close(self, mock_cls):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        wrapper = QdrantClientWrapper(url="http://test:6333")
        _ = wrapper.client
        await wrapper.close()
        mock_client.close.assert_awaited_once()
        assert wrapper._client is None

    @pytest.mark.asyncio
    @patch("reviewmind.vectorstore.client.AsyncQdrantClient")
    async def test_close_when_not_created(self, mock_cls):
        wrapper = QdrantClientWrapper(url="http://test:6333")
        # Closing without creating should be safe
        await wrapper.close()
        mock_cls.assert_not_called()

    @pytest.mark.asyncio
    @patch("reviewmind.vectorstore.client.AsyncQdrantClient")
    async def test_context_manager(self, mock_cls):
        mock_client = AsyncMock()
        mock_cls.return_value = mock_client
        async with QdrantClientWrapper(url="http://test:6333") as wrapper:
            assert wrapper._client is not None
        mock_client.close.assert_awaited_once()

    @pytest.mark.asyncio
    @patch("reviewmind.vectorstore.client.AsyncQdrantClient")
    async def test_health_check_success(self, mock_cls):
        mock_client = AsyncMock()
        mock_client.get_collections = AsyncMock(return_value=[])
        mock_cls.return_value = mock_client
        wrapper = QdrantClientWrapper(url="http://test:6333")
        assert await wrapper.health_check() is True

    @pytest.mark.asyncio
    @patch("reviewmind.vectorstore.client.AsyncQdrantClient")
    async def test_health_check_failure(self, mock_cls):
        mock_client = AsyncMock()
        mock_client.get_collections = AsyncMock(side_effect=Exception("connection refused"))
        mock_cls.return_value = mock_client
        wrapper = QdrantClientWrapper(url="http://test:6333")
        assert await wrapper.health_check() is False


# ═══════════════════════════════════════════════════════════════════════════════
# ensure_collection
# ═══════════════════════════════════════════════════════════════════════════════


class TestEnsureCollection:
    """Test ensure_collection function."""

    @pytest.mark.asyncio
    async def test_creates_new_collection(self):
        client = AsyncMock()
        client.collection_exists = AsyncMock(return_value=False)
        client.create_collection = AsyncMock()
        client.create_payload_index = AsyncMock()

        spec = CollectionSpec(
            name="test_col",
            payload_indexes=[PayloadIndex("field1", PayloadSchemaType.KEYWORD)],
        )

        result = await ensure_collection(client, spec)

        assert result is True
        client.create_collection.assert_awaited_once_with(
            collection_name="test_col",
            vectors_config=spec.vector_params,
        )
        client.create_payload_index.assert_awaited_once_with(
            collection_name="test_col",
            field_name="field1",
            field_schema=PayloadSchemaType.KEYWORD,
        )

    @pytest.mark.asyncio
    async def test_skips_existing_collection(self):
        client = AsyncMock()
        client.collection_exists = AsyncMock(return_value=True)

        spec = CollectionSpec(name="existing_col")
        result = await ensure_collection(client, spec)

        assert result is False
        client.create_collection.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_creates_multiple_indexes(self):
        client = AsyncMock()
        client.collection_exists = AsyncMock(return_value=False)

        indexes = [
            PayloadIndex("a", PayloadSchemaType.KEYWORD),
            PayloadIndex("b", PayloadSchemaType.BOOL),
            PayloadIndex("c", PayloadSchemaType.KEYWORD),
        ]
        spec = CollectionSpec(name="multi_idx", payload_indexes=indexes)

        await ensure_collection(client, spec)

        assert client.create_payload_index.await_count == 3

    @pytest.mark.asyncio
    async def test_auto_crawled_spec(self):
        client = AsyncMock()
        client.collection_exists = AsyncMock(return_value=False)

        await ensure_collection(client, AUTO_CRAWLED_SPEC)

        client.create_collection.assert_awaited_once()
        call_kwargs = client.create_collection.call_args
        assert call_kwargs.kwargs["collection_name"] == "auto_crawled"
        vp = call_kwargs.kwargs["vectors_config"]
        assert vp.size == 1536
        assert vp.distance == Distance.COSINE
        assert client.create_payload_index.await_count == 6

    @pytest.mark.asyncio
    async def test_curated_kb_spec(self):
        client = AsyncMock()
        client.collection_exists = AsyncMock(return_value=False)

        await ensure_collection(client, CURATED_KB_SPEC)

        client.create_collection.assert_awaited_once()
        call_kwargs = client.create_collection.call_args
        assert call_kwargs.kwargs["collection_name"] == "curated_kb"
        assert client.create_payload_index.await_count == 7  # 6 shared + category

    @pytest.mark.asyncio
    async def test_no_indexes_spec(self):
        client = AsyncMock()
        client.collection_exists = AsyncMock(return_value=False)

        spec = CollectionSpec(name="no_idx")
        await ensure_collection(client, spec)

        client.create_collection.assert_awaited_once()
        client.create_payload_index.assert_not_awaited()


# ═══════════════════════════════════════════════════════════════════════════════
# ensure_all_collections
# ═══════════════════════════════════════════════════════════════════════════════


class TestEnsureAllCollections:
    """Test ensure_all_collections function."""

    @pytest.mark.asyncio
    async def test_creates_both_collections(self):
        client = AsyncMock()
        client.collection_exists = AsyncMock(return_value=False)

        results = await ensure_all_collections(client)

        assert results == {"auto_crawled": True, "curated_kb": True}
        assert client.create_collection.await_count == 2

    @pytest.mark.asyncio
    async def test_both_exist(self):
        client = AsyncMock()
        client.collection_exists = AsyncMock(return_value=True)

        results = await ensure_all_collections(client)

        assert results == {"auto_crawled": False, "curated_kb": False}
        client.create_collection.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_partial_exists(self):
        client = AsyncMock()
        exists_map = {"auto_crawled": True, "curated_kb": False}
        client.collection_exists = AsyncMock(side_effect=lambda name: exists_map[name])

        results = await ensure_all_collections(client)

        assert results["auto_crawled"] is False  # already existed
        assert results["curated_kb"] is True  # was created
        # Only curated_kb was created
        client.create_collection.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_returns_dict_with_all_collection_names(self):
        client = AsyncMock()
        client.collection_exists = AsyncMock(return_value=False)
        results = await ensure_all_collections(client)
        assert set(results.keys()) == {"auto_crawled", "curated_kb"}


# ═══════════════════════════════════════════════════════════════════════════════
# delete_collection
# ═══════════════════════════════════════════════════════════════════════════════


class TestDeleteCollection:
    """Test delete_collection helper."""

    @pytest.mark.asyncio
    async def test_deletes_existing(self):
        client = AsyncMock()
        client.collection_exists = AsyncMock(return_value=True)
        client.delete_collection = AsyncMock()

        result = await delete_collection(client, "test_col")

        assert result is True
        client.delete_collection.assert_awaited_once_with("test_col")

    @pytest.mark.asyncio
    async def test_skip_nonexistent(self):
        client = AsyncMock()
        client.collection_exists = AsyncMock(return_value=False)

        result = await delete_collection(client, "nope")

        assert result is False
        client.delete_collection.assert_not_awaited()


# ═══════════════════════════════════════════════════════════════════════════════
# Vectorstore __init__ exports
# ═══════════════════════════════════════════════════════════════════════════════


class TestVectorstoreExports:
    """Test that __init__.py exports key symbols."""

    def test_import_client_wrapper(self):
        from reviewmind.vectorstore import QdrantClientWrapper

        assert QdrantClientWrapper is not None

    def test_import_collection_names(self):
        from reviewmind.vectorstore import COLLECTION_AUTO_CRAWLED, COLLECTION_CURATED_KB

        assert COLLECTION_AUTO_CRAWLED == "auto_crawled"
        assert COLLECTION_CURATED_KB == "curated_kb"

    def test_import_specs(self):
        from reviewmind.vectorstore import AUTO_CRAWLED_SPEC, CURATED_KB_SPEC

        assert AUTO_CRAWLED_SPEC.name == "auto_crawled"
        assert CURATED_KB_SPEC.name == "curated_kb"

    def test_import_ensure_functions(self):
        from reviewmind.vectorstore import ensure_all_collections, ensure_collection

        assert callable(ensure_all_collections)
        assert callable(ensure_collection)

    def test_import_embedding_dimension(self):
        from reviewmind.vectorstore import EMBEDDING_DIMENSION

        assert EMBEDDING_DIMENSION == 1536

    def test_import_source_type(self):
        from reviewmind.vectorstore import SourceType

        assert SourceType.YOUTUBE == "youtube"

    def test_import_payload_index(self):
        from reviewmind.vectorstore import PayloadIndex

        idx = PayloadIndex("x", PayloadSchemaType.KEYWORD)
        assert idx.field_name == "x"

    def test_import_collection_spec(self):
        from reviewmind.vectorstore import CollectionSpec

        spec = CollectionSpec(name="t")
        assert spec.name == "t"


# ═══════════════════════════════════════════════════════════════════════════════
# init_qdrant script
# ═══════════════════════════════════════════════════════════════════════════════


class TestInitQdrantScript:
    """Test scripts/init_qdrant.py module can be imported and runs main()."""

    @pytest.mark.asyncio
    async def test_main_success(self):
        """Test that init_qdrant.main() calls ensure_all_collections."""
        with (
            patch("reviewmind.vectorstore.client.AsyncQdrantClient") as mock_cls,
        ):
            mock_client = AsyncMock()
            mock_client.get_collections = AsyncMock(return_value=[])
            mock_client.collection_exists = AsyncMock(return_value=False)
            mock_client.create_collection = AsyncMock()
            mock_client.create_payload_index = AsyncMock()
            mock_client.close = AsyncMock()
            mock_cls.return_value = mock_client

            # Import the script module
            import importlib
            import sys

            # Ensure fresh import
            for mod in list(sys.modules.keys()):
                if "init_qdrant" in mod:
                    del sys.modules[mod]

            sys.path.insert(0, "scripts")
            try:
                import init_qdrant

                importlib.reload(init_qdrant)
                await init_qdrant.main()
            finally:
                sys.path.pop(0)

            # Both collections should have been created
            assert mock_client.create_collection.await_count == 2

    @pytest.mark.asyncio
    async def test_main_unhealthy_exits(self):
        """Test that init_qdrant.main() exits if Qdrant is unreachable."""
        with patch("reviewmind.vectorstore.client.AsyncQdrantClient") as mock_cls:
            mock_client = AsyncMock()
            mock_client.get_collections = AsyncMock(side_effect=Exception("refused"))
            mock_client.close = AsyncMock()
            mock_cls.return_value = mock_client

            import importlib
            import sys

            for mod in list(sys.modules.keys()):
                if "init_qdrant" in mod:
                    del sys.modules[mod]

            sys.path.insert(0, "scripts")
            try:
                import init_qdrant

                importlib.reload(init_qdrant)
                with pytest.raises(SystemExit) as exc_info:
                    await init_qdrant.main()
                assert exc_info.value.code == 1
            finally:
                sys.path.pop(0)


# ═══════════════════════════════════════════════════════════════════════════════
# Payload index schemas match PRD section 6.2
# ═══════════════════════════════════════════════════════════════════════════════


class TestPayloadSchemaMatchesPRD:
    """Verify payload indexes match PRD section 6.2 requirements."""

    def _index_map(self, spec: CollectionSpec) -> dict[str, PayloadSchemaType]:
        return {idx.field_name: idx.field_schema for idx in spec.payload_indexes}

    def test_auto_crawled_product_query_keyword(self):
        m = self._index_map(AUTO_CRAWLED_SPEC)
        assert m["product_query"] == PayloadSchemaType.KEYWORD

    def test_auto_crawled_source_type_keyword(self):
        m = self._index_map(AUTO_CRAWLED_SPEC)
        assert m["source_type"] == PayloadSchemaType.KEYWORD

    def test_auto_crawled_is_sponsored_bool(self):
        m = self._index_map(AUTO_CRAWLED_SPEC)
        assert m["is_sponsored"] == PayloadSchemaType.BOOL

    def test_auto_crawled_is_curated_bool(self):
        m = self._index_map(AUTO_CRAWLED_SPEC)
        assert m["is_curated"] == PayloadSchemaType.BOOL

    def test_curated_kb_category_keyword(self):
        m = self._index_map(CURATED_KB_SPEC)
        assert m["category"] == PayloadSchemaType.KEYWORD

    def test_curated_kb_source_url_keyword(self):
        m = self._index_map(CURATED_KB_SPEC)
        assert m["source_url"] == PayloadSchemaType.KEYWORD

    def test_curated_kb_language_keyword(self):
        m = self._index_map(CURATED_KB_SPEC)
        assert m["language"] == PayloadSchemaType.KEYWORD

    def test_auto_crawled_vector_params(self):
        vp = AUTO_CRAWLED_SPEC.vector_params
        assert vp.size == 1536
        assert vp.distance == Distance.COSINE

    def test_curated_kb_vector_params(self):
        vp = CURATED_KB_SPEC.vector_params
        assert vp.size == 1536
        assert vp.distance == Distance.COSINE
