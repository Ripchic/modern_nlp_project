"""Unit tests for reviewmind.vectorstore.client — upsert_chunks with deduplication.

Covers: ChunkPayload, UpsertResult, generate_point_id, _check_duplicates,
upsert_chunks, batch splitting, dedup logic, skip_dedup, error handling,
vectorstore __init__ exports.
"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest
from qdrant_client.models import (
    FieldCondition,
    Filter,
    PointStruct,
    ScoredPoint,
)

from reviewmind.vectorstore.client import (
    _NAMESPACE_UUID,
    DEDUP_SIMILARITY_THRESHOLD,
    DEFAULT_UPSERT_BATCH_SIZE,
    ChunkPayload,
    UpsertResult,
    _check_duplicates,
    generate_point_id,
    upsert_chunks,
)

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

DUMMY_VECTOR = [0.1] * 1536


def _make_payload(
    source_url: str = "https://example.com/review",
    chunk_index: int = 0,
    **kwargs,
) -> ChunkPayload:
    """Create a ChunkPayload with sensible defaults."""
    defaults = {
        "text": f"Chunk text for {source_url} idx={chunk_index}",
        "source_url": source_url,
        "source_type": "web",
        "product_query": "test product",
        "chunk_index": chunk_index,
        "language": "en",
        "is_sponsored": False,
        "is_curated": False,
        "source_id": "src-1",
        "author": "test_author",
        "date": "2026-01-01",
        "session_id": "sess-abc",
    }
    defaults.update(kwargs)
    return ChunkPayload(**defaults)


def _make_query_response(points: list[ScoredPoint] | None = None):
    """Create a mock QueryResponse with a .points attribute."""
    resp = MagicMock()
    resp.points = points or []
    return resp


def _make_scored_point(
    point_id: str = "pt-1",
    score: float = 0.96,
) -> ScoredPoint:
    """Create a ScoredPoint for testing (no payload needed for dedup)."""
    return ScoredPoint(id=point_id, version=1, score=score, payload=None)


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════


class TestConstants:
    """Test module-level constants."""

    def test_dedup_threshold_value(self):
        assert DEDUP_SIMILARITY_THRESHOLD == 0.95

    def test_dedup_threshold_is_float(self):
        assert isinstance(DEDUP_SIMILARITY_THRESHOLD, float)

    def test_default_batch_size_value(self):
        assert DEFAULT_UPSERT_BATCH_SIZE == 64

    def test_default_batch_size_is_int(self):
        assert isinstance(DEFAULT_UPSERT_BATCH_SIZE, int)

    def test_namespace_uuid_is_uuid(self):
        assert isinstance(_NAMESPACE_UUID, uuid.UUID)


# ═══════════════════════════════════════════════════════════════════════════════
# ChunkPayload dataclass
# ═══════════════════════════════════════════════════════════════════════════════


class TestChunkPayload:
    """Test ChunkPayload creation, defaults, and serialization."""

    def test_creation_minimal(self):
        p = ChunkPayload(text="Hello", source_url="https://x.com")
        assert p.text == "Hello"
        assert p.source_url == "https://x.com"
        assert p.source_type == ""
        assert p.chunk_index == 0
        assert p.is_sponsored is False
        assert p.is_curated is False

    def test_creation_full(self):
        p = _make_payload(
            source_url="https://yt.com/v",
            chunk_index=3,
            source_type="youtube",
            is_sponsored=True,
            is_curated=True,
        )
        assert p.source_url == "https://yt.com/v"
        assert p.chunk_index == 3
        assert p.source_type == "youtube"
        assert p.is_sponsored is True
        assert p.is_curated is True

    def test_to_dict_keys(self):
        p = _make_payload()
        d = p.to_dict()
        expected_keys = {
            "text", "source_url", "source_type", "product_query",
            "chunk_index", "language", "is_sponsored", "is_curated",
            "source_id", "author", "date", "session_id",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_values(self):
        p = _make_payload(source_url="https://test.com", chunk_index=5)
        d = p.to_dict()
        assert d["source_url"] == "https://test.com"
        assert d["chunk_index"] == 5
        assert d["is_sponsored"] is False

    def test_to_dict_is_plain_dict(self):
        """to_dict() should return a plain dict, not a dataclass."""
        p = _make_payload()
        d = p.to_dict()
        assert isinstance(d, dict)

    def test_payload_fields_match_prd_section_6_2(self):
        """All fields required by PRD section 6.2 are present."""
        p = _make_payload()
        d = p.to_dict()
        required_fields = [
            "text", "source_url", "source_type", "product_query",
            "chunk_index", "language", "is_sponsored", "is_curated",
            "source_id", "author", "date", "session_id",
        ]
        for field_name in required_fields:
            assert field_name in d, f"Missing PRD field: {field_name}"


# ═══════════════════════════════════════════════════════════════════════════════
# UpsertResult dataclass
# ═══════════════════════════════════════════════════════════════════════════════


class TestUpsertResult:
    """Test UpsertResult dataclass."""

    def test_defaults(self):
        r = UpsertResult()
        assert r.total == 0
        assert r.inserted == 0
        assert r.skipped == 0
        assert r.skipped_indices == []

    def test_custom_values(self):
        r = UpsertResult(total=10, inserted=7, skipped=3, skipped_indices=[2, 5, 8])
        assert r.total == 10
        assert r.inserted == 7
        assert r.skipped == 3
        assert r.skipped_indices == [2, 5, 8]

    def test_skipped_indices_independent_instances(self):
        """Each UpsertResult gets its own list for skipped_indices."""
        r1 = UpsertResult()
        r2 = UpsertResult()
        r1.skipped_indices.append(0)
        assert r2.skipped_indices == []


# ═══════════════════════════════════════════════════════════════════════════════
# generate_point_id
# ═══════════════════════════════════════════════════════════════════════════════


class TestGeneratePointId:
    """Test deterministic UUID generation for point IDs."""

    def test_returns_string(self):
        pid = generate_point_id("https://example.com", 0)
        assert isinstance(pid, str)

    def test_valid_uuid(self):
        pid = generate_point_id("https://example.com", 0)
        parsed = uuid.UUID(pid)
        assert parsed.version == 5

    def test_deterministic(self):
        """Same inputs always produce the same UUID."""
        pid1 = generate_point_id("https://example.com", 0)
        pid2 = generate_point_id("https://example.com", 0)
        assert pid1 == pid2

    def test_different_url_different_id(self):
        pid1 = generate_point_id("https://a.com", 0)
        pid2 = generate_point_id("https://b.com", 0)
        assert pid1 != pid2

    def test_different_index_different_id(self):
        pid1 = generate_point_id("https://example.com", 0)
        pid2 = generate_point_id("https://example.com", 1)
        assert pid1 != pid2

    def test_uses_namespace_uuid(self):
        """Point ID should match uuid5 with our namespace."""
        expected = str(uuid.uuid5(_NAMESPACE_UUID, "https://x.com::3"))
        actual = generate_point_id("https://x.com", 3)
        assert actual == expected


# ═══════════════════════════════════════════════════════════════════════════════
# _check_duplicates
# ═══════════════════════════════════════════════════════════════════════════════


class TestCheckDuplicates:
    """Test internal _check_duplicates helper."""

    @pytest.mark.asyncio
    async def test_no_existing_points_returns_false(self):
        """Empty search results → not a duplicate."""
        client = AsyncMock()
        client.query_points = AsyncMock(return_value=_make_query_response([]))
        result = await _check_duplicates(
            client, "auto_crawled", DUMMY_VECTOR, "https://example.com"
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_high_similarity_returns_true(self):
        """A match with score ≥ threshold → duplicate."""
        point = _make_scored_point(score=0.96)
        client = AsyncMock()
        client.query_points = AsyncMock(
            return_value=_make_query_response([point])
        )
        result = await _check_duplicates(
            client, "auto_crawled", DUMMY_VECTOR, "https://example.com"
        )
        assert result is True

    @pytest.mark.asyncio
    async def test_low_similarity_returns_false(self):
        """No match above threshold → not a duplicate (threshold is handled by Qdrant)."""
        client = AsyncMock()
        client.query_points = AsyncMock(return_value=_make_query_response([]))
        result = await _check_duplicates(
            client, "auto_crawled", DUMMY_VECTOR, "https://example.com",
            threshold=0.95,
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_custom_threshold(self):
        """Custom threshold value is passed to query_points."""
        client = AsyncMock()
        client.query_points = AsyncMock(return_value=_make_query_response([]))
        await _check_duplicates(
            client, "auto_crawled", DUMMY_VECTOR, "https://example.com",
            threshold=0.99,
        )
        call_kwargs = client.query_points.call_args.kwargs
        assert call_kwargs["score_threshold"] == 0.99

    @pytest.mark.asyncio
    async def test_filters_by_source_url(self):
        """Dedup check should filter by source_url."""
        client = AsyncMock()
        client.query_points = AsyncMock(return_value=_make_query_response([]))
        await _check_duplicates(
            client, "auto_crawled", DUMMY_VECTOR, "https://example.com/review"
        )
        call_kwargs = client.query_points.call_args.kwargs
        qf = call_kwargs["query_filter"]
        assert isinstance(qf, Filter)
        assert len(qf.must) == 1
        cond = qf.must[0]
        assert isinstance(cond, FieldCondition)
        assert cond.key == "source_url"
        assert cond.match.value == "https://example.com/review"

    @pytest.mark.asyncio
    async def test_exception_returns_false(self):
        """On error, proceed with upsert (don't block on dedup failure)."""
        client = AsyncMock()
        client.query_points = AsyncMock(side_effect=Exception("network error"))
        result = await _check_duplicates(
            client, "auto_crawled", DUMMY_VECTOR, "https://example.com"
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_query_params(self):
        """Verify correct query parameters are sent."""
        client = AsyncMock()
        client.query_points = AsyncMock(return_value=_make_query_response([]))
        await _check_duplicates(
            client, "test_collection", DUMMY_VECTOR, "https://example.com"
        )
        call_kwargs = client.query_points.call_args.kwargs
        assert call_kwargs["collection_name"] == "test_collection"
        assert call_kwargs["query"] == DUMMY_VECTOR
        assert call_kwargs["limit"] == 1
        assert call_kwargs["with_payload"] is False
        assert call_kwargs["with_vectors"] is False


# ═══════════════════════════════════════════════════════════════════════════════
# upsert_chunks — success paths
# ═══════════════════════════════════════════════════════════════════════════════


class TestUpsertChunksSuccess:
    """Test upsert_chunks with successful upserts."""

    @pytest.mark.asyncio
    async def test_basic_upsert(self):
        """Upsert 3 chunks with no existing data → all inserted."""
        client = AsyncMock()
        client.query_points = AsyncMock(return_value=_make_query_response([]))
        client.upsert = AsyncMock()

        vectors = [DUMMY_VECTOR] * 3
        payloads = [_make_payload(chunk_index=i) for i in range(3)]

        result = await upsert_chunks(client, "auto_crawled", vectors, payloads)

        assert result.total == 3
        assert result.inserted == 3
        assert result.skipped == 0
        assert result.skipped_indices == []
        assert client.upsert.await_count == 1

    @pytest.mark.asyncio
    async def test_upsert_calls_with_point_structs(self):
        """Verify that PointStruct objects are passed to client.upsert."""
        client = AsyncMock()
        client.query_points = AsyncMock(return_value=_make_query_response([]))
        client.upsert = AsyncMock()

        vectors = [DUMMY_VECTOR]
        payloads = [_make_payload()]

        await upsert_chunks(client, "auto_crawled", vectors, payloads)

        call_kwargs = client.upsert.call_args.kwargs
        assert call_kwargs["collection_name"] == "auto_crawled"
        points = call_kwargs["points"]
        assert len(points) == 1
        assert isinstance(points[0], PointStruct)

    @pytest.mark.asyncio
    async def test_point_has_deterministic_id(self):
        """Point IDs should be deterministic UUIDs."""
        client = AsyncMock()
        client.query_points = AsyncMock(return_value=_make_query_response([]))
        client.upsert = AsyncMock()

        payload = _make_payload(source_url="https://test.com", chunk_index=2)
        expected_id = generate_point_id("https://test.com", 2)

        await upsert_chunks(client, "auto_crawled", [DUMMY_VECTOR], [payload])

        point = client.upsert.call_args.kwargs["points"][0]
        assert point.id == expected_id

    @pytest.mark.asyncio
    async def test_point_has_correct_payload(self):
        """Each point should carry the full payload dict."""
        client = AsyncMock()
        client.query_points = AsyncMock(return_value=_make_query_response([]))
        client.upsert = AsyncMock()

        payload = _make_payload(
            source_url="https://test.com",
            source_type="youtube",
            is_sponsored=True,
        )
        await upsert_chunks(client, "auto_crawled", [DUMMY_VECTOR], [payload])

        point = client.upsert.call_args.kwargs["points"][0]
        assert point.payload["source_url"] == "https://test.com"
        assert point.payload["source_type"] == "youtube"
        assert point.payload["is_sponsored"] is True

    @pytest.mark.asyncio
    async def test_point_has_vector(self):
        """Each point should carry its vector."""
        client = AsyncMock()
        client.query_points = AsyncMock(return_value=_make_query_response([]))
        client.upsert = AsyncMock()

        vec = [0.5] * 1536
        await upsert_chunks(client, "auto_crawled", [vec], [_make_payload()])

        point = client.upsert.call_args.kwargs["points"][0]
        assert point.vector == vec

    @pytest.mark.asyncio
    async def test_empty_input(self):
        """Empty vectors/payloads → no upsert, zero counts."""
        client = AsyncMock()
        result = await upsert_chunks(client, "auto_crawled", [], [])
        assert result.total == 0
        assert result.inserted == 0
        assert result.skipped == 0
        client.upsert.assert_not_awaited()


# ═══════════════════════════════════════════════════════════════════════════════
# upsert_chunks — deduplication
# ═══════════════════════════════════════════════════════════════════════════════


class TestUpsertChunksDedup:
    """Test deduplication logic in upsert_chunks."""

    @pytest.mark.asyncio
    async def test_all_duplicates_skipped(self):
        """All chunks are duplicates → 0 inserted, all skipped."""
        client = AsyncMock()
        # Every dedup check returns a high-score match
        dup_point = _make_scored_point(score=0.98)
        client.query_points = AsyncMock(
            return_value=_make_query_response([dup_point])
        )
        client.upsert = AsyncMock()

        vectors = [DUMMY_VECTOR] * 3
        payloads = [_make_payload(chunk_index=i) for i in range(3)]

        result = await upsert_chunks(client, "auto_crawled", vectors, payloads)

        assert result.total == 3
        assert result.inserted == 0
        assert result.skipped == 3
        assert result.skipped_indices == [0, 1, 2]
        client.upsert.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_partial_duplicates(self):
        """Some chunks are duplicates, others are new."""
        client = AsyncMock()
        dup_point = _make_scored_point(score=0.98)
        # First chunk is dup, second is not, third is dup
        client.query_points = AsyncMock(
            side_effect=[
                _make_query_response([dup_point]),  # idx 0: dup
                _make_query_response([]),             # idx 1: new
                _make_query_response([dup_point]),  # idx 2: dup
            ]
        )
        client.upsert = AsyncMock()

        vectors = [DUMMY_VECTOR] * 3
        payloads = [_make_payload(chunk_index=i) for i in range(3)]

        result = await upsert_chunks(client, "auto_crawled", vectors, payloads)

        assert result.total == 3
        assert result.inserted == 1
        assert result.skipped == 2
        assert result.skipped_indices == [0, 2]
        assert client.upsert.await_count == 1

    @pytest.mark.asyncio
    async def test_different_source_urls_not_deduped(self):
        """Chunks from different source_urls are checked independently."""
        client = AsyncMock()
        client.query_points = AsyncMock(return_value=_make_query_response([]))
        client.upsert = AsyncMock()

        vectors = [DUMMY_VECTOR] * 2
        payloads = [
            _make_payload(source_url="https://a.com", chunk_index=0),
            _make_payload(source_url="https://b.com", chunk_index=0),
        ]

        result = await upsert_chunks(client, "auto_crawled", vectors, payloads)

        assert result.inserted == 2
        assert result.skipped == 0
        # Each chunk triggers its own dedup check with its own source_url
        assert client.query_points.await_count == 2

    @pytest.mark.asyncio
    async def test_skip_dedup_flag(self):
        """skip_dedup=True bypasses dedup checks entirely."""
        client = AsyncMock()
        client.upsert = AsyncMock()

        vectors = [DUMMY_VECTOR] * 3
        payloads = [_make_payload(chunk_index=i) for i in range(3)]

        result = await upsert_chunks(
            client, "auto_crawled", vectors, payloads, skip_dedup=True
        )

        assert result.total == 3
        assert result.inserted == 3
        assert result.skipped == 0
        # No dedup queries should have been made
        client.query_points.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_custom_dedup_threshold(self):
        """Custom dedup_threshold is forwarded to _check_duplicates."""
        client = AsyncMock()
        client.query_points = AsyncMock(return_value=_make_query_response([]))
        client.upsert = AsyncMock()

        vectors = [DUMMY_VECTOR]
        payloads = [_make_payload()]

        await upsert_chunks(
            client, "auto_crawled", vectors, payloads, dedup_threshold=0.99
        )

        call_kwargs = client.query_points.call_args.kwargs
        assert call_kwargs["score_threshold"] == 0.99

    @pytest.mark.asyncio
    async def test_idempotent_upsert(self):
        """Re-upserting the same data should skip all chunks (natural dedup)."""
        client = AsyncMock()
        # First call: no existing data → all inserted
        client.query_points = AsyncMock(return_value=_make_query_response([]))
        client.upsert = AsyncMock()

        vectors = [DUMMY_VECTOR] * 2
        payloads = [_make_payload(chunk_index=i) for i in range(2)]

        result1 = await upsert_chunks(client, "auto_crawled", vectors, payloads)
        assert result1.inserted == 2

        # Second call: data exists → all skipped via dedup
        dup_point = _make_scored_point(score=0.99)
        client.query_points = AsyncMock(
            return_value=_make_query_response([dup_point])
        )
        client.upsert.reset_mock()

        result2 = await upsert_chunks(client, "auto_crawled", vectors, payloads)
        assert result2.inserted == 0
        assert result2.skipped == 2


# ═══════════════════════════════════════════════════════════════════════════════
# upsert_chunks — batching
# ═══════════════════════════════════════════════════════════════════════════════


class TestUpsertChunksBatching:
    """Test batch splitting in upsert_chunks."""

    @pytest.mark.asyncio
    async def test_single_batch(self):
        """All chunks fit in one batch."""
        client = AsyncMock()
        client.query_points = AsyncMock(return_value=_make_query_response([]))
        client.upsert = AsyncMock()

        vectors = [DUMMY_VECTOR] * 5
        payloads = [_make_payload(chunk_index=i) for i in range(5)]

        await upsert_chunks(
            client, "auto_crawled", vectors, payloads, batch_size=10
        )

        assert client.upsert.await_count == 1
        points = client.upsert.call_args.kwargs["points"]
        assert len(points) == 5

    @pytest.mark.asyncio
    async def test_multiple_batches(self):
        """Chunks split across multiple batches."""
        client = AsyncMock()
        client.query_points = AsyncMock(return_value=_make_query_response([]))
        client.upsert = AsyncMock()

        vectors = [DUMMY_VECTOR] * 7
        payloads = [_make_payload(chunk_index=i) for i in range(7)]

        result = await upsert_chunks(
            client, "auto_crawled", vectors, payloads, batch_size=3
        )

        # 7 chunks / batch_size=3 → 3 batches (3, 3, 1)
        assert client.upsert.await_count == 3
        assert result.inserted == 7

    @pytest.mark.asyncio
    async def test_exact_batch_boundary(self):
        """Chunks that divide evenly into batches."""
        client = AsyncMock()
        client.query_points = AsyncMock(return_value=_make_query_response([]))
        client.upsert = AsyncMock()

        vectors = [DUMMY_VECTOR] * 6
        payloads = [_make_payload(chunk_index=i) for i in range(6)]

        await upsert_chunks(
            client, "auto_crawled", vectors, payloads, batch_size=3
        )

        # 6 / 3 = 2 batches
        assert client.upsert.await_count == 2

    @pytest.mark.asyncio
    async def test_batch_size_one(self):
        """Each chunk in its own batch."""
        client = AsyncMock()
        client.query_points = AsyncMock(return_value=_make_query_response([]))
        client.upsert = AsyncMock()

        vectors = [DUMMY_VECTOR] * 3
        payloads = [_make_payload(chunk_index=i) for i in range(3)]

        await upsert_chunks(
            client, "auto_crawled", vectors, payloads, batch_size=1
        )

        assert client.upsert.await_count == 3


# ═══════════════════════════════════════════════════════════════════════════════
# upsert_chunks — error handling
# ═══════════════════════════════════════════════════════════════════════════════


class TestUpsertChunksErrors:
    """Test error handling in upsert_chunks."""

    @pytest.mark.asyncio
    async def test_mismatched_lengths(self):
        """Vectors and payloads of different lengths → ValueError."""
        client = AsyncMock()
        with pytest.raises(ValueError, match="same length"):
            await upsert_chunks(
                client, "auto_crawled", [DUMMY_VECTOR], []
            )

    @pytest.mark.asyncio
    async def test_mismatched_lengths_reverse(self):
        """More payloads than vectors → ValueError."""
        client = AsyncMock()
        with pytest.raises(ValueError, match="same length"):
            await upsert_chunks(
                client, "auto_crawled", [], [_make_payload()]
            )

    @pytest.mark.asyncio
    async def test_upsert_exception_propagates(self):
        """Exception during client.upsert is re-raised."""
        client = AsyncMock()
        client.query_points = AsyncMock(return_value=_make_query_response([]))
        client.upsert = AsyncMock(side_effect=Exception("Qdrant error"))

        with pytest.raises(Exception, match="Qdrant error"):
            await upsert_chunks(
                client, "auto_crawled", [DUMMY_VECTOR], [_make_payload()]
            )

    @pytest.mark.asyncio
    async def test_dedup_error_does_not_block_upsert(self):
        """If dedup check fails, chunk is still upserted."""
        client = AsyncMock()
        client.query_points = AsyncMock(side_effect=Exception("network"))
        client.upsert = AsyncMock()

        result = await upsert_chunks(
            client, "auto_crawled", [DUMMY_VECTOR], [_make_payload()]
        )

        # Dedup error → not a dup → proceed with upsert
        assert result.inserted == 1
        assert result.skipped == 0


# ═══════════════════════════════════════════════════════════════════════════════
# upsert_chunks — payload coverage
# ═══════════════════════════════════════════════════════════════════════════════


class TestUpsertPayloadCoverage:
    """Verify that all PRD §6.2 payload fields are stored correctly."""

    @pytest.mark.asyncio
    async def test_all_payload_fields_stored(self):
        """All fields from ChunkPayload must be in the point's payload."""
        client = AsyncMock()
        client.query_points = AsyncMock(return_value=_make_query_response([]))
        client.upsert = AsyncMock()

        payload = ChunkPayload(
            text="Review text here",
            source_url="https://example.com/review",
            source_type="web",
            product_query="sony wh-1000xm5",
            chunk_index=2,
            language="ru",
            is_sponsored=True,
            is_curated=False,
            source_id="src-42",
            author="reviewer123",
            date="2026-01-15",
            session_id="sess-xyz",
        )

        await upsert_chunks(
            client, "auto_crawled", [DUMMY_VECTOR], [payload], skip_dedup=True
        )

        stored = client.upsert.call_args.kwargs["points"][0].payload
        assert stored["text"] == "Review text here"
        assert stored["source_url"] == "https://example.com/review"
        assert stored["source_type"] == "web"
        assert stored["product_query"] == "sony wh-1000xm5"
        assert stored["chunk_index"] == 2
        assert stored["language"] == "ru"
        assert stored["is_sponsored"] is True
        assert stored["is_curated"] is False
        assert stored["source_id"] == "src-42"
        assert stored["author"] == "reviewer123"
        assert stored["date"] == "2026-01-15"
        assert stored["session_id"] == "sess-xyz"

    @pytest.mark.asyncio
    async def test_curated_payload(self):
        """Curated chunks have is_curated=True in payload."""
        client = AsyncMock()
        client.upsert = AsyncMock()

        payload = _make_payload(is_curated=True, source_type="curated")
        await upsert_chunks(
            client, "curated_kb", [DUMMY_VECTOR], [payload], skip_dedup=True
        )

        stored = client.upsert.call_args.kwargs["points"][0].payload
        assert stored["is_curated"] is True
        assert stored["source_type"] == "curated"

    @pytest.mark.asyncio
    async def test_sponsored_payload(self):
        """Sponsored chunks have is_sponsored=True in payload."""
        client = AsyncMock()
        client.upsert = AsyncMock()

        payload = _make_payload(is_sponsored=True)
        await upsert_chunks(
            client, "auto_crawled", [DUMMY_VECTOR], [payload], skip_dedup=True
        )

        stored = client.upsert.call_args.kwargs["points"][0].payload
        assert stored["is_sponsored"] is True


# ═══════════════════════════════════════════════════════════════════════════════
# Vectorstore __init__ exports
# ═══════════════════════════════════════════════════════════════════════════════


class TestVectorstoreUpsertExports:
    """Verify that upsert-related symbols are exported from the package."""

    def test_chunk_payload_exported(self):
        from reviewmind.vectorstore import ChunkPayload as CP
        assert CP is ChunkPayload

    def test_upsert_result_exported(self):
        from reviewmind.vectorstore import UpsertResult as UR
        assert UR is UpsertResult

    def test_upsert_chunks_exported(self):
        from reviewmind.vectorstore import upsert_chunks as uc
        assert uc is upsert_chunks

    def test_generate_point_id_exported(self):
        from reviewmind.vectorstore import generate_point_id as gp
        assert gp is generate_point_id

    def test_dedup_threshold_exported(self):
        from reviewmind.vectorstore import DEDUP_SIMILARITY_THRESHOLD as dt
        assert dt == 0.95

    def test_batch_size_exported(self):
        from reviewmind.vectorstore import DEFAULT_UPSERT_BATCH_SIZE as bs
        assert bs == 64

    def test_all_exports_in___all__(self):
        import reviewmind.vectorstore as vs
        for name in [
            "ChunkPayload", "UpsertResult", "upsert_chunks",
            "generate_point_id", "DEDUP_SIMILARITY_THRESHOLD",
            "DEFAULT_UPSERT_BATCH_SIZE",
        ]:
            assert name in vs.__all__, f"{name} not in __all__"


# ═══════════════════════════════════════════════════════════════════════════════
# Integration scenarios
# ═══════════════════════════════════════════════════════════════════════════════


class TestIntegrationScenarios:
    """End-to-end scenarios combining multiple features."""

    @pytest.mark.asyncio
    async def test_fresh_ingest_full_pipeline(self):
        """Ingest 5 chunks into a fresh collection → all 5 inserted."""
        client = AsyncMock()
        client.query_points = AsyncMock(return_value=_make_query_response([]))
        client.upsert = AsyncMock()

        vectors = [DUMMY_VECTOR] * 5
        payloads = [
            _make_payload(
                source_url="https://youtube.com/v/abc",
                source_type="youtube",
                chunk_index=i,
                product_query="sony xm5",
            )
            for i in range(5)
        ]

        result = await upsert_chunks(client, "auto_crawled", vectors, payloads)

        assert result.total == 5
        assert result.inserted == 5
        assert result.skipped == 0

    @pytest.mark.asyncio
    async def test_reprocess_same_url_all_skipped(self):
        """Re-processing the same URL → all chunks skipped (dedup)."""
        client = AsyncMock()
        dup_point = _make_scored_point(score=0.99)
        client.query_points = AsyncMock(
            return_value=_make_query_response([dup_point])
        )
        client.upsert = AsyncMock()

        vectors = [DUMMY_VECTOR] * 5
        payloads = [
            _make_payload(
                source_url="https://youtube.com/v/abc",
                chunk_index=i,
            )
            for i in range(5)
        ]

        result = await upsert_chunks(client, "auto_crawled", vectors, payloads)

        assert result.total == 5
        assert result.inserted == 0
        assert result.skipped == 5
        client.upsert.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_mixed_sources_partial_dedup(self):
        """Mix of new and existing chunks from different sources."""
        client = AsyncMock()
        dup = _make_scored_point(score=0.97)
        # url-A idx=0 dup, url-A idx=1 new, url-B idx=0 new
        client.query_points = AsyncMock(
            side_effect=[
                _make_query_response([dup]),
                _make_query_response([]),
                _make_query_response([]),
            ]
        )
        client.upsert = AsyncMock()

        vectors = [DUMMY_VECTOR] * 3
        payloads = [
            _make_payload(source_url="https://a.com", chunk_index=0),
            _make_payload(source_url="https://a.com", chunk_index=1),
            _make_payload(source_url="https://b.com", chunk_index=0),
        ]

        result = await upsert_chunks(client, "auto_crawled", vectors, payloads)

        assert result.total == 3
        assert result.inserted == 2
        assert result.skipped == 1
        assert result.skipped_indices == [0]

    @pytest.mark.asyncio
    async def test_curated_seed_with_skip_dedup(self):
        """Seeding curated_kb with skip_dedup for performance."""
        client = AsyncMock()
        client.upsert = AsyncMock()

        vectors = [DUMMY_VECTOR] * 10
        payloads = [
            _make_payload(
                source_url="https://wirecutter.com/headphones",
                source_type="curated",
                is_curated=True,
                chunk_index=i,
            )
            for i in range(10)
        ]

        result = await upsert_chunks(
            client, "curated_kb", vectors, payloads,
            skip_dedup=True, batch_size=4,
        )

        assert result.total == 10
        assert result.inserted == 10
        assert result.skipped == 0
        # 10 / batch_size=4 → 3 batches (4, 4, 2)
        assert client.upsert.await_count == 3
        client.query_points.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_upsert_with_large_batch(self):
        """Upsert many chunks and verify correct batching."""
        client = AsyncMock()
        client.query_points = AsyncMock(return_value=_make_query_response([]))
        client.upsert = AsyncMock()

        n = 150
        vectors = [DUMMY_VECTOR] * n
        payloads = [_make_payload(chunk_index=i) for i in range(n)]

        result = await upsert_chunks(
            client, "auto_crawled", vectors, payloads, batch_size=64
        )

        assert result.inserted == n
        # 150 / 64 → 3 batches (64, 64, 22)
        assert client.upsert.await_count == 3
