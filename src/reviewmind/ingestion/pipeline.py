"""reviewmind.ingestion.pipeline — Оркестратор: parse → clean → sponsor detect → chunk → embed → upsert.

Full ingestion pipeline: determines URL type, invokes the appropriate scraper,
cleans text, detects sponsored content, chunks text, embeds chunks via OpenAI,
upserts vectors into Qdrant (auto_crawled), and persists source metadata in
PostgreSQL.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

import structlog
from qdrant_client import AsyncQdrantClient
from sqlalchemy.ext.asyncio import AsyncSession

from reviewmind.core.embeddings import EmbeddingError, EmbeddingService
from reviewmind.db.repositories.sources import SourceRepository
from reviewmind.ingestion.chunker import Chunk, chunk_text
from reviewmind.ingestion.cleaner import clean_text
from reviewmind.ingestion.sponsor import detect_sponsor_detailed
from reviewmind.scrapers.reddit import RedditScraper
from reviewmind.scrapers.web import WebScraper
from reviewmind.scrapers.youtube import YouTubeScraper
from reviewmind.vectorstore.client import ChunkPayload, UpsertResult, upsert_chunks
from reviewmind.vectorstore.collections import COLLECTION_AUTO_CRAWLED, SourceType

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# URL type detection (lightweight, inline — TASK-026 will add route_to_scraper)
# ---------------------------------------------------------------------------

_YOUTUBE_RE = re.compile(
    r"(?:youtube\.com|youtu\.be|youtube\.com/shorts|youtube\.com/embed|youtube\.com/live)",
    re.IGNORECASE,
)
_REDDIT_RE = re.compile(r"(?:reddit\.com|redd\.it)", re.IGNORECASE)


def detect_url_type(url: str) -> str:
    """Return ``'youtube'``, ``'reddit'``, or ``'web'`` for the given URL."""
    if _YOUTUBE_RE.search(url):
        return SourceType.YOUTUBE.value
    if _REDDIT_RE.search(url):
        return SourceType.REDDIT.value
    return SourceType.WEB.value


# ---------------------------------------------------------------------------
# Result data-classes
# ---------------------------------------------------------------------------


@dataclass
class SourceIngestionResult:
    """Result of ingesting a single URL."""

    url: str
    success: bool
    source_type: str = ""
    chunks_count: int = 0
    is_sponsored: bool = False
    error: str | None = None
    source_id: int | None = None


@dataclass
class IngestionResult:
    """Aggregate result returned by :meth:`IngestionPipeline.ingest_urls`."""

    success_count: int = 0
    failed_count: int = 0
    chunks_count: int = 0
    failed_urls: list[str] = field(default_factory=list)
    results: list[SourceIngestionResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


class IngestionPipeline:
    """Orchestrate: detect URL type → scrape → clean → sponsor detect → chunk → embed → upsert.

    Parameters
    ----------
    qdrant_client:
        An ``AsyncQdrantClient`` instance for upserting vectors.
    db_session:
        An ``AsyncSession`` (SQLAlchemy) for persisting source metadata.
    embedding_service:
        Optional pre-built :class:`EmbeddingService`. One is created on demand
        if not provided.
    collection_name:
        Qdrant collection to upsert into (default: ``auto_crawled``).
    """

    def __init__(
        self,
        qdrant_client: AsyncQdrantClient,
        db_session: AsyncSession | None = None,
        *,
        embedding_service: EmbeddingService | None = None,
        collection_name: str = COLLECTION_AUTO_CRAWLED,
    ) -> None:
        self._qdrant = qdrant_client
        self._db_session = db_session
        self._embedding: EmbeddingService | None = embedding_service
        self._owns_embedding = embedding_service is None
        self._collection = collection_name

        # Lazy-init scrapers
        self._youtube: YouTubeScraper | None = None
        self._reddit: RedditScraper | None = None
        self._web: WebScraper | None = None

    # -- lazy helpers -------------------------------------------------------

    @property
    def embedding_service(self) -> EmbeddingService:
        if self._embedding is None:
            self._embedding = EmbeddingService()
        return self._embedding

    def _get_youtube(self) -> YouTubeScraper:
        if self._youtube is None:
            self._youtube = YouTubeScraper()
        return self._youtube

    def _get_reddit(self) -> RedditScraper:
        if self._reddit is None:
            self._reddit = RedditScraper()
        return self._reddit

    def _get_web(self) -> WebScraper:
        if self._web is None:
            self._web = WebScraper()
        return self._web

    # -- public API ---------------------------------------------------------

    async def ingest_url(
        self,
        url: str,
        product_query: str,
        *,
        session_id: str | None = None,
        is_curated: bool = False,
    ) -> SourceIngestionResult:
        """Ingest a single URL through the full pipeline.

        Steps:
        1.  Detect URL type (youtube / reddit / web).
        2.  Scrape raw text + metadata via the appropriate parser.
        3.  Clean text (Unicode normalisation, HTML removal, etc.).
        4.  Detect sponsored content (regex heuristics).
        5.  Chunk text (``RecursiveCharacterTextSplitter``).
        6.  Embed chunks (``text-embedding-3-small``).
        7.  Upsert vectors + payload into Qdrant (``auto_crawled``).
        8.  Persist source metadata in PostgreSQL (``sources`` table).

        Returns a :class:`SourceIngestionResult` with success/failure details.
        """
        log = logger.bind(url=url, product_query=product_query)

        source_type = detect_url_type(url)
        log = log.bind(source_type=source_type)

        # ------ step 1: scrape ------
        try:
            raw_text, metadata = self._scrape(url, source_type)
        except Exception as exc:  # noqa: BLE001
            log.warning("scrape_failed", error=str(exc))
            return SourceIngestionResult(url=url, success=False, source_type=source_type, error=f"Scrape failed: {exc}")

        if not raw_text:
            log.info("scrape_empty")
            return SourceIngestionResult(url=url, success=False, source_type=source_type, error="No text extracted")

        # ------ step 2: clean ------
        cleaned = clean_text(raw_text)
        if not cleaned:
            log.info("clean_empty")
            return SourceIngestionResult(
                url=url, success=False, source_type=source_type, error="Text too short after cleaning",
            )

        # ------ step 3: sponsor detection ------
        sponsor_result = detect_sponsor_detailed(cleaned)
        is_sponsored = sponsor_result.is_sponsored
        log = log.bind(is_sponsored=is_sponsored)

        # ------ step 4: chunk ------
        chunk_metadata: dict = {
            "source_url": url,
            "source_type": source_type,
            "product_query": product_query,
            "is_sponsored": is_sponsored,
            "is_curated": is_curated,
        }
        if metadata.get("author"):
            chunk_metadata["author"] = metadata["author"]
        if metadata.get("language"):
            chunk_metadata["language"] = metadata["language"]

        chunks: list[Chunk] = chunk_text(cleaned, metadata=chunk_metadata)
        if not chunks:
            log.info("no_chunks")
            return SourceIngestionResult(url=url, success=False, source_type=source_type, error="No chunks produced")

        log = log.bind(chunks_count=len(chunks))

        # ------ step 5: embed ------
        try:
            texts = [c.text for c in chunks]
            vectors = await self.embedding_service.embed_batch(texts)
        except EmbeddingError as exc:
            log.error("embedding_failed", error=str(exc))
            return SourceIngestionResult(
                url=url, success=False, source_type=source_type, error=f"Embedding failed: {exc}",
            )

        # ------ step 6: upsert into Qdrant ------
        payloads = [
            ChunkPayload(
                text=c.text,
                source_url=url,
                source_type=source_type,
                product_query=product_query,
                chunk_index=c.chunk_index,
                language=metadata.get("language", ""),
                is_sponsored=is_sponsored,
                is_curated=is_curated,
                source_id=None,
                author=metadata.get("author"),
                date=metadata.get("date"),
                session_id=session_id,
            )
            for c in chunks
        ]

        try:
            upsert_result: UpsertResult = await upsert_chunks(
                self._qdrant,
                self._collection,
                vectors,
                payloads,
            )
        except Exception as exc:  # noqa: BLE001
            log.error("upsert_failed", error=str(exc))
            return SourceIngestionResult(
                url=url, success=False, source_type=source_type, error=f"Upsert failed: {exc}",
            )

        log.info(
            "upsert_done",
            inserted=upsert_result.inserted,
            skipped=upsert_result.skipped,
        )

        # ------ step 7: persist source in PostgreSQL ------
        source_id: int | None = None
        if self._db_session is not None:
            try:
                source_id = await self._persist_source(
                    url=url,
                    source_type=source_type,
                    product_query=product_query,
                    is_sponsored=is_sponsored,
                    is_curated=is_curated,
                    language=metadata.get("language"),
                    author=metadata.get("author"),
                )
            except Exception as exc:  # noqa: BLE001
                log.error("db_persist_failed", error=str(exc))
                # Source metadata failure is non-fatal; chunks are already in Qdrant.

        return SourceIngestionResult(
            url=url,
            success=True,
            source_type=source_type,
            chunks_count=upsert_result.inserted,
            is_sponsored=is_sponsored,
            source_id=source_id,
        )

    async def ingest_urls(
        self,
        urls: list[str],
        product_query: str,
        *,
        session_id: str | None = None,
        is_curated: bool = False,
    ) -> IngestionResult:
        """Ingest multiple URLs.  Failure of one URL does not block the rest."""
        result = IngestionResult()
        for url in urls:
            single = await self.ingest_url(
                url,
                product_query,
                session_id=session_id,
                is_curated=is_curated,
            )
            result.results.append(single)
            if single.success:
                result.success_count += 1
                result.chunks_count += single.chunks_count
            else:
                result.failed_count += 1
                result.failed_urls.append(url)
        return result

    # -- private helpers ----------------------------------------------------

    def _scrape(self, url: str, source_type: str) -> tuple[str, dict]:
        """Return ``(raw_text, metadata_dict)`` for the given URL."""
        metadata: dict = {}

        if source_type == SourceType.YOUTUBE.value:
            scraper = self._get_youtube()
            result = scraper.get_transcript_by_url(url)
            if result is None:
                return "", metadata
            metadata["language"] = result.language_code or result.language
            metadata["author"] = result.extra_metadata.get("author")
            metadata["date"] = result.extra_metadata.get("date")
            return result.text, metadata

        if source_type == SourceType.REDDIT.value:
            scraper = self._get_reddit()
            post = scraper.parse_url(url)
            if post is None:
                return "", metadata
            metadata["author"] = post.author
            metadata["language"] = None  # Reddit doesn't report language
            metadata["date"] = None
            return post.full_text, metadata

        # default: web
        scraper = self._get_web()
        page = scraper.parse_url(url)
        if page is None:
            return "", metadata
        metadata["author"] = page.author
        metadata["language"] = page.language
        metadata["date"] = page.date
        return page.text, metadata

    async def _persist_source(
        self,
        *,
        url: str,
        source_type: str,
        product_query: str,
        is_sponsored: bool,
        is_curated: bool,
        language: str | None,
        author: str | None,
    ) -> int | None:
        """Persist (or update) source metadata in PostgreSQL.  Returns source ID."""
        assert self._db_session is not None  # noqa: S101
        repo = SourceRepository(self._db_session)
        source, created = await repo.get_or_create(
            source_url=url,
            source_type=source_type,
            product_query=product_query,
            parsed_at=datetime.now(timezone.utc),
            is_sponsored=is_sponsored,
            is_curated=is_curated,
            language=language,
            author=author,
        )
        if not created:
            # Update existing source metadata
            await repo.update(
                source.id,
                parsed_at=datetime.now(timezone.utc),
                is_sponsored=is_sponsored,
                product_query=product_query,
                language=language,
                author=author,
            )
        return source.id

    # -- lifecycle ----------------------------------------------------------

    async def close(self) -> None:
        """Release owned resources."""
        if self._owns_embedding and self._embedding is not None:
            await self._embedding.close()
            self._embedding = None

    async def __aenter__(self) -> IngestionPipeline:
        return self

    async def __aexit__(self, *exc_info) -> None:
        await self.close()
