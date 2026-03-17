"""reviewmind/core/rag.py — RAG pipeline orchestrator.

Implements the full Retrieval-Augmented Generation pipeline:
1. Embed the user query via :class:`~reviewmind.core.embeddings.EmbeddingService`.
2. Hybrid search across ``curated_kb`` + ``auto_crawled`` via
   :func:`~reviewmind.vectorstore.search.hybrid_search`.
3. Rerank results via :func:`~reviewmind.core.reranker.rerank`.
4. Confidence check (≥3 chunks with score > 0.75).
5. Build context from up to 8 chunks + chat history.
6. Generate a structured answer via :meth:`~reviewmind.core.llm.LLMClient.generate_analysis`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog
from qdrant_client import AsyncQdrantClient

from reviewmind.core.embeddings import EmbeddingError, EmbeddingService
from reviewmind.core.llm import LLMClient, LLMError
from reviewmind.core.prompts import ChunkContext
from reviewmind.core.reranker import DEFAULT_RERANK_TOP_K, rerank
from reviewmind.services.language import detect_language
from reviewmind.vectorstore.search import SearchResult, hybrid_search

logger = structlog.get_logger("reviewmind.core.rag")

# ── Constants ────────────────────────────────────────────────────────────────

#: Minimum number of chunks with score above :data:`CONFIDENCE_SCORE_THRESHOLD`
#: required for the pipeline to consider the answer "confident".
CONFIDENCE_MIN_CHUNKS: int = 3

#: Minimum similarity score for a chunk to count towards the confidence check.
CONFIDENCE_SCORE_THRESHOLD: float = 0.75

#: Maximum number of chunks to include in the LLM context.
MAX_CONTEXT_CHUNKS: int = 8

#: Default number of results to request per collection in hybrid search.
DEFAULT_SEARCH_TOP_K: int = 5


# ── Data types ───────────────────────────────────────────────────────────────


@dataclass
class RAGResponse:
    """Result of a full RAG pipeline execution.

    Attributes
    ----------
    answer:
        The generated LLM analysis text.
    sources:
        List of source URLs used in the answer.
    used_curated:
        ``True`` if at least one curated (editorial) chunk was used.
    confidence_met:
        ``True`` if confidence check passed (≥3 chunks with score > 0.75).
    chunks_count:
        Total number of chunks fed to the LLM context.
    chunks_found:
        Total number of chunks returned from hybrid search (before rerank trim).
    used_sponsored:
        ``True`` if at least one sponsored chunk was used.
    error:
        Error message if the pipeline encountered a non-fatal issue
        (e.g. embedding failure caught gracefully).
    """

    answer: str
    sources: list[str] = field(default_factory=list)
    used_curated: bool = False
    confidence_met: bool = False
    chunks_count: int = 0
    chunks_found: int = 0
    used_sponsored: bool = False
    error: str | None = None


# ── Helpers ──────────────────────────────────────────────────────────────────


def _search_results_to_chunk_contexts(results: list[SearchResult]) -> list[ChunkContext]:
    """Convert :class:`SearchResult` objects into :class:`ChunkContext` objects
    suitable for prompt building.
    """
    return [
        ChunkContext(
            text=r.text,
            source_url=r.source_url,
            source_type=r.source_type,
            is_sponsored=r.is_sponsored,
            is_curated=r.is_curated,
            score=r.score,
            author=r.extra.get("author"),
            extra_metadata={
                "collection": r.collection,
                "language": r.language,
                "chunk_index": r.chunk_index,
                "point_id": r.point_id,
                "product_query": r.product_query,
            },
        )
        for r in results
    ]


def _extract_sources(results: list[SearchResult]) -> list[str]:
    """Return a deduplicated, order-preserving list of source URLs."""
    seen: set[str] = set()
    sources: list[str] = []
    for r in results:
        if r.source_url and r.source_url not in seen:
            seen.add(r.source_url)
            sources.append(r.source_url)
    return sources


def _check_confidence(results: list[SearchResult]) -> bool:
    """Return ``True`` if the confidence check passes.

    The check requires at least :data:`CONFIDENCE_MIN_CHUNKS` chunks
    with a score strictly above :data:`CONFIDENCE_SCORE_THRESHOLD`.
    """
    confident_count = sum(
        1 for r in results if r.score > CONFIDENCE_SCORE_THRESHOLD
    )
    return confident_count >= CONFIDENCE_MIN_CHUNKS


# ── RAG Pipeline ─────────────────────────────────────────────────────────────


class RAGPipeline:
    """Full Retrieval-Augmented Generation pipeline.

    Orchestrates: embed → search → rerank → confidence check → generate.

    Parameters
    ----------
    qdrant_client:
        An active :class:`AsyncQdrantClient` for vector search.
    embedding_service:
        An :class:`~reviewmind.core.embeddings.EmbeddingService` instance
        for query embedding.  If *None*, one is created lazily.
    llm_client:
        An :class:`~reviewmind.core.llm.LLMClient` instance for text
        generation.  If *None*, one is created lazily.
    search_top_k:
        Number of results per collection in hybrid search.
    rerank_top_k:
        Number of results to keep after reranking.
    """

    def __init__(
        self,
        qdrant_client: AsyncQdrantClient,
        embedding_service: EmbeddingService | None = None,
        llm_client: LLMClient | None = None,
        *,
        search_top_k: int = DEFAULT_SEARCH_TOP_K,
        rerank_top_k: int = DEFAULT_RERANK_TOP_K,
    ) -> None:
        self._qdrant = qdrant_client
        self._embedding: EmbeddingService | None = embedding_service
        self._llm: LLMClient | None = llm_client
        self._search_top_k = search_top_k
        self._rerank_top_k = rerank_top_k
        self._owns_embedding = embedding_service is None
        self._owns_llm = llm_client is None

    # ── Properties ────────────────────────────────────────────

    @property
    def embedding_service(self) -> EmbeddingService:
        """Return the embedding service, creating it lazily if needed."""
        if self._embedding is None:
            self._embedding = EmbeddingService()
        return self._embedding

    @property
    def llm_client(self) -> LLMClient:
        """Return the LLM client, creating it lazily if needed."""
        if self._llm is None:
            self._llm = LLMClient()
        return self._llm

    # ── Public API ────────────────────────────────────────────

    async def query(
        self,
        user_query: str,
        *,
        chat_history: list[dict[str, str]] | None = None,
        product_query: str | None = None,
        session_id: str | None = None,
    ) -> RAGResponse:
        """Execute the full RAG pipeline end-to-end.

        Steps:
        1. Embed the user query.
        2. Hybrid search across curated_kb + auto_crawled.
        3. Rerank results (curated boost, sponsored downweight).
        4. Confidence check (≥3 chunks with score > 0.75).
        5. Build context from up to 8 reranked chunks.
        6. Generate structured answer via LLM.

        Parameters
        ----------
        user_query:
            The user's natural-language question.
        chat_history:
            Optional prior messages for multi-turn context.
        product_query:
            Optional product-query string for Qdrant payload filtering.
        session_id:
            Optional session identifier for logging/tracking.

        Returns
        -------
        RAGResponse
            Contains the answer, sources, confidence state, and metadata.
        """
        log = logger.bind(
            session_id=session_id,
            product_query=product_query,
        )

        # ── Step 1: Embed query ──────────────────────────────
        log.info("rag_pipeline_start", user_query=user_query[:120])

        # Detect query language for response localisation
        query_language = detect_language(user_query)
        log.debug("rag_language_detected", language=query_language)

        try:
            query_vector = await self.embedding_service.embed_text(user_query)
        except EmbeddingError as exc:
            log.error("rag_embedding_error", error=str(exc))
            return RAGResponse(
                answer="",
                error=f"Embedding error: {exc}",
            )

        log.debug("rag_embedding_done", vector_dim=len(query_vector))

        # ── Step 2: Hybrid search ────────────────────────────
        try:
            search_results = await hybrid_search(
                client=self._qdrant,
                query_vector=query_vector,
                product_query=product_query,
                top_k=self._search_top_k,
            )
        except Exception as exc:
            log.error("rag_search_error", error=str(exc))
            return RAGResponse(
                answer="",
                error=f"Search error: {exc}",
            )

        log.info(
            "rag_search_done",
            results_count=len(search_results),
            curated_count=sum(1 for r in search_results if r.is_curated),
        )

        # ── Step 3: Rerank ───────────────────────────────────
        reranked = rerank(search_results, top_k=self._rerank_top_k)

        log.info(
            "rag_rerank_done",
            reranked_count=len(reranked),
            top_score=reranked[0].score if reranked else 0.0,
        )

        # ── Step 4: Confidence check ────────────────────────
        confidence_met = _check_confidence(reranked)

        log.info(
            "rag_confidence_check",
            confidence_met=confidence_met,
            confident_chunks=sum(
                1 for r in reranked if r.score > CONFIDENCE_SCORE_THRESHOLD
            ),
            threshold=CONFIDENCE_SCORE_THRESHOLD,
            min_required=CONFIDENCE_MIN_CHUNKS,
        )

        # ── Step 5: Build context ────────────────────────────
        context_chunks = reranked[:MAX_CONTEXT_CHUNKS]
        chunk_contexts = _search_results_to_chunk_contexts(context_chunks)
        sources = _extract_sources(context_chunks)

        used_curated = any(r.is_curated for r in context_chunks)
        used_sponsored = any(r.is_sponsored for r in context_chunks)

        log.info(
            "rag_context_built",
            context_chunks=len(chunk_contexts),
            sources_count=len(sources),
            used_curated=used_curated,
            used_sponsored=used_sponsored,
        )

        # ── Step 6: Generate answer ──────────────────────────
        try:
            answer = await self.llm_client.generate_analysis(
                user_query=user_query,
                chunks=chunk_contexts if chunk_contexts else None,
                chat_history=chat_history,
                language=query_language,
            )
        except LLMError as exc:
            log.error("rag_llm_error", error=str(exc))
            return RAGResponse(
                answer="",
                sources=sources,
                used_curated=used_curated,
                confidence_met=confidence_met,
                chunks_count=len(chunk_contexts),
                chunks_found=len(search_results),
                used_sponsored=used_sponsored,
                error=f"LLM error: {exc}",
            )

        log.info(
            "rag_pipeline_done",
            answer_length=len(answer),
            chunks_count=len(chunk_contexts),
            sources_count=len(sources),
            confidence_met=confidence_met,
        )

        return RAGResponse(
            answer=answer,
            sources=sources,
            used_curated=used_curated,
            confidence_met=confidence_met,
            chunks_count=len(chunk_contexts),
            chunks_found=len(search_results),
            used_sponsored=used_sponsored,
        )

    async def close(self) -> None:
        """Close owned resources (embedding service and LLM client).

        Only closes resources that were created lazily by this pipeline
        (not externally provided ones).
        """
        if self._owns_embedding and self._embedding is not None:
            await self._embedding.close()
            self._embedding = None
        if self._owns_llm and self._llm is not None:
            await self._llm.close()
            self._llm = None

    async def __aenter__(self) -> RAGPipeline:
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()
