"""reviewmind/services/comparison_service.py — Product comparison via parallel RAG.

Detects comparison queries ("X vs Y"), runs parallel RAG for each product,
and generates a structured comparison table via the LLM.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import structlog

from reviewmind.services.language import detect_language
from reviewmind.services.product_extractor import extract_product, is_comparison_query

if TYPE_CHECKING:
    from qdrant_client import AsyncQdrantClient

    from reviewmind.core.llm import LLMClient
    from reviewmind.core.rag import RAGResponse

logger = structlog.get_logger("reviewmind.services.comparison")

# ── Constants ────────────────────────────────────────────────────────────────

#: Minimum number of products required for a comparison.
MIN_PRODUCTS_FOR_COMPARISON: int = 2

#: Maximum number of products supported in a single comparison.
MAX_PRODUCTS_FOR_COMPARISON: int = 4

#: System prompt template for comparison generation.
COMPARISON_PROMPT_TEMPLATE = """\
Ты — беспристрастный аналитик потребительских отзывов. Твоя задача — сравнить товары
на основе реальных обзоров и предоставить сбалансированное сравнение.

ПРАВИЛА:
1. Отвечай ТОЛЬКО на основе предоставленного контекста. Не выдумывай факты.
2. Явно выделяй различия и сходства между товарами.
3. Источники с флагом [sponsored] считай менее достоверными, указывай это.
4. Структурируй ответ как таблицу сравнения с категориями.
5. Если данных по одному из товаров недостаточно — честно укажи это.
6. Отвечай на языке пользователя (определённый язык: {response_language}).
7. В конце дай краткий вывод: какой товар лучше подходит для каких сценариев.
8. Укажи количество источников и наличие проверенных материалов (📚).

ПАРАМЕТРЫ ГЕНЕРАЦИИ: temperature=0.3, max_tokens=1000, top_p=0.9

{product_sections}

ИСТОРИЯ ДИАЛОГА:
{chat_history}"""

#: Template for a single product's context section.
_PRODUCT_SECTION_TEMPLATE = """\
--- ДАННЫЕ О ТОВАРЕ: {product_name} ---
{chunks}"""

#: Message when no data is available for a product.
_NO_DATA_MSG = "Данные по этому товару отсутствуют."


# ── Data types ───────────────────────────────────────────────────────────────


@dataclass
class ProductRAGResult:
    """RAG result for a single product within a comparison."""

    product_name: str
    rag_response: RAGResponse | None = None
    error: str | None = None

    @property
    def has_data(self) -> bool:
        """``True`` if there is a valid answer with chunks."""
        return (
            self.rag_response is not None
            and self.rag_response.chunks_count > 0
        )


@dataclass
class ComparisonResult:
    """Result of a product comparison."""

    answer: str
    products: list[str] = field(default_factory=list)
    product_results: list[ProductRAGResult] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    used_curated: bool = False
    total_chunks: int = 0
    error: str | None = None

    @property
    def has_data_for_all(self) -> bool:
        """``True`` if every product has at least some data."""
        return all(pr.has_data for pr in self.product_results)


# ── Public API ───────────────────────────────────────────────────────────────


def detect_comparison(query: str) -> bool:
    """Check whether *query* is a product comparison request.

    Delegates to :func:`~reviewmind.services.product_extractor.is_comparison_query`.
    """
    return is_comparison_query(query)


async def compare_products(
    query: str,
    *,
    qdrant_client: AsyncQdrantClient,
    llm_client: LLMClient | None = None,
    chat_history: list[dict[str, str]] | None = None,
) -> ComparisonResult:
    """Run parallel RAG queries for each product and generate a comparison.

    Parameters
    ----------
    query:
        The user's comparison query (e.g. "iPhone 16 vs Samsung S25").
    qdrant_client:
        Active Qdrant client for vector search.
    llm_client:
        Optional LLM client.  If *None*, one is created internally.
    chat_history:
        Optional prior messages for multi-turn context.

    Returns
    -------
    ComparisonResult
    """
    log = logger.bind(query=query[:120])

    # ── Step 1: Extract product names ────────────────────────
    try:
        product_names = await extract_product(query)
    except Exception:
        product_names = []

    if len(product_names) < MIN_PRODUCTS_FOR_COMPARISON:
        log.info("comparison_too_few_products", count=len(product_names))
        return ComparisonResult(
            answer="",
            products=product_names,
            error="Не удалось определить два или более товаров для сравнения.",
        )

    # Cap at MAX_PRODUCTS_FOR_COMPARISON
    product_names = product_names[:MAX_PRODUCTS_FOR_COMPARISON]
    log.info("comparison_start", products=product_names)

    # ── Step 2: Parallel RAG for each product ────────────────
    product_results = await _parallel_rag(
        product_names=product_names,
        user_query=query,
        qdrant_client=qdrant_client,
        chat_history=chat_history,
    )

    # ── Step 3: Build comparison prompt and generate ─────────
    language = detect_language(query)
    all_sources: list[str] = []
    used_curated = False
    total_chunks = 0

    for pr in product_results:
        if pr.rag_response:
            all_sources.extend(pr.rag_response.sources)
            if pr.rag_response.used_curated:
                used_curated = True
            total_chunks += pr.rag_response.chunks_count

    # Deduplicate sources while preserving order
    seen: set[str] = set()
    unique_sources: list[str] = []
    for s in all_sources:
        if s not in seen:
            seen.add(s)
            unique_sources.append(s)

    # ── Step 4: Generate comparison via LLM ──────────────────
    from reviewmind.core.llm import LLMClient as _LLMClient  # noqa: PLC0415
    from reviewmind.core.llm import LLMError  # noqa: PLC0415
    from reviewmind.core.prompts import RAG_MAX_TOKENS, RAG_TEMPERATURE, RAG_TOP_P  # noqa: PLC0415

    owns_llm = llm_client is None
    client = llm_client or _LLMClient()

    try:
        system_prompt = _build_comparison_prompt(
            product_results=product_results,
            chat_history=chat_history,
            language=language,
        )

        answer = await client.generate(
            system_prompt=system_prompt,
            user_message=query,
            temperature=RAG_TEMPERATURE,
            max_tokens=RAG_MAX_TOKENS,
            top_p=RAG_TOP_P,
        )
    except LLMError as exc:
        log.error("comparison_llm_error", error=str(exc))
        return ComparisonResult(
            answer="",
            products=product_names,
            product_results=product_results,
            sources=unique_sources,
            used_curated=used_curated,
            total_chunks=total_chunks,
            error=f"LLM error: {exc}",
        )
    finally:
        if owns_llm:
            await client.close()

    log.info(
        "comparison_done",
        answer_len=len(answer),
        products=len(product_names),
        total_chunks=total_chunks,
        sources=len(unique_sources),
    )

    return ComparisonResult(
        answer=answer,
        products=product_names,
        product_results=product_results,
        sources=unique_sources,
        used_curated=used_curated,
        total_chunks=total_chunks,
    )


# ── Internal helpers ─────────────────────────────────────────────────────────


async def _parallel_rag(
    *,
    product_names: list[str],
    user_query: str,
    qdrant_client: AsyncQdrantClient,
    chat_history: list[dict[str, str]] | None = None,
) -> list[ProductRAGResult]:
    """Run RAG queries for each product in parallel using ``asyncio.gather``."""

    async def _single_rag(product_name: str) -> ProductRAGResult:
        try:
            from reviewmind.core.rag import RAGPipeline  # noqa: PLC0415

            async with RAGPipeline(qdrant_client=qdrant_client) as rag:
                response = await rag.query(
                    user_query=user_query,
                    product_query=product_name,
                    chat_history=chat_history,
                )
            return ProductRAGResult(product_name=product_name, rag_response=response)
        except Exception as exc:
            logger.warning(
                "comparison_single_rag_error",
                product=product_name,
                error=str(exc),
            )
            return ProductRAGResult(
                product_name=product_name,
                error=str(exc),
            )

    results = await asyncio.gather(
        *[_single_rag(name) for name in product_names],
        return_exceptions=False,
    )
    return list(results)


def _build_comparison_prompt(
    *,
    product_results: list[ProductRAGResult],
    chat_history: list[dict[str, str]] | None = None,
    language: str | None = None,
) -> str:
    """Build the full comparison system prompt from per-product RAG results."""
    from reviewmind.core.prompts import DEFAULT_PROMPT_LANGUAGE, format_chat_history  # noqa: PLC0415

    lang = language or DEFAULT_PROMPT_LANGUAGE
    sections: list[str] = []

    for pr in product_results:
        if pr.rag_response and pr.rag_response.chunks_count > 0:
            # Reconstruct ChunkContexts from the RAG response's data.
            # The RAG pipeline already executed search + rerank; we re-derive
            # chunk contexts from the stored answer metadata.  Since RAGResponse
            # doesn't store raw SearchResults, we build a minimal context block
            # by using the sources and the answer text.
            #
            # However, since we control the RAG pipeline call, we can actually
            # do a lightweight search+rerank ourselves to get the chunks.
            # But that would double the work.  Instead, we note that the
            # RAG response *answer* already contains the analysis.  For the
            # comparison prompt, we embed the per-product RAG *answer* as
            # the context block so the comparison LLM can synthesise them.
            chunk_text = pr.rag_response.answer if pr.rag_response.answer else _NO_DATA_MSG
            section = _PRODUCT_SECTION_TEMPLATE.format(
                product_name=pr.product_name,
                chunks=chunk_text,
            )
        else:
            reason = pr.error or _NO_DATA_MSG
            section = _PRODUCT_SECTION_TEMPLATE.format(
                product_name=pr.product_name,
                chunks=reason,
            )
        sections.append(section)

    product_sections = "\n\n".join(sections)
    history_text = format_chat_history(chat_history or [])

    return COMPARISON_PROMPT_TEMPLATE.format(
        product_sections=product_sections,
        chat_history=history_text,
        response_language=lang,
    )
