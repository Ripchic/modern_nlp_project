"""reviewmind/services/product_extractor.py — Extract product names from user queries.

Uses a combination of LLM extraction (gpt-4o-mini) and regex heuristics
to identify product names from natural language queries in Russian and English.
"""

from __future__ import annotations

import json
import re

import structlog

from reviewmind.core.llm import LLMClient, LLMError

logger: structlog.stdlib.BoundLogger = structlog.get_logger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

#: System prompt for the product-name extraction LLM call.
_EXTRACTION_PROMPT = (
    "Ты — помощник, который извлекает названия товаров из пользовательских запросов.\n\n"
    "ПРАВИЛА:\n"
    '1. Верни JSON-объект вида {"products": ["Product1", "Product2"]}.\n'
    '2. Если в запросе НЕТ товара — верни {"products": []}.\n'
    "3. Извлекай ТОЛЬКО конкретные модели/названия товаров (бренд + модель).\n"
    "4. НЕ выдумывай товары, которых нет в запросе.\n"
    "5. Для сравнительных запросов ('X vs Y', 'X или Y') — верни оба товара.\n"
    "6. Общие категории ('наушники', 'ноутбуки') НЕ являются товарами — верни [].\n"
    "7. Отвечай ТОЛЬКО JSON, без пояснений.\n\n"
    "Примеры:\n"
    'Запрос: "Стоит ли покупать Sony WH-1000XM5?" → {"products": ["Sony WH-1000XM5"]}\n'
    'Запрос: "iPhone 16 vs Samsung S25" → {"products": ["iPhone 16", "Samsung S25"]}\n'
    'Запрос: "обзор Dyson V15 Detect" → {"products": ["Dyson V15 Detect"]}\n'
    'Запрос: "какие наушники лучше?" → {"products": []}\n'
    'Запрос: "Привет, как дела?" → {"products": []}\n'
)

#: Temperature for extraction (low for determinism).
_EXTRACTION_TEMPERATURE = 0.0

#: Max tokens for the response (just a small JSON).
_EXTRACTION_MAX_TOKENS = 200

# ── Regex patterns ─────────────────────────────────────────────────────────────

#: Regex heuristics for fallback product extraction.
#: Matches "brand + model" patterns like "Sony WH-1000XM5", "iPhone 16 Pro Max".
_PRODUCT_PATTERN = re.compile(
    r"""
    (?:                             # Brand-model patterns
        (?:Sony|Samsung|Apple|Google|Xiaomi|Huawei|OnePlus|Oppo|Vivo|
           Dyson|Bose|JBL|Sennheiser|AKG|Jabra|Anker|Marshall|
           LG|Asus|Acer|Lenovo|HP|Dell|MSI|Razer|
           Canon|Nikon|Fujifilm|Garmin|Fitbit|
           DJI|GoPro|Nintendo|PlayStation|Xbox)
        \s+
        [\w\-]+                     # Model identifier
        (?:\s+(?:Pro|Max|Plus|Ultra|Air|Mini|SE|Lite|Edge|Neo|Fold|Flip))*
    )
    |
    (?:                             # iPhone / iPad / MacBook / AirPods patterns
        (?:iPhone|iPad|MacBook|AirPods|iMac|Mac\s?Pro|Apple\s?Watch|Galaxy|Pixel)
        (?:\s+[\w\-]+)?             # Optional model number/suffix
        (?:\s+(?:Pro|Max|Plus|Ultra|Air|Mini|SE|Lite|Fold|Flip))*
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

#: Comparison connectors (Russian and English).
_COMPARISON_RE = re.compile(
    r"\b(?:vs\.?|versus|или|or|против|compared?\s+to|сравнить?\s+с)\b",
    re.IGNORECASE,
)


# ── Public API ─────────────────────────────────────────────────────────────────


async def extract_product(
    query: str,
    *,
    llm_client: LLMClient | None = None,
) -> list[str]:
    """Extract product names from a natural-language user query.

    Uses the LLM for robust extraction; falls back to regex heuristics
    if the LLM call fails.

    Parameters
    ----------
    query:
        The user's message or search query.
    llm_client:
        An optional :class:`~reviewmind.core.llm.LLMClient`.
        If *None*, one is created (and closed) internally.

    Returns
    -------
    list[str]
        A list of extracted product names.  Empty if no concrete product
        was identified.
    """
    if not query or not query.strip():
        return []

    query = query.strip()

    # Try LLM extraction first
    products = await _extract_via_llm(query, llm_client=llm_client)
    if products is not None:
        return products

    # Fallback to regex
    logger.info("product_extraction_regex_fallback", query=query[:80])
    return extract_product_regex(query)


def extract_product_regex(query: str) -> list[str]:
    """Regex-only product extraction (synchronous fallback).

    Returns
    -------
    list[str]
        Unique matched product names in order of appearance.
    """
    if not query or not query.strip():
        return []

    matches = _PRODUCT_PATTERN.findall(query)
    # Deduplicate while preserving order
    seen: set[str] = set()
    results: list[str] = []
    for m in matches:
        cleaned = m.strip()
        if cleaned and cleaned.lower() not in seen:
            seen.add(cleaned.lower())
            results.append(cleaned)
    return results


def is_comparison_query(query: str) -> bool:
    """Check whether *query* looks like a product comparison request."""
    if not query:
        return False
    return bool(_COMPARISON_RE.search(query))


# ── Internal helpers ───────────────────────────────────────────────────────────


async def _extract_via_llm(
    query: str,
    *,
    llm_client: LLMClient | None = None,
) -> list[str] | None:
    """Call the LLM to extract product names.

    Returns *None* when the LLM call fails (caller should fall back).
    """
    owns_client = llm_client is None
    client = llm_client or LLMClient()

    try:
        raw = await client.generate(
            system_prompt=_EXTRACTION_PROMPT,
            user_message=query,
            temperature=_EXTRACTION_TEMPERATURE,
            max_tokens=_EXTRACTION_MAX_TOKENS,
        )
        return _parse_llm_response(raw)

    except LLMError as exc:
        logger.warning("product_extraction_llm_error", error=str(exc))
        return None

    except Exception as exc:
        logger.warning("product_extraction_unexpected_error", error=str(exc))
        return None

    finally:
        if owns_client:
            await client.close()


def _parse_llm_response(raw: str) -> list[str] | None:
    """Parse the JSON response from the LLM.

    Returns *None* if parsing fails so the caller can fall back.
    """
    # Strip markdown fences if present
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = text.strip()

    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        logger.warning("product_extraction_json_parse_error", raw=raw[:200])
        return None

    if isinstance(data, dict) and "products" in data:
        products = data["products"]
        if isinstance(products, list):
            return [str(p).strip() for p in products if p and str(p).strip()]

    logger.warning("product_extraction_unexpected_format", raw=raw[:200])
    return None
