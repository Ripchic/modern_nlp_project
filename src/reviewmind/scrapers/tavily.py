"""reviewmind/scrapers/tavily.py — Tavily API web search fallback.

Provides :class:`TavilyScraper` for web searching via the Tavily API.
Used as a fallback when the RAG pipeline lacks sufficient data
(< 3 confident chunks) to answer a query.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog
from tavily import AsyncTavilyClient, InvalidAPIKeyError, MissingAPIKeyError, UsageLimitExceededError

logger = structlog.get_logger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

#: Default maximum number of results to return from a Tavily search.
DEFAULT_MAX_RESULTS: int = 5

#: Default search depth — ``"basic"`` is cheaper; ``"advanced"`` is slower but richer.
DEFAULT_SEARCH_DEPTH: str = "basic"

#: Minimum text length (chars) to consider a result useful.
MIN_CONTENT_LENGTH: int = 50

#: Default timeout for a single Tavily search call (seconds).
DEFAULT_TIMEOUT: float = 30.0


# ── Data types ───────────────────────────────────────────────────────────────


@dataclass
class TavilyResult:
    """A single result from the Tavily API search.

    Attributes
    ----------
    url:
        The URL of the search result page.
    title:
        The page title returned by Tavily.
    content:
        The extracted/summarised text content (clean, not HTML).
    score:
        Relevance score assigned by Tavily (0.0–1.0).
    raw_content:
        Optional raw content (only if ``include_raw_content`` is requested).
    extra:
        Any extra metadata from the API response.
    """

    url: str
    title: str = ""
    content: str = ""
    score: float = 0.0
    raw_content: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def word_count(self) -> int:
        """Number of whitespace-delimited words in :attr:`content`."""
        return len(self.content.split()) if self.content else 0


# ── Scraper class ────────────────────────────────────────────────────────────


class TavilyScraper:
    """Search the web using the `Tavily API <https://tavily.com/>`_.

    Parameters
    ----------
    api_key:
        Tavily API key.  If *None*, the key is read lazily from
        :attr:`reviewmind.config.settings.tavily_api_key`.
    max_results:
        Default maximum number of results per search.
    search_depth:
        Search depth (``"basic"`` or ``"advanced"``).
    timeout:
        Request timeout in seconds.
    """

    def __init__(
        self,
        api_key: str | None = None,
        *,
        max_results: int = DEFAULT_MAX_RESULTS,
        search_depth: str = DEFAULT_SEARCH_DEPTH,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._api_key = api_key
        self._max_results = max_results
        self._search_depth = search_depth
        self._timeout = timeout
        self._client: AsyncTavilyClient | None = None

    # ── Lazy client ──────────────────────────────────────────

    def _get_api_key(self) -> str:
        """Resolve the API key, falling back to settings."""
        if self._api_key:
            return self._api_key
        from reviewmind.config import settings  # noqa: PLC0415

        return settings.tavily_api_key

    def _get_client(self) -> AsyncTavilyClient:
        """Return the async Tavily client, creating it lazily."""
        if self._client is None:
            api_key = self._get_api_key()
            if not api_key:
                raise ValueError("Tavily API key is not configured. Set TAVILY_API_KEY in .env.")
            self._client = AsyncTavilyClient(api_key=api_key)
        return self._client

    # ── Public API ───────────────────────────────────────────

    async def search(
        self,
        query: str,
        *,
        max_results: int | None = None,
        search_depth: str | None = None,
        include_raw_content: bool = False,
    ) -> list[TavilyResult]:
        """Execute a web search via the Tavily API.

        Parameters
        ----------
        query:
            The search query string.
        max_results:
            Override default ``max_results``.
        search_depth:
            Override default ``search_depth``.
        include_raw_content:
            If *True*, populate :attr:`TavilyResult.raw_content`.

        Returns
        -------
        list[TavilyResult]
            Parsed results sorted by relevance score (descending).
            Empty list on error or if no valid results are found.
        """
        if not query or not query.strip():
            logger.warning("tavily_search_empty_query")
            return []

        effective_max = max_results if max_results is not None else self._max_results
        effective_depth = search_depth or self._search_depth

        log = logger.bind(query=query[:120], max_results=effective_max, search_depth=effective_depth)
        log.info("tavily_search_start")

        try:
            client = self._get_client()
        except ValueError as exc:
            log.error("tavily_no_api_key", error=str(exc))
            return []

        try:
            response = await client.search(
                query=query,
                max_results=effective_max,
                search_depth=effective_depth,
                include_raw_content=include_raw_content,
                timeout=self._timeout,
            )
        except (InvalidAPIKeyError, MissingAPIKeyError) as exc:
            log.error("tavily_auth_error", error=str(exc))
            return []
        except UsageLimitExceededError as exc:
            log.error("tavily_usage_limit", error=str(exc))
            return []
        except Exception as exc:
            log.error("tavily_search_error", error=str(exc), error_type=type(exc).__name__)
            return []

        results = self._parse_response(response, include_raw_content=include_raw_content)

        log.info("tavily_search_done", results_count=len(results))
        return results

    # ── Parsing helpers ──────────────────────────────────────

    def _parse_response(
        self,
        response: dict[str, Any],
        *,
        include_raw_content: bool = False,
    ) -> list[TavilyResult]:
        """Parse the raw Tavily API response into :class:`TavilyResult` objects.

        Filters out results with content shorter than :data:`MIN_CONTENT_LENGTH`.
        """
        raw_results: list[dict[str, Any]] = response.get("results", [])

        parsed: list[TavilyResult] = []
        for item in raw_results:
            content = item.get("content", "")
            if len(content) < MIN_CONTENT_LENGTH:
                continue

            result = TavilyResult(
                url=item.get("url", ""),
                title=item.get("title", ""),
                content=content,
                score=float(item.get("score", 0.0)),
                raw_content=item.get("raw_content") if include_raw_content else None,
                extra={
                    k: v
                    for k, v in item.items()
                    if k not in {"url", "title", "content", "score", "raw_content"}
                },
            )
            parsed.append(result)

        # Sort by score descending
        parsed.sort(key=lambda r: r.score, reverse=True)
        return parsed
