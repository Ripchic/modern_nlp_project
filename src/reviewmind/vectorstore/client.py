"""reviewmind/vectorstore/client.py — Async Qdrant client wrapper."""

from __future__ import annotations

import structlog
from qdrant_client import AsyncQdrantClient

logger = structlog.get_logger("reviewmind.vectorstore.client")


class QdrantClientWrapper:
    """Async wrapper around Qdrant client with lifecycle management.

    Usage::

        async with QdrantClientWrapper() as wrapper:
            exists = await wrapper.client.collection_exists("my_col")

    Or create explicitly::

        wrapper = QdrantClientWrapper(url="http://localhost:6333")
        await wrapper.close()
    """

    def __init__(
        self,
        url: str | None = None,
        timeout: int = 10,
    ) -> None:
        if url is None:
            from reviewmind.config import settings

            url = settings.qdrant_url

        self._url = url
        self._timeout = timeout
        self._client: AsyncQdrantClient | None = None

    @property
    def client(self) -> AsyncQdrantClient:
        """Return the underlying async Qdrant client, creating it lazily."""
        if self._client is None:
            self._client = AsyncQdrantClient(url=self._url, timeout=self._timeout)
            logger.info("qdrant_client_created", url=self._url)
        return self._client

    async def close(self) -> None:
        """Close the underlying Qdrant client connection."""
        if self._client is not None:
            await self._client.close()
            logger.info("qdrant_client_closed")
            self._client = None

    async def __aenter__(self) -> QdrantClientWrapper:
        # Ensure the client is created on enter
        _ = self.client
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def health_check(self) -> bool:
        """Return True if Qdrant is reachable, False otherwise."""
        try:
            await self.client.get_collections()
            return True
        except Exception:
            return False
