#!/usr/bin/env python3
"""Инициализация Qdrant коллекций auto_crawled и curated_kb.

Скрипт идемпотентен — безопасен для повторного запуска.

Usage::

    python scripts/init_qdrant.py
    # or with custom URL:
    QDRANT_URL=http://localhost:6333 python scripts/init_qdrant.py
"""

from __future__ import annotations

import asyncio
import sys

import structlog

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger("scripts.init_qdrant")


async def main() -> None:
    """Create all required Qdrant collections with payload indexes."""
    from reviewmind.vectorstore.client import QdrantClientWrapper
    from reviewmind.vectorstore.collections import ensure_all_collections

    async with QdrantClientWrapper() as wrapper:
        # Verify connectivity
        healthy = await wrapper.health_check()
        if not healthy:
            logger.error("qdrant_unreachable", url=wrapper._url)
            sys.exit(1)

        logger.info("qdrant_connected", url=wrapper._url)

        results = await ensure_all_collections(wrapper.client)

        for name, created in results.items():
            status = "created" if created else "already_exists"
            logger.info("collection_status", collection=name, status=status)

        logger.info("init_complete", collections=len(results))


if __name__ == "__main__":
    asyncio.run(main())
