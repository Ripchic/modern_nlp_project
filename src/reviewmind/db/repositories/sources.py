# reviewmind/db/repositories/sources.py — CRUD для sources
from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from reviewmind.db.models import Source


class SourceRepository:
    """CRUD operations for the `sources` table."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_id(self, source_id: int) -> Source | None:
        result = await self._session.execute(select(Source).where(Source.id == source_id))
        return result.scalar_one_or_none()

    async def get_by_url(self, source_url: str) -> Source | None:
        result = await self._session.execute(select(Source).where(Source.source_url == source_url))
        return result.scalar_one_or_none()

    async def create(
        self,
        source_url: str,
        source_type: str,
        *,
        product_query: str | None = None,
        parsed_at: datetime | None = None,
        is_sponsored: bool = False,
        is_curated: bool = False,
        language: str | None = None,
        author: str | None = None,
    ) -> Source:
        source = Source(
            source_url=source_url,
            source_type=source_type,
            product_query=product_query,
            parsed_at=parsed_at,
            is_sponsored=is_sponsored,
            is_curated=is_curated,
            language=language,
            author=author,
        )
        self._session.add(source)
        await self._session.flush()
        return source

    async def get_or_create(self, source_url: str, source_type: str, **kwargs) -> tuple[Source, bool]:
        """Return (source, created)."""
        source = await self.get_by_url(source_url)
        if source is not None:
            return source, False
        source = await self.create(source_url, source_type, **kwargs)
        return source, True

    async def update(self, source_id: int, **kwargs) -> Source | None:
        source = await self.get_by_id(source_id)
        if source is None:
            return None
        for key, value in kwargs.items():
            setattr(source, key, value)
        await self._session.flush()
        return source

    async def delete(self, source_id: int) -> bool:
        source = await self.get_by_id(source_id)
        if source is None:
            return False
        await self._session.delete(source)
        await self._session.flush()
        return True
