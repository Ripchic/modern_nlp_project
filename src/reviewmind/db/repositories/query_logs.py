# reviewmind/db/repositories/query_logs.py — CRUD для query_logs
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from reviewmind.db.models import QueryLog


class QueryLogRepository:
    """CRUD operations for the `query_logs` table."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        user_id: int,
        *,
        session_id: str | None = None,
        mode: str | None = None,
        query_text: str | None = None,
        response_text: str | None = None,
        sources_used: list | None = None,
        response_time_ms: int | None = None,
        used_tavily: bool = False,
    ) -> QueryLog:
        log = QueryLog(
            user_id=user_id,
            session_id=session_id,
            mode=mode,
            query_text=query_text,
            response_text=response_text,
            sources_used=sources_used,
            response_time_ms=response_time_ms,
            used_tavily=used_tavily,
        )
        self._session.add(log)
        await self._session.flush()
        return log

    async def get_by_id(self, log_id: int) -> QueryLog | None:
        result = await self._session.execute(select(QueryLog).where(QueryLog.id == log_id))
        return result.scalar_one_or_none()

    async def update_rating(self, log_id: int, rating: int) -> QueryLog | None:
        log = await self.get_by_id(log_id)
        if log is None:
            return None
        log.rating = rating
        await self._session.flush()
        return log

    async def list_by_user(self, user_id: int, *, limit: int = 20) -> list[QueryLog]:
        result = await self._session.execute(
            select(QueryLog).where(QueryLog.user_id == user_id).order_by(QueryLog.created_at.desc()).limit(limit)
        )
        return list(result.scalars().all())
