# reviewmind/db/repositories/jobs.py — CRUD для jobs
from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from reviewmind.db.models import Job


class JobRepository:
    """CRUD operations for the `jobs` table."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_by_id(self, job_id: uuid.UUID) -> Job | None:
        result = await self._session.execute(select(Job).where(Job.id == job_id))
        return result.scalar_one_or_none()

    async def create(
        self,
        user_id: int,
        job_type: str,
        *,
        product_query: str | None = None,
        celery_task_id: str | None = None,
    ) -> Job:
        job = Job(
            id=uuid.uuid4(),
            user_id=user_id,
            job_type=job_type,
            status="pending",
            product_query=product_query,
            celery_task_id=celery_task_id,
        )
        self._session.add(job)
        await self._session.flush()
        return job

    async def update_status(
        self,
        job_id: uuid.UUID,
        status: str,
        *,
        celery_task_id: str | None = None,
        completed_at: datetime | None = None,
    ) -> Job | None:
        job = await self.get_by_id(job_id)
        if job is None:
            return None
        job.status = status
        if celery_task_id is not None:
            job.celery_task_id = celery_task_id
        if completed_at is not None:
            job.completed_at = completed_at
        await self._session.flush()
        return job

    async def list_by_user(self, user_id: int, *, status: str | None = None, limit: int = 20) -> list[Job]:
        stmt = select(Job).where(Job.user_id == user_id)
        if status is not None:
            stmt = stmt.where(Job.status == status)
        stmt = stmt.order_by(Job.created_at.desc()).limit(limit)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def delete(self, job_id: uuid.UUID) -> bool:
        job = await self.get_by_id(job_id)
        if job is None:
            return False
        await self._session.delete(job)
        await self._session.flush()
        return True
