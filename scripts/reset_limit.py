"""Reset daily limit counter for admin user."""
import asyncio
from datetime import datetime, timezone
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from reviewmind.config import get_settings
from reviewmind.db.repositories.limits import UserLimitRepository


async def reset():
    settings = get_settings()
    engine = create_async_engine(settings.database_url)
    factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as session:
        repo = UserLimitRepository(session)
        today = datetime.now(tz=timezone.utc).date()
        row = await repo.get(857056289, today)
        if row:
            print(f"Before: requests_used={row.requests_used}")
            row.requests_used = 0
            await session.commit()
            print("Reset to 0")
        else:
            print("No limit row found for today")
    await engine.dispose()


asyncio.run(reset())
