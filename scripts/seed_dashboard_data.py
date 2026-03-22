#!/usr/bin/env python3
"""Seed PostgreSQL with realistic historical data for Grafana dashboard testing.

Inserts ~30 days of activity across all tables:
- users (10 users)
- sources (50 sources across types)
- jobs (40 jobs)
- query_logs (300 log entries with realistic distributions)
- subscriptions (8 conversions)
- user_limits (daily limit tracking)

Usage:
    python scripts/seed_dashboard_data.py
"""
from __future__ import annotations

import os
import random
import uuid
from datetime import date, datetime, timedelta, timezone

import psycopg2

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://reviewmind:changeme@localhost:5433/reviewmind",
)

# Strip async driver prefix if present
DB_URL = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://").replace(
    "@postgres:", "@localhost:5433/".split("/")[0] + "@localhost:5433/"
    if "@postgres:" in DATABASE_URL
    else "@localhost:5433/"
)

# Simple sync URL for psycopg2
if "postgres:5432" in DATABASE_URL:
    DSN = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://").replace(
        "@postgres:5432/", "@localhost:5433/"
    )
elif "localhost" in DATABASE_URL and "5433" in DATABASE_URL:
    DSN = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")
else:
    # default local dev
    DSN = "postgresql://reviewmind:changeme@localhost:5433/reviewmind"

NOW = datetime.now(timezone.utc)
DAYS = 30

PRODUCTS = [
    "Sony WH-1000XM5",
    "iPhone 16 Pro",
    "Dyson V15 Detect",
    "Samsung Galaxy S25",
    "MacBook Pro M4",
    "AirPods Pro 2",
    "Xiaomi 14 Ultra",
    "LG OLED C4",
]

SOURCE_TYPES = ["youtube", "reddit", "web", "tavily", "curated"]
SOURCE_URLS = [
    ("youtube", "https://youtube.com/watch?v=test{i}", False, False),
    ("reddit", "https://reddit.com/r/headphones/test{i}", False, False),
    ("web", "https://rtings.com/review/test{i}", False, False),
    ("tavily", "https://tavily-result.com/test{i}", False, False),
    ("curated", "https://curated.reviewmind.app/test{i}", True, False),
]

MODES = ["auto", "manual"]
QUERY_TEXTS = [
    "Стоит ли брать {product}?",
    "{product} плюсы и минусы",
    "Сравнение {product} с аналогами",
    "{product} отзывы покупателей",
    "Качество звука {product}",
    "{product} цена качество",
    "Долго ли служит {product}?",
    "{product} vs конкуренты",
]


def rand_dt(days_ago_max: int, days_ago_min: int = 0) -> datetime:
    delta = timedelta(
        days=random.randint(days_ago_min, days_ago_max),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59),
    )
    return NOW - delta


def main() -> None:
    print(f"Connecting to: {DSN}")
    conn = psycopg2.connect(DSN)
    cur = conn.cursor()

    # ── 1. Users ──────────────────────────────────────────────────────────
    print("Inserting users...")
    user_ids = [100_000 + i for i in range(10)]
    for uid in user_ids:
        sub = "premium" if uid % 4 == 0 else "free"
        expires = (NOW + timedelta(days=30)).isoformat() if sub == "premium" else None
        created = rand_dt(DAYS, 25)
        cur.execute(
            """
            INSERT INTO users (user_id, subscription, sub_expires_at, created_at)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (user_id) DO NOTHING
            """,
            (uid, sub, expires, created),
        )

    # ── 2. Sources ────────────────────────────────────────────────────────
    print("Inserting sources...")
    source_ids = []
    for i in range(50):
        stype, url_tpl, is_curated, is_sponsored = random.choice(SOURCE_URLS)
        url = url_tpl.format(i=i) + f"_{random.randint(1000,9999)}"
        product = random.choice(PRODUCTS)
        parsed_at = rand_dt(DAYS, 1)
        is_sponsored = random.random() < 0.08
        cur.execute(
            """
            INSERT INTO sources (source_url, source_type, product_query, parsed_at,
                                 is_sponsored, is_curated, language)
            VALUES (%s, %s, %s, %s, %s, %s, 'ru')
            ON CONFLICT (source_url) DO NOTHING
            RETURNING id
            """,
            (url, stype, product, parsed_at, is_sponsored, is_curated),
        )
        row = cur.fetchone()
        if row:
            source_ids.append(row[0])

    # ── 3. Jobs ───────────────────────────────────────────────────────────
    print("Inserting jobs...")
    job_statuses = ["done"] * 25 + ["failed"] * 6 + ["pending"] * 5 + ["running"] * 4
    for i in range(40):
        uid = random.choice(user_ids)
        status = random.choice(job_statuses)
        job_type = random.choice(["auto_search", "manual_links"])
        product = random.choice(PRODUCTS)
        created_at = rand_dt(DAYS, 1)
        completed_at = (
            (created_at + timedelta(minutes=random.randint(1, 20)))
            if status in ("done", "failed")
            else None
        )
        cur.execute(
            """
            INSERT INTO jobs (id, user_id, job_type, status, product_query,
                              celery_task_id, created_at, completed_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
            """,
            (
                str(uuid.uuid4()),
                uid,
                job_type,
                status,
                product,
                str(uuid.uuid4()),
                created_at,
                completed_at,
            ),
        )

    # ── 4. Query logs ─────────────────────────────────────────────────────
    print("Inserting query_logs (300 entries)...")
    for _ in range(300):
        uid = random.choice(user_ids)
        product = random.choice(PRODUCTS)
        mode = random.choices(MODES, weights=[70, 30])[0]
        query_text = random.choice(QUERY_TEXTS).format(product=product)
        rating = random.choices([1, -1, None], weights=[55, 15, 30])[0]
        response_time_ms = random.randint(800, 8000)
        used_tavily = random.random() < 0.15
        created_at = rand_dt(DAYS)
        cur.execute(
            """
            INSERT INTO query_logs (user_id, session_id, mode, query_text,
                                    response_text, rating, response_time_ms,
                                    used_tavily, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                uid,
                str(uuid.uuid4())[:16],
                mode,
                query_text,
                f"Анализ {product}: [тестовый ответ ассистента]",
                rating,
                response_time_ms,
                used_tavily,
                created_at,
            ),
        )

    # ── 5. Subscriptions ──────────────────────────────────────────────────
    print("Inserting subscriptions...")
    premium_users = [uid for uid in user_ids if uid % 4 == 0]
    for uid in premium_users:
        activated_at = rand_dt(DAYS, 5)
        expires_at = activated_at + timedelta(days=30)
        cur.execute(
            """
            INSERT INTO subscriptions (user_id, telegram_payment_charge_id,
                                       amount_stars, activated_at, expires_at, status)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (telegram_payment_charge_id) DO NOTHING
            """,
            (
                uid,
                f"charge_{uid}_{random.randint(10000,99999)}",
                75,
                activated_at,
                expires_at,
                "active" if expires_at > NOW else "expired",
            ),
        )

    # ── 6. User limits ────────────────────────────────────────────────────
    print("Inserting user_limits...")
    today = date.today()
    for uid in user_ids:
        for days_ago in range(14):
            d = today - timedelta(days=days_ago)
            used = random.randint(0, 5)
            cur.execute(
                """
                INSERT INTO user_limits (user_id, date, requests_used)
                VALUES (%s, %s, %s)
                ON CONFLICT (user_id, date) DO UPDATE SET requests_used = EXCLUDED.requests_used
                """,
                (uid, d, used),
            )

    conn.commit()
    cur.close()
    conn.close()
    print("\n✅ Seed complete!")
    print("  • 10 users")
    print("  • 50 sources (youtube/reddit/web/tavily/curated)")
    print("  • 40 jobs (done/failed/pending/running)")
    print("  • 300 query_logs (30 days of activity)")
    print("  • subscription conversions")
    print("  • 14 days of user_limits")


if __name__ == "__main__":
    main()
