#!/usr/bin/env python3
"""Загрузка тестовых seed-данных в Qdrant + верификация качества RAG.

Скрипт предоставляет реалистичные обзоры для 3 товаров:
- Sony WH-1000XM5 (наушники)
- iPhone 16 Pro (смартфон)
- Dyson V15 Detect (пылесос)

Каждый товар содержит 2-3 рецензии (имитация YouTube / Web / Reddit),
которые проходят через ingestion pipeline (clean → chunk → embed → upsert)
в Qdrant ``auto_crawled`` коллекцию.

Команда ``verify`` запускает 20 тестовых RAG-запросов и оценивает качество.

Usage::

    # Seed data
    python scripts/seed_test_data.py seed

    # Verify RAG quality
    python scripts/seed_test_data.py verify

    # Both
    python scripts/seed_test_data.py seed verify
"""

from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass, field

import structlog

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger("scripts.seed_test_data")

# ── Seed review data ─────────────────────────────────────────────────────────

PRODUCT_SONY_WH1000XM5 = "Sony WH-1000XM5"
PRODUCT_IPHONE_16_PRO = "iPhone 16 Pro"
PRODUCT_DYSON_V15 = "Dyson V15 Detect"


@dataclass(frozen=True)
class SeedReview:
    """A single seed review to ingest."""

    product_query: str
    source_url: str
    source_type: str
    text: str
    author: str = ""
    language: str = "ru"
    is_sponsored: bool = False


# Sony WH-1000XM5 reviews
_SONY_REVIEW_1 = SeedReview(
    product_query=PRODUCT_SONY_WH1000XM5,
    source_url="https://example.com/sony-wh1000xm5-review",
    source_type="web",
    author="AudioExpert",
    text="""\
Sony WH-1000XM5: полный обзор после 6 месяцев использования.

Шумоподавление. Sony WH-1000XM5 оснащены системой активного шумоподавления нового поколения с 8 микрофонами
и двумя процессорами. Шумоподавление здесь лучшее на рынке в 2025 году — убирает практически весь внешний шум
в офисе, метро и самолёте. По сравнению с XM4 алгоритм стал заметно лучше справляться с голосами.

Звук. Новые 30-мм драйверы обеспечивают чистый, детализированный звук. Бас глубокий но контролируемый,
средние частоты чёткие и прозрачные. Поддерживается LDAC кодек для Hi-Res audio по Bluetooth.
Через приложение Sony Headphones Connect можно настроить эквалайзер под свои предпочтения.

Комфорт. Наушники стали легче — 250 грамм против 254 у XM4. Амбушюры из синтетической кожи мягкие
и хорошо обхватывают ухо. Оголовье не давит. Через 3-4 часа непрерывного ношения уши немного потеют.

Автономность. 30 часов работы с ANC — впечатляющий результат. Быстрая зарядка: 3 минуты дают 3 часа
воспроизведения. Полная зарядка через USB-C занимает 3.5 часа.

Минусы. Конструкция не складная — неудобно для транспортировки. Цена высокая — около 30 000 рублей.
Мультипоинт подключение к двум устройствам работает, но иногда переключение глючит.
Нет защиты от воды (IP-рейтинг отсутствует).

Итог: Sony WH-1000XM5 — лучшие полноразмерные наушники с шумоподавлением на рынке.
Рекомендую для ежедневного использования, работы и путешествий.""",
)

_SONY_REVIEW_2 = SeedReview(
    product_query=PRODUCT_SONY_WH1000XM5,
    source_url="https://example.com/youtube-sony-xm5",
    source_type="youtube",
    author="ТехноОбзор",
    text="""\
Привет друзья, сегодня у нас Sony WH-1000XM5.

Начнём с главного — шумоподавление. Я тестировал эти наушники в метро, в офисе open space и в самолёте.
В метро — тишина, слышно только если кто-то кричит рядом. В офисе коллег не слышно вообще.
В самолёте гул двигателей убирается процентов на 90. Это реально лучший ANC что я пробовал.

Теперь звук. Ребята, звук тут просто огонь. Бас мощный но при этом не бубнящий. Вокал отлично
проработан. На классической музыке чувствуется сцена. Через LDAC звучит ещё лучше, рекомендую
включить в настройках.

Про удобство — я ношу их по 6-8 часов на работе. К концу дня есть лёгкий дискомфорт, но терпимо.
Для больших голов будет немного тесновато, примерьте перед покупкой.

Аккумулятор — заряжаю раз в неделю, с учётом что слушаю по 5-6 часов в день. 30 часов — не враньё.

Из минусов: не складываются, чехол большой. Сенсорное управление иногда срабатывает случайно —
задел ухо и музыка на паузе. Микрофон для звонков средний — собеседники иногда жалуются на эхо.

Стоят ли они своих денег? На мой взгляд — да, если шумоподавление для вас приоритет.""",
)

_SONY_REVIEW_3 = SeedReview(
    product_query=PRODUCT_SONY_WH1000XM5,
    source_url="https://example.com/reddit-sony-xm5",
    source_type="reddit",
    author="audiophile_user42",
    is_sponsored=True,  # This one is sponsored for testing
    text="""\
Пользуюсь Sony WH-1000XM5 уже 3 месяца, вот мои впечатления.

Купил по скидке за 25 000 рублей на распродаже. Промокод AUDIO2025 для скидки 15%.

ANC: 9/10. Лучше чем Bose QC Ultra на мой слух. Единственное что проникает — высокочастотные
звуки типа сирены или детского плача.

Звук: 8/10. Для Bluetooth наушников отлично, но для аудиофилов не заменит проводные.
Бас немного раздут в стоковом эквалайзере, рекомендую убрать на пару дБ через приложение.

Удобство: 7/10. Синтетическая кожа хуже чем ткань на Bose. Летом жарко. Для 4+ часов
ношения подряд — тяжеловато.

Связь: поддержка мультипоинта — подключаю одновременно к ноуту и телефону. Переключение
между устройствами за секунду.

Микрофон: 5/10. Для созвонов в Zoom приемлемо, но шум ветра на улице не подавляет.

Батарея: 9/10. Хватает на рабочую неделю. Быстрая зарядка спасает если забыл зарядить.

Главный конкурент — Apple AirPods Max (но они дороже почти вдвое) и Bose QC Ultra (чуть хуже ANC).
Ссылка на покупку в описании.""",
)

# iPhone 16 Pro reviews
_IPHONE_REVIEW_1 = SeedReview(
    product_query=PRODUCT_IPHONE_16_PRO,
    source_url="https://example.com/iphone-16-pro-review",
    source_type="web",
    author="MobileTech",
    language="ru",
    text="""\
iPhone 16 Pro: обзор после 3 месяцев ежедневного использования.

Дизайн. Титановая рамка стала легче и прочнее. Экран 6.3 дюйма с ProMotion (до 120 Гц)
и пиковой яркостью 2000 нит. Always-On Display потребляет минимум энергии. Dynamic Island
стал полезнее с iOS 18 — теперь через него можно управлять музыкой, навигацией, таймерами.

Камера. Главная камера 48 Мп с сенсором нового поколения. Фото в условиях плохого освещения
значительно улучшились. Новый Ultra Wide 48 Мп даёт отличную детализацию для макро.
Телеобъектив 5x оптический зум (12 Мп) — впечатляющий для смартфона. Видео 4K 120fps
в Dolby Vision — уникальная возможность на мобильных.

Производительность. Чип A18 Pro — бесспорный лидер. Многозадачность без задержек,
игры на максимальных настройках, рендеринг видео в LumaFusion за секунды. 8 ГБ RAM
достаточно для всех задач.

Автономность. Батарея на 3577 мАч. В моём использовании: 6-7 часов экранного времени
при активном использовании камеры, мессенджеров и социальных сетей. Быстрая зарядка
до 50% за 30 минут через USB-C, MagSafe до 15 Вт.

Программное обеспечение. iOS 18 с Apple Intelligence — умный Siri стал заметно полезнее.
Генеративные функции пока ограничены и работают только на английском. Обновления будут
минимум 5-6 лет.

Минусы. Цена 120 000 рублей — очень высокая. Зарядное устройство не в комплекте.
Кнопка Action не такая удобная как хотелось бы. Wi-Fi 7 только в моделях с eSIM.
Нет USB 3.0 скоростей по кабелю (только Thunderbolt при записи ProRes).""",
)

_IPHONE_REVIEW_2 = SeedReview(
    product_query=PRODUCT_IPHONE_16_PRO,
    source_url="https://example.com/youtube-iphone16pro",
    source_type="youtube",
    author="ГаджетМаньяк",
    text="""\
iPhone 16 Pro — стоит ли обновляться? Разбираемся.

Если у вас iPhone 14 Pro или старше — да, обновление стоит того. Камера стала реально лучше,
особенно ночная съёмка и видео.

Если у вас iPhone 15 Pro — изменения минимальны. Новый чип чуть мощнее, экран чуть больше
(6.3 vs 6.1), камера улучшена, но разницу заметите только в специфических сценариях.

Главные плюсы:
- Камера: ночной режим, 5x зум, 4K120fps — лучшее видео на смартфоне
- Производительность: A18 Pro мощнее всех конкурентов
- Автономность: на час дольше чем iPhone 15 Pro
- Экосистема Apple: AirDrop, Handoff, Apple Watch интеграция

Главные минусы:
- Цена: от 120 000 рублей, модель 1ТБ стоит 170 000
- Зарядка: 20W проводная, 15W MagSafe — медленнее конкурентов
- Customization: iOS всё ещё ограничена по сравнению с Android
- Apple Intelligence в России не работает

Конкуренты: Samsung Galaxy S25 Ultra (лучше экран, стилус, дешевле), Google Pixel 9 Pro
(лучший AI, чистый Android), Xiaomi 15 Pro (вдвое дешевле, отличная камера).

Мой вердикт: лучший iPhone, но не лучший смартфон по соотношению цена-качество.""",
)

# Dyson V15 Detect reviews
_DYSON_REVIEW_1 = SeedReview(
    product_query=PRODUCT_DYSON_V15,
    source_url="https://example.com/dyson-v15-review",
    source_type="web",
    author="CleanHome",
    text="""\
Dyson V15 Detect: обзор беспроводного пылесоса.

Мощность всасывания. Dyson V15 Detect обеспечивает до 240 AW мощности всасывания в режиме Boost.
Это один из самых мощных беспроводных пылесосов на рынке. В автоматическом режиме пылесос
сам регулирует мощность в зависимости от типа покрытия и количества пыли — очень удобно.

Лазерная подсветка. Главная фишка — зелёный лазер на насадке Slim Fluffy, который подсвечивает
мельчайшую пыль на твёрдых полах. Это реально полезная функция — видно пыль, которую
обычно не замечаешь. Особенно хорошо работает в тёмных углах и под мебелью.

Пьезодатчик. Встроенный акустический датчик определяет размер и количество частиц пыли
в реальном времени. На LCD-экране отображается статистика: сколько какого размера пыли
собрано. Практической пользы от этого немного, но наглядно видно что пылесос работает.

Автономность. На минимальной мощности — до 60 минут. В автоматическом режиме — около 40 минут.
В режиме Boost — 8-10 минут. Для обычной квартиры (80-100 кв.м) хватает на 1-2 полных уборки.
Заряжается 4.5 часа через док-станцию.

Минусы. Вес — 3 кг, что ощутимо при длительной уборке. Ёмкость пылесборника — 0.77л, для
большой квартиры маловато. Шум в режиме Boost — высокий. Цена — около 60 000 рублей.
Лазерная подсветка бесполезна на коврах.

Конкуренты: Samsung Bespoke Jet (сильная альтернатива), Xiaomi G20 (вдвое дешевле),
Dreame V16 (лучше автономность).

Итог: Dyson V15 Detect — отличный пылесос премиум класса. Если бюджет позволяет —
однозначно рекомендую.""",
)

_DYSON_REVIEW_2 = SeedReview(
    product_query=PRODUCT_DYSON_V15,
    source_url="https://example.com/reddit-dyson-v15",
    source_type="reddit",
    author="home_cleaner_99",
    text="""\
Купил Dyson V15 Detect полгода назад, делюсь опытом.

В целом доволен, но есть нюансы.

Плюсы:
- Лазер на полу — после него обычный пылесос вообще не хочется использовать, видно всю грязь
- Мощность в авто-режиме достаточна для 90% уборки
- Фильтрация HEPA — жена аллергик, говорит помогает
- Можно использовать как ручной пылесос для мебели и авто
- Экран с графиками пыли — забавная штука

Минусы:
- Тяжёлый, через 20 минут уборки рука устаёт
- На коврах средней ворсистости насадка Motorbar иногда застревает
- Аккумулятор деградирует — через 6 месяцев на авто-режиме стал работать 35 минут вместо 40
- Дорогие сменные фильтры и батарея
- Док-станция занимает место на стене

Сравнение с Xiaomi G11: Dyson мощнее и качественнее, но Xiaomi вчетверо дешевле
и для маленькой квартиры его достаточно.

Стоит ли покупать за 60 000? Если вам важна максимальная чистота и лазер — да.
Если ищете просто хороший беспроводной пылесос — есть варианты дешевле.""",
)

# ── All seed reviews ─────────────────────────────────────────────────────────

ALL_SEED_REVIEWS: list[SeedReview] = [
    _SONY_REVIEW_1,
    _SONY_REVIEW_2,
    _SONY_REVIEW_3,
    _IPHONE_REVIEW_1,
    _IPHONE_REVIEW_2,
    _DYSON_REVIEW_1,
    _DYSON_REVIEW_2,
]

PRODUCT_QUERIES: list[str] = [
    PRODUCT_SONY_WH1000XM5,
    PRODUCT_IPHONE_16_PRO,
    PRODUCT_DYSON_V15,
]

# ── 20 test queries ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TestQuery:
    """A test query with expected attributes for verification."""

    query: str
    product_query: str | None = None
    expected_language: str = "ru"
    expects_sources: bool = True
    min_chunks: int = 1


TEST_QUERIES: list[TestQuery] = [
    # Sony WH-1000XM5 queries
    TestQuery(
        query="Sony WH-1000XM5 стоит ли покупать?",
        product_query=PRODUCT_SONY_WH1000XM5,
    ),
    TestQuery(
        query="Какое шумоподавление у Sony WH-1000XM5?",
        product_query=PRODUCT_SONY_WH1000XM5,
    ),
    TestQuery(
        query="Минусы Sony WH-1000XM5",
        product_query=PRODUCT_SONY_WH1000XM5,
    ),
    TestQuery(
        query="Как долго работает батарея Sony XM5?",
        product_query=PRODUCT_SONY_WH1000XM5,
    ),
    TestQuery(
        query="Удобно ли носить Sony WH-1000XM5 целый день?",
        product_query=PRODUCT_SONY_WH1000XM5,
    ),
    TestQuery(
        query="Sony XM5 vs Bose QC Ultra что лучше?",
        product_query=PRODUCT_SONY_WH1000XM5,
    ),
    TestQuery(
        query="Микрофон Sony WH-1000XM5 для звонков",
        product_query=PRODUCT_SONY_WH1000XM5,
    ),
    # iPhone 16 Pro queries
    TestQuery(
        query="iPhone 16 Pro стоит ли обновляться с 14 Pro?",
        product_query=PRODUCT_IPHONE_16_PRO,
    ),
    TestQuery(
        query="Камера iPhone 16 Pro — как снимает ночью?",
        product_query=PRODUCT_IPHONE_16_PRO,
    ),
    TestQuery(
        query="Время работы батареи iPhone 16 Pro",
        product_query=PRODUCT_IPHONE_16_PRO,
    ),
    TestQuery(
        query="Главные минусы iPhone 16 Pro",
        product_query=PRODUCT_IPHONE_16_PRO,
    ),
    TestQuery(
        query="iPhone 16 Pro vs Samsung Galaxy S25 Ultra",
        product_query=PRODUCT_IPHONE_16_PRO,
    ),
    TestQuery(
        query="Сколько стоит iPhone 16 Pro в 2026 году?",
        product_query=PRODUCT_IPHONE_16_PRO,
    ),
    # Dyson V15 Detect queries
    TestQuery(
        query="Dyson V15 Detect отзывы владельцев",
        product_query=PRODUCT_DYSON_V15,
    ),
    TestQuery(
        query="Нужен ли лазер в пылесосе Dyson?",
        product_query=PRODUCT_DYSON_V15,
    ),
    TestQuery(
        query="Сколько работает Dyson V15 от батареи?",
        product_query=PRODUCT_DYSON_V15,
    ),
    TestQuery(
        query="Dyson V15 или Xiaomi G11 — что выбрать?",
        product_query=PRODUCT_DYSON_V15,
    ),
    TestQuery(
        query="Минусы Dyson V15 Detect",
        product_query=PRODUCT_DYSON_V15,
    ),
    # Cross-product / edge-case queries
    TestQuery(
        query="Best noise cancelling headphones 2025",
        product_query=PRODUCT_SONY_WH1000XM5,
        expected_language="en",
    ),
    TestQuery(
        query="Какой гаджет лучше для работы из дома?",
        product_query=None,
        expects_sources=False,
        min_chunks=0,
    ),
]

# ── Verification result ──────────────────────────────────────────────────────


@dataclass
class VerificationResult:
    """Result of verifying a single test query through the RAG pipeline."""

    query: str
    answer: str = ""
    sources_count: int = 0
    chunks_count: int = 0
    confidence_met: bool = False
    has_answer: bool = False
    response_time_ms: float = 0.0
    error: str | None = None


@dataclass
class VerificationReport:
    """Aggregate report for all 20 test queries."""

    total: int = 0
    passed: int = 0
    failed: int = 0
    avg_response_time_ms: float = 0.0
    results: list[VerificationResult] = field(default_factory=list)

    @property
    def pass_rate(self) -> float:
        return (self.passed / self.total * 100) if self.total > 0 else 0.0


# ── Seed function ────────────────────────────────────────────────────────────


async def seed_data(qdrant_url: str | None = None) -> dict[str, int]:
    """Ingest all seed reviews into Qdrant ``auto_crawled``.

    Returns a dict mapping product_query → chunks_count.
    """
    from qdrant_client import AsyncQdrantClient

    from reviewmind.core.embeddings import EmbeddingService
    from reviewmind.ingestion.chunker import chunk_text
    from reviewmind.ingestion.cleaner import clean_text
    from reviewmind.ingestion.sponsor import detect_sponsor_detailed
    from reviewmind.vectorstore.client import ChunkPayload, upsert_chunks
    from reviewmind.vectorstore.collections import COLLECTION_AUTO_CRAWLED, ensure_all_collections

    url = qdrant_url or _get_qdrant_url()
    client = AsyncQdrantClient(url=url, timeout=30)

    try:
        # Ensure collections exist
        await ensure_all_collections(client)
        logger.info("collections_ensured")

        embedding = EmbeddingService()
        product_chunks: dict[str, int] = {}

        try:
            for review in ALL_SEED_REVIEWS:
                log = logger.bind(
                    product=review.product_query,
                    source=review.source_url,
                )

                # Clean
                cleaned = clean_text(review.text)
                if not cleaned:
                    log.warning("clean_empty")
                    continue

                # Sponsor detection
                sponsor_result = detect_sponsor_detailed(cleaned)

                # Chunk
                metadata = {
                    "source_url": review.source_url,
                    "source_type": review.source_type,
                    "product_query": review.product_query,
                    "is_sponsored": sponsor_result.is_sponsored or review.is_sponsored,
                    "is_curated": False,
                }
                chunks = chunk_text(cleaned, metadata=metadata)
                if not chunks:
                    log.warning("no_chunks")
                    continue

                # Embed
                texts = [c.text for c in chunks]
                vectors = await embedding.embed_batch(texts)

                # Build payloads
                payloads = [
                    ChunkPayload(
                        text=c.text,
                        source_url=review.source_url,
                        source_type=review.source_type,
                        product_query=review.product_query,
                        chunk_index=c.chunk_index,
                        language=review.language,
                        is_sponsored=sponsor_result.is_sponsored or review.is_sponsored,
                        is_curated=False,
                        author=review.author,
                    )
                    for c in chunks
                ]

                # Upsert (skip_dedup=True for seed data — idempotent via point ID)
                result = await upsert_chunks(
                    client, COLLECTION_AUTO_CRAWLED, vectors, payloads, skip_dedup=True,
                )
                inserted = result.inserted
                product_chunks[review.product_query] = (
                    product_chunks.get(review.product_query, 0) + inserted
                )
                log.info("review_ingested", chunks=inserted, skipped=result.skipped)

        finally:
            await embedding.close()

        logger.info(
            "seed_complete",
            products=len(product_chunks),
            total_chunks=sum(product_chunks.values()),
        )
        return product_chunks

    finally:
        await client.close()


# ── Verify function ──────────────────────────────────────────────────────────


async def verify_rag(qdrant_url: str | None = None) -> VerificationReport:
    """Run 20 test queries through the RAG pipeline and evaluate results.

    Returns a :class:`VerificationReport` with per-query results and stats.
    """
    from qdrant_client import AsyncQdrantClient

    from reviewmind.core.rag import RAGPipeline

    url = qdrant_url or _get_qdrant_url()
    client = AsyncQdrantClient(url=url, timeout=30)

    report = VerificationReport(total=len(TEST_QUERIES))

    try:
        pipeline = RAGPipeline(qdrant_client=client)

        try:
            for i, tq in enumerate(TEST_QUERIES, 1):
                log = logger.bind(query_num=i, query=tq.query[:60])

                vr = VerificationResult(query=tq.query)
                start = time.monotonic()

                try:
                    rag_response = await pipeline.query(
                        tq.query,
                        product_query=tq.product_query,
                    )
                    elapsed_ms = (time.monotonic() - start) * 1000
                    vr.answer = rag_response.answer
                    vr.sources_count = len(rag_response.sources)
                    vr.chunks_count = rag_response.chunks_count
                    vr.confidence_met = rag_response.confidence_met
                    vr.has_answer = bool(rag_response.answer and len(rag_response.answer) > 20)
                    vr.response_time_ms = elapsed_ms

                    # Pass/fail logic
                    passed = vr.has_answer
                    if tq.expects_sources:
                        passed = passed and vr.sources_count > 0
                    if tq.min_chunks > 0:
                        passed = passed and vr.chunks_count >= tq.min_chunks

                    if passed:
                        report.passed += 1
                        log.info("query_passed", time_ms=f"{elapsed_ms:.0f}")
                    else:
                        report.failed += 1
                        log.warning(
                            "query_failed",
                            has_answer=vr.has_answer,
                            sources=vr.sources_count,
                            chunks=vr.chunks_count,
                        )

                except Exception as exc:  # noqa: BLE001
                    elapsed_ms = (time.monotonic() - start) * 1000
                    vr.error = str(exc)
                    vr.response_time_ms = elapsed_ms
                    report.failed += 1
                    log.error("query_error", error=str(exc))

                report.results.append(vr)

        finally:
            await pipeline.close()

    finally:
        await client.close()

    # Compute average response time
    times = [r.response_time_ms for r in report.results if r.response_time_ms > 0]
    report.avg_response_time_ms = sum(times) / len(times) if times else 0.0

    return report


def _get_qdrant_url() -> str:
    """Return Qdrant URL from env or config."""
    import os

    url = os.environ.get("QDRANT_URL")
    if url:
        return url
    try:
        from reviewmind.config import settings
        return settings.qdrant_url
    except Exception:
        return "http://localhost:6333"


def _print_report(report: VerificationReport) -> None:
    """Print verification report to stdout."""
    print("\n" + "=" * 70)
    print("RAG VERIFICATION REPORT")
    print("=" * 70)
    print(f"Total queries:        {report.total}")
    print(f"Passed:               {report.passed}")
    print(f"Failed:               {report.failed}")
    print(f"Pass rate:            {report.pass_rate:.1f}%")
    print(f"Avg response time:    {report.avg_response_time_ms:.0f} ms")
    print("-" * 70)

    for i, r in enumerate(report.results, 1):
        status = "✅" if r.has_answer and not r.error else "❌"
        print(f"\n{status} Query {i}: {r.query[:60]}")
        if r.error:
            print(f"   Error: {r.error}")
        else:
            print(f"   Answer length: {len(r.answer)} chars")
            print(f"   Sources: {r.sources_count}, Chunks: {r.chunks_count}")
            print(f"   Confidence met: {r.confidence_met}")
            print(f"   Time: {r.response_time_ms:.0f} ms")
    print("\n" + "=" * 70)


# ── CLI entrypoint ───────────────────────────────────────────────────────────


async def main(args: list[str]) -> None:
    """Run seed and/or verify based on CLI arguments."""
    commands = {a.lower() for a in args} if args else {"seed", "verify"}

    if not commands.intersection({"seed", "verify"}):
        print(__doc__)
        sys.exit(1)

    if "seed" in commands:
        logger.info("seeding_start")
        product_chunks = await seed_data()
        for product, count in product_chunks.items():
            logger.info("product_seeded", product=product, chunks=count)

    if "verify" in commands:
        logger.info("verification_start")
        report = await verify_rag()
        _print_report(report)
        if report.pass_rate < 80:
            logger.warning("quality_below_threshold", pass_rate=report.pass_rate)
            sys.exit(1)
        else:
            logger.info("quality_ok", pass_rate=report.pass_rate)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
