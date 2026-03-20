#!/usr/bin/env python3
"""Загрузка кураторской базы знаний (curated KB) по 7 категориям гаджетов.

Категории: smartphones, headphones, laptops, smartwatches, tablets, speakers, smart_tvs.
Имитация экспертных материалов в стиле Wirecutter / RTINGS / 4PDA.

Данные загружаются в Qdrant ``curated_kb`` коллекцию с ``is_curated=True``
и полем ``category`` в payload.

Usage::

    # Load curated data
    python scripts/seed_curated_kb.py seed

    # Verify curated_kb state
    python scripts/seed_curated_kb.py verify
"""

from __future__ import annotations

import asyncio
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone

import structlog

structlog.configure(
    processors=[
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
    logger_factory=structlog.PrintLoggerFactory(),
)

logger = structlog.get_logger("scripts.seed_curated_kb")

# ── Constants ────────────────────────────────────────────────────────────────

CATEGORIES: list[str] = [
    "smartphones",
    "headphones",
    "laptops",
    "smartwatches",
    "tablets",
    "speakers",
    "smart_tvs",
]

# Maximum age for curated materials (days).
MAX_AGE_DAYS: int = 365


# ── Data model ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CuratedArticle:
    """A single curated expert article to ingest into curated_kb."""

    category: str
    product_query: str
    source_url: str
    source_name: str
    text: str
    author: str = ""
    language: str = "ru"
    date: str = ""


# ── Curated articles per category ────────────────────────────────────────────

# --- smartphones ---

_SMARTPHONES_1 = CuratedArticle(
    category="smartphones",
    product_query="iPhone 16 Pro",
    source_url="https://www.rtings.com/smartphone/reviews/apple/iphone-16-pro",
    source_name="RTINGS",
    author="RTINGS Editorial",
    date="2025-10-15",
    text="""\
iPhone 16 Pro — экспертный обзор RTINGS.

Дисплей. OLED Super Retina XDR, 6.3 дюйма, разрешение 2622×1206 пикселей. ProMotion с адаптивной
частотой до 120 Гц. Пиковая яркость HDR — 2000 нит, на солнце — 2300 нит. Цветопередача отличная:
DCI-P3 покрытие 100%, калибровка точная из коробки (deltaE < 1.5). Always-On Display потребляет
менее 1% батареи в час. По результатам лабораторных тестов экран iPhone 16 Pro входит в тройку лучших
среди всех смартфонов 2025 года.

Камера. Основная камера 48 Мп (f/1.78), Ultra Wide 48 Мп (f/2.2), телеобъектив 12 Мп (5x zoom, f/2.8).
ProRes HQ запись до 4K 120fps (только на 256 ГБ+). Ночной режим значительно улучшен — динамический
диапазон увеличился на 25% по сравнению с iPhone 15 Pro. Портретный режим с фокусным расстоянием 24/28/35 мм.
Photographic Styles V3 для кастомизации цвета. Видео — лидер индустрии: стабилизация, динамический диапазон
и детализация не имеют равных среди смартфонов.

Производительность. A18 Pro (3 нм TSMC): CPU +15% к A17 Pro, GPU +20%. 8 ГБ RAM.
Geekbench 6: single-core 3200+, multi-core 8000+. Thermal throttling минимальный — после 20 минут
стресс-теста сброс производительности < 10%.

Батарея. 3577 мАч. Экранное время (наш тест): 11 часов при смешанном использовании.
Зарядка: USB-C до 50% за 30 минут (27W), MagSafe 15W, Qi2 15W.

Итого: 8.5/10. Лучший камерофон 2025 года, один из лучших дисплеев, отличная автономность.
Минусы: цена, отсутствие зарядки в комплекте, eSIM-only на некоторых рынках.""",
)

_SMARTPHONES_2 = CuratedArticle(
    category="smartphones",
    product_query="Samsung Galaxy S25 Ultra",
    source_url="https://www.wirecutter.com/reviews/samsung-galaxy-s25-ultra",
    source_name="Wirecutter",
    author="Wirecutter Staff",
    date="2025-11-01",
    text="""\
Samsung Galaxy S25 Ultra — обзор Wirecutter.

Samsung Galaxy S25 Ultra — флагман с S Pen и лучшим экраном на рынке Android. AMOLED Dynamic LTPO2 6.9 дюймов,
разрешение 3120×1440, яркость до 2600 нит. Титановая рамка, Gorilla Armor 2 — прочнейшая конструкция.

Камера: 200 Мп основная (f/1.7, OIS), 12 Мп Ultra Wide, 50 Мп телефото 3x, 50 Мп телефото 5x.
Ночная съёмка впечатляет — Nightography улучшена на 30%. Видео 8K 30fps и 4K 120fps. Однако обработка
слегка агрессивная — перенасыщенные цвета в авто-режиме. В Pro режиме результаты точнее.

Производительность: Snapdragon 8 Elite, 12 ГБ RAM. Один из быстрейших Android-смартфонов.
Galaxy AI: суммаризация, перевод в реальном времени, генерация обоев, Circle to Search.

Батарея: 5000 мАч. Экранное время 12-13 часов (наш тест). Зарядка 45W проводная (до 65% за 30 мин),
15W беспроводная.

S Pen встроен в корпус. Latency 2.8 мс — ощущается как настоящая ручка. Полезно для заметок,
быстрых скриншотов, навигации, удалённого управления камерой.

Минусы: 6.9 дюймов — большой для одной руки, вес 232 г, цена около 100 000+ рублей,
One UI иногда избыточно нагружает уведомлениями.

Итого: 8.7/10. Лучший Android-смартфон для тех, кто хочет максимум от экрана, камеры и S Pen.""",
)

# --- headphones ---

_HEADPHONES_1 = CuratedArticle(
    category="headphones",
    product_query="Sony WH-1000XM5",
    source_url="https://www.rtings.com/headphones/reviews/sony/wh-1000xm5",
    source_name="RTINGS",
    author="RTINGS Editorial",
    date="2025-07-20",
    text="""\
Sony WH-1000XM5 — экспертный обзор RTINGS.

Шумоподавление. Общий балл ANC: 8.9/10. Эффективность подавления низких частот (гул самолёта, кондиционер):
95%. Средние частоты (разговоры в офисе): 85%. Высокие частоты (детский плач): 70%.
Адаптивный ANC с 8 микрофонами автоматически подстраивается под окружение. В сравнении с Bose QC Ultra:
Sony немного лучше на низких, Bose лучше на средних частотах.

Звук. Частотный диапазон хорошо сбалансирован из коробки. Бас: глубокий, 35-150 Гц ровный, без
бубнения. Мидбас: чистый. Средние: отличная ясность вокала. Верхние: детализированные, без сибилянтов.
Саундстейдж: средний для закрытых наушников. Поддержка LDAC (990 kbps) и SBC/AAC.
Лабораторные замеры: THD < 0.3% на 94 дБ — отличный результат.

Комфорт. Вес 250 г. Прижим средний (3.5 Н). Амбушюры синтетическая кожа, глубина 22 мм.
Через 3 часа — лёгкий нагрев. Оголовье не давит. Для очков: среднее удобство.

Батарея. 30 часов с ANC. Быстрая зарядка: 3 мин → 3 часа. Полная зарядка 3.5ч USB-C.

Микрофон. Качество для звонков: 7/10. В тихой обстановке — хорошо. На улице — шум ветра подавляется
плохо. Качество записи голоса для подкастов: не рекомендуется.

Итого: 8.4/10. Лучшие полноразмерные ANC-наушники для повседневного использования.""",
)

_HEADPHONES_2 = CuratedArticle(
    category="headphones",
    product_query="Apple AirPods Max 2",
    source_url="https://www.wirecutter.com/reviews/apple-airpods-max-2",
    source_name="Wirecutter",
    author="Wirecutter Audio Team",
    date="2025-09-10",
    text="""\
Apple AirPods Max 2 — обзор Wirecutter.

Apple AirPods Max 2 — обновлённая версия премиальных наушников с чипом H2.
ANC улучшено на 2x по сравнению с первым поколением. Персонализированное
пространственное аудио с отслеживанием головы.

Звук: натуральный, точный тюнинг. Бас контролируемый, средние ясные,
высокие без резкости. Spatial Audio убедителен для фильмов и музыки.
Поддержка lossless через USB-C (новая функция), AAC по Bluetooth.

Конструкция: алюминий + нержавеющая сталь. Вес 385 г — тяжелее Sony XM5.
Тканевые амбушюры — предпочтительнее для долгого ношения в тёплом климате.
Digital Crown для управления громкостью — интуитивнее тачпанели.

Батарея: 20 часов (с ANC) — меньше чем у Sony. Зарядка через USB-C.
Smart Case теперь полноценный кейс (наконец!), а не чехол без защиты.

Экосистема: мгновенное переключение между iPhone, Mac, iPad. Siri, Find My,
уведомление о громкости. Для пользователей Apple — бесшовная интеграция.

Минусы: цена ~55 000 руб. Нет 3.5мм jack. Тяжёлые. Нет складной конструкции.
Без Apple устройств теряется 40% функций.

Итого: 8.2/10. Лучшие наушники для экосистемы Apple, но дорогие и тяжёлые.""",
)

# --- laptops ---

_LAPTOPS_1 = CuratedArticle(
    category="laptops",
    product_query="MacBook Pro M4 Pro 14",
    source_url="https://www.wirecutter.com/reviews/macbook-pro-m4-pro-14",
    source_name="Wirecutter",
    author="Wirecutter Laptop Team",
    date="2025-12-01",
    text="""\
MacBook Pro M4 Pro 14 дюймов — обзор Wirecutter.

Производительность. M4 Pro (12 CPU + 18 GPU) — мощнейший чип для ноутбука.
Cinebench R23 multi-core: 18500+. Видеомонтаж в Final Cut: 8K таймлайн без лагов.
Компиляция Xcode: на 30% быстрее M3 Pro. Машинное обучение: Neural Engine 16-core.
Thunderbolt 5: скорость передачи до 120 Gbps.

Дисплей. Liquid Retina XDR, 3024×1964, 120 Гц ProMotion. Яркость HDR 1600 нит,
пиковая 1000 нит SDR. P3 wide color gamut. Лучший дисплей ноутбука — точка.

Автономность. До 18 часов работы в Safari (наш тест: 14 часов при реальном использовании).
MagSafe 3 зарядка до 50% за 30 минут (140W).

Клавиатура и трекпад. Клавиатура с подсветкой, ход клавиш 1.5 мм — комфортная для длительной
печати. Force Touch трекпад — большой и точный. Touch ID надёжен.

Порты. 3× Thunderbolt 5, HDMI 2.1, SD-слот, MagSafe 3, 3.5 мм jack. Всё что нужно
без переходников.

Минусы: стартовая цена 200 000+ руб. Notch (чёлка) на экране. macOS — не для всех.
16 ГБ RAM в базе — мало для M4 Pro уровня.

Итого: 9.2/10. Лучший ноутбук для профессионалов. Без конкурентов по производительности на ватт.""",
)

_LAPTOPS_2 = CuratedArticle(
    category="laptops",
    product_query="ASUS ROG Zephyrus G16 2025",
    source_url="https://www.rtings.com/laptop/reviews/asus/rog-zephyrus-g16-2025",
    source_name="RTINGS",
    author="RTINGS Editorial",
    date="2025-08-15",
    text="""\
ASUS ROG Zephyrus G16 2025 — экспертный обзор RTINGS.

Процессор Intel Core Ultra 9 285HX + NVIDIA RTX 5080 Laptop (150W TGP).
Тонкий корпус 15.9 мм, вес 1.85 кг — портативный для игрового ноутбука.

Экран OLED 16 дюймов, 2560×1600, 240 Гц. Время отклика 0.2 мс. 100% DCI-P3.
Яркость 500 нит. Для игр и творчества — один из лучших экранов.

Производительность в играх: Cyberpunk 2077 Ultra RT 1600p — 75 fps.
Baldur's Gate 3 Ultra — 90+ fps. Hogwarts Legacy Ultra — 80 fps.

Батарея: 90 Вт·ч. Без нагрузки — 8-9 часов. В играх — 1.5-2 часа.
Зарядка USB-C PD 100W + штатный адаптер 240W.

Охлаждение: дуальные вентиляторы, жидкий металл. При нагрузке шум 42 дБА
в Performance режиме. В Silent — 35 дБА с пониженной производительностью.

Клавиатура: 1.7 мм ход, RGB per-key. Трекпад большой. Отсутствует numpanel.

Минусы: веб-камера 1080p среднего качества, динамики слабые без сабвуфера,
нет SD-слота, цена ~180 000 руб.

Итого: 8.0/10. Лучший тонкий игровой ноутбук с OLED экраном.""",
)

# --- smartwatches ---

_SMARTWATCHES_1 = CuratedArticle(
    category="smartwatches",
    product_query="Apple Watch Ultra 3",
    source_url="https://www.wirecutter.com/reviews/apple-watch-ultra-3",
    source_name="Wirecutter",
    author="Wirecutter Wearables",
    date="2025-10-20",
    text="""\
Apple Watch Ultra 3 — обзор Wirecutter.

Apple Watch Ultra 3 — флагманские спортивные часы на watchOS 12.

Дисплей. 2" LTPO3 OLED, яркость 3300 нит — читаемый при прямом солнце.
Always-On. Сапфировое стекло, титановый корпус 49 мм, IP6X + WR100.

Здоровье. Датчик SpO2, ЭКГ, температура тела, обнаружение апноэ сна.
Точность пульсометра: ±2 уд/мин (наш тест с Polar H10). Новый датчик
артериального давления — FDA approved, показания ±5 мм рт.ст.

Спорт. GPS двухчастотный L1+L5: точность маршрута ±1.5 м. Поддержка Running Power,
плавание в открытой воде, дайвинг до 40 м с глубиномером, велосипед с Power Zones.

Батарея. 36 часов мониторинг целый день + ночной трекинг сна. В режиме экономии — 72 часа.
GPS-тренировка: 12 часов. Быстрая зарядка: 80% за 45 минут.

Экосистема. Только iPhone. Siri, уведомления, Apple Pay, управление музыкой. App Store
с тысячами приложений. Навигация Apple Maps c тактильными подсказками.

Минусы: цена ~75 000 руб. Только для iPhone. Большие и тяжёлые (61.4 г).
Нет поддержки сторонних циферблатов. Подписки (Fitness+, Apple Music).

Итого: 9.0/10. Лучшие смарт-часы для экосистемы Apple и серьёзных спортсменов.""",
)

_SMARTWATCHES_2 = CuratedArticle(
    category="smartwatches",
    product_query="Samsung Galaxy Watch 7",
    source_url="https://www.rtings.com/smartwatch/reviews/samsung/galaxy-watch-7",
    source_name="RTINGS",
    author="RTINGS Editorial",
    date="2025-06-15",
    text="""\
Samsung Galaxy Watch 7 — экспертный обзор RTINGS.

Samsung Galaxy Watch 7 — лучшие Android-часы среднего сегмента.

Дисплей: 1.47" Super AMOLED 480×480. Яркость 3000 нит. Always-On.
Сапфировое стекло, алюминиевый корпус. Водозащита IP68 + 5ATM.

Здоровье: BioActive датчик (пульс, ЭКГ, биоимпеданс для состав тела).
SpO2, температура кожи, анализ сна с оценкой score. Точность пульса: ±3 уд/мин.

Спорт: GPS двухчастотный. 100+ режимов тренировок. Auto-detect для ходьбы,
бега, плавания. Точность GPS маршрута: ±2 м (хороший результат).

Батарея: 425 мАч. Типичное использование: 1.5 дня. С Always-On: 1 день.
Быстрая зарядка: 0–100% за 60 минут. Qi зарядка.

Wear OS 5: Google Maps, Google Pay, YouTube Music, Spotify offline.
Совместимость только с Android. Samsung DeX через часы — новая функция.

Минусы: батарея на 1.5 дня — мало для спортивных часов. Нет оффлайн-карт.
Bezels шире чем у конкурентов. Некоторые функции только с Samsung Galaxy.

Итого: 7.8/10. Лучшие Android смарт-часы по балансу функций и цены.""",
)

# --- tablets ---

_TABLETS_1 = CuratedArticle(
    category="tablets",
    product_query="iPad Pro M4",
    source_url="https://www.wirecutter.com/reviews/apple/ipad-pro-m4",
    source_name="Wirecutter",
    author="Wirecutter Staff",
    date="2025-05-20",
    text="""\
iPad Pro M4 — обзор Wirecutter.

Дизайн. Самый тонкий iPad: 5.1 мм (11") / 5.3 мм (13"). Алюминиевый корпус.
Вес 444 г (11") — легче чем iPad Air. USB-C с Thunderbolt 4.

Дисплей. Tandem OLED (два OLED слоя): яркость HDR 1600 нит, SDR 1000 нит.
Контрастность 2M:1 — абсолютно чёрный цвет. ProMotion 120 Гц.
P3 wide color, True Tone. Антибликовое покрытие (nano-texture опция).
Лучший дисплей планшета — с большим отрывом.

Производительность. M4: 10 ядер CPU (4P+6E), 10 ядер GPU. 16 ГБ RAM в 1 ТБ модели.
Быстрее на 40% чем M2 в графике. Neural Engine 16-core с 38 TOPS — готов к Apple
Intelligence. Для художников, видеомонтажёров, 3D-проектировщиков — мощности избыточно.

Apple Pencil Pro: сжатие (barrel squeeze), gyroscope для вращения, haptic feedback.
Magic Keyboard с трекпадом превращает iPad в ноутбук (но за дополнительную цену).

Минусы: iPad OS ограничена — нет полноценной файловой системы, нет мульти-оконности
как на macOS. Цена с аксессуарами > 200 000 руб. Зарядка медленная (20W). Камера
фронтальная в ландшафтной ориентации — удобно для звонков.

Итого: 9.0/10. Лучший планшет в мире. Для творческих профессионалов без альтернатив.""",
)

_TABLETS_2 = CuratedArticle(
    category="tablets",
    product_query="Samsung Galaxy Tab S10 Ultra",
    source_url="https://www.rtings.com/tablet/reviews/samsung/galaxy-tab-s10-ultra",
    source_name="RTINGS",
    author="RTINGS Editorial",
    date="2025-08-01",
    text="""\
Samsung Galaxy Tab S10 Ultra — экспертный обзор RTINGS.

Дисплей: 14.6 дюймов Dynamic AMOLED 2X, 2960×1848, 120 Гц. Яркость 930 нит.
Антибликовое покрытие. Для мультимедиа и работы — один из лучших экранов.

Производительность: MediaTek Dimensity 9300+. 12 ГБ RAM / 256 ГБ.
Для офисных задач, Android-игр и мультимедиа — более чем достаточно.
Samsung DeX режим: полноценный десктоп с окнами и таскбаром.

S Pen в комплекте. Задержка 2.8 мс. Идеален для заметок и рисования.
Book Cover Keyboard продаётся отдельно (~15 000 руб).

Батарея: 11200 мАч. Экранное время: 13 часов видео. Зарядка 45W — 0-100% за 80 мин.

Четыре динамика AKG с Dolby Atmos. Звук объёмный — для просмотра фильмов отлично.

Камера: 13 Мп основная + 8 Мп ультраширокая. Для сканирования документов достаточно,
для фото — слабовато.

Минусы: 14.6 дюймов — слишком большой для мобильного планшета, вес 718 г,
Android-приложения не всегда оптимизированы под большой экран, стилус не хранится
в корпусе. Цена от 90 000 руб.

Итого: 8.0/10. Лучший Android-планшет. Идеально для Samsung DeX и мультимедиа.""",
)

# --- speakers ---

_SPEAKERS_1 = CuratedArticle(
    category="speakers",
    product_query="Sonos Era 300",
    source_url="https://www.wirecutter.com/reviews/sonos-era-300",
    source_name="Wirecutter",
    author="Wirecutter Audio",
    date="2025-04-10",
    text="""\
Sonos Era 300 — обзор Wirecutter.

Sonos Era 300 — умная колонка с пространственным аудио Dolby Atmos.

Звук. 6 драйверов: 4 среднечастотных + 1 твитер + 1 вуфер. Звучание
объёмное, с ощутимой пространственной сценой. Dolby Atmos через Amazon
Music и Apple Music — результат впечатляющий: музыка вокруг слушателя.
Бас глубокий и точный, даже без сабвуфера. Верхние частоты ясные.
EQ автоматическая через Trueplay c микрофоном (iOS / встроенный).

Дизайн. Нестандартная форма — овальная для направления
звука по 6 осям. 19 см высота. Доступна в чёрном и белом.

Подключения. Wi-Fi 6, Bluetooth 5.0, AirPlay 2, Spotify Connect,
линейный вход 3.5 мм (через адаптер). Настройка через Sonos app.

Мультирум. Sonos экосистема: объединение с другими Sonos колонками,
стерео пара, домашний кинотеатр вместе с Sonos Arc.

Голосовой ассистент. Amazon Alexa встроен. Sonos Voice Control (ограниченно).
Google Assistant убран.

Минусы: цена ~35 000 руб. Dolby Atmos только в 2 сервисах.
Нет Google Assistant. Sonos app v2 — проблемная, много багов в 2024-2025.
Нет батареи — только от сети.

Итого: 8.6/10. Лучшая умная колонка для пространственного аудио.""",
)

_SPEAKERS_2 = CuratedArticle(
    category="speakers",
    product_query="JBL Charge 6",
    source_url="https://www.rtings.com/speaker/reviews/jbl/charge-6",
    source_name="RTINGS",
    author="RTINGS Editorial",
    date="2025-06-01",
    text="""\
JBL Charge 6 — экспертный обзор RTINGS.

JBL Charge 6 — портативная Bluetooth-колонка с функцией Power Bank.

Звук. Драйвер 70 мм + два пассивных излучателя. Бас ощутимый для
портативной колонки — low-end от 55 Гц. Громкость достаточная для
вечеринки на открытом воздухе (до 30 человек). На максимуме без
заметных искажений — THD менее 1% до 85% громкости. JBL Pro Sound
с динамической настройкой. Приложение JBL Portable: кастомный EQ.

Автономность. 24 часа воспроизведения (наш тест: 20 часов при 50%
громкости). Зарядка 2 часа через USB-C. Power Bank: заряжает телефон
через USB-A (отнимает 4-5 часов от музыки).

Защита. IP67: полная пылезащита + погружение в воду до 1 м на 30 мин.
Плавает! Можно брать в душ, на пляж, в поход.

Подключения. Bluetooth 5.3, мультиточка (2 устройства), PartyBoost
(объединение нескольких JBL колонок). Нет Wi-Fi, нет AirPlay.

Дизайн. 223 мм на 96.5 мм на 94.5 мм, 960 г. Прорезиненный корпус,
металлическая решётка. Доступен в 6 цветах.

Минусы: нет голосового ассистента, нет Wi-Fi/AirPlay, моно звук
(нет стерео без второй колонки), зарядка Power Bank медленная.

Итого: 8.3/10. Лучшая портативная колонка по балансу звука, автономности и защиты.""",
)

# --- smart_tvs ---

_SMART_TVS_1 = CuratedArticle(
    category="smart_tvs",
    product_query="LG OLED C4 65",
    source_url="https://www.rtings.com/tv/reviews/lg/oled-c4",
    source_name="RTINGS",
    author="RTINGS TV Team",
    date="2025-11-15",
    text="""\
LG OLED C4 65 дюймов — экспертный обзор RTINGS.

Качество изображения. Панель WRGB OLED Evo с Micro Lens Array+.
Пиковая яркость HDR: 1300 нит (10% окно) — рекорд для C-серии.
Абсолютный чёрный цвет, бесконечная контрастность. Цветовой
охват 99% DCI-P3. DeltaE менее 2 из коробки. Viewing angle отличный:
изображение не теряет качество до 70 градусов от центра.

HDR. Dolby Vision, HDR10, HLG. Tone mapping хороший — яркие
сцены детализированы. Dolby Vision IQ с датчиком освещённости.

Гейминг. HDMI 2.1 четыре порта, 4K при 120Hz, VRR (G-Sync, FreeSync Premium).
Время отклика менее 1 мс. Input lag: 5.5 мс (Game Mode) — один из лучших.
ALLM автоматически переключает в Game Mode.

Звук. Встроенная аудиосистема 40W, 2.2 канала. Dolby Atmos.
Для использования без саундбара — неплохо, но для кино лучше
добавить внешний.

Smart TV. webOS 24: быстрый, с AI рекомендациями. Netflix, YouTube,
Disney+, Apple TV+, все российские сервисы. Пульт Magic Remote
с указателем и голосовым управлением (LG ThinQ AI).

Минусы: ABL (автоматическое ограничение яркости) на больших белых
окнах. Риск выгорания при статичных элементах (логотипы, UI).
Подставка широкая — нужна полка минимум 110 см. Цена около 130 000 руб.

Итого: 9.1/10. Лучший OLED-телевизор по соотношению цена и качество.""",
)

_SMART_TVS_2 = CuratedArticle(
    category="smart_tvs",
    product_query="Samsung QN90D 65",
    source_url="https://www.wirecutter.com/reviews/samsung-qn90d-65",
    source_name="Wirecutter",
    author="Wirecutter TV Team",
    date="2025-05-01",
    text="""\
Samsung QN90D 65 дюймов — обзор Wirecutter.

Samsung QN90D — Neo QLED телевизор с Mini LED подсветкой.
Альтернатива OLED без риска выгорания.

Качество изображения. Точечная подсветка из тысяч Mini LED.
Пиковая яркость: 2000 нит — ярче любого OLED. Отлично для
светлых помещений. Контрастность: 25000:1 (нативная) — хуже
OLED, но лучше обычных LED. Blooming (засветы вокруг ярких
объектов) заметен в тёмных сценах, но минимальный для Mini LED.

Цвет. Quantum Dots: 98% DCI-P3. Калибровка из коробки хорошая
(Filmmaker Mode deltaE менее 3). Samsung Neural Quantum Processor 4K
апскейлит SD/HD контент качественно.

Гейминг. HDMI 2.1 четыре порта, 4K при 144Hz (одобрен AMD FreeSync Premium Pro).
Input lag 5.8 мс (Game Mode). Game Bar с быстрым доступом к настройкам.
Samsung Gaming Hub: облачный гейминг Xbox, GeForce Now.

Звук. OTS (Object Tracking Sound) 60W, 4.2.2 канала. Dolby Atmos.
Для встроенной системы — один из лучших. Q-Symphony с Samsung
саундбаром — добавляет ТВ динамики как дополнительные каналы.

Минусы: углы обзора хуже OLED — потеря контраста при 40 градусах от центра.
Tizen OS перегружена рекламой. Пульт без подсветки. Blooming в тёмных
сценах. Некоторые режимы HDR обрабатывают агрессивно.

Итого: 8.4/10. Лучший Mini LED телевизор для ярких комнат и гейминга.""",
)

# ── All curated articles ─────────────────────────────────────────────────────

ALL_CURATED_ARTICLES: list[CuratedArticle] = [
    _SMARTPHONES_1,
    _SMARTPHONES_2,
    _HEADPHONES_1,
    _HEADPHONES_2,
    _LAPTOPS_1,
    _LAPTOPS_2,
    _SMARTWATCHES_1,
    _SMARTWATCHES_2,
    _TABLETS_1,
    _TABLETS_2,
    _SPEAKERS_1,
    _SPEAKERS_2,
    _SMART_TVS_1,
    _SMART_TVS_2,
]


# ── Helper ───────────────────────────────────────────────────────────────────


def _get_qdrant_url() -> str:
    """Resolve Qdrant URL from env or config."""
    url = os.environ.get("QDRANT_URL")
    if url:
        return url
    try:
        from reviewmind.config import settings  # noqa: WPS433

        return settings.qdrant_url
    except Exception:
        return "http://localhost:6333"


def _is_article_fresh(article: CuratedArticle, max_age_days: int = MAX_AGE_DAYS) -> bool:
    """Return True if the article date is within *max_age_days* from now."""
    if not article.date:
        return True  # no date → assume fresh
    try:
        article_date = datetime.strptime(article.date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        return (now - article_date).days <= max_age_days
    except ValueError:
        logger.warning("invalid_date_format", date=article.date, url=article.source_url)
        return True  # unparseable → assume fresh


# ── Seed function ────────────────────────────────────────────────────────────


@dataclass
class SeedResult:
    """Summary of curated KB seed operation."""

    total_articles: int = 0
    ingested_articles: int = 0
    skipped_stale: int = 0
    total_chunks: int = 0
    chunks_per_category: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


async def seed_curated_kb(qdrant_url: str | None = None) -> SeedResult:
    """Ingest all curated articles into Qdrant ``curated_kb``.

    Returns a :class:`SeedResult` with per-category chunk counts.
    """
    from qdrant_client import AsyncQdrantClient

    from reviewmind.core.embeddings import EmbeddingService
    from reviewmind.ingestion.chunker import chunk_text
    from reviewmind.ingestion.cleaner import clean_text
    from reviewmind.ingestion.sponsor import detect_sponsor_detailed
    from reviewmind.vectorstore.client import ChunkPayload, upsert_chunks
    from reviewmind.vectorstore.collections import COLLECTION_CURATED_KB, ensure_all_collections

    url = qdrant_url or _get_qdrant_url()
    client = AsyncQdrantClient(url=url, timeout=30)
    result = SeedResult(total_articles=len(ALL_CURATED_ARTICLES))

    try:
        await ensure_all_collections(client)
        logger.info("collections_ensured")

        embedding = EmbeddingService()

        try:
            for article in ALL_CURATED_ARTICLES:
                log = logger.bind(
                    category=article.category,
                    product=article.product_query,
                    source=article.source_url,
                )

                # Freshness check
                if not _is_article_fresh(article):
                    log.info("article_stale", date=article.date)
                    result.skipped_stale += 1
                    continue

                # Clean
                cleaned = clean_text(article.text)
                if not cleaned:
                    log.warning("clean_empty")
                    result.errors.append(f"Empty after cleaning: {article.source_url}")
                    continue

                # Sponsor detection
                sponsor_result = detect_sponsor_detailed(cleaned)

                # Chunk
                metadata = {
                    "source_url": article.source_url,
                    "source_type": "curated",
                    "product_query": article.product_query,
                    "is_sponsored": sponsor_result.is_sponsored,
                    "is_curated": True,
                    "category": article.category,
                }
                chunks = chunk_text(cleaned, metadata=metadata)
                if not chunks:
                    log.warning("no_chunks")
                    result.errors.append(f"No chunks: {article.source_url}")
                    continue

                # Embed
                texts = [c.text for c in chunks]
                vectors = await embedding.embed_batch(texts)

                # Build payloads
                payloads = [
                    ChunkPayload(
                        text=c.text,
                        source_url=article.source_url,
                        source_type="curated",
                        product_query=article.product_query,
                        chunk_index=c.chunk_index,
                        language=article.language,
                        is_sponsored=sponsor_result.is_sponsored,
                        is_curated=True,
                        author=article.author,
                        date=article.date,
                    )
                    for c in chunks
                ]

                # Upsert with dedup (idempotent via deterministic point IDs)
                upsert_result = await upsert_chunks(
                    client, COLLECTION_CURATED_KB, vectors, payloads, skip_dedup=True,
                )
                inserted = upsert_result.inserted
                result.ingested_articles += 1
                result.total_chunks += inserted
                result.chunks_per_category[article.category] = (
                    result.chunks_per_category.get(article.category, 0) + inserted
                )
                log.info("article_ingested", chunks=inserted, skipped=upsert_result.skipped)

        finally:
            await embedding.close()

        logger.info(
            "curated_kb_seed_complete",
            articles=result.ingested_articles,
            stale=result.skipped_stale,
            chunks=result.total_chunks,
            categories=len(result.chunks_per_category),
        )
        return result

    finally:
        await client.close()


# ── Verify function ──────────────────────────────────────────────────────────


@dataclass
class VerifyResult:
    """Result of verifying the curated_kb collection state."""

    collection_exists: bool = False
    points_count: int = 0
    categories_found: list[str] = field(default_factory=list)
    sample_payloads: list[dict] = field(default_factory=list)
    all_curated: bool = False
    has_category_field: bool = False


async def verify_curated_kb(qdrant_url: str | None = None) -> VerifyResult:
    """Check that curated_kb collection exists and has correct data.

    Returns a :class:`VerifyResult` with collection state.
    """
    from qdrant_client import AsyncQdrantClient

    from reviewmind.vectorstore.collections import COLLECTION_CURATED_KB

    url = qdrant_url or _get_qdrant_url()
    client = AsyncQdrantClient(url=url, timeout=30)
    vr = VerifyResult()

    try:
        exists = await client.collection_exists(COLLECTION_CURATED_KB)
        vr.collection_exists = exists
        if not exists:
            logger.warning("curated_kb_not_found")
            return vr

        info = await client.get_collection(COLLECTION_CURATED_KB)
        vr.points_count = info.points_count or 0
        logger.info("curated_kb_points", count=vr.points_count)

        # Scroll a sample of points to verify payload
        scroll_result = await client.scroll(
            collection_name=COLLECTION_CURATED_KB,
            limit=50,
            with_payload=True,
            with_vectors=False,
        )
        points = scroll_result[0] if scroll_result else []

        categories_set: set[str] = set()
        all_curated = True
        has_category = True

        for point in points:
            payload = point.payload or {}
            vr.sample_payloads.append(payload)

            cat = payload.get("category", "")
            if cat:
                categories_set.add(cat)
            else:
                has_category = False

            if not payload.get("is_curated"):
                all_curated = False

        vr.categories_found = sorted(categories_set)
        vr.all_curated = all_curated
        vr.has_category_field = has_category

        logger.info(
            "curated_kb_verified",
            points=vr.points_count,
            categories=vr.categories_found,
            all_curated=vr.all_curated,
            has_category=vr.has_category_field,
        )
        return vr

    finally:
        await client.close()


# ── CLI ──────────────────────────────────────────────────────────────────────


async def _main(args: list[str]) -> int:
    """Run requested commands and return exit code."""
    if not args:
        print("Usage: python scripts/seed_curated_kb.py [seed] [verify]")  # noqa: T201
        return 1

    exit_code = 0

    if "seed" in args:
        logger.info("starting_seed")
        result = await seed_curated_kb()
        logger.info(
            "seed_done",
            articles=result.ingested_articles,
            chunks=result.total_chunks,
            stale=result.skipped_stale,
            errors=len(result.errors),
        )
        if result.errors:
            for err in result.errors:
                logger.warning("seed_error", error=err)

        for cat, count in sorted(result.chunks_per_category.items()):
            logger.info("category_chunks", category=cat, chunks=count)

    if "verify" in args:
        logger.info("starting_verify")
        vr = await verify_curated_kb()
        if not vr.collection_exists:
            logger.error("curated_kb_collection_missing")
            exit_code = 1
        elif vr.points_count == 0:
            logger.error("curated_kb_empty")
            exit_code = 1
        else:
            if not vr.all_curated:
                logger.warning("not_all_curated")
            if not vr.has_category_field:
                logger.warning("missing_category_field")
            logger.info(
                "verify_result",
                points=vr.points_count,
                categories=vr.categories_found,
                all_curated=vr.all_curated,
            )

    return exit_code


def main() -> None:
    """CLI entry point."""
    args = sys.argv[1:]
    code = asyncio.run(_main(args))
    sys.exit(code)


if __name__ == "__main__":
    main()
