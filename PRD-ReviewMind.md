# PRD: ReviewMind — AI-агрегатор обзоров для принятия решений о покупке

**Версия:** 1.0  
**Дата:** 2026-03-01  
**Статус:** Ready for Development  
**Платформа:** Telegram Bot (MVP) + Web UI (Phase 2)  

---

## 1. Обзор приложения и цели

### 1.1 Проблема

Современный потребитель тратит 2–4 часа на изучение отзывов перед покупкой товара: YouTube-обзоры, Reddit-треды, сравнительные статьи. При этом:

- Большинство видеообзоров спонсированы и содержат предвзятую информацию.
- Каждый новый источник добавляет лишь 2–5% нового знания при 15–30 минутах просмотра.
- Критически важные «нестандартные» мнения (реальные дефекты, нишевые проблемы) теряются в потоке позитивного контента.
- Пользователь вынужден самостоятельно агрегировать и взвешивать противоречивые мнения.

### 1.2 Решение

**ReviewMind** — Telegram-бот (с последующим выходом в веб-интерфейс), который автоматически собирает, анализирует и синтезирует мнения из открытых источников (YouTube, Reddit, веб-статьи) и выдаёт пользователю структурированный беспристрастный анализ за секунды.

### 1.3 Критерии успеха (MVP KPIs)

| Метрика | Цель |
|---|---|
| Время до первого ответа (кэш) | < 3 секунды |
| Время уведомления (первичный сбор) | ≤ 5 минут |
| Доля релевантных ответов без внешних источников | ≥ 80% (после прогрева базы) |
| Положительных оценок (👍) | ≥ 70% |
| Возврат пользователя в течение 7 дней | ≥ 40% |
| % сессий с уточняющими вопросами | ≥ 30% |

---

## 2. Целевая аудитория

### 2.1 Основной пользователь

**Сегмент:** B2C — конечные потребители, планирующие покупку конкретного товара (электроника, гаджеты, смарт-девайсы, аксессуары).

**Портрет:** Технически грамотный пользователь Telegram, 18–45 лет, RU/CIS рынок (приоритет), EN рынок (вторично). Принимает решения о покупке на основе обзоров, экономит время.

### 2.2 Администраторы

Отдельный список `ADMIN_USER_IDS` в конфигурации. Доступ: безлимитные запросы, мониторинг алертов в admin Telegram-чат, управление кураторской базой.

### 2.3 Языковая поддержка

Бот отвечает **на языке запроса пользователя** (автодетекция через `langdetect`). Приоритетные языки MVP: **русский и английский**.

---

## 3. Архитектурные диаграммы

### 3.1 Общая схема системы

```mermaid
graph TB
    subgraph Clients["Client Layer"]
        TG[Telegram Bot<br/>aiogram 3.x<br/>long polling → webhook]
        WEB[Web UI<br/>React - Phase 2]
    end

    subgraph API["API Layer (FastAPI)"]
        GW[API Gateway<br/>REST endpoints]
        RL[Rate Limiter<br/>slowapi]
        MODE{User Mode<br/>Auto / Manual}
    end

    subgraph Core["Core Logic"]
        QP[Query Pipeline<br/>RAG Engine]
        IP[Ingestion Pipeline<br/>Parser + Embedder]
    end

    subgraph Queue["Task Queue"]
        REDIS_Q[Redis Broker]
        CW[Celery Workers<br/>Parallel scrapers]
        CB[Celery Beat<br/>Scheduled updates]
    end

    subgraph Storage["Storage Layer"]
        QD_AUTO[(Qdrant<br/>auto_crawled collection)]
        QD_CUR[(Qdrant<br/>curated_kb collection)]
        PG[(PostgreSQL<br/>Metadata + Logs)]
        REDIS_C[(Redis<br/>Session cache)]
    end

    subgraph Scrapers["Scraper Tools - shared by both modes"]
        YT[youtube-transcript-api<br/>YouTube Data API v3]
        RD[PRAW<br/>Reddit API]
        TF[trafilatura<br/>General web pages]
        TV[Tavily API<br/>Zero-shot web search fallback]
    end

    subgraph External["LLM / Embedding"]
        OAI[OpenAI API<br/>gpt-4o-mini + text-embedding-3-small<br/>GitHub Student Pack → production key]
    end

    TG -->|long polling / webhook| GW
    WEB -->|REST| GW
    GW --> RL --> MODE
    MODE -->|cache hit| QP
    MODE -->|auto: cache miss| REDIS_Q
    MODE -->|manual: user links| IP
    REDIS_Q --> CW --> IP
    CB --> CW
    IP --> YT
    IP --> RD
    IP --> TF
    IP -->|no data fallback| TV
    IP -->|embed| OAI
    IP -->|upsert| QD_AUTO
    IP -->|save source metadata| PG
    QP -->|search both collections| QD_AUTO
    QP -->|priority search| QD_CUR
    QP -->|generate| OAI
    QP -->|session ctx| REDIS_C
    CW -->|push result| TG
```

---

### 3.2 Request Flow (Sequence Diagram)

```mermaid
sequenceDiagram
    actor User
    participant Bot as Telegram Bot
    participant API as FastAPI
    participant CurKB as Qdrant (curated_kb)
    participant AutoKB as Qdrant (auto_crawled)
    participant Queue as Celery
    participant Scrapers as Scrapers (YT/Reddit/Web)
    participant LLM as OpenAI gpt-4o-mini

    User->>Bot: "Is the Dyson V15 worth it?"
    Bot->>User: [🔍 Auto-search] [🔗 Send my links]

    alt Manual Mode — user pastes links
        User->>Bot: youtube.com/... rtings.com/...
        Bot->>API: POST /ingest {user_id, urls[]}
        API->>Scrapers: Parse each URL (trafilatura / YT transcript)
        Scrapers-->>API: Clean text chunks
        API->>AutoKB: Embed + upsert chunks
        API->>LLM: RAG prompt + retrieved chunks
        LLM-->>Bot: Answer + source list
        Bot-->>User: ✅ Answer from your sources

    else Auto Mode — bot finds sources
        Bot->>API: POST /query {user_id, query, mode=auto}
        API->>CurKB: Search curated_kb for product
        CurKB-->>API: Curated chunks (priority, score boost)
        API->>AutoKB: Search auto_crawled for product

        alt Enough data (≥3 confident chunks total)
            AutoKB-->>API: Additional chunks
            API->>LLM: Prompt + curated + auto chunks
            LLM-->>API: Structured answer
            API-->>Bot: Answer + 📚 curated badge if used
            Bot-->>User: ✅ Instant answer (<3s)

        else Insufficient data — trigger background fetch
            AutoKB-->>API: Too few results
            API-->>Bot: "⏳ Collecting data (~3 min)..."
            Bot-->>User: Notified, please wait
            API->>Queue: Enqueue ingestion job
            Queue->>Scrapers: YouTube search + Reddit search
            Scrapers-->>Queue: Raw text
            Queue->>AutoKB: Chunk + embed + upsert
            Queue->>API: Job done callback
            API->>LLM: RAG prompt with fresh chunks
            API->>Bot: Push final answer
            Bot-->>User: ✅ Answer + [👍][👎][📎 Sources]
        end
    end
```

---

### 3.3 Data Ingestion Pipeline

```mermaid
flowchart LR
    subgraph Input["Data Sources"]
        YT_S[YouTube Search<br/>Data API v3]
        YT_T[YouTube Transcripts<br/>youtube-transcript-api]
        RD_S[Reddit Posts + Comments<br/>PRAW]
        WEB[General Web Pages<br/>trafilatura]
        CUR[Curated Materials<br/>Manual team curation]
    end

    subgraph Process["Processing"]
        CLEAN[Clean & Normalize<br/>Remove ads, timestamps<br/>HTML tags]
        SPONSOR[Sponsor Detection<br/>Regex heuristics<br/>is_sponsored flag]
        CHUNK[Text Chunking<br/>400-600 tokens<br/>50 token overlap]
        META[Metadata Tagging<br/>source_type, url<br/>date, author, is_curated]
    end

    subgraph Store["Storage"]
        EMBED[Embed<br/>text-embedding-3-small<br/>1536 dims]
        QD_A[(Qdrant<br/>auto_crawled<br/>collection)]
        QD_C[(Qdrant<br/>curated_kb<br/>collection<br/>no TTL, score boost)]
        PG_S[(PostgreSQL<br/>sources table)]
    end

    YT_S --> CLEAN
    YT_T --> CLEAN
    RD_S --> CLEAN
    WEB --> CLEAN
    CUR --> CHUNK
    CLEAN --> SPONSOR
    SPONSOR --> CHUNK
    CHUNK --> META
    META --> EMBED
    EMBED -->|is_curated=false| QD_A
    EMBED -->|is_curated=true| QD_C
    META --> PG_S
```

---

### 3.4 RAG Query Pipeline

```mermaid
flowchart TD
    Q[User Question] --> EMBED_Q[Embed Query<br/>text-embedding-3-small]
    EMBED_Q --> CUR_SEARCH[Search curated_kb<br/>Priority collection]
    CUR_SEARCH --> AUTO_SEARCH[Search auto_crawled<br/>Secondary collection]
    AUTO_SEARCH --> MERGE[Merge + Rerank<br/>Curated chunks get score boost<br/>Down-weight is_sponsored]
    MERGE --> CHECK{Enough data?<br/>≥3 chunks<br/>confidence > 0.75}
    CHECK -->|Yes| CONTEXT[Build Context<br/>up to 8 chunks<br/>+ metadata + history]
    CHECK -->|No, auto mode| TRIGGER[Trigger background<br/>Celery ingestion job]
    CHECK -->|No, manual mode| ASK[Ask user<br/>for more links]
    TRIGGER --> NOTIFY[Notify user<br/>'Collecting data...']
    CONTEXT --> PROMPT[System Prompt<br/>+ Context + Question<br/>+ Chat History]
    PROMPT --> LLM[gpt-4o-mini<br/>temp=0.3<br/>max_tokens=1000]
    LLM --> ANSWER[Structured Answer<br/>Pros / Cons / Controversies<br/>📚 badge if curated used<br/>Source count]
    ANSWER --> FEEDBACK[User Feedback<br/>👍 👎 📎 Sources]
```

---

## 4. Основные функции и функциональность

### 4.1 Feature F1 — Авто-режим (Auto Search)

**Описание:** Пользователь называет товар — бот самостоятельно находит источники, анализирует и отвечает.

**User Flow:**
1. `/start` → кнопка `[🔍 Авто-поиск]`
2. Пользователь пишет запрос (например: «Sony WH-1000XM5 стоит ли покупать?»)
3. Система проверяет Qdrant (`curated_kb` → `auto_crawled`)
4. **Если ≥ 3 чанка с confidence > 0.75:** немедленный RAG-ответ < 3 сек
5. **Если < 3 чанков:** Tavily API возвращает быстрый ответ (~2 сек) + параллельно Celery запускает полный сбор (YouTube + Reddit)
6. По завершении Celery — пуш-уведомление с обновлённым анализом

**Acceptance Criteria:**
- Запрос по товару из прогретой базы возвращает ответ менее чем за 3 секунды
- Запрос по новому товару: пользователь получает уведомление «⏳ Ищу данные...» не позже чем через 1 секунду, финальный ответ — в течение 5 минут
- Ответ содержит секции: `[✅ Плюсы] [❌ Минусы] [⚖️ Спорные моменты] [🏆 Вывод]`
- Ответ содержит кнопки `[👍] [👎] [📎 Источники]`
- Если использованы кураторские источники — виден бейдж `📚 Проверенные источники`

---

### 4.2 Feature F2 — Режим «Свои ссылки» (Manual Links)

**Описание:** Пользователь предоставляет конкретные ссылки — бот парсит именно их и отвечает на основе этого контента.

**User Flow:**
1. `/start` → кнопка `[🔗 Свои ссылки]`
2. Пользователь отправляет URL (по одному или списком)
3. Бот определяет тип: YouTube → `youtube-transcript-api`, Reddit → PRAW, остальное → `trafilatura`
4. Синхронный парсинг (MVP) → чанкинг → эмбеддинг → Qdrant `auto_crawled` (с `session_id`)
5. Немедленный RAG-ответ по загруженным чанкам

**Acceptance Criteria:**
- Корректно обрабатываются URL всех трёх типов: YouTube, Reddit, произвольные веб-страницы
- При недоступности URL или paywall — уведомление `«Не удалось загрузить: {url}. Попробуйте другую ссылку.»`, остальные ссылки обрабатываются
- Повторный вопрос по тем же ссылкам в рамках сессии отвечает мгновенно (из кэша Qdrant)
- Минимум 1 успешно спарсенная ссылка достаточна для генерации ответа

---

### 4.3 Feature F3 — RAG Pipeline (Retrieval-Augmented Generation)

**Описание:** Ядро системы — гибридный поиск + LLM-генерация.

**Порядок обработки запроса:**
1. Embed query → `text-embedding-3-small`
2. Параллельный поиск: `curated_kb` (Top-5) + `auto_crawled` (Top-5)
3. Reranking: кураторские чанки ×1.3, спонсорские ×0.7
4. Pre-filter по `product_query` payload
5. Если итого < 3 чанков с confidence > 0.75 → Tavily fallback
6. LLM генерация (gpt-4o-mini, temperature=0.3, max_tokens=1000)

**Acceptance Criteria:**
- Top-5 чанков релевантны запросу (верифицируется ручной проверкой 20 запросов)
- Ответ не содержит информации, не подтверждённой контекстом (RAG-промпт с явным ограничением)
- Спонсорские источники понижаются в ранжировании и помечаются в ответе
- Дубли чанков при upsert исключаются (cosine similarity > 0.95 → skip)

---

### 4.4 Feature F4 — Монетизация (Freemium + Telegram Stars)

**Описание:** Бесплатный уровень с лимитом + платная подписка через нативную систему Telegram Stars.

**Тарифы:**

| Тариф | Лимит | Стоимость |
|---|---|---|
| Free | 3 запроса в сутки | Бесплатно |
| Premium | Безлимит | ≈ 299 ₽/мес (в Telegram Stars) |
| Admin | Безлимит | Whitelist в конфиге |

**Логика лимитов:**
- Счётчик запросов хранится в PostgreSQL (`user_limits` таблица), сбрасывается ежесуточно в 00:00 UTC
- При достижении лимита: «Вы использовали 3/3 бесплатных запроса сегодня. Получите безлимитный доступ: [⭐ Подписка]»
- Оплата через Telegram Payments API с валютой XTR (Telegram Stars)
- Admin-список: массив `ADMIN_USER_IDS` в `.env`, проверяется до счётчика

**Acceptance Criteria:**
- Free-пользователь не может отправить 4-й запрос в сутки без подписки
- После оплаты через Telegram Stars — подписка активируется немедленно
- Администраторы из whitelist не ограничены
- Счётчик сбрасывается корректно в 00:00 UTC (Celery Beat задача)

---

### 4.5 Feature F5 — Кураторская база знаний (curated_kb)

**Описание:** Постоянная Qdrant-коллекция с проверенными командой материалами по популярным категориям товаров.

**MVP Seed-категории (гаджеты и смарт-девайсы):**
- Смартфоны (flagship + mid-range)
- Наушники (TWS, накладные, ANC)
- Ноутбуки (ультрабуки, игровые)
- Умные часы и фитнес-трекеры
- Планшеты
- Портативные колонки
- Смарт-телевизоры

**Источники для кураторской базы:** Wirecutter, RTINGS, 4PDA, специализированные обзорные издания.

**Требования к материалам:** не старше 1 года, авторитетный ресурс, не рекламный контент.

**Управление:** ручное добавление через `scripts/seed_curated_kb.py`, ревью ≤ раз в месяц.

**Acceptance Criteria:**
- Запрос по любой seed-категории возвращает ответ с бейджем `📚` без обращения к `auto_crawled`
- `updated_at` timestamp присутствует в каждом payload чанка
- Дубли исключаются при seed-загрузке

---

### 4.6 Feature F6 — Multi-turn диалог

**Описание:** Поддержка уточняющих вопросов в рамках сессии.

**Логика:**
- История последних 5 сообщений хранится в Redis (TTL 30 минут)
- Передаётся в LLM как `chat_history` в системном промпте
- Команда `/mode` переключает режим без сброса сессии

**Acceptance Criteria:**
- Уточняющий вопрос («а как насчёт батареи?») возвращает ответ с учётом предыдущего контекста
- Сессия истекает через 30 минут неактивности
- Переключение `/mode` не теряет историю текущей сессии

---

### 4.7 Feature F7 — Сравнение товаров

**Описание:** Структурированное сравнение двух и более товаров.

**Acceptance Criteria:**
- Запрос вида «iPhone 16 vs Samsung S25» возвращает таблицу сравнения
- Используются данные из обеих позиций базы (по каждому товару отдельный RAG-запрос, результаты мержатся)

---

### 4.8 Feature F8 — Фоновый сбор данных (Celery Async)

**Acceptance Criteria:**
- После постановки задачи пользователь немедленно получает «⏳ Ищу данные (~3 мин)...»
- Финальный push-ответ доставляется в течение 5 минут
- При падении задачи: retry ×3 → алерт в admin-чат → сообщение пользователю с извинением
- Celery Flower мониторинг: task success rate, очередь задач

---

## 5. Рекомендации по техническому стеку

### 5.1 Backend

| Компонент | Технология | Обоснование |
|---|---|---|
| API Framework | FastAPI (Python 3.11+) | Async-нативный, автодокументация, высокая производительность |
| Telegram Bot | aiogram 3.x (long polling → webhook) | Лучший async Python фреймворк для Telegram, активная поддержка |
| Task Queue | Celery + Redis broker | Стандарт для Python async задач, интеграция с Flower |
| Scheduler | Celery Beat | Периодические задачи (обновление кэша, сброс счётчиков) |

### 5.2 Data Layer

| Компонент | Технология | Обоснование |
|---|---|---|
| Vector DB | Qdrant (self-hosted) | Open-source, лучшая производительность для ANN-поиска, Docker-friendly |
| RDBMS | PostgreSQL + asyncpg + SQLAlchemy + Alembic | Надёжность, миграции, async поддержка |
| Cache / Session | Redis | TTL-кэш сессий, Celery broker — одна технология для двух задач |

### 5.3 AI / ML

| Компонент | Технология | Обоснование |
|---|---|---|
| LLM | gpt-4o-mini | $0.15/1M input tokens, 128K контекст, качество достаточное для агрегации |
| Embeddings | text-embedding-3-small | $0.02/1M tokens, 1536 dims, высокое качество для семантического поиска |
| API (dev) | GitHub Models API (бесплатно, Student Pack) | Тот же OpenAI-совместимый REST, нулевые затраты при разработке |
| API (prod) | OpenAI API | Переключение: только замена `OPENAI_API_KEY` + `OPENAI_BASE_URL` в `.env` |
| LLM Fallback | DeepSeek через OpenRouter ($0.07/1M) | Экономия при масштабировании, та же замена base_url |
| Web Search | Tavily API | Оптимизирован для LLM, возвращает чистый текст (не сырой HTML), free tier 1000 req/мес |

### 5.4 Data Collection

| Источник | Библиотека |
|---|---|
| YouTube транскрипты | `youtube-transcript-api` |
| YouTube поиск видео | YouTube Data API v3 (10K quota/день, бесплатно) |
| Reddit | PRAW (Python Reddit API Wrapper) |
| Веб-страницы | `trafilatura` |
| Chunking | `RecursiveCharacterTextSplitter` (LangChain) |

### 5.5 Infrastructure

| Компонент | Технология |
|---|---|
| Контейнеризация | Docker + Docker Compose |
| CI/CD | GitHub Actions (buildx multi-arch, push в GHCR, SSH deploy) |
| Rate Limiting | slowapi (FastAPI middleware) |
| Логирование | structlog → stdout |
| Мониторинг задач | Celery Flower (доступ через SSH-туннель) |
| Оркестрация Phase 3 | Kubernetes |

### 5.6 Монетизация

| Компонент | Технология |
|---|---|
| Платежи | Telegram Payments API (валюта: XTR — Telegram Stars) |
| Хранение подписок | PostgreSQL таблица `subscriptions` |

---

## 6. Модель данных

### 6.1 PostgreSQL — таблицы

#### `users`
```
user_id         BIGINT PRIMARY KEY         -- telegram_user_id
is_admin        BOOLEAN DEFAULT FALSE      -- whitelist admin
subscription    VARCHAR(20) DEFAULT 'free' -- 'free' | 'premium'
sub_expires_at  TIMESTAMP NULL             -- дата истечения подписки
created_at      TIMESTAMP DEFAULT NOW()
```

#### `user_limits`
```
user_id         BIGINT FK → users.user_id
date            DATE
requests_used   INTEGER DEFAULT 0
PRIMARY KEY (user_id, date)
```

#### `subscriptions`
```
id              SERIAL PRIMARY KEY
user_id         BIGINT FK → users.user_id
telegram_payment_charge_id  VARCHAR(256) UNIQUE
amount_stars    INTEGER
activated_at    TIMESTAMP
expires_at      TIMESTAMP
status          VARCHAR(20) -- 'active' | 'expired' | 'cancelled'
```

#### `sources`
```
id              SERIAL PRIMARY KEY
source_url      TEXT UNIQUE
source_type     VARCHAR(20) -- 'youtube' | 'reddit' | 'web' | 'tavily' | 'curated'
product_query   TEXT
parsed_at       TIMESTAMP
is_sponsored    BOOLEAN DEFAULT FALSE
is_curated      BOOLEAN DEFAULT FALSE
language        VARCHAR(10)
author          TEXT NULL
```

#### `jobs`
```
id              UUID PRIMARY KEY
user_id         BIGINT FK → users.user_id
job_type        VARCHAR(20) -- 'auto_search' | 'manual_links'
status          VARCHAR(20) -- 'pending' | 'running' | 'done' | 'failed'
product_query   TEXT
celery_task_id  VARCHAR(256)
created_at      TIMESTAMP
completed_at    TIMESTAMP NULL
```

#### `query_logs`
```
id              SERIAL PRIMARY KEY
user_id         BIGINT FK → users.user_id
session_id      VARCHAR(64)
mode            VARCHAR(20) -- 'auto' | 'manual'
query_text      TEXT
response_text   TEXT
sources_used    JSONB       -- [{url, source_type, is_curated}]
rating          SMALLINT NULL -- 1 (👍) | -1 (👎) | NULL
response_time_ms INTEGER
used_tavily     BOOLEAN DEFAULT FALSE
created_at      TIMESTAMP DEFAULT NOW()
```

### 6.2 Qdrant — коллекции

#### Коллекция `auto_crawled`
```
vector:    float[1536]   -- text-embedding-3-small
payload: {
  source_id:      int,
  source_url:     string,
  source_type:    string,   -- 'youtube' | 'reddit' | 'web' | 'tavily'
  product_query:  string,
  chunk_index:    int,
  language:       string,
  is_sponsored:   bool,
  is_curated:     bool,     -- always false in this collection
  author:         string | null,
  date:           string,   -- ISO 8601
  session_id:     string | null,  -- для режима ссылок
  text:           string
}
```

#### Коллекция `curated_kb`
```
vector:    float[1536]
payload: {
  source_id:      int,
  source_url:     string,
  source_type:    string,
  product_query:  string,
  category:       string,   -- 'smartphones' | 'headphones' | 'laptops' | 'smartwatches' | ...
  chunk_index:    int,
  language:       string,
  is_sponsored:   bool,     -- always false
  is_curated:     bool,     -- always true
  author:         string,
  date:           string,
  updated_at:     string,   -- ISO 8601, для контроля свежести
  text:           string
}
```

### 6.3 Redis — ключи

```
session:{user_id}:mode        STRING TTL=30min   -- 'auto' | 'manual'
session:{user_id}:history     LIST   TTL=30min   -- последние 5 сообщений (JSON)
session:{user_id}:chunks      LIST   TTL=30min   -- chunk_ids текущей сессии (режим ссылок)
```

---

## 7. Системный промпт LLM

```
Ты — беспристрастный аналитик потребительских отзывов. Твоя задача — агрегировать
мнения из реальных источников и предоставлять сбалансированный анализ.

ПРАВИЛА:
1. Отвечай ТОЛЬКО на основе предоставленного контекста. Не выдумывай факты.
2. Явно выделяй противоречивые мнения. Если есть критические отзывы — упомяни их.
3. Источники с флагом [sponsored] считай менее достоверными, указывай это в анализе.
4. Структурируй ответ: [✅ Плюсы] [❌ Минусы] [⚖️ Спорные моменты] [🏆 Вывод].
5. Если контекста недостаточно — честно скажи об этом, не додумывай.
6. Отвечай на языке пользователя.
7. В конце укажи: количество источников и наличие проверенных материалов (📚).
8. Отклоняй вопросы, не связанные с анализом товаров.

ПАРАМЕТРЫ ГЕНЕРАЦИИ: temperature=0.3, max_tokens=1000, top_p=0.9

КОНТЕКСТ: {retrieved_chunks}
ИСТОРИЯ ДИАЛОГА: {chat_history}
ВОПРОС ПОЛЬЗОВАТЕЛЯ: {user_query}
```

---

## 8. API эндпоинты

### FastAPI — публичные эндпоинты

```
POST /query
  Body: { user_id, session_id, query, mode: "auto"|"manual", urls?: [string] }
  Response: { answer, sources, job_id?, used_curated, used_tavily, response_time_ms }
  Notes: Проверяет лимит запросов. В auto-режиме может ответить немедленно или
         вернуть job_id для отслеживания.

POST /ingest
  Body: { user_id, session_id, urls: [string] }
  Response: { job_id, status: "queued" }
  Notes: Ставит задачу в Celery, немедленно возвращает job_id.

GET /status/{job_id}
  Response: { job_id, status, progress?, result? }

POST /feedback
  Body: { query_log_id, rating: 1|-1 }
  Response: { ok: true }

GET /health
  Response: { status: "ok", qdrant: bool, postgres: bool, redis: bool }
```

### Telegram Bot — команды

```
/start          Приветствие + выбор режима
/mode           Переключение авто/ссылки
/delete_my_data Удаление всей истории пользователя (GDPR)
/help           Справка
/subscribe      Оформить подписку через Telegram Stars
```

---

## 9. Принципы UI/UX (Telegram)

### Главное меню (`/start`)
```
Привет! Я ReviewMind — анализирую обзоры товаров.

Выбери режим:
[🔍 Авто-поиск]   [🔗 Свои ссылки]
```

### Ответ бота
```
📱 **Sony WH-1000XM5** — анализ 8 источников

✅ Плюсы
• Лучший ANC в классе по 6 источникам
• ...

❌ Минусы
• Пластиковая сборка — отмечена в 4 источниках
• ...

⚖️ Спорные моменты
• Звуковая сигнатура: часть аудиофилов считает её слишком «тёплой»

🏆 Вывод
Рекомендуется к покупке для пользователей, ценящих ANC.

📚 Использованы проверенные источники | 8 источников

[👍 Полезно]  [👎 Не то]  [📎 Источники]
```

### Кнопка «Источники»
```
📎 Источники:
1. 📚 Wirecutter — wirecutter.com/...
2. YouTube — youtube.com/watch?v=...
3. ⚠️ [sponsored] TechChannel — youtube.com/...
```

### Лимит исчерпан
```
Вы использовали 3/3 бесплатных запроса сегодня.

[⭐ Безлимит за 299₽/мес]   [Ждать до завтра]
```

---

## 10. Безопасность и соответствие требованиям

### Данные пользователя (GDPR / минимализм)
- Хранится только `telegram_user_id` (не имя, не номер телефона, не геолокация)
- Команда `/delete_my_data` удаляет: `query_logs`, `user_limits`, `subscriptions`, `users` (каскад)
- Политика конфиденциальности публикуется в `/start` боте

### Безопасность инфраструктуры
- API-ключи в `.env` (в `.gitignore`); в production — GitHub CI/CD Secrets
- Все сервисы в изолированной Docker-сети; наружу открыт только Telegram bot (long polling MVP)
- Rate limiting: slowapi middleware — 10 запросов/мин на `user_id`
- VPN (X-ray VLESS) на production-сервере для доступа к OpenAI API; проверяется в deploy-скрипте

### Этика
- Спонсорский контент явно помечается флагом `is_sponsored=true` и понижается в ранжировании (×0.7)
- При первом запуске бот явно сообщает, что является AI-системой
- Запросы вне тематики товаров отклоняются системным промптом
- Все ответы сопровождаются ссылками на источники

### Авторские права
- Используются только официальные API (YouTube, Reddit PRAW) и `trafilatura` для открытых страниц
- Хранятся фрагменты-чанки (не полный контент), всегда предоставляются ссылки на источники
- Fair use / information aggregation

---

## 11. Этапы разработки / Milestones

### 11.1 Roadmap (визуальный)

```mermaid
flowchart LR
    S0[Stage 0<br/>Foundation &<br/>CI/CD] --> S1[Stage 1<br/>Telegram ↔<br/>LLM Chat]
    S1 --> S2[Stage 2<br/>Data Layer<br/>Postgres + Qdrant]
    S2 --> S3[Stage 3<br/>Scrapers<br/>Shared Tools]
    S3 --> S4[Stage 4<br/>RAG Pipeline]
    S4 --> S5[Stage 5<br/>Manual Mode<br/>User Links]
    S5 --> S6[Stage 6<br/>Async Jobs<br/>Celery]
    S6 --> S7[Stage 7<br/>Monetization<br/>Stars + Limits]
    S7 --> S8[Stage 8<br/>Extra Features<br/>Curated KB + Tavily]
    S8 --> S9[Stage 9<br/>Auto Mode<br/>Full Pipeline]
```

### 11.2 Этапы в деталях
|---|---|---|---|
| **0. Фундамент и CI/CD** | Репозиторий, Docker Compose, GitHub Actions, multi-arch (arm64/amd64) | `git push` → автодеплой, все контейнеры зелёные | CI pipeline зелёный на PR |
| **1. Telegram ↔ LLM чат** | aiogram (long polling) + FastAPI + OpenAI клиент | Вопрос в Telegram → ответ LLM < 5 сек | E2E smoke-тест |
| **2. Слой данных** | PostgreSQL (миграции Alembic), Qdrant (2 коллекции), Redis | Все БД healthy, smoke-тесты проходят | `GET /health` → all green |
| **3. Парсеры** | YouTube, Reddit, trafilatura — изолированные модули | Каждый парсер работает автономно | Unit-тесты для каждого парсера |
| **4. RAG-пайплайн** | Embed → Qdrant upsert → гибридный поиск → LLM, seed-данные | Запрос по seed-продукту → релевантный ответ | Ручная проверка 20 запросов |
| **5. Режим ссылок** | Бот принимает URL → парсит → отвечает | Пользователь присылает 3 ссылки → получает анализ | E2E тест с реальными URL |
| **6. Celery async** | Фоновые задачи, push-уведомления, Celery Flower | 5 ссылок → «жду...» → push-ответ через ~3 мин | Celery task success rate ≥ 95% |
| **7. Монетизация** | User limits, Telegram Stars payments, admin whitelist | Free-пользователь ограничен 3 req/день; Stars оплата работает | Payment e2e тест |
| **8. Доп. фичи** | curated_kb seed (гаджеты), Tavily fallback, rate limiting | Популярная категория → `📚`; нишевый товар → Tavily | Интеграционные тесты |
| **9. Авто-режим** | Извлечение продукта → автопоиск YouTube + Reddit + полный pipeline | Только название товара → полный анализ | E2E тест авто-режима |

### Чек-лист готовности к запуску MVP (этапы 0–7)

- [ ] GitHub Actions: lint + тесты на PR, автодеплой на merge в main
- [ ] Контейнеры собираются на arm64 (dev) и amd64 (prod)
- [ ] VPN-проверка в деплой-скрипте (доступность OpenAI API)
- [ ] Telegram-бот: long polling, /start с выбором режима
- [ ] PostgreSQL: все миграции применены, таблицы созданы
- [ ] Qdrant: коллекции `auto_crawled` и `curated_kb` инициализированы и seed-загружены
- [ ] Парсеры YouTube / Reddit / trafilatura: unit-тесты зелёные
- [ ] RAG: ответы цитируют реальные источники, не галлюцинируют
- [ ] Режим ссылок: принимает URL всех трёх типов
- [ ] Celery + Flower: задачи выполняются, ошибки алертятся в admin-чат
- [ ] Монетизация: лимит 3 req/день работает, Stars-оплата активирует подписку
- [ ] Admins: ADMIN_USER_IDS whitelist обходит лимиты
- [ ] Rate limiting: 10 запросов/мин на user_id
- [ ] `/delete_my_data`: полное удаление данных пользователя

---

## 12. Потенциальные проблемы и митигация

| Риск | Вероятность | Влияние | Митигация |
|---|---|---|---|
| YouTube Transcript API: нет субтитров | Высокая | Среднее | Фильтрация видео < 500 слов, skip с уведомлением |
| Reddit API rate limit | Средняя | Среднее | Respect rate limits, агрессивное кэширование, только PRAW |
| trafilatura: paywall-страницы | Средняя | Низкое | Детектировать пустой результат → уведомить пользователя, skip |
| Галлюцинации LLM | Средняя | Высокое | Строгий RAG-промпт, порог confidence 0.75, всегда показывать источники |
| Рост стоимости OpenAI при масштабировании | Средняя | Среднее | Мониторинг costs, fallback на DeepSeek/Qwen через OpenRouter |
| Tavily free tier исчерпан (1000 req/мес) | Низкая | Низкое | 1000 req достаточно для MVP; платный тариф $9/мес |
| Кураторская база устарела | Средняя | Среднее | `updated_at` в payload, ревью раз в месяц |
| VPN недоступен → OpenAI недоступен | Средняя | Высокое | Проверка в deploy-скрипте, авто-рестарт VPN, Telegram-алерт |
| GitHub Models API нестабильность (dev) | Низкая | Низкое | Fallback на прямой OpenAI ключ; оба сконфигурированы в `.env` |
| Правовые риски: использование контента | Низкая | Высокое | Только официальные API, чанки (не полный контент), ссылки на источники |
| Telegram Stars: пользователи не конвертируют | Средняя | Среднее | A/B тест сообщений о лимите, возможно снизить Stars-цену |

---

## 13. Возможности будущего расширения (Phase 2+)

### Phase 2
- **Веб-интерфейс (React):** shared бэкенд с ботом, REST API уже готов
- **Telegram Webhook + Nginx:** переход с long polling, поддержка HTTPS
- **Amazon-отзывы:** интеграция через официальный Product Advertising API
- **Официальные страницы продуктов:** парсинг specs с производителей

### Phase 3
- **Kubernetes:** оркестрация при росте нагрузки (> 5K DAU)
- **Голосовой ввод:** Whisper API для транскрипции голосовых сообщений Telegram
- **Персонализация:** профиль предпочтений пользователя (приоритет характеристик)
- **Push-дайджест:** уведомление при появлении новых обзоров по сохранённым товарам

---

## 14. Бюджет и ресурсы

### Команда MVP
- 1 Full-stack Python разработчик (FastAPI + aiogram + Celery)
- 1 ML/NLP инженер (RAG pipeline, prompting, качество)
- (опционально) DevOps — часть времени

### Технологические затраты

| Ресурс | Разработка | MVP Production (500 DAU) |
|---|---|---|
| OpenAI / GitHub Models API | $0 (GitHub Student Pack) | ~$10–30/мес |
| YouTube Data API v3 | Бесплатно (10K quota/день) | Бесплатно |
| Reddit API (PRAW) | Бесплатно | Бесплатно |
| Tavily API | Бесплатно (1000 req/мес) | Бесплатно → $9/мес при росте |
| Qdrant | Бесплатно (self-hosted) | Бесплатно |
| PostgreSQL + Redis | Бесплатно (self-hosted) | Бесплатно |
| VPS / Сервер | **$0** (собственный Proxmox Xeon 2696v4) |
| **Итого** | **~$0/мес** | **~$10–36/мес** |

> **Embedding costs:** 500 users × 20 queries × ~1000 tokens = 10M tokens/мес → **$0.20** (`text-embedding-3-small`). Доминирующая статья — генерация ответов LLM.

### ROI
Экономия пользователю: 2–4 часа на одно решение о покупке → монетизация через freemium: подписка ≈ 299 ₽/мес через Telegram Stars.

---

## 15. Мониторинг и эксплуатация

| Компонент | Инструмент |
|---|---|
| Логирование | structlog → stdout |
| Мониторинг задач | Celery Flower (SSH-туннель) |
| Алерты ошибок | Telegram admin-чат (через aiogram) |
| Метрики качества | latency P50/P95, cost per session, Celery task success rate, % Tavily-ответов |
| Обновление auto_crawled | Celery Beat раз в 30 дней (топ-50 запросов) |
| Обновление curated_kb | Ручное, `scripts/seed_curated_kb.py`, ревью ≤ 1 раз в месяц |
| Деплой | `git push main` → GitHub Actions → `docker compose pull && up -d` |

---

## 16. Ключевые архитектурные решения

Этот раздел объясняет *почему* — обоснования каждого значимого технического выбора. Критически важно для разработчиков, чтобы не переосмыслять уже принятые решения.

**Почему две коллекции Qdrant, а не одна?**
`curated_kb` и `auto_crawled` имеют принципиально разные жизненные циклы и уровни доверия. Кураторский контент постоянный, проверен вручную, получает score boost при retrieval. Auto-crawled контент может быть устаревшим, предвзятым или редким. Раздельные коллекции позволяют буустить кураторские результаты без изменения алгоритма поиска, а также полностью пересобрать `auto_crawled` без влияния на надёжную базу.

**Почему парсеры строятся до режимов бота (Stage 3 до Stage 5 и 8)?**
Парсеры — это общий фундамент, используемый одинаково и в режиме ссылок (Stage 5), и в авто-режиме (Stage 8). Построить их один раз как чистые, независимо тестируемые модули — значит избежать дублирования кода. Stage 8 становится оркестрацией, а не новой логикой парсинга.

**Почему режим ссылок (Manual) до авто-режима (Auto)?**
Manual mode проще — нет извлечения названия продукта, нет вызовов search API. Он доказывает работоспособность полного RAG-пайплайна end-to-end и даёт рабочий продукт раньше. Auto mode — это лишь добавление автоматизированного поиска источников перед тем же пайплайном.

**Почему Qdrant, а не MongoDB Atlas Vector Search?**
Qdrant создан специально для векторного поиска: нативная фильтрация по payload, значительно меньшая latency для ANN-запросов, полностью бесплатен при self-hosting. Для системы, где векторный поиск является *основной* операцией, это правильный инструмент. PostgreSQL берёт на себя реляционный учёт.

**Почему gpt-4o-mini, а не o4-mini (reasoning model)?**
Reasoning-модели решают многошаговые логические задачи. Задача ReviewMind — синтез и суммаризация текста. gpt-4o-mini надёжно следует сложным структурированным промптам при цене ~10x ниже.

**Почему long polling сначала, webhook потом?**
Long polling не требует инфраструктуры: ни домена, ни HTTPS, ни Nginx. Webhook требует всего этого. Для разработки и раннего тестирования long polling запускается немедленно. Переход на webhook происходит только когда Web UI всё равно требует HTTPS.

**Почему Celery + Redis, а не FastAPI BackgroundTasks?**
FastAPI BackgroundTasks работают in-process и не переживают рестарт. Celery задачи долговременны, поддерживают retry, мониторятся через Flower и масштабируются горизонтально — критично, когда задачи парсинга занимают 3–5 минут.

**Почему GitHub Student Pack API для разработки, затем смена на production OpenAI?**
Оба используют одинаковый OpenAI-совместимый REST API. Отличаются только `OPENAI_API_KEY` и `OPENAI_BASE_URL`. Бесплатные dev-кредиты → нулевые затраты при разработке. Одно изменение `.env` для перехода в production.

**Почему Telegram Stars для монетизации, а не внешний эквайринг?**
Telegram Stars — нативная платёжная система Telegram, не требует выхода из мессенджера, интегрируется через стандартный Telegram Payments API, не требует merchant account на старте. Оптимально для MVP-аудитории, живущей в Telegram.

---

## 17. Деплой и требования к инфраструктуре

### Предварительные требования

Перед запуском должны быть получены следующие credentials:

| Ресурс | Где получить | Примечание |
|---|---|---|
| Telegram Bot Token | @BotFather в Telegram | Создать бота, получить токен |
| OpenAI API Key | platform.openai.com / GitHub Student Pack | Student Pack: github.com/education |
| YouTube Data API v3 Key | console.cloud.google.com | Включить YouTube Data API v3, создать API key |
| Reddit App Credentials | reddit.com/prefs/apps | Тип: script, получить client_id + client_secret |
| Tavily API Key | tavily.com | Free tier: 1000 req/мес |

### Инфраструктура

**Production-сервер:** Intel Xeon E5-2696 v4 (собственное железо, Proxmox), amd64  
**Dev-машина:** macOS Apple Silicon, arm64  
**Минимальные требования:** 2 vCPU, 4 GB RAM, 40 GB SSD  
**VPN:** X-ray VLESS обязателен на production-сервере для доступа к OpenAI API. Проверяется в deploy-скрипте перед стартом контейнеров.

### Переменные окружения (`.env`)

```env
# Telegram
TELEGRAM_BOT_TOKEN=
ADMIN_USER_IDS=123456789,987654321   # comma-separated

# OpenAI (dev: GitHub Models)
OPENAI_API_KEY=
OPENAI_BASE_URL=https://models.inference.ai.azure.com  # dev; https://api.openai.com/v1 for prod

# YouTube
YOUTUBE_API_KEY=

# Reddit
REDDIT_CLIENT_ID=
REDDIT_CLIENT_SECRET=
REDDIT_USER_AGENT=ReviewMind/1.0

# Tavily
TAVILY_API_KEY=

# PostgreSQL
POSTGRES_USER=reviewmind
POSTGRES_PASSWORD=
POSTGRES_DB=reviewmind
DATABASE_URL=postgresql+asyncpg://reviewmind:PASSWORD@postgres:5432/reviewmind

# Redis
REDIS_URL=redis://redis:6379/0

# Qdrant
QDRANT_URL=http://qdrant:6333
```

### Docker Compose сервисы (MVP)

```yaml
services:
  api:       # FastAPI — REST backend (port 8000, internal only)
  bot:       # aiogram — long polling (MVP), webhook Phase 2
  worker:    # Celery worker — parsing + embedding
  beat:      # Celery Beat — scheduled updates (30 days), daily limit reset
  flower:    # Celery Flower — task monitoring (SSH tunnel access)
  qdrant:    # Vector DB, port 6333 (internal)
  postgres:  # Relational DB, port 5432 (internal)
  redis:     # Broker + session cache, port 6379 (internal)
  # nginx:   # Добавляется в Phase 2 с webhook + Web UI
```

### CI/CD pipeline (GitHub Actions)

1. **On PR:** lint (ruff) + unit tests (pytest)
2. **On merge to main:** buildx multi-arch build (linux/amd64 + linux/arm64) → push to GHCR → SSH deploy: `docker compose pull && docker compose up -d`

---

## 18. Словарь терминов

| Термин | Определение |
|---|---|
| RAG | Retrieval-Augmented Generation — LLM получает релевантные фрагменты из базы знаний |
| Qdrant | Open-source векторная БД, оптимизированная для ANN-поиска |
| Embedding | Числовое векторное представление текста, сохраняющее семантическое сходство |
| Celery | Python-фреймворк для распределённой асинхронной обработки задач |
| Chunk | Фрагмент текста (400–600 токенов, overlap 50) для векторизации |
| `auto_crawled` | Qdrant-коллекция: контент, собранный ботом или предоставленный пользователем |
| `curated_kb` | Qdrant-коллекция: кураторские материалы команды, приоритетная при поиске |
| Score boost | Повышение релевантности кураторских чанков (×1.3) при merging результатов поиска |
| Tavily | API для веб-поиска, оптимизированный для LLM: возвращает чистый текст |
| trafilatura | Python-библиотека для извлечения основного текста с веб-страниц |
| aiogram | Async Python-библиотека для Telegram Bot API |
| PRAW | Python Reddit API Wrapper |
| Long polling | Режим работы Telegram-бота без домена и HTTPS (MVP) |
| Telegram Stars (XTR) | Нативная валюта Telegram для внутренних платежей |

---

## 19. Ссылки на документацию

- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [GitHub Models API](https://docs.github.com/en/github-models)
- [Telegram Payments API](https://core.telegram.org/bots/payments)
- [Telegram Stars](https://telegram.org/blog/telegram-stars)
- [youtube-transcript-api](https://github.com/jdepoix/youtube-transcript-api)
- [aiogram 3.x docs](https://docs.aiogram.dev/)
- [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
- [PRAW Documentation](https://praw.readthedocs.io/)
- [trafilatura](https://trafilatura.readthedocs.io/)
- [Tavily API](https://tavily.com/)
- [Celery Documentation](https://docs.celeryq.dev/)
- [slowapi Rate Limiting](https://github.com/laurentS/slowapi)
