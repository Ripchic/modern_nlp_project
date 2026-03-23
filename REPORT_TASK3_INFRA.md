# Отчет по Заданию 3: Инфраструктура и деплой

## Обзор

Инфраструктура ReviewMind включает 10 сервисов в Docker Compose, CI/CD пайплайн (GitHub Actions), мониторинг (Prometheus + Grafana), миграции базы данных (Alembic), скрипты инициализации и конфигурацию прокси для работы из РФ.

## Docker

### Dockerfile

Двухступенчатая сборка:
1. **builder** — `python:3.11-slim`, установка `uv 0.5.26`, разрешение зависимостей из `pyproject.toml`
2. **runtime** — чистый `python:3.11-slim`, копирование `.venv` из builder, запуск от `appuser` (UID 1001), non-root

Разделение этапов сокращает итоговый образ: в runtime нет pip, uv, кэша сборки.

### Docker Compose

Три файла:
- `docker-compose.yml` — базовая конфигурация, 10 сервисов, внутренняя сеть `reviewmind-network`, volumes для persistence
- `docker-compose.override.yml` — локальная разработка (порты наружу для отладки)
- `docker-compose.prod.yml` — production: экспозиция портов 8000/3000/9090/6333/5555, настройка HTTP-прокси через `host.docker.internal`

Сервисы:

| Сервис | Образ / target | Назначение |
|--------|---------------|------------|
| postgres | postgres:16-alpine | Хранение пользователей, запросов, лимитов, фидбэка |
| redis | redis:7-alpine | Сессии бота (TTL 30 мин), брокер Celery (db1), результаты (db2) |
| qdrant | qdrant/qdrant:v1.12.0 | Векторные коллекции: auto_crawled, curated_kb |
| api | Dockerfile:runtime | FastAPI, /health, /metrics, POST /query, POST /ingest |
| bot | Dockerfile:runtime | aiogram long polling, хэндлеры Telegram |
| worker | Dockerfile:runtime | Celery worker, concurrency=4, фоновый парсинг |
| beat | Dockerfile:runtime | Celery Beat, расписание: daily limit reset, monthly top-queries |
| flower | Dockerfile:runtime | Web-UI мониторинг Celery задач, порт 5555 |
| prometheus | prom/prometheus:v2.51.0 | Сбор метрик с FastAPI (/metrics/) и Qdrant (/metrics) |
| grafana | grafana/grafana:11.0.0 | 3 дашборда, 2 datasource (Prometheus, PostgreSQL) |

Healthcheck прописан для postgres, redis, qdrant, api, prometheus, grafana. Сервисы api/bot/worker стартуют только после `condition: service_healthy` зависимостей.

## CI/CD

### CI (`.github/workflows/ci.yml`)

Срабатывает на push и PR в main:
1. **Lint** — `ruff check .` + `ruff format --check .`
2. **Tests** — матрица Python 3.11 + 3.12, `pytest` с coverage, артефакт `coverage.xml`

Зависимости ставятся через `uv pip install --system -e '.[dev]'`. Fake-токены для Telegram/OpenAI передаются через env.

### CD (`.github/workflows/cd.yml`)

Срабатывает на push в main:
1. **Build & Push** — Docker Buildx, multi-arch `linux/amd64 + linux/arm64`, push в GHCR с тегами `sha` и `latest`, GHA cache
2. **Deploy** — SSH на production VM (порт 10202), `git pull --ff-only`, `docker compose build --no-cache`, миграции Alembic, `up -d --force-recreate`, health check wait loop (30 итераций по 1 секунде)

## Мониторинг

### Prometheus

Два scrape target:
- `fastapi` — `api:8000/metrics/`, сбор метрик `prometheus-client` (request count, latency, in-progress)
- `qdrant` — `qdrant:6333/metrics`, встроенные метрики Qdrant (collection size, search latency)

Интервал: 15 секунд.

### Grafana

Два datasource (provisioned, не требуют ручной настройки):
- Prometheus (default) — для метрик приложения и Qdrant
- PostgreSQL — для SQL-запросов по бизнес-данным (лимиты, запросы, фидбэк)

Три дашборда (JSON, auto-provisioned):
- **System Health** — HTTP request rate, latency percentiles, error rate, uptime
- **User Activity** — запросы по времени, режимы (auto/manual), лимиты, фидбэк
- **Database Stats** — размер таблиц, count записей, активные сессии

## Миграции и скрипты

### Alembic

Одна миграция `0001_initial_schema.py` — 6 таблиц: `users`, `queries`, `source_documents`, `feedback`, `daily_limits`, `payments`. Async driver `asyncpg`. Миграция запускается при деплое через `docker compose run --rm api alembic upgrade head`.

### Скрипты инициализации

- `scripts/init_qdrant.py` — создание коллекций `auto_crawled` и `curated_kb` (1536 dims, cosine)
- `scripts/seed_curated_kb.py` — загрузка 14 курированных статей по 7 категориям
- `scripts/seed_test_data.py` — тестовые данные для верификации RAG
- `scripts/reset_limit.py` — ручной сброс лимитов пользователя

## Прокси-конфигурация

Сервер развернут в РФ, доступ к Telegram API и внешним ресурсам требует прокси. Используется xray на хосте, контейнеры подключаются через `host.docker.internal:1081`.

В `docker-compose.prod.yml`:
```yaml
environment:
  - HTTPS_PROXY=http://host.docker.internal:1081
  - HTTP_PROXY=http://host.docker.internal:1081
  - NO_PROXY=localhost,127.0.0.1,postgres,redis,qdrant
```

Для aiogram (Telegram long polling) потребовалась отдельная настройка: стандартный `aiohttp` не поддерживает SOCKS/HTTP-прокси из коробки — добавлена зависимость `aiohttp-socks`.


## Проблемы при деплое

### 1. SSH порт на production VM

CD workflow использовал порт 22 по умолчанию. На VM SSH работает на порту 10202 (нестандартный, для защиты от брутфорса). Потребовалось 3 коммита:
- Попытка hardcode порта 22
- Исправление на 10202
- Финальная стабилизация workflow

### 2. Прокси-маршрутизация через xray

Контейнеры bot и worker не могли подключиться к Telegram API и внешним URL. Проблема: xray слушает на хосте, контейнеры изолированы в Docker network. Решение: `extra_hosts: host.docker.internal:host-gateway` + переменные `HTTP_PROXY`/`HTTPS_PROXY` в prod overlay.

### 3. aiogram + HTTP proxy

aiogram использует aiohttp для long polling. aiohttp не умеет работать с HTTP-прокси через переменные окружения — требуется явный proxy connector. Добавлена зависимость `aiohttp-socks`, настройка proxy в bot session initialization.

### 4. Celery Beat schedule path

Celery Beat по умолчанию пишет schedule-файл в текущую директорию. В контейнере с read-only файловой системой это вызывает ошибку. Решение: явный путь `--schedule=/tmp/celerybeat-schedule`.

### 5. CI/CD pipeline fixes

Несколько итераций исправлений:
- Ошибки в синтаксисе workflow YAML
- Отсутствие fake-токенов в env для тестов (Telegram, OpenAI)
- Проблемы с `uv` кэшированием в GitHub Actions

### 6. Tavily source URL persistence

Tavily scraper возвращал URL источников, но они не сохранялись в PostgreSQL при фоновой обработке через Celery. Причина: background job получал только тексты, но не метаданные об источниках. Исправлено передачей source URLs через Celery task arguments.

## Использование LLM

Генерация Dockerfile, compose-файлов, CI/CD workflow, Prometheus/Grafana конфигов — в основном через GPT-4o / Claude. Основная ручная работа — отладка деплоя: SSH-порты, прокси-маршрутизация, aiohttp-socks интеграция. Эти проблемы не решаются LLM, т.к. зависят от конкретной инфраструктуры (VPS, xray, нестандартные порты).
