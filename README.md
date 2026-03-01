# ReviewMind 🔍

> **AI-powered pre-purchase research aggregator** — ask a product question, get a balanced analysis from dozens of YouTube reviews and Reddit discussions in seconds.

---

## The Problem

Before buying anything, people spend **hours** watching YouTube reviews and browsing Reddit — yet:
- Most videos are sponsored and biased
- Each new source adds only ~5% of new knowledge
- The one honest critical review gets buried under 50 positive ones
- Manually aggregating and weighing contradictory opinions is exhausting

## The Solution

ReviewMind automatically collects transcripts from YouTube and discussions from Reddit, stores them in a vector database, and answers your specific questions with a **balanced, source-cited analysis** — highlighting both mainstream praise and critical minority opinions.

Two modes of operation:
- **Auto mode** — give the bot a product name, it finds and fetches sources itself
- **Manual mode** — paste your own links (YouTube, Reddit, articles), the bot analyzes exactly what you provided

A **curated knowledge base** maintained by the team covers the most popular product categories out of the box, so common queries get instant answers without any scraping delay.

---

## Architecture Overview

### High-Level System

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

### Request Flow

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

### Data Ingestion Pipeline

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

### RAG Query Pipeline

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

## Tech Stack

| Layer | Technology | Why |
|---|---|---|
| **Bot** | aiogram 3.x | Best async Telegram library for Python |
| **API** | FastAPI + uvicorn | Async, fast, auto-docs |
| **Task Queue** | Celery + Redis | Durable background jobs, survive restarts, easy retry |
| **Vector DB** | Qdrant (self-hosted, 2 collections) | Purpose-built ANN, free, payload filters, score boost |
| **Relational DB** | PostgreSQL + asyncpg | Metadata, logs, source deduplication |
| **Cache / Broker** | Redis | Session context (30min TTL) + Celery broker |
| **Embeddings** | OpenAI text-embedding-3-small | $0.02/1M tokens, multilingual, no GPU needed |
| **LLM** | gpt-4o-mini via OpenAI API | Best price/quality for synthesis; GitHub Student Pack for dev |
| **YouTube** | youtube-transcript-api + YouTube Data API v3 | Official + free transcript extraction |
| **Reddit** | PRAW (official API) | Rate-limit safe, stable |
| **Web scraping** | trafilatura | Extracts clean article text from any URL, no headless browser |
| **Web search fallback** | Tavily API (free tier: 1000 req/month) | LLM-ready results, zero-shot answers when DB is empty |
| **Containers** | Docker + Docker Compose | Isolation, reproducibility, arm64 + amd64 compatible |
| **CI/CD** | GitHub Actions | Auto test + build + deploy on push |
| **Proxy** | Nginx + Let's Encrypt | TLS termination (required for webhook, Phase 2+) |

> **API key strategy:** use GitHub Student Developer Pack (free GPT-4o mini access) during development. Switch `OPENAI_API_KEY` and `OPENAI_BASE_URL` to production OpenAI credentials before launch — zero code changes required as both use the same OpenAI-compatible REST interface.

---

## Project Structure
### WIP

---

## Deployment

### Prerequisites
- Docker + Docker Compose
- Domain name (for HTTPS webhook)
- OpenAI API key
- YouTube Data API v3 key
- Reddit app credentials (from reddit.com/prefs/apps)
- Telegram Bot token (from @BotFather)

### Quick Start

```bash
# TO-DO
```

### Environment Variables

```env
# TO-DO
```

---

## Development Roadmap

The project is built sequentially in clearly isolated stages. Each stage has defined acceptance criteria — **nothing moves forward until the current stage is verified end-to-end**. This allows confident incremental progress and easy debugging since only one new system is introduced at a time.

```mermaid
flowchart LR
    S0[Stage 0<br/>Foundation &<br/>CI/CD] --> S1[Stage 1<br/>Telegram ↔<br/>LLM Chat]
    S1 --> S2[Stage 2<br/>Data Layer<br/>Postgres + Qdrant]
    S2 --> S3[Stage 3<br/>Scrapers<br/>Shared Tools]
    S3 --> S4[Stage 4<br/>RAG Pipeline]
    S4 --> S5[Stage 5<br/>Manual Mode<br/>User Links]
    S5 --> S6[Stage 6<br/>Async Jobs<br/>Celery]
    S6 --> S7[Stage 7<br/>Extra Features<br/>Curated KB + Tavily]
    S7 --> S8[Stage 8<br/>Auto Mode<br/>Full Pipeline]
```

---

## Cost Estimate (MVP)

| Resource | Cost/month |
|---|---|
| VPS (private proxmox xeon2696v4) | Priceless |
| OpenAI API (500 users, ~20 queries/user) | ~$15–30 |
| YouTube Data API v3 | Free (10K quota/day) |
| Reddit PRAW | Free |
| Qdrant (self-hosted) | Free |
| **Total** | **~$15–30/month** |

> **Embedding cost breakdown:** 500 users × 20 queries × ~1000 tokens = 10M tokens/month → **$0.20** (text-embedding-3-small). The dominant cost is LLM generation, not embeddings — no need to self-host.

---

## Key Design Decisions

**Why two Qdrant collections instead of one?**
`curated_kb` and `auto_crawled` have fundamentally different lifecycles and trust levels. Curated content is permanent, manually vetted, and gets a score boost in retrieval. Auto-crawled content can be stale, biased, or sparse. Keeping them separate lets us boost curated results without touching the retrieval algorithm, and lets us wipe/rebuild `auto_crawled` without affecting the trusted baseline.

**Why scrapers are built before the bot modes?**
Scrapers (Stage 3) are the shared foundation used identically by both manual mode (Stage 5) and auto mode (Stage 8). Building them once as clean, independently testable modules avoids duplication and means Stage 8 is mostly orchestration, not new scraping logic.

**Why manual mode before auto mode?**
Manual mode is simpler (no topic extraction, no search API calls), proves the full RAG pipeline end-to-end, and gives you a working product earlier. Auto mode is just adding an automated source-discovery step before the same pipeline.

**Why Qdrant over MongoDB Atlas Vector Search?**
Qdrant is purpose-built for vector search with native payload filtering, significantly lower latency for ANN queries, and is completely free when self-hosted. For a system where vector search is the *core* operation, it's the right tool. PostgreSQL handles relational bookkeeping.

**Why gpt-4o-mini over o4-mini (reasoning model)?**
Reasoning models solve multi-step logical problems. Your task is text synthesis and summarization — gpt-4o-mini follows complex structured prompts reliably at ~10x lower cost.

**Why long polling first, webhook later?**
Long polling requires zero infrastructure (no domain, no HTTPS, no Nginx). Webhook requires all of that. For development and early testing, long polling gets you running immediately. Switch to webhook only when the web UI requires HTTPS anyway.

**Why Celery + Redis over FastAPI BackgroundTasks?**
FastAPI BackgroundTasks run in-process and don't survive a restart. Celery tasks are durable, retryable, monitorable via Flower, and scale horizontally — critical when scraping jobs take 3–5 minutes.

**Why GitHub Student Pack API for dev, then swap to production OpenAI?**
Both use the same OpenAI-compatible REST API. Only `OPENAI_API_KEY` and `OPENAI_BASE_URL` differ. Free dev credits → zero cost while building. One `.env` change to go production.

---