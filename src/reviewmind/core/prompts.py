"""reviewmind/core/prompts.py — LLM system prompts and context builders.

Implements the system prompt from PRD section 7.
Provides helpers for formatting retrieved chunks and chat history
before they are inserted into the prompt template.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ── Constants ──────────────────────────────────────────────────────────────────

#: Marker appended to sponsored source text in the context block.
SPONSORED_MARKER = "[sponsored]"

#: Marker prepended to curated/trusted source text in the context block.
CURATED_MARKER = "📚"

#: Message injected as the context block when no chunks are available.
NO_CONTEXT_TEXT = (
    "Контекст по данному запросу отсутствует. "
    "Информация из внешних источников ещё не была загружена."
)

#: Message injected into chat history slot when history is empty.
NO_HISTORY_TEXT = "История диалога отсутствует."

# ── Prompt template (PRD section 7) ───────────────────────────────────────────

#: Full RAG system-prompt template. Slots: {retrieved_chunks}, {chat_history}.
#: The user query is passed as the *user_message* in the chat-completion call,
#: so it is NOT embedded in this template.
SYSTEM_PROMPT_TEMPLATE = """\
Ты — ReviewMind, профессиональный AI-аналитик потребительских отзывов.
Твоя задача — изучить предоставленный контекст из реальных источников (интернет, ссылки, видеообзоры, статьи) и дать структурированный, полезный и объективный ответ.

ИНСТРУКЦИИ В ЗАВИСИМОСТИ ОТ СЦЕНАРИЯ ЗАПРОСА:

Сценарий А: Обзор конкретного товара (найденного в сети или по отправленной ссылке)
- Если пользователь просит обзор на один конкретный товар, сфокусируйся на нём.
- Оцени надежность мнений. Если обзор кажется рекламным или источник отмечен как [sponsored], ОБЯЗАТЕЛЬНО упомяни об этом.
- Выдели основные плюсы, минусы и спорные моменты.
- Оформи ответ так, как комфортно читать: используй заголовки (например: "✅ Плюсы", "❌ Минусы", "⚖️ Спорные моменты", "🏆 Вывод").

Сценарий Б: Выбор и сравнение товаров (рекомендация / общие запросы)
- Если запрос широкий (например, "лучшие беговые лыжи", "выбери хорошие товары Х"), найди в контексте наиболее упоминаемые и хорошо оцененные товары.
- Выдели 2-4 лучших модели или бренда из контекста (если они там есть).
- Для каждого варианта кратко опиши его сильные стороны, недостатки и кому он больше подойдет.
- В конце сделай небольшое резюме-сравнение.

ОБЩИЕ ПРАВИЛА (СТРОГО ОБЯЗАТЕЛЬНЫ):
1. Использование контекста: Если ниже приведен контекст, ТЫ ОБЯЗАН использовать его для ответа. Даже если контекст — это сырая транскрипция видео без чёткой структуры, извлеки из него всю полезную информацию.
2. Базовые знания: Не выдумывай факты и характеристики товаров для конкретных обзоров. Но если контекст по общему запросу скудный, ты можешь дополнить ответ общими знаниями и проверенными моделями (явно указав, что это "общие рекомендации").
3. Оформление заголовков: Используй только эмодзи и текст (например, ✅ Плюсы). НЕ ИСПОЛЬЗУЙ квадратные скобки [] и звёздочки ** для заголовков.
4. Оформление текста: НЕ ИСПОЛЬЗУЙ жирный шрифт (**текст**) внутри списков и абзацев, пиши чистым обычным текстом, чтобы избежать сломанной верстки в Telegram.
5. Язык: Отвечай на языке запроса пользователя (целевой язык: {response_language}).
6. Сдержанность: Отклоняй вопросы, совершенно не связанные с товарами, покупками или обзорами.

ПАРАМЕТРЫ ГЕНЕРАЦИИ: temperature=0.3, max_tokens=1000, top_p=0.9

КОНТЕКСТ:
{retrieved_chunks}

ИСТОРИЯ ДИАЛОГА:
{chat_history}"""

#: Fallback system prompt used when there are NO retrieved chunks at all
#: (e.g. direct-LLM mode or before RAG pipeline is ready).
FALLBACK_SYSTEM_PROMPT = """\
Ты — ReviewMind, профессиональный AI-помощник по выбору товаров. 
Сейчас ты работаешь в режиме общих знаний (без актуального контекста из интернета).

Твоя задача — помогать с общими вопросами по выбору товаров, давать советы и ориентировать в брендах/моделях.

ПРАВИЛА:
1. Если пользователь хочет выбрать товар (например, "лучшие беговые лыжи", "какие наушники купить"):
   - Порекомендуй 3-4 знаменитых бренда или проверенные временем модели из твоей базы.
   - Укажи, для каких задач подходит каждая рекомендация.
   - В конце всегда предлагай: "💡 Совет: назовите конкретную модель из этого списка или отправьте ссылку на её обзор, и я найду свежую детализированную информацию в интернете."
2. Будь объективен, указывай как плюсы, так и потенциальные минусы.
3. Оформление заголовков: Используй только эмодзи и текст (например, ✅ Плюсы, 🏆 Вывод). НЕ ИСПОЛЬЗУЙ квадратные скобки [] и звёздочки ** для заголовков.
4. Оформление списков: НЕ ИСПОЛЬЗУЙ жирный шрифт (**текст**) внутри списков и абзацев, используй простой текст. Платформа Telegram не всегда корректно рендерит сложный Markdown.
5. Язык: Отвечай на языке запроса пользователя.
6. Сдержанность: Отклоняй вопросы, совершенно не связанные с товарами (например, математика, политика).
"""

# ── RAG generation parameters ─────────────────────────────────────────────────

#: LLM parameters specified in PRD section 7.
RAG_TEMPERATURE: float = 0.3
RAG_MAX_TOKENS: int = 1000
RAG_TOP_P: float = 0.9

# ── Data types ────────────────────────────────────────────────────────────────


@dataclass
class ChunkContext:
    """A single retrieved text chunk with its metadata.

    These objects are passed to :func:`format_chunks_for_context` to build the
    ``{retrieved_chunks}`` slot of the system prompt.

    Attributes
    ----------
    text:
        The raw text of the chunk.
    source_url:
        URL of the original source.
    source_type:
        Type of source, e.g. ``"youtube"``, ``"reddit"``, ``"web"``.
    is_sponsored:
        ``True`` if the source is flagged as sponsored content.
    is_curated:
        ``True`` if the source is from the verified curated knowledge base.
    score:
        Similarity / reranking score (higher = more relevant).
    author:
        Optional author / channel name.
    extra_metadata:
        Arbitrary additional metadata (not used in prompt building).
    """

    text: str
    source_url: str
    source_type: str = "web"
    is_sponsored: bool = False
    is_curated: bool = False
    score: float = 0.0
    author: str | None = None
    extra_metadata: dict = field(default_factory=dict)


# ── Formatter helpers ─────────────────────────────────────────────────────────


def format_chunks_for_context(chunks: list[ChunkContext]) -> str:
    """Convert a list of :class:`ChunkContext` objects into a human-readable
    context block suitable for the ``{retrieved_chunks}`` template slot.

    Each chunk is numbered and annotated with its origin and any relevant flags
    (sponsored / curated).  Sponsored chunks are marked with
    :data:`SPONSORED_MARKER`; curated ones receive :data:`CURATED_MARKER`.

    Parameters
    ----------
    chunks:
        Ordered list of retrieved chunks (typically already reranked).

    Returns
    -------
    str
        Multi-line context string, or :data:`NO_CONTEXT_TEXT` when *chunks*
        is empty.
    """
    if not chunks:
        return NO_CONTEXT_TEXT

    # Add a summary line so the LLM knows the actual number of unique sources
    unique_urls = {c.source_url for c in chunks if c.source_url}
    lines: list[str] = [
        f"Всего уникальных источников: {len(unique_urls)}, фрагментов: {len(chunks)}",
        "",
    ]
    for i, chunk in enumerate(chunks, start=1):
        # Build the source label
        parts: list[str] = []

        if chunk.is_curated:
            parts.append(CURATED_MARKER)

        source_label = chunk.source_type.upper()
        if chunk.author:
            source_label = f"{source_label} ({chunk.author})"
        parts.append(source_label)

        if chunk.is_sponsored:
            parts.append(SPONSORED_MARKER)

        header = f"[{i}] {' '.join(parts)} | {chunk.source_url}"
        lines.append(header)
        lines.append(chunk.text.strip())
        lines.append("")  # blank line separator

    return "\n".join(lines).rstrip()


def format_chat_history(history: list[dict[str, str]]) -> str:
    """Convert a list of ``{"role": ..., "content": ...}`` messages into a
    readable history string for the ``{chat_history}`` template slot.

    Parameters
    ----------
    history:
        List of prior chat messages in OpenAI message format.

    Returns
    -------
    str
        Formatted history text, or :data:`NO_HISTORY_TEXT` when *history*
        is empty.
    """
    if not history:
        return NO_HISTORY_TEXT

    role_labels: dict[str, str] = {
        "user": "Пользователь",
        "assistant": "ReviewMind",
        "system": "Система",
    }

    lines: list[str] = []
    for msg in history:
        role = msg.get("role", "unknown")
        content = msg.get("content", "").strip()
        label = role_labels.get(role, role.capitalize())
        lines.append(f"{label}: {content}")

    return "\n".join(lines)


#: Default language code injected into the prompt when none is detected.
DEFAULT_PROMPT_LANGUAGE: str = "ru"


def build_rag_system_prompt(
    chunks: list[ChunkContext],
    chat_history: list[dict[str, str]] | None = None,
    *,
    language: str | None = None,
) -> str:
    """Build the full RAG system prompt by filling in the template slots.

    Parameters
    ----------
    chunks:
        Retrieved and reranked context chunks.  May be empty — in that case
        a "no context" notice is injected.
    chat_history:
        Optional conversation history.
    language:
        ISO 639-1 language code for the response (e.g. ``'ru'``, ``'en'``).
        When *None*, defaults to :data:`DEFAULT_PROMPT_LANGUAGE`.

    Returns
    -------
    str
        Fully rendered system prompt ready to be sent as the ``system``
        message in a chat-completion request.
    """
    context_text = format_chunks_for_context(chunks)
    history_text = format_chat_history(chat_history or [])
    lang = language or DEFAULT_PROMPT_LANGUAGE

    return SYSTEM_PROMPT_TEMPLATE.format(
        retrieved_chunks=context_text,
        chat_history=history_text,
        response_language=lang,
    )
