"""reviewmind/bot/handlers/start.py — /start и /help команды."""

from aiogram import Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message

from reviewmind.bot.keyboards import mode_keyboard

router = Router(name="start")

WELCOME_TEXT = (
    "👋 Привет! Я <b>ReviewMind</b> — AI-система для анализа обзоров товаров.\n\n"
    "🤖 <i>Обратите внимание: я являюсь AI-системой и генерирую ответы "
    "на основе анализа открытых источников. Мои выводы могут содержать неточности.</i>\n\n"
    "Я собираю обзоры с YouTube, Reddit, экспертных сайтов и формирую "
    "структурированный анализ: плюсы, минусы, спорные моменты и итоговый вывод.\n\n"
    "Выбери режим работы:"
)

HELP_TEXT = (
    "📖 <b>ReviewMind — справка</b>\n\n"
    "<b>Режимы работы:</b>\n"
    "🔍 <b>Авто-поиск</b> — напиши название товара, и я сам найду обзоры\n"
    "🔗 <b>Свои ссылки</b> — отправь ссылки на обзоры, и я проанализирую их\n\n"
    "<b>Команды:</b>\n"
    "/start — Начать заново, выбрать режим\n"
    "/mode — Переключить режим (авто-поиск / свои ссылки)\n"
    "/subscribe — Оформить подписку Premium\n"
    "/help — Показать эту справку\n"
    "/delete_my_data — Удалить все мои данные\n\n"
    "<b>Поддерживаемые источники:</b>\n"
    "• YouTube (субтитры видеообзоров)\n"
    "• Reddit (посты и комментарии)\n"
    "• Веб-страницы (статьи и обзоры)\n\n"
    "<b>Формат ответа:</b>\n"
    "✅ Плюсы | ❌ Минусы | ⚖️ Спорные моменты | 🏆 Вывод\n\n"
    "⚠️ Спонсорский контент помечается и понижается в ранжировании."
)


@router.message(CommandStart())
async def cmd_start(message: Message) -> None:
    """Handle /start command — send welcome message with mode selection keyboard."""
    await message.answer(WELCOME_TEXT, reply_markup=mode_keyboard(), parse_mode="HTML")


@router.message(Command("help"))
async def cmd_help(message: Message) -> None:
    """Handle /help command — send help and usage instructions."""
    await message.answer(HELP_TEXT, parse_mode="HTML")
