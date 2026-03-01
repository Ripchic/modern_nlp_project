"""reviewmind/bot/handlers/mode.py — Обработка выбора и переключения режима."""

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import CallbackQuery, Message

from reviewmind.bot.keyboards import MODE_AUTO, MODE_LINKS, mode_keyboard

router = Router(name="mode")

MODE_NAMES = {
    MODE_AUTO: "🔍 Авто-поиск",
    MODE_LINKS: "🔗 Свои ссылки",
}

MODE_DESCRIPTIONS = {
    MODE_AUTO: (
        "🔍 Режим <b>Авто-поиск</b> активирован!\n\n"
        "Просто напиши название товара — я сам найду обзоры "
        "на YouTube, Reddit и экспертных сайтах и сформирую анализ."
    ),
    MODE_LINKS: (
        "🔗 Режим <b>Свои ссылки</b> активирован!\n\n"
        "Отправь мне ссылки на обзоры (YouTube, Reddit или веб-страницы), "
        "и я проанализирую их содержимое."
    ),
}


@router.callback_query(F.data.in_({MODE_AUTO, MODE_LINKS}))
async def on_mode_selected(callback: CallbackQuery) -> None:
    """Handle inline button press for mode selection."""
    mode = callback.data
    description = MODE_DESCRIPTIONS.get(mode, "Режим выбран.")

    await callback.message.edit_text(description, parse_mode="HTML")  # type: ignore[union-attr]
    await callback.answer(f"Выбран режим: {MODE_NAMES.get(mode, mode)}")


@router.message(Command("mode"))
async def cmd_mode(message: Message) -> None:
    """Handle /mode command — show mode selection keyboard."""
    await message.answer(
        "Выбери режим работы:",
        reply_markup=mode_keyboard(),
        parse_mode="HTML",
    )
