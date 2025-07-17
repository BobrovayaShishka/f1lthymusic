from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
from music_generator import generate_music
from config import BOT_TOKEN
import logging
import os

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Словари для разных падежей
MOOD_NOMINATIVE = {
    "sad": "Грустная",
    "calm": "Спокойная",
    "happy": "Радостная",
    "energetic": "Энергичная"
}

MOOD_GENITIVE = {
    "sad": "грустной",
    "calm": "спокойной",
    "happy": "радостной",
    "energetic": "энергичной"
}

MOOD_KEYBOARD = [
    [InlineKeyboardButton("😢 Грустное", callback_data="sad")],
    [InlineKeyboardButton("😌 Спокойное", callback_data="calm")],
    [InlineKeyboardButton("😄 Радостное", callback_data="happy")],
    [InlineKeyboardButton("⚡ Энергичное", callback_data="energetic")]
]

# Клавиатура для повторной генерации
GENERATE_AGAIN_KEYBOARD = InlineKeyboardMarkup([
    [InlineKeyboardButton("🔄 Сгенерировать ещё", callback_data="generate_another")]
])


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /start"""
    welcome_text = (
        "🎹 *Привет, я f1lthy!*\n\n"
        "Я - твой персональный ИИ-композитор. С помощью нейросетей я создаю уникальные мелодии, "
        "основанные на выбранном тобой настроении.\n\n"
        "✨ Просто выбери эмоциональную окраску будущей мелодии:"
    )

    await update.message.reply_text(
        welcome_text,
        parse_mode="Markdown",
        reply_markup=InlineKeyboardMarkup(MOOD_KEYBOARD)
    )

async def handle_mood(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик выбора настроения"""
    query = update.callback_query
    await query.answer()
    mood = query.data

    # Получаем названия в разных падежах
    mood_name_nom = MOOD_NOMINATIVE.get(mood, "Музыкальная")
    mood_name_gen = MOOD_GENITIVE.get(mood, "музыкальной")

    # Сообщение о начале генерации с правильным склонением
    status_msg = await query.edit_message_text(f"🎹 Генерация {mood_name_gen} мелодии...")

    try:
        # Запускаем генерацию
        audio_path, midi_path = generate_music(mood)

        if not midi_path:
            await status_msg.edit_text("⚠️ Не удалось сгенерировать музыку")
            return

        # Отправка результата
        title = f"{mood_name_nom} мелодия"
        caption = f"🎵 {mood_name_nom} мелодия"

        # Если есть аудио, отправляем его с кнопкой "Сгенерировать ещё"
        if audio_path and os.path.exists(audio_path):
            with open(audio_path, 'rb') as audio_file:
                await context.bot.send_audio(
                    chat_id=query.message.chat_id,
                    audio=audio_file,
                    title=title,
                    caption=caption,
                    reply_markup=GENERATE_AGAIN_KEYBOARD
                )
        else:
            # Если аудио нет, отправляем MIDI с кнопкой "Сгенерировать ещё"
            with open(midi_path, 'rb') as midi_file:
                await context.bot.send_document(
                    chat_id=query.message.chat_id,
                    document=midi_file,
                    filename=f"{mood_name_nom}_мелодия.mid",
                    caption=caption,
                    reply_markup=GENERATE_AGAIN_KEYBOARD
                )

        # Редактируем сообщение о статусе
        await status_msg.edit_text(f"✅ {mood_name_nom} мелодия успешно сгенерирована!")

    except Exception as e:
        logger.error(f"Ошибка генерации: {str(e)}")
        await status_msg.edit_text(f"⚠️ Ошибка генерации: {str(e)}")


async def download_midi(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик скачивания MIDI"""
    query = update.callback_query
    await query.answer()
    midi_path = query.data[3:]

    if os.path.exists(midi_path):
        # Извлекаем название настроения из пути к файлу
        mood = os.path.basename(midi_path).split("_")[0]
        mood_name_nom = MOOD_NOMINATIVE.get(mood, "Музыкальная")

        with open(midi_path, 'rb') as midi_file:
            await context.bot.send_document(
                chat_id=query.message.chat_id,
                document=midi_file,
                filename=f"{mood_name_nom}_мелодия.mid",
                caption=f"🎵 MIDI-файл: {mood_name_nom} мелодия"
            )
    else:
        await query.message.reply_text("⚠️ Файл MIDI не найден")


async def generate_another(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик кнопки 'Сгенерировать ещё'"""
    query = update.callback_query
    await query.answer()

    # Удаляем текущую клавиатуру у сообщения с файлом
    await query.edit_message_reply_markup(reply_markup=None)

    # Отправляем новое сообщение с выбором настроения
    await query.message.reply_text(
        "🎵 Выберите настроение музыки:",
        reply_markup=InlineKeyboardMarkup(MOOD_KEYBOARD)
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.callback_query:
        query = update.callback_query
        await query.answer()
        message = query.message
    else:
        message = update.message

    help_text = (
        "🎼 *Как общаться с f1lthy?*\n\n"
        "1. Выбери настроение для мелодии из предложенных вариантов\n"
        "2. Подожди несколько секунд, пока нейросеть создаст уникальную композицию\n"
        "3. Прослушай результат и при необходимости скачай MIDI-файл\n"
        "4. Хочешь другую мелодию? Просто нажми 'Сгенерировать ещё'!\n\n"
        "💡 *Технологии:*\n"
        "- Генерация музыки с помощью нейросетей PyTorch\n"
        "- Контроль настроения и тональности\n"
        "- Профессиональное звучание через SoundFont\n\n"
    )

    keyboard = InlineKeyboardMarkup([
        [InlineKeyboardButton("🎹 Начать генерацию", callback_data="start_generation")]
    ])

    if update.callback_query:
        await query.edit_message_text(
            help_text,
            parse_mode="Markdown",
            reply_markup=keyboard
        )
    else:
        await message.reply_text(
            help_text,
            parse_mode="Markdown",
            reply_markup=keyboard
        )


# Новый обработчик для кнопки "Начать генерацию"
async def start_generation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    # Отправляем сообщение с выбором настроения
    await query.edit_message_text(
        "🎵 Выберите настроение музыки:",
        reply_markup=InlineKeyboardMarkup(MOOD_KEYBOARD)
    )


def main() -> None:
    application = Application.builder().token(BOT_TOKEN).build()

    # Добавляем обработчики команд
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))

    # Добавляем обработчики callback-запросов
    application.add_handler(CallbackQueryHandler(handle_mood, pattern="^(sad|calm|happy|energetic)$"))
    application.add_handler(CallbackQueryHandler(download_midi, pattern="^dl_"))
    application.add_handler(CallbackQueryHandler(generate_another, pattern="^generate_another$"))
    application.add_handler(CallbackQueryHandler(start_generation, pattern="^start_generation$"))

    logger.info("Бот запущен...")

    # Запускаем бота в режиме polling
    application.run_polling()


if __name__ == "__main__":
    main()  
