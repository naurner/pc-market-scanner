from __future__ import annotations

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.types import BotCommand
from sqlalchemy.ext.asyncio import async_sessionmaker

from lalascaner.bot import browse, handlers, insights
from lalascaner.bot.notifier import Notifier
from lalascaner.config import Settings
from lalascaner.pipeline import Pipeline

COMMANDS = [
    BotCommand(command="search", description="Поиск объявлений: /search 3060 до 25к"),
    BotCommand(command="price", description="Рыночная цена модели"),
    BotCommand(command="db", description="База: категории и модели"),
    BotCommand(command="deals", description="Объявления ниже рынка"),
    BotCommand(command="sell", description="За сколько продать и как быстро уйдёт"),
    BotCommand(command="watch", description="Уведомления о сделках"),
    BotCommand(command="watches", description="Мои подписки"),
    BotCommand(command="export", description="Выгрузка объявлений в CSV"),
    BotCommand(command="channels", description="Telegram-источники"),
    BotCommand(command="stats", description="Статистика базы"),
    BotCommand(command="backup", description="Прислать файл базы данных"),
    BotCommand(command="update", description="Обновиться с GitHub и перезапуститься"),
    BotCommand(command="help", description="Помощь"),
]


def build_bot(settings: Settings, sessionmaker: async_sessionmaker, pipeline: Pipeline) -> tuple[Bot, Dispatcher]:
    bot = Bot(settings.bot_token, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    dp = Dispatcher(settings=settings, sessionmaker=sessionmaker, pipeline=pipeline)
    access = handlers.AccessMiddleware(settings, sessionmaker)
    dp.message.outer_middleware(access)
    dp.callback_query.outer_middleware(access)
    # Порядок важен: в browse последним стоит перехват любого текста как поискового запроса.
    dp.include_router(handlers.router)
    dp.include_router(insights.router)
    dp.include_router(browse.router)
    pipeline.on_items = Notifier(bot, sessionmaker, settings)
    return bot, dp


def commands_for(settings: Settings) -> list[BotCommand]:
    if settings.telegram_enabled:
        return COMMANDS
    return [c for c in COMMANDS if c.command != "channels"]


async def start_polling(bot: Bot, dp: Dispatcher) -> None:
    await bot.set_my_commands(commands_for(dp["settings"]))
    await dp.start_polling(bot, handle_signals=False)
