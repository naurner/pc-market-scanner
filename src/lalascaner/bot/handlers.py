from __future__ import annotations

import asyncio
import re
from html import escape

from aiogram import BaseMiddleware, F, Router
from aiogram.filters import Command, CommandObject, CommandStart
from aiogram.types import (
    CallbackQuery,
    FSInputFile,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
)
from sqlalchemy.ext.asyncio import async_sessionmaker

from lalascaner.ai.normalize import pretty_key
from lalascaner.ai.schema import CATEGORY_NAMES_RU
from lalascaner.bot.backup import NotSqliteError, make_backup, split_file
from lalascaner.bot.browse import BTN_DB, BTN_SEARCH
from lalascaner.bot.formatting import listing_line, money
from lalascaner.bot.update import GitError, apply_update, restart_process
from lalascaner.config import Settings
from lalascaner.db import repo
from lalascaner.db.models import utcnow
from lalascaner.market.deals import DealParams
from lalascaner.pipeline import Pipeline

router = Router()

BTN_DEALS, BTN_WATCHES, BTN_STATS, BTN_UPDATE = "🔥 Сделки", "🔔 Подписки", "ℹ️ Статистика", "🔄 Обновить с GitHub"
BTN_SELL = "💰 За сколько продать"
MAIN_KEYBOARD = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text=BTN_SEARCH), KeyboardButton(text=BTN_DB)],
        [KeyboardButton(text=BTN_DEALS), KeyboardButton(text=BTN_WATCHES), KeyboardButton(text=BTN_STATS)],
        [KeyboardButton(text=BTN_SELL), KeyboardButton(text=BTN_UPDATE)],
    ],
    resize_keyboard=True,
)

HELP = """<b>Lalascaner</b> — рынок ПК-железа Кыргызстана ({sources}).

<b>🔍 Поиск</b> — просто напишите модель, можно с фильтрами:
<code>3060 12gb</code> · <code>ryzen 5600 до 10к</code> · <code>ddr4 16gb б/у</code> · <code>ноутбук 4060 от 50к</code>
Покажу рыночную цену и объявления с листанием.

<b>📦 База</b> — категории → модели → цены и все предложения. /db gpu — сразу категория.
/export 3060 — выгрузить объявления в CSV

<b>🔥 Сделки</b> — объявления ниже рынка за 48 ч (/deals)

<b>💰 За сколько продать</b> — цена и как быстро уходит: /sell rtx 3060 12gb б/у

<b>🔔 Уведомления о новых объявлениях ниже рынка</b>
/watch all — все сделки
/watch 4070 до 60000 — по модели с потолком цены
/watch new 5700x3d — все новые объявления по модели
/watches — мои подписки · в карточке модели есть кнопка «Следить»

{sources_help}/stats — размер базы
/backup — прислать файл базы данных
/update — обновиться с GitHub и перезапуститься"""


def help_text(settings: Settings) -> str:
    if settings.telegram_enabled:
        sources_help = "<b>Источники</b>: /channels · /addchannel @name · /delchannel @name\n"
        return HELP.format(sources="lalafo + Telegram", sources_help=sources_help)
    return HELP.format(sources="lalafo", sources_help="")


class AccessMiddleware(BaseMiddleware):
    """Бот личный: доступ из BOT_ALLOWED_USERS или у первого, кто нажал /start."""

    def __init__(self, settings: Settings, sessionmaker: async_sessionmaker):
        self.settings = settings
        self.sessionmaker = sessionmaker

    async def __call__(self, handler, event, data):  # noqa: ANN001
        user = data.get("event_from_user")
        if user is None:
            return None
        allowed = self.settings.allowed_user_ids
        if allowed:
            ok = user.id in allowed
        else:
            async with self.sessionmaker() as session:
                owner = await repo.get_kv(session, "bot_owner")
                if owner is None and isinstance(event, Message) and (event.text or "").startswith("/start"):
                    await repo.set_kv(session, "bot_owner", user.id)
                    await session.commit()
                    ok = True
                else:
                    ok = owner == user.id
        if not ok:
            if isinstance(event, Message):
                await event.answer("⛔ Это приватный бот.")
            return None
        return await handler(event, data)


@router.message(CommandStart())
async def cmd_start(message: Message, sessionmaker: async_sessionmaker, settings: Settings) -> None:
    async with sessionmaker() as session:
        if not await repo.list_subscriptions(session, message.chat.id):
            await repo.add_subscription(session, message.chat.id, query="", mode="deals")
            await session.commit()
            note = "\n\n✅ Подписал на все сделки ниже рынка. Настроить: /watches"
        else:
            note = ""
    await message.answer(help_text(settings) + note, reply_markup=MAIN_KEYBOARD, disable_web_page_preview=True)


@router.message(Command("help"))
async def cmd_help(message: Message, settings: Settings) -> None:
    await message.answer(help_text(settings), reply_markup=MAIN_KEYBOARD, disable_web_page_preview=True)


@router.message(Command("deals"))
@router.message(F.text == BTN_DEALS)
async def cmd_deals(
    message: Message, sessionmaker: async_sessionmaker, settings: Settings, command: CommandObject | None = None
) -> None:
    category = (command.args or "").strip().lower() if command else ""
    active = DealParams.from_settings(settings).active_clause()
    async with sessionmaker() as session:
        deals = await repo.recent_deals(session, hours=48, limit=15, category=category or None, active=active)
    if not deals:
        await message.answer("За 48 часов не было объявлений дешевле самых дешёвых актуальных.")
        return
    lines = ["<b>Рекордно дешёвые за 48 ч</b> (ещё актуальны):"]
    for item in deals:
        warn = " ⚠️" if item.suspicious else ""
        if item.deal_kind in ("model", "variant") and item.ref_min_price:
            why = f"до этого самое дешёвое {money(item.ref_min_price)} из {item.ref_count}"
        else:
            why = f"медиана рынка {money(item.market_median)}"
        lines.append(listing_line(item) + f"\n   {why}{warn}")
    await message.answer("\n".join(lines), disable_web_page_preview=True)


_RE_MAX = re.compile(r"\s+до\s+(\d[\d\s]*)\s*(к|k)?\s*(сом)?\s*$", re.I)


def parse_watch_args(args: str) -> tuple[str, float | None, str]:
    text = args.strip()
    mode = "deals"
    if text.lower().startswith("new "):
        mode, text = "all", text[4:].strip()
    max_price = None
    if m := _RE_MAX.search(text):
        max_price = float(m.group(1).replace(" ", "")) * (1000 if m.group(2) else 1)
        text = text[: m.start()].strip()
    if text.lower() in ("all", "все"):
        text = ""
    return text, max_price, mode


@router.message(Command("watch"))
async def cmd_watch(message: Message, command: CommandObject, sessionmaker: async_sessionmaker) -> None:
    if not command.args:
        await message.answer("Использование: <code>/watch all</code>, <code>/watch 4070 до 60000</code>, <code>/watch new 5700x3d</code>")
        return
    query, max_price, mode = parse_watch_args(command.args)
    if mode == "all" and not query:
        await message.answer("Для режима new нужна модель, иначе будет спам всеми объявлениями.")
        return
    async with sessionmaker() as session:
        sub = await repo.add_subscription(session, message.chat.id, query=query, max_price_kgs=max_price, mode=mode)
        await session.commit()
    what = f"«{escape(query)}»" if query else "все модели"
    kind = "все новые объявления" if mode == "all" else "сделки ниже рынка"
    limit = f", до {money(max_price)}" if max_price else ""
    await message.answer(f"🔔 Подписка #{sub.id}: {kind}, {what}{limit}.")


@router.message(Command("watches"))
@router.message(F.text == BTN_WATCHES)
async def cmd_watches(message: Message, sessionmaker: async_sessionmaker) -> None:
    async with sessionmaker() as session:
        subs = await repo.list_subscriptions(session, message.chat.id)
    if not subs:
        await message.answer("Подписок нет. Добавить: <code>/watch all</code>")
        return
    lines, buttons = ["<b>Подписки:</b>"], []
    for sub in subs:
        if sub.query.startswith("key:"):
            what = escape(pretty_key(sub.query[4:]))
        else:
            what = escape(sub.query) if sub.query else "все модели"
        kind = "все новые" if sub.mode == "all" else "сделки"
        limit = f", до {money(sub.max_price_kgs)}" if sub.max_price_kgs else ""
        lines.append(f"#{sub.id} — {kind}: {what}{limit}")
        buttons.append([InlineKeyboardButton(text=f"✖ Удалить #{sub.id}", callback_data=f"unwatch:{sub.id}")])
    await message.answer("\n".join(lines), reply_markup=InlineKeyboardMarkup(inline_keyboard=buttons))


@router.callback_query(F.data.startswith("unwatch:"))
async def cb_unwatch(callback: CallbackQuery, sessionmaker: async_sessionmaker) -> None:
    sub_id = int(callback.data.split(":", 1)[1])
    async with sessionmaker() as session:
        ok = await repo.delete_subscription(session, sub_id, callback.message.chat.id)
        await session.commit()
    await callback.answer("Удалено" if ok else "Не найдено")
    if ok:
        await callback.message.answer(f"Подписка #{sub_id} удалена.")


@router.message(Command("channels"))
async def cmd_channels(message: Message, sessionmaker: async_sessionmaker, settings: Settings) -> None:
    if not settings.telegram_enabled:
        await message.answer("Telegram-источники сейчас выключены — бот работает только с lalafo.")
        return
    async with sessionmaker() as session:
        channels = await repo.list_channels(session)
    if not channels:
        await message.answer("Каналов нет. Добавить: <code>/addchannel @name</code>")
        return
    lines = ["<b>Telegram-источники:</b>"]
    for ch in channels:
        state = "✅" if ch.enabled else "⏸"
        kind = "группа" if ch.kind == "group" else "канал"
        lines.append(f"{state} @{escape(ch.username)} ({kind}) — {escape(ch.title or '')} · до #{ch.last_message_id}")
    await message.answer("\n".join(lines), disable_web_page_preview=True)


@router.message(Command("addchannel"))
async def cmd_addchannel(message: Message, command: CommandObject, pipeline: Pipeline, settings: Settings) -> None:
    if not settings.telegram_enabled:
        await message.answer("Telegram-источники сейчас выключены — бот работает только с lalafo.")
        return
    raw = (command.args or "").strip()
    username = raw.rsplit("/", 1)[-1].lstrip("@")
    if not re.fullmatch(r"[A-Za-z0-9_]{4,64}", username):
        await message.answer("Использование: <code>/addchannel @name</code> или ссылка t.me/name")
        return
    await message.answer(f"Проверяю @{escape(username)}…")
    info = await pipeline.add_channel(username)
    if info is None:
        await message.answer("Не получилось: канал приватный или не существует.")
        return
    kind = "группа" if info.kind == "group" else "канал"
    await message.answer(f"✅ Подключил @{escape(info.username)} ({kind}): {escape(info.title or '')}. История подтянется при следующем опросе.")


@router.message(Command("delchannel"))
async def cmd_delchannel(
    message: Message, command: CommandObject, sessionmaker: async_sessionmaker, settings: Settings
) -> None:
    if not settings.telegram_enabled:
        await message.answer("Telegram-источники сейчас выключены — бот работает только с lalafo.")
        return
    username = (command.args or "").strip().rsplit("/", 1)[-1].lstrip("@")
    async with sessionmaker() as session:
        ok = await repo.remove_channel(session, username)
        await session.commit()
    await message.answer("⏸ Отключён." if ok else "Такого канала нет.")


@router.message(Command("stats"))
@router.message(F.text == BTN_STATS)
async def cmd_stats(message: Message, sessionmaker: async_sessionmaker, settings: Settings) -> None:
    async with sessionmaker() as session:
        stats = await repo.db_stats(session)
        usage = await repo.get_kv(session, "ai_usage", {}) or {}
        batches = await repo.get_kv(session, "ai_batches", []) or []
    await message.answer(stats_text(settings, stats, usage, batches))


def stats_text(settings: Settings, stats: dict, usage: dict, batches: list) -> str:
    src = stats["listings_by_source"]
    status = stats["listings_by_status"]
    cats = sorted(stats["items_by_category"].items(), key=lambda kv: -kv[1])
    sources = f"lalafo {src.get('lalafo', 0)}"
    if settings.telegram_enabled:
        sources += f", Telegram {src.get('telegram', 0)}"
    queue = f"В очереди на разбор: {status.get('new', 0)} · ошибок: {status.get('failed', 0)}"
    if settings.ai_enabled:
        if status.get("refine"):
            queue += f" · ждут доработки ИИ: {status['refine']}"
        if settings.ai_provider == "claude":
            queue += f" · в Batch API: {status.get('batch', 0)}"
    lines = [
        "<b>База</b>",
        f"Объявлений: {sources}",
        queue,
        f"Позиций на продажу: {stats['items_total']} · моделей с ценой: {stats['models_tracked']}",
        f"Сделок за 24 ч: {stats['deals_24h']}",
        "",
        "<b>По категориям:</b> " + ", ".join(f"{CATEGORY_NAMES_RU.get(c, c)} {n}" for c, n in cats[:10]),
    ]
    events = stats.get("events_by_kind") or {}
    lines += [
        "",
        "<b>История</b>",
        f"Смен цены: {events.get('price', 0)} · текста: {events.get('text', 0)} · поднятий в ленте: {events.get('bump', 0)}",
        f"Пропало с lalafo (продано/снято): {stats.get('gone', 0)} · вернулось: {events.get('back', 0)}",
        f"Снимков интереса (просмотры, избранное, звонки): {stats.get('metrics', 0)} · дней истории цен: {stats.get('history_days', 0)}",
    ]
    if settings.effective_embed:
        lines.append(f"Векторов объявлений: {stats.get('vectors', 0)}")
    if not settings.ai_enabled:
        lines += ["", "<b>ИИ</b>: выключен, разбор на правилах"]
        return "\n".join(lines)
    provider = {"claude": "Claude", "ollama": "Ollama", "openai": "OpenAI-совместимый"}[settings.ai_provider]
    lines += ["", f"<b>ИИ</b>: {provider}, режим {settings.effective_classifier}, модель {escape(settings.ai_model)}"]
    if usage:
        lines.append(
            f"Запросов {usage.get('requests', 0)} · вход {usage.get('input_tokens', 0):,} · выход {usage.get('output_tokens', 0):,} "
            f"· из кэша {usage.get('cache_read_input_tokens', 0):,} токенов".replace(",", " ")
        )
    if batches:
        lines.append(f"Batch API в обработке: {len(batches)}")
    return "\n".join(lines)


@router.message(Command("backup"))
async def cmd_backup(message: Message, settings: Settings) -> None:
    try:
        path = await asyncio.to_thread(make_backup, settings.database_url)
    except NotSqliteError:
        await message.answer("База не SQLite — выгрузка файлом не поддерживается.")
        return
    parts = [path]
    try:
        parts = await asyncio.to_thread(split_file, path)
        name = f"lalascaner-{utcnow():%Y%m%d-%H%M}.db"
        if len(parts) == 1:
            await message.answer_document(FSInputFile(path, filename=name), caption="📦 Снимок базы данных.")
            return
        for i, part in enumerate(parts, 1):
            suffix = part.name.rsplit(".", 1)[1]
            await message.answer_document(
                FSInputFile(part, filename=f"{name}.{suffix}"), caption=f"📦 База, часть {i}/{len(parts)}"
            )
        await message.answer(
            f"База в {len(parts)} частях по 25 МБ. Сохраните их в одну папку и склейте:\n"
            f"Mac/Linux: <code>cat {name}.part* &gt; {name}</code>\n"
            f"Windows: <code>copy /b {name}.part* {name}</code>"
        )
    finally:
        for p in {path, *parts}:
            p.unlink(missing_ok=True)


@router.message(Command("update"))
@router.message(F.text == BTN_UPDATE)
async def cmd_update(message: Message, settings: Settings) -> None:
    status = await message.answer("🔄 Проверяю обновления на GitHub…")
    try:
        result = await apply_update(settings.bot_repo_path)
    except GitError as exc:
        await status.edit_text(
            f"❌ Не удалось обновиться: {escape(str(exc))}\n"
            "Обновите вручную на сервере: <code>git pull</code>"
        )
        return
    if not result.updated:
        await status.edit_text(f"✅ Уже последняя версия (<code>{escape(result.head)}</code>).")
        return
    log = escape("\n".join(result.commits[:15]))
    if result.needs_rebuild:
        await status.edit_text(
            f"⬇️ Код обновлён до <code>{escape(result.head)}</code>:\n<pre>{log}</pre>\n\n"
            "⚠️ Изменились зависимости (pyproject.toml/Dockerfile) — одной кнопкой это не поставить. "
            "На сервере выполните: <code>docker compose up -d --build</code>"
        )
        return
    await status.edit_text(f"⬇️ Обновлено до <code>{escape(result.head)}</code>:\n<pre>{log}</pre>\n\n🔄 Перезапускаюсь…")
    await asyncio.sleep(0.5)
    restart_process()
