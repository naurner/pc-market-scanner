"""База в боте: категории → модели → карточка модели, поиск с фильтрами и листанием, выгрузка CSV."""

from __future__ import annotations

import csv
import io
import secrets
from collections import OrderedDict
from html import escape

from aiogram import F, Router
from aiogram.exceptions import TelegramBadRequest
from aiogram.filters import Command, CommandObject
from aiogram.types import BufferedInputFile, CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message
from sqlalchemy.ext.asyncio import async_sessionmaker

from lalascaner.ai.schema import CATEGORY_NAMES_RU
from lalascaner.ai.normalize import pretty_key
from lalascaner.ai.variants import pretty_variant
from lalascaner.bot.formatting import listing_line, money, short_money, stat_block
from lalascaner.config import Settings
from lalascaner.db import repo
from lalascaner.market.deals import DealParams
from lalascaner.market.query import parse_search
from lalascaner.pipeline import Pipeline

router = Router()

BTN_DB, BTN_SEARCH = "📦 База", "🔍 Поиск"
BROWSE_CATEGORIES = ("gpu", "cpu", "ram", "ssd", "hdd", "motherboard", "psu", "cooler", "case", "monitor", "laptop", "pc", "peripheral")
MODELS_PER_PAGE = 10
ITEMS_PER_PAGE = 8

# Запросы поиска для кнопок листания (в callback_data помещается только короткий токен).
_SEARCHES: OrderedDict[str, str] = OrderedDict()


def remember_search(text: str) -> str:
    token = secrets.token_hex(4)
    _SEARCHES[token] = text
    while len(_SEARCHES) > 500:
        _SEARCHES.popitem(last=False)
    return token


def _btn(text: str, data: str) -> InlineKeyboardButton:
    return InlineKeyboardButton(text=text, callback_data=data)


def _pager(prefix: str, page: int, total: int, per_page: int) -> list[InlineKeyboardButton]:
    pages = max(1, (total + per_page - 1) // per_page)
    row = []
    if page > 0:
        row.append(_btn("◀", f"{prefix}:{page - 1}"))
    row.append(_btn(f"{page + 1}/{pages}", "noop"))
    if page + 1 < pages:
        row.append(_btn("▶", f"{prefix}:{page + 1}"))
    return row


async def _show(target: Message | CallbackQuery, text: str, markup: InlineKeyboardMarkup | None = None) -> None:
    if isinstance(target, CallbackQuery):
        try:
            await target.message.edit_text(text, reply_markup=markup, disable_web_page_preview=True)
        except TelegramBadRequest:  # «message is not modified» и т.п.
            pass
        await target.answer()
    else:
        await target.answer(text, reply_markup=markup, disable_web_page_preview=True)


# ── категории и модели ───────────────────────────────────────────────────────


def categories_markup() -> InlineKeyboardMarkup:
    buttons = [_btn(CATEGORY_NAMES_RU[c], f"cat:{c}:0") for c in BROWSE_CATEGORIES]
    rows = [buttons[i : i + 2] for i in range(0, len(buttons), 2)]
    rows.append([_btn("📄 Выгрузить весь рынок в CSV", "exp:all")])
    return InlineKeyboardMarkup(inline_keyboard=rows)


@router.message(Command("db", "market"))
@router.message(F.text == BTN_DB)
async def cmd_db(message: Message, sessionmaker: async_sessionmaker, command: CommandObject | None = None) -> None:
    category = (command.args or "").strip().lower() if command else ""
    if category in CATEGORY_NAMES_RU:
        await show_category(message, sessionmaker, category, 0)
        return
    async with sessionmaker() as session:
        stats = await repo.db_stats(session)
    counts = stats["items_by_category"]
    lines = [
        "<b>📦 База рынка</b>",
        f"Позиций на продажу: {stats['items_total']} · моделей с ценой: {stats['models_tracked']}",
        "",
        "Выберите категорию:",
    ]
    lines += [f"• {CATEGORY_NAMES_RU[c]}: {counts[c]}" for c in BROWSE_CATEGORIES if counts.get(c)]
    await message.answer("\n".join(lines), reply_markup=categories_markup())


@router.callback_query(F.data == "db")
async def cb_db(callback: CallbackQuery) -> None:
    await _show(callback, "<b>📦 База рынка</b>\nВыберите категорию:", categories_markup())


async def show_category(target: Message | CallbackQuery, sessionmaker: async_sessionmaker, category: str, page: int) -> None:
    async with sessionmaker() as session:
        stats, total = await repo.models_in_category(
            session, category, offset=page * MODELS_PER_PAGE, limit=MODELS_PER_PAGE
        )
    name = CATEGORY_NAMES_RU.get(category, category)
    if not stats:
        markup = InlineKeyboardMarkup(inline_keyboard=[[_btn("⬅ Категории", "db")]])
        await _show(target, f"<b>{escape(name)}</b>\nПока нет моделей с достаточным числом предложений.", markup)
        return
    text = f"<b>{escape(name)}</b> — {total} моделей\nМедиана цены · (кол-во предложений). Нажмите модель:"
    rows = [
        [_btn(f"{s.title} · {short_money(s.median)} ({s.n})"[:60], f"m:{repo.key_token(s.key)}:0")] for s in stats
    ]
    rows.append(_pager(f"cat:{category}", page, total, MODELS_PER_PAGE))
    rows.append([_btn("📄 CSV категории", f"exp:{category}"), _btn("⬅ Категории", "db")])
    await _show(target, text, InlineKeyboardMarkup(inline_keyboard=rows))


@router.callback_query(F.data.startswith("cat:"))
async def cb_category(callback: CallbackQuery, sessionmaker: async_sessionmaker) -> None:
    _, category, page = callback.data.split(":")
    await show_category(callback, sessionmaker, category, int(page))


async def model_card(
    sessionmaker: async_sessionmaker, key: str, page: int, params: DealParams
) -> tuple[str, list[list[InlineKeyboardButton]]]:
    active = params.active_clause()
    async with sessionmaker() as session:
        stats = await repo.stats_by_key(session, key)
        floor = stats["all"].median * params.suspicious_ratio if "all" in stats else None
        items, total = await repo.items_for_key(
            session, key, active, offset=page * ITEMS_PER_PAGE, limit=ITEMS_PER_PAGE, min_price=floor
        )
        variants = await repo.active_variants(session, key, active, min_price=floor) if page == 0 else []
    parts = [stat_block(stats) or f"<b>{escape(pretty_key(key))}</b>"]
    if page == 0 and items:
        parts.append(f"🟢 Сейчас актуально: <b>{total}</b> · самое дешёвое <b>{money(items[0].price_kgs)}</b>")
    if len(variants) > 1 or (variants and variants[0][0]):
        lines = ["<b>Самое дешёвое по версиям:</b>"]
        for variant_key, count, min_price in variants:
            name = pretty_variant(variant_key).split(" · ", 1)[-1] if variant_key else "версия не указана"
            lines.append(f"• {escape(name)} — от {money(min_price)} ({count})")
        parts.append("\n".join(lines))
    if items:
        start = page * ITEMS_PER_PAGE
        parts.append(f"<b>Актуальные предложения от дешёвых</b> ({start + 1}–{start + len(items)} из {total}):")
        parts.append("\n".join(listing_line(item) for item in items))
    else:
        parts.append("Актуальных предложений с ценой сейчас нет.")
    token = repo.key_token(key)
    rows = []
    if total > ITEMS_PER_PAGE:
        rows.append(_pager(f"m:{token}", page, total, ITEMS_PER_PAGE))
    category = stats["all"].category if "all" in stats else key.split(":")[0]
    rows.append([_btn("🔔 Следить за сделками", f"w:{token}"), _btn("⬅ Модели", f"cat:{category}:0")])
    return "\n\n".join(parts), rows


@router.callback_query(F.data.startswith("m:"))
async def cb_model(callback: CallbackQuery, sessionmaker: async_sessionmaker, settings: Settings) -> None:
    _, token, page = callback.data.split(":")
    async with sessionmaker() as session:
        key = await repo.key_by_token(session, token)
    if key is None:
        await callback.answer("Модель не найдена — статистика обновилась, откройте заново", show_alert=True)
        return
    text, rows = await model_card(sessionmaker, key, int(page), DealParams.from_settings(settings))
    await _show(callback, text, InlineKeyboardMarkup(inline_keyboard=rows))


@router.callback_query(F.data.startswith("w:"))
async def cb_watch_model(callback: CallbackQuery, sessionmaker: async_sessionmaker) -> None:
    token = callback.data.split(":", 1)[1]
    async with sessionmaker() as session:
        key = await repo.key_by_token(session, token)
        if key is None:
            await callback.answer("Модель не найдена", show_alert=True)
            return
        existing = [s for s in await repo.list_subscriptions(session, callback.message.chat.id) if s.query == f"key:{key}"]
        if not existing:
            await repo.add_subscription(session, callback.message.chat.id, query=f"key:{key}", mode="deals")
            await session.commit()
    await callback.answer("🔔 Буду присылать сделки по этой модели" if not existing else "Уже слежу", show_alert=True)


@router.callback_query(F.data == "noop")
async def cb_noop(callback: CallbackQuery) -> None:
    await callback.answer()


# ── поиск ────────────────────────────────────────────────────────────────────

SEARCH_HELP = (
    "Напишите модель, можно с фильтрами:\n"
    "<code>3060 12gb</code> · <code>ryzen 5600 до 10к</code> · <code>ddr4 16gb б/у</code> · "
    "<code>ноутбук 4060 от 50к до 90к</code>"
)


async def search_view(
    sessionmaker: async_sessionmaker, text: str, page: int, order: str, params: DealParams, pipeline: Pipeline | None = None
) -> tuple[str, InlineKeyboardMarkup | None]:
    qf, flt = parse_search(text)
    if qf.empty:
        return SEARCH_HELP, None
    active = params.active_clause()
    async with sessionmaker() as session:
        total = await repo.count_items(session, qf, flt=flt, order=order, active=active)
        items = await repo.search_items(
            session, qf, limit=ITEMS_PER_PAGE, order=order, offset=page * ITEMS_PER_PAGE, flt=flt, active=active
        )
        stats = await repo.stats_for(session, qf, limit=1) if page == 0 else []
        blocks = [stat_block(await repo.stats_by_key(session, s.key)) for s in stats]
    header = f"🔍 <b>{escape(text)}</b>"
    if flt.active:
        header += f" <i>({escape(flt.describe())})</i>"
    parts = [header]
    parts += [b for b in blocks if b]
    if not items:
        similar = await pipeline.semantic_search(text) if pipeline is not None and page == 0 else None
        if similar:
            parts.append("Точных совпадений нет. <b>Похожие по смыслу</b>:")
            parts.append("\n".join(listing_line(item) for item in similar))
        else:
            parts.append("Актуальных объявлений не найдено.")
        return "\n\n".join(parts), None
    start = page * ITEMS_PER_PAGE
    order_name = "от дешёвых" if order == "cheap" else "сначала новые"
    parts.append(f"<b>Актуальные объявления</b> {order_name} ({start + 1}–{start + len(items)} из {total}):")
    parts.append("\n".join(listing_line(item) for item in items))
    token = remember_search(text)
    other = "recent" if order == "cheap" else "cheap"
    rows = []
    if total > ITEMS_PER_PAGE:
        rows.append(_pager(f"s:{token}:{order[0]}", page, total, ITEMS_PER_PAGE))
    toggle = "🕒 Сначала новые" if order == "cheap" else "💰 От дешёвых"
    rows.append([_btn(toggle, f"s:{token}:{other[0]}:0"), _btn("📄 CSV", f"xs:{token}")])
    if stats:
        rows.append([_btn(f"🔔 Следить: {stats[0].title}"[:60], f"w:{repo.key_token(stats[0].key)}")])
    return "\n\n".join(parts), InlineKeyboardMarkup(inline_keyboard=rows)


async def run_search(
    message: Message, text: str, sessionmaker: async_sessionmaker, settings: Settings, pipeline: Pipeline, order: str = "cheap"
) -> None:
    body, markup = await search_view(sessionmaker, text, 0, order, DealParams.from_settings(settings), pipeline)
    await message.answer(body, reply_markup=markup, disable_web_page_preview=True)


@router.message(Command("search", "price"))
async def cmd_search(
    message: Message, command: CommandObject, sessionmaker: async_sessionmaker, settings: Settings, pipeline: Pipeline
) -> None:
    order = "recent" if command.command == "search" else "cheap"
    await run_search(message, command.args or "", sessionmaker, settings, pipeline, order)


@router.message(F.text == BTN_SEARCH)
async def btn_search(message: Message) -> None:
    await message.answer(SEARCH_HELP)


@router.callback_query(F.data.startswith("s:"))
async def cb_search(callback: CallbackQuery, sessionmaker: async_sessionmaker, settings: Settings) -> None:
    _, token, order_code, page = callback.data.split(":")
    text = _SEARCHES.get(token)
    if text is None:
        await callback.answer("Поиск устарел — отправьте запрос ещё раз", show_alert=True)
        return
    order = "cheap" if order_code == "c" else "recent"
    body, markup = await search_view(sessionmaker, text, int(page), order, DealParams.from_settings(settings))
    await _show(callback, body, markup)


# ── выгрузка CSV ─────────────────────────────────────────────────────────────


def _csv_file(rows: list[list], header: list[str], name: str) -> BufferedInputFile:
    buf = io.StringIO()
    writer = csv.writer(buf, delimiter=";")
    writer.writerow(header)
    writer.writerows(rows)
    return BufferedInputFile(buf.getvalue().encode("utf-8-sig"), filename=name)


@router.callback_query(F.data.startswith("exp:"))
async def cb_export_market(callback: CallbackQuery, sessionmaker: async_sessionmaker) -> None:
    category = callback.data.split(":", 1)[1]
    async with sessionmaker() as session:
        stats = await repo.market_rows(session, None if category == "all" else category)
    rows = [
        [CATEGORY_NAMES_RU.get(s.category, s.category), s.title, s.key, s.condition, s.n, round(s.median), round(s.p25), round(s.p75), round(s.min), round(s.max)]
        for s in stats
    ]
    header = ["категория", "модель", "ключ", "состояние", "предложений", "медиана", "p25", "p75", "мин", "макс"]
    await callback.message.answer_document(
        _csv_file(rows, header, f"market_{category}.csv"), caption=f"Рынок: {len(rows)} строк"
    )
    await callback.answer()


async def _export_listings(message: Message, text: str, sessionmaker: async_sessionmaker, params: DealParams) -> None:
    qf, flt = parse_search(text)
    if qf.empty:
        await message.answer("Использование: <code>/export 3060</code> или кнопка «📄 Выгрузить» в разделе База")
        return
    async with sessionmaker() as session:
        items = await repo.search_items(session, qf, limit=5000, order="cheap", flt=flt, active=params.active_clause())
    rows = [
        [
            round(i.price_kgs) if i.price_kgs else "", i.condition, i.model, pretty_key(i.canonical_key),
            pretty_variant(i.variant_key).split(" · ", 1)[-1] if i.variant_key else "",
            i.listing.source, i.listing.city or "", i.listing.changed_at.strftime("%Y-%m-%d %H:%M"), i.listing.url,
        ]
        for i in items
    ]
    header = ["цена_сом", "состояние", "название", "модель", "версия", "источник", "город", "дата", "ссылка"]
    await message.answer_document(_csv_file(rows, header, "listings.csv"), caption=f"«{escape(text)}»: {len(rows)} объявлений")


@router.message(Command("export"))
async def cmd_export(message: Message, command: CommandObject, sessionmaker: async_sessionmaker, settings: Settings) -> None:
    await _export_listings(message, command.args or "", sessionmaker, DealParams.from_settings(settings))


@router.callback_query(F.data.startswith("xs:"))
async def cb_export_search(callback: CallbackQuery, sessionmaker: async_sessionmaker, settings: Settings) -> None:
    text = _SEARCHES.get(callback.data.split(":", 1)[1])
    if text is None:
        await callback.answer("Поиск устарел — отправьте запрос ещё раз", show_alert=True)
        return
    await callback.answer()
    await _export_listings(callback.message, text, sessionmaker, DealParams.from_settings(settings))


# Любой текст без команды — поиск.
@router.message(F.text & ~F.text.startswith("/"))
async def free_text(message: Message, sessionmaker: async_sessionmaker, settings: Settings, pipeline: Pipeline) -> None:
    await run_search(message, message.text, sessionmaker, settings, pipeline)
