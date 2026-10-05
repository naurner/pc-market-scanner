"""Рассылка уведомлений по подпискам."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import timedelta

from aiogram import Bot
from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError, TelegramRetryAfter
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from sqlalchemy.ext.asyncio import async_sessionmaker

from lalascaner.bot.formatting import deal_message
from lalascaner.config import Settings
from lalascaner.db import repo
from lalascaner.db.models import Item, Subscription, utcnow
from lalascaner.market.query import parse_query

log = logging.getLogger(__name__)


def item_matches_query(item: Item, query: str) -> bool:
    if not query:
        return True
    qf = parse_query(query)
    key = item.canonical_key
    if qf.key_prefixes:
        ok = any(key == p or key.startswith(p + ":") for p in qf.key_prefixes)
    else:
        haystack = f"{key} {item.model}".lower()
        ok = all(tok in haystack for tok in qf.tokens)
    if ok and qf.category and not qf.key_prefixes:
        ok = item.category == qf.category
    return ok


def subscription_matches(sub: Subscription, item: Item, default_threshold: float) -> bool:
    if item.intent != "sell":
        return False
    if sub.mode == "deals":
        threshold = sub.min_discount if sub.min_discount is not None else default_threshold
        if not item.is_deal or item.discount is None or item.discount < threshold:
            return False
    if sub.max_price_kgs is not None and (item.price_kgs is None or item.price_kgs > sub.max_price_kgs):
        return False
    return item_matches_query(item, sub.query)


# Уже отправленное объявление присылаем снова, только если подешевело хотя бы на столько.
REPEAT_DROP = 0.03
# Тот же продавец + модель + цена в пределах 3% под другим номером в течение месяца — перезалив, не новость.
REPOST_TOLERANCE = 0.03
REPOST_WINDOW = timedelta(days=30)
# Правка, сделанная позже первого появления на столько, — это правка, а не новое объявление.
EDIT_GRACE = timedelta(minutes=10)


@dataclass
class Decision:
    send: bool
    previous_price: float | None = None  # «было», если присылаем из-за снижения цены


async def alert_decision(session, chat_id: int, item: Item) -> Decision:  # noqa: ANN001
    listing = item.listing
    last = await repo.last_alert(session, chat_id, listing.id, item.canonical_key)
    if last is not None:
        dropped = item.price_kgs is not None and last.price_kgs and item.price_kgs <= last.price_kgs * (1 - REPEAT_DROP)
        return Decision(True, last.price_kgs) if dropped else Decision(False)
    if listing.dup_of and await repo.alerted_listing(session, chat_id, listing.dup_of):
        return Decision(False)
    if await repo.alerted_same_offer(session, chat_id, item, utcnow() - REPOST_WINDOW, REPOST_TOLERANCE):
        return Decision(False)
    if listing.changed_at > listing.fetched_at + EDIT_GRACE:
        # Давно висящее объявление отредактировали: новость, только если снизили цену.
        drop = await repo.price_drop_event(session, listing.id, listing.changed_at - EDIT_GRACE)
        if drop is None:
            return Decision(False)
        same_currency = drop.currency in (None, "KGS")
        return Decision(True, drop.old_price if same_currency else None)
    return Decision(True)


class Notifier:
    # Защита от лавины при первом запуске: не больше стольких сообщений в чат за одну рассылку.
    MAX_PER_RUN = 25

    def __init__(self, bot: Bot, sessionmaker: async_sessionmaker, settings: Settings):
        self.bot = bot
        self.sessionmaker = sessionmaker
        self.settings = settings

    async def __call__(self, item_ids: list[int]) -> None:
        async with self.sessionmaker() as session:
            subs = await repo.list_subscriptions(session)
            if not subs:
                return
            items = await repo.get_items(session, item_ids)
            items.sort(key=lambda i: -(i.discount or 0))
            sent: dict[int, int] = {}
            for item in items:
                chats = {sub.chat_id for sub in subs if subscription_matches(sub, item, self.settings.deal_threshold)}
                for chat_id in chats:
                    if sent.get(chat_id, 0) >= self.MAX_PER_RUN:
                        continue
                    decision = await alert_decision(session, chat_id, item)
                    if not decision.send:
                        continue
                    if await self._send(chat_id, item, decision.previous_price):
                        sent[chat_id] = sent.get(chat_id, 0) + 1
                        await repo.record_alert(session, chat_id, item)
                        await session.commit()

    async def _send(self, chat_id: int, item: Item, previous_price: float | None = None) -> bool:
        row = [InlineKeyboardButton(text="Открыть объявление", url=item.listing.url)]
        if self.settings.effective_embed:
            row.append(InlineKeyboardButton(text="🔎 Похожие", callback_data=f"sim:{item.listing_id}"))
        markup = InlineKeyboardMarkup(inline_keyboard=[row])
        for _ in range(3):
            try:
                text = deal_message(item, previous_price)
                await self.bot.send_message(chat_id, text, reply_markup=markup, disable_web_page_preview=True)
                await asyncio.sleep(0.05)
                return True
            except TelegramRetryAfter as exc:
                await asyncio.sleep(exc.retry_after + 1)
            except TelegramForbiddenError:
                log.warning("бот заблокирован в чате %s", chat_id)
                return False
            except TelegramBadRequest as exc:
                log.warning("не удалось отправить уведомление в %s: %s", chat_id, exc)
                return False
        return False
