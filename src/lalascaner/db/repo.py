"""Все обращения к БД собраны здесь, чтобы бот, веб и пайплайн работали через один слой."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import delete, func, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from lalascaner.ai.grounding import key_grounded, price_grounded
from lalascaner.ai.normalize import build_keys, norm_resolution
from lalascaner.ai.variants import build_variant_key
from lalascaner.ai.schema import PostResult
from lalascaner.db.models import (
    KV,
    Alert,
    Channel,
    Item,
    Listing,
    ListingEvent,
    ListingMetric,
    ListingVector,
    PriceHistory,
    PriceStat,
    Subscription,
    utcnow,
)
from lalascaner.market.currency import to_kgs
from lalascaner.market.query import ItemFilter, QueryFilter
from lalascaner.market.sanity import plausible_price
from lalascaner.sources.base import RawListing

CHUNK = 400


# ── объявления ───────────────────────────────────────────────────────────────


@dataclass
class UpsertResult:
    new_ids: list[int] = field(default_factory=list)
    changed_ids: list[int] = field(default_factory=list)
    unchanged: int = 0


async def upsert_listings(session: AsyncSession, raws: list[RawListing]) -> UpsertResult:
    result = UpsertResult()
    if not raws:
        return result
    now = utcnow()
    unique: dict[tuple[str, str], RawListing] = {}
    for raw in raws:
        unique.setdefault((raw.source, raw.external_id), raw)

    by_source: dict[str, list[str]] = {}
    for source, ext in unique:
        by_source.setdefault(source, []).append(ext)
    existing: dict[tuple[str, str], Listing] = {}
    for source, ext_ids in by_source.items():
        for i in range(0, len(ext_ids), CHUNK):
            rows = await session.scalars(
                select(Listing).where(Listing.source == source, Listing.external_id.in_(ext_ids[i : i + CHUNK]))
            )
            existing.update({(row.source, row.external_id): row for row in rows})

    fresh: list[tuple[Listing, list[dict], ListingMetric | None]] = []
    for key, raw in unique.items():
        content_hash = raw.content_hash()
        seen_at = now if raw.source == "lalafo" else raw.posted_at
        row = existing.get(key)
        if row is None:
            listing = Listing(
                source=raw.source,
                source_ref=raw.source_ref,
                external_id=raw.external_id,
                url=raw.url,
                title=raw.title,
                text=raw.text,
                price=raw.price,
                currency=raw.currency,
                city=raw.city,
                author_id=raw.author_id,
                author_name=raw.author_name,
                images=raw.images,
                topic_id=raw.topic_id,
                posted_at=raw.posted_at,
                fetched_at=now,
                last_seen_at=seen_at,
                changed_at=raw.posted_at,
                content_hash=content_hash,
                status="new",
            )
            metric = _track_meta(listing, raw.meta, now)
            session.add(listing)
            events = [{"kind": "new", "at": raw.posted_at, "price": raw.price, "currency": raw.currency}]
            fresh.append((listing, events, metric))
            continue
        row.last_seen_at = max(row.last_seen_at, seen_at)
        if raw.source == "lalafo" and row.gone_at is not None:
            row.gone_at = None
            session.add(ListingEvent(listing_id=row.id, at=now, kind="back", price=raw.price, currency=raw.currency))
        old_push = (row.meta or {}).get("push_up")
        new_push = raw.meta.get("push_up")
        if old_push and new_push and new_push > old_push:
            session.add(ListingEvent(listing_id=row.id, at=now, kind="bump", price=raw.price, currency=raw.currency))
        if (metric := _track_meta(row, raw.meta, now)) is not None:
            metric.listing_id = row.id
            session.add(metric)
        if row.content_hash == content_hash:
            result.unchanged += 1
            continue
        price_changed = raw.price != row.price or raw.currency != row.currency
        session.add(
            ListingEvent(
                listing_id=row.id, at=now, kind="price" if price_changed else "text",
                price=raw.price, currency=raw.currency, old_price=row.price if price_changed else None,
            )
        )
        row.title, row.text, row.price, row.currency = raw.title, raw.text, raw.price, raw.currency
        row.city, row.images, row.url, row.source_ref = raw.city, raw.images, raw.url, raw.source_ref
        row.topic_id = raw.topic_id
        row.content_hash = content_hash
        row.changed_at = now
        row.status, row.intent, row.classifier, row.error = "new", None, None, None
        await session.execute(delete(Item).where(Item.listing_id == row.id))
        result.changed_ids.append(row.id)

    await session.flush()
    for listing, events, metric in fresh:
        session.add_all(ListingEvent(listing_id=listing.id, **event) for event in events)
        if metric is not None:
            metric.listing_id = listing.id
            session.add(metric)
    await session.flush()
    result.new_ids = [listing.id for listing, _, _ in fresh]
    return result


METRIC_KEYS = ("views", "impressions", "favorites", "callers", "writers")
# Показы в ленте растут постоянно — сами по себе не повод для снимка, в отличие от интереса покупателей.
METRIC_TRIGGERS = ("views", "favorites", "callers", "writers")
METRICS_EVERY = timedelta(hours=23)


def _track_meta(row: Listing, meta: dict, now: datetime) -> ListingMetric | None:
    """Обновляет row.meta и решает, пора ли снимок метрик: раз в сутки и только если интерес изменился.
    Значения последнего снимка лежат в meta["_snap"], чтобы сравнивать со снимком, а не с прошлым обходом."""
    old = row.meta or {}
    new = dict(meta)
    snap = old.get("_snap")
    metric = None
    if any(new.get(k) is not None for k in METRIC_KEYS):
        due = row.metrics_at is None or now - row.metrics_at >= METRICS_EVERY
        changed = snap is None or any(new.get(k) != snap.get(k) for k in METRIC_TRIGGERS)
        if due and changed:
            metric = ListingMetric(at=now, **{k: new.get(k) for k in METRIC_KEYS})
            row.metrics_at = now
            snap = {k: new.get(k) for k in METRIC_TRIGGERS}
    if snap is not None:
        new["_snap"] = snap
    if new != old and (new or old):
        row.meta = new or None
    return metric


async def mark_gone(session: AsyncSession, not_seen_since: datetime) -> int:
    """lalafo: объявления, не встречавшиеся в ленте с not_seen_since, считаем пропавшими (продано/снято).
    Время ухода — последний раз, когда видели: точнее не знаем, а для скорости продажи важен порядок."""
    rows = list(
        await session.scalars(
            select(Listing).where(
                Listing.source == "lalafo", Listing.gone_at.is_(None), Listing.last_seen_at < not_seen_since
            )
        )
    )
    for row in rows:
        row.gone_at = row.last_seen_at
        session.add(ListingEvent(listing_id=row.id, at=row.last_seen_at, kind="gone", price=row.price, currency=row.currency))
    await session.flush()
    return len(rows)


async def known_external_ids(session: AsyncSession, source: str, external_ids: list[str]) -> set[str]:
    known: set[str] = set()
    for i in range(0, len(external_ids), CHUNK):
        rows = await session.scalars(
            select(Listing.external_id).where(
                Listing.source == source, Listing.external_id.in_(external_ids[i : i + CHUNK])
            )
        )
        known.update(rows)
    return known


async def pending_listings(session: AsyncSession, limit: int, include_failed: bool = False) -> list[Listing]:
    statuses = ["new", "failed"] if include_failed else ["new"]
    rows = await session.scalars(
        select(Listing).where(Listing.status.in_(statuses)).order_by(Listing.changed_at.desc()).limit(limit)
    )
    return list(rows)


async def count_pending(session: AsyncSession) -> int:
    return await session.scalar(select(func.count()).select_from(Listing).where(Listing.status == "new")) or 0


async def find_done_by_hash(session: AsyncSession, content_hash: str, exclude_id: int) -> Listing | None:
    return await session.scalar(
        select(Listing)
        .where(Listing.content_hash == content_hash, Listing.status == "done", Listing.id != exclude_id)
        .options(selectinload(Listing.items))
        .limit(1)
    )


# Уверенность источника разбора. Локальная модель (7B) надёжнее правил на сложных постах, но слабее Claude.
CONFIDENCE = {"claude": 0.9, "local": 0.75, "rules": 0.6}


def _item_from_result(listing: Listing, extracted, rates: dict[str, float], classifier: str) -> Item:  # noqa: ANN001
    canonical, family = build_keys(extracted)
    price = extracted.price
    currency = extracted.currency or ("KGS" if price is not None else None)
    price_kgs = to_kgs(price, currency, rates)
    attrs = extracted.spec_attrs()
    if not plausible_price(extracted.category, price_kgs):
        attrs["implausible_price"] = price
        price = currency = price_kgs = None
    quantity = max(1, extracted.quantity or 1)
    return Item(
        listing_id=listing.id,
        category=extracted.category,
        intent=extracted.intent,
        brand=extracted.brand,
        series=(extracted.series or None) and extracted.series[:64],
        model=extracted.model[:256],
        canonical_key=canonical[:128],
        family_key=family[:128],
        variant_key=build_variant_key(extracted, canonical[:128]),
        attrs=attrs,
        condition=extracted.condition,
        price=price,
        currency=currency,
        price_kgs=price_kgs,
        quantity=quantity,
        is_bundle=extracted.is_bundle,
        confidence=CONFIDENCE.get(classifier, 0.6),
        posted_at=listing.posted_at,
    )


async def save_result(
    session: AsyncSession,
    listing: Listing,
    result: PostResult,
    classifier: str,
    rates: dict[str, float],
    confidence: float | None = None,
) -> list[Item]:
    await session.execute(delete(Item).where(Item.listing_id == listing.id))
    items = [_item_from_result(listing, extracted, rates, classifier) for extracted in result.items]
    if confidence is not None:
        for item in items:
            item.confidence = confidence
    session.add_all(items)
    listing.intent = result.intent
    listing.status = "done"
    listing.classifier = classifier
    listing.error = None
    await session.flush()
    return items


async def copy_items(session: AsyncSession, listing: Listing, donor: Listing) -> list[Item]:
    await session.execute(delete(Item).where(Item.listing_id == listing.id))
    skip = {
        "id", "listing_id", "posted_at", "created_at", "market_key", "market_median", "market_n", "discount",
        "is_deal", "suspicious", "deal_kind", "ref_min_price", "ref_count",
    }
    items = []
    for src in donor.items:
        data = {c.name: getattr(src, c.name) for c in Item.__table__.columns if c.name not in skip}
        items.append(Item(listing_id=listing.id, posted_at=listing.posted_at, **data))
    session.add_all(items)
    listing.intent = donor.intent
    listing.status = "done"
    listing.classifier = f"copy:{donor.id}"
    listing.error = None
    await session.flush()
    return items


async def mark_failed(session: AsyncSession, listings: list[Listing], error: str) -> None:
    for listing in listings:
        listing.status = "failed"
        listing.error = error[:2000]
    await session.flush()


async def refine_listings(session: AsyncSession, limit: int) -> list[Listing]:
    """Объявления, разобранные правилами и ждущие ИИ: сначала Telegram (там ИИ полезнее всего), потом свежие."""
    rows = await session.scalars(
        select(Listing)
        .where(Listing.status == "refine")
        .order_by((Listing.source == "telegram").desc(), Listing.changed_at.desc())
        .limit(limit)
    )
    return list(rows)


async def count_refine(session: AsyncSession) -> int:
    return await session.scalar(select(func.count()).select_from(Listing).where(Listing.status == "refine")) or 0


async def count_unavailable(session: AsyncSession) -> int:
    """Сколько постов не разобрано из-за недоступности ИИ (error начинается с «unavailable:»)."""
    return (
        await session.scalar(
            select(func.count()).select_from(Listing).where(Listing.status == "failed", Listing.error.like("unavailable:%"))
        )
        or 0
    )


async def requeue_unavailable(session: AsyncSession) -> int:
    rows = await session.scalars(select(Listing).where(Listing.status == "failed", Listing.error.like("unavailable:%")))
    count = 0
    for listing in rows:
        listing.status, listing.error = "new", None
        count += 1
    await session.flush()
    return count


async def listings_missing_topic(session: AsyncSession, source: str, limit: int) -> list[Listing]:
    """Объявления, сохранённые до появления topic_id — их нужно один раз дозапросить, чтобы узнать тему."""
    rows = await session.scalars(
        select(Listing).where(Listing.source == source, Listing.topic_id.is_(None)).limit(limit)
    )
    return list(rows)


async def mark_skipped(session: AsyncSession, listing: Listing, reason: str) -> None:
    listing.status = "done"
    listing.intent = "other"
    listing.classifier = reason
    await session.execute(delete(Item).where(Item.listing_id == listing.id))
    await session.flush()


# ── key-value ────────────────────────────────────────────────────────────────


async def get_kv(session: AsyncSession, key: str, default: Any = None) -> Any:
    row = await session.get(KV, key)
    return row.value if row else default


async def set_kv(session: AsyncSession, key: str, value: Any) -> None:
    row = await session.get(KV, key)
    if row:
        row.value = value
    else:
        session.add(KV(key=key, value=value))
    await session.flush()


# ── каналы ───────────────────────────────────────────────────────────────────


async def list_channels(session: AsyncSession, enabled_only: bool = False) -> list[Channel]:
    stmt = select(Channel).order_by(Channel.id)
    if enabled_only:
        stmt = stmt.where(Channel.enabled.is_(True))
    return list(await session.scalars(stmt))


async def get_channel(session: AsyncSession, username: str) -> Channel | None:
    return await session.scalar(select(Channel).where(Channel.username == username.lstrip("@").lower()))


async def add_channel(session: AsyncSession, username: str, kind: str, title: str | None) -> Channel:
    username = username.lstrip("@").lower()
    channel = await get_channel(session, username)
    if channel:
        channel.enabled, channel.kind, channel.title = True, kind, title or channel.title
    else:
        channel = Channel(username=username, kind=kind, title=title)
        session.add(channel)
    await session.flush()
    return channel


async def remove_channel(session: AsyncSession, username: str) -> bool:
    channel = await get_channel(session, username)
    if not channel:
        return False
    channel.enabled = False
    await session.flush()
    return True


# ── подписки и уведомления ───────────────────────────────────────────────────


async def add_subscription(
    session: AsyncSession,
    chat_id: int,
    query: str = "",
    max_price_kgs: float | None = None,
    min_discount: float | None = None,
    mode: str = "deals",
) -> Subscription:
    sub = Subscription(chat_id=chat_id, query=query, max_price_kgs=max_price_kgs, min_discount=min_discount, mode=mode)
    session.add(sub)
    await session.flush()
    return sub


async def list_subscriptions(session: AsyncSession, chat_id: int | None = None) -> list[Subscription]:
    stmt = select(Subscription).order_by(Subscription.id)
    if chat_id is not None:
        stmt = stmt.where(Subscription.chat_id == chat_id)
    return list(await session.scalars(stmt))


async def delete_subscription(session: AsyncSession, sub_id: int, chat_id: int) -> bool:
    res = await session.execute(
        delete(Subscription).where(Subscription.id == sub_id, Subscription.chat_id == chat_id)
    )
    return (res.rowcount or 0) > 0


async def last_alert(session: AsyncSession, chat_id: int, listing_id: int, canonical_key: str) -> Alert | None:
    return await session.scalar(
        select(Alert)
        .where(Alert.chat_id == chat_id, Alert.listing_id == listing_id, Alert.canonical_key == canonical_key)
        .order_by(Alert.sent_at.desc())
        .limit(1)
    )


async def alerted_listing(session: AsyncSession, chat_id: int, listing_id: int) -> bool:
    return await session.scalar(
        select(Alert.id).where(Alert.chat_id == chat_id, Alert.listing_id == listing_id).limit(1)
    ) is not None


async def alerted_same_offer(
    session: AsyncSession, chat_id: int, item: Item, since: datetime, tolerance: float
) -> bool:
    """Тот же продавец, та же модель, почти та же цена, но другое объявление — перезалив под новым номером."""
    author = item.listing.author_id
    if not author or item.price_kgs is None:
        return False
    rows = await session.scalars(
        select(Alert.price_kgs).where(
            Alert.chat_id == chat_id,
            Alert.author_id == author,
            Alert.canonical_key == item.canonical_key,
            Alert.listing_id != item.listing_id,
            Alert.sent_at >= since,
        )
    )
    return any(p is not None and abs(p - item.price_kgs) <= tolerance * max(p, item.price_kgs) for p in rows)


async def price_drop_event(session: AsyncSession, listing_id: int, since: datetime) -> ListingEvent | None:
    """Последняя смена цены объявления, если цену снизили (после since)."""
    event = await session.scalar(
        select(ListingEvent)
        .where(ListingEvent.listing_id == listing_id, ListingEvent.kind == "price", ListingEvent.at >= since)
        .order_by(ListingEvent.at.desc(), ListingEvent.id.desc())
        .limit(1)
    )
    if event is None or event.price is None or event.old_price is None or event.price >= event.old_price:
        return None
    return event


async def record_alert(session: AsyncSession, chat_id: int, item: Item) -> None:
    session.add(
        Alert(
            chat_id=chat_id, listing_id=item.listing_id, canonical_key=item.canonical_key,
            author_id=item.listing.author_id, price_kgs=item.price_kgs,
        )
    )
    await session.flush()


async def migrate_old_notifications(session: AsyncSession) -> int:
    """Переносит журнал из старой таблицы notifications (была привязана к позициям) в alerts и удаляет её."""
    exists = await session.scalar(text("SELECT name FROM sqlite_master WHERE type='table' AND name='notifications'"))
    if not exists:
        return 0
    result = await session.execute(
        text(
            "INSERT INTO alerts (chat_id, listing_id, canonical_key, author_id, price_kgs, sent_at) "
            "SELECT n.chat_id, i.listing_id, i.canonical_key, l.author_id, i.price_kgs, n.sent_at "
            "FROM notifications n JOIN items i ON i.id = n.item_id JOIN listings l ON l.id = i.listing_id"
        )
    )
    await session.execute(text("DROP TABLE notifications"))
    return result.rowcount or 0


# ── выборки для бота и веба ──────────────────────────────────────────────────


async def get_items(session: AsyncSession, item_ids: list[int]) -> list[Item]:
    if not item_ids:
        return []
    rows = await session.scalars(
        select(Item).where(Item.id.in_(item_ids)).options(selectinload(Item.listing))
    )
    return list(rows)


def _not_copy():  # noqa: ANN202
    """Репосты одного и того же текста в списках не показываем — только оригинал."""
    return or_(Listing.classifier.is_(None), Listing.classifier.not_like("copy:%"))


def _items_stmt(qf, days, flt, order, only_priced, active=None):  # noqa: ANN001, ANN202
    """active — SQL-условие «объявление актуально» (DealParams.active_clause); иначе окно в days дней."""
    window = active if active is not None else Listing.last_seen_at >= utcnow() - timedelta(days=days)
    stmt = select(Item).join(Listing).where(Item.intent == "sell", window, _not_copy())
    if qf is not None:
        stmt = stmt.where(qf.clause(Item.canonical_key, Item.model, Item.category))
    if only_priced or order == "cheap" or (flt and (flt.min_price or flt.max_price)):
        stmt = stmt.where(Item.price_kgs.is_not(None), Item.is_bundle.is_(False))
    if flt:
        if flt.min_price:
            stmt = stmt.where(Item.price_kgs >= flt.min_price)
        if flt.max_price:
            stmt = stmt.where(Item.price_kgs <= flt.max_price)
        if flt.condition:
            stmt = stmt.where(Item.condition == flt.condition)
    return stmt


async def search_items(
    session: AsyncSession,
    qf: QueryFilter,
    limit: int = 10,
    days: int = 30,
    order: str = "recent",
    only_priced: bool = False,
    offset: int = 0,
    flt: ItemFilter | None = None,
    active=None,  # noqa: ANN001
) -> list[Item]:
    stmt = _items_stmt(qf, days, flt, order, only_priced, active).options(selectinload(Item.listing))
    stmt = stmt.order_by(Item.price_kgs.asc() if order == "cheap" else Listing.changed_at.desc())
    return list(await session.scalars(stmt.offset(offset).limit(limit)))


async def count_items(
    session: AsyncSession,
    qf: QueryFilter,
    days: int = 30,
    flt: ItemFilter | None = None,
    order: str = "recent",
    active=None,  # noqa: ANN001
) -> int:
    stmt = _items_stmt(qf, days, flt, order, False, active).with_only_columns(func.count(Item.id))
    return await session.scalar(stmt) or 0


def key_clause(key: str):  # noqa: ANN202
    return or_(Item.canonical_key == key, Item.canonical_key.like(key + ":%"), Item.family_key == key)


def _offers_base(key: str, active, min_price: float | None = None):  # noqa: ANN001, ANN202
    stmt = (
        select(Item)
        .join(Listing)
        .where(key_clause(key), Item.intent == "sell", Item.price_kgs.is_not(None), Item.is_bundle.is_(False))
        .where(Item.condition != "broken", active, _not_copy())
    )
    if min_price:
        # Фейково дешёвые цены (из прайс-листов, скам) не показываем как «самое дешёвое».
        stmt = stmt.where(Item.price_kgs >= min_price)
    return stmt


async def items_for_key(
    session: AsyncSession, key: str, active, offset: int = 0, limit: int = 8, min_price: float | None = None  # noqa: ANN001
) -> tuple[list[Item], int]:
    """Актуальные предложения модели от дешёвых."""
    base = _offers_base(key, active, min_price)
    total = await session.scalar(base.with_only_columns(func.count(Item.id))) or 0
    rows = await session.scalars(
        base.options(selectinload(Item.listing)).order_by(Item.price_kgs.asc()).offset(offset).limit(limit)
    )
    return list(rows), total


async def active_variants(
    session: AsyncSession, key: str, active, limit: int = 8, min_price: float | None = None  # noqa: ANN001
) -> list[tuple[str | None, int, float]]:
    """Самое дешёвое актуальное предложение по каждой версии модели: [(variant_key, кол-во, мин. цена)]."""
    base = _offers_base(key, active, min_price).subquery()
    rows = await session.execute(
        select(base.c.variant_key, func.count(), func.min(base.c.price_kgs))
        .group_by(base.c.variant_key)
        .order_by(func.min(base.c.price_kgs))
        .limit(limit)
    )
    return [(variant, count, price) for variant, count, price in rows.all()]


async def models_in_category(
    session: AsyncSession, category: str, offset: int = 0, limit: int = 10, min_n: int = 2
) -> tuple[list[PriceStat], int]:
    base = select(PriceStat).where(PriceStat.condition == "all", PriceStat.category == category, PriceStat.n >= min_n)
    total = await session.scalar(base.with_only_columns(func.count())) or 0
    rows = await session.scalars(base.order_by(PriceStat.n.desc(), PriceStat.key).offset(offset).limit(limit))
    return list(rows), total


def key_token(key: str) -> str:
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


async def key_by_token(session: AsyncSession, token: str) -> str | None:
    """callback_data в Telegram ограничена 64 байтами, поэтому в кнопках — короткий хэш ключа модели."""
    keys = await session.scalars(select(PriceStat.key).where(PriceStat.condition == "all"))
    return next((key for key in keys if key_token(key) == token), None)


async def market_rows(session: AsyncSession, category: str | None = None) -> list[PriceStat]:
    stmt = select(PriceStat).order_by(PriceStat.category, PriceStat.n.desc())
    if category:
        stmt = stmt.where(PriceStat.category == category)
    return list(await session.scalars(stmt))


async def stats_for(session: AsyncSession, qf: QueryFilter, limit: int = 10) -> list[PriceStat]:
    stmt = (
        select(PriceStat)
        .where(qf.clause(PriceStat.key, None, PriceStat.category))
        .where(PriceStat.condition == "all")
        .order_by(PriceStat.n.desc())
        .limit(limit)
    )
    return list(await session.scalars(stmt))


async def stats_by_key(session: AsyncSession, key: str) -> dict[str, PriceStat]:
    rows = await session.scalars(select(PriceStat).where(PriceStat.key == key))
    return {row.condition: row for row in rows}


async def recent_deals(
    session: AsyncSession, hours: int = 48, limit: int = 20, category: str | None = None, active=None  # noqa: ANN001
) -> list[Item]:
    since = utcnow() - timedelta(hours=hours)
    # Сортировка: насколько дешевле самого дешёвого конкурента (для режима median — насколько ниже медианы).
    gap = func.coalesce((Item.ref_min_price - Item.price_kgs) / Item.ref_min_price, Item.discount)
    stmt = (
        select(Item)
        .join(Listing)
        .where(Item.is_deal.is_(True), Listing.changed_at >= since, _not_copy())
        .options(selectinload(Item.listing))
        .order_by(gap.desc())
        .limit(limit)
    )
    if active is not None:
        stmt = stmt.where(active)
    if category:
        stmt = stmt.where(Item.category == category)
    return list(await session.scalars(stmt))


async def db_stats(session: AsyncSession) -> dict[str, Any]:
    listings_by_source = dict(
        (await session.execute(select(Listing.source, func.count()).group_by(Listing.source))).all()
    )
    listings_by_status = dict(
        (await session.execute(select(Listing.status, func.count()).group_by(Listing.status))).all()
    )
    items_by_category = dict(
        (
            await session.execute(
                select(Item.category, func.count()).where(Item.intent == "sell").group_by(Item.category)
            )
        ).all()
    )
    events_by_kind = dict(
        (await session.execute(select(ListingEvent.kind, func.count()).group_by(ListingEvent.kind))).all()
    )
    return {
        "listings_by_source": listings_by_source,
        "listings_by_status": listings_by_status,
        "items_by_category": items_by_category,
        "events_by_kind": events_by_kind,
        "gone": await session.scalar(select(func.count()).select_from(Listing).where(Listing.gone_at.is_not(None))),
        "metrics": await session.scalar(select(func.count()).select_from(ListingMetric)),
        "vectors": await session.scalar(select(func.count()).select_from(ListingVector)),
        "history_days": await session.scalar(select(func.count(func.distinct(PriceHistory.day)))),
        "items_total": sum(items_by_category.values()),
        "models_tracked": await session.scalar(
            select(func.count()).select_from(PriceStat).where(PriceStat.condition == "all")
        ),
        "deals_24h": await session.scalar(
            select(func.count())
            .select_from(Item)
            .join(Listing)
            .where(Item.is_deal.is_(True), Listing.changed_at >= utcnow() - timedelta(hours=24))
        ),
    }


def freshness_cutoff(hours: int) -> datetime:
    return utcnow() - timedelta(hours=hours)


# ── история цен, векторы, скорость продажи ───────────────────────────────────


async def snapshot_price_history(session: AsyncSession, day: str, min_n: int = 3) -> int:
    """Сохраняет текущие price_stats как снимок дня (модели хотя бы с min_n предложениями — хвост из
    единичных объявлений в динамике только шумит). Повторный вызов в тот же день перезаписывает снимок."""
    await session.execute(delete(PriceHistory).where(PriceHistory.day == day))
    stats = list(await session.scalars(select(PriceStat).where(PriceStat.n >= min_n)))
    session.add_all(
        PriceHistory(
            day=day, key=s.key, condition=s.condition, category=s.category, n=s.n,
            median=s.median, p25=s.p25, p75=s.p75, min=s.min, max=s.max,
        )
        for s in stats
    )
    await session.flush()
    return len(stats)


async def price_trend(session: AsyncSession, key: str, condition: str = "all", days: int = 90) -> list[PriceHistory]:
    since = (utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
    rows = await session.scalars(
        select(PriceHistory)
        .where(PriceHistory.key == key, PriceHistory.condition == condition, PriceHistory.day >= since)
        .order_by(PriceHistory.day)
    )
    return list(rows)


async def listings_needing_vectors(session: AsyncSession, model: str, limit: int) -> list[Listing]:
    """Разобранные объявления с позициями, у которых нет вектора этой модели или текст менялся после него.
    Свежие первыми — они нужнее для похожих, дублей и оценок."""
    has_items = select(Item.id).where(Item.listing_id == Listing.id).exists()
    has_vector = (
        select(ListingVector.listing_id)
        .where(
            ListingVector.listing_id == Listing.id,
            ListingVector.model == model,
            ListingVector.created_at >= Listing.changed_at,
        )
        .exists()
    )
    rows = await session.scalars(
        select(Listing)
        .where(Listing.status.in_(["done", "refine"]), has_items, ~has_vector)
        .options(selectinload(Listing.items))
        .order_by(Listing.changed_at.desc())
        .limit(limit)
    )
    return list(rows)


async def save_vectors(session: AsyncSession, model: str, vectors: dict[int, bytes]) -> None:
    now = utcnow()
    ids = list(vectors)
    existing = {
        v.listing_id: v
        for v in await session.scalars(select(ListingVector).where(ListingVector.listing_id.in_(ids)))
    }
    for listing_id, blob in vectors.items():
        row = existing.get(listing_id)
        if row is None:
            session.add(ListingVector(listing_id=listing_id, model=model, vec=blob, created_at=now))
        else:
            row.model, row.vec, row.created_at = model, blob, now
    await session.flush()


async def load_vectors(session: AsyncSession, model: str) -> list[tuple[int, bytes]]:
    rows = await session.execute(select(ListingVector.listing_id, ListingVector.vec).where(ListingVector.model == model))
    return [(lid, blob) for lid, blob in rows.all()]


async def sell_items_of_listings(
    session: AsyncSession, listing_ids: list[int], active=None, since: datetime | None = None  # noqa: ANN001
) -> list[Item]:
    """Позиции «продаю» с ценой у этих объявлений — кандидаты для похожих и оценок по соседям."""
    if not listing_ids:
        return []
    stmt = (
        select(Item)
        .join(Listing)
        .where(
            Item.listing_id.in_(listing_ids),
            Item.intent == "sell",
            Item.price_kgs.is_not(None),
            Item.is_bundle.is_(False),
            Item.condition != "broken",
        )
        .options(selectinload(Item.listing))
    )
    if active is not None:
        stmt = stmt.where(active)
    if since is not None:
        stmt = stmt.where(Listing.last_seen_at >= since)
    return list(await session.scalars(stmt))


async def liquidity_rows(session: AsyncSession, since: datetime) -> list[tuple]:
    """Позиции lalafo для расчёта скорости продажи: цена относительно рынка + когда появилось и пропало.
    Только объявления, чьё появление мы застали (иначе не знаем, сколько оно уже висело до нас),
    и не копии/перезаливы — они не самостоятельные продажи."""
    rows = await session.execute(
        select(
            Listing.id, Item.category, Item.price_kgs, Item.market_median,
            Listing.posted_at, Listing.gone_at,
        )
        .join(Listing)
        .where(
            Listing.source == "lalafo",
            Listing.posted_at >= since,
            func.julianday(Listing.fetched_at) - func.julianday(Listing.posted_at) <= 2,
            Listing.classifier.not_like("copy:%"),
            Listing.dup_of.is_(None),
            Item.intent == "sell",
            Item.is_bundle.is_(False),
            Item.price_kgs.is_not(None),
            Item.market_median > 0,
            Item.market_n >= 5,
        )
    )
    return list(rows.all())


async def fix_monitor_resolution(session: AsyncSession) -> int:
    """Разовая починка: правила раньше записывали в attrs.resolution всю строку объявления."""
    fixed = 0
    for item in await session.scalars(select(Item).where(Item.category == "monitor")):
        attrs = dict(item.attrs or {})
        raw = attrs.get("resolution")
        if not isinstance(raw, str) or re.fullmatch(r"\d{3,4}x\d{3,4}", raw):
            continue
        clean = norm_resolution(raw)
        if clean:
            attrs["resolution"] = clean
        else:
            attrs.pop("resolution")
        item.attrs = attrs
        fixed += 1
    await session.flush()
    return fixed


async def ground_ai_items(session: AsyncSession, text_of) -> tuple[int, int]:  # noqa: ANN001
    """Разовая чистка позиций, разобранных ИИ до проверки по тексту: выдуманная цена убирается (исходное
    значение остаётся в attrs.ungrounded_price), выдуманная модель удаляется. (без цены, удалено)."""
    unpriced = removed = 0
    rows = await session.scalars(
        select(Item).join(Listing).where(Listing.classifier.in_(["local", "claude"])).options(selectinload(Item.listing))
    )
    for item in rows:
        text = text_of(item.listing)
        if not key_grounded(item.category, item.canonical_key, text):
            await session.delete(item)
            removed += 1
            continue
        if item.price is not None and not price_grounded(item.price, text):
            item.attrs = {**(item.attrs or {}), "ungrounded_price": item.price}
            item.price = item.currency = item.price_kgs = None
            item.is_deal, item.deal_kind = False, None
            unpriced += 1
    await session.flush()
    return unpriced, removed


async def reposted_listing_ids(session: AsyncSession) -> set[int]:
    """Объявления, которые потом перезалили: их исчезновение — не продажа."""
    dups = set(await session.scalars(select(Listing.dup_of).where(Listing.dup_of.is_not(None)).distinct()))
    for classifier in await session.scalars(select(Listing.classifier).where(Listing.classifier.like("copy:%"))):
        donor = classifier.split(":", 1)[1]
        if donor.isdigit():
            dups.add(int(donor))
    return dups
