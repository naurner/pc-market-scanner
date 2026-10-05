"""Сбор максимума данных: метаданные lalafo, история изменений, исчезновения, снимки интереса и цен."""

import json
from datetime import timedelta

from sqlalchemy import select

from lalascaner.ai.rules import classify_rules
from lalascaner.db import repo
from lalascaner.db.models import Item, Listing, ListingEvent, ListingMetric, PriceHistory, PriceStat, utcnow
from lalascaner.sources.base import RawListing
from lalascaner.sources.lalafo import parse_item
from lalascaner.sources.telegram_web import parse_channel_page


def raw(ext_id="1", price=10000.0, text="RTX 3060 12GB", **meta):
    return RawListing(
        source="lalafo", external_id=ext_id, source_ref="1355", url=f"https://lalafo.kg/ads/{ext_id}",
        title="Видеокарта", text=text, price=price, currency="KGS", author_id="u1",
        posted_at=utcnow() - timedelta(hours=2), meta=meta,
    )


async def events(maker, kind=None):
    async with maker() as session:
        stmt = select(ListingEvent).order_by(ListingEvent.id)
        if kind:
            stmt = stmt.where(ListingEvent.kind == kind)
        return list(await session.scalars(stmt))


async def store(maker, *raws):
    async with maker() as session:
        result = await repo.upsert_listings(session, list(raws))
        await session.commit()
        return result


def test_lalafo_meta_keeps_interest_seller_and_hashes_phone(fixtures):
    data = json.loads((fixtures / "lalafo_feed_cpu.json").read_text(encoding="utf-8"))
    item = data["items"][0]
    meta = parse_item(item).meta
    assert meta["favorites"] == 173 and meta["callers"] == 88 and meta["writers"] == 129
    assert meta["impressions"] == 178964 and meta["old_price"] == 14000
    assert "seller_response_rate" in meta and "lat" in meta and "push_up" in meta
    assert len(meta["phone_hash"]) == 16
    digits = "".join(ch for ch in item["mobile"] if ch.isdigit())
    assert digits not in json.dumps(meta)  # сам номер не храним
    assert all(v is not None for v in meta.values())


def test_telegram_channel_views_are_parsed(fixtures):
    listings, _ = parse_channel_page((fixtures / "tg_channel_page.html").read_text(encoding="utf-8"), "durov")
    views = [item.meta.get("views") for item in listings]
    assert all(isinstance(v, int) and v > 100_000 for v in views)


async def test_new_price_text_changes_are_recorded(maker):
    await store(maker, raw(price=10000))
    await store(maker, raw(price=10000))  # без изменений — ничего не пишем
    await store(maker, raw(price=9000))
    await store(maker, raw(price=9000, text="RTX 3060 12GB, торг"))
    kinds = [(e.kind, e.price, e.old_price) for e in await events(maker)]
    assert kinds == [("new", 10000, None), ("price", 9000, 10000), ("text", 9000, None)]


async def test_bump_is_recorded_when_seller_raises_listing(maker):
    await store(maker, raw(push_up=1000))
    await store(maker, raw(push_up=1000))
    await store(maker, raw(push_up=2000))
    assert len(await events(maker, "bump")) == 1


async def test_metrics_snapshot_daily_and_only_on_change(maker):
    await store(maker, raw(views=1, favorites=0, impressions=10))
    await store(maker, raw(views=5, favorites=1, impressions=20))  # меньше суток — снимка нет
    async with maker() as session:
        listing = await session.scalar(select(Listing))
        listing.metrics_at = utcnow() - timedelta(hours=25)
        await session.commit()
    await store(maker, raw(views=5, favorites=1, impressions=30))
    async with maker() as session:
        listing = await session.scalar(select(Listing))
        listing.metrics_at = utcnow() - timedelta(hours=25)
        await session.commit()
    await store(maker, raw(views=5, favorites=1, impressions=99))  # изменились только показы — не повод
    async with maker() as session:
        metrics = list(await session.scalars(select(ListingMetric).order_by(ListingMetric.id)))
        listing = await session.scalar(select(Listing))
    assert [(m.views, m.favorites) for m in metrics] == [(1, 0), (5, 1)]
    assert listing.meta["impressions"] == 99  # последние значения всё равно сохраняются


async def test_gone_and_back(maker):
    await store(maker, raw("1"), raw("2"))
    async with maker() as session:
        stale = await session.scalar(select(Listing).where(Listing.external_id == "1"))
        stale.last_seen_at = utcnow() - timedelta(days=1)
        telegram = Listing(
            source="telegram", external_id="g/1", source_ref="g", text="x",
            last_seen_at=utcnow() - timedelta(days=5), posted_at=utcnow() - timedelta(days=5),
        )
        session.add(telegram)
        await session.commit()
        gone = await repo.mark_gone(session, utcnow() - timedelta(hours=14))
        await session.commit()
    assert gone == 1  # Telegram не отслеживаем: пропажу поста так не заметить
    async with maker() as session:
        row = await session.scalar(select(Listing).where(Listing.external_id == "1"))
        assert row.gone_at == row.last_seen_at
    await store(maker, raw("1"))
    async with maker() as session:
        row = await session.scalar(select(Listing).where(Listing.external_id == "1"))
    assert row.gone_at is None
    assert [e.kind for e in await events(maker) if e.kind in ("gone", "back")] == ["gone", "back"]


async def test_price_history_snapshot_skips_tiny_samples_and_is_idempotent(maker):
    async with maker() as session:
        session.add_all(
            [
                PriceStat(key="gpu:rtx-3060:12gb", condition="all", category="gpu", title="RTX 3060", n=10,
                          median=20000, p25=18000, p75=22000, min=15000, max=25000),
                PriceStat(key="gpu:rare", condition="all", category="gpu", title="rare", n=1,
                          median=5000, p25=5000, p75=5000, min=5000, max=5000),
            ]
        )
        await session.commit()
        assert await repo.snapshot_price_history(session, "2026-10-01") == 1
        assert await repo.snapshot_price_history(session, "2026-10-01") == 1
        await session.commit()
        rows = list(await session.scalars(select(PriceHistory)))
        trend = await repo.price_trend(session, "gpu:rtx-3060:12gb", days=36500)
    assert [(r.day, r.key, r.median) for r in rows] == [("2026-10-01", "gpu:rtx-3060:12gb", 20000)]
    assert len(trend) == 1


def test_monitor_rules_store_only_resolution():
    outcome = classify_rules("1", "Монитор Samsung 27 дюймов 2560x1440 144hz", title="Монитор, Samsung, Б/у", price=15000, category_hint="monitor")
    assert outcome.result.items[0].resolution == "2560x1440"


async def test_fix_monitor_resolution_repairs_old_rows(maker):
    await store(maker, raw())
    async with maker() as session:
        listing = await session.scalar(select(Listing))
        bad = {"diagonal_in": 27.0, "resolution": 'Монитор, Benq, Б/у, LED, 27" 1920x1080'}
        junk = {"diagonal_in": 18.0, "resolution": 'Монитор, Benq, Б/у, LED, 18" - 19"'}
        good = {"diagonal_in": 24.0, "resolution": "1920x1080"}
        for attrs in (bad, junk, good):
            session.add(Item(listing_id=listing.id, category="monitor", model="m", canonical_key="mon:x", family_key="mon", attrs=attrs))
        await session.commit()
        assert await repo.fix_monitor_resolution(session) == 2
        await session.commit()
        items = list(await session.scalars(select(Item).order_by(Item.id)))
    assert [i.attrs.get("resolution") for i in items] == ["1920x1080", None, "1920x1080"]
