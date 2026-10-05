"""Сделка = дешевле самого дешёвого АКТИВНОГО объявления той же модели (или той же версии)."""

from datetime import timedelta

import pytest
from sqlalchemy import update

from lalascaner.ai.schema import ExtractedItem, PostResult
from lalascaner.db import repo
from lalascaner.db.models import Listing, utcnow
from lalascaner.db.session import init_db, make_engine, make_sessionmaker
from lalascaner.market.deals import DealParams, evaluate_items
from lalascaner.market.stats import recompute_stats
from lalascaner.sources.base import RawListing

RATES = {"KGS": 1.0}
_counter = iter(range(10_000))


@pytest.fixture
async def session(tmp_path):
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path / 'c.db'}")
    await init_db(engine)
    async with make_sessionmaker(engine)() as s:
        yield s
    await engine.dispose()


async def add(session, price, seller, *, condition="used", brand=None, series=None, seen_hours_ago=0, classifier="claude",
              chip="RTX 3060", vram=12):
    n = next(_counter)
    raw = RawListing(
        source="lalafo", external_id=f"l{n}", source_ref="1355", url=f"https://lalafo.kg/{n}", title="Видеокарта",
        text=f"{brand or ''} {chip} {series or ''} #{n}", price=price, currency="KGS", author_id=seller,
        posted_at=utcnow() - timedelta(hours=1),
    )
    res = await repo.upsert_listings(session, [raw])
    listing = await session.get(Listing, res.new_ids[0])
    model = " ".join(filter(None, [brand, chip, series, f"{vram}GB"]))
    result = PostResult(post_id="x", intent="sell", items=[ExtractedItem(
        category="gpu", brand=brand, series=series, model=model, chip=chip, vram_gb=vram,
        condition=condition, price=price, currency="KGS",
    )])
    items = await repo.save_result(session, listing, result, classifier, RATES)
    if seen_hours_ago:
        await session.execute(
            update(Listing).where(Listing.id == listing.id).values(last_seen_at=utcnow() - timedelta(hours=seen_hours_ago))
        )
    await session.flush()
    return items[0]


async def market(session):
    for price, seller in [(20000, "a"), (21000, "b"), (22000, "c")]:
        await add(session, price, seller)
    await recompute_stats(session, window_days=60)


PARAMS = DealParams(mode="cheapest", min_competitors=2, min_gap_percent=0)


async def test_cheaper_than_every_active_listing_is_model_deal(session):
    await market(session)
    new = await add(session, 19500, "d")
    deals = await evaluate_items(session, [new], PARAMS)
    assert deals == [new]
    assert (new.deal_kind, new.ref_min_price, new.ref_count) == ("model", 20000, 3)


async def test_not_cheapest_is_not_a_deal_even_if_below_median(session):
    await market(session)
    new = await add(session, 20500, "d")
    assert await evaluate_items(session, [new], PARAMS) == []
    assert new.deal_kind is None and not new.is_deal


async def test_variant_record_when_not_cheapest_overall(session):
    await market(session)
    await add(session, 23000, "e", brand="Palit", series="Dual")
    await add(session, 24000, "f", brand="Palit", series="Dual")
    new = await add(session, 21500, "g", brand="Palit", series="Dual")
    assert new.variant_key == "gpu:rtx-3060:12gb|palit-dual"
    deals = await evaluate_items(session, [new], PARAMS)
    assert deals == [new]
    assert (new.deal_kind, new.ref_min_price, new.ref_count) == ("variant", 23000, 2)

    no_variant = DealParams(mode="cheapest", min_competitors=2, variant_alerts=False, min_gap_percent=0)
    assert await evaluate_items(session, [new], no_variant) == []


async def test_own_listings_of_same_seller_are_ignored(session):
    await market(session)
    await add(session, 18000, "a")  # у продавца «a» есть ещё дешевле — это не конкуренты
    new = await add(session, 19000, "a")
    deals = await evaluate_items(session, [new], PARAMS)
    assert deals == [new] and new.ref_min_price == 21000


async def test_sold_listings_are_not_competitors(session):
    await market(session)
    await add(session, 15000, "old", seen_hours_ago=72)  # давно не встречалось в ленте — продано/снято
    new = await add(session, 19000, "d")
    assert await evaluate_items(session, [new], PARAMS) == [new]
    assert new.ref_min_price == 20000


async def test_new_items_compete_only_with_new(session):
    await market(session)  # б/у 20–22к
    await add(session, 32000, "n1", condition="new")
    await add(session, 33000, "n2", condition="new")
    new = await add(session, 30000, "d", condition="new")
    assert await evaluate_items(session, [new], PARAMS) == [new]
    assert new.ref_min_price == 32000 and new.ref_count == 2


async def test_used_items_compete_with_new_too(session):
    await market(session)
    await add(session, 19000, "n1", condition="new")
    used = await add(session, 19500, "d")
    assert await evaluate_items(session, [used], PARAMS) == []


async def test_needs_enough_competitors(session):
    await add(session, 20000, "a")
    new = await add(session, 15000, "b")
    assert await evaluate_items(session, [new], PARAMS) == []


async def test_scam_priced_competitor_does_not_block_real_deals(session):
    for price, seller in [(20000, "a"), (21000, "b"), (22000, "c"), (23000, "e"), (24000, "f")]:
        await add(session, price, seller)
    await recompute_stats(session, window_days=60)
    await add(session, 5000, "scammer")  # 5000 < 35% медианы — фейк, не ориентир
    new = await add(session, 19000, "d")
    assert await evaluate_items(session, [new], PARAMS) == [new]
    assert new.ref_min_price == 20000


async def test_broken_items_are_neither_deals_nor_competitors(session):
    await market(session)
    broken = await add(session, 5000, "d", condition="broken")
    assert await evaluate_items(session, [broken], PARAMS) == []
    await add(session, 15000, "junk", condition="broken")  # нерабочая за 15к не мешает рабочей за 19.5к
    new = await add(session, 19500, "e")
    assert await evaluate_items(session, [new], PARAMS) == [new] and new.ref_min_price == 20000


async def test_pc_and_laptop_need_noticeable_gap(session):
    for price, seller in [(58900, "a"), (60000, "b"), (62000, "c")]:
        await add(session, price, seller, chip="RTX 3060")
    # для видеокарт 58000 vs 58900 — рекорд, а для ПК разница в 1.5% — это просто другая конфигурация
    item = await add(session, 58000, "d")
    item.category = "pc"
    assert await evaluate_items(session, [item], PARAMS) == []
    item.category = "gpu"
    assert await evaluate_items(session, [item], PARAMS) == [item]


async def test_monitor_without_resolution_is_not_compared(session):
    from lalascaner.market.deals import _comparable_key

    assert not _comparable_key("mon:19")
    assert _comparable_key("mon:27:2560x1440")


async def test_unconfident_items_compete_but_never_alert(session):
    await market(session)
    unsure = await add(session, 15000, "d", classifier="rules")
    unsure.confidence = 0.55  # правила не уверены (прайс-лист и т.п.)
    assert await evaluate_items(session, [unsure], PARAMS) == []
    new = await add(session, 16000, "e")  # но как конкурент дешёвое «неуверенное» учитывается
    assert await evaluate_items(session, [new], PARAMS) == []


async def test_min_gap_percent(session):
    await market(session)
    new = await add(session, 19700, "d")  # на 1.5% дешевле 20000
    assert await evaluate_items(session, [new], DealParams(mode="cheapest", min_competitors=2, min_gap_percent=3)) == []
    assert await evaluate_items(session, [new], DealParams(mode="cheapest", min_competitors=2, min_gap_percent=1)) == [new]


async def test_min_gap(session):
    await market(session)
    new = await add(session, 19800, "d")
    assert await evaluate_items(session, [new], DealParams(mode="cheapest", min_competitors=2, min_gap_kgs=500, min_gap_percent=0)) == []
    assert await evaluate_items(session, [new], DealParams(mode="cheapest", min_competitors=2, min_gap_kgs=100, min_gap_percent=0)) == [new]
