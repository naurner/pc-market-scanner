from datetime import timedelta

import pytest

from lalascaner.ai.schema import ExtractedItem, PostResult
from lalascaner.db import repo
from lalascaner.db.models import utcnow
from lalascaner.db.session import init_db, make_engine, make_sessionmaker
from lalascaner.market.currency import parse_nbkr, to_kgs
from lalascaner.market.deals import DealParams, evaluate_items
from lalascaner.market.query import parse_query
from lalascaner.market.stats import recompute_stats, summarize
from lalascaner.sources.base import RawListing

RATES = {"KGS": 1.0, "USD": 87.0}


@pytest.fixture
async def session(tmp_path):
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")
    await init_db(engine)
    maker = make_sessionmaker(engine)
    async with maker() as s:
        yield s
    await engine.dispose()


def raw(ext_id: str, text: str = "rtx 3060 12gb", price: float | None = 20000, author: str = "a", source="lalafo", **kw):
    return RawListing(
        source=source,
        external_id=ext_id,
        source_ref="1355",
        url=f"https://example/{ext_id}",
        title="Видеокарта",
        text=text,
        price=price,
        currency="KGS",
        author_id=author,
        posted_at=kw.pop("posted_at", utcnow() - timedelta(hours=1)),
        **kw,
    )


def gpu_result(post_id: str, price: float | None, condition="used", currency="KGS", vram=12, chip="RTX 3060"):
    return PostResult(
        post_id=post_id,
        intent="sell",
        items=[
            ExtractedItem(
                category="gpu", model=f"{chip} {vram}GB", chip=chip, vram_gb=vram, condition=condition,
                price=price, currency=currency,
            )
        ],
    )


async def test_upsert_new_unchanged_changed(session):
    first = await repo.upsert_listings(session, [raw("1"), raw("2")])
    assert len(first.new_ids) == 2
    listing = (await repo.pending_listings(session, 10))[0]
    await repo.save_result(session, listing, gpu_result("x", 20000), "rules", RATES)

    again = await repo.upsert_listings(session, [raw("1"), raw("2")])
    assert again.new_ids == [] and again.changed_ids == [] and again.unchanged == 2

    changed = await repo.upsert_listings(session, [raw(listing.external_id, price=15000)])
    assert changed.changed_ids == [listing.id]
    await session.refresh(listing, ["items"])
    assert listing.status == "new" and listing.items == []
    assert listing.changed_at > listing.posted_at


async def test_save_result_builds_keys_and_converts_currency(session):
    res = await repo.upsert_listings(session, [raw("1", price=None)])
    listing = (await repo.pending_listings(session, 10))[0]
    items = await repo.save_result(session, listing, gpu_result("1", 250, currency="USD"), "claude", RATES)
    assert listing.status == "done" and listing.intent == "sell"
    assert items[0].canonical_key == "gpu:rtx-3060:12gb"
    assert items[0].family_key == "gpu:rtx-3060"
    assert items[0].price_kgs == 250 * 87.0
    assert res.new_ids == [listing.id]


async def test_copy_items_from_donor(session):
    await repo.upsert_listings(session, [raw("1", source="telegram"), raw("2", source="telegram")])
    donor, target = sorted(await repo.pending_listings(session, 10), key=lambda row: row.external_id)
    await repo.save_result(session, donor, gpu_result("1", 20000), "claude", RATES)
    found = await repo.find_done_by_hash(session, target.content_hash, exclude_id=target.id)
    assert found is not None and found.id == donor.id
    items = await repo.copy_items(session, target, found)
    assert target.status == "done" and target.classifier == f"copy:{donor.id}"
    assert items[0].canonical_key == "gpu:rtx-3060:12gb"


def test_summarize_filters_outliers():
    s = summarize([20000, 21000, 22000, 23000, 24000, 90000])
    assert s.n == 5 and s.median == 22000 and s.max == 24000


async def seed_market(session, prices, chip="RTX 3060", vram=12, condition="used"):
    for i, price in enumerate(prices):
        ext = f"m-{chip}-{vram}-{i}"
        await repo.upsert_listings(session, [raw(ext, text=f"{chip} {vram} #{i}", price=price, author=f"seller{i}")])
    for listing in await repo.pending_listings(session, 1000):
        await repo.save_result(session, listing, gpu_result(listing.external_id, listing.price, condition, vram=vram, chip=chip), "rules", RATES)


async def test_recompute_stats_dedups_same_seller(session):
    await seed_market(session, [20000, 21000, 22000, 23000, 24000, 90000])
    # тот же продавец повторил объявление — не должен удваивать вес
    await repo.upsert_listings(session, [raw("repost", text="rtx 3060 repost", price=20000, author="seller0")])
    listing = (await repo.pending_listings(session, 10))[0]
    await repo.save_result(session, listing, gpu_result("r", 20000), "rules", RATES)

    await recompute_stats(session, window_days=60)
    stats = await repo.stats_by_key(session, "gpu:rtx-3060:12gb")
    assert stats["used"].n == 5
    assert stats["used"].median == 22000
    assert "all" in stats and "new" not in stats
    assert (await repo.stats_by_key(session, "gpu:rtx-3060"))["all"].n == 5


async def test_evaluate_items_flags_deals_and_suspicious(session):
    await seed_market(session, [20000, 21000, 22000, 23000, 24000])
    await recompute_stats(session, window_days=60)

    await repo.upsert_listings(
        session, [raw("cheap", text="a", price=15000), raw("scam", text="b", price=5000), raw("fair", text="c", price=21000)]
    )
    items = []
    for listing in await repo.pending_listings(session, 10):
        # правила уверенно разобрали однозначные объявления
        items += await repo.save_result(
            session, listing, gpu_result(listing.external_id, listing.price), "rules", RATES, confidence=0.75
        )

    deals = await evaluate_items(session, items, DealParams(min_samples=5, threshold=0.2, suspicious_ratio=0.35))
    by_price = {item.price: item for item in items}
    # 5000 от правил — подозрительно дёшево, скорее ошибка разбора: не сделка (случай ИИ — в тесте ниже)
    assert {d.price for d in deals} == {15000}
    assert by_price[15000].discount == pytest.approx(1 - 15000 / 22000, abs=1e-4)
    assert by_price[5000].suspicious and not by_price[15000].suspicious
    assert not by_price[21000].is_deal
    assert by_price[15000].market_key == "gpu:rtx-3060:12gb|used"


async def test_deal_filters_small_savings_and_low_confidence_scams(session):
    await seed_market(session, [2000, 2100, 2200, 2300, 2400], chip="GT 1030", vram=2)
    await seed_market(session, [20000, 21000, 22000, 23000, 24000])
    await recompute_stats(session, window_days=60)
    await repo.upsert_listings(
        session,
        [raw("tiny", text="x", price=1000), raw("rulescam", text="y", price=3000), raw("aiscam", text="z", price=3000)],
    )
    listings = {l.external_id: l for l in await repo.pending_listings(session, 10)}
    tiny = await repo.save_result(session, listings["tiny"], gpu_result("t", 1000, chip="GT 1030", vram=2), "rules", RATES)
    rules_scam = await repo.save_result(session, listings["rulescam"], gpu_result("r", 3000), "rules", RATES)
    ai_scam = await repo.save_result(session, listings["aiscam"], gpu_result("a", 3000), "claude", RATES)
    params = DealParams(min_samples=5, threshold=0.2, suspicious_ratio=0.35, min_saving_kgs=1500, mode="median")
    deals = await evaluate_items(session, tiny + rules_scam + ai_scam, params)
    assert tiny[0].discount > 0.5 and not tiny[0].is_deal  # экономия 1200 сом — мелочь
    assert rules_scam[0].suspicious and not rules_scam[0].is_deal  # правила могли ошибиться с ценой
    assert ai_scam[0].suspicious and ai_scam[0].is_deal
    assert deals == ai_scam


async def test_evaluate_falls_back_to_family(session):
    await seed_market(session, [20000, 21000, 22000, 23000, 24000], vram=12)
    await recompute_stats(session, window_days=60)
    await repo.upsert_listings(session, [raw("novram", text="rtx 3060 8gb", price=12000)])
    listing = (await repo.pending_listings(session, 10))[0]
    items = await repo.save_result(session, listing, gpu_result("n", 12000, vram=8), "rules", RATES)
    await evaluate_items(session, items, DealParams(min_samples=5))
    assert items[0].market_key == "gpu:rtx-3060|used"


async def test_query_search_and_stats(session):
    await seed_market(session, [20000, 21000, 22000], chip="RTX 3060", vram=12)
    await seed_market(session, [30000, 31000, 32000], chip="RTX 3060 Ti", vram=8)
    await recompute_stats(session, window_days=60)

    qf = parse_query("3060 12gb")
    assert qf.key_prefixes == ["gpu:rtx-3060:12gb"]
    stats = await repo.stats_for(session, qf)
    assert [s.key for s in stats] == ["gpu:rtx-3060:12gb"]

    qf = parse_query("rtx 3060")
    keys = {s.key for s in await repo.stats_for(session, qf)}
    assert keys == {"gpu:rtx-3060", "gpu:rtx-3060:12gb"}

    cheap = await repo.search_items(session, parse_query("3060 ti"), order="cheap")
    assert [i.price for i in cheap] == [30000, 31000, 32000]


@pytest.mark.parametrize(
    "query,prefixes,category",
    [
        ("ryzen 5 5600", ["cpu:ryzen-5-5600"], "cpu"),
        ("12400f", ["cpu:i5-12400f"], "cpu"),
        ("ddr4 16gb", ["ram:ddr4:16gb", "ram:ddr4-so:16gb"], "ram"),
        ("ssd 1tb", ["ssd:1tb", "ssd:nvme:1tb", "ssd:sata:1tb"], "ssd"),
        ("бп 750", ["psu:750w"], "psu"),
    ],
)
def test_parse_query(query, prefixes, category):
    qf = parse_query(query)
    assert qf.key_prefixes == prefixes
    assert qf.category == category


def test_parse_query_free_text():
    qf = parse_query("razer deathadder")
    assert qf.key_prefixes == [] and qf.tokens == ["razer", "deathadder"]


def test_currency():
    xml = '<CurrencyRates><Currency ISOCode="USD">\n<Nominal>1</Nominal>\n<Value>87,4500</Value>\n</Currency>' \
          '<Currency ISOCode="KZT">\n<Nominal>1</Nominal>\n<Value>0,1952</Value>\n</Currency></CurrencyRates>'
    rates = parse_nbkr(xml)
    assert rates["USD"] == 87.45 and rates["KZT"] == 0.1952 and rates["KGS"] == 1.0
    assert to_kgs(100, "USD", rates) == 8745.0
    assert to_kgs(None, "USD", rates) is None
