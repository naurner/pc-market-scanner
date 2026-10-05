from datetime import timedelta

import pytest

from lalascaner.ai.schema import ExtractedItem, PostResult
from lalascaner.bot.browse import ITEMS_PER_PAGE, model_card, search_view
from lalascaner.bot.notifier import item_matches_query
from lalascaner.db import repo
from lalascaner.db.models import utcnow
from lalascaner.db.session import init_db, make_engine, make_sessionmaker
from lalascaner.market.deals import DealParams
from lalascaner.market.query import parse_search
from lalascaner.market.stats import recompute_stats
from lalascaner.sources.base import RawListing

RATES = {"KGS": 1.0, "USD": 87.0}


@pytest.fixture
async def maker(tmp_path):
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path / 'b.db'}")
    await init_db(engine)
    yield make_sessionmaker(engine)
    await engine.dispose()


async def seed(maker, prices, chip="RTX 3060", vram=12, condition="used", prefix="g"):
    async with maker() as session:
        raws = [
            RawListing(
                source="lalafo", external_id=f"{prefix}{i}", source_ref="1355", url=f"https://lalafo.kg/{prefix}{i}",
                title="Видеокарта", text=f"{chip} {vram}gb #{prefix}{i}", price=p, currency="KGS",
                author_id=f"seller-{prefix}{i}", posted_at=utcnow() - timedelta(hours=i + 1),
            )
            for i, p in enumerate(prices)
        ]
        await repo.upsert_listings(session, raws)
        for listing in await repo.pending_listings(session, 1000):
            result = PostResult(
                post_id="x", intent="sell",
                items=[ExtractedItem(category="gpu", model=f"{chip} {vram}GB", chip=chip, vram_gb=vram,
                                     condition=condition, price=listing.price, currency="KGS")],
            )
            await repo.save_result(session, listing, result, "claude", RATES)
        await recompute_stats(session, window_days=60)
        await session.commit()


@pytest.mark.parametrize(
    "text,prefixes,tokens,flt",
    [
        ("3060 12gb до 25к б/у", ["gpu:rtx-3060:12gb"], [], (None, 25000.0, "used")),
        ("ryzen 5600 до 10 000", ["cpu:ryzen-5-5600"], [], (None, 10000.0, None)),
        ("ноутбук 4060 от 50к до 90к", [], ["4060"], (50000.0, 90000.0, None)),
        ("новый 4070 ti", ["gpu:rtx-4070-ti"], [], (None, None, "new")),
    ],
)
def test_parse_search(text, prefixes, tokens, flt):
    qf, item_filter = parse_search(text)
    assert qf.key_prefixes == prefixes
    assert qf.tokens == tokens
    assert (item_filter.min_price, item_filter.max_price, item_filter.condition) == flt


async def test_search_view_filters_and_paginates(maker):
    await seed(maker, [18000 + i * 500 for i in range(12)])
    text, markup = await search_view(maker, "3060 12gb до 22000", page=0, order="cheap", params=DealParams())
    assert "RTX 3060 12GB" in text and "Медиана" in text
    assert "(1–8 из 9)" in text
    callbacks = [b.callback_data for row in markup.inline_keyboard for b in row]
    assert any(cb and cb.startswith("s:") and cb.endswith(":1") for cb in callbacks)  # кнопка «▶»

    text2, _ = await search_view(maker, "3060 12gb до 22000", page=1, order="cheap", params=DealParams())
    assert "(9–9 из 9)" in text2


async def test_search_view_empty_query_shows_help(maker):
    text, markup = await search_view(maker, "   ", page=0, order="cheap", params=DealParams())
    assert "Напишите модель" in text and markup is None


async def test_model_card_and_key_token(maker):
    await seed(maker, [20000, 21000, 22000, 23000, 24000, 25000, 26000, 27000, 28000, 29000])
    async with maker() as session:
        token = repo.key_token("gpu:rtx-3060:12gb")
        assert await repo.key_by_token(session, token) == "gpu:rtx-3060:12gb"
        models, total = await repo.models_in_category(session, "gpu")
    assert total == 2 and {m.key for m in models} == {"gpu:rtx-3060", "gpu:rtx-3060:12gb"}
    text, rows = await model_card(maker, "gpu:rtx-3060:12gb", page=0, params=DealParams())
    assert "Актуальные предложения от дешёвых</b> (1–8 из 10)" in text
    assert "Сейчас актуально: <b>10</b> · самое дешёвое <b>20 000 сом</b>" in text
    assert text.index("20 000 сом") < text.index("27 000 сом")
    callbacks = [b.callback_data for row in rows for b in row]
    assert f"w:{token}" in callbacks and f"m:{token}:1" in callbacks
    assert ITEMS_PER_PAGE == 8


async def test_model_card_hides_fake_cheap_prices(maker):
    await seed(maker, [20000, 21000, 22000, 23000, 24000, 25000])
    await seed(maker, [2499], prefix="junk")  # цена чужой позиции из прайс-листа
    async with maker() as session:
        await recompute_stats(session, window_days=60)
        await session.commit()
    text, _ = await model_card(maker, "gpu:rtx-3060:12gb", page=0, params=DealParams())
    assert "самое дешёвое <b>20 000 сом</b>" in text
    assert "2 499" not in text


def test_key_subscription_matching():
    from lalascaner.db.models import Item

    item = Item(canonical_key="gpu:rtx-3060:12gb", family_key="gpu:rtx-3060", model="x", category="gpu")
    assert item_matches_query(item, "key:gpu:rtx-3060:12gb")
    assert item_matches_query(item, "key:gpu:rtx-3060")
    assert not item_matches_query(item, "key:gpu:rtx-3060-ti")
