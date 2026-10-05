from datetime import timedelta
from types import SimpleNamespace

import pytest

from lalascaner.bot.formatting import deal_message, money, short_money
from lalascaner.bot.handlers import parse_watch_args
from lalascaner.bot.notifier import subscription_matches
from lalascaner.db.models import Item, Listing, Subscription, utcnow


@pytest.mark.parametrize(
    "args,expected",
    [
        ("all", ("", None, "deals")),
        ("все", ("", None, "deals")),
        ("4070 до 60000", ("4070", 60000.0, "deals")),
        ("rtx 3060 до 25к", ("rtx 3060", 25000.0, "deals")),
        ("new 5700x3d", ("5700x3d", None, "all")),
        ("new ddr4 16gb до 4 000 сом", ("ddr4 16gb", 4000.0, "all")),
    ],
)
def test_parse_watch_args(args, expected):
    assert parse_watch_args(args) == expected


def make_item(**kw):
    listing = Listing(
        source="lalafo", source_ref="1355", external_id="1", url="https://lalafo.kg/x", title="t",
        text="RTX 3060 12GB отличная", city="Бишкек", changed_at=utcnow() - timedelta(hours=2),
    )
    base = dict(
        category="gpu", intent="sell", model="Palit RTX 3060 Dual", canonical_key="gpu:rtx-3060:12gb",
        family_key="gpu:rtx-3060", condition="used", price=15000, currency="KGS", price_kgs=15000, quantity=1,
        is_bundle=False, is_deal=True, discount=0.3, market_median=21500, market_n=34,
        market_key="gpu:rtx-3060:12gb|used", suspicious=False,
    )
    base.update(kw)
    item = Item(**base)
    item.listing = listing
    return item


def sub(**kw):
    return Subscription(**{"chat_id": 1, "query": "", "mode": "deals", "max_price_kgs": None, "min_discount": None, **kw})


def test_subscription_matching():
    deal = make_item()
    assert subscription_matches(sub(), deal, 0.2)
    assert subscription_matches(sub(query="3060"), deal, 0.2)
    assert subscription_matches(sub(query="rtx 3060 12gb"), deal, 0.2)
    assert not subscription_matches(sub(query="3060 ti"), deal, 0.2)
    assert not subscription_matches(sub(max_price_kgs=10000), deal, 0.2)
    assert not subscription_matches(sub(min_discount=0.4), deal, 0.2)

    regular = make_item(is_deal=False, discount=0.05)
    assert not subscription_matches(sub(), regular, 0.2)
    assert subscription_matches(sub(mode="all", query="3060"), regular, 0.2)
    assert not subscription_matches(sub(mode="all", query="3060"), make_item(intent="buy"), 0.2)
    assert subscription_matches(sub(query="palit"), deal, 0.2)


def test_deal_message_cheapest_model():
    text = deal_message(make_item(deal_kind="model", ref_min_price=19000, ref_count=7, suspicious=True))
    assert "Самая низкая цена на RTX 3060 12GB" in text
    assert "15 000 сом" in text
    assert "Дешевле всех 7 актуальных объявлений «RTX 3060 12GB»" in text and "19 000 сом (−4 000 сом)" in text
    assert "Медиана рынка (б/у): 21 500 сом" in text and "Подозрительно" in text and "Бишкек" in text


def test_deal_message_variant_and_median():
    variant = make_item(deal_kind="variant", variant_key="gpu:rtx-3060:12gb|palit-dual", ref_min_price=23000, ref_count=2)
    text = deal_message(variant)
    assert "Самая низкая цена на версию RTX 3060 12GB · Palit Dual" in text
    assert "Дешевле всех 2 актуальных объявлений «RTX 3060 12GB · Palit Dual»" in text
    assert "На 30% ниже медианы" in deal_message(make_item(deal_kind="median"))
    assert deal_message(make_item(is_deal=False, discount=None)).startswith("🆕")


def test_money_formatting():
    assert money(1234567) == "1 234 567 сом"
    assert short_money(13500) == "13.5к" and short_money(105500) == "106к" and short_money(900) == "900"
    assert money(None) == "—"
    usd = deal_message(make_item(price=250, currency="USD", price_kgs=21862.5))
    assert "250 USD ≈ 21 862 сом" in usd or "250 USD ≈ 21 863 сом" in usd


def test_bot_app_builds():
    from lalascaner.bot.app import build_bot
    from lalascaner.config import Settings

    settings = Settings(_env_file=None, bot_token="123456:ABCdefGhIJKlmNoPQRsTUVwxyZ")
    pipeline = SimpleNamespace(on_items=None)
    bot, dp = build_bot(settings, sessionmaker=None, pipeline=pipeline)
    assert pipeline.on_items is not None
    assert dp["settings"] is settings


def test_old_lalafo_listing_shows_since_date():
    from lalascaner.bot.formatting import freshness

    old = Listing(source="lalafo", changed_at=utcnow() - timedelta(days=900))
    fresh = Listing(source="lalafo", changed_at=utcnow() - timedelta(hours=3))
    tg_old = Listing(source="telegram", changed_at=utcnow() - timedelta(days=40))
    assert freshness(old).startswith("висит с ")
    assert freshness(fresh) == "3 ч назад"
    assert freshness(tg_old) == "40 дн назад"


def test_deal_message_shows_buyer_interest_and_similar_estimate():
    item = make_item(market_median=None, market_n=None, est_price=19500, est_n=6)
    item.listing.meta = {"favorites": 4, "callers": 0, "writers": 2, "views": 10}
    text = deal_message(item)
    assert "❤️ в избранном 4 · 💬 писали 2" in text and "звонили" not in text
    assert "По 6 похожим объявлениям: около 19 500 сом" in text


def test_stats_text_reports_collected_history():
    from lalascaner.bot.handlers import stats_text
    from lalascaner.config import Settings

    stats = {
        "listings_by_source": {"lalafo": 10}, "listings_by_status": {}, "items_by_category": {"gpu": 3},
        "items_total": 3, "models_tracked": 1, "deals_24h": 0,
        "events_by_kind": {"price": 7, "bump": 2, "back": 1}, "gone": 5, "metrics": 40, "vectors": 9, "history_days": 3,
    }
    on = Settings(_env_file=None, ai_enabled=True, ai_provider="ollama")
    text = stats_text(on, stats, usage={}, batches=[])
    assert "Смен цены: 7" in text and "поднятий в ленте: 2" in text
    assert "Пропало с lalafo (продано/снято): 5 · вернулось: 1" in text and "Векторов объявлений: 9" in text
    assert "Векторов" not in stats_text(Settings(_env_file=None), stats, usage={}, batches=[])
