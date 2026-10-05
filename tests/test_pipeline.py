from datetime import timedelta

import pytest
from sqlalchemy import select

from lalascaner.ai.schema import ExtractedItem, PostResult
from lalascaner.config import Settings
from lalascaner.db import repo
from lalascaner.db.models import Listing, utcnow
from lalascaner.db.session import init_db, make_engine, make_sessionmaker
from lalascaner.pipeline import Pipeline
from lalascaner.sources.base import RawListing
from lalascaner.sources.lalafo import FeedPage
from lalascaner.sources.telegram_web import MISSING


def lalafo_raw(ext_id, text, price=9900, category="4379", promoted=False, title="Процессор для ПК, Новый"):
    return RawListing(
        source="lalafo", external_id=str(ext_id), source_ref=category, url=f"https://lalafo.kg/ads/{ext_id}",
        title=title, text=text, price=price, currency="KGS", author_id=f"u{ext_id}",
        posted_at=utcnow() - timedelta(hours=1), is_promoted=promoted,
    )


def tg_raw(msg_id, text, author="seller", group="compshop_kg", hours_ago=1, topic_id=None):
    return RawListing(
        source="telegram", external_id=f"{group}/{msg_id}", source_ref=group, url=f"https://t.me/{group}/{msg_id}",
        text=text, author_id=author, posted_at=utcnow() - timedelta(hours=hours_ago), topic_id=topic_id,
    )


class FakeLalafo:
    def __init__(self, pages: dict[int, list[list[RawListing]]]):
        self.pages = pages
        self.calls = []

    async def fetch_page(self, category_id, page=1, per_page=40):
        self.calls.append((category_id, page))
        pages = self.pages.get(category_id, [[]])
        listings = pages[page - 1] if page <= len(pages) else []
        return FeedPage(listings=listings, page=page, page_count=len(pages), total=0)

    async def close(self):
        pass


class FakeTelegram:
    def __init__(self, messages: dict[int, RawListing | None]):
        self.messages = messages

    async def fetch_group_message(self, username, msg_id):
        if msg_id not in self.messages:
            return MISSING
        return self.messages[msg_id]

    async def scan_group(self, username, start_id, max_messages=400, miss_limit=40, window=8, until_id=None):
        found, last, misses, cur = {}, start_id - 1, 0, start_id
        while cur < start_id + max_messages and (misses < miss_limit or (until_id and cur <= until_id)):
            res = self.messages.get(cur, MISSING) if cur in self.messages else MISSING
            if res == MISSING:
                misses += 1
            else:
                misses, last = 0, cur
                if res is not None:
                    found.setdefault(res.external_id, res)
            cur += 1
        return list(found.values()), last

    async def latest_group_message_id(self, username):
        return max(self.messages)

    async def close(self):
        pass


class FakeClaude:
    def __init__(self):
        self.seen = []
        self.usage = type("U", (), {"as_dict": lambda self: {"requests": 1}})()

    async def classify(self, posts):
        self.seen.extend(p.post_id for p in posts)
        results = {
            p.post_id: PostResult(
                post_id=p.post_id, intent="sell",
                items=[ExtractedItem(category="gpu", model="RTX 3060", chip="RTX 3060", vram_gb=12, price=15000, currency="KGS", condition="used")],
            )
            for p in posts
        }
        return results, {}


@pytest.fixture
async def maker(tmp_path):
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path / 'p.db'}")
    await init_db(engine)
    yield make_sessionmaker(engine)
    await engine.dispose()


def settings(**kw) -> Settings:
    base = dict(_env_file=None, classifier="rules", telegram_channels="", min_samples=3, deal_threshold=0.2)
    base.update(kw)
    return Settings(**base)


async def test_poll_lalafo_stops_on_page_without_new_listings(maker):
    page1 = [lalafo_raw(1, "i5 12400f", promoted=True), lalafo_raw(2, "ryzen 5 5600")]
    page2 = [lalafo_raw(3, "i7 12700k")]
    page3 = [lalafo_raw(4, "i9 14900k")]
    fake = FakeLalafo({1345: [page1, page2, page3]})
    pipeline = Pipeline(settings(), maker, lalafo=fake, telegram=FakeTelegram({}))
    # страница 2 уже известна → дальше не листаем
    async with maker() as session:
        await repo.upsert_listings(session, page2)
        await session.commit()
    new = await pipeline.poll_lalafo(max_pages=5)
    assert new == 2
    assert [c for c in fake.calls if c[0] == 1345] == [(1345, 1), (1345, 2)]


async def test_classify_rules_mode_prefilter_and_copy(maker):
    pipeline = Pipeline(settings(), maker, lalafo=FakeLalafo({}), telegram=FakeTelegram({}))
    async with maker() as session:
        await repo.upsert_listings(
            session,
            [
                lalafo_raw(10, "Новый, запечатанный Ryzen 5 5600", price=9900),
                tg_raw(1, "брат там космические спутники если надо сигнал быстрее отправят"),
                tg_raw(2, "Новый Ryzen 5 5600 box - 9900", author="a"),
                tg_raw(3, "Новый Ryzen 5 5600 box - 9900", author="a", hours_ago=30),
            ],
        )
        await session.commit()
    report = await pipeline.classify_pending(limit=10)
    assert report.processed == 4
    assert report.skipped == 1
    assert report.copied == 1
    assert report.by_rules == 2
    async with maker() as session:
        items = await repo.get_items(session, report.item_ids)
    assert {i.canonical_key for i in items} == {"cpu:ryzen-5-5600"}


async def test_blocked_topic_is_skipped_even_if_it_looks_like_an_offer(maker):
    pipeline = Pipeline(settings(telegram_blocked_topics="6460,6547"), maker, lalafo=FakeLalafo({}), telegram=FakeTelegram({}))
    async with maker() as session:
        await repo.upsert_listings(
            session,
            [
                tg_raw(1, "Продам rtx 3060 12gb, цена 25000", topic_id=6460),  # похоже на объявление, но Болтовня
                tg_raw(2, "Ищу материнку MSI B660, куплю или обменяю", topic_id=6547),  # Ищу/Обмен
                tg_raw(3, "Продам rx 6700 xt 12gb, цена 27000", topic_id=9999),  # обычная тема — не трогаем
                tg_raw(4, "Продам i5 12400f, цена 11000"),  # без темы вообще — не трогаем
            ],
        )
        await session.commit()
    report = await pipeline.classify_pending(limit=10)
    assert report.skipped == 2 and report.by_rules == 2
    async with maker() as session:
        rows = {l.external_id: (l.status, l.classifier) for l in await session.scalars(select(Listing))}
    assert rows["compshop_kg/1"] == ("done", "topic")
    assert rows["compshop_kg/2"] == ("done", "topic")
    assert rows["compshop_kg/3"] == ("done", "rules")
    assert rows["compshop_kg/4"] == ("done", "rules")


async def test_refresh_telegram_topics_backfills_old_listings_and_hides_blocked(maker):
    old_no_topic = tg_raw(10, "Продаю видеокарту, недорого")  # как до появления TELEGRAM_BLOCKED_TOPICS
    telegram = FakeTelegram({10: tg_raw(10, "Продаю видеокарту, недорого", topic_id=6460)})
    pipeline = Pipeline(settings(telegram_blocked_topics="6460"), maker, lalafo=FakeLalafo({}), telegram=telegram)
    async with maker() as session:
        await repo.upsert_listings(session, [old_no_topic])
        await session.commit()

    checked, hidden = await pipeline.refresh_telegram_topics()
    assert (checked, hidden) == (1, 1)
    async with maker() as session:
        listing = await session.scalar(select(Listing))
    assert listing.topic_id == 6460 and listing.status == "done" and listing.classifier == "topic"

    # Второй прогон ничего не находит — topic_id уже проставлен, к Telegram больше не ходим.
    telegram.messages.clear()
    assert await pipeline.refresh_telegram_topics() == (0, 0)


async def test_refresh_telegram_topics_marks_no_topic_as_checked_not_null(maker):
    telegram = FakeTelegram({20: tg_raw(20, "Продаю видеокарту")})  # topic_id=None — обычный пост
    pipeline = Pipeline(settings(), maker, lalafo=FakeLalafo({}), telegram=telegram)
    async with maker() as session:
        await repo.upsert_listings(session, [tg_raw(20, "Продаю видеокарту")])
        await session.commit()

    assert await pipeline.refresh_telegram_topics() == (1, 0)
    async with maker() as session:
        listing = await session.scalar(select(Listing))
    assert listing.topic_id == 0  # «проверено, темы нет» — не NULL, повторно не переспросим

    telegram.messages.clear()
    assert await pipeline.refresh_telegram_topics() == (0, 0)


async def test_copied_reposts_are_not_reevaluated_as_new_deals(maker):
    pipeline = Pipeline(settings(), maker, lalafo=FakeLalafo({}), telegram=FakeTelegram({}))
    async with maker() as session:
        await repo.upsert_listings(session, [tg_raw(1, "Новый Ryzen 5 5600 box - 9900", author="a")])
        await session.commit()
    first = await pipeline.classify_pending(limit=10)
    async with maker() as session:
        await repo.upsert_listings(session, [tg_raw(2, "Новый Ryzen 5 5600 box - 9900", author="a")])
        await session.commit()
    emitted = []

    async def on_items(ids):
        emitted.extend(ids)

    pipeline.on_items = on_items
    second = await pipeline.classify_pending(limit=10)
    assert first.by_rules == 1 and second.copied == 1
    assert emitted == []


async def test_notifications_held_back_while_big_queue_is_processed(maker):
    pipeline = Pipeline(settings(notify_warmup_pending=1), maker, lalafo=FakeLalafo({}), telegram=FakeTelegram({}))
    emitted = []

    async def on_items(ids):
        emitted.extend(ids)

    pipeline.on_items = on_items
    async with maker() as session:
        await repo.upsert_listings(session, [lalafo_raw(i, f"Новый Ryzen 5 5600 box #{i}", price=9900 + i) for i in range(5)])
        await session.commit()
    await pipeline.classify_pending(limit=2)  # в очереди остаётся 3 > 1
    assert emitted == [] and pipeline.warming_up
    await pipeline.classify_pending(limit=10)  # очередь пуста
    assert emitted  # теперь уведомления идут


async def test_hybrid_sends_only_unconfident_to_ai(maker):
    claude = FakeClaude()
    pipeline = Pipeline(
        settings(classifier="hybrid", ai_enabled=True), maker, lalafo=FakeLalafo({}), telegram=FakeTelegram({}), claude=claude
    )
    async with maker() as session:
        res = await repo.upsert_listings(
            session,
            [
                tg_raw(1, "I5-8500 \n4000 сом"),
                tg_raw(2, "gigabyte h510m v2\ni5 11400\n\nцена 12к"),
            ],
        )
        await session.commit()
    report = await pipeline.classify_pending(limit=10)
    assert report.by_rules == 1 and report.by_ai == 1
    assert claude.seen == [str(res.new_ids[1])]


async def test_deals_emitted_only_for_fresh_listings(maker):
    pipeline = Pipeline(settings(deal_mode="median"), maker, lalafo=FakeLalafo({}), telegram=FakeTelegram({}))
    async with maker() as session:
        market = [lalafo_raw(100 + i, f"видеокарта rtx 3060 12gb #{i}", price=p, category="1355", title="Видеокарта, Б/у")
                  for i, p in enumerate([20000, 21000, 22000, 23000])]
        await repo.upsert_listings(session, market)
        await session.commit()
    await pipeline.classify_pending(limit=10)
    await pipeline.recompute()

    emitted = []

    async def on_items(ids):
        emitted.extend(ids)

    pipeline.on_items = on_items
    old = lalafo_raw(200, "видеокарта rtx 3060 12gb дёшево", price=14000, category="1355", title="Видеокарта, Б/у")
    old.posted_at = utcnow() - timedelta(days=10)
    fresh = lalafo_raw(201, "видеокарта rtx 3060 12gb срочно", price=15000, category="1355", title="Видеокарта, Б/у")
    async with maker() as session:
        await repo.upsert_listings(session, [old, fresh])
        await session.commit()
    report = await pipeline.classify_pending(limit=10)
    assert len(report.deal_ids) == 2
    async with maker() as session:
        emitted_items = await repo.get_items(session, emitted)
    assert [i.listing.external_id for i in emitted_items] == ["201"]


async def test_group_poll_initial_backfill_and_cursor(maker):
    messages = {i: None for i in range(1, 60)}
    messages[50] = tg_raw(50, "Новый Ryzen 5 5600 box - 9900")
    tg = FakeTelegram(messages)
    pipeline = Pipeline(settings(telegram_backfill_messages=20), maker, lalafo=FakeLalafo({}), telegram=tg)
    async with maker() as session:
        await repo.add_channel(session, "compshop_kg", "group", "Test")
        await session.commit()
    assert await pipeline.poll_telegram() == 1
    async with maker() as session:
        assert (await repo.get_channel(session, "compshop_kg")).last_message_id == 59

    # дыра из удалённых сообщений: 60..99 нет, 100 есть → курсор сдвигается вперёд
    tg.messages[100] = None
    assert await pipeline.poll_telegram() == 0
    async with maker() as session:
        assert (await repo.get_channel(session, "compshop_kg")).last_message_id == 99
