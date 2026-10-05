"""Одно и то же объявление не должно приходить снова: ни после правки, ни после перезалива под новым номером.
Повтор — только если цену снизили заметно, с пометкой «было»."""

from datetime import timedelta

from sqlalchemy import select, text

from lalascaner.bot.notifier import Notifier
from lalascaner.config import Settings
from lalascaner.db import repo
from lalascaner.db.models import Alert, Item, Listing, utcnow
from lalascaner.sources.base import RawListing


class FakeBot:
    def __init__(self):
        self.sent = []

    async def send_message(self, chat_id, text, **kwargs):
        self.sent.append(text)


def raw(ext_id, price, text="RTX 3060 12GB Palit", author="seller1"):
    return RawListing(
        source="lalafo", external_id=str(ext_id), source_ref="1355", url=f"https://lalafo.kg/ads/{ext_id}",
        title="Видеокарта", text=text, price=price, currency="KGS", author_id=author,
        posted_at=utcnow() - timedelta(hours=1),
    )


async def store(maker, *raws):
    async with maker() as session:
        await repo.upsert_listings(session, list(raws))
        await session.commit()


async def deal(maker, ext_id) -> int:
    """Позиция-сделка для объявления — как будто его только что (пере)разобрали."""
    async with maker() as session:
        listing = await session.scalar(select(Listing).where(Listing.external_id == str(ext_id)))
        item = Item(
            listing_id=listing.id, category="gpu", intent="sell", model="RTX 3060", canonical_key="gpu:rtx-3060:12gb",
            family_key="gpu:rtx-3060", condition="used", price=listing.price, currency="KGS", price_kgs=listing.price,
            is_deal=True, discount=0.3, deal_kind="median", confidence=0.9,
        )
        session.add(item)
        await session.commit()
        return item.id


async def notifier(maker):
    bot = FakeBot()
    async with maker() as session:
        await repo.add_subscription(session, 1, query="", mode="deals")
        await session.commit()
    return Notifier(bot, maker, Settings(_env_file=None)), bot


async def make_old(maker, ext_id):
    async with maker() as session:
        listing = await session.scalar(select(Listing).where(Listing.external_id == str(ext_id)))
        listing.fetched_at = utcnow() - timedelta(days=10)
        await session.commit()


async def test_same_item_and_edited_listing_are_not_resent(maker):
    notify, bot = await notifier(maker)
    await store(maker, raw(1, 20000))
    item_id = await deal(maker, 1)
    await notify([item_id])
    await notify([item_id])  # пересчёт рынка каждые полчаса присылает те же позиции
    assert len(bot.sent) == 1

    await store(maker, raw(1, 20000, text="RTX 3060 12GB Palit, торг"))  # правка текста пересоздаёт позиции
    await notify([await deal(maker, 1)])
    await store(maker, raw(1, 19800, text="RTX 3060 12GB Palit, торг"))  # −1% — не повод
    await notify([await deal(maker, 1)])
    assert len(bot.sent) == 1


async def test_price_drop_is_resent_with_previous_price(maker):
    notify, bot = await notifier(maker)
    await store(maker, raw(1, 20000))
    await notify([await deal(maker, 1)])
    await store(maker, raw(1, 18000))
    await notify([await deal(maker, 1)])
    assert len(bot.sent) == 2
    assert bot.sent[1].startswith("📉 <b>Снизили цену</b>: было 20 000 сом")


async def test_repost_by_same_seller_is_skipped_other_seller_is_not(maker):
    notify, bot = await notifier(maker)
    await store(maker, raw(1, 20000))
    await notify([await deal(maker, 1)])
    await store(maker, raw(2, 20300, text="RTX 3060 12GB Palit срочно"))  # тот же продавец, новый номер
    await notify([await deal(maker, 2)])
    await store(maker, raw(3, 20000, author="other"))
    await notify([await deal(maker, 3)])
    assert len(bot.sent) == 2


async def test_reposts_detected_by_vectors_are_skipped(maker):
    notify, bot = await notifier(maker)
    await store(maker, raw(1, 20000), raw(2, 15000, author="другой-аккаунт"))
    await notify([await deal(maker, 1)])
    async with maker() as session:
        first = await session.scalar(select(Listing).where(Listing.external_id == "1"))
        second = await session.scalar(select(Listing).where(Listing.external_id == "2"))
        second.dup_of = first.id
        await session.commit()
    await notify([await deal(maker, 2)])
    assert len(bot.sent) == 1


async def test_old_listing_edit_is_news_only_when_price_drops(maker):
    notify, bot = await notifier(maker)
    await store(maker, raw(1, 20000))
    await make_old(maker, 1)
    await store(maker, raw(1, 20000, text="RTX 3060 12GB Palit, подниму"))
    await notify([await deal(maker, 1)])
    assert bot.sent == []  # висит 10 дней, просто отредактировали

    await store(maker, raw(1, 17000, text="RTX 3060 12GB Palit, подниму"))
    await notify([await deal(maker, 1)])
    assert len(bot.sent) == 1 and "было 20 000 сом" in bot.sent[0]


async def test_old_notifications_are_migrated(maker):
    await store(maker, raw(1, 20000))
    item_id = await deal(maker, 1)
    async with maker() as session:
        await session.execute(text("CREATE TABLE notifications (id INTEGER PRIMARY KEY, chat_id BIGINT, item_id INT, sent_at DATETIME)"))
        await session.execute(text(f"INSERT INTO notifications (chat_id, item_id, sent_at) VALUES (1, {item_id}, '2026-09-30 10:00:00')"))
        await session.commit()
        assert await repo.migrate_old_notifications(session) == 1
        await session.commit()
        assert await repo.migrate_old_notifications(session) == 0
        alert = await session.scalar(select(Alert))
    assert alert.canonical_key == "gpu:rtx-3060:12gb" and alert.author_id == "seller1" and alert.price_kgs == 20000

    notify, bot = await notifier(maker)
    await notify([item_id])
    assert bot.sent == []  # уже отправленное до обновления не приходит снова
