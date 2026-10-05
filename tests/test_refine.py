"""Медленный локальный ИИ не блокирует базу: правила разбирают всё сразу, ИИ дорабатывает сомнительное в фоне."""

from datetime import timedelta

import pytest
from sqlalchemy import select

from lalascaner.ai.claude import AIUnavailable
from lalascaner.ai.schema import ExtractedItem, PostResult
from lalascaner.config import Settings
from lalascaner.db import repo
from lalascaner.db.models import Item, Listing, utcnow
from lalascaner.db.session import init_db, make_engine, make_sessionmaker
from lalascaner.pipeline import Pipeline
from lalascaner.scheduler import plan_jobs
from lalascaner.sources.base import RawListing

# Ноутбук с кучей цифр: правила разбирают в одну позицию, но неуверенно (confident=False)
LAPTOP = (
    "Продам Игровой ноутбук \n•ASUS TUF Gaming a15\n•Процессор-мощный Ryzen 7-6800H 8-Ядер/16 Потоков\n"
    "•Видеокарта-RTX 3060 6gb\n•ОЗУ-16гб DDR5\n•SSD-512\n•Экран 15.6 FullHD-IPS<144hz>\n60000"
)
# Однозначное объявление: правилам верим сразу
CLEAR = "Новый Ryzen 5 5600 box - 9900"


class FakeLocalAI:
    kind = "local"
    supports_batch = False
    background_refine = True

    def __init__(self, mode: str = "ok"):
        self.mode = mode
        self.calls = 0
        self.usage = type("U", (), {"as_dict": lambda self: {"requests": 1}})()

    async def classify(self, posts):
        self.calls += 1
        if self.mode == "unavailable":
            raise AIUnavailable("нет соединения")
        if self.mode == "error":
            return {}, {p.post_id: "модель не вернула результат для поста" for p in posts}
        results = {
            p.post_id: PostResult(
                post_id=p.post_id, intent="sell",
                items=[ExtractedItem(category="laptop", model="ASUS TUF A15", cpu_model="Ryzen 7 6800H", gpu_model="RTX 3060",
                                     price=60000, currency="KGS", condition="used")],
            )
            for p in posts
        }
        return results, {}

    async def health(self):
        return (self.mode != "unavailable", "ok")


def make(**kw) -> Settings:
    kw = {"ai_provider": "ollama", **kw}
    return Settings(_env_file=None, ai_enabled=True, classifier="hybrid", **kw)


@pytest.fixture
async def maker(tmp_path):
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path / 'r.db'}")
    await init_db(engine)
    yield make_sessionmaker(engine)
    await engine.dispose()


def tg(msg_id: int, text: str) -> RawListing:
    return RawListing(
        source="telegram", external_id=f"g/{msg_id}", source_ref="g", url=f"https://t.me/g/{msg_id}", text=text,
        author_id="a", posted_at=utcnow() - timedelta(hours=1),
    )


def lalafo(ext_id: int, days_old: float, text: str = "Отличная видеокарта, пишите") -> RawListing:
    return RawListing(
        source="lalafo", external_id=str(ext_id), source_ref="1355", url=f"https://lalafo.kg/{ext_id}",
        title="Видеокарта для ПК", text=text, price=5000, currency="KGS", author_id=f"u{ext_id}",
        posted_at=utcnow() - timedelta(days=days_old),
    )


async def statuses(maker) -> dict[str, str]:
    async with maker() as session:
        return {l.external_id: l.status for l in await session.scalars(select(Listing))}


async def add(maker, raws):
    async with maker() as session:
        await repo.upsert_listings(session, raws)
        await session.commit()


async def test_rules_result_is_saved_immediately_and_ai_is_not_awaited(maker):
    ai = FakeLocalAI()
    pipeline = Pipeline(make(), maker, claude=ai)
    await add(maker, [tg(1, LAPTOP), tg(2, CLEAR)])
    report = await pipeline.classify_pending(limit=10)

    assert ai.calls == 0  # быстрый путь не ждёт медленную модель
    assert report.by_rules == 2 and report.by_ai == 0
    assert await statuses(maker) == {"g/1": "refine", "g/2": "done"}
    async with maker() as session:
        items = list(await session.scalars(select(Item)))
        assert {i.category for i in items} == {"laptop", "cpu"} and {i.confidence for i in items} == {0.55, 0.75}
        assert await repo.count_pending(session) == 0  # очередь «новых» пуста — уведомления не придерживаются


async def test_only_fresh_lalafo_is_worth_the_ai_time(maker):
    pipeline = Pipeline(make(ai_refine_max_age_days=3), maker, claude=FakeLocalAI())
    await add(maker, [lalafo(1, days_old=0.5), lalafo(2, days_old=30)])
    await pipeline.classify_pending(limit=10)
    assert await statuses(maker) == {"1": "refine", "2": "done"}


async def test_refine_upgrades_rules_result_with_ai(maker):
    ai = FakeLocalAI()
    pipeline = Pipeline(make(), maker, claude=ai)
    await add(maker, [tg(1, LAPTOP)])
    await pipeline.classify_pending(limit=10)

    report = await pipeline.refine_pending()
    assert ai.calls == 1 and report.by_ai == 1
    assert await statuses(maker) == {"g/1": "done"}
    async with maker() as session:
        item = await session.scalar(select(Item))
        listing = await session.scalar(select(Listing))
    assert item.confidence == 0.75 and listing.classifier == "local"
    await pipeline.refine_pending()
    assert ai.calls == 1  # очередь пуста — модель не дёргаем


async def test_telegram_is_refined_before_lalafo(maker):
    ai = FakeLocalAI()
    pipeline = Pipeline(make(ai_batch_size=1), maker, claude=ai)
    await add(maker, [lalafo(1, days_old=0.1), tg(2, LAPTOP)])
    await pipeline.classify_pending(limit=10)
    async with maker() as session:
        order = [l.external_id for l in await repo.refine_listings(session, 10)]
    assert order == ["g/2", "1"]


async def test_ai_outage_keeps_rules_result_and_retries_later(maker):
    ai = FakeLocalAI(mode="unavailable")
    pipeline = Pipeline(make(), maker, claude=ai)
    await add(maker, [tg(1, LAPTOP)])
    await pipeline.classify_pending(limit=10)

    await pipeline.refine_pending()
    assert await statuses(maker) == {"g/1": "refine"}  # не потеряно и не «failed»
    async with maker() as session:
        assert (await session.scalar(select(Item))).confidence == 0.55  # результат правил на месте
    calls = ai.calls
    await pipeline.refine_pending()
    assert ai.calls == calls  # пока сервер «лежит», к нему не ходим

    ai.mode = "ok"
    pipeline._ai_down_until = 0.0
    await pipeline.refine_pending()
    assert await statuses(maker) == {"g/1": "done"}


async def test_model_failure_on_a_post_keeps_rules_result_as_final(maker):
    pipeline = Pipeline(make(), maker, claude=FakeLocalAI(mode="error"))
    await add(maker, [tg(1, LAPTOP)])
    await pipeline.classify_pending(limit=10)
    await pipeline.refine_pending()
    assert await statuses(maker) == {"g/1": "done"}
    async with maker() as session:
        assert await session.scalar(select(Item)) is not None


async def test_claude_style_classifier_still_blocks_on_ai(maker):
    class FastAI(FakeLocalAI):
        kind = "claude"
        background_refine = False

    ai = FastAI()
    pipeline = Pipeline(make(ai_provider="claude", anthropic_api_key="k"), maker, claude=ai)
    await add(maker, [tg(1, LAPTOP)])
    report = await pipeline.classify_pending(limit=10)
    assert ai.calls == 1 and report.by_ai == 1
    assert await statuses(maker) == {"g/1": "done"}


def test_scheduler_has_refine_job_only_for_background_classifiers():
    names = {name for name, *_ in plan_jobs(Pipeline(make(), sessionmaker=None, claude=FakeLocalAI()))}
    assert "ИИ-доработка" in names
    fast = FakeLocalAI()
    fast.background_refine = False
    names = {name for name, *_ in plan_jobs(Pipeline(make(), sessionmaker=None, claude=fast))}
    assert "ИИ-доработка" not in names
