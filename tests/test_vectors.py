"""Векторы: индекс, оценка по соседям, перезаливы, похожие, поиск по смыслу, «за сколько продать».
Ollama подменяется детерминированным «мешком слов» — близкие тексты дают близкие векторы."""

import json
import re
import zlib
from datetime import timedelta

import httpx
import numpy as np
import pytest
import respx
from sqlalchemy import select

from lalascaner.ai.embed import EmbedModelMissing, EmbedUnavailable
from lalascaner.bot.insights import appraisal_text, nice_price
from lalascaner.config import Settings
from lalascaner.db import repo
from lalascaner.db.models import Item, Listing, ListingVector, PriceStat, utcnow
from lalascaner.market import vectors as vec
from lalascaner.pipeline import Appraisal, Pipeline
from lalascaner.sources.base import RawListing

DIM = 256


class FakeEmbedder:
    model = "fake"

    def __init__(self, missing=False, down=False):
        self.missing, self.down, self.pulled, self.calls = missing, down, 0, 0

    async def embed(self, texts):
        self.calls += 1
        if self.down:
            raise EmbedUnavailable("нет связи")
        if self.missing:
            raise EmbedModelMissing("нет модели")
        out = []
        for text in texts:
            v = [0.0] * DIM
            for token in re.findall(r"\w+", text.lower()):
                v[zlib.crc32(token.encode()) % DIM] += 1
            out.append(v)
        return out

    async def pull(self):
        self.pulled += 1
        self.missing = False

    async def aclose(self):
        pass


class Nothing:
    async def close(self):
        pass


def settings(**kw):
    return Settings(**{"_env_file": None, "classifier": "rules", "telegram_channels": "", "min_samples": 3, **kw})


BASE = "Видеокарта RTX 3060 12GB Palit Dual отличное состояние не майнила коробка документы гарантия магазин Бишкек самовывоз доставка возможна проверка на месте"


def gpu(ext_id, price, text=BASE, author=None, hours_ago=1):
    return RawListing(
        source="lalafo", external_id=str(ext_id), source_ref="1355", url=f"https://lalafo.kg/ads/{ext_id}",
        title="Видеокарта", text=text, price=price, currency="KGS", author_id=author or f"u{ext_id}",
        posted_at=utcnow() - timedelta(hours=hours_ago),
    )


async def seeded(maker, embedder=None):
    pipeline = Pipeline(settings(), maker, lalafo=Nothing(), telegram=Nothing(), embedder=embedder or FakeEmbedder())
    listings = [
        gpu(1, 20000),
        gpu(2, 19000, BASE.replace("коробка документы", "без коробки")),
        gpu(3, 21000, BASE.replace("гарантия магазин", "гарантии нет")),
        gpu(4, 22000, BASE.replace("самовывоз", "торг уместен")),
        gpu(5, 20000, BASE + " срочно"),  # перезалив первого: та же модель, цена и почти тот же текст
    ]
    async with maker() as session:
        await repo.upsert_listings(session, listings)
        await session.commit()
    await pipeline.classify_pending(limit=50)
    return pipeline


async def listing(maker, ext_id):
    async with maker() as session:
        return await session.scalar(select(Listing).where(Listing.external_id == str(ext_id)))


def test_pack_normalizes_and_index_ranks_and_replaces():
    index = vec.VectorIndex()
    a, b, c = (vec.unpack(vec.pack(v)) for v in ([3, 0, 0], [1, 1, 0], [0, 0, 5]))
    assert np.linalg.norm(a) == pytest.approx(1, abs=1e-3)
    for lid, v in ((1, a), (2, b), (3, c)):
        index.add(lid, v)
    assert [lid for lid, _ in index.search(a, k=2)] == [1, 2]
    assert [lid for lid, _ in index.search(a, k=2, exclude={1})] == [2, 3]
    index.add(3, a)  # тот же id — замена, а не дубль
    assert len(index) == 3 and index.search(a, k=1, exclude={1})[0][0] == 3


def test_estimate_uses_close_neighbors_weighted():
    assert vec.weighted_median([(100, 1), (200, 1), (300, 5)]) == 300
    assert vec.estimate_price([(100, 0.9), (200, 0.9)]) is None  # мало соседей
    assert vec.estimate_price([(100, 0.9), (200, 0.9), (9999, 0.2)]) is None  # далёкий не считается
    est = vec.estimate_price([(100, 0.9), (200, 0.85), (300, 0.8)])
    assert est.price == 200 and est.n == 3


def test_is_repost_needs_same_model_and_close_price():
    assert vec.is_repost(0.99, True, 20000, 19500)
    assert not vec.is_repost(0.99, True, 20000, 15000)
    assert not vec.is_repost(0.99, False, 20000, 20000)
    assert not vec.is_repost(0.9, True, 20000, 20000)


async def test_embed_pending_links_reposts_and_estimates(maker):
    pipeline = await seeded(maker)
    assert await pipeline.embed_pending(limit=100) == 5
    assert await pipeline.embed_pending(limit=100) == 0  # уже посчитано
    first, repost = await listing(maker, 1), await listing(maker, 5)
    assert repost.dup_of == first.id and first.dup_of is None
    async with maker() as session:
        item = await session.scalar(select(Item).where(Item.listing_id == first.id))
        stored = await session.scalar(select(ListingVector).where(ListingVector.listing_id == first.id))
    assert item.est_n >= 3 and 19000 <= item.est_price <= 22000
    assert stored.model == "fake" and len(stored.vec) == DIM * 2  # float16

    fresh = Pipeline(settings(), maker, lalafo=Nothing(), telegram=Nothing(), embedder=FakeEmbedder())
    similar = await fresh.similar_items(first.id)  # векторы подгружаются из БД после перезапуска
    assert similar and first.id not in {i.listing_id for i in similar}
    found = await fresh.semantic_search("видеокарта rtx 3060 palit")
    assert found and all(i.category == "gpu" for i in found)


async def test_repost_disappearance_is_not_counted_as_sale(maker):
    pipeline = await seeded(maker)
    await pipeline.embed_pending(limit=100)
    async with maker() as session:
        reposted = await repo.reposted_listing_ids(session)
    assert (await listing(maker, 1)).id in reposted


async def test_missing_model_is_pulled_once_and_outage_pauses(maker):
    embedder = FakeEmbedder(missing=True)
    pipeline = await seeded(maker, embedder)
    assert await pipeline.embed_pending(limit=100) == 5
    assert embedder.pulled == 1

    down = FakeEmbedder(down=True)
    paused = await seeded_again(maker, down)
    assert await paused.embed_pending() == 0
    calls = down.calls
    assert await paused.embed_pending() == 0 and down.calls == calls  # пауза — Ollama не дёргаем


async def seeded_again(maker, embedder):
    pipeline = Pipeline(settings(), maker, lalafo=Nothing(), telegram=Nothing(), embedder=embedder)
    async with maker() as session:
        await repo.upsert_listings(session, [gpu(99, 30000, "Видеокарта RTX 4070 новая")])
        await session.commit()
    await pipeline.classify_pending(limit=50)
    return pipeline


async def test_appraise_uses_market_then_falls_back_to_similar(maker):
    pipeline = await seeded(maker)
    await pipeline.embed_pending(limit=100)
    await pipeline.recompute()
    known = await pipeline.appraise("rtx 3060 12gb б/у")
    assert known.stat is not None and 19000 <= known.reference <= 22000
    assert known.category == "gpu" and known.active_n >= 3 and known.cheapest == 19000

    vague = await pipeline.appraise("видеокарта palit dual отличное состояние не майнила коробка документы гарантия магазин")
    assert vague.stat is None and vague.estimate is not None and vague.category == "gpu"


async def test_appraise_without_vectors_and_unknown_model(maker):
    pipeline = Pipeline(settings(), maker, lalafo=Nothing(), telegram=Nothing())
    appraisal = await pipeline.appraise("что-то непонятное")
    assert appraisal.reference is None
    assert "Не нашёл такую модель" in appraisal_text(appraisal)


def test_appraisal_text_shows_tiers_with_speed():
    stat = PriceStat(key="gpu:rtx-3060:12gb", condition="used", category="gpu", title="RTX 3060 12GB", n=40,
                     median=20400, p25=18000, p75=23000, min=12000, max=30000)
    liquidity = {
        "gpu": {"cheap": {"p7": 0.42, "median_days": 9.0}, "market": {"p7": 0.29, "median_days": None}},
        "all": {"high": {"p3": 0.1, "p7": None, "median_days": None}},
    }
    text = appraisal_text(Appraisal(query="3060", title="RTX 3060 12GB", category="gpu", condition="used",
                                    stat=stat, active_n=12, cheapest=17500, liquidity=liquidity))
    assert "⚡ Быстро — <b>18 400 сом</b> (-10%): за неделю уходит ~42%, обычно за ~9 дн." in text
    assert "⚖️ По рынку — <b>20 500 сом</b>: за неделю уходит ~29%" in text
    assert "🐢 Подороже — <b>23 500 сом</b> (+15%): за 3 дня уходит ~10% <i>(по всему рынку)</i>" in text
    assert "Сейчас продаётся: 12, самое дешёвое 17 500 сом" in text and "обычно 18 000–23 000 сом" in text
    assert "Скорость — по категории «" in text


def test_nice_price():
    assert nice_price(23460) == 23500 and nice_price(18360) == 18400 and nice_price(987) == 990


@respx.mock
async def test_ollama_embedder_protocol():
    from lalascaner.ai.embed import OllamaEmbedder

    embed = respx.post("http://ollama:11434/api/embed")
    embed.side_effect = [
        httpx.Response(404, json={"error": 'model "bge-m3" not found, try pulling it first'}),
        httpx.Response(200, json={"embeddings": [[0.1, 0.2], [0.3, 0.4]]}),
        httpx.Response(500, text="out of memory"),
    ]
    pull = respx.post("http://ollama:11434/api/pull").mock(return_value=httpx.Response(200, json={"status": "success"}))
    client = OllamaEmbedder(base_url="http://ollama:11434/", model="bge-m3")
    try:
        with pytest.raises(EmbedModelMissing):
            await client.embed(["a", "b"])
        await client.pull()
        assert await client.embed(["a", "b"]) == [[0.1, 0.2], [0.3, 0.4]]
        with pytest.raises(EmbedUnavailable, match="500"):
            await client.embed(["a"])
    finally:
        await client.aclose()
    assert json.loads(embed.calls[1].request.content) == {"model": "bge-m3", "input": ["a", "b"]}
    assert json.loads(pull.calls[0].request.content) == {"model": "bge-m3", "stream": False}
