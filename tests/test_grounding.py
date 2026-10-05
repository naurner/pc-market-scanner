"""Ответ ИИ сверяется с текстом поста: выдуманные цены и модели не попадают в базу и в уведомления."""

import pytest
from sqlalchemy import select

from lalascaner.ai.grounding import ground_result, price_grounded, text_numbers
from lalascaner.ai.schema import ExtractedItem, PostResult
from lalascaner.db import repo
from lalascaner.db.models import Item, Listing
from lalascaner.pipeline import grounding_text

REAL_FALSE_DEAL = "Ребята есть 6700xt 12 gb какая цена ей хочу продать и обновится"


def gpu(model, chip, price, vram=12):
    return ExtractedItem(category="gpu", model=model, chip=chip, vram_gb=vram, price=price, currency="KGS", condition="used")


def result(*items):
    return PostResult(post_id="1", intent="sell", items=list(items))


@pytest.mark.parametrize(
    "text,present,absent",
    [
        ("цена 9к", {9000}, set()),
        ("есть 6700xt 12 gb", set(), {6700, 12, 12000}),  # номер модели и объём — не цены
        ("25 000 сом", {25000}, set()),
        ("i5 13400 - 13.500", {13500}, set()),
        ("3060 за 25", {25000}, set()),
        ("клавиатура 2000с", {2000}, set()),
        ("650вт 80+ bronze, 8 ядер, 27\" 144hz", set(), {650, 8, 27, 144, 27000}),
        ("Corsair 2000d-4500", {4500}, {90000}),
    ],
)
def test_text_numbers(text, present, absent):
    found = text_numbers(text)
    assert present <= found and not (absent & found)


def test_real_false_deal_is_dropped_entirely():
    grounded = ground_result(result(gpu("RTX 3070 XT", "RTX 3070", 12000)), REAL_FALSE_DEAL)
    assert grounded.items == []


def test_hallucinated_price_is_removed_but_item_kept():
    grounded = ground_result(result(gpu("RTX 3060", "RTX 3060", 12000)), "Продаю rtx 3060 12gb, пишите в лс")
    assert len(grounded.items) == 1 and grounded.items[0].price is None and grounded.items[0].currency is None


def test_legit_answer_passes_unchanged():
    item = gpu("Palit RTX 3060 Dual", "RTX 3060", 25000)
    grounded = ground_result(result(item), "Palit rtx 3060 dual 12gb, цена 25к, торг")
    assert grounded.items == [item]


def test_lalafo_price_field_counts_as_text():
    listing = Listing(title="Видеокарта", text="RTX 3060 12GB, отличное состояние", price=21000, currency="KGS")
    assert price_grounded(21000, grounding_text(listing))
    assert not price_grounded(21000, f"{listing.title}\n{listing.text}")


async def test_old_ai_items_are_cleaned(maker):
    async with maker() as session:
        bad = Listing(source="telegram", external_id="g/1", source_ref="g", text=REAL_FALSE_DEAL, classifier="local", status="done")
        half = Listing(source="telegram", external_id="g/2", source_ref="g", text="rtx 3060 12gb продам", classifier="local", status="done")
        rules = Listing(source="telegram", external_id="g/3", source_ref="g", text="текст без модели", classifier="rules", status="done")
        session.add_all([bad, half, rules])
        await session.flush()
        for listing, chip in ((bad, "rtx-3070"), (half, "rtx-3060"), (rules, "rtx-4090")):
            session.add(
                Item(listing_id=listing.id, category="gpu", model=chip, canonical_key=f"gpu:{chip}:12gb", family_key=f"gpu:{chip}",
                     price=12000, currency="KGS", price_kgs=12000, is_deal=True, deal_kind="model")
            )
        await session.commit()
        assert await repo.ground_ai_items(session, grounding_text) == (1, 1)
        await session.commit()
        items = {i.canonical_key: i for i in await session.scalars(select(Item))}
    assert "gpu:rtx-3070:12gb" not in items  # выдуманная модель удалена
    fixed = items["gpu:rtx-3060:12gb"]
    assert fixed.price_kgs is None and not fixed.is_deal and fixed.attrs["ungrounded_price"] == 12000
    assert items["gpu:rtx-4090:12gb"].price_kgs == 12000  # результаты правил не трогаем
