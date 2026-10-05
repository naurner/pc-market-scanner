"""Поиск выгодных объявлений.

Основной режим (cheapest): объявление — сделка, если оно дешевле САМОГО ДЕШЁВОГО из активных сейчас
объявлений той же модели (RTX 3060 12GB) или той же версии (Palit Dual RTX 3060 12GB).
В сравнении не участвуют: объявления того же продавца, репосты, снятые/проданные объявления,
фейково дешёвые цены. Новый товар сравнивается только с новыми, б/у — со всеми.

Режим median оставлен как запасной: сделка — на DEAL_THRESHOLD ниже медианы рынка.
Медиана в любом режиме сохраняется в позиции как контекст («рынок: 26 200 сом»).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy import and_, func, or_, select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from lalascaner.db.models import Item, Listing, PriceStat, utcnow

TRUSTED_CONFIDENCE = 0.8
# Уведомлять только о позициях, разобранных уверенно (ИИ или однозначное объявление для правил).
ALERT_MIN_CONFIDENCE = 0.7


@dataclass
class DealParams:
    min_samples: int = 5
    threshold: float = 0.2
    suspicious_ratio: float = 0.35
    # Режим median: экономия меньше этой суммы — не повод для уведомления.
    min_saving_kgs: float = 1500
    mode: str = "cheapest"
    min_competitors: int = 3
    min_gap_kgs: float = 0
    min_gap_percent: float = 3
    variant_alerts: bool = True
    lalafo_active_hours: int = 14
    other_active_days: int = 7

    @classmethod
    def from_settings(cls, s) -> DealParams:  # noqa: ANN001
        return cls(
            min_samples=s.min_samples,
            threshold=s.deal_threshold,
            suspicious_ratio=s.suspicious_ratio,
            min_saving_kgs=s.deal_min_saving_kgs,
            mode=s.deal_mode,
            min_competitors=s.deal_min_competitors,
            min_gap_kgs=s.deal_min_gap_kgs,
            min_gap_percent=s.deal_min_gap_percent,
            variant_alerts=s.deal_variant_alerts,
            lalafo_active_hours=s.lalafo_active_hours,
            other_active_days=s.telegram_active_days,
        )

    def active_clause(self, now: datetime | None = None):  # noqa: ANN201
        now = now or utcnow()
        return or_(
            and_(Listing.source == "lalafo", Listing.last_seen_at >= now - timedelta(hours=self.lalafo_active_hours)),
            and_(Listing.source != "lalafo", Listing.posted_at >= now - timedelta(days=self.other_active_days)),
        )


def _comparable_key(key: str) -> bool:
    """Слишком общие ключи не годятся для сравнения: «монитор 22"» без разрешения смешивает всё подряд."""
    if key.startswith("mon:"):
        return len(key.split(":")) >= 3
    return True


def reference_candidates(item: Item) -> list[tuple[str, str]]:
    conditions = [item.condition, "all"] if item.condition in ("new", "used") else ["all"]
    keys = [item.canonical_key] + ([item.family_key] if item.family_key != item.canonical_key else [])
    return [(key, cond) for key in keys if _comparable_key(key) for cond in conditions]


def pick_reference(item: Item, stats: dict[tuple[str, str], PriceStat], min_samples: int) -> PriceStat | None:
    for candidate in reference_candidates(item):
        stat = stats.get(candidate)
        if stat is not None and stat.n >= min_samples:
            return stat
    return None


# Готовые ПК и ноутбуки отличаются памятью/SSD/экраном: разница меньше 5% — это другая конфигурация, не выгода.
MIN_RELATIVE_GAP = {"pc": 0.05, "laptop": 0.05}


def _eligible(item: Item) -> bool:
    return item.price_kgs is not None and item.intent == "sell" and not item.is_bundle and item.condition != "broken"


def apply_market_context(item: Item, stat: PriceStat | None, params: DealParams) -> None:
    """Медиана рынка и пометка «подозрительно дёшево» — в любом режиме."""
    item.market_key = item.market_median = item.market_n = item.discount = None
    item.is_deal = item.suspicious = False
    item.deal_kind = item.ref_min_price = item.ref_count = None
    if stat is None or not _eligible(item) or stat.median <= 0:
        return
    item.market_key = f"{stat.key}|{stat.condition}"
    item.market_median = stat.median
    item.market_n = stat.n
    item.discount = round(1 - item.price_kgs / stat.median, 4)
    item.suspicious = item.price_kgs < stat.median * params.suspicious_ratio


def _trusted(item: Item) -> bool:
    # Подозрительно низкая цена от правил — скорее ошибка разбора, чем сделка; от ИИ — показываем с пометкой.
    return (item.confidence or 0) >= TRUSTED_CONFIDENCE or not item.suspicious


def median_deal(item: Item, params: DealParams) -> bool:
    if item.discount is None or item.market_median is None:
        return False
    saving = item.market_median - item.price_kgs
    return item.discount >= params.threshold and saving >= params.min_saving_kgs


async def _competitors(
    session: AsyncSession, item: Item, clause, author: str | None, params: DealParams, floor: float | None  # noqa: ANN001
) -> tuple[int, float | None]:
    stmt = (
        select(func.count(Item.id), func.min(Item.price_kgs))
        .join(Listing, Listing.id == Item.listing_id)
        .where(
            clause,
            Item.intent == "sell",
            Item.price_kgs.is_not(None),
            Item.is_bundle.is_(False),
            Item.condition != "broken",
            Item.listing_id != item.listing_id,
            or_(Listing.classifier.is_(None), Listing.classifier.not_like("copy:%")),
            params.active_clause(),
        )
    )
    if author:
        stmt = stmt.where(or_(Listing.author_id.is_(None), Listing.author_id != author))
    if item.condition == "new":
        stmt = stmt.where(Item.condition == "new")
    if floor:
        stmt = stmt.where(Item.price_kgs >= floor)
    count, min_price = (await session.execute(stmt)).one()
    return count or 0, min_price


async def cheapest_deal(
    session: AsyncSession, item: Item, author: str | None, params: DealParams
) -> tuple[str, float, int] | None:
    """(вид сделки, самая низкая цена среди активных, сколько их) или None."""
    floor = item.market_median * params.suspicious_ratio if item.market_median else None
    levels = [("model", Item.canonical_key == item.canonical_key)] if _comparable_key(item.canonical_key) else []
    if params.variant_alerts and item.variant_key:
        levels.append(("variant", Item.variant_key == item.variant_key))
    for kind, clause in levels:
        count, min_price = await _competitors(session, item, clause, author, params, floor)
        if count < params.min_competitors or min_price is None:
            continue
        gap = min_price - item.price_kgs
        relative = max(params.min_gap_percent / 100, MIN_RELATIVE_GAP.get(item.category, 0))
        min_gap = max(params.min_gap_kgs, min_price * relative)
        if gap > 0 and gap >= min_gap:
            return kind, min_price, count
    return None


async def evaluate_items(session: AsyncSession, items: list[Item], params: DealParams) -> list[Item]:
    """Проставляет рыночную оценку позициям. Возвращает те, что оказались сделками."""
    wanted_list = list({cand for item in items for cand in reference_candidates(item)})
    stats: dict[tuple[str, str], PriceStat] = {}
    for i in range(0, len(wanted_list), 300):
        chunk = wanted_list[i : i + 300]
        rows = await session.scalars(select(PriceStat).where(tuple_(PriceStat.key, PriceStat.condition).in_(chunk)))
        stats.update({(row.key, row.condition): row for row in rows})

    authors: dict[int, str | None] = {}
    listing_ids = list({item.listing_id for item in items})
    for i in range(0, len(listing_ids), 500):
        rows = await session.execute(select(Listing.id, Listing.author_id).where(Listing.id.in_(listing_ids[i : i + 500])))
        authors.update(dict(rows.all()))

    deals = []
    for item in items:
        apply_market_context(item, pick_reference(item, stats, params.min_samples), params)
        if not _eligible(item) or not _trusted(item) or (item.confidence or 0) < ALERT_MIN_CONFIDENCE:
            continue
        if params.mode == "median":
            if median_deal(item, params):
                item.is_deal, item.deal_kind = True, "median"
        elif found := await cheapest_deal(session, item, authors.get(item.listing_id), params):
            item.is_deal = True
            item.deal_kind, item.ref_min_price, item.ref_count = found
        if item.is_deal:
            deals.append(item)
    await session.flush()
    return deals
