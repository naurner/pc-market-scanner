"""Пересчёт рыночных цен по каждой модели."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import timedelta

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from lalascaner.ai.normalize import pretty_key
from lalascaner.db.models import Item, Listing, PriceStat, utcnow

log = logging.getLogger(__name__)


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("empty")
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


@dataclass
class Summary:
    n: int
    median: float
    p25: float
    p75: float
    min: float
    max: float


def summarize(prices: list[float], iqr_k: float = 1.5, min_for_filter: int = 5) -> Summary | None:
    values = sorted(p for p in prices if p and p > 0)
    if not values:
        return None
    if len(values) >= min_for_filter:
        q1, q3 = percentile(values, 0.25), percentile(values, 0.75)
        spread = q3 - q1
        low, high = q1 - iqr_k * spread, q3 + iqr_k * spread
        filtered = [v for v in values if low <= v <= high]
        values = filtered or values
    return Summary(
        n=len(values),
        median=percentile(values, 0.5),
        p25=percentile(values, 0.25),
        p75=percentile(values, 0.75),
        min=values[0],
        max=values[-1],
    )


async def recompute_stats(session: AsyncSession, window_days: int) -> int:
    cutoff = utcnow() - timedelta(days=window_days)
    rows = (
        await session.execute(
            select(
                Item.canonical_key,
                Item.family_key,
                Item.category,
                Item.condition,
                Item.price_kgs,
                Listing.author_id,
                Listing.content_hash,
                Listing.changed_at,
            )
            .join(Listing)
            .where(
                Item.intent == "sell",
                Item.price_kgs.is_not(None),
                Item.is_bundle.is_(False),
                Item.condition != "broken",
                Listing.last_seen_at >= cutoff,
            )
            .order_by(Listing.changed_at.desc())
        )
    ).all()

    # Один продавец, повторяющий одно и то же объявление, — это одна точка данных (берём самое свежее).
    seen: set[tuple] = set()
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    categories: dict[str, str] = {}
    for canonical, family, category, condition, price, author, content_hash, _changed in rows:
        seller = author or content_hash
        dedup_key = (canonical, seller, condition)
        if dedup_key in seen:
            continue
        seen.add(dedup_key)
        for key in {canonical, family}:
            categories[key] = category
            groups[(key, "all")].append(price)
            if condition in ("new", "used"):
                groups[(key, condition)].append(price)

    now = utcnow()
    stats = []
    for (key, condition), prices in groups.items():
        summary = summarize(prices)
        if summary is None:
            continue
        stats.append(
            PriceStat(
                key=key,
                condition=condition,
                category=categories[key],
                title=pretty_key(key)[:256],
                n=summary.n,
                median=round(summary.median, 2),
                p25=round(summary.p25, 2),
                p75=round(summary.p75, 2),
                min=summary.min,
                max=summary.max,
                updated_at=now,
            )
        )

    await session.execute(delete(PriceStat))
    session.add_all(stats)
    await session.flush()
    log.info("статистика пересчитана: %d ключей из %d позиций", len(stats), len(rows))
    return len(stats)
