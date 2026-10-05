"""Скорость продажи в зависимости от цены относительно рынка.

Считается по истории lalafo: объявление появилось → пропало из ленты (продано или снято) либо ещё висит.
Висящие ещё не «проданы», но и не «непродаваемы» — про них известно только, что за N дней не ушли.
Такие наблюдения корректно учитывает кривая выживаемости Каплана–Мейера: доля «ушедших к дню t».
Перезаливы (пропало и тут же выложено заново) — не продажа, а продолжение того же наблюдения.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

# (название, от, до) — цена / медиана рынка. Дешевле 0.75 — чаще ошибка распознавания или обман, не берём.
BUCKETS = (
    ("cheap", 0.75, 0.93),
    ("market", 0.93, 1.07),
    ("high", 1.07, 1.25),
    ("very_high", 1.25, 3.0),
)
BUCKET_RU = {"cheap": "дешевле рынка", "market": "по рынку", "high": "дороже рынка", "very_high": "сильно дороже"}
HORIZONS = (3, 7, 14)
# По всему рынку хватает и 15 объявлений в корзине; внутри категории меньше 30 — уже шум
# (на реальных данных «дороже рынка» по памяти «уходило» быстрее, чем «по рынку», на 15 объявлениях).
MIN_OBSERVATIONS = 15
MIN_CATEGORY_OBSERVATIONS = 30


def bucket_of(ratio: float) -> str | None:
    for name, low, high in BUCKETS:
        if low <= ratio < high:
            return name
    return None


@dataclass
class Observation:
    days: float
    sold: bool


def kaplan_meier(observations: list[Observation]) -> list[tuple[float, float]]:
    """[(день, доля ещё не проданных)] — ступенчатая кривая по моментам продаж."""
    at_risk = len(observations)
    survival = 1.0
    curve: list[tuple[float, float]] = []
    by_time: dict[float, list[bool]] = defaultdict(list)
    for obs in observations:
        by_time[obs.days].append(obs.sold)
    for t in sorted(by_time):
        events = by_time[t]
        sold = sum(events)
        if sold and at_risk:
            survival *= 1 - sold / at_risk
            curve.append((t, survival))
        at_risk -= len(events)
    return curve


def survival_at(curve: list[tuple[float, float]], day: float) -> float:
    value = 1.0
    for t, s in curve:
        if t > day:
            break
        value = s
    return value


def summarize(observations: list[Observation]) -> dict:
    curve = kaplan_meier(observations)
    longest = max(o.days for o in observations)
    summary: dict = {"n": len(observations), "sold": sum(o.sold for o in observations)}
    for day in HORIZONS:
        # Дальше самого долгого наблюдения кривая ничего не знает — честнее промолчать.
        summary[f"p{day}"] = round(1 - survival_at(curve, day), 3) if longest >= day else None
    median = next((t for t, s in curve if s <= 0.5), None)
    summary["median_days"] = round(median, 1) if median is not None else None
    return summary


def build_liquidity(rows: list[tuple], reposted: set[int], now: datetime) -> dict:
    """rows: (listing_id, категория, цена, медиана рынка, появилось, пропало или None).
    Объявления с несколькими позициями пропускаем: неясно, какая из них определила скорость."""
    per_listing: dict[int, list[tuple]] = defaultdict(list)
    for row in rows:
        per_listing[row[0]].append(row)
    groups: dict[str, dict[str, list[Observation]]] = defaultdict(lambda: defaultdict(list))
    for listing_id, items in per_listing.items():
        if len(items) != 1:
            continue
        _, category, price, median, posted_at, gone_at = items[0]
        bucket = bucket_of(price / median)
        if bucket is None:
            continue
        sold = gone_at is not None and listing_id not in reposted
        end = gone_at if gone_at is not None else now
        days = max((end - posted_at).total_seconds() / 86400, 0.01)
        obs = Observation(days=days, sold=sold)
        groups["all"][bucket].append(obs)
        groups[category][bucket].append(obs)
    result: dict = {"updated": now.isoformat(timespec="seconds")}
    for scope, buckets in groups.items():
        need = MIN_OBSERVATIONS if scope == "all" else MIN_CATEGORY_OBSERVATIONS
        summaries = {b: summarize(obs) for b, obs in buckets.items() if len(obs) >= need}
        if summaries:
            result[scope] = summaries
    return result


def for_category(liquidity: dict, category: str | None, bucket: str) -> tuple[dict | None, bool]:
    """Статистика по категории, а если там мало данных — по всему рынку. (данные, это ли общий рынок)."""
    if category and bucket in liquidity.get(category, {}):
        return liquidity[category][bucket], False
    if bucket in liquidity.get("all", {}):
        return liquidity["all"][bucket], True
    return None, False
