"""Проверка правдоподобия цен: мусорные цены не должны попадать ни в статистику, ни в сделки."""

from __future__ import annotations

# Ниже этих цен (в сомах) позиция почти наверняка разобрана неверно: номер модели, объём, цена комплектующей
# внутри поста про ноутбук и т.п. Такие цены обнуляются, сама позиция остаётся в базе.
MIN_PRICE_KGS: dict[str, float] = {
    "gpu": 800,
    "cpu": 300,
    "ram": 250,
    "motherboard": 500,
    "ssd": 400,
    "hdd": 300,
    "psu": 500,
    "cooler": 150,
    "case": 400,
    "monitor": 800,
    "laptop": 6000,
    "pc": 6000,
    "peripheral": 100,
    "other": 100,
}
MAX_PRICE_KGS = 3_000_000


def plausible_price(category: str, price_kgs: float | None) -> bool:
    if price_kgs is None:
        return True
    return MIN_PRICE_KGS.get(category, 100) <= price_kgs <= MAX_PRICE_KGS
