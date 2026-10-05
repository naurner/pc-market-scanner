"""Поиск похожих объявлений по векторам: индекс в памяти, оценка цены по соседям, признаки перезалива.

Векторы хранятся нормированными (float16 в БД), поэтому сходство — просто скалярное произведение.
10 тыс. объявлений × 1024 числа — ~40 МБ памяти и миллисекунды на запрос.
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import numpy as np
except ImportError:  # образ собран до появления numpy — векторы выключаются, остальное работает
    np = None

# Соседи слабее этого по смыслу уже про другое — в оценку цены не берём.
MIN_SIMILARITY = 0.75
# Короткий запрос («rtx 3060 б/у») по смыслу всегда дальше от длинного объявления, чем объявления друг от друга.
QUERY_MIN_SIMILARITY = 0.6
# Почти тот же текст. Числа векторы различают плохо, поэтому перезалив ещё проверяется по модели и цене.
DUP_SIMILARITY = 0.96
DUP_PRICE_TOLERANCE = 0.1


def available() -> bool:
    return np is not None


def pack(vec: list[float]) -> bytes:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm > 0:
        arr = arr / norm
    return arr.astype(np.float16).tobytes()


def unpack(blob: bytes):  # noqa: ANN201
    return np.frombuffer(blob, dtype=np.float16).astype(np.float32)


class VectorIndex:
    def __init__(self) -> None:
        self._ids: list[int] = []
        self._pos: dict[int, int] = {}
        self._rows: list = []
        self._matrix = None
        self._id_arr = None

    def __len__(self) -> int:
        return len(self._ids)

    def __contains__(self, listing_id: int) -> bool:
        return listing_id in self._pos

    def add(self, listing_id: int, vec) -> None:  # noqa: ANN001
        if listing_id in self._pos:
            self._rows[self._pos[listing_id]] = vec
        else:
            self._pos[listing_id] = len(self._ids)
            self._ids.append(listing_id)
            self._rows.append(vec)
        self._matrix = None

    def get(self, listing_id: int):  # noqa: ANN201
        pos = self._pos.get(listing_id)
        return None if pos is None else self._rows[pos]

    def search(self, query, k: int, exclude: set[int] | frozenset = frozenset()) -> list[tuple[int, float]]:  # noqa: ANN001
        if not self._ids:
            return []
        if self._matrix is None:
            self._matrix = np.vstack(self._rows)
            self._id_arr = np.asarray(self._ids)
        scores = self._matrix @ query
        take = min(len(scores), k + len(exclude))
        top = np.argpartition(-scores, take - 1)[:take]
        top = top[np.argsort(-scores[top])]
        found = [(int(self._id_arr[i]), float(scores[i])) for i in top]
        return [(lid, score) for lid, score in found if lid not in exclude][:k]


def weighted_median(pairs: list[tuple[float, float]]) -> float:
    """pairs: (значение, вес). Медиана, где похожие соседи весят больше далёких."""
    ordered = sorted(pairs)
    total = sum(w for _, w in ordered)
    acc = 0.0
    for value, weight in ordered:
        acc += weight
        if acc >= total / 2:
            return value
    return ordered[-1][0]


@dataclass
class Estimate:
    price: float
    n: int


def estimate_price(
    neighbors: list[tuple[float, float]], min_n: int = 3, max_n: int = 15, min_similarity: float = MIN_SIMILARITY
) -> Estimate | None:
    """neighbors: (цена, сходство) от самых похожих. Оценка — взвешенная медиана цен близких соседей."""
    close = [(price, score) for price, score in neighbors if score >= min_similarity][:max_n]
    if len(close) < min_n:
        return None
    return Estimate(price=weighted_median(close), n=len(close))


def is_repost(score: float, same_key: bool, price: float | None, other_price: float | None) -> bool:
    if score < DUP_SIMILARITY or not same_key:
        return False
    if price is None or other_price is None:
        return price is None and other_price is None
    return abs(price - other_price) <= DUP_PRICE_TOLERANCE * max(price, other_price)
