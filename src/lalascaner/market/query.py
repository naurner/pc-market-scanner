"""Разбор пользовательского запроса («3060 12gb», «ryzen 5600», «ddr4 16») в фильтр по ключам."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from sqlalchemy import ColumnElement, and_, func, or_

from lalascaner.ai.normalize import norm_cpu, norm_gpu_chip, norm_memory_type, ram_capacity, slugify, storage_bucket

_CATEGORY_WORDS = (
    (r"видеокарт|видюх|\bgpu\b", "gpu"),
    (r"процессор|\bпроц|\bcpu\b", "cpu"),
    (r"\bозу\b|оператив|\bram\b|\bddr", "ram"),
    (r"\bssd\b|nvme", "ssd"),
    (r"\bhdd\b|жестк|жёстк", "hdd"),
    (r"\bбп\b|блок\w* питани|\bpsu\b", "psu"),
    (r"материнк|мат\.?\s*плат", "motherboard"),
    (r"монитор|моник", "monitor"),
    (r"ноутбук|\bноут\b|laptop|macbook", "laptop"),
    (r"\bпк\b|компьютер|системник|сборк", "pc"),
    (r"кулер|охлажден", "cooler"),
    (r"корпус", "case"),
)


@dataclass
class QueryFilter:
    category: str | None = None
    key_prefixes: list[str] = field(default_factory=list)
    tokens: list[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not (self.category or self.key_prefixes or self.tokens)

    def clause(self, key_col, model_col=None, category_col=None) -> ColumnElement[bool]:  # noqa: ANN001
        parts = []
        if self.key_prefixes:
            parts.append(or_(*(or_(key_col == p, key_col.like(p + ":%")) for p in self.key_prefixes)))
        elif self.tokens:
            haystack = key_col if model_col is None else func.lower(key_col + " " + model_col)
            parts.extend(haystack.like(f"%{tok}%") for tok in self.tokens)
        if self.category and category_col is not None:
            parts.append(category_col == self.category)
        elif self.category and not self.key_prefixes:
            prefix = {"motherboard": "mb", "monitor": "mon"}.get(self.category, self.category)
            parts.append(key_col.like(prefix + ":%"))
        return and_(*parts) if parts else and_(True)


@dataclass
class ItemFilter:
    min_price: float | None = None
    max_price: float | None = None
    condition: str | None = None  # new | used

    @property
    def active(self) -> bool:
        return bool(self.min_price or self.max_price or self.condition)

    def describe(self) -> str:
        parts = []
        if self.condition:
            parts.append("б/у" if self.condition == "used" else "новые")
        if self.min_price:
            parts.append(f"от {self.min_price:,.0f}".replace(",", " "))
        if self.max_price:
            parts.append(f"до {self.max_price:,.0f}".replace(",", " "))
        return ", ".join(parts)


_RE_PRICE_BOUND = re.compile(r"\b(до|от)\s*(\d[\d\s]*(?:[.,]\d+)?)\s*(к|k|тыс\.?)?\s*(сом\w*|с\b)?", re.I)
_RE_USED_WORD = re.compile(r"(?<![а-яa-z])(б/у|бу|б\.у\.?|used)(?![а-яa-z])", re.I)
_RE_NEW_WORD = re.compile(r"(?<![а-яa-z])(нов(ый|ая|ое|ые|ую)|new)(?![а-яa-z])", re.I)


def parse_search(text: str) -> tuple[QueryFilter, ItemFilter]:
    """«3060 12gb до 25к б/у» → фильтр модели + ценовой/состояния."""
    flt = ItemFilter()
    rest = text
    for m in _RE_PRICE_BOUND.finditer(text):
        value = float(re.sub(r"\s", "", m.group(2)).replace(",", ".")) * (1000 if m.group(3) else 1)
        if m.group(1).lower() == "до":
            flt.max_price = value
        else:
            flt.min_price = value
        rest = rest.replace(m.group(0), " ")
    if _RE_USED_WORD.search(rest):
        flt.condition = "used"
        rest = _RE_USED_WORD.sub(" ", rest)
    elif _RE_NEW_WORD.search(rest):
        flt.condition = "new"
        rest = _RE_NEW_WORD.sub(" ", rest)
    return parse_query(rest), flt


def parse_query(query: str) -> QueryFilter:
    q = query.strip().lower()
    qf = QueryFilter()
    if q.startswith("key:"):
        key = q[4:].strip()
        qf.category = {"mb": "motherboard", "mon": "monitor"}.get(key.split(":")[0], key.split(":")[0])
        qf.key_prefixes = [key]
        return qf
    for pattern, category in _CATEGORY_WORDS:
        if re.search(pattern, q):
            qf.category = category
            break

    gb = re.search(r"(\d{1,4})\s*(gb|гб|g\b|tb|тб)", q)
    gb_value = None
    if gb:
        gb_value = float(gb.group(1)) * (1000 if gb.group(2) in ("tb", "тб") else 1)

    if qf.category in (None, "gpu"):
        chip = norm_gpu_chip(q, allow_bare=qf.category == "gpu" or not re.search(r"ddr|ssd|hdd", q))
        if chip and not norm_cpu(q, allow_bare=False):
            qf.category = "gpu"
            qf.key_prefixes = [f"gpu:{chip}:{int(gb_value)}gb" if gb_value and gb_value <= 48 else f"gpu:{chip}"]
            return qf

    if qf.category in (None, "cpu"):
        cpu = norm_cpu(q)
        if cpu:
            qf.category = "cpu"
            qf.key_prefixes = [f"cpu:{cpu}"]
            return qf

    if qf.category == "ram":
        mem = norm_memory_type(q)
        cap = ram_capacity(gb_value) if gb_value else None
        if not cap and (m := re.search(r"\b(4|8|16|32|64|128)\b", q)):
            cap = f"{m.group(1)}gb"
        if mem and cap:
            qf.key_prefixes = [f"ram:{mem}:{cap}", f"ram:{mem}-so:{cap}"]
            return qf

    if qf.category in ("ssd", "hdd") and gb_value:
        cap = storage_bucket(gb_value)
        if qf.category == "hdd":
            qf.key_prefixes = [f"hdd:{cap}"]
        else:
            qf.key_prefixes = [f"ssd:{cap}", f"ssd:nvme:{cap}", f"ssd:sata:{cap}"]
        return qf

    if qf.category == "psu" and (m := re.search(r"(\d{3,4})\s*(w|вт)?", q)):
        qf.key_prefixes = [f"psu:{m.group(1)}w"]
        return qf

    stop = {"видеокарта", "видеокарты", "процессор", "ноутбук", "монитор", "пк", "бп", "озу", "ssd", "hdd"}
    qf.tokens = [tok for tok in slugify(q).split("-") if tok and tok not in stop]
    return qf
