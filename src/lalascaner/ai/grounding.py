"""Проверка ответа ИИ по тексту объявления.

Маленькая локальная модель иногда додумывает: «есть 6700xt 12 gb, какая цена?» превращается в
«RTX 3070 за 12 000 сом» — и уходит уведомлением о рекордно дешёвой сделке. Поэтому:
* цена должна найтись в тексте числом (не размером вроде «12 gb» и не номером модели вроде «6700xt»);
* номер модели видеокарты/процессора (3070, 12400) должен встречаться в тексте.
Не прошла цена — позиция остаётся без цены; не прошла модель — позиция выброшена.
"""

from __future__ import annotations

import logging
import re

from lalascaner.ai.normalize import build_keys
from lalascaner.ai.schema import ExtractedItem, PostResult

log = logging.getLogger(__name__)

_NUMBER = re.compile(r"(?<![\w.,])(\d{1,3}(?:[  .,]\d{3})+(?!\d)|\d+(?:[.,]\d+)?)")
_THOUSANDS = re.compile(r"\d{1,3}(?:[  .,]\d{3})+")
# После числа — единица измерения: это размер/характеристика, а не цена.
_UNIT_AFTER = re.compile(
    r"\s*(?:гб|gb|tb|тб|мб|mb|w\b|вт|мгц|mhz|ghz|ггц|hz|гц|герц|ядер|ядра|поток|шт|%|дюйм|inch|\"|″|bit|бит|nm|нм|"
    r"mm\b|мм|rpm|fps|фпс|sp\b|х\d|x\d|×)",
    re.I,
)
_THOUSAND_AFTER = re.compile(r"\s*(?:к|k|тыс)(?![a-zа-я])", re.I)
_CURRENCY_AFTER = re.compile(r"\s*(?:сом|som|kgs|с(?![а-яa-z])|c(?![а-яa-z])|\$|usd|долл|у\.?е|руб|₽|тг|тенге)", re.I)
_LETTER_AFTER = re.compile(r"[a-zа-яё]", re.I)
# Номер модели: группы от 3 цифр в ключе (rtx-3070 → 3070, i5-12400f → 12400, ryzen-5-5600 → 5600).
_MODEL_NUMBER = re.compile(r"\d{3,5}")
MODEL_CHECKED = {"gpu", "cpu"}
PRICE_TOLERANCE = 0.01


def _parse(raw: str) -> float:
    if _THOUSANDS.fullmatch(raw):
        return float(re.sub(r"[  .,]", "", raw))
    return float(raw.replace(",", "."))


def text_numbers(text: str) -> set[float]:
    """Все числа, которые могут быть ценой: «25000», «25 000», «25к» (и 25 000), «за 25» (и 25 000)."""
    found: set[float] = set()
    for m in _NUMBER.finditer(text):
        tail = text[m.end() : m.end() + 12]
        if _UNIT_AFTER.match(tail):
            continue
        thousand = _THOUSAND_AFTER.match(tail)
        # Буквы вплотную к числу — часть названия модели (6700xt, 12400f, 3060ti), если это не «к»/валюта.
        if not thousand and _LETTER_AFTER.match(tail) and not _CURRENCY_AFTER.match(tail):
            continue
        value = _parse(m.group(1))
        found.add(value)
        if thousand or value < 1000:
            # «3060 за 25» в чатах — 25 000; «25к» — тоже.
            found.add(value * 1000)
    return found


def price_grounded(price: float, text: str) -> bool:
    return any(abs(n - price) <= PRICE_TOLERANCE * price for n in text_numbers(text))


def key_grounded(category: str, canonical_key: str, text: str) -> bool:
    if category not in MODEL_CHECKED:
        return True
    model_part = canonical_key.split(":")[1] if ":" in canonical_key else canonical_key
    lowered = text.lower()
    return all(re.search(rf"(?<!\d){n}(?!\d)", lowered) for n in _MODEL_NUMBER.findall(model_part))


def model_grounded(item: ExtractedItem, text: str) -> bool:
    return key_grounded(item.category, build_keys(item)[0], text)


def ground_result(result: PostResult, text: str) -> PostResult:
    items: list[ExtractedItem] = []
    for item in result.items:
        if not model_grounded(item, text):
            log.info("ИИ: позиция «%s» не упоминается в тексте поста %s — выброшена", item.model, result.post_id)
            continue
        if item.price is not None and not price_grounded(item.price, text):
            log.info("ИИ: цены %s нет в тексте поста %s — позиция без цены", item.price, result.post_id)
            item = item.model_copy(update={"price": None, "currency": None})
        items.append(item)
    return result.model_copy(update={"items": items})
