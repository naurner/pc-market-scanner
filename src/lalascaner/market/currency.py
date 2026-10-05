"""Курсы валют Нацбанка КР — всё приводим к сомам."""

from __future__ import annotations

import logging
import re

import httpx

log = logging.getLogger(__name__)

NBKR_URL = "https://www.nbkr.kg/XML/daily.xml"
DEFAULT_RATES: dict[str, float] = {"KGS": 1.0, "USD": 87.45, "EUR": 100.9, "RUB": 1.04, "KZT": 0.195}

_RE_CURRENCY = re.compile(
    r'<Currency ISOCode="([A-Z]{3})">\s*<Nominal>(\d+)</Nominal>\s*<Value>([\d,\.]+)</Value>', re.S
)


def parse_nbkr(xml: str) -> dict[str, float]:
    rates = {"KGS": 1.0}
    for code, nominal, value in _RE_CURRENCY.findall(xml):
        rates[code] = float(value.replace(",", ".")) / int(nominal)
    return rates


async def fetch_rates(http: httpx.AsyncClient | None = None) -> dict[str, float]:
    client = http or httpx.AsyncClient(timeout=20)
    try:
        resp = await client.get(NBKR_URL)
        resp.raise_for_status()
        rates = parse_nbkr(resp.content.decode("windows-1251", errors="replace"))
        if "USD" not in rates:
            raise ValueError("в ответе НБКР нет USD")
        return rates
    finally:
        if http is None:
            await client.aclose()


def to_kgs(price: float | None, currency: str | None, rates: dict[str, float]) -> float | None:
    if price is None:
        return None
    rate = rates.get((currency or "KGS").upper()) or DEFAULT_RATES.get((currency or "KGS").upper())
    if rate is None:
        return None
    return round(price * rate, 2)
