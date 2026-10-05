"""Клиент JSON API lalafo.kg (тот же, что использует их сайт)."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import random
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx

from lalascaner.sources.base import USER_AGENT, RawListing

log = logging.getLogger(__name__)

FEED_URL = "https://lalafo.kg/api/search/v3/feed/search"
SITE = "https://lalafo.kg"
HEADERS = {
    "device": "pc",
    "language": "ru_RU",
    "country-id": "12",
    "User-Agent": USER_AGENT,
    "Accept": "application/json",
}

# Родительские категории, которые обходим (API отдаёт в них объявления всех подкатегорий).
FEED_CATEGORIES: dict[int, str] = {
    1345: "Комплектующие для ПК",
    1342: "Настольные ПК",
    1350: "Мониторы",
    1343: "Ноутбуки",
}

# Листовая категория lalafo → подсказка для классификатора. None — категорию разбираем, но без подсказки.
LEAF_HINTS: dict[int, str | None] = {
    1355: "gpu",
    4379: "cpu",
    4337: "ram",
    4453: "motherboard",
    1346: None,  # жёсткие диски и SSD
    4405: "psu",
    4404: "cooler",
    7464: "case",
    13263: None,  # комплекты для сборки
    13281: "other",
    13268: "other",
    13283: "other",
    4406: None,
    1342: "pc",
    1350: "monitor",
    # ноутбуки по брендам
    **{
        cid: "laptop"
        for cid in (
            11534, 11535, 11537, 11572, 12027, 11538, 11577, 11541, 11881, 11589, 11542,
            11543, 12033, 12031, 12030, 12682, 12683, 12684, 12685, 12686, 12687, 12286,
        )
    },
}


def is_relevant_leaf(category_id: int | str | None) -> bool:
    try:
        return int(category_id) in LEAF_HINTS  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False


def leaf_hint(category_id: int | str | None) -> str | None:
    try:
        return LEAF_HINTS.get(int(category_id))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _ts(value: int | None) -> datetime:
    if not value:
        return datetime.now(UTC).replace(tzinfo=None)
    return datetime.fromtimestamp(int(value), UTC).replace(tzinfo=None)


METRIC_FIELDS = {
    "views": "views",
    "impressions": "impressions",
    "favorites": "favorite_count",
    "callers": "callers_count",
    "writers": "writers_count",
}


def _params(item: dict) -> list[dict]:
    out = []
    for param in item.get("params") or []:
        if isinstance(param, dict) and param.get("name"):
            out.append({"name": param["name"], "value": param.get("value")})
    return out


def listing_meta(item: dict) -> dict:
    """Всё полезное из ответа lalafo, что не текст и не цена. Телефон — только хэшем: достаточно, чтобы
    узнать одного продавца под разными аккаунтами, и не храним сами номера."""
    user = item.get("user") or {}
    phone = "".join(ch for ch in str(item.get("mobile") or "") if ch.isdigit())
    meta = {
        **{name: item.get(src) for name, src in METRIC_FIELDS.items()},
        "old_price": item.get("old_price"),
        "negotiable": item.get("is_negotiable"),
        "vip": item.get("is_vip"),
        "premium": item.get("is_premium"),
        "paid": bool(item.get("paid_features") or item.get("paid_packages")) or None,
        "updated": item.get("updated_time"),
        "push_up": item.get("last_push_up"),
        "lat": item.get("lat"),
        "lng": item.get("lng"),
        "label": item.get("ad_label"),
        "params": _params(item) or None,
        "seller_pro": user.get("pro"),
        "seller_company": user.get("company_name"),
        "seller_response_rate": user.get("response_rate"),
        "seller_response_time": user.get("response_time"),
        "phone_hash": hashlib.sha1(phone.encode()).hexdigest()[:16] if phone else None,
    }
    return {k: v for k, v in meta.items() if v is not None}


def parse_item(item: dict) -> RawListing:
    user = item.get("user") or {}
    images = [
        img.get("original_url") or img.get("thumbnail_url")
        for img in item.get("images") or []
        if img.get("original_url") or img.get("thumbnail_url")
    ]
    price = item.get("price")
    return RawListing(
        source="lalafo",
        external_id=str(item["id"]),
        source_ref=str(item.get("category_id") or ""),
        url=SITE + item["url"] if item.get("url") else f"{SITE}/kyrgyzstan/ads/{item['id']}",
        title=(item.get("title") or "").strip(),
        text=(item.get("description") or "").strip(),
        price=float(price) if price else None,
        currency=item.get("currency") or None,
        city=item.get("city") or None,
        author_id=str(item.get("user_id")) if item.get("user_id") else None,
        author_name=user.get("username") or None,
        images=images[:10],
        posted_at=_ts(item.get("created_time")),
        is_promoted=bool(item.get("is_vip") or item.get("is_premium") or item.get("campaign_show")),
        meta=listing_meta(item),
    )


@dataclass
class FeedPage:
    listings: list[RawListing]
    page: int
    page_count: int
    total: int


# 490 — нестандартный код lalafo: временная блокировка за слишком частые запросы.
RETRYABLE = {429, 490, 500, 502, 503, 504}


class LalafoClient:
    def __init__(self, http: httpx.AsyncClient | None = None, delay: float = 1.2):
        self._http = http or httpx.AsyncClient(headers=HEADERS, timeout=30, follow_redirects=True)
        self._owns_http = http is None
        self._delay = delay
        self._lock = asyncio.Lock()

    async def close(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def fetch_page(self, category_id: int, page: int = 1, per_page: int = 40) -> FeedPage:
        params = {
            "expand": "url",
            "per-page": per_page,
            "category_id": category_id,
            "page": page,
            "sort_by": "newest",
        }
        data = await self._get_json(FEED_URL, params)
        meta = data.get("_meta") or {}
        listings = []
        for raw in data.get("items") or []:
            if not raw.get("id") or not is_relevant_leaf(raw.get("category_id")):
                continue
            try:
                listings.append(parse_item(raw))
            except (KeyError, TypeError, ValueError) as exc:
                log.warning("lalafo: не удалось разобрать объявление %s: %s", raw.get("id"), exc)
        return FeedPage(
            listings=listings,
            page=int(meta.get("currentPage") or page),
            page_count=int(meta.get("pageCount") or 1),
            total=int(meta.get("totalCount") or 0),
        )

    async def _get_json(self, url: str, params: dict) -> dict:
        backoff = 20.0
        for _attempt in range(6):
            async with self._lock:
                try:
                    resp = await self._http.get(url, params=params, headers=HEADERS)
                except httpx.TransportError as exc:
                    resp = None
                    log.warning("lalafo: сетевая ошибка %s", exc)
                finally:
                    await asyncio.sleep(self._delay + random.uniform(0, self._delay / 2))
                if resp is not None and resp.status_code == 200:
                    return resp.json()
                if resp is not None and resp.status_code not in RETRYABLE:
                    resp.raise_for_status()
                # Пауза внутри блокировки: пока ждём, остальные запросы к lalafo тоже стоят.
                status = resp.status_code if resp is not None else "network"
                log.warning("lalafo: HTTP %s на странице %s, пауза %.0f с", status, params.get("page"), backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 600)
        raise RuntimeError(f"lalafo: не удалось получить страницу {params.get('page')} категории {params.get('category_id')}")
