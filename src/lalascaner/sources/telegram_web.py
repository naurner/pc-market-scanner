"""Чтение публичных Telegram-каналов и групп без авторизации.

* Каналы: веб-лента https://t.me/s/<name> (по 20 постов, пагинация ?before=<id>).
* Группы: веб-лента для них закрыта, поэтому читаем поштучно embed-виджет
  https://t.me/<name>/<id>?embed=1 — он работает для публичных групп.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx
from bs4 import BeautifulSoup, Tag

from lalascaner.sources.base import USER_AGENT, RawListing

log = logging.getLogger(__name__)

TME = "https://t.me"
MISSING = "missing"


@dataclass
class ChatInfo:
    username: str
    kind: str  # channel | group
    title: str | None


def _to_naive_utc(value: str | None) -> datetime:
    if not value:
        return datetime.now(UTC).replace(tzinfo=None)
    return datetime.fromisoformat(value).astimezone(UTC).replace(tzinfo=None)


def _message_text(msg: Tag) -> str:
    node = msg.select_one(".tgme_widget_message_text.js-message_text")
    if node is None:
        return ""
    for br in node.find_all("br"):
        br.replace_with("\n")
    return node.get_text().strip()


_RE_BG_URL = re.compile(r"background-image:url\('([^']+)'\)")
_RE_POST_ID = re.compile(r"/(\d+)(?:\?|$)")


def _photos(msg: Tag) -> tuple[list[str], list[int]]:
    urls: list[str] = []
    ids: list[int] = []
    for a in msg.select("a.tgme_widget_message_photo_wrap"):
        if m := _RE_BG_URL.search(a.get("style", "")):
            urls.append(m.group(1))
        if m := _RE_POST_ID.search(a.get("href", "")):
            ids.append(int(m.group(1)))
    return urls, ids


_RE_REPLY_ID = re.compile(r"/(\d+)$")


def _reply_target_id(msg: Tag) -> int | None:
    """Id сообщения, на которое отвечает это (класс tgme_widget_message_reply). В группах с темами
    каждый прямой пост в теме технически «отвечает» на служебное сообщение о создании темы — так
    telegram_blocked_topics в конфиге может скрыть целую тему (болтовня, blacklist, ищу/обмен...)."""
    node = msg.select_one("a.tgme_widget_message_reply")
    if node is None:
        return None
    if m := _RE_REPLY_ID.search(node.get("href", "")):
        return int(m.group(1))
    return None


_VIEW_MULT = {"K": 1_000, "M": 1_000_000}


def _meta(msg: Tag) -> dict:
    """Просмотры поста («951K», «2.54M») — Telegram показывает их у каналов, у групп их нет."""
    node = msg.select_one(".tgme_widget_message_views")
    raw = node.get_text(strip=True).upper() if node else ""
    if not raw:
        return {}
    try:
        views = float(raw[:-1]) * _VIEW_MULT[raw[-1]] if raw[-1] in _VIEW_MULT else float(raw)
    except ValueError:
        return {}
    return {"views": int(views)}


def _author(msg: Tag) -> tuple[str | None, str | None]:
    node = msg.select_one(".tgme_widget_message_author > .tgme_widget_message_author_name")
    if node is None:
        node = msg.select_one(".tgme_widget_message_from_author")
    if node is None:
        return None, None
    name = node.get_text(strip=True) or None
    href = node.get("href") if node.name == "a" else None
    author_id = href.rstrip("/").rsplit("/", 1)[-1] if href else name
    return author_id, name


def parse_channel_page(html: str, username: str) -> tuple[list[RawListing], str | None]:
    soup = BeautifulSoup(html, "html.parser")
    title_node = soup.select_one(".tgme_channel_info_header_title")
    title = title_node.get_text(strip=True) if title_node else None
    listings = []
    for msg in soup.select("div.tgme_widget_message[data-post]"):
        post = msg.get("data-post", "")
        _, _, msg_id = post.partition("/")
        if not msg_id.isdigit():
            continue
        text = _message_text(msg)
        if not text:
            continue
        time_node = msg.select_one(".tgme_widget_message_date time")
        photos, _ = _photos(msg)
        listings.append(
            RawListing(
                source="telegram",
                external_id=f"{username}/{msg_id}",
                source_ref=username,
                url=f"{TME}/{username}/{msg_id}",
                text=text,
                posted_at=_to_naive_utc(time_node.get("datetime") if time_node else None),
                author_id=username,
                author_name=title,
                images=photos[:10],
                topic_id=_reply_target_id(msg),
                meta=_meta(msg),
            )
        )
    listings.sort(key=lambda r: int(r.external_id.rsplit("/", 1)[1]))
    return listings, title


def parse_embed(html: str, username: str, msg_id: int) -> RawListing | str | None:
    """RawListing — пост с текстом; MISSING — сообщения нет; None — есть, но без текста."""
    soup = BeautifulSoup(html, "html.parser")
    if soup.select_one(".tgme_widget_message_error"):
        return MISSING
    msg = soup.select_one("div.tgme_widget_message")
    if msg is None:
        return MISSING
    text = _message_text(msg)
    if not text:
        return None
    photos, album_ids = _photos(msg)
    # Альбом приходит несколькими сообщениями с одинаковым текстом — ключуем по первому фото.
    base_id = min(album_ids + [msg_id]) if album_ids else msg_id
    time_node = msg.select_one(".tgme_widget_message_date time")
    author_id, author_name = _author(msg)
    return RawListing(
        source="telegram",
        external_id=f"{username}/{base_id}",
        source_ref=username,
        url=f"{TME}/{username}/{msg_id}",
        text=text,
        posted_at=_to_naive_utc(time_node.get("datetime") if time_node else None),
        author_id=author_id,
        author_name=author_name,
        images=photos[:10],
        topic_id=_reply_target_id(msg),
        meta=_meta(msg),
    )


class TelegramWebClient:
    def __init__(self, http: httpx.AsyncClient | None = None, concurrency: int = 6):
        self._http = http or httpx.AsyncClient(
            headers={"User-Agent": USER_AGENT}, timeout=30, follow_redirects=False
        )
        self._owns_http = http is None
        self._sem = asyncio.Semaphore(concurrency)

    async def close(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def _get(self, url: str, params: dict | None = None) -> httpx.Response:
        backoff = 5.0
        for _ in range(5):
            async with self._sem:
                resp = await self._http.get(url, params=params)
            if resp.status_code == 429 or resp.status_code >= 500:
                log.warning("telegram: HTTP %s на %s, пауза %.0f с", resp.status_code, url, backoff)
                await asyncio.sleep(backoff)
                backoff *= 2
                continue
            return resp
        return resp

    async def probe(self, username: str) -> ChatInfo | None:
        username = username.lstrip("@").lower()
        resp = await self._get(f"{TME}/s/{username}")
        if resp.status_code == 200 and "tgme_widget_message" in resp.text:
            _, title = parse_channel_page(resp.text, username)
            return ChatInfo(username, "channel", title)
        page = await self._get(f"{TME}/{username}")
        if page.status_code != 200:
            return None
        soup = BeautifulSoup(page.text, "html.parser")
        title_node = soup.select_one(".tgme_page_title")
        extra = soup.select_one(".tgme_page_extra")
        if title_node is None:
            return None
        extra_text = extra.get_text(" ", strip=True) if extra else ""
        kind = "group" if re.search(r"member|участник", extra_text, re.I) else "channel"
        return ChatInfo(username, kind, title_node.get_text(strip=True))

    async def fetch_channel_page(self, username: str, before: int | None = None) -> list[RawListing]:
        params = {"before": before} if before else None
        resp = await self._get(f"{TME}/s/{username}", params=params)
        if resp.status_code != 200:
            return []
        listings, _ = parse_channel_page(resp.text, username)
        return listings

    async def fetch_group_message(self, username: str, msg_id: int) -> RawListing | str | None:
        resp = await self._get(f"{TME}/{username}/{msg_id}", params={"embed": "1", "mode": "tme"})
        if resp.status_code != 200:
            return MISSING
        return parse_embed(resp.text, username, msg_id)

    # В живых группах много удалённых сообщений: дыры в десятки id — обычное дело.
    GAP_SPAN = 240

    async def _exists_near(self, username: str, msg_id: int, span: int = GAP_SPAN, step: int = 8) -> bool:
        results = await asyncio.gather(
            *(self.fetch_group_message(username, i) for i in range(msg_id, msg_id + span, step))
        )
        return any(r != MISSING for r in results)

    async def latest_group_message_id(self, username: str) -> int:
        """Поиск последнего id сообщения в группе (с учётом дыр от удалённых сообщений)."""
        lo, hi = 0, 1024
        while await self._exists_near(username, hi):
            lo, hi = hi, hi * 2
            if hi > 100_000_000:
                break
        while hi - lo > self.GAP_SPAN:
            mid = (lo + hi) // 2
            if await self._exists_near(username, mid):
                lo = mid
            else:
                hi = mid
        _, last = await self.scan_group(
            username, max(1, lo), max_messages=hi - lo + 2 * self.GAP_SPAN, miss_limit=self.GAP_SPAN
        )
        return max(last, lo)

    async def scan_group(
        self,
        username: str,
        start_id: int,
        max_messages: int = 400,
        miss_limit: int = 40,
        window: int = 8,
        until_id: int | None = None,
    ) -> tuple[list[RawListing], int]:
        """Читает сообщения начиная с start_id. Возвращает (посты с текстом, последний существующий id).

        until_id — известный последний id: до него дыры из удалённых сообщений не останавливают чтение.
        """
        listings: dict[str, RawListing] = {}
        last_existing = start_id - 1
        misses = 0
        cur = start_id
        while cur < start_id + max_messages and (misses < miss_limit or (until_id is not None and cur <= until_id)):
            ids = list(range(cur, cur + window))
            results = await asyncio.gather(*(self.fetch_group_message(username, i) for i in ids))
            for msg_id, res in zip(ids, results, strict=True):
                if res == MISSING:
                    misses += 1
                    continue
                misses = 0
                last_existing = msg_id
                if isinstance(res, RawListing):
                    listings.setdefault(res.external_id, res)
            cur += window
        return list(listings.values()), last_existing
