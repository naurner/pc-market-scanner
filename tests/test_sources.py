import json
from datetime import datetime

import httpx
import respx

from lalascaner.sources.lalafo import FEED_URL, LalafoClient, is_relevant_leaf, leaf_hint, parse_item
from lalascaner.sources.telegram_web import MISSING, TelegramWebClient, parse_channel_page, parse_embed


def test_parse_lalafo_item(fixtures):
    data = json.loads((fixtures / "lalafo_feed_cpu.json").read_text(encoding="utf-8"))
    raw = next(i for i in data["items"] if i["id"] == 114650290)
    listing = parse_item(raw)
    assert listing.source == "lalafo"
    assert listing.external_id == "114650290"
    assert listing.source_ref == "4379"
    assert listing.price == 9900
    assert listing.currency == "KGS"
    assert "ryzen 5 5600" in listing.text.lower()
    assert not listing.is_promoted
    assert listing.url.startswith("https://lalafo.kg/")
    assert isinstance(listing.posted_at, datetime) and listing.posted_at.tzinfo is None
    assert listing.images


def test_lalafo_leaf_hints():
    assert leaf_hint(1355) == "gpu"
    assert leaf_hint("4379") == "cpu"
    assert leaf_hint(11537) == "laptop"
    assert is_relevant_leaf(1346) and leaf_hint(1346) is None
    assert not is_relevant_leaf(12703)  # гарнитуры
    assert not is_relevant_leaf(None)


@respx.mock
async def test_lalafo_fetch_page(fixtures):
    body = (fixtures / "lalafo_feed_cpu.json").read_text(encoding="utf-8")
    route = respx.get(FEED_URL).mock(return_value=httpx.Response(200, text=body))
    client = LalafoClient(delay=0)
    try:
        page = await client.fetch_page(4379, page=1)
    finally:
        await client.close()
    assert route.called
    request = route.calls.last.request
    assert request.headers["language"] == "ru_RU"
    assert request.url.params["sort_by"] == "newest"
    assert page.page_count == 16
    assert len(page.listings) == 12
    assert all(item.source_ref == "4379" for item in page.listings)


def test_listing_content_hash_changes_with_price(fixtures):
    data = json.loads((fixtures / "lalafo_feed_cpu.json").read_text(encoding="utf-8"))
    a = parse_item(data["items"][1])
    b = parse_item(data["items"][1])
    assert a.content_hash() == b.content_hash()
    b.price = (b.price or 0) + 100
    assert a.content_hash() != b.content_hash()


def test_parse_channel_page(fixtures):
    listings, _ = parse_channel_page((fixtures / "tg_channel_page.html").read_text(encoding="utf-8"), "durov")
    assert len(listings) == 3
    ids = [int(item.external_id.split("/")[1]) for item in listings]
    assert ids == sorted(ids)
    assert all(item.text for item in listings)
    assert listings[0].url == f"https://t.me/durov/{ids[0]}"


def test_parse_embed_album_uses_first_photo_id(fixtures):
    html = (fixtures / "tg_group_album.html").read_text(encoding="utf-8")
    listing = parse_embed(html, "compshop_kg", 10000)
    assert listing is not MISSING and listing is not None
    assert listing.external_id == "compshop_kg/9999"
    assert listing.url == "https://t.me/compshop_kg/10000"
    assert listing.text.startswith("12400F+3070 за 54900 сом")
    assert "Colorful RTX 3070 LHR\nGigabyte H610" in listing.text
    assert len(listing.images) == 2
    assert listing.posted_at == datetime(2025, 12, 21, 17, 34, 56)
    assert listing.topic_id == 401  # альбом тоже отвечает на служебное сообщение темы, как и все посты в ней


def test_parse_embed_reply_takes_own_text_not_quote(fixtures):
    html = (fixtures / "tg_group_reply.html").read_text(encoding="utf-8")
    listing = parse_embed(html, "compshop_kg", 67944)
    assert listing.text == "@mxss01"
    assert listing.author_id == "erbol333"
    assert listing.author_name == "Эрбол"
    # Сообщение технически «отвечает» на 67943 — этот id можно занести в TELEGRAM_BLOCKED_TOPICS,
    # если это тема вроде болтовни/blacklist, чтобы такие посты не разбирались как объявления.
    assert listing.topic_id == 67943


def test_parse_embed_missing(fixtures):
    html = (fixtures / "tg_group_notfound.html").read_text(encoding="utf-8")
    assert parse_embed(html, "compshop_kg", 99999) == MISSING


class GappyGroup(TelegramWebClient):
    """Группа с дырами от удалённых сообщений, как в реальном compshop_kg."""

    def __init__(self, existing: set[int]):
        super().__init__()
        self.existing = existing

    async def fetch_group_message(self, username, msg_id):
        return None if msg_id in self.existing else MISSING


async def test_latest_group_message_id_survives_large_gaps():
    existing = set(range(1, 5000, 3)) | set(range(5150, 40000, 2)) | set(range(40180, 68079, 5)) | {68078}
    client = GappyGroup(existing)
    try:
        assert await client.latest_group_message_id("g") == 68078
    finally:
        await client.close()


async def test_scan_group_does_not_stop_on_gap_before_known_latest():
    existing = set(range(100, 120)) | set(range(300, 310))
    client = GappyGroup(existing)
    try:
        _, last_default = await client.scan_group("g", 100, max_messages=1000)
        _, last_until = await client.scan_group("g", 100, max_messages=1000, until_id=309)
    finally:
        await client.close()
    assert last_default == 119
    assert last_until == 309


@respx.mock
async def test_scan_group_dedups_albums_and_stops_on_misses(fixtures):
    album = (fixtures / "tg_group_album.html").read_text(encoding="utf-8")
    missing = (fixtures / "tg_group_notfound.html").read_text(encoding="utf-8")

    def handler(request: httpx.Request) -> httpx.Response:
        msg_id = int(request.url.path.rsplit("/", 1)[1])
        return httpx.Response(200, text=album if msg_id in (9999, 10000) else missing)

    respx.get(url__regex=r"https://t\.me/compshop_kg/\d+").mock(side_effect=handler)
    client = TelegramWebClient()
    try:
        listings, last = await client.scan_group("compshop_kg", 9999, miss_limit=10, window=4)
    finally:
        await client.close()
    assert [item.external_id for item in listings] == ["compshop_kg/9999"]
    assert last == 10000
