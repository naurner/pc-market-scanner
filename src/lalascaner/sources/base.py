from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/140.0 Safari/537.36"
)


def _norm_for_hash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


@dataclass
class RawListing:
    """Объявление в виде, независимом от источника. Все datetime — naive UTC."""

    source: str
    external_id: str
    source_ref: str
    url: str
    text: str
    posted_at: datetime
    title: str = ""
    price: float | None = None
    currency: str | None = None
    city: str | None = None
    author_id: str | None = None
    author_name: str | None = None
    images: list[str] = field(default_factory=list)
    is_promoted: bool = False
    # Telegram: id сообщения, на которое это отвечает. В группах с темами (форум) каждый пост «отвечает»
    # на служебное сообщение о создании темы — так можно отличить тему (болтовня, blacklist...) от другой.
    topic_id: int | None = None
    # Дополнительные данные источника (просмотры, избранное, продавец…) — в хэш не входят: их изменение
    # не делает объявление «новым», а пишется в историю метрик.
    meta: dict = field(default_factory=dict)

    def content_hash(self) -> str:
        payload = "|".join(
            [
                _norm_for_hash(self.title),
                _norm_for_hash(self.text),
                "" if self.price is None else f"{self.price:.0f}",
                self.currency or "",
            ]
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()
