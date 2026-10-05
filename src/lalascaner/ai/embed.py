"""Векторы (эмбеддинги) текста объявлений через Ollama: /api/embed.

Модель по умолчанию — bge-m3: многоязычная, хорошо понимает русский, ~1.2 ГБ. Она в разы быстрее
генеративной модели, которая разбирает объявления, поэтому векторы считаются фоном для всей базы.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from lalascaner.db.models import Listing

MAX_CHARS = 2000


class EmbedUnavailable(Exception):
    """Ollama не отвечает или вернула ошибку — повторим позже."""


class EmbedModelMissing(EmbedUnavailable):
    """Модель векторов не скачана в Ollama."""


def embed_text(listing: Listing) -> str:
    return f"{listing.title}\n{listing.text}".strip()[:MAX_CHARS]


@dataclass
class OllamaEmbedder:
    base_url: str
    model: str
    timeout: float = 120

    def __post_init__(self) -> None:
        self._http = httpx.AsyncClient(base_url=self.base_url.rstrip("/"), timeout=self.timeout)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        try:
            resp = await self._http.post("/api/embed", json={"model": self.model, "input": texts})
        except httpx.HTTPError as exc:
            raise EmbedUnavailable(f"Ollama недоступна ({self.base_url}): {exc}") from exc
        if resp.status_code == 404:
            raise EmbedModelMissing(f"модель {self.model} не скачана")
        if resp.status_code != 200:
            raise EmbedUnavailable(f"Ollama: HTTP {resp.status_code} {resp.text[:200]}")
        vectors = resp.json().get("embeddings") or []
        if len(vectors) != len(texts):
            raise EmbedUnavailable(f"Ollama вернула {len(vectors)} векторов на {len(texts)} текстов")
        return vectors

    async def pull(self) -> None:
        """Скачивает модель (один раз, ~1 ГБ) — чтобы векторы заработали без ручных команд на сервере."""
        try:
            resp = await self._http.post(
                "/api/pull", json={"model": self.model, "stream": False}, timeout=httpx.Timeout(3600, connect=30)
            )
        except httpx.HTTPError as exc:
            raise EmbedUnavailable(f"не удалось скачать {self.model}: {exc}") from exc
        if resp.status_code != 200:
            raise EmbedUnavailable(f"не удалось скачать {self.model}: HTTP {resp.status_code} {resp.text[:200]}")
