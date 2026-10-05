"""Классификаторы для бесплатных и локальных моделей.

* LocalClassifier — любой OpenAI-совместимый сервер: Groq, OpenRouter, LM Studio, vLLM и т.п.
* OllamaClassifier — нативный API Ollama (/api/chat). Ответ жёстко ограничивается JSON-схемой
  (structured outputs): небольшая локальная модель физически не может выдать невалидный JSON или
  лишние поля. Плюс явный num_ctx — по умолчанию Ollama берёт 4096 токенов и молча отрезает начало
  промпта, то есть как раз инструкции.

Промпт и схема результата те же, что у Claude (ai/claude.py); отличается транспорт. Batch API и кэш
промпта у этих провайдеров другие или отсутствуют: supports_batch=False, чтобы планировщик и CLI
не пытались его использовать.

Сбои делятся на два вида. Мусор в ответе / обрезка ответа по длине — пачка делится пополам и повторяется
(RetryableSplit): возможно, дело в одном посте. Сервер недоступен, нет модели, неверный ключ, не хватает
памяти, таймаут — AIUnavailable: это про саму модель или железо, а не про конкретный пост, поэтому делить
бессмысленно и вредно — на слабом Mac таймаут на пачке иначе обернётся таймаутом на её половинах, потом
на четвертинках, и один медленный ответ растянется на десятки минут. Посты просто откладываются до
восстановления сервера.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

import httpx
from pydantic import ValidationError

from lalascaner.ai.claude import AIUnavailable, PostInput, RetryableSplit
from lalascaner.ai.prompt import SYSTEM_PROMPT, user_message
from lalascaner.ai.schema import BatchResult, PostResult

log = logging.getLogger(__name__)

# Поля, которые модель обязана указать у каждой позиции. Остальные характеристики — по желанию:
# так ответ заметно короче (а локальная модель быстрее), чем если требовать все ~30 полей.
_CORE_FIELDS = ["category", "intent", "model", "condition", "price", "currency", "quantity", "is_bundle"]


def _inline_refs(node: Any, defs: dict[str, Any]) -> Any:
    """Разворачивает $ref/$defs и убирает title/default/description — грамматика получается компактнее,
    а разные серверы по-разному понимают ссылки."""
    if isinstance(node, dict):
        if "$ref" in node:
            return _inline_refs(defs[node["$ref"].rsplit("/", 1)[-1]], defs)
        out = {k: _inline_refs(v, defs) for k, v in node.items() if k not in ("$defs", "title", "default", "description")}
        if out.get("type") == "object" and "properties" in out:
            out["additionalProperties"] = False
        return out
    if isinstance(node, list):
        return [_inline_refs(v, defs) for v in node]
    return node


def build_local_schema() -> dict[str, Any]:
    schema = BatchResult.model_json_schema()
    item = schema["$defs"]["ExtractedItem"]
    item["required"] = sorted(set(item.get("required", [])) | set(_CORE_FIELDS))
    return _inline_refs(schema, schema["$defs"])


LOCAL_SCHEMA = build_local_schema()


@dataclass
class Usage:
    """Та же форма, что и Usage в ai/claude.py (нужна для /stats и общего учёта в pipeline)."""

    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    def add(self, usage: dict[str, Any] | None) -> None:
        """Токены в формате OpenAI (prompt_tokens/completion_tokens) или Ollama (prompt_eval_count/eval_count)."""
        self.requests += 1
        if not usage:
            return
        self.input_tokens += usage.get("prompt_tokens") or usage.get("prompt_eval_count") or 0
        self.output_tokens += usage.get("completion_tokens") or usage.get("eval_count") or 0
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        self.cache_read_input_tokens += cached or 0

    def as_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


def parse_response_text(text: str, expected_ids: list[str]) -> dict[str, PostResult]:
    try:
        batch = BatchResult.model_validate_json(text)
    except ValidationError as exc:
        raise RetryableSplit(f"невалидный JSON от модели: {exc.errors()[:2]}") from exc
    return {post.post_id: post for post in batch.posts if post.post_id in expected_ids}


def _in_docker() -> bool:
    return os.path.exists("/.dockerenv")


@dataclass
class LocalClassifier:
    base_url: str
    api_key: str = "not-needed"
    model: str = "llama3.1"
    max_tokens: int = 4096
    timeout: float = 300.0
    supports_batch: bool = False
    usage: Usage = field(default_factory=Usage)
    _http: httpx.AsyncClient = field(init=False, repr=False)

    chat_path = "/chat/completions"
    label = "OpenAI-совместимый сервер"
    kind = "local"  # метка в listings.classifier и ключ уверенности в repo.CONFIDENCE
    background_refine = True  # медленный: правила дают результат сразу, ИИ дорабатывает в фоне

    def __post_init__(self) -> None:
        self._http = httpx.AsyncClient(
            base_url=self.base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=httpx.Timeout(self.timeout, connect=10.0),
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    # ── формат запроса и ответа (у Ollama свой — см. OllamaClassifier) ──────────

    def request_params(self, posts: list[PostInput]) -> dict[str, Any]:
        posts_json = json.dumps([p.as_json() for p in posts], ensure_ascii=False)
        return {
            "model": self.model,
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message(posts_json)},
            ],
        }

    def _extract(self, data: dict[str, Any]) -> tuple[str, bool]:
        """(текст ответа, обрезан ли он по длине)."""
        choice = data["choices"][0]
        return (choice.get("message") or {}).get("content") or "", choice.get("finish_reason") == "length"

    # ── транспорт ──────────────────────────────────────────────────────────────

    def unreachable_hint(self, exc: Exception) -> str:
        hint = f"нет соединения с {self.base_url} ({type(exc).__name__})"
        if _in_docker() and ("localhost" in self.base_url or "127.0.0.1" in self.base_url):
            hint += ". Внутри Docker localhost — это сам контейнер: укажите адрес http://host.docker.internal:<порт>"
        return hint

    def _explain_error(self, resp: httpx.Response) -> str | None:
        """Ответ сервера, после которого повторять бессмысленно, — текст для пользователя; иначе None."""
        if resp.status_code in (401, 403):
            return "сервер отклонил ключ — проверьте AI_API_KEY"
        if resp.status_code == 404:
            return f"404 от {self.base_url}: неверный AI_BASE_URL или нет модели «{self.model}»"
        if resp.status_code == 429:
            return "провайдер ограничил частоту запросов (лимит бесплатного тарифа)"
        return None

    async def _request(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            resp = await self._http.post(path, json=payload)
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise AIUnavailable(self.unreachable_hint(exc)) from exc
        except httpx.TimeoutException as exc:
            raise AIUnavailable(
                f"модель «{self.model}» не успела ответить за {self.timeout:.0f} с. Похоже, ей не хватает "
                "ресурсов Mac — попробуйте модель полегче (например qwen2.5:3b) или увеличьте AI_TIMEOUT"
            ) from exc
        if reason := self._explain_error(resp):
            raise AIUnavailable(reason)
        resp.raise_for_status()
        return resp.json()

    async def _call(self, posts: list[PostInput]) -> dict[str, PostResult]:
        data = await self._request(self.chat_path, self.request_params(posts))
        self.usage.add(data.get("usage") or data)
        text, truncated = self._extract(data)
        if truncated:
            raise RetryableSplit("ответ обрезан по max_tokens")
        return parse_response_text(text, [p.post_id for p in posts])

    async def classify(self, posts: list[PostInput]) -> tuple[dict[str, PostResult], dict[str, str]]:
        """Возвращает (результаты по post_id, ошибки по post_id). Проблемные пачки делятся пополам —
        как у ClaudeClassifier. AIUnavailable не ловится: его обрабатывает pipeline (посты подождут)."""
        if not posts:
            return {}, {}
        try:
            results = await self._call(posts)
        except (RetryableSplit, httpx.HTTPError, ValueError, KeyError, IndexError) as exc:
            if len(posts) == 1:
                return {}, {posts[0].post_id: str(exc)}
            log.warning("%s: %s — делю пачку из %d постов", self.label, exc, len(posts))
            mid = len(posts) // 2
            # Последовательно: локальная модель обрабатывает запросы по одному, параллель только замедлит.
            left = await self.classify(posts[:mid])
            right = await self.classify(posts[mid:])
            return {**left[0], **right[0]}, {**left[1], **right[1]}
        missing = [p for p in posts if p.post_id not in results]
        errors: dict[str, str] = {}
        if missing and len(missing) < len(posts):
            retry_results, errors = await self.classify(missing)
            results.update(retry_results)
        elif missing:
            errors = {p.post_id: "модель не вернула результат для поста" for p in missing}
        return results, errors

    # ── проверка доступности ───────────────────────────────────────────────────

    async def health(self) -> tuple[bool, str]:
        """(всё в порядке, пояснение). Дешёвый запрос списка моделей — токены не тратятся."""
        try:
            resp = await self._http.get("/models")
        except httpx.TransportError as exc:
            return False, self.unreachable_hint(exc)
        if reason := self._explain_error(resp):
            # 404 на /models — не ошибка: не все серверы отдают список моделей.
            return (True, f"{self.base_url} отвечает") if resp.status_code == 404 else (False, reason)
        if resp.status_code != 200:
            return False, f"{self.base_url}: HTTP {resp.status_code}"
        ids = [m.get("id") for m in (resp.json().get("data") or [])]
        if ids and self.model not in ids:
            return False, f"у провайдера нет модели «{self.model}». Доступно, например: {', '.join(map(str, ids[:5]))}"
        return True, f"{self.base_url} · модель {self.model}"


@dataclass
class OllamaClassifier(LocalClassifier):
    """Нативный API Ollama: ответ ограничен JSON-схемой, размер контекста задаётся явно."""

    base_url: str = "http://localhost:11434"
    model: str = "qwen2.5:7b"
    # Промпт (~5–6 тыс. токенов) + пачка постов + ответ должны поместиться в контекст.
    num_ctx: int = 8192
    keep_alive: str = "30m"

    chat_path = "/api/chat"
    label = "Ollama"

    def request_params(self, posts: list[PostInput]) -> dict[str, Any]:
        posts_json = json.dumps([p.as_json() for p in posts], ensure_ascii=False)
        return {
            "model": self.model,
            "stream": False,
            "format": LOCAL_SCHEMA,
            "keep_alive": self.keep_alive,
            "options": {"temperature": 0, "seed": 0, "num_ctx": self.num_ctx, "num_predict": self.max_tokens},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message(posts_json)},
            ],
        }

    def _extract(self, data: dict[str, Any]) -> tuple[str, bool]:
        return (data.get("message") or {}).get("content") or "", data.get("done_reason") == "length"

    def unreachable_hint(self, exc: Exception) -> str:
        return super().unreachable_hint(exc) + ". Запущена ли Ollama? На Mac: brew services start ollama"

    def _explain_error(self, resp: httpx.Response) -> str | None:
        if resp.status_code == 404:
            return f"модель «{self.model}» не скачана — выполните: ollama pull {self.model}"
        if resp.status_code == 500 and "memory" in resp.text.lower():
            return f"модели «{self.model}» не хватает памяти — возьмите поменьше (например qwen2.5:3b): {resp.text[:200]}"
        return super()._explain_error(resp)

    async def health(self) -> tuple[bool, str]:
        try:
            resp = await self._http.get("/api/tags")
        except httpx.TransportError as exc:
            return False, self.unreachable_hint(exc)
        if resp.status_code != 200:
            return False, f"{self.base_url}: HTTP {resp.status_code} — это точно Ollama?"
        names = [m.get("name") or m.get("model") or "" for m in resp.json().get("models") or []]
        if self.model not in names and f"{self.model}:latest" not in names:
            have = ", ".join(names) if names else "ни одной"
            return False, f"Ollama работает, но модель «{self.model}» не скачана (сейчас есть: {have}). Выполните: ollama pull {self.model}"
        return True, f"Ollama {self.base_url} · модель {self.model} готова"
