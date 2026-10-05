"""Классификатор постов на Claude: пачки постов, structured outputs, кэш промпта, Batch API."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

import anthropic
from pydantic import ValidationError

from lalascaner.ai.prompt import SYSTEM_PROMPT, user_message
from lalascaner.ai.schema import BatchResult, PostResult

log = logging.getLogger(__name__)

# Серверный фолбэк при отказе модели — поддерживают только эти модели (и только не в Batch API).
FALLBACK_MODELS = {"claude-opus-5", "claude-fable-5-1"}
FALLBACK_BETA = "server-side-fallback-2026-07-01"
MAX_TEXT_CHARS = 6000


@dataclass
class PostInput:
    post_id: str
    source: str
    text: str
    title: str = ""
    price: float | None = None
    currency: str | None = None
    category_hint: str | None = None

    def as_json(self) -> dict[str, Any]:
        return {
            "post_id": self.post_id,
            "source": self.source,
            "title": self.title,
            "text": self.text[:MAX_TEXT_CHARS],
            "listed_price": self.price,
            "listed_currency": self.currency,
            "category_hint": self.category_hint,
        }


@dataclass
class Usage:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    def add(self, usage: Any) -> None:
        self.requests += 1
        for name in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"):
            setattr(self, name, getattr(self, name) + (getattr(usage, name, 0) or 0))

    def as_dict(self) -> dict[str, int]:
        return dict(self.__dict__)


class RetryableSplit(Exception):
    """Ответ нельзя использовать целиком (обрезан/отказ/битый JSON) — пачку стоит разделить."""


class AIUnavailable(Exception):
    """Сервер ИИ не отвечает / не та модель / неверный ключ. Делить пачку бессмысленно — надо ждать и
    повторить позже, а посты пока не считать безнадёжными."""


def _strict(schema: Any) -> Any:
    """Все поля обязательные (nullable), без title и подсказок про default — так схема строже и короче."""
    if isinstance(schema, dict):
        out = {}
        for key, value in schema.items():
            if key == "title":
                continue
            if key == "description" and isinstance(value, str) and value.startswith("{default"):
                continue
            out[key] = _strict(value)
        if out.get("type") == "object" and "properties" in out:
            out["required"] = list(out["properties"].keys())
        return out
    if isinstance(schema, list):
        return [_strict(value) for value in schema]
    return schema


RESPONSE_SCHEMA = _strict(anthropic.transform_schema(BatchResult))


def parse_response_text(text: str, expected_ids: list[str]) -> dict[str, PostResult]:
    try:
        batch = BatchResult.model_validate_json(text)
    except ValidationError as exc:
        raise RetryableSplit(f"невалидный JSON от модели: {exc.errors()[:2]}") from exc
    results = {post.post_id: post for post in batch.posts if post.post_id in expected_ids}
    return results


@dataclass
class ClaudeClassifier:
    client: anthropic.AsyncAnthropic
    model: str = "claude-opus-5"
    effort: str = "low"
    max_tokens: int = 16000
    supports_batch: bool = True
    usage: Usage = field(default_factory=Usage)

    kind = "claude"  # метка в listings.classifier и ключ уверенности в repo.CONFIDENCE
    background_refine = False  # быстрый и дешёвый: сомнительные посты разбираем сразу, без фоновой очереди

    def request_params(self, posts: list[PostInput]) -> dict[str, Any]:
        posts_json = json.dumps([p.as_json() for p in posts], ensure_ascii=False, indent=None)
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": user_message(posts_json)}],
            "output_config": {"effort": self.effort, "format": {"type": "json_schema", "schema": RESPONSE_SCHEMA}},
        }

    def _supports_effort(self) -> bool:
        return not self.model.startswith("claude-haiku")

    def _finalize_params(self, params: dict[str, Any]) -> dict[str, Any]:
        if not self._supports_effort():
            params["output_config"] = {"format": params["output_config"]["format"]}
        return params

    def _parse_message(self, message: Any, posts: list[PostInput]) -> dict[str, PostResult]:
        if message.stop_reason == "refusal":
            raise RetryableSplit("модель отказалась отвечать")
        if message.stop_reason == "max_tokens":
            raise RetryableSplit("ответ обрезан по max_tokens")
        text = next((block.text for block in message.content if block.type == "text"), "")
        return parse_response_text(text, [p.post_id for p in posts])

    async def _call(self, posts: list[PostInput]) -> dict[str, PostResult]:
        params = self._finalize_params(self.request_params(posts))
        if self.model in FALLBACK_MODELS:
            message = await self.client.beta.messages.create(betas=[FALLBACK_BETA], fallbacks="default", **params)
        else:
            message = await self.client.messages.create(**params)
        self.usage.add(message.usage)
        return self._parse_message(message, posts)

    async def classify(self, posts: list[PostInput]) -> tuple[dict[str, PostResult], dict[str, str]]:
        """Возвращает (результаты по post_id, ошибки по post_id). Проблемные пачки делятся пополам."""
        if not posts:
            return {}, {}
        try:
            results = await self._call(posts)
        except RetryableSplit as exc:
            if len(posts) == 1:
                return {}, {posts[0].post_id: str(exc)}
            log.warning("claude: %s — делю пачку из %d постов", exc, len(posts))
            mid = len(posts) // 2
            left, right = await asyncio.gather(self.classify(posts[:mid]), self.classify(posts[mid:]))
            return {**left[0], **right[0]}, {**left[1], **right[1]}
        missing = [p for p in posts if p.post_id not in results]
        errors: dict[str, str] = {}
        if missing and len(missing) < len(posts):
            retry_results, errors = await self.classify(missing)
            results.update(retry_results)
        elif missing:
            errors = {p.post_id: "модель не вернула результат для поста" for p in missing}
        return results, errors

    # ── Batch API: массовый разбор со скидкой 50% ───────────────────────────

    async def submit_batch(self, groups: list[list[PostInput]]) -> str:
        requests = [
            {"custom_id": f"g{idx}", "params": self._finalize_params(self.request_params(group))}
            for idx, group in enumerate(groups)
        ]
        batch = await self.client.messages.batches.create(requests=requests)
        return batch.id

    async def batch_status(self, batch_id: str) -> str:
        batch = await self.client.messages.batches.retrieve(batch_id)
        return batch.processing_status

    async def collect_batch(
        self, batch_id: str, groups: dict[str, list[PostInput]]
    ) -> tuple[dict[str, PostResult], dict[str, str]]:
        results: dict[str, PostResult] = {}
        errors: dict[str, str] = {}
        stream = await self.client.messages.batches.results(batch_id)
        async for entry in stream:
            posts = groups.get(entry.custom_id, [])
            if entry.result.type != "succeeded":
                for post in posts:
                    errors[post.post_id] = f"batch: {entry.result.type}"
                continue
            message = entry.result.message
            self.usage.add(message.usage)
            try:
                parsed = self._parse_message(message, posts)
            except RetryableSplit as exc:
                for post in posts:
                    errors[post.post_id] = str(exc)
                continue
            results.update(parsed)
            for post in posts:
                if post.post_id not in parsed:
                    errors[post.post_id] = "модель не вернула результат для поста"
        return results, errors
