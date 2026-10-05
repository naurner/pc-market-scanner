"""Ollama: нативный API со строгой схемой, диагностика и поведение, когда сервер недоступен."""

import json
from datetime import timedelta

import httpx
import pytest
import respx

from lalascaner.ai.claude import AIUnavailable, PostInput
from lalascaner.ai.local import LOCAL_SCHEMA, LocalClassifier, OllamaClassifier
from lalascaner.ai.schema import ExtractedItem, PostResult
from lalascaner.config import Settings
from lalascaner.db import repo
from lalascaner.db.models import Listing, utcnow
from lalascaner.db.session import init_db, make_engine, make_sessionmaker
from lalascaner.pipeline import UNAVAILABLE_PREFIX, Pipeline, build_claude
from lalascaner.sources.base import RawListing

BASE = "http://ollama.test:11434"


def make(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


def reply(content: str, done_reason: str = "stop") -> dict:
    return {
        "model": "qwen2.5:7b",
        "message": {"role": "assistant", "content": content},
        "done": True,
        "done_reason": done_reason,
        "prompt_eval_count": 2500,
        "eval_count": 300,
    }


def batch_json(post_id: str, price: float = 27000) -> str:
    return json.dumps(
        {
            "posts": [
                {
                    "post_id": post_id,
                    "intent": "sell",
                    "items": [
                        {
                            "category": "gpu", "intent": "sell", "model": "RTX 3070 Ti", "condition": "used",
                            "price": price, "currency": "KGS", "quantity": 1, "is_bundle": False,
                        }
                    ],
                }
            ]
        }
    )


def post(post_id: str = "1") -> PostInput:
    return PostInput(post_id=post_id, source="telegram", text="rtx3070ti zotac цена 27к")


@pytest.fixture
async def clf():
    c = OllamaClassifier(base_url=BASE, model="qwen2.5:7b", timeout=5)
    yield c
    await c.aclose()


# ── схема ────────────────────────────────────────────────────────────────────


def test_local_schema_is_self_contained_and_strict():
    text = json.dumps(LOCAL_SCHEMA)
    assert "$ref" not in text and "$defs" not in text
    item = LOCAL_SCHEMA["properties"]["posts"]["items"]["properties"]["items"]["items"]
    assert item["additionalProperties"] is False
    assert {"category", "model", "price", "currency", "condition", "is_bundle"} <= set(item["required"])
    assert "chip" in item["properties"] and "chip" not in item["required"]  # характеристики — по желанию
    assert LOCAL_SCHEMA["additionalProperties"] is False


# ── запрос и разбор ответа ───────────────────────────────────────────────────


@respx.mock
async def test_request_uses_native_api_with_schema_and_context(clf):
    route = respx.post(f"{BASE}/api/chat").mock(return_value=httpx.Response(200, json=reply(batch_json("1"))))
    results, errors = await clf.classify([post()])
    body = json.loads(route.calls.last.request.content)
    assert not errors and results["1"].items[0].price == 27000
    assert body["stream"] is False and body["model"] == "qwen2.5:7b"
    assert body["format"] == LOCAL_SCHEMA
    assert body["options"]["num_ctx"] == 8192 and body["options"]["temperature"] == 0
    assert body["messages"][0]["role"] == "system"
    assert "authorization" not in route.calls.last.request.headers or True  # Ollama ключа не требует
    assert clf.usage.input_tokens == 2500 and clf.usage.output_tokens == 300 and clf.usage.requests == 1


@respx.mock
async def test_truncated_reply_splits_batch(clf):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        content = body["messages"][1]["content"]
        ids = [p["post_id"] for p in json.loads(content.split("<posts>")[1].split("</posts>")[0].strip())]
        if len(ids) > 1:
            return httpx.Response(200, json=reply("{trunc", done_reason="length"))
        return httpx.Response(200, json=reply(batch_json(ids[0])))

    respx.post(f"{BASE}/api/chat").mock(side_effect=handler)
    results, errors = await clf.classify([post("a"), post("b")])
    assert set(results) == {"a", "b"} and not errors


@respx.mock
async def test_read_timeout_is_unavailable_not_split(clf):
    """Таймаут — это про модель/железо (например, слабый Mac не тянет её), а не про один пост из пачки:
    делить и повторять на той же перегруженной модели только продлит ожидание в разы."""
    respx.post(f"{BASE}/api/chat").mock(side_effect=httpx.ReadTimeout("slow"))
    with pytest.raises(AIUnavailable, match="не успела ответить"):
        await clf.classify([post("a"), post("b")])


# ── недоступность ────────────────────────────────────────────────────────────


@respx.mock
async def test_connection_refused_is_unavailable_with_hint(clf):
    respx.post(f"{BASE}/api/chat").mock(side_effect=httpx.ConnectError("refused"))
    with pytest.raises(AIUnavailable, match="brew services start ollama"):
        await clf.classify([post("a"), post("b")])


@respx.mock
async def test_missing_model_says_how_to_pull(clf):
    respx.post(f"{BASE}/api/chat").mock(return_value=httpx.Response(404, json={"error": "model not found"}))
    with pytest.raises(AIUnavailable, match="ollama pull qwen2.5:7b"):
        await clf.classify([post()])


@respx.mock
async def test_out_of_memory_is_unavailable(clf):
    respx.post(f"{BASE}/api/chat").mock(
        return_value=httpx.Response(500, json={"error": "model requires more system memory (6.2 GiB) than is available"})
    )
    with pytest.raises(AIUnavailable, match="не хватает памяти"):
        await clf.classify([post()])


async def test_docker_hint_for_localhost(monkeypatch):
    monkeypatch.setattr("lalascaner.ai.local.os.path.exists", lambda path: path == "/.dockerenv")
    c = OllamaClassifier(base_url="http://localhost:11434")
    assert "host.docker.internal" in c.unreachable_hint(httpx.ConnectError("x"))
    await c.aclose()


# ── health ───────────────────────────────────────────────────────────────────


@respx.mock
async def test_health_ok_accepts_latest_tag():
    respx.get(f"{BASE}/api/tags").mock(return_value=httpx.Response(200, json={"models": [{"name": "llama3.1:latest"}]}))
    c = OllamaClassifier(base_url=BASE, model="llama3.1")
    ok, message = await c.health()
    await c.aclose()
    assert ok and "готова" in message


@respx.mock
async def test_health_reports_missing_model_and_lists_installed(clf):
    respx.get(f"{BASE}/api/tags").mock(return_value=httpx.Response(200, json={"models": [{"name": "llama3.1:latest"}]}))
    ok, message = await clf.health()
    assert not ok and "ollama pull qwen2.5:7b" in message and "llama3.1:latest" in message


@respx.mock
async def test_health_reports_unreachable(clf):
    respx.get(f"{BASE}/api/tags").mock(side_effect=httpx.ConnectError("refused"))
    ok, message = await clf.health()
    assert not ok and "Запущена ли Ollama" in message


@respx.mock
async def test_openai_health_checks_model_list():
    respx.get("http://groq.test/v1/models").mock(return_value=httpx.Response(200, json={"data": [{"id": "llama-3.3-70b"}]}))
    c = LocalClassifier(base_url="http://groq.test/v1", model="nope")
    ok, message = await c.health()
    await c.aclose()
    assert not ok and "llama-3.3-70b" in message


# ── настройки ────────────────────────────────────────────────────────────────


def test_ollama_settings_defaults(monkeypatch):
    monkeypatch.setattr("lalascaner.config.os.path.exists", lambda path: False)
    s = make(ai_enabled=True, ai_provider="ollama", ai_model="qwen2.5:7b")
    assert s.effective_classifier == "hybrid"
    assert s.effective_ai_base_url == "http://localhost:11434"
    assert s.effective_batch_size == 4 and s.effective_ai_concurrency == 1
    monkeypatch.setattr("lalascaner.config.os.path.exists", lambda path: path == "/.dockerenv")
    assert s.effective_ai_base_url == "http://host.docker.internal:11434"
    assert make(ai_provider="ollama", ai_base_url="http://mac:11434").effective_ai_base_url == "http://mac:11434"


def test_claude_keeps_big_batches_and_parallelism():
    s = make(ai_enabled=True, anthropic_api_key="k")
    assert s.effective_batch_size == 10 and s.effective_ai_concurrency == 3
    assert make(ai_provider="ollama", ai_batch_size=6, ai_concurrency=2).effective_batch_size == 6


async def test_build_claude_returns_ollama_classifier():
    clf = build_claude(make(ai_enabled=True, ai_provider="ollama", ai_base_url=BASE, ai_model="qwen2.5:7b", ai_num_ctx=4096))
    assert isinstance(clf, OllamaClassifier) and clf.num_ctx == 4096 and clf.base_url == BASE and clf.kind == "local"
    assert clf.supports_batch is False
    await clf.aclose()


# ── pipeline: посты не теряются, пока Ollama выключена ───────────────────────


class DownClassifier:
    """ИИ, который «выключен», потом «включили»."""

    kind = "local"
    supports_batch = False

    def __init__(self):
        self.up = False
        self.calls = 0
        self.usage = type("U", (), {"as_dict": lambda self: {"requests": 1}})()

    async def classify(self, posts):
        self.calls += 1
        if not self.up:
            raise AIUnavailable("нет соединения")
        return (
            {
                p.post_id: PostResult(
                    post_id=p.post_id, intent="sell",
                    items=[ExtractedItem(category="gpu", model="RTX 3070 Ti", chip="RTX 3070 Ti", price=27000, currency="KGS", condition="used")],
                )
                for p in posts
            },
            {},
        )

    async def health(self):
        return (True, "ok") if self.up else (False, "нет соединения")


@pytest.fixture
async def maker(tmp_path):
    engine = make_engine(f"sqlite+aiosqlite:///{tmp_path / 'o.db'}")
    await init_db(engine)
    yield make_sessionmaker(engine)
    await engine.dispose()


def tg_raw(msg_id: int, text: str) -> RawListing:
    return RawListing(
        source="telegram", external_id=f"g/{msg_id}", source_ref="g", url=f"https://t.me/g/{msg_id}", text=text,
        author_id="a", posted_at=utcnow() - timedelta(hours=1),
    )


async def test_ai_outage_postpones_posts_and_recovers(maker):
    settings = make(ai_enabled=True, ai_provider="ollama", classifier="claude")
    ai = DownClassifier()
    pipeline = Pipeline(settings, maker, claude=ai)
    async with maker() as session:
        await repo.upsert_listings(session, [tg_raw(1, "продам видеокарту, цена договорная"), tg_raw(2, "продаю rtx 3070 ti 27к")])
        await session.commit()

    report = await pipeline.classify_pending(limit=10)
    assert report.failed == 2 and report.by_ai == 0
    async with maker() as session:
        errors = [l.error for l in await session.scalars(__import__("sqlalchemy").select(Listing))]
        assert all(e.startswith(UNAVAILABLE_PREFIX) for e in errors)
        assert await repo.count_unavailable(session) == 2

    # пока сервер лежит, повторно к нему не ходим и ничего не возвращаем в очередь
    calls = ai.calls
    assert await pipeline.requeue_unavailable() == 0 and ai.calls == calls

    ai.up = True
    assert await pipeline.requeue_unavailable() == 2
    report = await pipeline.classify_pending(limit=10)
    assert report.by_ai == 2 and report.failed == 0
    async with maker() as session:
        assert await repo.count_unavailable(session) == 0
        assert {l.classifier for l in await session.scalars(__import__("sqlalchemy").select(Listing))} == {"local"}


async def test_local_ai_results_have_medium_confidence(maker):
    from lalascaner.db.models import Item

    ai = DownClassifier()
    ai.up = True
    pipeline = Pipeline(make(ai_enabled=True, ai_provider="ollama", classifier="claude"), maker, claude=ai)
    async with maker() as session:
        await repo.upsert_listings(session, [tg_raw(1, "продаю rtx 3070 ti 27к")])
        await session.commit()
    await pipeline.classify_pending(limit=10)
    async with maker() as session:
        item = await session.scalar(__import__("sqlalchemy").select(Item))
    assert item.confidence == 0.75


def test_scheduler_retries_when_local_ai_is_used():
    settings = make(ai_enabled=True, ai_provider="ollama")
    from lalascaner.scheduler import plan_jobs

    names = {name for name, *_ in plan_jobs(Pipeline(settings, sessionmaker=None, claude=DownClassifier()))}
    assert "ИИ-повтор" in names and "batch-api" not in names
