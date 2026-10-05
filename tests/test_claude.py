import json
from types import SimpleNamespace

from lalascaner.ai.claude import FALLBACK_BETA, RESPONSE_SCHEMA, ClaudeClassifier, PostInput


def post_json(post_id: str, price: float = 27000) -> dict:
    return {
        "post_id": post_id,
        "intent": "sell",
        "items": [
            {
                "category": "gpu", "intent": "sell", "brand": "Zotac", "model": "Zotac RTX 3070 Ti",
                "chip": "RTX 3070 Ti", "vram_gb": 8, "cpu_model": None, "gpu_model": None, "memory_type": None,
                "capacity_gb": None, "modules": None, "speed_mhz": None, "interface": None, "form_factor": None,
                "wattage": None, "efficiency": None, "chipset": None, "socket": None, "diagonal_in": None,
                "resolution": None, "refresh_hz": None, "ram_gb": None, "storage_gb": None, "condition": "used",
                "price": price, "currency": "KGS", "quantity": 1, "is_bundle": False,
            }
        ],
    }


def message(posts: list[dict], stop_reason: str = "end_turn"):
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=json.dumps({"posts": posts}))],
        usage=SimpleNamespace(input_tokens=100, output_tokens=50, cache_read_input_tokens=900, cache_creation_input_tokens=0),
    )


class FakeMessages:
    def __init__(self, responder):
        self.calls = []
        self._responder = responder

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._responder(kwargs)


def ids_in(kwargs) -> list[str]:
    content = kwargs["messages"][0]["content"]
    payload = content.split("<posts>\n", 1)[1].rsplit("\n</posts>", 1)[0]
    return [p["post_id"] for p in json.loads(payload)]


def fake_client(responder):
    messages = FakeMessages(responder)
    beta_messages = FakeMessages(responder)
    return SimpleNamespace(messages=messages, beta=SimpleNamespace(messages=beta_messages)), messages, beta_messages


def posts(*ids):
    return [PostInput(post_id=i, source="telegram", text=f"rtx 3070 ti {i}") for i in ids]


def test_schema_is_strict():
    def walk(node):
        if isinstance(node, dict):
            assert "title" not in node
            if node.get("type") == "object" and "properties" in node:
                assert node["additionalProperties"] is False
                assert set(node["required"]) == set(node["properties"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(RESPONSE_SCHEMA)


async def test_opus_uses_fallbacks_effort_and_cached_system():
    client, plain, beta = fake_client(lambda kw: message([post_json(i) for i in ids_in(kw)]))
    clf = ClaudeClassifier(client=client, model="claude-opus-5", effort="low")
    results, errors = await clf.classify(posts("1", "2"))
    assert errors == {}
    assert set(results) == {"1", "2"}
    assert results["1"].items[0].chip == "RTX 3070 Ti"
    assert plain.calls == []
    call = beta.calls[0]
    assert call["betas"] == [FALLBACK_BETA] and call["fallbacks"] == "default"
    assert call["output_config"]["effort"] == "low"
    assert call["output_config"]["format"]["type"] == "json_schema"
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert clf.usage.requests == 1 and clf.usage.cache_read_input_tokens == 900


async def test_haiku_uses_plain_endpoint_without_effort():
    client, plain, beta = fake_client(lambda kw: message([post_json(i) for i in ids_in(kw)]))
    clf = ClaudeClassifier(client=client, model="claude-haiku-4-5")
    results, _ = await clf.classify(posts("1"))
    assert "1" in results and beta.calls == []
    assert "effort" not in plain.calls[0]["output_config"]


async def test_truncated_batch_is_split():
    def responder(kw):
        ids = ids_in(kw)
        if len(ids) > 1:
            return message([], stop_reason="max_tokens")
        return message([post_json(ids[0])])

    client, _, beta = fake_client(responder)
    clf = ClaudeClassifier(client=client)
    results, errors = await clf.classify(posts("1", "2", "3"))
    assert set(results) == {"1", "2", "3"} and errors == {}
    assert len(beta.calls) >= 4


async def test_missing_post_is_retried_alone_and_error_reported():
    def responder(kw):
        ids = ids_in(kw)
        return message([post_json(i) for i in ids if i != "bad"])

    client, _, _ = fake_client(responder)
    clf = ClaudeClassifier(client=client)
    results, errors = await clf.classify(posts("ok", "bad"))
    assert set(results) == {"ok"}
    assert set(errors) == {"bad"}


async def test_real_sdk_serializes_request():
    import anthropic
    import httpx2

    captured = {}

    def handler(request):
        body = json.loads(request.content)
        captured["body"], captured["beta"] = body, request.headers.get("anthropic-beta")
        out = {"posts": [{"post_id": "1", "intent": "other", "items": []}]}
        return httpx2.Response(
            200,
            json={
                "id": "msg_1", "type": "message", "role": "assistant", "model": body["model"],
                "content": [{"type": "text", "text": json.dumps(out)}], "stop_reason": "end_turn",
                "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 1},
            },
        )

    client = anthropic.AsyncAnthropic(
        api_key="test", http_client=anthropic.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handler))
    )
    results, errors = await ClaudeClassifier(client=client).classify(posts("1"))
    assert results["1"].intent == "other" and errors == {}
    assert captured["beta"] == FALLBACK_BETA
    assert captured["body"]["fallbacks"] == "default"
    assert captured["body"]["output_config"]["effort"] == "low"


async def test_refusal_on_single_post_becomes_error():
    client, _, _ = fake_client(lambda kw: message([], stop_reason="refusal"))
    clf = ClaudeClassifier(client=client)
    results, errors = await clf.classify(posts("1"))
    assert results == {} and "отказ" in errors["1"]
