"""LocalClassifier — любой OpenAI-совместимый сервер (Ollama/Groq/OpenRouter)."""

import json

import httpx
import pytest
import respx

from lalascaner.ai.claude import PostInput
from lalascaner.ai.local import LocalClassifier

BASE = "http://localhost:11434/v1"


def _completion(content: str, finish_reason: str = "stop", usage: dict | None = None) -> dict:
    return {
        "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": finish_reason}],
        "usage": usage or {"prompt_tokens": 100, "completion_tokens": 50},
    }


def _batch_json(post_id: str, category: str = "gpu") -> str:
    return json.dumps(
        {
            "posts": [
                {
                    "post_id": post_id,
                    "intent": "sell",
                    "items": [
                        {
                            "category": category,
                            "intent": "sell",
                            "model": "RTX 3060",
                            "chip": "RTX 3060",
                            "vram_gb": 12,
                            "condition": "used",
                            "price": 20000,
                            "currency": "KGS",
                            "quantity": 1,
                            "is_bundle": False,
                        }
                    ],
                }
            ]
        }
    )


@pytest.fixture
def classifier():
    clf = LocalClassifier(base_url=BASE, api_key="ollama", model="llama3.1")
    yield clf


async def aclose(clf: LocalClassifier) -> None:
    await clf.aclose()


@respx.mock
async def test_classify_success_parses_result_and_tracks_usage(classifier):
    route = respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(200, json=_completion(_batch_json("a")))
    )
    posts = [PostInput(post_id="a", source="lalafo", text="rtx 3060 12gb, 20000 сом")]
    results, errors = await classifier.classify(posts)
    await aclose(classifier)

    assert route.called
    request_body = json.loads(route.calls.last.request.content)
    assert request_body["model"] == "llama3.1"
    assert request_body["response_format"] == {"type": "json_object"}
    assert not errors
    assert results["a"].items[0].chip == "RTX 3060"
    assert classifier.usage.requests == 1
    assert classifier.usage.input_tokens == 100
    assert classifier.usage.output_tokens == 50


@respx.mock
async def test_auth_header_sent(classifier):
    route = respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(200, json=_completion(_batch_json("a")))
    )
    await classifier.classify([PostInput(post_id="a", source="lalafo", text="x")])
    await aclose(classifier)
    assert route.calls.last.request.headers["authorization"] == "Bearer ollama"


@respx.mock
async def test_invalid_json_splits_batch_and_recovers(classifier):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        posts_json = body["messages"][1]["content"].split("<posts>")[1].split("</posts>")[0].strip()
        ids = [p["post_id"] for p in json.loads(posts_json)]
        # Пачка из 2 постов — сервер отвечает мусором; после деления пополам — по одному успешно.
        if len(ids) > 1:
            return httpx.Response(200, json=_completion("не json"))
        return httpx.Response(200, json=_completion(_batch_json(ids[0])))

    respx.post(f"{BASE}/chat/completions").mock(side_effect=handler)
    posts = [
        PostInput(post_id="a", source="lalafo", text="rtx 3060 12gb, 20000 сом"),
        PostInput(post_id="b", source="lalafo", text="rtx 3060 12gb, 21000 сом"),
    ]
    results, errors = await classifier.classify(posts)
    await aclose(classifier)
    assert set(results) == {"a", "b"}
    assert not errors


@respx.mock
async def test_finish_reason_length_is_treated_as_retryable(classifier):
    respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(200, json=_completion("{truncated", finish_reason="length"))
    )
    results, errors = await classifier.classify([PostInput(post_id="a", source="lalafo", text="x")])
    await aclose(classifier)
    assert results == {}
    assert "a" in errors


@respx.mock
async def test_missing_post_id_in_response_is_reported_as_error(classifier):
    respx.post(f"{BASE}/chat/completions").mock(
        return_value=httpx.Response(200, json=_completion(_batch_json("a")))
    )
    results, errors = await classifier.classify(
        [PostInput(post_id="a", source="lalafo", text="x"), PostInput(post_id="b", source="lalafo", text="y")]
    )
    await aclose(classifier)
    assert "a" in results
    assert "b" in errors


@respx.mock
async def test_server_error_produces_error_not_exception(classifier):
    respx.post(f"{BASE}/chat/completions").mock(return_value=httpx.Response(500, text="oops"))
    results, errors = await classifier.classify([PostInput(post_id="a", source="lalafo", text="x")])
    await aclose(classifier)
    assert results == {}
    assert "a" in errors


def test_supports_batch_is_false(classifier):
    assert classifier.supports_batch is False


async def test_empty_posts_short_circuits(classifier):
    results, errors = await classifier.classify([])
    await aclose(classifier)
    assert results == {} and errors == {}
