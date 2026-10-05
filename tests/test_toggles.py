"""Режим «только lalafo, без ИИ»: ничего платного и никаких Telegram-источников."""

import pytest

from lalascaner.ai.claude import ClaudeClassifier
from lalascaner.ai.local import LocalClassifier
from lalascaner.bot.app import commands_for
from lalascaner.bot.handlers import help_text, stats_text
from lalascaner.config import Settings
from lalascaner.pipeline import Pipeline, build_claude
from lalascaner.scheduler import plan_jobs


def make(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


def test_defaults_read_telegram_but_ai_is_opt_in():
    s = make(anthropic_api_key="sk-test", classifier="claude")
    assert s.telegram_enabled and not s.ai_enabled
    assert s.effective_classifier == "rules"
    assert build_claude(s) is None


def test_ai_can_be_enabled_back():
    s = make(ai_enabled=True, anthropic_api_key="sk-test")
    assert s.effective_classifier == "hybrid"
    assert build_claude(s) is not None


def test_scheduler_skips_telegram_and_batch_api_when_disabled():
    names = {name for name, *_ in plan_jobs(Pipeline(make(telegram_enabled=False), sessionmaker=None))}
    assert "telegram" not in names and "batch-api" not in names
    assert {"lalafo", "lalafo-полный", "классификация", "статистика"} <= names

    enabled = {name for name, *_ in plan_jobs(Pipeline(make(telegram_enabled=True), sessionmaker=None))}
    assert "telegram" in enabled


def test_bot_hides_telegram_and_ai_when_disabled():
    s = make(telegram_enabled=False)
    assert "/channels" not in help_text(s) and "lalafo + Telegram" not in help_text(s)
    assert "channels" not in {c.command for c in commands_for(s)}
    stats = {
        "listings_by_source": {"lalafo": 10, "telegram": 5},
        "listings_by_status": {"new": 1, "batch": 2},
        "items_by_category": {"gpu": 3},
        "items_total": 3,
        "models_tracked": 2,
        "deals_24h": 1,
    }
    text = stats_text(s, stats, usage={"requests": 5}, batches=[{"id": "x"}])
    assert "Telegram" not in text and "Batch API" not in text and "Запросов" not in text
    assert "ИИ</b>: выключен" in text

    on = make(telegram_enabled=True, ai_enabled=True)
    assert "/channels" in help_text(on)
    text_on = stats_text(on, stats, usage={"requests": 5}, batches=[{"id": "x"}])
    assert "Telegram 5" in text_on and "Batch API в обработке: 1" in text_on


# ── бесплатный ИИ (Ollama/Groq/OpenRouter через AI_PROVIDER=openai) ─────────────


def test_openai_provider_ready_only_with_base_url():
    s = make(ai_enabled=True, ai_provider="openai")
    assert s.effective_classifier == "rules"  # адрес сервера не задан
    assert build_claude(s) is None

    ready = make(ai_enabled=True, ai_provider="openai", ai_base_url="http://localhost:11434/v1")
    assert ready.effective_classifier == "hybrid"


def test_build_claude_returns_local_classifier_for_openai_provider():
    s = make(ai_enabled=True, ai_provider="openai", ai_base_url="http://localhost:11434/v1", ai_api_key="k", ai_model="llama3.1")
    clf = build_claude(s)
    assert isinstance(clf, LocalClassifier)
    assert not isinstance(clf, ClaudeClassifier)
    assert clf.base_url == "http://localhost:11434/v1" and clf.model == "llama3.1" and clf.api_key == "k"
    assert clf.supports_batch is False


def test_claude_classifier_supports_batch_by_default():
    s = make(ai_enabled=True, anthropic_api_key="sk-test")
    clf = build_claude(s)
    assert isinstance(clf, ClaudeClassifier)
    assert clf.supports_batch is True


def test_scheduler_skips_batch_api_for_local_classifier():
    settings = make(ai_enabled=True, ai_provider="openai", ai_base_url="http://x")
    local = LocalClassifier(base_url="http://x")
    names = {name for name, *_ in plan_jobs(Pipeline(settings, sessionmaker=None, claude=local))}
    assert "batch-api" not in names


async def test_submit_ai_batch_refuses_for_local_classifier():
    settings = make(ai_enabled=True, ai_provider="openai", ai_base_url="http://x")
    local = LocalClassifier(base_url="http://x")
    pipeline = Pipeline(settings, sessionmaker=None, claude=local)
    with pytest.raises(RuntimeError, match="AI_PROVIDER=claude"):
        await pipeline.submit_ai_batch()
    await local.aclose()
