"""Пользователь копирует .env.example в .env как есть — он обязан разбираться без ошибок."""

import collections
import re
from pathlib import Path

from lalascaner.config import Settings

ENV_EXAMPLE = Path(__file__).parent.parent / ".env.example"


def test_env_example_has_no_duplicate_keys():
    keys = re.findall(r"^([A-Z][A-Z0-9_]*)=", ENV_EXAMPLE.read_text(encoding="utf-8"), re.M)
    assert [k for k, n in collections.Counter(keys).items() if n > 1] == []


def test_env_example_parses_and_enables_telegram_and_local_ai():
    s = Settings(_env_file=ENV_EXAMPLE)
    assert s.telegram_enabled and s.channel_list == ["compshop_kg"]
    assert s.blocked_topic_ids == {419, 4819, 6460, 6523, 6547}
    assert s.ai_enabled and s.ai_provider == "ollama" and s.ai_model == "qwen2.5:7b"
    assert s.effective_classifier == "hybrid"
    assert s.effective_batch_size == 4 and s.effective_ai_concurrency == 1  # пустые значения = авто


def test_every_key_in_env_example_is_a_known_setting():
    known = {name.upper() for name in Settings.model_fields}
    keys = set(re.findall(r"^([A-Z][A-Z0-9_]*)=", ENV_EXAMPLE.read_text(encoding="utf-8"), re.M))
    assert keys - known == set()
