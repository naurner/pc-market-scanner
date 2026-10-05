from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


class Settings(BaseSettings):
    # env_ignore_empty: «AI_BATCH_SIZE=» в .env означает «по умолчанию», а не ошибку разбора числа
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", env_ignore_empty=True)

    bot_token: str = ""
    bot_allowed_users: str = ""

    # Главные переключатели. Telegram-чат читаем по умолчанию; ИИ включается явно (AI_ENABLED=true).
    ai_enabled: bool = False
    telegram_enabled: bool = True

    # claude — Anthropic API (платно). ollama — Ollama у себя дома (бесплатно, нативный API со строгой
    # JSON-схемой). openai — любой OpenAI-совместимый сервер: Groq, OpenRouter (бесплатный тариф в облаке), LM Studio.
    ai_provider: Literal["claude", "ollama", "openai"] = "claude"
    anthropic_api_key: str = ""
    # Адрес сервера. Для ollama можно не задавать: в Docker это http://host.docker.internal:11434, иначе localhost.
    ai_base_url: str = ""
    # Ключ провайдера (только для openai; Ollama ключа не требует).
    ai_api_key: str = "not-needed"
    # Сколько секунд ждать ответ локальной модели на одну пачку (на слабом железе первая пачка долгая: загрузка модели).
    ai_timeout: float = 300
    # Размер контекста Ollama в токенах: промпт (~2.5 тыс.) + пачка постов + ответ.
    ai_num_ctx: int = 8192
    # Сколько пачек обрабатывать одновременно. 0 = авто: 3 для Claude, 1 для локальных моделей.
    ai_concurrency: int = 0
    # Локальная модель медленная (минута на пачку), поэтому сначала всё разбирают правила, а ИИ фоном
    # дорабатывает сомнительное: посты Telegram и объявления lalafo не старше стольких дней.
    ai_refine_max_age_days: int = 3
    classifier: Literal["", "claude", "rules", "hybrid"] = ""
    ai_model: str = "claude-opus-5"
    ai_effort: Literal["low", "medium", "high", "xhigh", "max"] = "low"
    # Постов в одном запросе. 0 = авто: 10 для Claude, 4 для локальных моделей (маленькие модели теряются в длинных пачках).
    ai_batch_size: int = 0
    # Если в очереди больше стольких постов — отправляем их через Batch API (дешевле вдвое, ответ до часа).
    # Работает только для ai_provider=claude — у бесплатных провайдеров такого API нет.
    ai_batch_api_threshold: int = 300

    # Векторы объявлений (похожие, поиск по смыслу, перезаливы, оценка редких моделей, «за сколько продать»).
    # Работают через ту же Ollama, что и ИИ (AI_PROVIDER=ollama). Модель скачивается сама при первом запуске.
    embed_enabled: bool = True
    embed_model: str = "bge-m3"
    embed_batch: int = 32

    telegram_channels: str = "compshop_kg"
    # Id тем форума compshop_kg, которые пропускаем целиком — там нет объявлений, только спам/оффтоп:
    # 419 Правила, 4819 Blacklist, 6460 Болтовня, 6523 (по указанию владельца), 6547 Ищу/Обмен.
    # Узнать id темы: открыть её в Telegram → «⋮» → Копировать ссылку — число в конце ссылки.
    # Работает только для прямых постов в теме, не для веток внутри неё (см. telegram_web._reply_target_id).
    telegram_blocked_topics: str = "419,4819,6460,6523,6547"
    lalafo_poll_minutes: int = 10
    lalafo_max_pages_per_poll: int = 3
    # Полный обход всех страниц: обновляет «актуальность» объявлений (проданные перестают попадаться)
    lalafo_full_crawl_hours: int = 6
    # Объявление lalafo считается активным, если встречалось в ленте за последние N часов
    lalafo_active_hours: int = 14
    telegram_active_days: int = 7
    telegram_poll_minutes: int = 5
    # Сколько истории подтянуть при первом подключении: сообщений для групп, страниц (по 20 постов) для каналов
    telegram_backfill_messages: int = 1500
    telegram_backfill_pages: int = 10
    telegram_group_scan_limit: int = 600

    stats_window_days: int = 60
    min_samples: int = 5
    # cheapest — сделка, если дешевле самого дешёвого активного объявления той же модели/версии;
    # median — старый режим: на DEAL_THRESHOLD ниже медианы
    deal_mode: Literal["cheapest", "median"] = "cheapest"
    # Сколько активных объявлений должно быть в сравнении, чтобы «самое дешёвое» что-то значило
    deal_min_competitors: int = 3
    # Насколько (в сомах) нужно быть дешевле самого дешёвого. 0 — достаточно быть дешевле хоть на сом
    deal_min_gap_kgs: float = 0
    # ...и не меньше чем на столько процентов (отсекает «на 10 сом дешевле»). 0 — любой отрыв
    deal_min_gap_percent: float = 3
    # Уведомлять и о рекорде по версии (Palit Dual), а не только по модели целиком
    deal_variant_alerts: bool = True
    deal_threshold: float = 0.2
    deal_min_saving_kgs: float = 1500
    # Пока в очереди на разбор больше стольких объявлений (первичная загрузка), уведомления копятся
    notify_warmup_pending: int = 200
    suspicious_ratio: float = 0.35
    notify_max_age_hours: int = 48

    database_url: str = "sqlite+aiosqlite:///data/lalascaner.db"
    log_level: str = "INFO"
    # Каталог git-репозитория для кнопки «Обновить с GitHub» в боте. "." — рабочий каталог процесса
    # (в Docker это /app, если docker-compose.yml монтирует туда весь репозиторий, а не только код).
    bot_repo_path: str = "."

    @property
    def allowed_user_ids(self) -> set[int]:
        return {int(x) for x in _split_csv(self.bot_allowed_users)}

    @property
    def channel_list(self) -> list[str]:
        return [c.lstrip("@").lower() for c in _split_csv(self.telegram_channels)]

    @property
    def blocked_topic_ids(self) -> set[int]:
        return {int(x) for x in _split_csv(self.telegram_blocked_topics) if x.lstrip("-").isdigit()}

    @property
    def effective_classifier(self) -> str:
        if not self.ai_enabled:
            return "rules"  # ИИ выключен — никаких запросов к Claude и расходов
        if self.classifier:
            return self.classifier
        ready = {
            "claude": bool(self.anthropic_api_key),
            "openai": bool(self.ai_base_url),
            "ollama": True,  # адрес подбирается сам; недоступность Ollama обрабатывается на лету
        }[self.ai_provider]
        return "hybrid" if ready else "rules"

    @property
    def effective_embed(self) -> bool:
        return self.embed_enabled and self.ai_enabled and self.ai_provider == "ollama"

    @property
    def effective_ai_base_url(self) -> str:
        if self.ai_base_url:
            return self.ai_base_url
        if self.ai_provider == "ollama":
            return "http://host.docker.internal:11434" if os.path.exists("/.dockerenv") else "http://localhost:11434"
        return ""

    @property
    def effective_batch_size(self) -> int:
        return self.ai_batch_size or (10 if self.ai_provider == "claude" else 4)

    @property
    def effective_ai_concurrency(self) -> int:
        return self.ai_concurrency or (3 if self.ai_provider == "claude" else 1)


@lru_cache
def get_settings() -> Settings:
    return Settings()
