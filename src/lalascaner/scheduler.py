"""Фоновые периодические задачи."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta

from lalascaner.db import repo
from lalascaner.db.models import utcnow
from lalascaner.pipeline import Pipeline

log = logging.getLogger(__name__)


async def periodic(name: str, interval_s: float, job: Callable[[], Awaitable[object]], initial_delay: float = 0) -> None:
    await asyncio.sleep(initial_delay)
    while True:
        started = time.monotonic()
        try:
            await job()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 — задача должна переживать любые сбои
            log.exception("задача %s упала, повтор через %.0f с", name, interval_s)
        await asyncio.sleep(max(1.0, interval_s - (time.monotonic() - started)))


class Jobs:
    def __init__(self, pipeline: Pipeline):
        self.p = pipeline
        self.s = pipeline.settings

    async def lalafo_full_if_due(self) -> None:
        async with self.p.sessionmaker() as session:
            last = await repo.get_kv(session, "lalafo_full_crawl_at")
        due = last is None or utcnow() - datetime.fromisoformat(last) > timedelta(hours=self.s.lalafo_full_crawl_hours)
        if due:
            log.info("lalafo: запускаю полный обход")
            await self.p.crawl_lalafo_full()

    async def classify(self) -> None:
        async with self.p.sessionmaker() as session:
            pending = await repo.count_pending(session)
        if not pending:
            return
        if self.p.claude and getattr(self.p.claude, "supports_batch", False) and pending > self.s.ai_batch_api_threshold:
            batch_id, report = await self.p.submit_ai_batch()
            log.info("большая очередь (%d): правила/копии — %s; остальное в Batch API %s", pending, report, batch_id)
            return
        # Небольшая очередь — разбираем сразу, пачками, пока не опустеет.
        for _ in range(20):
            report = await self.p.classify_pending(limit=60)
            if report.processed < 60:
                break
        async with self.p.sessionmaker() as session:
            pending = await repo.count_pending(session)
        if self.p.warming_up and pending <= self.s.notify_warmup_pending:
            # Разбор догнал: пересчитываем рынок и только теперь рассылаем настоящие сделки.
            self.p.warming_up = False
            log.info("первичный разбор завершён — пересчитываю рынок и проверяю сделки")
            await self.stats()

    async def collect_batches(self) -> None:
        report = await self.p.collect_ai_batches()
        if report.processed:
            log.info("Batch API: %s", report)
            await self.stats()

    async def refresh_topics(self) -> None:
        checked, hidden = await self.p.refresh_telegram_topics()
        if checked:
            log.info("темы Telegram: проверено %d старых постов, скрыто %d", checked, hidden)

    async def stats(self) -> None:
        await self.p.recompute()
        deals = await self.p.reevaluate_recent(hours=self.s.notify_max_age_hours)
        if deals and self.p.on_items:
            async with self.p.sessionmaker() as session:
                fresh = await repo.recent_deals(session, hours=self.s.notify_max_age_hours, limit=200)
            await self.p.on_items([i.id for i in fresh])


async def run_scheduler(pipeline: Pipeline) -> None:
    await pipeline.refresh_rates()
    await pipeline.maintenance()
    if pipeline.settings.telegram_enabled:
        await pipeline.sync_channels_from_settings()
    await asyncio.gather(*(periodic(name, interval, job, delay) for name, interval, job, delay in plan_jobs(pipeline)))


def plan_jobs(pipeline: Pipeline) -> list[tuple[str, float, Callable[[], Awaitable[object]], float]]:
    """Список фоновых задач с учётом включённых источников и ИИ: (имя, интервал, задача, задержка старта)."""
    s = pipeline.settings
    jobs = Jobs(pipeline)
    plan = [
        ("курсы", 12 * 3600, pipeline.refresh_rates, 12 * 3600),
        ("lalafo", s.lalafo_poll_minutes * 60, pipeline.poll_lalafo, 2),
        ("lalafo-полный", 3600, jobs.lalafo_full_if_due, 30),
        ("классификация", 20, jobs.classify, 15),
        ("статистика", 30 * 60, jobs.stats, 90),
        # Снимок цен за день + пересчёт скорости продажи (данных о продажах прибавляется постоянно).
        ("история", 6 * 3600, pipeline.history_snapshot, 300),
    ]
    if pipeline.vectors is not None:
        plan.append(("векторы", 30, pipeline.embed_pending, 60))
    if s.telegram_enabled:
        plan.append(("telegram", s.telegram_poll_minutes * 60, pipeline.poll_telegram, 10))
        # Разово дозаполняет тему у постов, сохранённых до появления TELEGRAM_BLOCKED_TOPICS — пока
        # очередь не опустеет (проверяет частями), дальше почти сразу выходит без единого запроса.
        plan.append(("темы-telegram", 60, jobs.refresh_topics, 20))
    if pipeline.claude and getattr(pipeline.claude, "supports_batch", False):
        plan.append(("batch-api", 5 * 60, jobs.collect_batches, 60))
    if pipeline.refine_in_background:
        plan.append(("ИИ-доработка", 20, pipeline.refine_pending, 45))
    if pipeline.claude and hasattr(pipeline.claude, "health"):
        # Локальный сервер мог быть выключен: как только он отвечает, возвращаем отложенные посты в очередь.
        plan.append(("ИИ-повтор", 5 * 60, pipeline.requeue_unavailable, 120))
    return plan
