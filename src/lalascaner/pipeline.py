"""Оркестрация: источники → БД → классификация → рыночная оценка → уведомления."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import timedelta
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

import anthropic
from sqlalchemy import select
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.orm import selectinload

from lalascaner.ai.claude import AIUnavailable, ClaudeClassifier, PostInput
from lalascaner.ai.embed import EmbedModelMissing, EmbedUnavailable, OllamaEmbedder, embed_text
from lalascaner.ai.grounding import ground_result
from lalascaner.ai.local import LocalClassifier, OllamaClassifier
from lalascaner.ai.normalize import build_keys, pretty_key
from lalascaner.ai.rules import classify_rules, looks_like_offer
from lalascaner.config import Settings
from lalascaner.db import repo
from lalascaner.db.models import Item, Listing, PriceStat, utcnow
from lalascaner.market import vectors as vec
from lalascaner.market.currency import DEFAULT_RATES, fetch_rates
from lalascaner.market.deals import DealParams, evaluate_items
from lalascaner.market.liquidity import build_liquidity
from lalascaner.market.stats import recompute_stats
from lalascaner.sources.base import RawListing
from lalascaner.sources.lalafo import FEED_CATEGORIES, LalafoClient, leaf_hint
from lalascaner.sources.telegram_web import MISSING, ChatInfo, TelegramWebClient

log = logging.getLogger(__name__)

ItemsCallback = Callable[[list[int]], Awaitable[None]]
RULES_CONFIDENT, RULES_UNSURE = 0.75, 0.55


@dataclass
class ClassifyReport:
    processed: int = 0
    copied: int = 0
    skipped: int = 0
    by_rules: int = 0
    by_ai: int = 0
    failed: int = 0
    item_ids: list[int] = field(default_factory=list)
    deal_ids: list[int] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"разобрано {self.processed}: ИИ {self.by_ai}, правила {self.by_rules}, копии {self.copied}, "
            f"пропущено {self.skipped}, ошибок {self.failed}; позиций {len(self.item_ids)}, сделок {len(self.deal_ids)}"
        )


Classifier = ClaudeClassifier | LocalClassifier | OllamaClassifier

# Сервер ИИ недоступен: столько секунд не пытаемся к нему обращаться (посты ждут в очереди повторов).
AI_DOWN_PAUSE_S = 300
UNAVAILABLE_PREFIX = "unavailable:"


def build_claude(settings: Settings) -> Classifier | None:
    if settings.effective_classifier == "rules":
        return None
    if settings.ai_provider == "ollama":
        return OllamaClassifier(
            base_url=settings.effective_ai_base_url,
            model=settings.ai_model,
            timeout=settings.ai_timeout,
            num_ctx=settings.ai_num_ctx,
        )
    if settings.ai_provider == "openai":
        return LocalClassifier(
            base_url=settings.ai_base_url,
            api_key=settings.ai_api_key,
            model=settings.ai_model,
            timeout=settings.ai_timeout,
        )
    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key or None, max_retries=4)
    return ClaudeClassifier(client=client, model=settings.ai_model, effort=settings.ai_effort)


def grounding_text(listing: Listing) -> str:
    """Всё, что ИИ видел о посте: у lalafo цена — отдельное поле, в описании её может не быть."""
    price = f"\n{listing.price:.0f} {listing.currency or ''}" if listing.price is not None else ""
    return f"{listing.title}\n{listing.text}{price}"


def build_embedder(settings: Settings) -> OllamaEmbedder | None:
    if not settings.effective_embed:
        return None
    if not vec.available():
        log.warning("векторы выключены: в образе нет numpy — пересоберите его: docker compose up -d --build")
        return None
    return OllamaEmbedder(base_url=settings.effective_ai_base_url, model=settings.embed_model)


# Ollama не отвечает на запросы векторов — столько секунд не пробуем.
EMBED_DOWN_PAUSE_S = 600
LIQUIDITY_WINDOW_DAYS = 90


@dataclass
class Appraisal:
    """«За сколько продать»: что распознали в описании и на какую цену опираться."""

    query: str
    title: str | None = None
    key: str | None = None
    category: str | None = None
    condition: str = "unknown"
    stat: PriceStat | None = None
    estimate: vec.Estimate | None = None
    active_n: int = 0
    cheapest: float | None = None
    liquidity: dict = field(default_factory=dict)

    @property
    def reference(self) -> float | None:
        if self.stat is not None:
            return self.stat.median
        return self.estimate.price if self.estimate else None


class Pipeline:
    def __init__(
        self,
        settings: Settings,
        sessionmaker: async_sessionmaker,
        lalafo: LalafoClient | None = None,
        telegram: TelegramWebClient | None = None,
        claude: Classifier | None = None,
        embedder: OllamaEmbedder | None = None,
    ):
        self.settings = settings
        self.sessionmaker = sessionmaker
        self.lalafo = lalafo or LalafoClient()
        self.telegram = telegram or TelegramWebClient()
        self.claude = claude
        self.embedder = embedder
        self.vectors = vec.VectorIndex() if embedder is not None else None
        self._vectors_loaded = False
        self._embed_down_until = 0.0
        self._embed_pull_tried = False
        self._embed_lock = asyncio.Lock()
        self.on_items: ItemsCallback | None = None
        self._classify_lock = asyncio.Lock()
        # Идёт первичный разбор большой очереди: «самое дешёвое» ещё неизвестно, уведомления придерживаем.
        self.warming_up = False
        # SQLite допускает одного писателя: все записи внутри процесса идут по очереди, транзакции короткие.
        self._write_lock = asyncio.Lock()
        self._ai_down_until = 0.0
        self._refine_lock = asyncio.Lock()

    @asynccontextmanager
    async def write_session(self) -> AsyncIterator[AsyncSession]:
        async with self._write_lock, self.sessionmaker() as session:
            yield session

    def deal_params(self) -> DealParams:
        return DealParams.from_settings(self.settings)

    async def close(self) -> None:
        await self.lalafo.close()
        await self.telegram.close()
        if self.claude and hasattr(self.claude, "aclose"):
            await self.claude.aclose()
        if self.embedder is not None:
            await self.embedder.aclose()

    # ── курсы ────────────────────────────────────────────────────────────────

    async def refresh_rates(self) -> dict[str, float]:
        try:
            rates = await fetch_rates()
        except Exception as exc:  # noqa: BLE001 — курс не критичен, работаем на последних известных
            log.warning("не удалось обновить курсы НБКР: %s", exc)
            return await self.rates()
        async with self.write_session() as session:
            await repo.set_kv(session, "rates", rates)
            await session.commit()
        log.info("курсы НБКР: USD=%.2f", rates.get("USD", 0))
        return rates

    async def rates(self) -> dict[str, float]:
        async with self.sessionmaker() as session:
            return await repo.get_kv(session, "rates", DEFAULT_RATES)

    # ── lalafo ───────────────────────────────────────────────────────────────

    async def _store(self, raws: list[RawListing]) -> repo.UpsertResult:
        for attempt in range(5):
            try:
                async with self.write_session() as session:
                    result = await repo.upsert_listings(session, raws)
                    await session.commit()
                return result
            except OperationalError as exc:  # БД занята другим процессом (например, CLI) — подождём
                if "locked" not in str(exc) or attempt == 4:
                    raise
                log.warning("БД занята, повтор записи через %d с", 5 * (attempt + 1))
                await asyncio.sleep(5 * (attempt + 1))
        raise RuntimeError("unreachable")

    async def poll_lalafo(self, max_pages: int | None = None) -> int:
        """Свежие объявления: листаем «сначала новые», пока на странице есть незнакомые."""
        max_pages = max_pages or self.settings.lalafo_max_pages_per_poll
        total_new = 0
        for category_id, name in FEED_CATEGORIES.items():
            for page_no in range(1, max_pages + 1):
                page = await self.lalafo.fetch_page(category_id, page_no)
                async with self.sessionmaker() as session:
                    known = await repo.known_external_ids(session, "lalafo", [l.external_id for l in page.listings])
                fresh_regular = [l for l in page.listings if not l.is_promoted and l.external_id not in known]
                result = await self._store(page.listings)
                total_new += len(result.new_ids)
                if not fresh_regular or page_no >= page.page_count:
                    break
            log.debug("lalafo %s: готово", name)
        if total_new:
            log.info("lalafo: новых объявлений %d", total_new)
        return total_new

    async def crawl_lalafo_full(self, progress: Callable[[str], None] | None = None) -> int:
        """Полный обход всех страниц: наполняет базу и обновляет last_seen/цены у старых объявлений."""
        total_new = total_changed = 0
        for category_id, name in FEED_CATEGORIES.items():
            first = await self.lalafo.fetch_page(category_id, 1)
            pages = first.page_count
            result = await self._store(first.listings)
            total_new += len(result.new_ids)
            total_changed += len(result.changed_ids)
            for page_no in range(2, pages + 1):
                try:
                    page = await self.lalafo.fetch_page(category_id, page_no)
                except Exception as exc:  # noqa: BLE001 — одна страница не должна срывать весь обход
                    log.warning("lalafo %s: страница %d пропущена: %s", name, page_no, exc)
                    continue
                result = await self._store(page.listings)
                total_new += len(result.new_ids)
                total_changed += len(result.changed_ids)
                if progress and page_no % 10 == 0:
                    progress(f"lalafo {name}: страница {page_no}/{pages}, новых {total_new}")
            if progress:
                progress(f"lalafo {name}: {pages} стр. пройдено")
        async with self.write_session() as session:
            await repo.set_kv(session, "lalafo_full_crawl_at", utcnow().isoformat())
            # Обход прошёл целиком — что не встречалось в ленте дольше «активного» окна, пропало с сайта.
            gone = await repo.mark_gone(session, utcnow() - timedelta(hours=self.settings.lalafo_active_hours))
            await session.commit()
        log.info("lalafo полный обход: новых %d, изменённых %d, пропало %d", total_new, total_changed, gone)
        return total_new

    # ── Telegram ─────────────────────────────────────────────────────────────

    async def sync_channels_from_settings(self) -> None:
        async with self.sessionmaker() as session:
            known = {c.username for c in await repo.list_channels(session)}
        for username in self.settings.channel_list:
            if username not in known:
                info = await self.add_channel(username)
                if info is None:
                    log.warning("telegram: не удалось открыть @%s (приватный или не существует)", username)

    async def add_channel(self, username: str) -> ChatInfo | None:
        info = await self.telegram.probe(username)
        if info is None:
            return None
        async with self.write_session() as session:
            await repo.add_channel(session, info.username, info.kind, info.title)
            await session.commit()
        log.info("telegram: подключён @%s (%s) — %s", info.username, info.kind, info.title)
        return info

    async def poll_telegram(self) -> int:
        async with self.sessionmaker() as session:
            channels = await repo.list_channels(session, enabled_only=True)
        total = 0
        for channel in channels:
            try:
                if channel.kind == "group":
                    total += await self._poll_group(channel.username, channel.last_message_id)
                else:
                    total += await self._poll_channel(channel.username, channel.last_message_id)
            except Exception:  # noqa: BLE001 — один сломанный канал не должен останавливать остальные
                log.exception("telegram: ошибка при опросе @%s", channel.username)
        return total

    async def _set_cursor(self, username: str, last_id: int) -> None:
        async with self.write_session() as session:
            channel = await repo.get_channel(session, username)
            if channel and last_id > channel.last_message_id:
                channel.last_message_id = last_id
            await session.commit()

    @staticmethod
    def _msg_id(listing: RawListing) -> int:
        return int(listing.url.rstrip("/").rsplit("/", 1)[1])

    async def _poll_channel(self, username: str, last_id: int) -> int:
        max_pages = self.settings.telegram_backfill_pages if last_id == 0 else 5
        collected: list[RawListing] = []
        before = None
        for _ in range(max_pages):
            page = await self.telegram.fetch_channel_page(username, before=before)
            if not page:
                break
            fresh = [l for l in page if self._msg_id(l) > last_id]
            collected.extend(fresh)
            if len(fresh) < len(page):
                break
            before = min(self._msg_id(l) for l in page)
        if not collected:
            return 0
        result = await self._store(collected)
        await self._set_cursor(username, max(self._msg_id(l) for l in collected))
        log.info("telegram @%s: новых постов %d", username, len(result.new_ids))
        return len(result.new_ids)

    async def _poll_group(self, username: str, last_id: int) -> int:
        until = None
        if last_id == 0:
            latest = await self.telegram.latest_group_message_id(username)
            start = max(1, latest - self.settings.telegram_backfill_messages + 1)
            limit, until = self.settings.telegram_backfill_messages + 50, latest
            log.info("telegram @%s: первая загрузка, сообщения %d…%d", username, start, latest)
        else:
            start, limit = last_id + 1, self.settings.telegram_group_scan_limit
        listings, last_existing = await self.telegram.scan_group(username, start, max_messages=limit, until_id=until)
        if last_existing < start:
            # Ничего не нашли подряд — возможно, дыра из удалённых сообщений. Проверяем, есть ли что-то дальше.
            for step in (40, 80, 150, 400):
                if await self.telegram.fetch_group_message(username, start + step) != MISSING:
                    await self._set_cursor(username, start + 39)
                    break
            return 0
        result = await self._store(listings)
        await self._set_cursor(username, last_existing)
        if result.new_ids:
            log.info("telegram @%s: новых постов %d (до #%d)", username, len(result.new_ids), last_existing)
        return len(result.new_ids)

    async def refresh_telegram_topics(self, limit: int = 200) -> tuple[int, int]:
        """Дозаполняет topic_id у Telegram-объявлений, сохранённых до появления этого поля (для них
        нужен ещё один запрос к Telegram), и прячет те, что попали в TELEGRAM_BLOCKED_TOPICS.
        Возвращает (сколько проверено, сколько скрыто)."""
        async with self.sessionmaker() as session:
            listings = await repo.listings_missing_topic(session, "telegram", limit)
        if not listings:
            return 0, 0
        # Запросы к Telegram — без блокировки записи, как и для ИИ: сбор данных в это время не простаивает.
        resolved: dict[int, int] = {}
        for listing in listings:
            result = await self.telegram.fetch_group_message(listing.source_ref, self._msg_id(listing))
            topic_id = result.topic_id if isinstance(result, RawListing) else None
            # 0 — «проверили, темы нет»: отличаем от NULL («ещё не проверяли»), чтобы не спрашивать снова.
            resolved[listing.id] = topic_id or 0
        hidden = 0
        async with self.write_session() as session:
            rows = {row.id: row for row in await self._listings_by_ids(session, list(resolved))}
            for listing_id, topic_id in resolved.items():
                row = rows.get(listing_id)
                if row is None:
                    continue
                row.topic_id = topic_id
                if topic_id in self.settings.blocked_topic_ids:
                    await repo.mark_skipped(session, row, "topic")
                    hidden += 1
            await session.commit()
        return len(listings), hidden

    # ── классификация ────────────────────────────────────────────────────────

    def _post_input(self, listing: Listing) -> PostInput:
        return PostInput(
            post_id=str(listing.id),
            source=listing.source,
            title=listing.title,
            text=listing.text,
            price=listing.price,
            currency=listing.currency,
            category_hint=leaf_hint(listing.source_ref) if listing.source == "lalafo" else None,
        )

    def _rules(self, listing: Listing):  # noqa: ANN202
        hint = leaf_hint(listing.source_ref) if listing.source == "lalafo" else None
        if listing.source == "lalafo":
            return classify_rules(
                str(listing.id), listing.text, listing.title, listing.price, listing.currency or "KGS", hint
            )
        return classify_rules(str(listing.id), listing.text)

    async def _triage(self, session, listings: list[Listing], rates, report: ClassifyReport) -> list[Listing]:  # noqa: ANN001
        """Дешёвые шаги до ИИ. Возвращает посты, которым нужен ИИ."""
        mode = self.settings.effective_classifier if self.claude else "rules"
        need_ai: list[Listing] = []
        for listing in listings:
            report.processed += 1
            donor = await repo.find_done_by_hash(session, listing.content_hash, exclude_id=listing.id)
            if donor is not None:
                # Репост уже разобранного текста: копируем позиции в базу, но не оцениваем и не уведомляем повторно.
                await repo.copy_items(session, listing, donor)
                report.copied += 1
                continue
            if listing.source == "telegram" and listing.topic_id in self.settings.blocked_topic_ids:
                await repo.mark_skipped(session, listing, "topic")
                report.skipped += 1
                continue
            if listing.source == "telegram" and not looks_like_offer(listing.text):
                await repo.mark_skipped(session, listing, "prefilter")
                report.skipped += 1
                continue
            if mode == "claude":
                need_ai.append(listing)
                continue
            outcome = self._rules(listing)
            if mode == "rules" or outcome.confident:
                # Однозначное объявление (одна модель, одна цена) — правилам можно верить и для уведомлений.
                confidence = RULES_CONFIDENT if outcome.confident else RULES_UNSURE
                items = await repo.save_result(session, listing, outcome.result, "rules", rates, confidence)
                report.by_rules += 1
                report.item_ids += [i.id for i in items]
            elif self.refine_in_background:
                # Медленный локальный ИИ не должен задерживать всё остальное: результат правил (пусть неуверенный)
                # сохраняем сразу, а доработку ставим в очередь — но только для того, что стоит машинного времени.
                items = await repo.save_result(session, listing, outcome.result, "rules", rates, RULES_UNSURE)
                report.by_rules += 1
                report.item_ids += [i.id for i in items]
                if self._worth_refining(listing):
                    listing.status = "refine"
            else:
                need_ai.append(listing)
        return need_ai

    @property
    def refine_in_background(self) -> bool:
        return bool(self.claude and getattr(self.claude, "background_refine", False))

    def _worth_refining(self, listing: Listing) -> bool:
        """Telegram — всегда; lalafo — только свежее (старый хвост каталога правилам хватает)."""
        if listing.source != "lalafo":
            return True
        return listing.posted_at >= utcnow() - timedelta(days=self.settings.ai_refine_max_age_days)

    async def _save_ai(
        self, session, by_id: dict[str, Listing], results, errors, rates, report, keep_on_error: bool = False  # noqa: ANN001
    ) -> None:
        for post_id, result in results.items():
            listing = by_id[post_id]
            # ИИ может додумать цену или модель, которых в посте нет, — сверяем с текстом.
            result = ground_result(result, grounding_text(listing))
            items = await repo.save_result(session, listing, result, getattr(self.claude, "kind", "claude"), rates)
            report.by_ai += 1
            report.item_ids += [i.id for i in items]
        failed = [by_id[pid] for pid in errors if pid in by_id]
        for listing in failed:
            if keep_on_error:
                # Результат правил уже в базе. Сервер недоступен — доработаем позже (остаётся «refine»);
                # модель просто не справилась с постом — оставляем результат правил как окончательный.
                if not errors[str(listing.id)].startswith(UNAVAILABLE_PREFIX):
                    listing.status = "done"
                    report.failed += 1
                continue
            await repo.mark_failed(session, [listing], errors[str(listing.id)])
            report.failed += 1

    async def classify_pending(self, limit: int = 60, include_failed: bool = False) -> ClassifyReport:
        async with self._classify_lock:
            report = ClassifyReport()
            rates = await self.rates()
            async with self.write_session() as session:
                listings = await repo.pending_listings(session, limit, include_failed=include_failed)
                if not listings:
                    return report
                need_ai_ids = [l.id for l in await self._triage(session, listings, rates, report)]
                await session.commit()

            if need_ai_ids and self.claude:
                # Запросы к ИИ идут без блокировки записи — сбор данных в это время не простаивает.
                async with self.sessionmaker() as session:
                    need_ai = await self._listings_by_ids(session, need_ai_ids)
                outputs = await self._run_ai(need_ai)
                async with self.write_session() as session:
                    by_id = {str(l.id): l for l in await self._listings_by_ids(session, need_ai_ids)}
                    for results, errors in outputs:
                        await self._save_ai(session, by_id, results, errors, rates, report)
                    await self._persist_usage(session)
                    await session.commit()

            async with self.write_session() as session:
                await self._evaluate(session, report)
                await session.commit()

            log.info("классификация: %s", report)
            await self._emit(report)
            return report

    async def refine_pending(self, limit: int | None = None) -> ClassifyReport:
        """Одна порция фоновой доработки: ИИ перечитывает объявления, которые правила разобрали неуверенно."""
        report = ClassifyReport()
        if not self.refine_in_background or time.monotonic() < self._ai_down_until:
            return report
        async with self._refine_lock:
            async with self.sessionmaker() as session:
                size = max(1, self.settings.effective_batch_size)
                listings = await repo.refine_listings(session, limit or size * 2)
            if not listings:
                return report
            rates = await self.rates()
            outputs = await self._run_ai(listings)
            async with self.write_session() as session:
                by_id = {str(l.id): l for l in await self._listings_by_ids(session, [l.id for l in listings])}
                for results, errors in outputs:
                    await self._save_ai(session, by_id, results, errors, rates, report, keep_on_error=True)
                report.processed = len(listings)
                await self._persist_usage(session)
                await self._evaluate(session, report)
                await session.commit()
        log.info("доработка ИИ: %s", report)
        await self._emit(report)
        return report

    async def _run_ai(self, listings: list[Listing]) -> list[tuple[dict, dict]]:
        size = max(1, self.settings.effective_batch_size)
        groups = [listings[i : i + size] for i in range(0, len(listings), size)]
        sem = asyncio.Semaphore(max(1, self.settings.effective_ai_concurrency))

        def unavailable(group: list[Listing], reason: str) -> tuple[dict, dict]:
            return {}, {str(l.id): f"{UNAVAILABLE_PREFIX} {reason}" for l in group}

        async def run(group: list[Listing]) -> tuple[dict, dict]:
            async with sem:
                if time.monotonic() < self._ai_down_until:
                    return unavailable(group, "ИИ недоступен (см. лог)")
                try:
                    return await self.claude.classify([self._post_input(l) for l in group])
                except AIUnavailable as exc:
                    self._ai_down_until = time.monotonic() + AI_DOWN_PAUSE_S
                    log.warning("ИИ недоступен: %s. Сложные объявления подождут, разбор на правилах продолжается.", exc)
                    return unavailable(group, str(exc))
                except anthropic.APIError as exc:
                    log.error("claude: ошибка API: %s", exc)
                    return {}, {str(l.id): f"API: {exc}" for l in group}

        return list(await asyncio.gather(*(run(g) for g in groups)))

    async def _evaluate(self, session, report: ClassifyReport) -> None:  # noqa: ANN001
        items = await repo.get_items(session, report.item_ids)
        deals = await evaluate_items(session, items, self.deal_params())
        report.deal_ids = [d.id for d in deals]

    async def _emit(self, report: ClassifyReport) -> None:
        if not (self.on_items and report.item_ids):
            return
        async with self.sessionmaker() as session:
            pending = await repo.count_pending(session)
        if pending > self.settings.notify_warmup_pending:
            if not self.warming_up:
                log.info("очередь %d объявлений: уведомления придержу до конца разбора", pending)
            self.warming_up = True
            return
        cutoff = repo.freshness_cutoff(self.settings.notify_max_age_hours)
        async with self.sessionmaker() as session:
            items = await repo.get_items(session, report.item_ids)
        fresh = [i.id for i in items if i.listing.changed_at >= cutoff]
        if fresh:
            try:
                await self.on_items(fresh)
            except Exception:  # noqa: BLE001
                log.exception("ошибка отправки уведомлений")

    async def _persist_usage(self, session) -> None:  # noqa: ANN001
        if not self.claude:
            return
        total = await repo.get_kv(session, "ai_usage", {}) or {}
        for key, value in self.claude.usage.as_dict().items():
            total[key] = total.get(key, 0) + value
        await repo.set_kv(session, "ai_usage", total)
        self.claude.usage = type(self.claude.usage)()

    # ── доступность ИИ ───────────────────────────────────────────────────────

    async def ai_health(self) -> tuple[bool, str]:
        """(доступен ли ИИ, пояснение). Для Claude не проверяем — запрос стоил бы денег."""
        if not self.claude:
            return False, "ИИ выключен"
        check = getattr(self.claude, "health", None)
        return await check() if check else (True, "проверка не требуется")

    async def requeue_unavailable(self) -> int:
        """Возвращает в очередь посты, которые не разобрали из-за недоступности ИИ, если он снова отвечает."""
        async with self.sessionmaker() as session:
            waiting = await repo.count_unavailable(session)
        if not waiting:
            return 0
        ok, message = await self.ai_health()
        if not ok:
            log.info("ИИ всё ещё недоступен (%d постов ждут): %s", waiting, message)
            return 0
        self._ai_down_until = 0.0
        async with self.write_session() as session:
            count = await repo.requeue_unavailable(session)
            await session.commit()
        log.info("ИИ снова доступен — возвращаю в очередь %d постов", count)
        return count

    # ── Batch API ────────────────────────────────────────────────────────────

    async def submit_ai_batch(self, limit: int = 20000) -> tuple[str | None, ClassifyReport]:
        if not self.claude:
            raise RuntimeError("Batch API требует ANTHROPIC_API_KEY и CLASSIFIER=claude|hybrid")
        if not getattr(self.claude, "supports_batch", False):
            raise RuntimeError("Batch API поддерживает только AI_PROVIDER=claude")
        report = ClassifyReport()
        rates = await self.rates()
        async with self.write_session() as session:
            listings = await repo.pending_listings(session, limit)
            need_ai = await self._triage(session, listings, rates, report)
            if not need_ai:
                await session.commit()
                return None, report
            size = max(1, self.settings.effective_batch_size)
            groups = [need_ai[i : i + size] for i in range(0, len(need_ai), size)]
            batch_id = await self.claude.submit_batch([[self._post_input(l) for l in g] for g in groups])
            for listing in need_ai:
                listing.status = "batch"
            batches = await repo.get_kv(session, "ai_batches", []) or []
            batches.append({"id": batch_id, "groups": {f"g{i}": [l.id for l in g] for i, g in enumerate(groups)}})
            await repo.set_kv(session, "ai_batches", batches)
            await self._evaluate(session, report)
            await session.commit()
        log.info("Batch API: отправлено %d постов в %d запросах, batch %s", len(need_ai), len(groups), batch_id)
        await self._emit(report)
        return batch_id, report

    async def collect_ai_batches(self) -> ClassifyReport:
        report = ClassifyReport()
        if not self.claude:
            return report
        rates = await self.rates()
        async with self.write_session() as session:
            batches = await repo.get_kv(session, "ai_batches", []) or []
            remaining = []
            for batch in batches:
                status = await self.claude.batch_status(batch["id"])
                if status != "ended":
                    remaining.append(batch)
                    continue
                listing_ids = [lid for ids in batch["groups"].values() for lid in ids]
                rows = {l.id: l for l in await self._listings_by_ids(session, listing_ids)}
                groups = {
                    cid: [self._post_input(rows[lid]) for lid in ids if lid in rows]
                    for cid, ids in batch["groups"].items()
                }
                results, errors = await self.claude.collect_batch(batch["id"], groups)
                by_id = {str(lid): row for lid, row in rows.items()}
                await self._save_ai(session, by_id, results, errors, rates, report)
                report.processed += len(rows)
                log.info("Batch API %s: получено %d, ошибок %d", batch["id"], len(results), len(errors))
            await repo.set_kv(session, "ai_batches", remaining)
            await self._persist_usage(session)
            await self._evaluate(session, report)
            await session.commit()
        await self._emit(report)
        return report

    @staticmethod
    async def _listings_by_ids(session, ids: list[int]) -> list[Listing]:  # noqa: ANN001
        rows = []
        for i in range(0, len(ids), 500):
            rows += list(await session.scalars(select(Listing).where(Listing.id.in_(ids[i : i + 500]))))
        return rows

    # ── статистика ───────────────────────────────────────────────────────────

    async def recompute(self) -> int:
        async with self.write_session() as session:
            count = await recompute_stats(session, self.settings.stats_window_days)
            await session.commit()
        return count

    async def reevaluate_recent(self, hours: int = 72) -> int:
        """После пересчёта рынка обновляет оценку свежих позиций (без повторных уведомлений)."""
        async with self.write_session() as session:
            since = repo.freshness_cutoff(hours)
            items = list(
                await session.scalars(
                    select(Item)
                    .join(Listing)
                    .where(Listing.changed_at >= since, Item.intent == "sell", Listing.classifier.not_like("copy:%"))
                )
            )
            deals = await evaluate_items(session, items, self.deal_params())
            await session.commit()
        return len(deals)

    # ── история ──────────────────────────────────────────────────────────────

    async def history_snapshot(self) -> None:
        """Раз в день — снимок рыночных цен; скорость продажи пересчитывается при каждом вызове."""
        day = utcnow().strftime("%Y-%m-%d")
        async with self.write_session() as session:
            if await repo.get_kv(session, "history_day") != day:
                count = await repo.snapshot_price_history(session, day)
                await repo.set_kv(session, "history_day", day)
                log.info("история цен: снимок за %s, моделей %d", day, count)
            await session.commit()
        await self.recompute_liquidity()

    async def recompute_liquidity(self) -> dict:
        async with self.sessionmaker() as session:
            rows = await repo.liquidity_rows(session, utcnow() - timedelta(days=LIQUIDITY_WINDOW_DAYS))
            reposted = await repo.reposted_listing_ids(session)
        data = build_liquidity(rows, reposted, utcnow())
        async with self.write_session() as session:
            await repo.set_kv(session, "liquidity", data)
            await session.commit()
        return data

    async def maintenance(self) -> None:
        """Разовые починки данных при запуске."""
        async with self.write_session() as session:
            if not await repo.get_kv(session, "fix_monitor_resolution_v1"):
                fixed = await repo.fix_monitor_resolution(session)
                await repo.set_kv(session, "fix_monitor_resolution_v1", True)
                if fixed:
                    log.info("починил разрешение у %d мониторов", fixed)
            if moved := await repo.migrate_old_notifications(session):
                log.info("журнал уведомлений перенесён: %d записей", moved)
            cleaned = not await repo.get_kv(session, "ground_ai_items_v1")
            if cleaned:
                unpriced, removed = await repo.ground_ai_items(session, grounding_text)
                await repo.set_kv(session, "ground_ai_items_v1", True)
                log.info("проверка ответов ИИ по тексту: убрано выдуманных цен %d, выдуманных моделей %d", unpriced, removed)
            await session.commit()
            has_liquidity = await repo.get_kv(session, "liquidity") is not None
        if cleaned:
            await self.recompute()  # медианы без выдуманных цен
        if not has_liquidity:
            await self.recompute_liquidity()

    # ── векторы ──────────────────────────────────────────────────────────────

    async def _ensure_vectors_loaded(self) -> None:
        if self._vectors_loaded or self.vectors is None:
            return
        async with self.sessionmaker() as session:
            rows = await repo.load_vectors(session, self.embedder.model)
        for listing_id, blob in rows:
            self.vectors.add(listing_id, vec.unpack(blob))
        self._vectors_loaded = True
        log.info("векторы: загружено %d", len(rows))

    async def _embed(self, texts: list[str]) -> list[list[float]] | None:
        """None — Ollama или модель векторов недоступны (пауза, работа продолжается без векторов)."""
        if time.monotonic() < self._embed_down_until:
            return None
        try:
            return await self.embedder.embed(texts)
        except EmbedModelMissing:
            if not self._embed_pull_tried:
                self._embed_pull_tried = True
                log.info("векторы: скачиваю модель %s в Ollama (один раз, около 1 ГБ)…", self.embedder.model)
                try:
                    await self.embedder.pull()
                    log.info("векторы: модель %s скачана", self.embedder.model)
                    return await self.embedder.embed(texts)
                except EmbedUnavailable as exc:
                    log.warning("векторы: %s", exc)
            else:
                log.warning("векторы: модели %s нет в Ollama", self.embedder.model)
        except EmbedUnavailable as exc:
            log.warning("векторы: %s — повтор через %d мин", exc, EMBED_DOWN_PAUSE_S // 60)
        self._embed_down_until = time.monotonic() + EMBED_DOWN_PAUSE_S
        return None

    async def _query_vector(self, text: str):  # noqa: ANN202
        raw = await self._embed([text[:2000]])
        return vec.unpack(vec.pack(raw[0])) if raw else None

    async def embed_pending(self, limit: int | None = None) -> int:
        """Порция фоновой работы: векторы для новых/изменённых объявлений, перезаливы и оценка цены по соседям."""
        if self.vectors is None:
            return 0
        async with self._embed_lock:
            await self._ensure_vectors_loaded()
            async with self.sessionmaker() as session:
                listings = await repo.listings_needing_vectors(
                    session, self.embedder.model, limit or self.settings.embed_batch
                )
            if not listings:
                return 0
            raw = await self._embed([embed_text(listing) for listing in listings])
            if raw is None:
                return 0
            packed = {listing.id: vec.pack(v) for listing, v in zip(listings, raw, strict=True)}
            for listing_id, blob in packed.items():
                self.vectors.add(listing_id, vec.unpack(blob))
            async with self.write_session() as session:
                await repo.save_vectors(session, self.embedder.model, packed)
                await self._link_neighbors(session, list(packed))
                await session.commit()
            return len(packed)

    async def _neighbor_items(self, session, query, exclude: set[int], k: int = 60, active=None) -> list[tuple[Item, float]]:  # noqa: ANN001
        """Позиции ближайших по смыслу объявлений: по одной на объявление, от самых похожих."""
        found = self.vectors.search(query, k=k, exclude=exclude)
        scores = dict(found)
        since = utcnow() - timedelta(days=self.settings.stats_window_days) if active is None else None
        items = await repo.sell_items_of_listings(session, list(scores), active=active, since=since)
        per_listing: dict[int, list[Item]] = {}
        for item in items:
            per_listing.setdefault(item.listing_id, []).append(item)
        pairs = [(group[0], scores[lid]) for lid, group in per_listing.items() if len(group) == 1]
        return sorted(pairs, key=lambda pair: -pair[1])

    async def _link_neighbors(self, session, listing_ids: list[int]) -> None:  # noqa: ANN001
        rows = await session.scalars(
            select(Listing).where(Listing.id.in_(listing_ids)).options(selectinload(Listing.items))
        )
        for listing in rows:
            query = self.vectors.get(listing.id)
            sells = [i for i in listing.items if i.intent == "sell" and not i.is_bundle]
            if query is None or not sells:
                continue
            neighbors = await self._neighbor_items(session, query, exclude={listing.id})
            main = sells[0]
            if listing.dup_of is None:
                for other, score in neighbors:
                    older = other.listing.posted_at <= listing.posted_at and other.listing_id < listing.id
                    same_key = other.canonical_key == main.canonical_key
                    if older and vec.is_repost(score, same_key, main.price_kgs, other.price_kgs):
                        listing.dup_of = other.listing_id
                        break
            reposts = {listing.dup_of} if listing.dup_of else set()
            for item in sells:
                pool = [
                    (other, score) for other, score in neighbors
                    if other.category == item.category and other.listing_id not in reposts
                    and (other.listing.author_id is None or other.listing.author_id != listing.author_id)
                ]
                same_condition = [p for p in pool if item.condition in ("new", "used") and p[0].condition == item.condition]
                chosen = same_condition if len(same_condition) >= 3 else pool
                estimate = vec.estimate_price([(other.price_kgs, score) for other, score in chosen])
                item.est_price, item.est_n = (round(estimate.price), estimate.n) if estimate else (None, None)
        await session.flush()

    async def similar_items(self, listing_id: int, k: int = 6) -> list[Item] | None:
        """Похожие актуальные объявления. None — векторов для этого объявления (пока) нет."""
        if self.vectors is None:
            return None
        await self._ensure_vectors_loaded()
        query = self.vectors.get(listing_id)
        if query is None:
            return None
        return await self._nearest_active(query, k, exclude={listing_id})

    async def semantic_search(self, text: str, k: int = 8) -> list[Item] | None:
        if self.vectors is None:
            return None
        await self._ensure_vectors_loaded()
        query = await self._query_vector(text)
        if query is None:
            return None
        return await self._nearest_active(query, k)

    async def _nearest_active(self, query, k: int, exclude: set[int] | None = None) -> list[Item]:  # noqa: ANN001
        async with self.sessionmaker() as session:
            pairs = await self._neighbor_items(
                session, query, exclude=exclude or set(), k=max(80, k * 10), active=self.deal_params().active_clause()
            )
        chosen: list[Item] = []
        seen: set[int] = set()
        for item, _score in pairs:
            # Перезаливы одного объявления показываем один раз.
            if item.listing_id in seen or (item.listing.dup_of and item.listing.dup_of in seen):
                continue
            seen.add(item.listing_id)
            chosen.append(item)
            if len(chosen) >= k:
                break
        return chosen

    # ── за сколько продать ───────────────────────────────────────────────────

    async def appraise(self, text: str) -> Appraisal:
        appraisal = Appraisal(query=text)
        outcome = classify_rules("appraise", text, title=text)
        extracted = next((i for i in outcome.result.items if i.intent == "sell"), None)
        min_samples = self.settings.min_samples
        async with self.sessionmaker() as session:
            appraisal.liquidity = await repo.get_kv(session, "liquidity", {}) or {}
            if extracted is not None:
                canonical, family = build_keys(extracted)
                appraisal.category, appraisal.condition = extracted.category, extracted.condition
                for key in dict.fromkeys([canonical, family]):
                    stats = await repo.stats_by_key(session, key)
                    stat = stats.get(appraisal.condition) if appraisal.condition in ("new", "used") else None
                    if stat is None or stat.n < min_samples:
                        stat = stats.get("all")
                    if stat is not None and stat.n >= min_samples:
                        appraisal.key, appraisal.stat, appraisal.title = key, stat, stat.title
                        break
                if appraisal.key is None:
                    appraisal.key, appraisal.title = canonical, pretty_key(canonical)
                params = self.deal_params()
                floor = appraisal.stat.median * params.suspicious_ratio if appraisal.stat else None
                items, appraisal.active_n = await repo.items_for_key(
                    session, appraisal.key, params.active_clause(), limit=1, min_price=floor
                )
                appraisal.cheapest = items[0].price_kgs if items else None
        if appraisal.stat is None and self.vectors is not None:
            await self._ensure_vectors_loaded()
            query = await self._query_vector(text)
            if query is not None:
                async with self.sessionmaker() as session:
                    pairs = await self._neighbor_items(session, query, exclude=set())
                if appraisal.category:
                    pairs = [p for p in pairs if p[0].category == appraisal.category]
                elif pairs:
                    appraisal.category = pairs[0][0].category
                appraisal.estimate = vec.estimate_price(
                    [(item.price_kgs, score) for item, score in pairs], min_similarity=vec.QUERY_MIN_SIMILARITY
                )
        return appraisal
