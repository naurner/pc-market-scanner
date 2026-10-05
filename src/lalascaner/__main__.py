"""CLI: python -m lalascaner <команда>."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from contextlib import asynccontextmanager

from lalascaner.config import Settings, get_settings
from lalascaner.db import repo
from lalascaner.db.session import init_db, make_engine, make_sessionmaker
from lalascaner.pipeline import Pipeline, build_claude, build_embedder

log = logging.getLogger("lalascaner")


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    for noisy in ("httpx", "httpx2", "aiogram.event", "anthropic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@asynccontextmanager
async def app_context(settings: Settings, with_ai: bool = True):
    engine = make_engine(settings.database_url)
    await init_db(engine)
    sessionmaker = make_sessionmaker(engine)
    pipeline = Pipeline(
        settings,
        sessionmaker,
        claude=build_claude(settings) if with_ai else None,
        embedder=build_embedder(settings) if with_ai else None,
    )
    try:
        yield pipeline
    finally:
        await pipeline.close()
        await engine.dispose()


async def cmd_run(settings: Settings, args: argparse.Namespace) -> None:
    from lalascaner.scheduler import run_scheduler

    async with app_context(settings) as pipeline:
        tasks = [run_scheduler(pipeline)]
        if settings.bot_token:
            from lalascaner.bot.app import build_bot, start_polling

            bot, dp = build_bot(settings, pipeline.sessionmaker, pipeline)
            tasks.append(start_polling(bot, dp))
        else:
            log.warning("BOT_TOKEN не задан — работаю без Telegram-бота (только сбор данных)")
        log.info(
            "источники: lalafo%s · ИИ: %s",
            " + Telegram" if settings.telegram_enabled else "",
            f"{settings.ai_provider}/{settings.effective_classifier} ({settings.ai_model})" if settings.ai_enabled else "выключен",
        )
        if pipeline.claude is not None and hasattr(pipeline.claude, "health"):
            ok, message = await pipeline.ai_health()
            if ok:
                log.info("ИИ готов: %s", message)
            else:
                log.warning("ИИ НЕДОСТУПЕН: %s. Пока разбор идёт на правилах, сложные посты дождутся ИИ.", message)
        await asyncio.gather(*tasks)


async def cmd_backfill(settings: Settings, args: argparse.Namespace) -> None:
    async with app_context(settings, with_ai=False) as pipeline:
        await pipeline.refresh_rates()
        if args.source in ("lalafo", "all"):
            await pipeline.crawl_lalafo_full(progress=lambda msg: print(msg, flush=True))
        if args.source in ("telegram", "all"):
            if not settings.telegram_enabled:
                print("telegram: источники выключены (TELEGRAM_ENABLED=false) — пропускаю")
                return
            if args.messages:
                pipeline.settings.telegram_backfill_messages = args.messages
            await pipeline.sync_channels_from_settings()
            new = await pipeline.poll_telegram()
            print(f"telegram: новых постов {new}")


async def cmd_classify(settings: Settings, args: argparse.Namespace) -> None:
    if (args.collect or args.batch_api) and not settings.ai_enabled:
        print("ИИ выключен (AI_ENABLED=false) — Batch API недоступен. Обычный разбор: python -m lalascaner classify")
        return
    if (args.collect or args.batch_api) and settings.ai_provider != "claude":
        print("Batch API поддерживает только AI_PROVIDER=claude. Обычный разбор: python -m lalascaner classify")
        return
    async with app_context(settings) as pipeline:
        if args.collect:
            print(await pipeline.collect_ai_batches())
            return
        if args.batch_api:
            batch_id, report = await pipeline.submit_ai_batch(limit=args.limit)
            print(report)
            print(f"Batch API: {batch_id or 'нечего отправлять'}. Забрать результаты: python -m lalascaner classify --collect")
            return
        total = 0
        while total < args.limit:
            step = min(100, args.limit - total)
            report = await pipeline.classify_pending(limit=step, include_failed=args.retry_failed)
            print(report, flush=True)
            total += report.processed
            if report.processed < step:
                break
        if args.recompute:
            print(f"статистика: {await pipeline.recompute()} ключей")


AI_CHECK_SAMPLES = [
    "rtx3070ti zotac gaming trio\nне помешает обслуживание\nцена 27к",
    "Продам: i5 13400 2шт - по 13.5к  32gb ddr5 (16x2) 5600mhz cl36 king. fury - 36.5k",
    "сколько щас стоит 8гб ддр4 одной планкой?",
]


async def cmd_ai_check(settings: Settings, args: argparse.Namespace) -> None:
    """Проверка ИИ: настройки, доступность сервера, наличие модели и пробный разбор нескольких постов."""
    import time

    from lalascaner.ai.claude import AIUnavailable, PostInput

    print(f"AI_ENABLED={settings.ai_enabled} · AI_PROVIDER={settings.ai_provider} · AI_MODEL={settings.ai_model}")
    print(f"адрес: {settings.effective_ai_base_url or '(не нужен)'} · пачка {settings.effective_batch_size} постов")
    if not settings.ai_enabled:
        print("❌ ИИ выключен. В .env поставьте AI_ENABLED=true и перезапустите: docker compose up -d")
        return
    async with app_context(settings) as pipeline:
        if pipeline.claude is None:
            print("❌ ИИ не используется: CLASSIFIER=rules или не задан ключ/адрес провайдера")
            return
        ok, message = await pipeline.ai_health()
        print(("✅ " if ok else "❌ ") + message)
        if not ok:
            return
        if settings.ai_provider == "claude" and not args.test:
            print("Пробный разбор для Claude платный (копейки) — добавьте --test, если он нужен.")
            return
        posts = [PostInput(post_id=str(i), source="telegram", text=text) for i, text in enumerate(AI_CHECK_SAMPLES, 1)]
        started = time.monotonic()
        try:
            results, errors = await pipeline.claude.classify(posts)
        except AIUnavailable as exc:
            print(f"❌ {exc}")
            return
        print(f"пробный разбор {len(posts)} постов: {time.monotonic() - started:.0f} с, токены {pipeline.claude.usage.as_dict()}")
        for post_id, result in sorted(results.items()):
            what = "; ".join(f"{i.category} {i.model!r} {i.price} {i.currency}" for i in result.items) or "—"
            print(f"  пост {post_id}: {result.intent} → {what}")
        for post_id, error in errors.items():
            print(f"  пост {post_id}: ОШИБКА {error}")
        print("Ожидаемо: пост 1 — видеокарта RTX 3070 Ti ~27000, пост 2 — процессор i5-13400 и память DDR5, пост 3 — other без позиций.")


async def cmd_recompute(settings: Settings, args: argparse.Namespace) -> None:
    async with app_context(settings, with_ai=False) as pipeline:
        print(f"статистика: {await pipeline.recompute()} ключей")
        print(f"сделок среди свежих: {await pipeline.reevaluate_recent(hours=args.hours)}")


async def cmd_retag_topics(settings: Settings, args: argparse.Namespace) -> None:
    """Дозаполняет тему у Telegram-постов, сохранённых до TELEGRAM_BLOCKED_TOPICS, и скрывает заблокированные.
    В обычной работе `run` делает то же самое сам в фоне — команда нужна только чтобы не ждать."""
    async with app_context(settings, with_ai=False) as pipeline:
        total_checked = total_hidden = 0
        while True:
            checked, hidden = await pipeline.refresh_telegram_topics(limit=200)
            if not checked:
                break
            total_checked, total_hidden = total_checked + checked, total_hidden + hidden
            print(f"проверено {total_checked}, скрыто {total_hidden}", flush=True)
        if total_hidden:
            print(f"статистика: {await pipeline.recompute()} ключей")
        else:
            print("нечего скрывать (или все посты уже проверены)")


async def cmd_price(settings: Settings, args: argparse.Namespace) -> None:
    from lalascaner.bot.formatting import money
    from lalascaner.market.query import parse_query

    async with app_context(settings, with_ai=False) as pipeline:
        qf = parse_query(" ".join(args.query))
        async with pipeline.sessionmaker() as session:
            for stat in await repo.stats_for(session, qf, limit=10):
                by_cond = await repo.stats_by_key(session, stat.key)
                extra = " · ".join(
                    f"{c}: {money(by_cond[c].median)} (n={by_cond[c].n})" for c in ("used", "new") if c in by_cond
                )
                print(f"{stat.title:<40} медиана {money(stat.median):>12}  n={stat.n:<4} {extra}")
            print("\nсамые дешёвые за 14 дней:")
            for item in await repo.search_items(session, qf, limit=10, days=14, order="cheap"):
                print(f"  {money(item.price_kgs):>12}  {item.condition:<7} {item.model[:50]:<50} {item.listing.url}")


async def cmd_stats(settings: Settings, args: argparse.Namespace) -> None:
    async with app_context(settings, with_ai=False) as pipeline:
        async with pipeline.sessionmaker() as session:
            stats = await repo.db_stats(session)
            stats["classifier"] = settings.effective_classifier
            if settings.ai_enabled:
                stats["ai_usage"] = await repo.get_kv(session, "ai_usage", {})
            if settings.telegram_enabled:
                stats["channels"] = [(c.username, c.kind, c.last_message_id) for c in await repo.list_channels(session)]
        for key, value in stats.items():
            print(f"{key}: {value}")


async def cmd_add_channel(settings: Settings, args: argparse.Namespace) -> None:
    if not settings.telegram_enabled:
        print("Telegram-источники выключены (TELEGRAM_ENABLED=false)")
        return
    async with app_context(settings, with_ai=False) as pipeline:
        info = await pipeline.add_channel(args.username)
        print(info or "не удалось подключить")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="lalascaner", description="Мониторинг рынка ПК-комплектующих")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("run", help="бот + сбор данных + классификация по расписанию").set_defaults(func=cmd_run)

    p = sub.add_parser("ai-check", help="проверить, что ИИ подключён и отвечает")
    p.add_argument("--test", action="store_true", help="для Claude тоже сделать пробный (платный) разбор")
    p.set_defaults(func=cmd_ai_check)

    p = sub.add_parser("backfill", help="первичная загрузка истории")
    p.add_argument("source", choices=["lalafo", "telegram", "all"])
    p.add_argument("--messages", type=int, help="сколько последних сообщений группы загрузить")
    p.set_defaults(func=cmd_backfill)

    p = sub.add_parser("classify", help="разобрать очередь постов")
    p.add_argument("--limit", type=int, default=100000)
    p.add_argument("--batch-api", action="store_true", help="через Message Batches API (−50%% стоимости)")
    p.add_argument("--collect", action="store_true", help="забрать готовые результаты Batch API")
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--recompute", action="store_true", help="после разбора пересчитать рынок")
    p.set_defaults(func=cmd_classify)

    p = sub.add_parser("recompute", help="пересчитать рыночные цены")
    p.add_argument("--hours", type=int, default=48)
    p.set_defaults(func=cmd_recompute)

    sub.add_parser(
        "retag-topics", help="скрыть Telegram-посты из TELEGRAM_BLOCKED_TOPICS без ожидания фоновой задачи"
    ).set_defaults(func=cmd_retag_topics)

    p = sub.add_parser("price", help="рыночная цена из консоли")
    p.add_argument("query", nargs="+")
    p.set_defaults(func=cmd_price)

    sub.add_parser("stats", help="статистика базы").set_defaults(func=cmd_stats)

    p = sub.add_parser("add-channel", help="подключить Telegram-канал или группу")
    p.add_argument("username")
    p.set_defaults(func=cmd_add_channel)

    args = parser.parse_args(argv)
    settings = get_settings()
    setup_logging(settings.log_level)
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    try:
        asyncio.run(args.func(settings, args))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
