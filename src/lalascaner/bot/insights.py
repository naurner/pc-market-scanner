"""«За сколько продать» и «похожие объявления» — всё, что опирается на историю продаж и векторы."""

from __future__ import annotations

from html import escape

from aiogram import F, Router
from aiogram.filters import Command, CommandObject
from aiogram.types import CallbackQuery, ForceReply, Message

from lalascaner.ai.schema import CATEGORY_NAMES_RU
from lalascaner.bot.formatting import CONDITION_RU, listing_line, money
from lalascaner.bot.handlers import BTN_SELL, MAIN_KEYBOARD
from lalascaner.market.liquidity import for_category
from lalascaner.pipeline import Appraisal, Pipeline

router = Router()

SELL_PROMPT = "Опишите, что продаёте"
# (подпись, множитель к рыночной цене, корзина скорости продажи из market.liquidity)
TIERS = (
    ("⚡ Быстро", 0.9, "cheap"),
    ("⚖️ По рынку", 1.0, "market"),
    ("🐢 Подороже", 1.15, "high"),
)


def nice_price(value: float) -> float:
    step = 500 if value >= 20000 else 100 if value >= 2000 else 10
    return round(value / step) * step


def speed_phrase(summary: dict | None) -> str:
    if not summary:
        return "пока мало данных о продажах"
    parts = []
    if summary.get("p7") is not None:
        parts.append(f"за неделю уходит ~{summary['p7'] * 100:.0f}%")
    elif summary.get("p3") is not None:
        parts.append(f"за 3 дня уходит ~{summary['p3'] * 100:.0f}%")
    if summary.get("median_days") is not None:
        parts.append(f"обычно за ~{summary['median_days']:.0f} дн.")
    return ", ".join(parts) or "пока мало данных о продажах"


def appraisal_text(a: Appraisal) -> str:
    title = a.title or a.query
    lines = [f"💰 <b>За сколько продать: {escape(title)}</b>"]
    if a.condition in ("new", "used"):
        lines.append(f"Состояние: {CONDITION_RU[a.condition]}")
    reference = a.reference
    if a.stat is not None:
        lines.append(
            f"📊 Рынок: медиана <b>{money(a.stat.median)}</b> · обычно {money(a.stat.p25, '')}–{money(a.stat.p75)} · n={a.stat.n}"
        )
    elif a.estimate is not None:
        lines.append(f"🧮 По {a.estimate.n} похожим объявлениям: около <b>{money(a.estimate.price)}</b>")
    if a.active_n:
        lines.append(f"🟢 Сейчас продаётся: {a.active_n}, самое дешёвое {money(a.cheapest)}")
    if reference is None:
        lines.append(
            "\nНе нашёл такую модель в базе. Напишите точнее — производитель, модель, объём памяти, "
            "например <code>rtx 3060 12gb б/у</code> или <code>ryzen 5 5600</code>."
        )
        return "\n".join(lines)
    lines.append("\n<b>Варианты цены</b>:")
    category_name = CATEGORY_NAMES_RU.get(a.category, a.category) if a.category else None
    for label, mult, bucket in TIERS:
        summary, pooled = for_category(a.liquidity, a.category, bucket)
        delta = f" ({(mult - 1) * 100:+.0f}%)" if mult != 1 else ""
        # Скорость по всему рынку, когда в категории мало данных, — помечаем именно эту строку.
        note = " <i>(по всему рынку)</i>" if pooled and category_name else ""
        lines.append(f"{label} — <b>{money(nice_price(reference * mult))}</b>{delta}: {speed_phrase(summary)}{note}")
    scope = f"по категории «{category_name}»" if category_name else "по всему рынку"
    lines.append(
        f"\n<i>Скорость — {scope}, по истории lalafo за 3 месяца. «Уходит» = объявление пропало с сайта "
        "(продано или снято), поэтому это верхняя оценка. Точность растёт с каждой неделей сбора.</i>"
    )
    return "\n".join(lines)


async def ask_description(message: Message) -> None:
    await message.answer(
        f"{SELL_PROMPT} — модель и состояние, например: <code>rtx 3060 12gb б/у</code>",
        reply_markup=ForceReply(input_field_placeholder="rtx 3060 12gb б/у"),
    )


async def answer_appraisal(message: Message, text: str, pipeline: Pipeline) -> None:
    appraisal = await pipeline.appraise(text.strip())
    await message.answer(appraisal_text(appraisal), reply_markup=MAIN_KEYBOARD, disable_web_page_preview=True)


@router.message(Command("sell"))
async def cmd_sell(message: Message, command: CommandObject, pipeline: Pipeline) -> None:
    if not (command.args or "").strip():
        await ask_description(message)
        return
    await answer_appraisal(message, command.args, pipeline)


@router.message(F.text == BTN_SELL)
async def btn_sell(message: Message) -> None:
    await ask_description(message)


@router.message(F.text, F.reply_to_message.text.startswith(SELL_PROMPT))
async def sell_reply(message: Message, pipeline: Pipeline) -> None:
    await answer_appraisal(message, message.text, pipeline)


@router.callback_query(F.data.startswith("sim:"))
async def cb_similar(callback: CallbackQuery, pipeline: Pipeline) -> None:
    if pipeline.vectors is None:
        await callback.answer("Похожие работают через локальную Ollama (AI_PROVIDER=ollama)", show_alert=True)
        return
    items = await pipeline.similar_items(int(callback.data.split(":", 1)[1]))
    if items is None:
        await callback.answer("Для этого объявления векторы ещё считаются — загляните позже", show_alert=True)
        return
    if not items:
        await callback.answer("Похожих актуальных объявлений нет", show_alert=True)
        return
    await callback.answer()
    await callback.message.answer(
        "<b>🔎 Похожие актуальные объявления:</b>\n" + "\n".join(listing_line(i) for i in items),
        disable_web_page_preview=True,
    )
