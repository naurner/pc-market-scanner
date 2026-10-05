from __future__ import annotations

from datetime import datetime
from html import escape

from lalascaner.ai.normalize import pretty_key
from lalascaner.ai.variants import pretty_variant
from lalascaner.db.models import Item, Listing, PriceStat, utcnow

CONDITION_RU = {"new": "новое", "used": "б/у", "broken": "нерабочее", "unknown": "состояние ?", "all": "все"}
SOURCE_RU = {"lalafo": "lalafo", "telegram": "Telegram"}


def money(value: float | None, currency: str = "сом") -> str:
    if value is None:
        return "—"
    return f"{value:,.0f}".replace(",", " ") + (f" {currency}" if currency else "")


def short_money(value: float) -> str:
    if value >= 1000:
        k = value / 1000
        return (f"{k:.1f}".rstrip("0").rstrip(".") if k < 100 else f"{k:.0f}") + "к"
    return f"{value:.0f}"


def age(dt: datetime) -> str:
    minutes = int((utcnow() - dt).total_seconds() // 60)
    if minutes < 60:
        return f"{max(minutes, 1)} мин назад"
    if minutes < 48 * 60:
        return f"{minutes // 60} ч назад"
    return f"{minutes // (60 * 24)} дн назад"


def freshness(listing: Listing) -> str:
    """lalafo-объявление может годами «висеть» активным — вместо «915 дн назад» пишем, с какого времени."""
    if listing.source == "lalafo" and (utcnow() - listing.changed_at).days > 30:
        return f"висит с {listing.changed_at:%m.%Y}"
    return age(listing.changed_at)


def item_title(item: Item) -> str:
    """Модель, а если известна версия — модель · производитель серия."""
    return pretty_variant(item.variant_key) if item.variant_key else pretty_key(item.canonical_key)


def original_price(item: Item) -> str:
    if item.price is None:
        return "цена не указана"
    if item.currency and item.currency != "KGS":
        return f"{money(item.price, item.currency)} ≈ {money(item.price_kgs)}"
    return money(item.price)


def snippet(text: str, limit: int = 220) -> str:
    text = " ".join(text.split())
    return escape(text[:limit] + ("…" if len(text) > limit else ""))


def deal_reason(item: Item) -> str:
    """Почему это сделка — одной строкой."""
    if item.deal_kind in ("model", "variant") and item.ref_min_price and item.price_kgs is not None:
        scope = pretty_key(item.canonical_key) if item.deal_kind == "model" else pretty_variant(item.variant_key or "")
        gap = item.ref_min_price - item.price_kgs
        return (
            f"📉 Дешевле всех {item.ref_count} актуальных объявлений «{escape(scope)}»: "
            f"самое дешёвое было {money(item.ref_min_price)} (−{money(gap)})"
        )
    if item.deal_kind == "median" and item.discount is not None:
        return f"📉 На {item.discount * 100:.0f}% ниже медианы рынка"
    return ""


INTEREST_ICONS = (("favorites", "❤️ в избранном"), ("callers", "📞 звонили"), ("writers", "💬 писали"))


def interest_line(listing: Listing) -> str:
    """Интерес покупателей по данным lalafo — если уже кто-то проявил."""
    meta = listing.meta or {}
    parts = [f"{label} {meta[key]}" for key, label in INTEREST_ICONS if meta.get(key)]
    return " · ".join(parts)


def deal_message(item: Item, previous_price: float | None = None) -> str:
    listing = item.listing
    if item.deal_kind == "model":
        head = f"🔥 <b>Самая низкая цена на {escape(pretty_key(item.canonical_key))}</b>"
    elif item.deal_kind == "variant":
        head = f"🔥 <b>Самая низкая цена на версию {escape(pretty_variant(item.variant_key or ''))}</b>"
    elif item.is_deal:
        head = f"🔥 <b>{escape(item_title(item))}</b>"
    else:
        head = f"🆕 <b>{escape(item_title(item))}</b>"
    lines = [head, f"💰 <b>{original_price(item)}</b>"]
    if previous_price is not None:
        lines.insert(0, f"📉 <b>Снизили цену</b>: было {money(previous_price)}")
    if reason := deal_reason(item):
        lines.append(reason)
    if item.market_median is not None:
        _, _, condition = (item.market_key or "").partition("|")
        lines.append(f"📊 Медиана рынка ({CONDITION_RU.get(condition, condition)}): {money(item.market_median)} · n={item.market_n}")
    elif item.est_price is not None:
        lines.append(f"🧮 По {item.est_n} похожим объявлениям: около {money(item.est_price)}")
    if interest := interest_line(listing):
        lines.append(interest)
    if item.suspicious:
        lines.append("⚠️ <b>Подозрительно дёшево</b> — возможен скам или ошибка в объявлении")
    if item.model and item.model.lower() != item_title(item).lower():
        lines.append(f"🏷 {escape(item.model)}")
    meta = [SOURCE_RU.get(listing.source, listing.source)]
    if listing.source == "telegram":
        meta[0] += f" @{listing.source_ref}"
    if listing.city:
        meta.append(escape(listing.city))
    meta.append(CONDITION_RU.get(item.condition, item.condition))
    meta.append(freshness(listing))
    lines.append("📍 " + " · ".join(meta))
    if item.quantity > 1:
        lines.append(f"📦 {item.quantity} шт.")
    lines.append(f"<i>{snippet(listing.text or listing.title)}</i>")
    return "\n".join(lines)


DEAL_TAGS = {"model": " 🔥мин. цена", "variant": " 🔥мин. версии"}


def listing_line(item: Item) -> str:
    listing = item.listing
    tag = ""
    if item.is_deal:
        tag = DEAL_TAGS.get(item.deal_kind or "", f" 🔥−{(item.discount or 0) * 100:.0f}%")
    return (
        f"• <a href=\"{escape(listing.url)}\">{escape(item_title(item))}</a> — {original_price(item)}{tag}\n"
        f"   {CONDITION_RU.get(item.condition, '')} · {SOURCE_RU.get(listing.source, listing.source)} · {freshness(listing)}"
    )


def stat_block(stats: dict[str, PriceStat]) -> str:
    main = stats.get("all")
    if main is None:
        return ""
    lines = [
        f"<b>{escape(main.title)}</b>",
        f"Медиана: <b>{money(main.median)}</b> · n={main.n}",
        f"Обычный диапазон: {short_money(main.p25)}–{short_money(main.p75)} · мин {short_money(main.min)} · макс {short_money(main.max)}",
    ]
    parts = [f"{CONDITION_RU[c]}: {money(stats[c].median)} (n={stats[c].n})" for c in ("used", "new") if c in stats]
    if parts:
        lines.append(" · ".join(parts))
    return "\n".join(lines)
