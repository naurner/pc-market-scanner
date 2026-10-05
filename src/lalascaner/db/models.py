from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class Listing(Base):
    """Сырой пост/объявление из источника. Все datetime — naive UTC."""

    __tablename__ = "listings"
    __table_args__ = (
        UniqueConstraint("source", "external_id", name="uq_listing_source_ext"),
        Index("ix_listing_status", "status"),
        Index("ix_listing_hash", "content_hash"),
        Index("ix_listing_posted", "posted_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str] = mapped_column(String(16))  # lalafo | telegram
    source_ref: Mapped[str] = mapped_column(String(64), default="")  # категория lalafo / username канала
    external_id: Mapped[str] = mapped_column(String(64))
    url: Mapped[str] = mapped_column(String(512), default="")
    title: Mapped[str] = mapped_column(String(512), default="")
    text: Mapped[str] = mapped_column(Text, default="")
    price: Mapped[float | None] = mapped_column(Float, nullable=True)
    currency: Mapped[str | None] = mapped_column(String(8), nullable=True)
    city: Mapped[str | None] = mapped_column(String(64), nullable=True)
    author_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    author_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    images: Mapped[list[str]] = mapped_column(JSON, default=list)
    posted_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    fetched_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    # Когда объявление появилось или поменялось (цена/текст) — по нему решаем, «свежее» ли оно для уведомлений.
    changed_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    content_hash: Mapped[str] = mapped_column(String(40), default="")

    status: Mapped[str] = mapped_column(String(16), default="new")  # new | done | failed
    intent: Mapped[str | None] = mapped_column(String(16), nullable=True)  # sell | buy | service | other
    classifier: Mapped[str | None] = mapped_column(String(16), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Telegram: id темы (см. RawListing.topic_id). NULL — ещё не проверяли (старые записи до этого поля),
    # 0 — проверили, темы нет, другое число — тема, по которой можно скрыть болтовню/blacklist/ищу-обмен.
    topic_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Всё, что источник знает сверх текста: просмотры, избранное, звонки, VIP, продавец, координаты,
    # параметры (см. sources.lalafo.listing_meta). Последние значения; история — в ListingMetric.
    meta: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    # lalafo: когда объявление пропало из ленты (продано или снято). NULL — активно или не отслеживается.
    gone_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    metrics_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    # Перезалив: то же объявление (та же модель и цена, почти тот же текст), выложенное раньше под другим id.
    dup_of: Mapped[int | None] = mapped_column(Integer, nullable=True)

    items: Mapped[list[Item]] = relationship(
        back_populates="listing", cascade="all, delete-orphan", order_by="Item.id"
    )


class Item(Base):
    """Конкретная позиция (видеокарта, процессор, ...) извлечённая из поста."""

    __tablename__ = "items"
    __table_args__ = (
        Index("ix_item_key", "canonical_key"),
        Index("ix_item_family", "family_key"),
        Index("ix_item_category_posted", "category", "posted_at"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    listing_id: Mapped[int] = mapped_column(ForeignKey("listings.id", ondelete="CASCADE"))
    category: Mapped[str] = mapped_column(String(16))
    intent: Mapped[str] = mapped_column(String(10), default="sell")  # sell | buy | exchange
    brand: Mapped[str | None] = mapped_column(String(64), nullable=True)
    series: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model: Mapped[str] = mapped_column(String(256))
    canonical_key: Mapped[str] = mapped_column(String(128))
    family_key: Mapped[str] = mapped_column(String(128))
    # Модель + производитель + серия: gpu:rtx-3060:12gb|palit-dual. None — версию определить не удалось.
    variant_key: Mapped[str | None] = mapped_column(String(160), nullable=True, index=True)
    attrs: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    condition: Mapped[str] = mapped_column(String(8), default="unknown")  # new | used | unknown
    price: Mapped[float | None] = mapped_column(Float, nullable=True)
    currency: Mapped[str | None] = mapped_column(String(8), nullable=True)
    price_kgs: Mapped[float | None] = mapped_column(Float, nullable=True)
    quantity: Mapped[int] = mapped_column(Integer, default=1)
    is_bundle: Mapped[bool] = mapped_column(Boolean, default=False)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    posted_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    # Оценка относительно рынка на момент разбора
    market_key: Mapped[str | None] = mapped_column(String(128), nullable=True)
    market_median: Mapped[float | None] = mapped_column(Float, nullable=True)
    market_n: Mapped[int | None] = mapped_column(Integer, nullable=True)
    discount: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_deal: Mapped[bool] = mapped_column(Boolean, default=False)
    suspicious: Mapped[bool] = mapped_column(Boolean, default=False)
    # Почему сделка: model — дешевле всех активных этой модели, variant — дешевле всех этой версии, median — ниже медианы
    deal_kind: Mapped[str | None] = mapped_column(String(8), nullable=True)
    ref_min_price: Mapped[float | None] = mapped_column(Float, nullable=True)  # самое дешёвое из активных на момент оценки
    ref_count: Mapped[int | None] = mapped_column(Integer, nullable=True)  # сколько активных было в сравнении
    # Оценка цены по похожим объявлениям (векторы) — для моделей, по которым мало данных для медианы.
    est_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    est_n: Mapped[int | None] = mapped_column(Integer, nullable=True)

    listing: Mapped[Listing] = relationship(back_populates="items")


class ListingEvent(Base):
    """История объявления: new, price (сменилась цена), text, bump (поднято в ленте), gone, back."""

    __tablename__ = "listing_events"
    __table_args__ = (Index("ix_event_listing", "listing_id"), Index("ix_event_kind_at", "kind", "at"))

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    listing_id: Mapped[int] = mapped_column(ForeignKey("listings.id", ondelete="CASCADE"))
    at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    kind: Mapped[str] = mapped_column(String(8))
    price: Mapped[float | None] = mapped_column(Float, nullable=True)
    currency: Mapped[str | None] = mapped_column(String(8), nullable=True)
    old_price: Mapped[float | None] = mapped_column(Float, nullable=True)


class ListingMetric(Base):
    """Снимок интереса покупателей к объявлению lalafo — не чаще раза в сутки и только если что-то изменилось."""

    __tablename__ = "listing_metrics"
    __table_args__ = (Index("ix_metric_listing", "listing_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    listing_id: Mapped[int] = mapped_column(ForeignKey("listings.id", ondelete="CASCADE"))
    at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    views: Mapped[int | None] = mapped_column(Integer, nullable=True)
    impressions: Mapped[int | None] = mapped_column(Integer, nullable=True)
    favorites: Mapped[int | None] = mapped_column(Integer, nullable=True)
    callers: Mapped[int | None] = mapped_column(Integer, nullable=True)
    writers: Mapped[int | None] = mapped_column(Integer, nullable=True)


class PriceHistory(Base):
    """Ежедневный снимок рыночных цен (price_stats) — динамика цен по моделям."""

    __tablename__ = "price_history"

    day: Mapped[str] = mapped_column(String(10), primary_key=True)  # YYYY-MM-DD
    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    condition: Mapped[str] = mapped_column(String(8), primary_key=True)
    category: Mapped[str] = mapped_column(String(16))
    n: Mapped[int] = mapped_column(Integer)
    median: Mapped[float] = mapped_column(Float)
    p25: Mapped[float] = mapped_column(Float)
    p75: Mapped[float] = mapped_column(Float)
    min: Mapped[float] = mapped_column(Float)
    max: Mapped[float] = mapped_column(Float)


class ListingVector(Base):
    """Эмбеддинг текста объявления (float16, нормирован). Производные данные — можно пересчитать заново."""

    __tablename__ = "listing_vectors"

    listing_id: Mapped[int] = mapped_column(ForeignKey("listings.id", ondelete="CASCADE"), primary_key=True)
    model: Mapped[str] = mapped_column(String(64))
    vec: Mapped[bytes] = mapped_column(LargeBinary)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class PriceStat(Base):
    __tablename__ = "price_stats"

    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    condition: Mapped[str] = mapped_column(String(8), primary_key=True)  # new | used | all
    category: Mapped[str] = mapped_column(String(16))
    title: Mapped[str] = mapped_column(String(256))
    n: Mapped[int] = mapped_column(Integer)
    median: Mapped[float] = mapped_column(Float)
    p25: Mapped[float] = mapped_column(Float)
    p75: Mapped[float] = mapped_column(Float)
    min: Mapped[float] = mapped_column(Float)
    max: Mapped[float] = mapped_column(Float)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Channel(Base):
    __tablename__ = "channels"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True)
    kind: Mapped[str] = mapped_column(String(8), default="channel")  # channel | group
    title: Mapped[str | None] = mapped_column(String(256), nullable=True)
    enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    last_message_id: Mapped[int] = mapped_column(Integer, default=0)
    added_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Subscription(Base):
    __tablename__ = "subscriptions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    chat_id: Mapped[int] = mapped_column(BigInteger, index=True)
    query: Mapped[str] = mapped_column(String(128), default="")  # пусто = все сделки
    category: Mapped[str | None] = mapped_column(String(16), nullable=True)
    max_price_kgs: Mapped[float | None] = mapped_column(Float, nullable=True)
    min_discount: Mapped[float | None] = mapped_column(Float, nullable=True)
    # deals — только ниже рынка; all — любые новые объявления по запросу
    mode: Mapped[str] = mapped_column(String(8), default="deals")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Alert(Base):
    """Отправленное в чат уведомление. Привязано к объявлению, а не к позиции: позиции пересоздаются при каждой
    правке объявления, и проверка «уже отправлял» по ним стиралась — те же объявления приходили снова."""

    __tablename__ = "alerts"
    __table_args__ = (
        Index("ix_alert_chat_listing", "chat_id", "listing_id"),
        Index("ix_alert_chat_author_key", "chat_id", "author_id", "canonical_key"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    chat_id: Mapped[int] = mapped_column(BigInteger)
    listing_id: Mapped[int] = mapped_column(ForeignKey("listings.id", ondelete="CASCADE"))
    canonical_key: Mapped[str] = mapped_column(String(128))
    author_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    price_kgs: Mapped[float | None] = mapped_column(Float, nullable=True)
    sent_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class KV(Base):
    __tablename__ = "kv"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[Any] = mapped_column(JSON)
