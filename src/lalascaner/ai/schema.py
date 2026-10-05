"""Структура, которую классификатор (ИИ или правила) возвращает по каждому посту."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Category = Literal[
    "gpu",
    "cpu",
    "ram",
    "motherboard",
    "ssd",
    "hdd",
    "psu",
    "cooler",
    "case",
    "monitor",
    "laptop",
    "pc",
    "peripheral",
    "other",
]
CATEGORIES: tuple[str, ...] = Category.__args__  # type: ignore[attr-defined]

CATEGORY_NAMES_RU: dict[str, str] = {
    "gpu": "Видеокарты",
    "cpu": "Процессоры",
    "ram": "Оперативная память",
    "motherboard": "Материнские платы",
    "ssd": "SSD",
    "hdd": "HDD",
    "psu": "Блоки питания",
    "cooler": "Охлаждение",
    "case": "Корпуса",
    "monitor": "Мониторы",
    "laptop": "Ноутбуки",
    "pc": "Готовые ПК",
    "peripheral": "Периферия",
    "other": "Другое",
}

Intent = Literal["sell", "buy", "exchange"]
PostIntent = Literal["sell", "buy", "service", "other"]
Condition = Literal["new", "used", "broken", "unknown"]
Currency = Literal["KGS", "USD", "RUB", "KZT", "EUR"]


class ExtractedItem(BaseModel):
    category: Category
    intent: Intent = "sell"
    brand: str | None = None
    series: str | None = None
    model: str = Field(description="Короткое нормализованное название позиции")
    chip: str | None = None
    vram_gb: float | None = None
    cpu_model: str | None = None
    gpu_model: str | None = None
    memory_type: str | None = None
    capacity_gb: float | None = None
    modules: int | None = None
    speed_mhz: int | None = None
    interface: str | None = None
    form_factor: str | None = None
    wattage: int | None = None
    efficiency: str | None = None
    chipset: str | None = None
    socket: str | None = None
    diagonal_in: float | None = None
    resolution: str | None = None
    refresh_hz: int | None = None
    ram_gb: float | None = None
    storage_gb: float | None = None
    condition: Condition = "unknown"
    price: float | None = None
    currency: Currency | None = None
    quantity: int = 1
    is_bundle: bool = False

    def spec_attrs(self) -> dict:
        """Технические поля без мета-информации — хранятся в items.attrs."""
        skip = {"category", "intent", "brand", "series", "model", "condition", "price", "currency", "quantity", "is_bundle"}
        return {k: v for k, v in self.model_dump().items() if k not in skip and v is not None}


class PostResult(BaseModel):
    post_id: str
    intent: PostIntent
    items: list[ExtractedItem]


class BatchResult(BaseModel):
    posts: list[PostResult]
