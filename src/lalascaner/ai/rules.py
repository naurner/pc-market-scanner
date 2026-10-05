"""Классификатор на правилах.

Нужен для трёх вещей: работа без ключа API, предфильтр болтовни перед ИИ (экономия)
и гибридный режим — простые однозначные объявления разбираются без вызова модели.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from lalascaner.ai.normalize import (
    model_spans,
    norm_chipset,
    norm_cpu,
    norm_efficiency,
    norm_gpu_chip,
    norm_resolution,
)
from lalascaner.ai.schema import ExtractedItem, PostResult
from lalascaner.ai.variants import detect_series, detect_vendor

# ── цены ─────────────────────────────────────────────────────────────────────

_NUM = r"(?<![\d.,])(\d{1,3}(?:[  .,]\d{3})+(?!\d)|\d+(?:[.,]\d+)?)"
_CUR = (
    r"(\$|usd|у\.?е\.?|долл\w*|сом\w*|som\w*|kgs|с(?![а-яa-z])|c(?![а-яa-z])|₽|руб\w*|р\.(?![а-я])"
    r"|тг(?![а-я])|₸|тенге|kzt)"
)
_UNITS_AFTER = r"(?!\s*(?:гб|gb|tb|тб|w\b|вт|мгц|mhz|hz|гц|ядер|шт|%|\d|x|х|×))"

_P_USD_PREFIX = re.compile(r"\$\s*" + _NUM, re.I)
_P_WITH_CUR = re.compile(_NUM + r"\s*(к|k|тыс\.?)?\s*" + _CUR, re.I)
_P_K = re.compile(_NUM + r"\s*(к|k)(?![а-яa-z])", re.I)
_P_LABEL = re.compile(
    r"(?:цена|стоимость|прошу|price|отдам\s+за|продам\s+за|\bза|\bпо)\s*[:\-–—]?\s*" + _NUM + _UNITS_AFTER, re.I
)
_P_DASH_END = re.compile(r"[-–—]\s*" + _NUM + r"\s*$", re.I | re.M)
_P_LINE_NUM = re.compile(r"^\s*" + _NUM + r"\s*(?:\(|$)", re.I | re.M)

_LIMITS = {"KGS": (100, 5_000_000), "USD": (5, 50_000), "RUB": (100, 5_000_000), "KZT": (500, 30_000_000), "EUR": (5, 50_000)}


@dataclass
class Price:
    value: float
    currency: str
    start: int
    end: int


def _parse_num(raw: str) -> float:
    raw = raw.replace(" ", " ")
    if re.fullmatch(r"\d{1,3}(?:[ .,]\d{3})+", raw):
        return float(re.sub(r"[ .,]", "", raw))
    return float(raw.replace(",", "."))


def _currency(marker: str | None) -> str:
    if not marker:
        return "KGS"
    m = marker.lower()
    if m in ("$", "usd") or m.startswith(("долл", "у")):
        return "USD"
    if m in ("₽", "р.") or m.startswith("руб"):
        return "RUB"
    if m in ("тг", "₸", "тенге", "kzt"):
        return "KZT"
    return "KGS"


def find_prices(text: str) -> list[Price]:
    blocked = model_spans(text)
    accepted: list[Price] = []

    def free(start: int, end: int) -> bool:
        spans = blocked + [(p.start, p.end) for p in accepted]
        return all(end <= s or start >= e for s, e in spans)

    def add(start: int, end: int, value: float, currency: str) -> None:
        low, high = _LIMITS[currency]
        if low <= value <= high and free(start, end):
            accepted.append(Price(value, currency, start, end))

    for m in _P_USD_PREFIX.finditer(text):
        add(*m.span(), _parse_num(m.group(1)), "USD")
    for m in _P_WITH_CUR.finditer(text):
        value = _parse_num(m.group(1)) * (1000 if m.group(2) else 1)
        add(*m.span(), value, _currency(m.group(3)))
    for m in _P_K.finditer(text):
        num = _parse_num(m.group(1))
        if num < 1000:
            add(*m.span(), num * 1000, "KGS")
    for m in _P_LABEL.finditer(text):
        add(m.start(1), m.end(1), _parse_num(m.group(1)), "KGS")
    for m in _P_DASH_END.finditer(text):
        value = _parse_num(m.group(1))
        if value >= 300:
            add(m.start(1), m.end(1), value, "KGS")
    for m in _P_LINE_NUM.finditer(text):
        value = _parse_num(m.group(1))
        if value >= 500:
            add(m.start(1), m.end(1), value, "KGS")

    accepted.sort(key=lambda p: p.start)
    return accepted


# ── намерение и состояние ────────────────────────────────────────────────────

_RE_BUY = re.compile(r"\b(куплю|скупка|скупаю|скупаем|покупаю|покупаем|выкуп\w*|приму|ищу|нужен|нужна|нужны)\b", re.I)
_RE_SELL = re.compile(r"\b(прода\w*|отдам|цена|продам|в наличии|есть в наличии)\b", re.I)
_RE_SERVICE = re.compile(r"\b(ремонт\w*|чистк\w*|диагностик\w*|прошивк\w*|установк\w* windows|сборка на заказ)\b", re.I)
_RE_NEW = re.compile(r"(?<![а-яa-z])(нов(ый|ая|ое|ые|ую)|new|запечатан\w*|в пленке|в плёнке)(?![а-яa-z])", re.I)
_RE_USED = re.compile(
    r"(б/у|б\.у\.?|(?<![а-яa-z])бу(?![а-яa-z])|used|пользовал\w*|open\s?box|опенбокс|в (хорошем|отличном|идеальном) состоянии)",
    re.I,
)


_RE_BROKEN = re.compile(
    r"(не\s*работа\w*|нерабоч\w*|на\s*запчаст\w*|под\s*ремонт|требует\s*ремонт\w*|сломан\w*|битая\s*матриц\w*|"
    r"нет\s*изображени\w*|артефакт\w*|не\s*включа\w*|не\s*запуска\w*|(экран|матриц\w*|стекло)\s*разбит\w*|трещин\w*)",
    re.I,
)


def detect_condition(text: str, title: str = "") -> str:
    if _RE_BROKEN.search(f"{title}\n{text}"):
        return "broken"
    for source in (title, text):
        if not source:
            continue
        if _RE_USED.search(source):
            return "used"
        if _RE_NEW.search(source):
            return "new"
    return "unknown"


# ── сущности ─────────────────────────────────────────────────────────────────

_COMPONENT_WORDS = re.compile(
    r"(видеокарт|видюх|процессор|\bпроц\b|\bозу\b|оперативк|\bопер[уа]?\b|\bddr\d|\bssd\b|\bhdd\b|nvme|\bm\.2\b|"
    r"жестк|жёстк|винчестер|блок\w* питани|\bбп\b|материнк|мат\.?\s*плат|монитор|моник|ноутбук|\bноут\b|macbook|"
    r"\bпк\b|компьютер|системник|сборк|кулер|корпус|\bмыш|клавиатур|наушник|гарнитур|\brtx|\bgtx|\brx\s?\d|ryzen|"
    r"\bi[3579][ -]?\d{3,5}|xeon)",
    re.I,
)
_RE_GPU_CTX = re.compile(r"(\bti\b|super|\bxt\b|\d+\s*(gb|гб)\b|видео|видюх|zotac|palit|colorful|inno3d|sapphire|powercolor|msi|gigabyte|asus)", re.I)
_RE_CPU_CTX = re.compile(r"(процессор|\bпроц\b|\bcpu\b|сокет|lga\s?\d|\bam[345]\b|ядер)", re.I)
_RE_RAM_CTX = re.compile(r"(\bозу\b|оператив|\bопер[уа]?\b|\bram\b|планк|плашк|so-?dimm)", re.I)
_RE_SSD_CTX = re.compile(r"(\bssd\b|nvme|\bm\.?2\b)", re.I)
_RE_HDD_CTX = re.compile(r"(\bhdd\b|жестк|жёстк|винчестер)", re.I)
_RE_PSU_CTX = re.compile(r"(\bбп\b|блок\w* питани|\bpsu\b|bronze|gold|silver|platinum|titanium|80\s*\+|модульн)", re.I)
_RE_MB_CTX = re.compile(r"(материнк|мат\.?\s*плат|\bплат[аы]\b|motherboard|asus|msi|gigabyte|asrock|biostar|maxsun|jginyue|huananzhi|machinist)", re.I)
_RE_MON_CTX = re.compile(r"(монитор|моник|monitor|дюйм)", re.I)
_RE_LAPTOP_CTX = re.compile(r"(ноутбук|\bноут\b|laptop|macbook)", re.I)
_RE_PC_CTX = re.compile(r"(\bпк\b|компьютер|системник|системный блок|сборк)", re.I)
_RE_MOBILE_GPU = re.compile(r"(laptop\s*gpu|\bmobile\b|max-?q|\bмобильн)", re.I)
_RE_MOBILE_CPU = re.compile(r"\d(h|hx|hk|hs|u)$")  # i5-11400h, ryzen-7-6800hs — ноутбучные процессоры


_RE_DEVICE_BRAND = re.compile(
    r"(asus|lenovo|acer|\bhp\b|dell|msi|macbook|apple|huawei|honor|xiaomi|redmi|gigabyte|thunderobot|hasee|"
    r"samsung|legion|predator|nitro|omen|victus|rog|tuf|thinkpad|ideapad|aorus|machenike|ardor|dexp)",
    re.I,
)


def _first_line(lines: list[str]) -> str:
    """Строка с названием устройства: сначала ищем строку с брендом/серией, иначе первая непустая."""
    cleaned = [re.sub(r"^[\W_]+", "", line).strip() for line in lines]
    cleaned = [line for line in cleaned if line]
    for line in cleaned[:6]:
        if _RE_DEVICE_BRAND.search(line):
            return line[:80]
    return (cleaned[0] if cleaned else "")[:80]

_RE_GB = re.compile(r"(\d{1,4}(?:[.,]\d)?)\s*(gb|гб|tb|тб)\b", re.I)
_RE_KIT = re.compile(r"\b(\d)\s*[xх×*]\s*(\d{1,2})\s*(gb|гб)?\b", re.I)
_RE_SPEED = re.compile(r"\b(1333|1600|2133|2400|2666|2933|3000|3200|3466|3600|4000|4800|5200|5600|6000|6400|6800|7200|8000)\s*(mhz|мгц)?", re.I)
_RE_WATT = re.compile(r"\b(\d{3,4})\s*(w|вт|ватт)\b", re.I)
_RE_DIAG = re.compile(r"\b(\d{2}(?:[.,]\d)?)\s*(\"|”|''|дюйм|inch)", re.I)
_RE_HZ = re.compile(r"\b(\d{2,3})\s*(hz|гц)\b", re.I)
_RE_MEM_TYPE = re.compile(r"(?<!g)ddr\s*([2-5])", re.I)

_BRANDS = (
    "asus", "msi", "gigabyte", "zotac", "palit", "colorful", "sapphire", "powercolor", "inno3d", "asrock", "biostar",
    "kingston", "corsair", "samsung", "crucial", "adata", "xpg", "patriot", "hyperx", "g.skill", "teamgroup",
    "seagate", "toshiba", "wd", "cougar", "deepcool", "zalman", "aerocool", "chieftec", "thermaltake", "gamemax",
    "lenovo", "acer", "hp", "dell", "apple", "xiaomi", "huawei", "intel", "amd", "nvidia", "lg", "aoc", "benq", "philips",
)


def _brand(text: str) -> str | None:
    low = text.lower()
    for brand in _BRANDS:
        if re.search(rf"(?<![a-z]){re.escape(brand)}(?![a-z])", low):
            return brand.upper() if len(brand) <= 3 else brand.capitalize()
    return None


def _capacity_gb(text: str) -> float | None:
    values = []
    for m in _RE_GB.finditer(text):
        value = float(m.group(1).replace(",", "."))
        values.append(value * 1000 if m.group(2).lower() in ("tb", "тб") else value)
    return max(values) if values else None


def _ram_capacity(text: str) -> float | None:
    explicit = [v for v in (_capacity_gb(text),) if v and v <= 512]
    if explicit:
        return explicit[0]
    if m := _RE_KIT.search(text):
        return float(int(m.group(1)) * int(m.group(2)))
    return None


@dataclass
class Entity:
    category: str
    line: int
    fields: dict = field(default_factory=dict)

    def key(self) -> tuple:
        return (self.category, tuple(sorted((k, str(v)) for k, v in self.fields.items() if k != "model")))


def _entities_in(chunk: str, line_no: int, hint: str | None, whole: str) -> list[Entity]:
    found: list[Entity] = []
    model = chunk.strip()[:80]

    chip = norm_gpu_chip(chunk, allow_bare=False)
    if not chip and (hint == "gpu" or _RE_GPU_CTX.search(chunk)):
        chip = norm_gpu_chip(chunk)
    if chip:
        vram = None
        if m := re.search(r"\b(\d{1,2})\s*(gb|гб|g)\b", chunk, re.I):
            vram = float(m.group(1))
        found.append(Entity("gpu", line_no, {"chip": chip, "vram_gb": vram, "model": model}))

    cpu = norm_cpu(chunk, allow_bare=False) or norm_cpu(chunk, allow_bare="suffixed")
    if not cpu and (hint == "cpu" or _RE_CPU_CTX.search(chunk)):
        cpu = norm_cpu(chunk)
    if cpu:
        found.append(Entity("cpu", line_no, {"cpu_model": cpu, "model": model}))

    mem = _RE_MEM_TYPE.search(chunk)
    if mem or hint == "ram" or _RE_RAM_CTX.search(chunk):
        cap = _ram_capacity(chunk)
        if cap and (mem or _RE_RAM_CTX.search(whole)):
            speed = _RE_SPEED.search(chunk)
            found.append(
                Entity(
                    "ram",
                    line_no,
                    {
                        "memory_type": f"DDR{mem.group(1)}" if mem else None,
                        "capacity_gb": cap,
                        "speed_mhz": int(speed.group(1)) if speed else None,
                        "form_factor": "SO-DIMM" if re.search(r"so-?dimm|ноут", chunk, re.I) else None,
                        "model": model,
                    },
                )
            )

    is_ssd = bool(_RE_SSD_CTX.search(chunk)) or hint == "ssd"
    is_hdd = bool(_RE_HDD_CTX.search(chunk)) or hint == "hdd"
    if (is_ssd or is_hdd) and not (is_ssd and is_hdd):
        cap = _capacity_gb(chunk)
        if cap and cap >= 60:
            if is_ssd:
                if re.search(r"внешн\w*|external|portable|usb", whole, re.I):
                    iface = "usb"
                elif re.search(r"nvme|m\.?2|pcie", chunk, re.I):
                    iface = "nvme"
                else:
                    iface = "sata" if re.search(r"sata|сата", chunk, re.I) else None
                found.append(Entity("ssd", line_no, {"capacity_gb": cap, "interface": iface, "model": model}))
            else:
                found.append(Entity("hdd", line_no, {"capacity_gb": cap, "model": model}))

    if m := _RE_WATT.search(chunk):
        if hint == "psu" or _RE_PSU_CTX.search(chunk) or _RE_PSU_CTX.search(whole):
            watt = int(m.group(1))
            if 200 <= watt <= 2000:
                found.append(
                    Entity("psu", line_no, {"wattage": watt, "efficiency": norm_efficiency(chunk), "model": model})
                )

    chipset = norm_chipset(chunk)
    if chipset and not chip and (hint == "motherboard" or _RE_MB_CTX.search(chunk)):
        found.append(Entity("motherboard", line_no, {"chipset": chipset, "model": model}))

    if hint == "monitor" or _RE_MON_CTX.search(chunk):
        if m := _RE_DIAG.search(chunk):
            hz = _RE_HZ.search(chunk)
            found.append(
                Entity(
                    "monitor",
                    line_no,
                    {
                        "diagonal_in": float(m.group(1).replace(",", ".")),
                        "resolution": norm_resolution(chunk),
                        "refresh_hz": int(hz.group(1)) if hz else None,
                        "model": model,
                    },
                )
            )
    return found


@dataclass
class RuleOutcome:
    result: PostResult
    confident: bool


def looks_like_offer(text: str) -> bool:
    text = text.strip()
    if len(text) < 10:
        return False
    has_component = bool(_COMPONENT_WORDS.search(text)) or bool(model_spans(text))
    has_price = bool(find_prices(text))
    has_marker = bool(_RE_SELL.search(text) or _RE_BUY.search(text))
    if has_component and (has_price or has_marker):
        return True
    return has_price and has_marker


def _make_item(entity: Entity, condition: str, price: float | None, currency: str | None, brand_src: str) -> ExtractedItem:
    fields = {k: v for k, v in entity.fields.items() if v is not None}
    model = fields.pop("model", "") or entity.category
    # Производитель и серия: сначала из строки с моделью, затем из всего текста.
    vendor = detect_vendor(entity.category, model) or detect_vendor(entity.category, brand_src)
    series = detect_series(entity.category, vendor, model) or detect_series(entity.category, vendor, brand_src)
    return ExtractedItem(
        category=entity.category,  # type: ignore[arg-type]
        brand=vendor or _brand(brand_src),
        series=series,
        model=model,
        condition=condition,  # type: ignore[arg-type]
        price=price,
        currency=currency if price is not None else None,  # type: ignore[arg-type]
        **fields,
    )


# Однозначно аксессуар — ищем в заголовке и первых строках (бывает, что список дисков, а ниже «кейс для…»).
_RE_ACCESSORY_STRONG = re.compile(
    r"(кейс\s*для|корпус\s*для\s*(ssd|hdd|диск|жестк|m\.?2)|\bбокс\s*для|enclosure|\bдля\s+(ssd|m\.?2|nvme|hdd)\b|"
    r"\bибп\b|\bups\b|бесперебойн\w*|стабилизатор)",
    re.I,
)
# Слова, которые могут быть и частью комплекта («карта и переходник»), — только в названии товара.
_RE_ACCESSORY = re.compile(
    r"(адаптер|переходник|креплени\w*|кронштейн|\bvesa\b|подставк\w*\s*для|райзер|riser|удлинител\w*|шлейф|"
    r"салазк\w*|радиатор\s*для\s*(ssd|m\.?2)|док-?станци\w*)",
    re.I,
)
_RE_BUY_STRONG = re.compile(r"\b(куплю|скупка|скупаю|скупаем|покупаю|покупаем|выкуп\w*)\b", re.I)


def _post_intent(text: str, title: str, single_listing: bool = False) -> str:
    full = f"{title}\n{text}"
    # В карточке объявления «Нужен надёжный компьютер?» — реклама, а не покупка: там считаем только явные слова.
    buy = (_RE_BUY_STRONG if single_listing else _RE_BUY).search(full)
    sell = _RE_SELL.search(full)
    if buy and not sell:
        return "buy"
    if buy and sell and buy.start() < sell.start() and not find_prices(full):
        return "buy"
    if not single_listing and "?" in text and not sell and len(find_prices(text)) <= 1:
        return "other"
    if _RE_SERVICE.search(full) and not sell:
        return "service"
    return "sell"


def classify_rules(
    post_id: str,
    text: str,
    title: str = "",
    price: float | None = None,
    currency: str | None = None,
    category_hint: str | None = None,
) -> RuleOutcome:
    single_listing = bool(title) or price is not None
    lines = [line for line in text.split("\n") if line.strip()]
    name = "\n".join([title] + lines[:1])
    head = "\n".join([title] + lines[:3])
    if _RE_ACCESSORY_STRONG.search(head) or _RE_ACCESSORY.search(name):
        # Корпус для SSD, крепление монитора, ИБП, райзер — это не сам компонент.
        return RuleOutcome(PostResult(post_id=post_id, intent="sell", items=[]), confident=False)
    intent = _post_intent(text, title, single_listing)
    if intent != "sell":
        return RuleOutcome(PostResult(post_id=post_id, intent=intent, items=[]), confident=True)  # type: ignore[arg-type]

    condition = detect_condition(text, title)
    if single_listing:
        return _classify_single(post_id, text, title, price, currency, category_hint, condition)
    return _classify_multi(post_id, text, category_hint, condition)


def _classify_single(post_id, text, title, price, currency, hint, condition) -> RuleOutcome:
    whole = f"{title}\n{text}"
    category = hint
    if category is None:
        if _RE_LAPTOP_CTX.search(whole):
            category = "laptop"
        elif _RE_PC_CTX.search(whole) and not _RE_SSD_CTX.search(title):
            category = "pc"

    if category in ("laptop", "pc"):
        cpu = norm_cpu(whole, allow_bare=False) or norm_cpu(whole, allow_bare="suffixed")
        gpu = norm_gpu_chip(whole, allow_bare=False)
        mobile = bool(cpu and _RE_MOBILE_CPU.search(cpu)) or bool(_RE_MOBILE_GPU.search(whole))
        if category == "pc" and (mobile or _RE_LAPTOP_CTX.search(whole)):
            category = "laptop"  # ноутбук, выложенный в «Настольные ПК»
        elif category == "laptop" and cpu and not mobile and _RE_PC_CTX.search(whole):
            category = "pc"  # системник, выложенный в «Ноутбуки»
        if cpu and gpu:
            model = _first_line(text.split("\n")) if category == "laptop" else (title or text[:80])
            entity = Entity(category, 0, {"cpu_model": cpu, "gpu_model": gpu, "model": model})
            item = _make_item(entity, condition, price, currency or "KGS", whole)
            confident = category == "pc" and price is not None
            return RuleOutcome(PostResult(post_id=post_id, intent="sell", items=[item]), confident=confident)
        return RuleOutcome(PostResult(post_id=post_id, intent="sell", items=[]), confident=False)

    # Сначала текст (там реальная модель), заголовок lalafo — только как запасной вариант.
    entities: list[Entity] = []
    for chunk in (text, title, whole):
        entities = [e for e in _entities_in(chunk, 0, category, whole) if category is None or e.category == category]
        if entities:
            break
    if not entities:
        return RuleOutcome(PostResult(post_id=post_id, intent="sell", items=[]), confident=False)

    entity = entities[0]
    if entity.category == "gpu" and not entity.fields.get("vram_gb"):
        if m := re.search(r"\b(\d{1,2})\s*(gb|гб)\b", whole, re.I):
            entity.fields["vram_gb"] = float(m.group(1))
    # Несколько разных моделей одной категории в одном объявлении — это прайс, а не одна позиция.
    line_entities = [
        (line, e)
        for no, line in enumerate(text.split("\n"))
        for e in _entities_in(line, no, None, whole)
        if e.category == entity.category
    ]
    per_line = {e.key() for _, e in line_entities}
    # Название позиции — строка, где нашлась модель, а не весь текст объявления.
    if line_entities:
        entity.fields["model"] = line_entities[0][0].strip()[:80]
    elif title:
        entity.fields["model"] = title[:80]
    distinct = {e.category for e in entities}
    item = _make_item(entity, condition, price, currency or "KGS", whole)
    confident = price is not None and len(distinct) == 1 and len(per_line) <= 1
    return RuleOutcome(PostResult(post_id=post_id, intent="sell", items=[item]), confident=confident)


def _classify_multi(post_id, text, hint, condition) -> RuleOutcome:
    lines = text.split("\n")
    entities: list[Entity] = []
    seen: set = set()
    for no, line in enumerate(lines):
        if _RE_BUY.search(line):
            continue  # «куплю монитор, бюджет 5-6к» — не предложение о продаже
        for entity in _entities_in(line, no, hint, text):
            if entity.key() not in seen:
                seen.add(entity.key())
                entities.append(entity)

    prices = find_prices(text)
    line_starts = [0]
    for line in lines[:-1]:
        line_starts.append(line_starts[-1] + len(line) + 1)

    def line_of(pos: int) -> int:
        return max(i for i, start in enumerate(line_starts) if start <= pos)

    if not entities:
        return RuleOutcome(PostResult(post_id=post_id, intent="sell", items=[]), confident=not prices)

    categories = {e.category for e in entities}
    cpu = next((e.fields["cpu_model"] for e in entities if e.category == "cpu"), None)
    gpu = next((e.fields["chip"] for e in entities if e.category == "gpu"), None)
    mobile_cpu = bool(cpu and _RE_MOBILE_CPU.search(cpu))

    def whole_device(category: str) -> RuleOutcome:
        # Ноутбук/сборка — одна позиция. Если цен несколько, цена устройства — самая большая.
        price = max(prices, key=lambda p: p.value * (87 if p.currency == "USD" else 1)) if prices else None
        entity = Entity(category, 0, {"cpu_model": cpu, "gpu_model": gpu, "model": _first_line(lines)})
        item = _make_item(entity, condition, price.value if price else None, price.currency if price else None, text)
        return RuleOutcome(PostResult(post_id=post_id, intent="sell", items=[item]), confident=False)

    if _RE_LAPTOP_CTX.search(text) or _RE_MOBILE_GPU.search(text) or (mobile_cpu and len(categories) >= 2):
        return whole_device("laptop")

    if len(entities) == 1:
        price = prices[0] if prices else None
        item = _make_item(entities[0], condition, price.value if price else None, price.currency if price else None, text)
        return RuleOutcome(PostResult(post_id=post_id, intent="sell", items=[item]), confident=len(prices) == 1)

    itemized = sum(1 for e in entities if any(line_of(p.start) == e.line for p in prices)) >= 2
    if ("gpu" in categories or "cpu" in categories) and not itemized and (
        len(categories) >= 3 or {"gpu", "cpu"} <= categories or _RE_PC_CTX.search(text)
    ):
        return whole_device("pc")

    items = []
    used: set[int] = set()
    for entity in entities:
        candidates = [
            (i, p) for i, p in enumerate(prices) if i not in used and 0 <= line_of(p.start) - entity.line <= 2
        ]
        if len(prices) >= 2 and candidates:
            idx, price = candidates[0]
            used.add(idx)
            items.append(_make_item(entity, condition, price.value, price.currency, lines[entity.line]))
        else:
            items.append(_make_item(entity, condition, None, None, lines[entity.line]))
    return RuleOutcome(PostResult(post_id=post_id, intent="sell", items=items), confident=False)
