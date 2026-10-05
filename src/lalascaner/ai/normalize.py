"""Детерминированная нормализация моделей в канонические ключи.

Ключ строит код, а не ИИ: так «RTX3060 12G» и «GeForce RTX 3060 12GB» гарантированно
попадают в одну корзину статистики. canonical_key — точная модификация,
family_key — более широкая группа для случаев, когда по точному ключу мало данных.
"""

from __future__ import annotations

import re

from lalascaner.ai.schema import ExtractedItem

_CYR_LOOKALIKES = {
    "ртх": "rtx",
    "гтх": "gtx",
    "райзен": "ryzen",
    "интел": "intel",
    "ддр": "ddr",
}


def _prep(text: str) -> str:
    t = text.lower().replace("ё", "е")
    for cyr, lat in _CYR_LOOKALIKES.items():
        t = t.replace(cyr, lat)
    return t


def slugify(text: str, max_len: int = 80) -> str:
    slug = re.sub(r"[^0-9a-zа-я]+", "-", _prep(text)).strip("-")
    return slug[:max_len].strip("-")


# ── GPU ──────────────────────────────────────────────────────────────────────

_GPU_SUFFIX = r"(?:\s*-?\s*(ti|super))?(?:\s*-?\s*(super))?"
_RE_NV = re.compile(r"\b(rtx|gtx|gt)\s*-?\s*(\d{3,4})(?!\d)" + _GPU_SUFFIX)
_RE_NV_PRO = re.compile(r"\brtx\s*-?\s*a(\d{4})\b")
_RE_QUADRO = re.compile(r"\b(quadro|tesla)[\s-]*(?:rtx[\s-]*)?([a-z]?\d{3,5})\b")
_RE_RX = re.compile(r"\b(?:radeon\s*)?rx[\s-]*(\d{3,4})(?!\d)(?:[\s-]*(xtx|xt|gre)\b)?")
_RE_RADEON = re.compile(r"\bradeon\s+(?:r[579]\s+)?(\d{3,4})(?!\d)(?:\s*(xtx|xt|gre))?")
_RE_ARC = re.compile(r"\barc\s*-?\s*([ab]\d{3})\b")
_RE_NV_BARE = re.compile(
    r"(?<![\d.,])\b(10[5-8]0|16[56]0|20[6-8]0|30[5-9]0|40[6-9]0|50[5-9]0)(?!\d)" + _GPU_SUFFIX
)
_RE_RX_BARE4 = re.compile(r"(?<![\d.,])\b([5-9]\d00)\s*(xtx|xt)\b")
_RE_RX_BARE3 = re.compile(r"(?<![\d.,])\b(470|480|570|580|590)(?!\d)")


def _nv_chip(prefix: str, number: str, s1: str | None, s2: str | None) -> str:
    parts = [prefix, number] + [s for s in (s1, s2) if s]
    return "-".join(parts)


def _earliest(t: str, rules) -> str | None:  # noqa: ANN001
    """Из всех шаблонов берём самое раннее упоминание: в «rx 5700 xt … как rtx 3060» модель — первая."""
    best: tuple[int, int, str] | None = None
    for order, (pattern, build) in enumerate(rules):
        for m in pattern.finditer(t):
            value = build(m)
            if value is None:
                continue
            if best is None or (m.start(), order) < best[:2]:
                best = (m.start(), order, value)
            break
    return best[2] if best else None


def _bare_nv(m: re.Match) -> str:
    num = m.group(1)
    return _nv_chip("gtx" if num[:2] in ("10", "16") else "rtx", num, m.group(2), m.group(3))


_RE_MINING = re.compile(r"\b(p10[246]|cmp)[\s-]*(\d{3}[a-z]{0,2})\b")

_GPU_RULES = (
    # Майнинговые карты без видеовыходов (P106-100, CMP 30HX) — не равны игровым аналогам.
    (_RE_MINING, lambda m: f"{m.group(1)}-{m.group(2)}"),
    (_RE_NV_PRO, lambda m: f"rtx-a{m.group(1)}"),
    (_RE_NV, lambda m: _nv_chip(m.group(1), m.group(2), m.group(3), m.group(4))),
    (_RE_RX, lambda m: "-".join(p for p in ("rx", m.group(1), m.group(2)) if p)),
    (_RE_RADEON, lambda m: "-".join(p for p in ("rx", m.group(1), m.group(2)) if p)),
    (_RE_ARC, lambda m: f"arc-{m.group(1)}"),
    (_RE_QUADRO, lambda m: f"{m.group(1)}-{m.group(2)}"),
)
_GPU_BARE_RULES = (
    (_RE_NV_BARE, _bare_nv),
    (_RE_RX_BARE4, lambda m: f"rx-{m.group(1)}-{m.group(2)}"),
    (_RE_RX_BARE3, lambda m: f"rx-{m.group(1)}"),
)


def norm_gpu_chip(text: str | None, allow_bare: bool = True) -> str | None:
    if not text:
        return None
    t = _prep(text)
    if found := _earliest(t, _GPU_RULES):
        return found
    return _earliest(t, _GPU_BARE_RULES) if allow_bare else None


# ── CPU ──────────────────────────────────────────────────────────────────────

_RE_ULTRA = re.compile(r"\b(?:core[\s-]*)?ultra[\s-]*([3579])[\s-]*(\d{3})([a-z]{0,2})\b")
_RE_CORE = re.compile(r"\bi([3579])\s*-?\s*(\d{4,5})([a-z]{0,2})\b")
_RE_RYZEN = re.compile(
    r"\b(?:ryzen|r)[\s-]*([3579])[\s-]*(?:pro[\s-]*)?(\d{4})\s*(x3d|xt|hx|hs|ge|x|g|f|h|u)?\b"
)
_RE_RYZEN_NO_TIER = re.compile(r"\bryzen[\s-]*(\d{4})\s*(x3d|xt|x|g|f)?\b")
_RE_XEON = re.compile(r"\bxeon[\s-]*(?:(e[357])[\s-]*)?(\d{4})(?:[\s-]*(v\d))?")
_RE_PENTIUM = re.compile(r"\b(pentium|celeron)[\s-]*(?:gold[\s-]*)?([gjn]?\d{3,4}[a-z]?)\b")
_RE_ATHLON = re.compile(r"\bathlon\s*(?:ii\s*)?(?:x4\s*)?(\d{3,4}[a-z]{0,2})\b")
_RE_INTEL_BARE5 = re.compile(r"(?<![\d.,])\b(1[0-4])(\d)(\d{2})(kf|ks|k|f|t)?\b")
_RE_INTEL_BARE4 = re.compile(r"(?<![\d.,])\b([2-9])(\d)(\d{2})(kf|ks|k|f|t)\b")
_RE_AMD_BARE4 = re.compile(r"(?<![\d.,])\b([1-9])(\d)(\d{2})(x3d|xt|x|g)\b")


def _intel_tier(gen: str, digit: str) -> str:
    if gen == "2" and digit == "6":
        return "7"
    return {"1": "3", "3": "5", "4": "5", "5": "5", "6": "5", "7": "7", "8": "7", "9": "9"}.get(digit, "5")


def _amd_tier(digit: str) -> str:
    return {"1": "3", "2": "3", "7": "7", "8": "7", "9": "9"}.get(digit, "5")


def norm_cpu(text: str | None, allow_bare: bool | str = True) -> str | None:
    """allow_bare: True — разрешить «голые» номера (12400f, 5600x, 11400), False — только с префиксом
    (i5/ryzen/xeon...), "suffixed" — голые только с буквенным суффиксом (12400f, 5600x)."""
    if not text:
        return None
    t = _prep(text)
    if found := _earliest(t, _CPU_RULES):
        return found
    if not allow_bare:
        return None

    def intel5(m: re.Match) -> str | None:
        gen, digit, rest, suffix = m.groups()
        if allow_bare == "suffixed" and not suffix:
            return None
        return f"i{_intel_tier(gen, digit)}-{gen}{digit}{rest}{suffix or ''}"

    bare_rules = (
        (_RE_INTEL_BARE5, intel5),
        (_RE_INTEL_BARE4, lambda m: f"i{_intel_tier(m.group(1), m.group(2))}-{''.join(m.groups())}"),
        (_RE_AMD_BARE4, lambda m: f"ryzen-{_amd_tier(m.group(2))}-{''.join(m.groups())}"),
    )
    return _earliest(t, bare_rules)


_CPU_RULES = (
    (_RE_ULTRA, lambda m: f"ultra-{m.group(1)}-{m.group(2)}{m.group(3)}"),
    (_RE_CORE, lambda m: f"i{m.group(1)}-{m.group(2)}{m.group(3)}"),
    (_RE_RYZEN, lambda m: f"ryzen-{m.group(1)}-{m.group(2)}{m.group(3) or ''}"),
    # «ryzen 5600x» без номера серии — серию выводим из модели
    (_RE_RYZEN_NO_TIER, lambda m: f"ryzen-{_amd_tier(m.group(1)[1])}-{m.group(1)}{m.group(2) or ''}"),
    (_RE_XEON, lambda m: "-".join(p for p in ("xeon", m.group(1), m.group(2), m.group(3)) if p)),
    (_RE_PENTIUM, lambda m: f"{m.group(1)}-{m.group(2)}"),
    (_RE_ATHLON, lambda m: f"athlon-{m.group(1)}"),
)


# ── прочие поля ──────────────────────────────────────────────────────────────

_STORAGE_STEPS = (8, 16, 32, 64, 80, 128, 160, 256, 320, 512, 750)


def storage_bucket(gb: float | None) -> str | None:
    if gb is None or gb <= 0:
        return None
    if gb >= 900:
        return f"{max(1, round(gb / 1000))}tb"
    nearest = min(_STORAGE_STEPS, key=lambda step: abs(step - gb) / step)
    return f"{nearest}gb"


def ram_capacity(gb: float | None) -> str | None:
    if gb is None or gb <= 0:
        return None
    return f"{int(round(gb))}gb"


_RES_WORDS = (
    (r"\buwqhd\b", "3440x1440"),
    (r"\b(?:w?qhd|2k|1440p)\b", "2560x1440"),
    (r"\b(?:uhd|4k|2160p)\b", "3840x2160"),
    (r"\bwfhd\b", "2560x1080"),
    (r"\b(?:full\s*hd|fhd|1080p)\b", "1920x1080"),
    (r"\bhd\+", "1600x900"),
    (r"\bhd\b", "1366x768"),
)


def norm_resolution(text: str | None) -> str | None:
    if not text:
        return None
    t = _prep(text)
    if m := re.search(r"(\d{3,4})\s*[xх×*]\s*(\d{3,4})", t):
        return f"{m.group(1)}x{m.group(2)}"
    for pattern, value in _RES_WORDS:
        if re.search(pattern, t):
            return value
    return None


def norm_efficiency(text: str | None) -> str | None:
    if not text:
        return None
    t = _prep(text)
    for level in ("titanium", "platinum", "gold", "silver", "bronze"):
        if level in t:
            return level
    return None


_RE_CHIPSET = re.compile(r"\b([abhpqxz])(\d{2,3})(e)?(?=[mi]|\b|[-_ ])")


def norm_chipset(text: str | None) -> str | None:
    if not text:
        return None
    if m := _RE_CHIPSET.search(_prep(text)):
        return f"{m.group(1)}{m.group(2)}{m.group(3) or ''}"
    return None


def norm_memory_type(text: str | None) -> str | None:
    if not text:
        return None
    if m := re.search(r"(?<!g)ddr\s*([2-5])", _prep(text)):
        return f"ddr{m.group(1)}"
    return None


_SPAN_PATTERNS = (
    _RE_NV, _RE_NV_PRO, _RE_RX, _RE_RADEON, _RE_ARC, _RE_QUADRO, _RE_NV_BARE, _RE_RX_BARE4,
    _RE_ULTRA, _RE_CORE, _RE_RYZEN, _RE_XEON, _RE_PENTIUM, _RE_ATHLON, _RE_INTEL_BARE4, _RE_AMD_BARE4,
)


def model_spans(text: str) -> list[tuple[int, int]]:
    """Позиции упоминаний моделей CPU/GPU в тексте — чтобы не принять «i5-8500» за цену 8500."""
    lowered = text.lower()
    spans = []
    for pattern in _SPAN_PATTERNS:
        spans.extend(m.span() for m in pattern.finditer(lowered))
    for m in _RE_INTEL_BARE5.finditer(lowered):
        if m.group(4):
            spans.append(m.span())
    return spans


_RE_SATA_MODELS = re.compile(r"\b(bx500|mx500|8[67]0\s*evo|870\s*qvo|s?a400|su[56][35]0|a55|s11\s*pro|ultra\s*3d|wds\d+g\d?b0a)\b")
_RE_NVME_MODELS = re.compile(r"\b(9[789]0\s*(evo|pro)?|sn\d{3}|nv[123]|kc3000|fury\s*renegade|p[35]\s*plus|mp[67]00|a2000|p300|lexar\s*nm\d+)\b")


def norm_ssd_interface(text: str | None) -> str | None:
    if not text:
        return None
    t = _prep(text)
    if t.strip() == "usb" or re.search(r"внешн|external|portable", t):
        return "usb"
    if "nvme" in t or "pcie" in t or re.search(r"\bgen\s*[345]\b", t) or _RE_NVME_MODELS.search(t):
        return "nvme"
    if _RE_SATA_MODELS.search(t):
        return "sata"
    if "sata" in t or "сата" in t or "2.5" in t:
        return "sata"
    if "m.2" in t or "m2" in t:
        return "nvme"
    return None


def _is_sodimm(text: str | None) -> bool:
    return bool(text) and bool(re.search(r"so\s*-?\s*dimm|ноут", _prep(text)))


# ── ключи ────────────────────────────────────────────────────────────────────


MAX_SLUG_WORDS = 6


def _named_slug(item: ExtractedItem) -> str:
    name = item.model
    if item.brand and item.brand.lower() not in name.lower():
        name = f"{item.brand} {name}"
    words = slugify(name).split("-")
    return "-".join(words[:MAX_SLUG_WORDS]) or "unknown"


def _is_sentence(text: str) -> bool:
    return len(slugify(text).split("-")) > MAX_SLUG_WORDS


def build_keys(item: ExtractedItem) -> tuple[str, str]:
    cat = item.category
    context = " ".join(filter(None, [item.model, item.brand]))

    if cat == "gpu":
        chip = norm_gpu_chip(item.chip) or norm_gpu_chip(item.model)
        if not chip:
            slug = f"gpu:{_named_slug(item)}"
            return slug, slug
        # Версии с тем же названием, но заметно другой начинкой и ценой — отдельные модели.
        variant_ctx = " ".join(filter(None, [item.chip, item.model, item.series])).lower()
        if chip in ("rx-580", "rx-570") and re.search(r"2048\s*-?\s*sp", variant_ctx):
            chip += "-2048sp"
        elif chip == "gt-1030" and re.search(r"(?<!g)ddr4", variant_ctx):
            chip += "-ddr4"
        family = f"gpu:{chip}"
        if item.vram_gb:
            return f"{family}:{int(round(item.vram_gb))}gb", family
        return family, family

    if cat == "cpu":
        cpu = norm_cpu(item.cpu_model) or norm_cpu(item.model)
        key = f"cpu:{cpu}" if cpu else f"cpu:{_named_slug(item)}"
        return key, key

    if cat == "ram":
        mem = norm_memory_type(item.memory_type) or norm_memory_type(context)
        cap = ram_capacity(item.capacity_gb)
        if not (mem and cap):
            slug = f"ram:{_named_slug(item)}"
            return slug, slug
        so = "-so" if _is_sodimm(item.form_factor) else ""
        family = f"ram:{mem}:{cap}"
        base = f"ram:{mem}{so}:{cap}"
        return (f"{base}:{item.speed_mhz}" if item.speed_mhz else base), family

    if cat == "ssd":
        cap = storage_bucket(item.capacity_gb)
        if not cap:
            slug = f"ssd:{_named_slug(item)}"
            return slug, slug
        iface = norm_ssd_interface(item.interface) or norm_ssd_interface(context)
        family = f"ssd:{cap}"
        return (f"ssd:{iface}:{cap}" if iface else family), family

    if cat == "hdd":
        cap = storage_bucket(item.capacity_gb)
        key = f"hdd:{cap}" if cap else f"hdd:{_named_slug(item)}"
        return key, key

    if cat == "psu":
        if not item.wattage:
            slug = f"psu:{_named_slug(item)}"
            return slug, slug
        family = f"psu:{item.wattage}w"
        eff = norm_efficiency(item.efficiency) or norm_efficiency(context)
        return (f"{family}:{eff}" if eff else family), family

    if cat == "motherboard":
        chipset = norm_chipset(item.chipset) or norm_chipset(item.model)
        key = f"mb:{_named_slug(item)}"
        family = f"mb:{chipset}" if chipset else key
        # Название-предложение («Продаю материнскую плату … не работают USB») — не модель, берём чипсет.
        if slugify(item.model) in ("", chipset) or (chipset and _is_sentence(item.model)):
            key = family
        return key, family

    if cat == "monitor":
        if not item.diagonal_in:
            slug = f"mon:{_named_slug(item)}"
            return slug, slug
        parts = [f"mon:{int(item.diagonal_in + 0.5)}"]
        res = norm_resolution(item.resolution)
        if res:
            parts.append(res)
        family = ":".join(parts)
        if res and item.refresh_hz:
            parts.append(str(item.refresh_hz))
        return ":".join(parts), family

    if cat == "laptop":
        # Серия — только латинские слова и цифры: «Lenovo Loq 15 Игровой Ноутбук» → lenovo-loq-15.
        name = item.model if not item.brand or item.brand.lower() in item.model.lower() else f"{item.brand} {item.model}"
        series = "-".join([tok for tok in slugify(name).split("-") if re.fullmatch(r"[a-z0-9]+", tok)][:5])
        cpu = norm_cpu(item.cpu_model)
        gpu = norm_gpu_chip(item.gpu_model)
        if cpu and gpu:
            config = f"{cpu}+{gpu}"
            return (f"laptop:{series}:{config}" if series else f"laptop:{config}"), f"laptop:{config}"
        slug = f"laptop:{series or _named_slug(item)}"
        return slug, slug

    if cat == "pc":
        cpu = norm_cpu(item.cpu_model)
        gpu = norm_gpu_chip(item.gpu_model)
        if gpu:
            return f"pc:{cpu or 'unknown'}+{gpu}", f"pc:{gpu}"
        if cpu:
            key = f"pc:{cpu}+igpu"
            return key, key
        slug = f"pc:{_named_slug(item)}"
        return slug, slug

    slug = f"{cat}:{_named_slug(item)}"
    return slug, slug


# ── человекочитаемые названия ────────────────────────────────────────────────

_UPPER_TOKENS = {"rtx", "gtx", "gt", "rx", "xt", "xtx", "gre"}


def _pretty_gpu(chip: str) -> str:
    tokens = chip.split("-")
    out = []
    for tok in tokens:
        if tok in _UPPER_TOKENS:
            out.append(tok.upper())
        elif tok == "ti":
            out.append("Ti")
        elif tok in ("super", "arc", "quadro", "tesla"):
            out.append(tok.capitalize())
        else:
            out.append(tok.upper())
    return " ".join(out)


def _pretty_cpu(cpu: str, short: bool = False) -> str:
    if m := re.fullmatch(r"i([3579])-(\d{4,5})([a-z]*)", cpu):
        core = f"i{m.group(1)}-{m.group(2)}{m.group(3).upper()}"
        return core if short else f"Core {core}"
    if m := re.fullmatch(r"ultra-([3579])-(\d{3})([a-z]*)", cpu):
        return f"Core Ultra {m.group(1)} {m.group(2)}{m.group(3).upper()}"
    if m := re.fullmatch(r"ryzen-([3579])-(\d{4})([a-z0-9]*)", cpu):
        return f"Ryzen {m.group(1)} {m.group(2)}{m.group(3).upper()}"
    if cpu.startswith("xeon-"):
        parts = cpu.split("-")[1:]
        return "Xeon " + "-".join(p.upper() for p in parts if not p.startswith("v")) + "".join(
            f" {p}" for p in parts if p.startswith("v")
        )
    return _pretty_slug(cpu)


def _pretty_slug(slug: str) -> str:
    words = []
    for tok in slug.split("-"):
        if not tok:
            continue
        words.append(tok.upper() if any(ch.isdigit() for ch in tok) else tok.capitalize())
    return " ".join(words)


def pretty_key(key: str) -> str:
    cat, _, rest = key.partition(":")
    parts = rest.split(":")

    if cat == "gpu":
        name = _pretty_gpu(parts[0]) if norm_gpu_chip(parts[0].replace("-", " ")) else _pretty_slug(parts[0])
        return f"{name} {parts[1].upper()}" if len(parts) > 1 else name
    if cat == "cpu":
        return _pretty_cpu(parts[0])
    if cat == "ram" and len(parts) >= 2:
        mem = parts[0].replace("-so", "").upper()
        so = " SO-DIMM" if parts[0].endswith("-so") else ""
        speed = f" {parts[2]}MHz" if len(parts) > 2 else ""
        return f"{mem}{so} {parts[1].upper()}{speed}"
    if cat == "ssd":
        if len(parts) == 2:
            iface = "NVMe" if parts[0] == "nvme" else parts[0].upper()
            return f"SSD {iface} {parts[1].upper()}"
        return f"SSD {parts[0].upper()}" if storage_like(parts[0]) else f"SSD {_pretty_slug(parts[0])}"
    if cat == "hdd":
        return f"HDD {parts[0].upper()}" if storage_like(parts[0]) else f"HDD {_pretty_slug(parts[0])}"
    if cat == "psu":
        if re.fullmatch(r"\d+w", parts[0]):
            eff = f" {parts[1].capitalize()}" if len(parts) > 1 else ""
            return f"БП {parts[0][:-1]}W{eff}"
        return f"БП {_pretty_slug(parts[0])}"
    if cat == "mb":
        return f"Плата {_pretty_slug(parts[0])}"
    if cat == "mon":
        if parts[0].isdigit():
            out = f'Монитор {parts[0]}"'
            if len(parts) > 1:
                out += f" {parts[1]}"
            if len(parts) > 2:
                out += f" {parts[2]}Hz"
            return out
        return f"Монитор {_pretty_slug(parts[0])}"
    if cat == "laptop":
        if len(parts) == 1 and "+" in parts[0]:
            return f"Ноутбук {_pretty_config(parts[0])}"
        out = f"Ноутбук {_pretty_slug(parts[0])}"
        if len(parts) > 1:
            out += f" ({_pretty_config(parts[1])})"
        return out
    if cat == "pc":
        if "+" in parts[0]:
            return f"ПК {_pretty_config(parts[0])}"
        return f"ПК {_pretty_slug(parts[0])}"
    return _pretty_slug(parts[0])


def storage_like(token: str) -> bool:
    return bool(re.fullmatch(r"\d+(gb|tb)", token))


def _pretty_config(config: str) -> str:
    cpu, _, gpu = config.partition("+")
    cpu_name = _pretty_cpu(cpu, short=True) if cpu != "unknown" else "?"
    if not gpu:
        return cpu_name
    if gpu == "igpu":
        return f"{cpu_name} (без видеокарты)"
    return f"{cpu_name} + {_pretty_gpu(gpu)}"
