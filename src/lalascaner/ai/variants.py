"""Производитель и серия товара — «версия» внутри модели.

RTX 3060 12GB — это модель, а Palit Dual и ASUS ROG Strix — её версии с разной ценой. Сделки ищутся
и по модели (дешевле всех RTX 3060 12GB), и по версии (дешевле всех Palit Dual RTX 3060 12GB).
Ключ версии: «<канонический ключ модели>|<производитель>-<серия>», например gpu:rtx-3060:12gb|palit-dual.
"""

from __future__ import annotations

import re
from collections.abc import Callable

from lalascaner.ai.normalize import pretty_key, slugify
from lalascaner.ai.schema import ExtractedItem

Token = str | Callable[[re.Match], str] | None
Rule = tuple[str, Token]

VARIANT_CATEGORIES = {"gpu", "ram", "ssd", "hdd", "psu", "monitor"}

_GPU_VENDORS: list[Rule] = [
    (r"\basus\b|\brog\b|\bstrix\b|\btuf\b", "asus"),
    (r"\bmsi\b|\bsuprim\b|\bventus\b", "msi"),
    (r"\bgigabyte\b|\baorus\b", "gigabyte"),
    (r"\bpalit\b", "palit"),
    (r"\bzotac\b", "zotac"),
    (r"\bcolorful\b|\bigame\b", "colorful"),
    (r"\binno\s*3d\b|\bichill\b", "inno3d"),
    (r"\bsapphire\b", "sapphire"),
    (r"\bpower\s*color\b|\bred\s*devil\b|\bred\s*dragon\b|\bhellhound\b", "powercolor"),
    (r"\bxfx\b", "xfx"),
    (r"\basrock\b", "asrock"),
    (r"\bpny\b", "pny"),
    (r"\bmanli\b", "manli"),
    (r"\bmaxsun\b", "maxsun"),
    (r"\bafox\b", "afox"),
    (r"\bbiostar\b", "biostar"),
    (r"\bgalax\b|\bkfa2\b", "galax"),
    (r"\bgainward\b", "gainward"),
    (r"\byeston\b", "yeston"),
    (r"\bsoyo\b", "soyo"),
    (r"\bpeladn\b", "peladn"),
    (r"\bmllse\b", "mllse"),
    (r"\bsinotex\b|\bninja\b", "sinotex"),
    (r"\barktek\b", "arktek"),
    (r"founders\s*edition", "nvidia"),
]
_GPU_SERIES: list[Rule] = [
    (r"\brog\s*strix\b|\bstrix\b", "strix"),
    (r"\btuf\b", "tuf"),
    (r"\bdual\b", "dual"),
    (r"\bphoenix\b", "phoenix"),
    (r"\bprime\b", "prime"),
    (r"\bproart\b", "proart"),
    (r"\bturbo\b", "turbo"),
    (r"\bsuprim\b", "suprim"),
    (r"\bgaming\s*x?\s*trio\b", "gaming-trio"),
    (r"\bgaming\s*[xz]\b", "gaming-x"),
    (r"\bventus\b", "ventus"),
    (r"\bmech\b", "mech"),
    (r"\barmor\b", "armor"),
    (r"\baero\b", "aero"),
    (r"\baorus\b", "aorus"),
    (r"\bgaming\s*oc\b", "gaming-oc"),
    (r"\beagle\b", "eagle"),
    (r"\bwindforce\b", "windforce"),
    (r"\bvision\b", "vision"),
    (r"\bgamerock\b", "gamerock"),
    (r"\bgaming\s*pro\b", "gamingpro"),
    (r"\bjetstream\b", "jetstream"),
    (r"\bstormx\b", "stormx"),
    (r"\bamp\b", "amp"),
    (r"\btrinity\b", "trinity"),
    (r"\btwin\s*edge\b", "twin-edge"),
    (r"\btwin\s*x2\b", "twin-x2"),
    (r"\bichill\b", "ichill"),
    (r"\bigame\b", "igame"),
    (r"\bbattle[\s-]*ax\b", "battle-ax"),
    (r"\btomahawk\b", "tomahawk"),
    (r"\bnitro\b", "nitro"),
    (r"\bpulse\b", "pulse"),
    (r"\btoxic\b", "toxic"),
    (r"\bred\s*devil\b", "red-devil"),
    (r"\bred\s*dragon\b", "red-dragon"),
    (r"\bhellhound\b", "hellhound"),
    (r"\bfighter\b", "fighter"),
    (r"\bspeedster\b", "speedster"),
    (r"\bthicc\b", "thicc"),
    (r"founders\s*edition", "fe"),
    (r"\bphantom\s*gaming\b", "phantom"),
    (r"\bchallenger\b", "challenger"),
]

_SSD_VENDORS: list[Rule] = [
    (r"\bsamsung\b", "samsung"),
    (r"\bkingston\b", "kingston"),
    (r"\bwd\b|western\s*digital|\bwdc\b", "wd"),
    (r"\bcrucial\b", "crucial"),
    (r"\badata\b|\bxpg\b", "adata"),
    (r"\btranscend\b", "transcend"),
    (r"\bnetac\b", "netac"),
    (r"\bkioxia\b", "kioxia"),
    (r"\bhikvision\b|\bhiksemi\b", "hikvision"),
    (r"\bapacer\b", "apacer"),
    (r"silicon\s*power", "silicon-power"),
    (r"\bpatriot\b", "patriot"),
    (r"team\s*group|t-?force", "teamgroup"),
    (r"\blexar\b", "lexar"),
    (r"\bsandisk\b", "sandisk"),
    (r"\bintel\b", "intel"),
    (r"\bseagate\b", "seagate"),
    (r"\bgigabyte\b", "gigabyte"),
    (r"\bsmartbuy\b", "smartbuy"),
    (r"\bkingspec\b", "kingspec"),
    (r"\bgoldenfir\b", "goldenfir"),
    (r"hynix", "hynix"),
    (r"\bmicron\b", "micron"),
]
_SSD_SERIES: list[Rule] = [
    (r"\b9[789]0\s*(evo\s*plus|evo|pro)\b|\b8[67]0\s*(evo|qvo|pro)\b", None),
    (r"\bpm9[0-9a-z]{2}\b", None),
    (r"\bnv[123]\b", None),
    (r"\bs?a400\b", "a400"),
    (r"\bkc\d{4}\b", None),
    (r"fury\s*renegade", "fury-renegade"),
    (r"\bsn\d{3}x?\b", None),
    (r"\bp[35]\s*plus\b", None),
    (r"\bp[235]\b", None),
    (r"\b[mb]x500\b", None),
    (r"\bt[57]00\b", None),
    (r"\blegend\s*\d{3}\b|\blegend\b", None),
    (r"\bsu\d{3}\b|\bsx\d{4}\w*\b", None),
    (r"\bnv\d{4}\b|\bn5\d{2}s\b", None),
    (r"\bnm\d{3}\b", None),
    (r"ultra\s*3d", "ultra-3d"),
    (r"\b(blue|black|green|red)\b", None),
]

_RAM_VENDORS: list[Rule] = [
    (r"\bhyperx\b", "hyperx"),
    (r"\bkingston\b", "kingston"),
    (r"\bcorsair\b", "corsair"),
    (r"g\.?\s*skill", "g-skill"),
    (r"\bcrucial\b", "crucial"),
    (r"\badata\b|\bxpg\b", "adata"),
    (r"\bpatriot\b", "patriot"),
    (r"team\s*group|t-?force", "teamgroup"),
    (r"\bsamsung\b", "samsung"),
    (r"hynix", "hynix"),
    (r"\bmicron\b", "micron"),
    (r"\bgoodram\b", "goodram"),
    (r"\bnetac\b", "netac"),
    (r"\bkingbank\b", "kingbank"),
    (r"silicon\s*power", "silicon-power"),
    (r"\bapacer\b", "apacer"),
    (r"\bgeil\b", "geil"),
    (r"\bhikvision\b", "hikvision"),
    (r"\blexar\b", "lexar"),
    (r"\bkllisre\b", "kllisre"),
    (r"\batermiter\b", "atermiter"),
    (r"\bseiwhale\b", "seiwhale"),
    (r"great\s*wall", "great-wall"),
    (r"\basgard\b", "asgard"),
    (r"\bcolorful\b", "colorful"),
]
_RAM_SERIES: list[Rule] = [
    (r"fury\s*beast", "fury-beast"),
    (r"fury\s*renegade", "fury-renegade"),
    (r"\bfury\b", "fury"),
    (r"vengeance\s*lpx", "vengeance-lpx"),
    (r"vengeance\s*rgb", "vengeance-rgb"),
    (r"\bvengeance\b", "vengeance"),
    (r"\bdominator\b", "dominator"),
    (r"trident\s*z", "trident-z"),
    (r"\bripjaws\b", "ripjaws"),
    (r"\bballistix\b", "ballistix"),
    (r"\bspectrix\b", "spectrix"),
    (r"\bgammix\b", "gammix"),
    (r"\bviper\b", "viper"),
    (r"t-?force\s*delta|\bdelta\b", "delta"),
    (r"\bvulcan\b", "vulcan"),
    (r"\bpredator\b", "predator"),
    (r"battle[\s-]*ax", "battle-ax"),
    (r"\bpolaris\b", "polaris"),
]

_PSU_VENDORS: list[Rule] = [
    (r"\bcorsair\b", "corsair"),
    (r"\bseasonic\b", "seasonic"),
    (r"be\s*quiet", "be-quiet"),
    (r"\bdeepcool\b", "deepcool"),
    (r"\bchieftec\b", "chieftec"),
    (r"\baerocool\b", "aerocool"),
    (r"\bcougar\b", "cougar"),
    (r"\bthermaltake\b", "thermaltake"),
    (r"\bzalman\b", "zalman"),
    (r"1st\s*player", "1stplayer"),
    (r"\bfsp\b", "fsp"),
    (r"\bgamemax\b", "gamemax"),
    (r"\bmontech\b", "montech"),
    (r"super\s*flower", "super-flower"),
    (r"cooler\s*master", "cooler-master"),
    (r"\bnzxt\b", "nzxt"),
    (r"\bhuntkey\b", "huntkey"),
    (r"\bxpg\b", "xpg"),
    (r"\bmsi\b", "msi"),
    (r"\basus\b", "asus"),
    (r"\bgigabyte\b", "gigabyte"),
    (r"\bpowercase\b", "powercase"),
    (r"\bexegate\b", "exegate"),
    (r"\bginzzu\b", "ginzzu"),
    (r"\baccord\b", "accord"),
]
_PSU_SERIES: list[Rule] = [
    (r"\b(rm|cx|hx|sf|cv)\s*\d{3,4}\s*(x|e|m|i)?\b", lambda m: m.group(1) + (m.group(2) or "")),
    (r"\b(cmx|gex|gx|stx|vte|xtc)\b", lambda m: m.group(1)),
    (r"\b(pk|pq|dq|pf|pm|pn|px)\s*\d{3,4}", lambda m: m.group(1)),
    (r"\b(pure|system|straight|dark)\s*power", lambda m: f"{m.group(1)}-power"),
    (r"\b(proton|gpe|gpc|gps|atmos)\b", lambda m: m.group(1)),
    (r"\b(kcas|vx)\b", lambda m: m.group(1)),
    (r"\b(smart|toughpower|litepower)\b", lambda m: m.group(1)),
    (r"\b(focus|core)\b", lambda m: m.group(1)),
    (r"\bmwe\b", "mwe"),
    (r"\bmag\s*a\d{3,4}", "mag"),
    (r"\b(century|gamma)\b", lambda m: m.group(1)),
]

_MONITOR_VENDORS: list[Rule] = [
    (r"\bsamsung\b", "samsung"),
    (r"\blg\b", "lg"),
    (r"\baoc\b", "aoc"),
    (r"\basus\b", "asus"),
    (r"\bmsi\b", "msi"),
    (r"\bacer\b", "acer"),
    (r"\bbenq\b", "benq"),
    (r"\bphilips\b", "philips"),
    (r"\bdell\b", "dell"),
    (r"\bhp\b", "hp"),
    (r"\bxiaomi\b|\bredmi\b", "xiaomi"),
    (r"\bgigabyte\b", "gigabyte"),
    (r"\bviewsonic\b", "viewsonic"),
    (r"\biiyama\b", "iiyama"),
    (r"\blenovo\b", "lenovo"),
    (r"\bhuawei\b", "huawei"),
    (r"titan\s*army", "titan-army"),
    (r"\bkoorui\b", "koorui"),
    (r"\bjump\b", "jump"),
    (r"\bhikvision\b", "hikvision"),
]
_MONITOR_SERIES: list[Rule] = [(r"\b(ips|va|tn|oled)\b", lambda m: m.group(1))]

_HDD_VENDORS: list[Rule] = [
    (r"\bseagate\b", "seagate"),
    (r"\bwd\b|western\s*digital|\bwdc\b", "wd"),
    (r"\btoshiba\b", "toshiba"),
    (r"hitachi|hgst", "hgst"),
    (r"\bsamsung\b", "samsung"),
]
_HDD_SERIES: list[Rule] = [
    (r"\b(barracuda|ironwolf|skyhawk|exos)\b", lambda m: m.group(1)),
    (r"\b([pxns]300)\b", lambda m: m.group(1)),
    (r"\b(purple|blue|black|red|gold)\b", lambda m: m.group(1)),
]

_VENDORS = {"gpu": _GPU_VENDORS, "ssd": _SSD_VENDORS, "ram": _RAM_VENDORS, "psu": _PSU_VENDORS,
            "monitor": _MONITOR_VENDORS, "hdd": _HDD_VENDORS}
_SERIES = {"gpu": _GPU_SERIES, "ssd": _SSD_SERIES, "ram": _RAM_SERIES, "psu": _PSU_SERIES,
           "monitor": _MONITOR_SERIES, "hdd": _HDD_SERIES}
_COMPILED: dict[str, re.Pattern] = {}


def _compiled(pattern: str) -> re.Pattern:
    if pattern not in _COMPILED:
        _COMPILED[pattern] = re.compile(pattern, re.I)
    return _COMPILED[pattern]


def _token(match: re.Match, token: Token) -> str:
    if token is None:
        return slugify(match.group(0))
    if callable(token):
        return slugify(token(match))
    return token


def _earliest(rules: list[Rule], text: str) -> str | None:
    """Самое раннее совпадение в тексте; при равной позиции — правило выше в списке."""
    best: tuple[int, int, str] | None = None
    for order, (pattern, token) in enumerate(rules):
        m = _compiled(pattern).search(text)
        if m and (best is None or (m.start(), order) < best[:2]):
            best = (m.start(), order, _token(m, token))
    return best[2] if best else None


def _first_by_priority(rules: list[Rule], text: str) -> str | None:
    for pattern, token in rules:
        if m := _compiled(pattern).search(text):
            return _token(m, token)
    return None


def detect_vendor(category: str, text: str | None) -> str | None:
    if not text or category not in _VENDORS:
        return None
    return _earliest(_VENDORS[category], text.lower())


def detect_series(category: str, vendor: str | None, text: str | None) -> str | None:
    if not text or category not in _SERIES:
        return None
    t = text.lower()
    if category == "gpu":
        return _earliest(_SERIES["gpu"], t)
    rules = _SERIES[category]
    if category == "ssd" and vendor != "samsung":
        rules = rules[1:]  # «980 pro» и т.п. — только у Samsung
    return _first_by_priority(rules, t)


_RE_KIT = re.compile(r"\b([248])\s*[xх×*]\s*\d{1,2}\s*(gb|гб)?\b", re.I)


def variant_signature(item: ExtractedItem) -> str | None:
    category = item.category
    if category not in VARIANT_CATEGORIES:
        return None
    text = " ".join(filter(None, [item.brand, item.series, item.model]))
    vendor = detect_vendor(category, item.brand) or detect_vendor(category, text)
    if vendor is None and item.brand:
        vendor = slugify(item.brand) or None
    if vendor is None:
        return None
    series = detect_series(category, vendor, item.series) or detect_series(category, vendor, text)
    if series is None and item.series:
        series = "-".join(slugify(item.series).split("-")[:2]) or None
    parts = [vendor] + ([series] if series and series != vendor else [])
    if category == "ram":
        modules = item.modules
        if not modules and (m := _RE_KIT.search(item.model)):
            modules = int(m.group(1))
        if modules:
            parts.append(f"{modules}x")
    return "-".join(parts)


def build_variant_key(item: ExtractedItem, canonical_key: str) -> str | None:
    signature = variant_signature(item)
    return f"{canonical_key}|{signature}"[:160] if signature else None


_PRETTY_VENDORS = {
    "asus": "ASUS", "msi": "MSI", "wd": "WD", "xfx": "XFX", "pny": "PNY", "aoc": "AOC", "lg": "LG", "hp": "HP",
    "fsp": "FSP", "g-skill": "G.Skill", "inno3d": "Inno3D", "powercolor": "PowerColor", "deepcool": "DeepCool",
    "be-quiet": "be quiet!", "hgst": "HGST", "adata": "ADATA", "nzxt": "NZXT", "benq": "BenQ", "hyperx": "HyperX",
}


def _pretty_signature(signature: str) -> str:
    for vendor, name in sorted(_PRETTY_VENDORS.items(), key=lambda kv: -len(kv[0])):
        if signature == vendor or signature.startswith(vendor + "-"):
            rest = signature[len(vendor) + 1 :]
            return " ".join([name] + [_pretty_word(w) for w in rest.split("-") if w])
    return " ".join(_pretty_word(w) for w in signature.split("-") if w)


def _pretty_word(word: str) -> str:
    if re.fullmatch(r"\d+x", word):
        return word
    if any(ch.isdigit() for ch in word) or (len(word) <= 3 and word not in ("pro", "amp", "red", "evo")):
        return word.upper()
    return word.capitalize()


def pretty_variant(variant_key: str) -> str:
    canonical, _, signature = variant_key.partition("|")
    title = pretty_key(canonical)
    return f"{title} · {_pretty_signature(signature)}" if signature else title
