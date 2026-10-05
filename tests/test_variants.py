import pytest

from lalascaner.ai.normalize import build_keys
from lalascaner.ai.schema import ExtractedItem
from lalascaner.ai.variants import build_variant_key, detect_series, detect_vendor, pretty_variant


def item(**kw):
    kw.setdefault("model", "x")
    return ExtractedItem(**kw)


@pytest.mark.parametrize(
    "category,text,vendor,series",
    [
        ("gpu", "Palit GeForce RTX 3060 Dual 12GB", "palit", "dual"),
        ("gpu", "ASUS ROG STRIX RTX 3070 OC", "asus", "strix"),
        ("gpu", "Видеокарта ROG Strix 3080", "asus", "strix"),
        ("gpu", "Colorful iGame RTX 3060 Ultra W OC", "colorful", "igame"),
        ("gpu", "MSI RTX 4070 Gaming X Trio", "msi", "gaming-trio"),
        ("gpu", "Gigabyte AORUS RX 6800 XT Master", "gigabyte", "aorus"),
        ("gpu", "gigabyte aorus radeon rx 5700 xt 8g", "gigabyte", "aorus"),
        ("gpu", "Sapphire Nitro+ RX 580 8GB", "sapphire", "nitro"),
        ("gpu", "PowerColor Red Devil RX 6700 XT", "powercolor", "red-devil"),
        ("gpu", "Zotac RTX 3070 Ti Twin Edge", "zotac", "twin-edge"),
        ("gpu", "rtx3070ti zotac gaming trio", "zotac", "gaming-trio"),
        ("gpu", "RTX 3060 12GB", None, None),
        ("ssd", "Samsung 980 Pro 1TB NVMe", "samsung", "980-pro"),
        ("ssd", "Kingston NV2 500GB", "kingston", "nv2"),
        ("ssd", "SSD WD Blue SN580 1TB", "wd", "sn580"),
        ("ssd", "SSD Crucial BX500 500 ГБ", "crucial", "bx500"),
        ("ram", "Kingston Fury Beast DDR5 64 GB 2X32 5200", "kingston", "fury-beast"),
        ("ram", "Corsair Vengeance LPX DDR4 16GB", "corsair", "vengeance-lpx"),
        ("ram", "G.Skill Trident Z RGB 2x8 3600", "g-skill", "trident-z"),
        ("psu", "COUGAR CMX 850W BRONZE", "cougar", "cmx"),
        ("psu", "Corsair RM750x 750W Gold", "corsair", "rmx"),
        ("psu", "DeepCool PK650D 650W", "deepcool", "pk"),
        ("monitor", 'Игровой монитор AOC 27" IPS 144Hz', "aoc", "ips"),
        ("hdd", "Seagate BarraCuda 2TB", "seagate", "barracuda"),
    ],
)
def test_detect_vendor_and_series(category, text, vendor, series):
    found_vendor = detect_vendor(category, text)
    assert found_vendor == vendor
    assert detect_series(category, found_vendor, text) == series


def test_gpu_variant_key_and_title():
    it = item(category="gpu", brand="Palit", model="Palit GeForce RTX 3060 Dual 12GB", chip="RTX 3060", vram_gb=12)
    canonical, _ = build_keys(it)
    key = build_variant_key(it, canonical)
    assert key == "gpu:rtx-3060:12gb|palit-dual"
    assert pretty_variant(key) == "RTX 3060 12GB · Palit Dual"


def test_ram_variant_includes_kit_layout():
    it = item(category="ram", model="Kingston Fury Beast DDR5 2x16GB 6000", memory_type="DDR5", capacity_gb=32, speed_mhz=6000, modules=2)
    canonical, _ = build_keys(it)
    assert build_variant_key(it, canonical) == "ram:ddr5:32gb:6000|kingston-fury-beast-2x"


def test_no_variant_for_exact_categories_or_unknown_vendor():
    cpu = item(category="cpu", model="Core i5-12400F", cpu_model="i5-12400F")
    assert build_variant_key(cpu, build_keys(cpu)[0]) is None
    noname = item(category="gpu", model="RTX 3060 12GB", chip="RTX 3060", vram_gb=12)
    assert build_variant_key(noname, build_keys(noname)[0]) is None


@pytest.mark.parametrize(
    "model,chip,expected",
    [
        ("RX 580 8GB 2048SP", "RX 580", "gpu:rx-580-2048sp:8gb"),
        ("Radeon RX 580 2048 SP", "RX 580", "gpu:rx-580-2048sp:8gb"),
        ("GT 1030 2GB DDR4", "GT 1030", "gpu:gt-1030-ddr4:2gb"),
        ("GT 1030 2GB GDDR5", "GT 1030", "gpu:gt-1030:2gb"),
    ],
)
def test_gpu_model_level_variants(model, chip, expected):
    vram = 2 if "1030" in chip else 8
    assert build_keys(item(category="gpu", model=model, chip=chip, vram_gb=vram))[0] == expected
