import pytest

from lalascaner.ai.normalize import (
    build_keys,
    norm_chipset,
    norm_cpu,
    norm_gpu_chip,
    norm_resolution,
    pretty_key,
    slugify,
    storage_bucket,
)
from lalascaner.ai.schema import ExtractedItem


@pytest.mark.parametrize(
    "text,expected",
    [
        ("GeForce RTX 3060 Ti", "rtx-3060-ti"),
        ("rtx3070ti zotac gaming trio", "rtx-3070-ti"),
        ("RTX 4070 Ti Super", "rtx-4070-ti-super"),
        ("4080 super 16gb zotac", "rtx-4080-super"),
        ("Gigabyte 5060 ti", "rtx-5060-ti"),
        ("GTX 1660 Super", "gtx-1660-super"),
        ("gtx1050ti", "gtx-1050-ti"),
        ("1060 6gb", "gtx-1060"),
        ("Radeon RX 6700 XT", "rx-6700-xt"),
        ("Rx 580 8", "rx-580"),
        ("RX 7900 XTX", "rx-7900-xtx"),
        ("Intel Arc B580", "arc-b580"),
        ("GT 1030", "gt-1030"),
        ("Quadro P2000", "quadro-p2000"),
        ("какая-то карта", None),
        (None, None),
    ],
)
def test_norm_gpu_chip(text, expected):
    assert norm_gpu_chip(text) == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("Intel Core i5 12400F", "i5-12400f"),
        ("I5-8500", "i5-8500"),
        ("i7 4790K", "i7-4790k"),
        ("Core i5-13400", "i5-13400"),
        ("Intel Core i7-9750H", "i7-9750h"),
        ("12400f", "i5-12400f"),
        ("i5 11400", "i5-11400"),
        ("Ryzen 5 5600", "ryzen-5-5600"),
        ("ryzen 7 5700x3d", "ryzen-7-5700x3d"),
        ("R9 7940HX", "ryzen-9-7940hx"),
        ("Ryzen 5 9500f", "ryzen-5-9500f"),
        ("райзен 7 5700", "ryzen-7-5700"),
        ("ryzen 5600", "ryzen-5-5600"),
        ("Ryzen 7800X3D", "ryzen-7-7800x3d"),
        ("5600x", "ryzen-5-5600x"),
        ("Core Ultra 5 245K", "ultra-5-245k"),
        ("Xeon E5-2680 v4", "xeon-e5-2680-v4"),
        ("Pentium G4560", "pentium-g4560"),
        ("3600", None),
        ("процессор", None),
    ],
)
def test_norm_cpu(text, expected):
    assert norm_cpu(text) == expected


@pytest.mark.parametrize(
    "text",
    ["Ryzen 5 9500f", "ryzen 7 5700x3d", "Core Ultra 5 245K", "Xeon E5-2680 v4", "i5 12400F", "Pentium G4560", "R9 7940HX"],
)
def test_norm_cpu_is_idempotent(text):
    once = norm_cpu(text)
    assert once is not None
    assert norm_cpu(once) == once


@pytest.mark.parametrize("text", ["RTX 4070 Ti Super", "RX 7900 XTX", "Arc B580", "gtx1050ti", "Quadro P2000", "RTX A4000"])
def test_norm_gpu_is_idempotent(text):
    once = norm_gpu_chip(text)
    assert once is not None
    assert norm_gpu_chip(once) == once


@pytest.mark.parametrize(
    "gb,expected",
    [
        (120, "128gb"),
        (128, "128gb"),
        (240, "256gb"),
        (256, "256gb"),
        (480, "512gb"),
        (500, "512gb"),
        (512, "512gb"),
        (750, "750gb"),
        (960, "1tb"),
        (1000, "1tb"),
        (1024, "1tb"),
        (2000, "2tb"),
        (8000, "8tb"),
        (None, None),
    ],
)
def test_storage_bucket(gb, expected):
    assert storage_bucket(gb) == expected


def test_earliest_mention_wins():
    text = "gigabyte aorus radeon rx 5700 xt 8g\nпо производительности равна Nvidia GeForce RTX 3060 (12 ГБ)"
    assert norm_gpu_chip(text) == "rx-5700-xt"
    assert norm_cpu("Ryzen 5 5600, быстрее чем i5 10400") == "ryzen-5-5600"


def test_long_sentence_model_does_not_become_key():
    long_model = "Продаю Материнскую Плату 1151 v2 r.1.02 ASROCK H310M-HDV Не работают USB порты"
    assert build_keys(item(category="motherboard", model=long_model)) == ("mb:h310", "mb:h310")
    key, _ = build_keys(item(category="peripheral", model="Мышь игровая беспроводная очень лёгкая с зарядной станцией белая"))
    assert len(key.split(":")[1].split("-")) <= 6


def test_norm_resolution():
    assert norm_resolution("2K") == "2560x1440"
    assert norm_resolution("Full HD") == "1920x1080"
    assert norm_resolution("3840 x 2160") == "3840x2160"
    assert norm_resolution("4k uhd") == "3840x2160"
    assert norm_resolution(None) is None


def test_norm_chipset():
    assert norm_chipset("Gigabyte H510M V2") == "h510"
    assert norm_chipset("ASUS TUF B660M-PLUS") == "b660"
    assert norm_chipset("z590i vision d") == "z590"
    assert norm_chipset("X99") == "x99"
    assert norm_chipset("какая-то плата") is None


def test_slugify():
    assert slugify("Razer DeathAdder V3 PRO") == "razer-deathadder-v3-pro"
    assert slugify("  Мышь  ATK!! ") == "мышь-atk"


def item(**kw):
    kw.setdefault("model", "x")
    return ExtractedItem(**kw)


@pytest.mark.parametrize(
    "fields,expected",
    [
        (dict(category="gpu", chip="RTX 3060", vram_gb=12), ("gpu:rtx-3060:12gb", "gpu:rtx-3060")),
        (dict(category="gpu", model="Zotac RTX 4080 Super 16GB", vram_gb=16), ("gpu:rtx-4080-super:16gb", "gpu:rtx-4080-super")),
        (dict(category="gpu", chip="RX 580"), ("gpu:rx-580", "gpu:rx-580")),
        (dict(category="cpu", cpu_model="i5 12400F"), ("cpu:i5-12400f", "cpu:i5-12400f")),
        (
            dict(category="ram", memory_type="DDR5", capacity_gb=32, speed_mhz=6000),
            ("ram:ddr5:32gb:6000", "ram:ddr5:32gb"),
        ),
        (
            dict(category="ram", memory_type="ddr4", capacity_gb=16, speed_mhz=3200, form_factor="SO-DIMM"),
            ("ram:ddr4-so:16gb:3200", "ram:ddr4:16gb"),
        ),
        (dict(category="ssd", interface="NVMe", capacity_gb=512), ("ssd:nvme:512gb", "ssd:512gb")),
        (dict(category="ssd", interface="M.2 SATA", capacity_gb=1000), ("ssd:sata:1tb", "ssd:1tb")),
        (dict(category="hdd", capacity_gb=8000), ("hdd:8tb", "hdd:8tb")),
        (dict(category="psu", wattage=850, efficiency="80+ Bronze"), ("psu:850w:bronze", "psu:850w")),
        (dict(category="psu", wattage=600), ("psu:600w", "psu:600w")),
        (dict(category="motherboard", model="Gigabyte H510M V2"), ("mb:gigabyte-h510m-v2", "mb:h510")),
        (
            dict(category="monitor", diagonal_in=27, resolution="2K", refresh_hz=165),
            ("mon:27:2560x1440:165", "mon:27:2560x1440"),
        ),
        (dict(category="monitor", diagonal_in=23.8, resolution="1920x1080"), ("mon:24:1920x1080", "mon:24:1920x1080")),
        (
            dict(category="laptop", model="Acer Predator Helios 300", cpu_model="i7-9750H", gpu_model="GTX 1660 Ti"),
            ("laptop:acer-predator-helios-300:i7-9750h+gtx-1660-ti", "laptop:i7-9750h+gtx-1660-ti"),
        ),
        (dict(category="laptop", model="MacBook Pro 14 M5 Pro"), ("laptop:macbook-pro-14-m5-pro", "laptop:macbook-pro-14-m5-pro")),
        (dict(category="pc", cpu_model="I3 8100", gpu_model="Rx 580 8"), ("pc:i3-8100+rx-580", "pc:rx-580")),
        (dict(category="pc", cpu_model="i5 12400f"), ("pc:i5-12400f+igpu", "pc:i5-12400f+igpu")),
        (
            dict(category="peripheral", brand="Razer", model="DeathAdder V3 Pro"),
            ("peripheral:razer-deathadder-v3-pro", "peripheral:razer-deathadder-v3-pro"),
        ),
        (dict(category="case", brand="Corsair", model="Corsair 2000D"), ("case:corsair-2000d", "case:corsair-2000d")),
    ],
)
def test_build_keys(fields, expected):
    assert build_keys(item(**fields)) == expected


@pytest.mark.parametrize(
    "model,expected",
    [
        ("Lenovo Loq 15 Игровой Ноутбук С", "laptop:lenovo-loq-15:ryzen-7-7435hs+rtx-4060"),
        ("Colorful Встречайте Лисян В Мире Ноутбуков", "laptop:colorful:ryzen-7-7435hs+rtx-4060"),
        ("Игровой ноутбук", "laptop:ryzen-7-7435hs+rtx-4060"),
    ],
)
def test_laptop_key_drops_marketing_words(model, expected):
    it = item(category="laptop", model=model, cpu_model="Ryzen 7 7435HS", gpu_model="RTX 4060")
    assert build_keys(it) == (expected, "laptop:ryzen-7-7435hs+rtx-4060")


def test_pretty_laptop_family():
    assert pretty_key("laptop:i7-13650hx+rtx-4060") == "Ноутбук i7-13650HX + RTX 4060"
    assert pretty_key("laptop:msi-katana-17:i5-12450h+rtx-4060") == "Ноутбук Msi Katana 17 (i5-12450H + RTX 4060)"


def test_gpu_without_parseable_chip_falls_back_to_slug():
    assert build_keys(item(category="gpu", model="Matrox G200")) == ("gpu:matrox-g200", "gpu:matrox-g200")


@pytest.mark.parametrize(
    "key,expected",
    [
        ("gpu:rtx-3060-ti:8gb", "RTX 3060 Ti 8GB"),
        ("gpu:rx-6700-xt", "RX 6700 XT"),
        ("cpu:i5-12400f", "Core i5-12400F"),
        ("cpu:ryzen-7-5700x3d", "Ryzen 7 5700X3D"),
        ("ram:ddr4:16gb:3200", "DDR4 16GB 3200MHz"),
        ("ram:ddr4-so:8gb", "DDR4 SO-DIMM 8GB"),
        ("ssd:nvme:1tb", "SSD NVMe 1TB"),
        ("ssd:512gb", "SSD 512GB"),
        ("hdd:2tb", "HDD 2TB"),
        ("psu:750w:gold", "БП 750W Gold"),
        ("mb:b660", "Плата B660"),
        ("mon:27:2560x1440:165", 'Монитор 27" 2560x1440 165Hz'),
        ("pc:i3-8100+rx-580", "ПК i3-8100 + RX 580"),
    ],
)
def test_pretty_key(key, expected):
    assert pretty_key(key) == expected
