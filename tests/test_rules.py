import pytest

from lalascaner.ai.normalize import build_keys
from lalascaner.ai.rules import classify_rules, find_prices, looks_like_offer


@pytest.mark.parametrize(
    "text,expected",
    [
        ("цена 27к", [(27000, "KGS")]),
        ("I5-8500 \n4000 сом", [(4000, "KGS")]),
        ("Новый Ryzen 5 5600 box - 9900", [(9900, "KGS")]),
        ("DDR5 64 GB 2X32 5200 CL40 INTEL XMP -  57 000 сом", [(57000, "KGS")]),
        ("Цена: 3500", [(3500, "KGS")]),
        ("i5 13400 2шт - по 13.5к", [(13500, "KGS")]),
        ("32gb ddr5 (16x2) 5600mhz cl36 - 36.5k", [(36500, "KGS")]),
        ("продам за 20000с", [(20000, "KGS")]),
        ("Продаю клавиатуру, полный комплект. 2000с", [(2000, "KGS")]),
        ("rtx 3060 250$", [(250, "USD")]),
        ("цена $1 200", [(1200, "USD")]),
        ("\n10 000 (новая в Бишкеке~13000)", [(10000, "KGS")]),
        ("Corsair 2000d-4500", [(4500, "KGS")]),
        ("RTX 3060 12gb, 600w", []),
        ("i5 12400f 6 ядер 12 потоков", []),
    ],
)
def test_find_prices(text, expected):
    assert [(p.value, p.currency) for p in find_prices(text)] == expected


@pytest.mark.parametrize(
    "text,expected",
    [
        ("брат там космические спутники если надо сигнал быстрее отправят", False),
        ("Upp", False),
        ("3к гдето", False),
        ("сколько щас стоит 8гб ддр4 одной планкой?", False),
        ("I5-8500 \n4000 сом", True),
        ("продаю 8TB Жесткий диск", True),
        ("rtx3070ti zotac gaming trio\nне помешает обслуживание \nцена 27к", True),
        ("Продается мышки, новые\nAJAZZ AJ159P MC\nЦена: 2100 сом", True),
    ],
)
def test_looks_like_offer(text, expected):
    assert looks_like_offer(text) is expected


def one_item(outcome):
    assert len(outcome.result.items) == 1, outcome.result.items
    return outcome.result.items[0]


def test_tg_gpu_with_k_price():
    out = classify_rules("1", text="rtx3070ti zotac gaming trio\nне помешает обслуживание \nцена 27к")
    item = one_item(out)
    assert out.confident
    assert out.result.intent == "sell"
    assert build_keys(item)[1] == "gpu:rtx-3070-ti"
    assert (item.price, item.currency) == (27000, "KGS")


def test_tg_gpu_bare_number_with_vram():
    out = classify_rules("1", text="4080 super 16gb zotac \n105500 сом \nКомплект сама карта и переходник")
    item = one_item(out)
    assert build_keys(item) == ("gpu:rtx-4080-super:16gb", "gpu:rtx-4080-super")
    assert item.price == 105500


def test_tg_new_cpu():
    item = one_item(classify_rules("1", text="Новый Ryzen 5 5600 box - 9900"))
    assert build_keys(item)[0] == "cpu:ryzen-5-5600"
    assert item.condition == "new"
    assert item.price == 9900


def test_tg_psu():
    item = one_item(
        classify_rules(
            "1",
            text="Продам блок питания \nCOUGAR CMX 850W BRONZE CERTIFICATE полу модульный все провода есть\nЦена: 3500",
        )
    )
    assert build_keys(item)[0] == "psu:850w:bronze"
    assert item.price == 3500


def test_tg_ram_kit():
    item = one_item(
        classify_rules(
            "1",
            text="Kingston Fury Beast DDR5 64 GB 2X32 5200 CL40 INTEL XMP -  57 000 сом\n\nесть 2 комплекта",
        )
    )
    assert build_keys(item)[0] == "ram:ddr5:64gb:5200"
    assert item.price == 57000


def test_tg_pc_build_single_price():
    text = (
        "12400F+3070 за 54900 сом🔥\n\n12400f\nColorful RTX 3070 LHR\nGigabyte H610\n"
        "HyperX 16GB\nNvme 512GB\nZalman 600-GT Bronze\nAG400 LED"
    )
    out = classify_rules("1", text=text)
    item = one_item(out)
    assert item.category == "pc"
    assert build_keys(item) == ("pc:i5-12400f+rtx-3070", "pc:rtx-3070")
    assert item.price == 54900
    assert not out.confident


def test_tg_bundle_is_not_confident():
    out = classify_rules("1", text="gigabyte h510m v2\ni5 11400\n\nцена 12к")
    assert not out.confident


def test_tg_buy_request():
    out = classify_rules("1", text="Куплю мобайл лежденс аккаунт за 2000-1500 в лс кидайте")
    assert out.result.intent == "buy"
    assert out.result.items == []


def test_tg_question_is_other():
    out = classify_rules("1", text="Какую мышь лучше взять от АТК?  Желательно в средне бюджетном сегменте. До 3к")
    assert out.result.intent == "other"
    assert out.result.items == []


def test_lalafo_cpu_with_hint():
    out = classify_rules(
        "1",
        title="Процессор для ПК, Новый, AMD Ryzen 5, 6 ядер",
        text="Новый, запечатанный Ryzen 5 5600.\n▫️AM4\n▫️6 ядер/12 потоков\n▫️4.4 ГГц",
        price=9900,
        currency="KGS",
        category_hint="cpu",
    )
    item = one_item(out)
    assert out.confident
    assert build_keys(item)[0] == "cpu:ryzen-5-5600"
    assert item.condition == "new"
    assert (item.price, item.currency) == (9900, "KGS")


def test_lalafo_used_gpu_title_condition():
    out = classify_rules(
        "1",
        title="Видеокарта, Б/у, NVidia, GeForce RTX, 8 ГБ, Для ПК",
        text="Продаю видеокарту rtx 3060ti в хорошем состоянии",
        price=21000,
        currency="KGS",
        category_hint="gpu",
    )
    item = one_item(out)
    assert build_keys(item)[1] == "gpu:rtx-3060-ti"
    assert item.vram_gb == 8
    assert item.condition == "used"


def test_lalafo_skupka_is_buy():
    out = classify_rules(
        "1",
        title="Скупка видеокарт быстро дорого с выездом",
        text="Скупаю видеокарты !!!\nОцениваем дороже других",
        price=None,
        category_hint="gpu",
    )
    assert out.result.intent == "buy"
    assert out.result.items == []


def test_lalafo_disk_category_detects_ssd():
    out = classify_rules(
        "1",
        title="SSD, Новый, Samsung, 1 ТБ, M.2",
        text="Samsung 980 Pro 1TB NVMe",
        price=8500,
        currency="KGS",
        category_hint=None,
    )
    item = one_item(out)
    assert build_keys(item)[0] == "ssd:nvme:1tb"


def test_tg_laptop_post():
    text = (
        "Acer Predator Helios 300\nПроцессор: Intel Core i7-9750H (6 ядер, до 4.5 ГГц)\n"
        "Видеокарта: NVIDIA GeForce GTX 1660 Ti (6 GB GDDR6)\nЭкран: 17.3\" IPS, 144 Гц (Full HD)\n"
        "Оперативная память: 16 GB DDR4  двумя плашками по 8 ГБ)\nНакопители: 256 GB SSD (M.2 NVMe) + 1 TB HDD\n\n"
        "привез с Кореи\n\n30 000 сомей"
    )
    item = one_item(classify_rules("1", text=text))
    assert item.category == "laptop"
    assert build_keys(item) == ("laptop:acer-predator-helios-300:i7-9750h+gtx-1660-ti", "laptop:i7-9750h+gtx-1660-ti")
    assert item.price == 30000


def test_lalafo_several_models_in_one_listing_is_not_confident():
    out = classify_rules(
        "1",
        title="Процессор для ПК, Б/у, AMD Ryzen 9, 12 ядер",
        text="Процессоры на am4 сокет, все райзены\nathlon 3000g 500c\nryzen 3 1200 2000c\nryzen 5 3600 5000c",
        price=500,
        category_hint="cpu",
    )
    assert not out.confident


def test_tg_laptop_with_many_numbers_stays_one_item():
    """Регрессия: ноутбук разбирался на «RTX 3060 за 512 сом» и т.п."""
    text = (
        "Продам Игровой ноутбук \n•ASUS TUF Gaming a15\n•Процессор-мощный Ryzen 7-6800H 8-Ядер/16 Потоков\n"
        "•Видеокарта-RTX 3060 6gb\n•ОЗУ-16гб DDR5\n•SSD-512\n•Экран 15.6 FullHD-IPS<144hz>\n•Батарея-90 Вт•ч\n"
        "CS2-300FPS(Высокие-Средние)\nPUBG-150-180FPS(Высокие-Средние)\n•Состояние Отличное\n"
        "Неделю назад производилась Чистка Ноутбука от пыли , замена термопасты\n60000"
    )
    out = classify_rules("1", text=text)
    item = one_item(out)
    assert item.category == "laptop"
    assert build_keys(item) == ("laptop:asus-tuf-gaming-a15:ryzen-7-6800h+rtx-3060", "laptop:ryzen-7-6800h+rtx-3060")
    assert item.price == 60000


def test_tg_pc_build_with_several_prices_stays_one_item():
    text = "Игровой ПК\nRyzen 5 5600\nRTX 3060 12gb\n16gb DDR4\nЦена 45000, торг до 43000"
    item = one_item(classify_rules("1", text=text))
    assert item.category == "pc" and item.price == 45000


def test_lalafo_laptop_posted_in_desktops_is_laptop():
    out = classify_rules(
        "1",
        title="Настольный ПК, Б/у",
        text="Asus tuf gaming\ni5-11400H\nram 16\nsdd 512\nrtx 3050\nсостояние идеальное.",
        price=48000,
        category_hint="pc",
    )
    item = one_item(out)
    assert item.category == "laptop"
    assert build_keys(item)[1] == "laptop:i5-11400h+rtx-3050"


@pytest.mark.parametrize(
    "text",
    [
        "не работает,продаю на запчасти.", "Монитор 19 дюймов, битая матрица", "Видеокарта RX 580 под ремонт, нет изображения",
        "нерабочий блок питания 600W", "Игровой монитор Xiaomi G27i 27\"\nСостояние б/у\nЭкран разбит",
    ],
)
def test_broken_condition(text):
    from lalascaner.ai.rules import detect_condition

    assert detect_condition(text) == "broken"


@pytest.mark.parametrize(
    "title,text,hint",
    [
        ("SSD, Новый", "UGREEN CM559, SSD M.2 NVMe/SATA, 10Gbps. Кейс для SSD. Новый в упаковке.", None),
        ("Монитор", "Крепление для монитора VESA NB H100 — настольный кронштейн до 32 дюймов", "monitor"),
        ("Блок питания", "ИБП SVC V-650F / Требуется замена АКБ.", "psu"),
        ("Видеокарта", "Райзер для видеокарты PCI-E x16, кабель 20 см", "gpu"),
        ("SSD", "UGREEN CM559 для SSD M.2 NVMe/SATA, 10Gbps.\nВ наличии", None),
        ("SSD", "SSD диски M.2 NVMe\n\nКейс для внешнего жёсткого диска MOKiN M.2 NVMe SATA", None),
    ],
)
def test_accessories_are_not_the_component(title, text, hint):
    out = classify_rules("1", title=title, text=text, price=1900, category_hint=hint)
    assert out.result.items == [] and not out.confident


def test_external_ssd_is_separate_from_internal():
    item = one_item(classify_rules("1", title="SSD", text="Внешний SSD-накопитель MAIWO 1 ТБ USB-C", price=10000, category_hint=None))
    assert build_keys(item)[0] == "ssd:usb:1tb"


def test_p106_mining_card_is_not_gtx_1060():
    item = one_item(classify_rules("1", title="Видеокарта", text="P106-100 6gb Gigabyte - аналог 1060 6gb, без видеовыходов", price=3000, category_hint="gpu"))
    assert build_keys(item)[0] == "gpu:p106-100:6gb"


def test_price_list_items_are_not_confident():
    out = classify_rules(
        "1", title="Процессоры", text="Процессоры 115х\ni5 4460 - 2500\ni3 4150 - 1200\ni3 3220 - 800", price=300, category_hint="cpu"
    )
    assert not out.confident


def test_lalafo_desktop_posted_in_laptops_is_pc():
    out = classify_rules(
        "1",
        title="Ноутбук, Б/у",
        text="Нужен надежный компьютер с хорошей производительностью?\nIntel Core i5-12400\nRTX 4060 8GB\n16GB DDR4",
        price=74990,
        category_hint="laptop",
    )
    item = one_item(out)
    assert item.category == "pc"
    assert build_keys(item)[0] == "pc:i5-12400+rtx-4060"


def test_known_sata_ssd_model():
    item = one_item(
        classify_rules("1", title="SSD, Б/у", text="SSD Crucial BX500  500 ГБ\nЗдоровье 78%", price=3700, category_hint=None)
    )
    assert build_keys(item)[0] == "ssd:sata:512gb"


def test_tg_buy_lines_are_not_sell_items():
    text = (
        "Меняю оперу 1 планка 16 гб DDR4 2666mhz на 2×8 такой же, с моей доплатой.\nИли продам 4500с.\n\n"
        "Куплю монитор 24-27 дюйма, Full HD, от 165гц, бюджет 5-6к."
    )
    out = classify_rules("1", text=text)
    assert all(item.category != "monitor" for item in out.result.items)


def test_cyrillic_sata():
    item = one_item(classify_rules("1", text="Ssd сата 256gb 86% здоровье\n1500с все отлично работает"))
    assert build_keys(item)[0] == "ssd:sata:256gb"


def test_lalafo_single_uses_line_as_model():
    out = classify_rules(
        "1",
        title="Материнская плата, Б/у",
        text="Продаю Материнскую Плату 1151 v2 r.1.02\nASROCK H310M-HDV\n\nНе работают USB порты. А в остальном все работает",
        price=500,
        category_hint="motherboard",
    )
    item = one_item(out)
    assert item.model == "ASROCK H310M-HDV"
    assert build_keys(item) == ("mb:asrock-h310m-hdv", "mb:h310")


def test_lalafo_gpu_comparison_mention_ignored():
    out = classify_rules(
        "1",
        title="Gigabyte aorus radeon rx 5700 xt 8g",
        text="gigabyte aorus radeon rx 5700 xt 8g \nОбслужена!\nпо производительности равна Nvidia GeForce RTX 3060 (12 ГБ)",
        price=15500,
        category_hint="gpu",
    )
    assert build_keys(one_item(out)) == ("gpu:rx-5700-xt:8gb", "gpu:rx-5700-xt")


def test_lalafo_without_model_is_not_confident():
    out = classify_rules("1", title="Видеокарта для ПК", text="Отличная видеокарта, пишите", price=5000, category_hint="gpu")
    assert not out.confident
