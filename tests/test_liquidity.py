from datetime import datetime, timedelta

import pytest

from lalascaner.market.liquidity import (
    MIN_CATEGORY_OBSERVATIONS,
    MIN_OBSERVATIONS,
    Observation,
    bucket_of,
    build_liquidity,
    for_category,
    kaplan_meier,
    summarize,
)

NOW = datetime(2026, 10, 1)


def test_kaplan_meier_counts_still_listed_as_at_risk_not_as_unsold():
    # 4 объявления: продано на 1-й и 3-й день, двое ещё висят (2 и 5 дней)
    obs = [Observation(1, True), Observation(2, False), Observation(3, True), Observation(5, False)]
    curve = kaplan_meier(obs)
    assert curve == [(1, 0.75), (3, pytest.approx(0.375))]  # на 3-й день в риске 2, продан 1 → 0.75 * 1/2


def test_summary_is_silent_beyond_longest_observation():
    summary = summarize([Observation(1, True), Observation(4, False)])
    assert summary["p3"] == 0.5 and summary["p7"] is None and summary["p14"] is None
    assert summary["median_days"] == 1


def test_buckets():
    assert bucket_of(0.5) is None and bucket_of(0.8) == "cheap" and bucket_of(1.0) == "market"
    assert bucket_of(1.1) == "high" and bucket_of(2.0) == "very_high" and bucket_of(5) is None


def rows(start_id, n, ratio, sold_after_days, category="gpu"):
    out = []
    for i in range(n):
        posted = NOW - timedelta(days=10)
        gone = posted + timedelta(days=sold_after_days) if sold_after_days is not None else None
        out.append((start_id + i, category, 1000 * ratio, 1000, posted, gone))
    return out


def test_cheap_sells_faster_and_reposts_are_not_sales():
    # дешёвые: 24 пропали на 1-й день (6 из них — перезалиты), ещё 6 висят все 10 дней
    data = rows(0, 24, 0.85, 1) + rows(200, 6, 0.85, None) + rows(100, MIN_CATEGORY_OBSERVATIONS, 1.2, None)
    reposted = set(range(6))
    result = build_liquidity(data, reposted, NOW)
    cheap, high = result["gpu"]["cheap"], result["gpu"]["high"]
    # 18 продаж из 30 в риске; если бы перезаливы считались продажами, было бы 24/30
    assert cheap["sold"] == 18 and cheap["p7"] == pytest.approx(0.6)
    assert high["sold"] == 0 and high["p7"] == 0
    assert result["all"]["cheap"]["n"] == 30


def test_category_needs_more_observations_than_whole_market():
    result = build_liquidity(rows(0, MIN_OBSERVATIONS, 1.0, 2), set(), NOW)
    assert "market" in result["all"] and "gpu" not in result


def test_multi_item_listings_and_small_groups_are_skipped():
    data = rows(0, MIN_OBSERVATIONS - 1, 1.0, 2)
    data += [(500, "gpu", 1000, 1000, NOW - timedelta(days=3), None), (500, "cpu", 1000, 1000, NOW - timedelta(days=3), None)]
    result = build_liquidity(data, set(), NOW)
    assert "gpu" not in result and "all" not in result


def test_for_category_falls_back_to_whole_market():
    liquidity = {"all": {"market": {"n": 50}}, "gpu": {"cheap": {"n": 20}}}
    assert for_category(liquidity, "gpu", "cheap") == ({"n": 20}, False)
    assert for_category(liquidity, "gpu", "market") == ({"n": 50}, True)
    assert for_category(liquidity, "cpu", "high") == (None, False)
