from __future__ import annotations

import numpy as np

from stat_arb_engine.risk import estimate_liquidity_capacity


def test_liquidity_capacity_identifies_adv_bottleneck() -> None:
    report = estimate_liquidity_capacity(
        weights=np.array([0.4, -0.2, 0.0]),
        average_daily_volume=np.array([1_000_000.0, 50_000.0, 10_000.0]),
        prices=np.array([20.0, 10.0, 30.0]),
        max_adv_participation=0.05,
    )

    assert report.max_strategy_capital == 125_000.0
    assert report.bottleneck_index == 1
    assert np.isinf(report.per_asset_capacity[2])


def test_liquidity_capacity_applies_short_borrow_limit() -> None:
    report = estimate_liquidity_capacity(
        weights=np.array([0.1, -0.5]),
        average_daily_volume=np.array([1_000_000.0, 1_000_000.0]),
        prices=np.array([20.0, 20.0]),
        borrow_availability=np.array([0.0, 250_000.0]),
    )

    assert report.max_strategy_capital == 500_000.0
    assert report.bottleneck_index == 1
