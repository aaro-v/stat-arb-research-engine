from __future__ import annotations

import numpy as np

from stat_arb_engine import estimate_ou, screen_pair
from stat_arb_engine.research import dynamic_hedge_ratio


def test_estimate_ou_returns_positive_half_life() -> None:
    spread = np.array([1.0, 0.8, 0.5, 0.4, 0.2, 0.1])
    params = estimate_ou(spread)
    assert params.theta > 0
    assert params.half_life > 0


def test_screen_pair_shapes_spread() -> None:
    rng = np.random.default_rng(7)
    x = np.arange(20, dtype=float)
    y = 2.0 * x + rng.normal(0.0, 0.2, size=x.shape)
    result = screen_pair(y, x)
    assert result.spread.shape == x.shape
    assert result.hedge_ratio > 1.9


def test_dynamic_hedge_ratio_tracks_series_length() -> None:
    x = np.linspace(1.0, 5.0, 10)
    y = 1.5 * x
    ratios = dynamic_hedge_ratio(y, x)
    assert ratios.shape == x.shape
