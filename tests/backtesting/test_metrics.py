from __future__ import annotations

import numpy as np
import pytest

from stat_arb_engine.backtesting import rolling_splits, summarize_pnl
from stat_arb_engine.execution import CostModel, estimate_round_trip_cost
from stat_arb_engine.signals import ThresholdSignal, classify_zscore


def test_summarize_pnl() -> None:
    summary = summarize_pnl(np.array([0.01, -0.005, 0.02]))
    assert summary.total_return == 0.025
    assert summary.trades >= 1
    assert summary.hit_rate == 2 / 3
    assert summary.profit_factor == 6.0


def test_summarize_pnl_with_position_path_diagnostics() -> None:
    summary = summarize_pnl(
        np.array([0.0, 0.02, -0.01, 0.03, -0.005, 0.0]),
        positions=np.array([0.0, 1.0, 1.0, 0.0, -1.0, 0.0]),
    )

    assert summary.trades == 4
    assert summary.turnover == 4.0
    assert summary.average_holding_period == 1.5
    assert summary.hit_rate == 0.5


def test_summarize_pnl_validates_position_shape() -> None:
    with pytest.raises(ValueError, match="same length"):
        summarize_pnl(np.array([0.01, -0.01]), positions=np.array([1.0]))


def test_round_trip_cost_includes_borrow() -> None:
    costs = estimate_round_trip_cost(
        notional=np.array([100_000.0]),
        holding_days=np.array([10.0]),
        model=CostModel(),
    )
    assert costs[0] > 0


def test_threshold_classification() -> None:
    assert classify_zscore(2.5) == ThresholdSignal.ENTER_SHORT_SPREAD
    assert classify_zscore(0.1) == ThresholdSignal.EXIT


def test_walk_forward_splits() -> None:
    splits = rolling_splits(length=100, train_size=40, test_size=10)
    assert splits[0].train_start == 0
    assert splits[-1].test_end <= 100
