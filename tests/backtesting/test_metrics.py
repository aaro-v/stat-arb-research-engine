from __future__ import annotations

import numpy as np
import pytest

from stat_arb_engine.backtesting import (
    WalkForwardSplit,
    aggregate_walk_forward_diagnostics,
    block_bootstrap_stress,
    rolling_splits,
    summarize_pnl,
    summarize_walk_forward_pnl,
)
from stat_arb_engine.execution import CostModel, estimate_round_trip_cost
from stat_arb_engine.signals import ThresholdSignal, classify_zscore


def test_summarize_pnl() -> None:
    summary = summarize_pnl(np.array([0.01, -0.005, 0.02]))
    assert summary.total_return == 0.025
    assert summary.trades >= 1
    assert summary.hit_rate == 2 / 3
    assert summary.profit_factor == 6.0
    assert summary.annualized_volatility > 0.0
    assert summary.downside_deviation > 0.0
    assert summary.sortino > 0.0
    assert summary.tail_loss_95 > 0.0
    assert summary.expected_shortfall_95 > 0.0


def test_summarize_pnl_reports_zero_tail_risk_without_losses() -> None:
    summary = summarize_pnl(np.array([0.0, 0.01, 0.02]))

    assert summary.profit_factor == float("inf")
    assert summary.downside_deviation == 0.0
    assert summary.sortino == 0.0
    assert summary.tail_loss_95 == 0.0
    assert summary.expected_shortfall_95 == 0.0


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


def test_walk_forward_splits_validate_step() -> None:
    with pytest.raises(ValueError, match="step"):
        rolling_splits(length=100, train_size=40, test_size=10, step=0)


def test_summarize_walk_forward_pnl_uses_out_of_sample_windows() -> None:
    pnl = np.array([0.50, 0.40, 0.03, 0.02, 0.30, 0.25, -0.01, 0.04])
    positions = np.array([0.0, 1.0, 1.0, 0.0, 0.0, -1.0, -1.0, 0.0])
    splits = [
        WalkForwardSplit(train_start=0, train_end=2, test_start=2, test_end=4),
        WalkForwardSplit(train_start=4, train_end=6, test_start=6, test_end=8),
    ]

    folds = summarize_walk_forward_pnl(pnl, splits, positions=positions)
    diagnostics = aggregate_walk_forward_diagnostics(folds)

    assert [fold.fold for fold in folds] == [0, 1]
    assert [fold.summary.total_return for fold in folds] == [0.05, 0.03]
    assert folds[0].summary.turnover == 2.0
    assert diagnostics.folds == 2
    assert diagnostics.total_return == pytest.approx(0.08)
    assert diagnostics.positive_fold_rate == 1.0


def test_summarize_walk_forward_pnl_validates_split_bounds() -> None:
    with pytest.raises(ValueError, match="invalid walk-forward split"):
        summarize_walk_forward_pnl(
            np.array([0.01, -0.01]),
            [WalkForwardSplit(train_start=0, train_end=1, test_start=1, test_end=3)],
        )


def test_block_bootstrap_stress_is_deterministic_with_seed() -> None:
    pnl = np.array([0.01, -0.02, 0.015, 0.005, -0.01, 0.02])

    first = block_bootstrap_stress(pnl, simulations=200, horizon=12, block_size=3, seed=7)
    second = block_bootstrap_stress(pnl, simulations=200, horizon=12, block_size=3, seed=7)

    assert first == second
    assert first.simulations == 200
    assert first.horizon == 12
    assert first.block_size == 3
    assert 0.0 <= first.loss_probability <= 1.0
    assert first.total_return_p05 <= first.median_total_return <= first.total_return_p95
    assert first.max_drawdown_p05 <= 0.0


def test_block_bootstrap_stress_validates_inputs() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        block_bootstrap_stress(np.array([]))
    with pytest.raises(ValueError, match="simulations"):
        block_bootstrap_stress(np.array([0.01]), simulations=0)
    with pytest.raises(ValueError, match="horizon"):
        block_bootstrap_stress(np.array([0.01]), horizon=0)
    with pytest.raises(ValueError, match="block_size"):
        block_bootstrap_stress(np.array([0.01]), block_size=0)
