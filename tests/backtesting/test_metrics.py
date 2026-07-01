from __future__ import annotations

import numpy as np
import pytest

from stat_arb_engine.backtesting import (
    WalkForwardSplit,
    aggregate_decay_diagnostics,
    aggregate_walk_forward_diagnostics,
    block_bootstrap_stress,
    drawdown_episodes,
    rolling_window_summaries,
    rolling_splits,
    summarize_pnl,
    summarize_pnl_by_regime,
    summarize_walk_forward_pnl,
)
from stat_arb_engine.signals import ThresholdSignal, classify_zscore


def test_summarize_pnl() -> None:
    summary = summarize_pnl(np.array([0.01, -0.005, 0.02]))
    assert summary.total_return == 0.025
    assert summary.max_drawdown_duration == 1
    assert summary.time_under_water == 1
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

    assert summary.max_drawdown_duration == 0
    assert summary.time_under_water == 0
    assert summary.profit_factor == float("inf")
    assert summary.downside_deviation == 0.0
    assert summary.sortino == 0.0
    assert summary.tail_loss_95 == 0.0
    assert summary.expected_shortfall_95 == 0.0


def test_summarize_pnl_by_regime_preserves_label_order() -> None:
    summaries = summarize_pnl_by_regime(
        np.array([0.01, -0.02, 0.03, -0.01]),
        np.array(["calm", "stress", "calm", "stress"]),
    )

    assert [summary.regime for summary in summaries] == ["calm", "stress"]
    assert summaries[0].observations == 2
    assert summaries[0].observation_fraction == pytest.approx(0.5)
    assert summaries[0].summary.total_return == pytest.approx(0.04)
    assert summaries[0].return_share == pytest.approx(0.04 / 0.07)
    assert summaries[1].summary.total_return == pytest.approx(-0.03)
    assert summaries[1].return_share == pytest.approx(0.03 / 0.07)


def test_summarize_pnl_by_regime_reports_position_concentration() -> None:
    summaries = summarize_pnl_by_regime(
        np.array([0.01, -0.02, 0.03, -0.01]),
        np.array(["calm", "stress", "calm", "stress"]),
        positions=np.array([0.0, 1.0, -2.0, 0.0]),
    )

    assert summaries[0].gross_exposure == pytest.approx(2.0)
    assert summaries[0].exposure_share == pytest.approx(2 / 3)
    assert summaries[0].active_fraction == pytest.approx(0.5)
    assert summaries[0].long_fraction == pytest.approx(0.0)
    assert summaries[0].short_fraction == pytest.approx(0.5)
    assert summaries[0].flat_fraction == pytest.approx(0.5)
    assert summaries[1].gross_exposure == pytest.approx(1.0)
    assert summaries[1].exposure_share == pytest.approx(1 / 3)
    assert summaries[1].active_fraction == pytest.approx(0.5)
    assert summaries[1].long_fraction == pytest.approx(0.5)
    assert summaries[1].short_fraction == pytest.approx(0.0)
    assert summaries[1].flat_fraction == pytest.approx(0.5)


def test_summarize_pnl_by_regime_validates_shape() -> None:
    with pytest.raises(ValueError, match="same length"):
        summarize_pnl_by_regime(np.array([0.01, -0.01]), np.array(["calm"]))

    with pytest.raises(ValueError, match="same length"):
        summarize_pnl_by_regime(
            np.array([0.01, -0.01]),
            np.array(["calm", "stress"]),
            positions=np.array([1.0]),
        )


def test_summarize_pnl_with_position_path_diagnostics() -> None:
    summary = summarize_pnl(
        np.array([0.0, 0.02, -0.01, 0.03, -0.005, 0.0]),
        positions=np.array([0.0, 1.0, 1.0, 0.0, -1.0, 0.0]),
    )

    assert summary.trades == 4
    assert summary.turnover == 4.0
    assert summary.average_holding_period == 1.5
    assert summary.active_fraction == pytest.approx(3 / 6)
    assert summary.long_fraction == pytest.approx(2 / 6)
    assert summary.short_fraction == pytest.approx(1 / 6)
    assert summary.flat_fraction == pytest.approx(3 / 6)
    assert summary.average_abs_position == pytest.approx(0.5)
    assert summary.max_abs_position == pytest.approx(1.0)
    assert summary.gross_exposure == pytest.approx(3.0)
    assert summary.return_per_gross_exposure == pytest.approx(0.035 / 3.0)
    assert summary.hit_rate == 0.5


def test_summarize_pnl_respects_initial_position_for_turnover() -> None:
    summary = summarize_pnl(
        np.array([0.01, -0.005, 0.0]),
        positions=np.array([1.0, 1.0, 0.0]),
        initial_position=1.0,
    )

    assert summary.trades == 1
    assert summary.turnover == 1.0
    assert summary.gross_exposure == pytest.approx(2.0)
    assert summary.return_per_gross_exposure == pytest.approx(0.005 / 2.0)


def test_summarize_pnl_validates_initial_position() -> None:
    with pytest.raises(ValueError, match="initial_position"):
        summarize_pnl(np.array([0.01]), initial_position=float("nan"))


def test_summarize_pnl_reports_drawdown_duration() -> None:
    summary = summarize_pnl(np.array([0.04, -0.01, -0.01, 0.005, 0.02, -0.005]))

    assert summary.max_drawdown == pytest.approx(-0.02)
    assert summary.max_drawdown_duration == 3
    assert summary.time_under_water == 4
    assert summary.underwater_fraction == pytest.approx(4 / 6)
    assert summary.average_drawdown == pytest.approx(-0.0125)
    assert summary.ulcer_index == pytest.approx(np.sqrt(0.00075 / 6))
    assert summary.drawdown_recovery_ratio == pytest.approx(2.0)


def test_drawdown_episodes_report_recovery_paths() -> None:
    episodes = drawdown_episodes(np.array([0.04, -0.01, -0.02, 0.01, 0.03, -0.02, 0.01]))

    assert episodes[0].start == 1
    assert episodes[0].trough == 2
    assert episodes[0].end == 4
    assert episodes[0].depth == pytest.approx(-0.03)
    assert episodes[0].duration == 3
    assert episodes[0].recovery_duration == 4
    assert episodes[1].start == 5
    assert episodes[1].trough == 5
    assert episodes[1].end is None
    assert episodes[1].depth == pytest.approx(-0.02)
    assert episodes[1].duration == 2
    assert episodes[1].recovery_duration is None


def test_drawdown_episodes_validate_inputs() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        drawdown_episodes(np.array([]))

    with pytest.raises(ValueError, match="one-dimensional"):
        drawdown_episodes(np.array([[0.01, -0.01]]))


def test_summarize_pnl_validates_position_shape() -> None:
    with pytest.raises(ValueError, match="same length"):
        summarize_pnl(np.array([0.01, -0.01]), positions=np.array([1.0]))


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
    assert folds[0].summary.turnover == 1.0
    assert folds[0].summary.gross_exposure == pytest.approx(1.0)
    assert folds[0].summary.return_per_gross_exposure == pytest.approx(0.05)
    assert folds[1].summary.short_fraction == 0.5
    assert folds[1].summary.flat_fraction == 0.5
    assert folds[1].summary.gross_exposure == pytest.approx(1.0)
    assert diagnostics.folds == 2
    assert diagnostics.total_return == pytest.approx(0.08)
    assert diagnostics.mean_return == pytest.approx(0.04)
    assert diagnostics.return_std > 0.0
    assert diagnostics.return_standard_error == pytest.approx(0.01)
    assert diagnostics.return_t_stat == pytest.approx(4.0)
    assert diagnostics.return_ci95_lower == pytest.approx(0.0204)
    assert diagnostics.return_ci95_upper == pytest.approx(0.0596)
    assert diagnostics.return_consistency > 0.0
    assert diagnostics.sharpe_std > 0.0
    assert diagnostics.positive_fold_rate == 1.0
    assert diagnostics.positive_return_concentration == pytest.approx(0.625)
    assert diagnostics.worst_fold == 1
    assert diagnostics.worst_fold_return == pytest.approx(0.03)
    assert diagnostics.best_fold == 0
    assert diagnostics.best_fold_return == pytest.approx(0.05)


def test_walk_forward_diagnostics_reports_single_fold_consistency() -> None:
    folds = summarize_walk_forward_pnl(
        np.array([0.0, 0.0, 0.02, 0.01]),
        [WalkForwardSplit(train_start=0, train_end=2, test_start=2, test_end=4)],
    )

    diagnostics = aggregate_walk_forward_diagnostics(folds)

    assert diagnostics.return_std == 0.0
    assert diagnostics.return_standard_error == 0.0
    assert diagnostics.return_t_stat == float("inf")
    assert diagnostics.return_ci95_lower == pytest.approx(0.03)
    assert diagnostics.return_ci95_upper == pytest.approx(0.03)
    assert diagnostics.return_consistency == float("inf")
    assert diagnostics.positive_fold_rate == 1.0
    assert diagnostics.positive_return_concentration == 1.0
    assert diagnostics.worst_fold_return == pytest.approx(0.03)
    assert diagnostics.best_fold_return == pytest.approx(0.03)


def test_summarize_walk_forward_pnl_carries_position_across_test_boundary() -> None:
    pnl = np.array([0.0, 0.0, 0.01, -0.005, 0.0])
    positions = np.array([0.0, 1.0, 1.0, 1.0, 0.0])
    splits = [WalkForwardSplit(train_start=0, train_end=2, test_start=2, test_end=5)]

    [fold] = summarize_walk_forward_pnl(pnl, splits, positions=positions)

    assert fold.summary.trades == 1
    assert fold.summary.turnover == 1.0


def test_summarize_walk_forward_pnl_validates_split_bounds() -> None:
    with pytest.raises(ValueError, match="invalid walk-forward split"):
        summarize_walk_forward_pnl(
            np.array([0.01, -0.01]),
            [WalkForwardSplit(train_start=0, train_end=1, test_start=1, test_end=3)],
        )


def test_rolling_window_summaries_measure_decay() -> None:
    pnl = np.array([0.03, 0.02, 0.01, 0.01, -0.02, -0.01, -0.03, -0.02])
    positions = np.array([0.0, 1.0, 1.0, 0.0, -1.0, -1.0, 0.0, 0.0])

    windows = rolling_window_summaries(pnl, window_size=4, step=2, positions=positions)
    diagnostics = aggregate_decay_diagnostics(windows, comparison_windows=1)

    assert [window.start for window in windows] == [0, 2, 4]
    assert [window.end for window in windows] == [4, 6, 8]
    assert windows[0].summary.total_return == pytest.approx(0.07)
    assert windows[1].summary.turnover == 2.0
    assert windows[1].summary.gross_exposure == pytest.approx(3.0)
    assert windows[1].summary.return_per_gross_exposure == pytest.approx(-0.01 / 3.0)
    assert diagnostics.windows == 3
    assert diagnostics.window_size == 4
    assert diagnostics.recent_window_return == pytest.approx(-0.08)
    assert diagnostics.return_decay == pytest.approx(-0.15)
    assert diagnostics.sharpe_decay < 0.0
    assert diagnostics.negative_window_rate == pytest.approx(2 / 3)
    assert diagnostics.worst_window == 2
    assert diagnostics.best_window == 0


def test_rolling_window_summaries_validate_inputs() -> None:
    with pytest.raises(ValueError, match="window_size"):
        rolling_window_summaries(np.array([0.01]), window_size=0)
    with pytest.raises(ValueError, match="no larger"):
        rolling_window_summaries(np.array([0.01]), window_size=2)
    with pytest.raises(ValueError, match="step"):
        rolling_window_summaries(np.array([0.01, 0.02]), window_size=1, step=0)
    with pytest.raises(ValueError, match="same length"):
        rolling_window_summaries(
            np.array([0.01, 0.02]),
            window_size=1,
            positions=np.array([1.0]),
        )


def test_decay_diagnostics_validate_inputs() -> None:
    with pytest.raises(ValueError, match="windows"):
        aggregate_decay_diagnostics([])

    windows = rolling_window_summaries(np.array([0.01, 0.02]), window_size=1)
    with pytest.raises(ValueError, match="comparison_windows"):
        aggregate_decay_diagnostics(windows, comparison_windows=0)


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
