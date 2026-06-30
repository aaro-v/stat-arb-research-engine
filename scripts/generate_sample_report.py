from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from stat_arb_engine.backtesting import (
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
from stat_arb_engine.execution import CostModel, estimate_position_costs
from stat_arb_engine.reporting import write_artifact_manifest
from stat_arb_engine.reporting.charts import plot_mean_reversion_diagnostic
from stat_arb_engine.signals import ThresholdPolicy, ThresholdSignal, classify_zscore


def build_sample_diagnostic(days: int = 320, seed: int = 17) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    shocks = rng.normal(0.0, 0.8, size=days)
    spread = np.zeros(days)
    for index in range(1, days):
        spread[index] = 0.94 * spread[index - 1] + shocks[index]

    mean = pd.Series(spread).rolling(63, min_periods=20).mean()
    vol = pd.Series(spread).rolling(63, min_periods=20).std(ddof=0).replace(0.0, np.nan)
    zscore = ((pd.Series(spread) - mean) / vol).fillna(0.0)

    policy = ThresholdPolicy(entry_z=1.6, exit_z=0.35)
    positions: list[float] = []
    position = 0.0
    for value in zscore:
        signal = classify_zscore(float(value), policy)
        if signal == ThresholdSignal.ENTER_LONG_SPREAD:
            position = 1.0
        elif signal == ThresholdSignal.ENTER_SHORT_SPREAD:
            position = -1.0
        elif signal == ThresholdSignal.EXIT:
            position = 0.0
        positions.append(position)

    position_series = pd.Series(positions)
    spread_change = pd.Series(spread).diff().fillna(0.0)
    trading_costs = estimate_position_costs(
        position_series.to_numpy(),
        CostModel(commission_bps=10.0, slippage_bps=10.0, borrow_bps_per_year=0.0),
    )
    pnl = -position_series.shift(1).fillna(0.0) * spread_change - trading_costs
    return pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-02", periods=days),
            "spread": spread,
            "zscore": zscore,
            "position": positions,
            "pnl": pnl,
        }
    )


def main() -> None:
    output_dir = Path("reports/generated")
    frame = build_sample_diagnostic()
    summary = summarize_pnl(frame["pnl"].to_numpy(), positions=frame["position"].to_numpy())
    splits = rolling_splits(length=len(frame), train_size=126, test_size=42, step=42)
    folds = summarize_walk_forward_pnl(
        frame["pnl"].to_numpy(),
        splits,
        positions=frame["position"].to_numpy(),
    )
    walk_forward = aggregate_walk_forward_diagnostics(folds)
    rolling_windows = rolling_window_summaries(
        frame["pnl"].to_numpy(),
        window_size=63,
        step=21,
        positions=frame["position"].to_numpy(),
    )
    decay = aggregate_decay_diagnostics(rolling_windows, comparison_windows=2)
    stress = block_bootstrap_stress(
        frame["pnl"].to_numpy(),
        simulations=1_000,
        horizon=126,
        block_size=10,
        seed=23,
    )
    drawdowns = drawdown_episodes(frame["pnl"].to_numpy())
    volatility_regimes = classify_volatility_regime(frame["spread"].diff().fillna(0.0).to_numpy())
    regime_summaries = summarize_pnl_by_regime(frame["pnl"].to_numpy(), volatility_regimes)
    summary_frame = pd.DataFrame(
        [
            {
                "dataset": "deterministic_synthetic_engineering_validation",
                "total_pnl": summary.total_return,
                "sharpe": summary.sharpe,
                "max_drawdown": summary.max_drawdown,
                "max_drawdown_duration": summary.max_drawdown_duration,
                "trades": summary.trades,
                "hit_rate": summary.hit_rate,
                "profit_factor": summary.profit_factor,
                "turnover": summary.turnover,
                "average_holding_period": summary.average_holding_period,
                "active_fraction": summary.active_fraction,
                "long_fraction": summary.long_fraction,
                "short_fraction": summary.short_fraction,
                "flat_fraction": summary.flat_fraction,
                "average_abs_position": summary.average_abs_position,
                "max_abs_position": summary.max_abs_position,
                "gross_exposure": summary.gross_exposure,
                "return_per_gross_exposure": summary.return_per_gross_exposure,
                "annualized_volatility": summary.annualized_volatility,
                "downside_deviation": summary.downside_deviation,
                "sortino": summary.sortino,
                "tail_loss_95": summary.tail_loss_95,
                "expected_shortfall_95": summary.expected_shortfall_95,
                "time_under_water": summary.time_under_water,
                "underwater_fraction": summary.underwater_fraction,
                "average_drawdown": summary.average_drawdown,
                "ulcer_index": summary.ulcer_index,
                "drawdown_recovery_ratio": summary.drawdown_recovery_ratio,
                "walk_forward_folds": walk_forward.folds,
                "walk_forward_mean_sharpe": walk_forward.mean_sharpe,
                "walk_forward_median_sharpe": walk_forward.median_sharpe,
                "walk_forward_sharpe_std": walk_forward.sharpe_std,
                "walk_forward_total_return": walk_forward.total_return,
                "walk_forward_mean_return": walk_forward.mean_return,
                "walk_forward_return_std": walk_forward.return_std,
                "walk_forward_return_standard_error": walk_forward.return_standard_error,
                "walk_forward_return_t_stat": walk_forward.return_t_stat,
                "walk_forward_return_ci95_lower": walk_forward.return_ci95_lower,
                "walk_forward_return_ci95_upper": walk_forward.return_ci95_upper,
                "walk_forward_return_consistency": walk_forward.return_consistency,
                "walk_forward_positive_fold_rate": walk_forward.positive_fold_rate,
                "walk_forward_positive_return_concentration": (
                    walk_forward.positive_return_concentration
                ),
                "walk_forward_worst_drawdown": walk_forward.worst_drawdown,
                "walk_forward_worst_fold": walk_forward.worst_fold,
                "walk_forward_worst_fold_return": walk_forward.worst_fold_return,
                "walk_forward_best_fold": walk_forward.best_fold,
                "walk_forward_best_fold_return": walk_forward.best_fold_return,
                "decay_windows": decay.windows,
                "decay_window_size": decay.window_size,
                "decay_recent_window_return": decay.recent_window_return,
                "decay_early_mean_return": decay.early_mean_return,
                "decay_recent_mean_return": decay.recent_mean_return,
                "decay_return_decay": decay.return_decay,
                "decay_early_mean_sharpe": decay.early_mean_sharpe,
                "decay_recent_mean_sharpe": decay.recent_mean_sharpe,
                "decay_sharpe_decay": decay.sharpe_decay,
                "decay_negative_window_rate": decay.negative_window_rate,
                "decay_worst_window": decay.worst_window,
                "decay_worst_window_return": decay.worst_window_return,
                "decay_best_window": decay.best_window,
                "decay_best_window_return": decay.best_window_return,
            }
        ]
    )
    fold_frame = pd.DataFrame(
        [
            {
                "dataset": "deterministic_synthetic_engineering_validation",
                "fold": fold.fold,
                "train_start": fold.split.train_start,
                "train_end": fold.split.train_end,
                "test_start": fold.split.test_start,
                "test_end": fold.split.test_end,
                "total_pnl": fold.summary.total_return,
                "sharpe": fold.summary.sharpe,
                "max_drawdown": fold.summary.max_drawdown,
                "max_drawdown_duration": fold.summary.max_drawdown_duration,
                "trades": fold.summary.trades,
                "turnover": fold.summary.turnover,
                "active_fraction": fold.summary.active_fraction,
                "long_fraction": fold.summary.long_fraction,
                "short_fraction": fold.summary.short_fraction,
                "flat_fraction": fold.summary.flat_fraction,
                "average_abs_position": fold.summary.average_abs_position,
                "max_abs_position": fold.summary.max_abs_position,
                "gross_exposure": fold.summary.gross_exposure,
                "return_per_gross_exposure": fold.summary.return_per_gross_exposure,
                "hit_rate": fold.summary.hit_rate,
                "profit_factor": fold.summary.profit_factor,
                "time_under_water": fold.summary.time_under_water,
                "underwater_fraction": fold.summary.underwater_fraction,
                "average_drawdown": fold.summary.average_drawdown,
                "ulcer_index": fold.summary.ulcer_index,
                "drawdown_recovery_ratio": fold.summary.drawdown_recovery_ratio,
            }
            for fold in folds
        ]
    )
    rolling_frame = pd.DataFrame(
        [
            {
                "dataset": "deterministic_synthetic_engineering_validation",
                "window": window.window,
                "start": window.start,
                "end": window.end,
                "total_pnl": window.summary.total_return,
                "sharpe": window.summary.sharpe,
                "max_drawdown": window.summary.max_drawdown,
                "max_drawdown_duration": window.summary.max_drawdown_duration,
                "trades": window.summary.trades,
                "turnover": window.summary.turnover,
                "active_fraction": window.summary.active_fraction,
                "long_fraction": window.summary.long_fraction,
                "short_fraction": window.summary.short_fraction,
                "flat_fraction": window.summary.flat_fraction,
                "average_abs_position": window.summary.average_abs_position,
                "max_abs_position": window.summary.max_abs_position,
                "gross_exposure": window.summary.gross_exposure,
                "return_per_gross_exposure": window.summary.return_per_gross_exposure,
                "hit_rate": window.summary.hit_rate,
                "profit_factor": window.summary.profit_factor,
                "time_under_water": window.summary.time_under_water,
                "underwater_fraction": window.summary.underwater_fraction,
                "average_drawdown": window.summary.average_drawdown,
                "ulcer_index": window.summary.ulcer_index,
                "drawdown_recovery_ratio": window.summary.drawdown_recovery_ratio,
            }
            for window in rolling_windows
        ]
    )
    stress_frame = pd.DataFrame(
        [
            {
                "dataset": "deterministic_synthetic_engineering_validation",
                "simulations": stress.simulations,
                "horizon": stress.horizon,
                "block_size": stress.block_size,
                "mean_total_return": stress.mean_total_return,
                "median_total_return": stress.median_total_return,
                "total_return_p05": stress.total_return_p05,
                "total_return_p95": stress.total_return_p95,
                "loss_probability": stress.loss_probability,
                "max_drawdown_p05": stress.max_drawdown_p05,
                "expected_shortfall_95": stress.expected_shortfall_95,
            }
        ]
    )
    drawdown_frame = pd.DataFrame(
        [
            {
                "dataset": "deterministic_synthetic_engineering_validation",
                "episode": index,
                "start": episode.start,
                "trough": episode.trough,
                "end": episode.end,
                "depth": episode.depth,
                "duration": episode.duration,
                "recovery_duration": episode.recovery_duration,
            }
            for index, episode in enumerate(drawdowns)
        ]
    ).astype({"end": "Int64", "recovery_duration": "Int64"})
    regime_frame = pd.DataFrame(
        [
            {
                "dataset": "deterministic_synthetic_engineering_validation",
                "regime": regime.regime,
                "observations": regime.observations,
                "observation_fraction": regime.observation_fraction,
                "return_share": regime.return_share,
                "total_pnl": regime.summary.total_return,
                "mean_pnl": regime.summary.total_return / regime.observations,
                "sharpe": regime.summary.sharpe,
                "max_drawdown": regime.summary.max_drawdown,
                "hit_rate": regime.summary.hit_rate,
                "tail_loss_95": regime.summary.tail_loss_95,
                "expected_shortfall_95": regime.summary.expected_shortfall_95,
            }
            for regime in regime_summaries
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_path = output_dir / "mean_reversion_diagnostic.csv"
    summary_path = output_dir / "mean_reversion_summary.csv"
    walk_forward_path = output_dir / "mean_reversion_walk_forward.csv"
    decay_path = output_dir / "mean_reversion_decay.csv"
    stress_path = output_dir / "mean_reversion_stress.csv"
    drawdown_path = output_dir / "mean_reversion_drawdowns.csv"
    regime_path = output_dir / "mean_reversion_regimes.csv"
    chart_path = output_dir / "mean_reversion_diagnostic.png"
    frame.to_csv(diagnostic_path, index=False)
    summary_frame.to_csv(summary_path, index=False)
    fold_frame.to_csv(walk_forward_path, index=False)
    rolling_frame.to_csv(decay_path, index=False)
    stress_frame.to_csv(stress_path, index=False)
    drawdown_frame.to_csv(drawdown_path, index=False)
    regime_frame.to_csv(regime_path, index=False)
    plot_mean_reversion_diagnostic(
        frame,
        chart_path,
        entry_z=1.6,
        exit_z=0.35,
    )
    write_artifact_manifest(
        [
            decay_path,
            diagnostic_path,
            chart_path,
            drawdown_path,
            regime_path,
            stress_path,
            summary_path,
            walk_forward_path,
        ],
        output_dir / "manifest.csv",
        root=Path.cwd(),
    )
    print(f"Wrote sample report artifacts to {output_dir}")


def classify_volatility_regime(spread_changes: np.ndarray) -> np.ndarray:
    absolute_changes = np.abs(np.asarray(spread_changes, dtype=float))
    median_change = float(np.median(absolute_changes))
    return np.where(absolute_changes <= median_change, "low_volatility", "high_volatility")


if __name__ == "__main__":
    main()
