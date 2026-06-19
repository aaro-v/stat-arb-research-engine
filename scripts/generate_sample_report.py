from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from stat_arb_engine.backtesting import (
    aggregate_walk_forward_diagnostics,
    rolling_splits,
    summarize_pnl,
    summarize_walk_forward_pnl,
)
from stat_arb_engine.execution import CostModel, estimate_position_costs
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
                "annualized_volatility": summary.annualized_volatility,
                "downside_deviation": summary.downside_deviation,
                "sortino": summary.sortino,
                "tail_loss_95": summary.tail_loss_95,
                "expected_shortfall_95": summary.expected_shortfall_95,
                "time_under_water": summary.time_under_water,
                "walk_forward_folds": walk_forward.folds,
                "walk_forward_mean_sharpe": walk_forward.mean_sharpe,
                "walk_forward_median_sharpe": walk_forward.median_sharpe,
                "walk_forward_sharpe_std": walk_forward.sharpe_std,
                "walk_forward_total_return": walk_forward.total_return,
                "walk_forward_mean_return": walk_forward.mean_return,
                "walk_forward_return_std": walk_forward.return_std,
                "walk_forward_positive_fold_rate": walk_forward.positive_fold_rate,
                "walk_forward_worst_drawdown": walk_forward.worst_drawdown,
                "walk_forward_worst_fold": walk_forward.worst_fold,
                "walk_forward_best_fold": walk_forward.best_fold,
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
                "hit_rate": fold.summary.hit_rate,
                "profit_factor": fold.summary.profit_factor,
                "time_under_water": fold.summary.time_under_water,
            }
            for fold in folds
        ]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "mean_reversion_diagnostic.csv", index=False)
    summary_frame.to_csv(output_dir / "mean_reversion_summary.csv", index=False)
    fold_frame.to_csv(output_dir / "mean_reversion_walk_forward.csv", index=False)
    plot_mean_reversion_diagnostic(
        frame,
        output_dir / "mean_reversion_diagnostic.png",
        entry_z=1.6,
        exit_z=0.35,
    )
    print(f"Wrote sample report artifacts to {output_dir}")


if __name__ == "__main__":
    main()
