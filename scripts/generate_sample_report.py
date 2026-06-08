from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

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

    spread_change = pd.Series(spread).diff().fillna(0.0)
    pnl = -pd.Series(positions).shift(1).fillna(0.0) * spread_change - 0.002 * pd.Series(
        positions
    ).diff().abs().fillna(0.0)
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
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "mean_reversion_diagnostic.csv", index=False)
    plot_mean_reversion_diagnostic(
        frame,
        output_dir / "mean_reversion_diagnostic.png",
        entry_z=1.6,
        exit_z=0.35,
    )
    print(f"Wrote sample report artifacts to {output_dir}")


if __name__ == "__main__":
    main()
