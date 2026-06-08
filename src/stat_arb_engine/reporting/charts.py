from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_mean_reversion_diagnostic(
    frame: pd.DataFrame,
    output_path: Path,
    *,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
) -> None:
    """Plot spread z-score, threshold positions, and cumulative PnL."""

    required = {"date", "zscore", "position", "pnl"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"diagnostic frame missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("diagnostic frame must not be empty")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dates = pd.to_datetime(frame["date"])
    cumulative_pnl = np.cumsum(frame["pnl"].to_numpy(dtype=float))

    fig, (z_ax, pnl_ax) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    z_ax.plot(dates, frame["zscore"], color="#315c72", linewidth=1.6, label="Spread z-score")
    z_ax.axhline(entry_z, color="#b64926", linestyle="--", linewidth=1.0, label="Entry band")
    z_ax.axhline(-entry_z, color="#b64926", linestyle="--", linewidth=1.0)
    z_ax.axhline(exit_z, color="#758e4f", linestyle=":", linewidth=1.0, label="Exit band")
    z_ax.axhline(-exit_z, color="#758e4f", linestyle=":", linewidth=1.0)
    z_ax.fill_between(
        dates,
        frame["position"].clip(upper=0.0),
        frame["position"].clip(lower=0.0),
        color="#d7b377",
        alpha=0.18,
        step="mid",
        label="Position",
    )
    z_ax.set_title("Deterministic Mean-Reversion Diagnostic")
    z_ax.set_ylabel("Z-score / position")
    z_ax.legend(loc="upper left", ncols=2)
    z_ax.grid(alpha=0.25)

    pnl_ax.plot(dates, cumulative_pnl, color="#2f6f4e", linewidth=1.8)
    pnl_ax.set_ylabel("Cumulative PnL")
    pnl_ax.set_xlabel("Date")
    pnl_ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
