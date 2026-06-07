from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BacktestSummary:
    sharpe: float
    max_drawdown: float
    total_return: float
    trades: int


def summarize_pnl(pnl: np.ndarray, periods_per_year: int = 252) -> BacktestSummary:
    """Summarize a daily PnL vector with standard research metrics."""

    values = np.asarray(pnl, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("pnl must be a non-empty one-dimensional array")

    equity = np.cumsum(values)
    drawdowns = equity - np.maximum.accumulate(equity)
    volatility = np.std(values, ddof=1) if values.size > 1 else 0.0
    sharpe = 0.0 if volatility == 0 else float(np.mean(values) / volatility * np.sqrt(periods_per_year))
    return BacktestSummary(
        sharpe=sharpe,
        max_drawdown=float(drawdowns.min(initial=0.0)),
        total_return=float(equity[-1]),
        trades=int(np.count_nonzero(np.diff(np.sign(values), prepend=0))),
    )
