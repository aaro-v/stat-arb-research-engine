from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import BacktestSummary, summarize_pnl


@dataclass(frozen=True)
class RegimePnlSummary:
    regime: str
    observations: int
    observation_fraction: float
    return_share: float
    summary: BacktestSummary


def summarize_pnl_by_regime(
    pnl: np.ndarray,
    regimes: np.ndarray,
    *,
    periods_per_year: int = 252,
) -> list[RegimePnlSummary]:
    """Summarize PnL slices by an exogenous regime label."""

    values = np.asarray(pnl, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("pnl must be a non-empty one-dimensional array")

    labels = np.asarray(regimes)
    if labels.ndim != 1 or labels.size != values.size:
        raise ValueError("regimes must be a one-dimensional array with the same length as pnl")

    regime_names = _ordered_regimes(labels)
    total_abs_return = float(sum(abs(values[labels == name].sum()) for name in regime_names))
    summaries: list[RegimePnlSummary] = []
    for name in regime_names:
        regime_pnl = values[labels == name]
        summary = summarize_pnl(regime_pnl, periods_per_year=periods_per_year)
        summaries.append(
            RegimePnlSummary(
                regime=str(name),
                observations=int(regime_pnl.size),
                observation_fraction=float(regime_pnl.size / values.size),
                return_share=_return_share(summary.total_return, total_abs_return),
                summary=summary,
            )
        )
    return summaries


def _ordered_regimes(labels: np.ndarray) -> list[object]:
    ordered: list[object] = []
    for label in labels:
        if label not in ordered:
            ordered.append(label)
    return ordered


def _return_share(total_return: float, total_abs_return: float) -> float:
    if total_abs_return == 0.0:
        return 0.0
    return float(abs(total_return) / total_abs_return)
