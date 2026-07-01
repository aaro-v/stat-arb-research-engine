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
    gross_exposure: float = 0.0
    exposure_share: float = 0.0
    active_fraction: float = 0.0
    long_fraction: float = 0.0
    short_fraction: float = 0.0
    flat_fraction: float = 0.0


def summarize_pnl_by_regime(
    pnl: np.ndarray,
    regimes: np.ndarray,
    *,
    periods_per_year: int = 252,
    positions: np.ndarray | None = None,
) -> list[RegimePnlSummary]:
    """Summarize PnL slices by an exogenous regime label."""

    values = np.asarray(pnl, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("pnl must be a non-empty one-dimensional array")

    labels = np.asarray(regimes)
    if labels.ndim != 1 or labels.size != values.size:
        raise ValueError("regimes must be a one-dimensional array with the same length as pnl")

    position_values = None if positions is None else np.asarray(positions, dtype=float)
    if position_values is not None and (
        position_values.ndim != 1 or position_values.size != values.size
    ):
        raise ValueError("positions must be a one-dimensional array with the same length as pnl")

    regime_names = _ordered_regimes(labels)
    total_abs_return = float(sum(abs(values[labels == name].sum()) for name in regime_names))
    total_gross_exposure = (
        0.0 if position_values is None else float(np.abs(position_values).sum())
    )
    summaries: list[RegimePnlSummary] = []
    for name in regime_names:
        mask = labels == name
        regime_pnl = values[mask]
        summary = summarize_pnl(regime_pnl, periods_per_year=periods_per_year)
        (
            gross_exposure,
            active_fraction,
            long_fraction,
            short_fraction,
            flat_fraction,
        ) = _regime_position_metrics(None if position_values is None else position_values[mask])
        summaries.append(
            RegimePnlSummary(
                regime=str(name),
                observations=int(regime_pnl.size),
                observation_fraction=float(regime_pnl.size / values.size),
                return_share=_return_share(summary.total_return, total_abs_return),
                gross_exposure=gross_exposure,
                exposure_share=_return_share(gross_exposure, total_gross_exposure),
                active_fraction=active_fraction,
                long_fraction=long_fraction,
                short_fraction=short_fraction,
                flat_fraction=flat_fraction,
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


def _regime_position_metrics(positions: np.ndarray | None) -> tuple[float, float, float, float, float]:
    if positions is None:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    active = positions != 0.0
    long = positions > 0.0
    short = positions < 0.0
    periods = positions.size
    return (
        float(np.abs(positions).sum()),
        float(np.count_nonzero(active) / periods),
        float(np.count_nonzero(long) / periods),
        float(np.count_nonzero(short) / periods),
        float(np.count_nonzero(~active) / periods),
    )
