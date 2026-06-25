from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import BacktestSummary, summarize_pnl


@dataclass(frozen=True)
class RollingWindowSummary:
    window: int
    start: int
    end: int
    summary: BacktestSummary


@dataclass(frozen=True)
class DecayDiagnostics:
    windows: int
    window_size: int
    recent_window_return: float
    early_mean_return: float
    recent_mean_return: float
    return_decay: float
    early_mean_sharpe: float
    recent_mean_sharpe: float
    sharpe_decay: float
    negative_window_rate: float
    worst_window: int
    worst_window_return: float
    best_window: int
    best_window_return: float


def rolling_window_summaries(
    pnl: np.ndarray,
    *,
    window_size: int,
    step: int | None = None,
    periods_per_year: int = 252,
    positions: np.ndarray | None = None,
) -> list[RollingWindowSummary]:
    """Summarize overlapping PnL windows for strategy decay monitoring."""

    values = np.asarray(pnl, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("pnl must be a non-empty one-dimensional array")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if window_size > values.size:
        raise ValueError("window_size must be no larger than pnl length")
    if step is not None and step <= 0:
        raise ValueError("step must be positive")

    position_values = None if positions is None else np.asarray(positions, dtype=float)
    if position_values is not None and (
        position_values.ndim != 1 or position_values.size != values.size
    ):
        raise ValueError("positions must be a one-dimensional array with the same length as pnl")

    stride = step or window_size
    windows: list[RollingWindowSummary] = []
    window_index = 0
    for start in range(0, values.size - window_size + 1, stride):
        end = start + window_size
        window_positions = None if position_values is None else position_values[start:end]
        initial_position = 0.0 if position_values is None or start == 0 else float(position_values[start - 1])
        summary = summarize_pnl(
            values[start:end],
            periods_per_year=periods_per_year,
            positions=window_positions,
            initial_position=initial_position,
        )
        windows.append(
            RollingWindowSummary(
                window=window_index,
                start=start,
                end=end,
                summary=summary,
            )
        )
        window_index += 1
    return windows


def aggregate_decay_diagnostics(
    windows: list[RollingWindowSummary],
    *,
    comparison_windows: int = 2,
) -> DecayDiagnostics:
    """Compare early and recent rolling windows to flag performance decay."""

    if not windows:
        raise ValueError("windows must not be empty")
    if comparison_windows <= 0:
        raise ValueError("comparison_windows must be positive")

    sample_size = min(comparison_windows, len(windows))
    returns = np.array([window.summary.total_return for window in windows], dtype=float)
    sharpes = np.array([window.summary.sharpe for window in windows], dtype=float)
    early_returns = returns[:sample_size]
    recent_returns = returns[-sample_size:]
    early_sharpes = sharpes[:sample_size]
    recent_sharpes = sharpes[-sample_size:]
    worst_window_index = int(np.argmin(returns))
    best_window_index = int(np.argmax(returns))

    early_mean_return = float(np.mean(early_returns))
    recent_mean_return = float(np.mean(recent_returns))
    early_mean_sharpe = float(np.mean(early_sharpes))
    recent_mean_sharpe = float(np.mean(recent_sharpes))
    return DecayDiagnostics(
        windows=len(windows),
        window_size=windows[0].end - windows[0].start,
        recent_window_return=float(returns[-1]),
        early_mean_return=early_mean_return,
        recent_mean_return=recent_mean_return,
        return_decay=float(recent_mean_return - early_mean_return),
        early_mean_sharpe=early_mean_sharpe,
        recent_mean_sharpe=recent_mean_sharpe,
        sharpe_decay=float(recent_mean_sharpe - early_mean_sharpe),
        negative_window_rate=float(np.count_nonzero(returns < 0.0) / len(returns)),
        worst_window=windows[worst_window_index].window,
        worst_window_return=float(returns[worst_window_index]),
        best_window=windows[best_window_index].window,
        best_window_return=float(returns[best_window_index]),
    )
