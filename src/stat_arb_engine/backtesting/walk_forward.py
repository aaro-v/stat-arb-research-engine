from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import BacktestSummary, summarize_pnl


@dataclass(frozen=True)
class WalkForwardSplit:
    train_start: int
    train_end: int
    test_start: int
    test_end: int


@dataclass(frozen=True)
class WalkForwardFoldSummary:
    fold: int
    split: WalkForwardSplit
    summary: BacktestSummary


@dataclass(frozen=True)
class WalkForwardDiagnostics:
    folds: int
    mean_sharpe: float
    median_sharpe: float
    worst_drawdown: float
    total_return: float
    positive_fold_rate: float


def rolling_splits(
    length: int,
    train_size: int,
    test_size: int,
    step: int | None = None,
) -> list[WalkForwardSplit]:
    if min(length, train_size, test_size) <= 0:
        raise ValueError("length, train_size, and test_size must be positive")
    if step is not None and step <= 0:
        raise ValueError("step must be positive")

    stride = step or test_size
    splits: list[WalkForwardSplit] = []
    start = 0
    while start + train_size + test_size <= length:
        train_end = start + train_size
        test_end = train_end + test_size
        splits.append(WalkForwardSplit(start, train_end, train_end, test_end))
        start += stride
    return splits


def summarize_walk_forward_pnl(
    pnl: np.ndarray,
    splits: list[WalkForwardSplit],
    *,
    periods_per_year: int = 252,
    positions: np.ndarray | None = None,
) -> list[WalkForwardFoldSummary]:
    """Summarize each out-of-sample walk-forward test window."""

    values = np.asarray(pnl, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("pnl must be a non-empty one-dimensional array")
    if not splits:
        raise ValueError("splits must not be empty")

    position_values = None if positions is None else np.asarray(positions, dtype=float)
    if position_values is not None and (
        position_values.ndim != 1 or position_values.size != values.size
    ):
        raise ValueError("positions must be a one-dimensional array with the same length as pnl")

    fold_summaries: list[WalkForwardFoldSummary] = []
    for index, split in enumerate(splits):
        _validate_split(split, values.size)
        test_positions = None if position_values is None else position_values[split.test_start : split.test_end]
        summary = summarize_pnl(
            values[split.test_start : split.test_end],
            periods_per_year=periods_per_year,
            positions=test_positions,
        )
        fold_summaries.append(WalkForwardFoldSummary(fold=index, split=split, summary=summary))
    return fold_summaries


def aggregate_walk_forward_diagnostics(
    folds: list[WalkForwardFoldSummary],
) -> WalkForwardDiagnostics:
    """Aggregate fold summaries into stability diagnostics for research review."""

    if not folds:
        raise ValueError("folds must not be empty")

    sharpes = np.array([fold.summary.sharpe for fold in folds], dtype=float)
    returns = np.array([fold.summary.total_return for fold in folds], dtype=float)
    drawdowns = np.array([fold.summary.max_drawdown for fold in folds], dtype=float)
    return WalkForwardDiagnostics(
        folds=len(folds),
        mean_sharpe=float(np.mean(sharpes)),
        median_sharpe=float(np.median(sharpes)),
        worst_drawdown=float(np.min(drawdowns)),
        total_return=float(np.sum(returns)),
        positive_fold_rate=float(np.count_nonzero(returns > 0.0) / len(returns)),
    )


def _validate_split(split: WalkForwardSplit, length: int) -> None:
    if not (0 <= split.train_start < split.train_end <= split.test_start < split.test_end <= length):
        raise ValueError(f"invalid walk-forward split bounds: {split}")
