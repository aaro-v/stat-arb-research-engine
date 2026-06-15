from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BacktestSummary:
    sharpe: float
    max_drawdown: float
    max_drawdown_duration: int
    total_return: float
    trades: int
    hit_rate: float = 0.0
    profit_factor: float = 0.0
    turnover: float = 0.0
    average_holding_period: float = 0.0
    annualized_volatility: float = 0.0
    downside_deviation: float = 0.0
    sortino: float = 0.0
    tail_loss_95: float = 0.0
    expected_shortfall_95: float = 0.0
    time_under_water: int = 0


def summarize_pnl(
    pnl: np.ndarray,
    periods_per_year: int = 252,
    positions: np.ndarray | None = None,
) -> BacktestSummary:
    """Summarize a PnL vector with path diagnostics used in research review."""

    values = np.asarray(pnl, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("pnl must be a non-empty one-dimensional array")
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")

    position_values = None if positions is None else np.asarray(positions, dtype=float)
    if position_values is not None and (
        position_values.ndim != 1 or position_values.size != values.size
    ):
        raise ValueError("positions must be a one-dimensional array with the same length as pnl")

    equity = np.cumsum(values)
    drawdowns = equity - np.maximum.accumulate(equity)
    max_drawdown_duration, time_under_water = _drawdown_duration(drawdowns)
    volatility = np.std(values, ddof=1) if values.size > 1 else 0.0
    sharpe = 0.0 if volatility == 0 else float(np.mean(values) / volatility * np.sqrt(periods_per_year))
    downside_deviation = _downside_deviation(values)
    sortino = (
        0.0
        if downside_deviation == 0.0
        else float(np.mean(values) / downside_deviation * np.sqrt(periods_per_year))
    )
    tail_loss_95, expected_shortfall_95 = _tail_loss(values, confidence=0.95)
    gains = values[values > 0.0].sum(initial=0.0)
    losses = values[values < 0.0].sum(initial=0.0)
    active_periods = np.count_nonzero(values)
    trades = _count_pnl_sign_changes(values)
    turnover = 0.0
    average_holding_period = 0.0
    if position_values is not None:
        trades = int(np.count_nonzero(np.diff(position_values, prepend=0.0)))
        turnover = float(np.abs(np.diff(position_values, prepend=0.0)).sum())
        average_holding_period = _average_holding_period(position_values)

    return BacktestSummary(
        sharpe=sharpe,
        max_drawdown=float(drawdowns.min(initial=0.0)),
        max_drawdown_duration=max_drawdown_duration,
        total_return=float(equity[-1]),
        trades=trades,
        hit_rate=0.0 if active_periods == 0 else float(np.count_nonzero(values > 0.0) / active_periods),
        profit_factor=float("inf") if losses == 0.0 and gains > 0.0 else _safe_profit_factor(gains, losses),
        turnover=turnover,
        average_holding_period=average_holding_period,
        annualized_volatility=float(volatility * np.sqrt(periods_per_year)),
        downside_deviation=downside_deviation,
        sortino=sortino,
        tail_loss_95=tail_loss_95,
        expected_shortfall_95=expected_shortfall_95,
        time_under_water=time_under_water,
    )


def _count_pnl_sign_changes(values: np.ndarray) -> int:
    return int(np.count_nonzero(np.diff(np.sign(values), prepend=0.0)))


def _safe_profit_factor(gains: float, losses: float) -> float:
    if losses == 0.0:
        return 0.0
    return float(gains / abs(losses))


def _downside_deviation(values: np.ndarray) -> float:
    downside = np.minimum(values, 0.0)
    if not np.any(downside):
        return 0.0
    return float(np.sqrt(np.mean(np.square(downside))))


def _tail_loss(values: np.ndarray, confidence: float) -> tuple[float, float]:
    quantile = np.quantile(values, 1.0 - confidence)
    tail = values[values <= quantile]
    tail_loss = max(0.0, -float(quantile))
    expected_shortfall = 0.0 if tail.size == 0 else max(0.0, -float(np.mean(tail)))
    return tail_loss, expected_shortfall


def _drawdown_duration(drawdowns: np.ndarray) -> tuple[int, int]:
    max_duration = 0
    current_duration = 0
    time_under_water = 0
    for drawdown in drawdowns:
        if drawdown < 0.0:
            current_duration += 1
            time_under_water += 1
            max_duration = max(max_duration, current_duration)
        else:
            current_duration = 0
    return max_duration, time_under_water


def _average_holding_period(positions: np.ndarray) -> float:
    holding_periods: list[int] = []
    current_length = 0
    was_active = False
    for position in positions:
        is_active = position != 0.0
        if is_active:
            current_length = current_length + 1 if was_active else 1
        elif was_active:
            holding_periods.append(current_length)
            current_length = 0
        was_active = is_active
    if was_active:
        holding_periods.append(current_length)
    return 0.0 if not holding_periods else float(np.mean(holding_periods))
