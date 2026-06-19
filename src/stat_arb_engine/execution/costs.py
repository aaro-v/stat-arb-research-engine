from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CostModel:
    commission_bps: float = 0.25
    slippage_bps: float = 1.0
    borrow_bps_per_year: float = 30.0
    trading_days: int = 252

    @property
    def one_way_bps(self) -> float:
        return self.commission_bps + self.slippage_bps


def estimate_position_costs(
    positions: np.ndarray,
    model: CostModel,
    gross_notional: float | np.ndarray = 1.0,
    initial_position: float = 0.0,
    days_per_period: float = 1.0,
) -> np.ndarray:
    """Estimate period trading and borrow costs for a normalized position path."""

    position_values = np.asarray(positions, dtype=float)
    if position_values.ndim != 1 or position_values.size == 0:
        raise ValueError("positions must be a non-empty one-dimensional array")
    if not np.isfinite(initial_position):
        raise ValueError("initial_position must be finite")
    if days_per_period < 0.0:
        raise ValueError("days_per_period must be non-negative")

    notional = _broadcast_notional(gross_notional, position_values.shape)
    position_changes = np.diff(position_values, prepend=initial_position)
    trading_cost = np.abs(position_changes) * notional * (model.one_way_bps / 10_000.0)
    borrow_cost = (
        np.abs(np.minimum(position_values, 0.0))
        * notional
        * (model.borrow_bps_per_year / 10_000.0)
        * (days_per_period / model.trading_days)
    )
    return trading_cost + borrow_cost


def estimate_round_trip_cost(notional: np.ndarray, holding_days: np.ndarray, model: CostModel) -> np.ndarray:
    gross = np.asarray(notional, dtype=float)
    days = np.asarray(holding_days, dtype=float)
    if gross.shape != days.shape:
        raise ValueError("notional and holding_days must have the same shape")

    trading_cost = gross * (2.0 * model.one_way_bps / 10_000.0)
    borrow_cost = gross * (model.borrow_bps_per_year / 10_000.0) * (days / model.trading_days)
    return trading_cost + borrow_cost


def _broadcast_notional(gross_notional: float | np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    notional = np.asarray(gross_notional, dtype=float)
    if notional.ndim == 0:
        return np.full(shape, float(notional))
    if notional.shape != shape:
        raise ValueError("gross_notional must be scalar or have the same shape as positions")
    return notional
