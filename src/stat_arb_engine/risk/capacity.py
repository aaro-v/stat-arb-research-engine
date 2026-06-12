from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CapacityReport:
    max_strategy_capital: float
    bottleneck_index: int
    bottleneck_limit: float
    per_asset_capacity: np.ndarray


def estimate_liquidity_capacity(
    weights: np.ndarray,
    average_daily_volume: np.ndarray,
    prices: np.ndarray,
    *,
    max_adv_participation: float = 0.05,
    borrow_availability: np.ndarray | None = None,
) -> CapacityReport:
    """Estimate strategy capital constrained by ADV participation and borrow supply.

    Weights are interpreted as signed dollar exposure per dollar of strategy capital.
    For example, a weight of -0.25 means a 25 cent short for each dollar allocated.
    """

    w = np.asarray(weights, dtype=float)
    adv = np.asarray(average_daily_volume, dtype=float)
    px = np.asarray(prices, dtype=float)
    _validate_capacity_inputs(w, adv, px, max_adv_participation)

    dollar_liquidity = adv * px * max_adv_participation
    dollar_limits = dollar_liquidity.copy()
    per_asset_capacity = np.divide(
        dollar_liquidity,
        np.abs(w),
        out=np.full_like(w, np.inf, dtype=float),
        where=np.abs(w) > 0.0,
    )

    if borrow_availability is not None:
        borrow = np.asarray(borrow_availability, dtype=float)
        if borrow.shape != w.shape or borrow.ndim != 1:
            raise ValueError("borrow_availability must match the one-dimensional weights shape")
        if np.any(borrow < 0.0):
            raise ValueError("borrow_availability must be non-negative")
        short_capacity = np.divide(
            borrow,
            np.abs(w),
            out=np.full_like(w, np.inf, dtype=float),
            where=w < 0.0,
        )
        per_asset_capacity = np.minimum(per_asset_capacity, short_capacity)
        dollar_limits = np.minimum(
            dollar_limits,
            np.where(w < 0.0, borrow, np.inf),
        )

    if np.all(np.isinf(per_asset_capacity)):
        return CapacityReport(
            max_strategy_capital=float("inf"),
            bottleneck_index=-1,
            bottleneck_limit=float("inf"),
            per_asset_capacity=per_asset_capacity,
        )

    bottleneck_index = int(np.argmin(per_asset_capacity))
    return CapacityReport(
        max_strategy_capital=float(per_asset_capacity[bottleneck_index]),
        bottleneck_index=bottleneck_index,
        bottleneck_limit=float(dollar_limits[bottleneck_index]),
        per_asset_capacity=per_asset_capacity,
    )


def _validate_capacity_inputs(
    weights: np.ndarray,
    average_daily_volume: np.ndarray,
    prices: np.ndarray,
    max_adv_participation: float,
) -> None:
    if weights.ndim != 1 or average_daily_volume.ndim != 1 or prices.ndim != 1:
        raise ValueError("weights, average_daily_volume, and prices must be one-dimensional")
    if weights.shape != average_daily_volume.shape or weights.shape != prices.shape:
        raise ValueError("weights, average_daily_volume, and prices must have the same shape")
    if weights.size == 0:
        raise ValueError("capacity inputs must not be empty")
    if np.any(average_daily_volume < 0.0):
        raise ValueError("average_daily_volume must be non-negative")
    if np.any(prices <= 0.0):
        raise ValueError("prices must be positive")
    if max_adv_participation <= 0.0 or max_adv_participation > 1.0:
        raise ValueError("max_adv_participation must be in (0, 1]")
