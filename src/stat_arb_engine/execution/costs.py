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


def estimate_round_trip_cost(notional: np.ndarray, holding_days: np.ndarray, model: CostModel) -> np.ndarray:
    gross = np.asarray(notional, dtype=float)
    days = np.asarray(holding_days, dtype=float)
    if gross.shape != days.shape:
        raise ValueError("notional and holding_days must have the same shape")

    trading_cost = gross * (2.0 * model.one_way_bps / 10_000.0)
    borrow_cost = gross * (model.borrow_bps_per_year / 10_000.0) * (days / model.trading_days)
    return trading_cost + borrow_cost
