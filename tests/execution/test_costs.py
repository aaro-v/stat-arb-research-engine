from __future__ import annotations

import numpy as np
import pytest

from stat_arb_engine.execution import CostModel, estimate_position_costs, estimate_round_trip_cost


def test_round_trip_cost_includes_borrow() -> None:
    costs = estimate_round_trip_cost(
        notional=np.array([100_000.0]),
        holding_days=np.array([10.0]),
        model=CostModel(),
    )
    assert costs[0] > 0


def test_position_costs_charge_turnover_and_short_borrow() -> None:
    costs = estimate_position_costs(
        positions=np.array([0.0, 1.0, -1.0, -1.0, 0.0]),
        model=CostModel(commission_bps=5.0, slippage_bps=5.0, borrow_bps_per_year=252.0),
        gross_notional=100.0,
    )

    assert costs == pytest.approx(np.array([0.0, 0.10, 0.21, 0.01, 0.10]))


def test_position_costs_respect_initial_position() -> None:
    costs = estimate_position_costs(
        positions=np.array([1.0, 0.0]),
        model=CostModel(commission_bps=5.0, slippage_bps=5.0),
        initial_position=1.0,
    )

    assert costs == pytest.approx(np.array([0.0, 0.001]))


def test_position_costs_accept_time_varying_notional() -> None:
    costs = estimate_position_costs(
        positions=np.array([0.0, 1.0, 0.0]),
        model=CostModel(commission_bps=10.0, slippage_bps=0.0),
        gross_notional=np.array([100.0, 200.0, 300.0]),
    )

    assert costs == pytest.approx(np.array([0.0, 0.20, 0.30]))


def test_position_costs_validate_inputs() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        estimate_position_costs(np.array([]), CostModel())
    with pytest.raises(ValueError, match="same shape"):
        estimate_position_costs(np.array([1.0, 0.0]), CostModel(), gross_notional=np.array([1.0]))
    with pytest.raises(ValueError, match="days_per_period"):
        estimate_position_costs(np.array([1.0]), CostModel(), days_per_period=-1.0)
