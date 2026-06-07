from __future__ import annotations

import cvxpy as cp
import numpy as np

from stat_arb_engine.portfolio.constraints import PortfolioLimits
from stat_arb_engine.utils import ensure_1d_pair


def build_market_neutral_weights(
    expected_returns: np.ndarray,
    betas: np.ndarray,
    leverage_limit: float = 2.0,
    risk_aversion: float = 1.0,
) -> np.ndarray:
    """Build beta-neutral weights under a leverage constraint."""

    limits = PortfolioLimits(gross_leverage=leverage_limit)
    limits.validate()
    alpha, market_beta = ensure_1d_pair(expected_returns, betas, "expected_returns", "betas")

    weights = cp.Variable(alpha.size)
    objective = cp.Maximize(alpha @ weights - risk_aversion * cp.sum_squares(weights))
    constraints = [
        cp.sum(weights) == 0,
        market_beta @ weights == 0,
        cp.norm1(weights) <= limits.gross_leverage,
        weights <= limits.max_position_weight,
        weights >= -limits.max_position_weight,
    ]
    problem = cp.Problem(objective, constraints)
    problem.solve(solver=cp.CLARABEL)

    if weights.value is None:
        raise RuntimeError("portfolio optimization failed")
    return np.asarray(weights.value).round(12)
