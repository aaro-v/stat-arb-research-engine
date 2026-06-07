from __future__ import annotations

import numpy as np

from stat_arb_engine.utils import ensure_1d_pair


def dynamic_hedge_ratio(
    y: np.ndarray,
    x: np.ndarray,
    process_var: float = 1e-5,
    obs_var: float = 1e-3,
) -> np.ndarray:
    """Estimate a time-varying hedge ratio with a compact Kalman filter."""

    lhs, rhs = ensure_1d_pair(y, x, "y", "x")

    beta = 0.0
    variance = 1.0
    estimates = np.empty_like(lhs)

    for i, (yi, xi) in enumerate(zip(lhs, rhs, strict=True)):
        variance += process_var
        innovation_var = xi * variance * xi + obs_var
        gain = variance * xi / innovation_var
        beta = beta + gain * (yi - xi * beta)
        variance = (1.0 - gain * xi) * variance
        estimates[i] = beta

    return estimates
