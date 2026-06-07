from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OUParams:
    """Discrete-time estimate of an Ornstein-Uhlenbeck process."""

    theta: float
    mu: float
    sigma: float
    half_life: float


def estimate_ou(spread: np.ndarray) -> OUParams:
    """Estimate OU parameters from a spread using AR(1) regression."""

    values = np.asarray(spread, dtype=float)
    if values.ndim != 1 or values.size < 3:
        raise ValueError("spread must be a one-dimensional array with at least 3 observations")

    x = values[:-1]
    y = values[1:]
    design = np.column_stack([np.ones_like(x), x])
    intercept, phi = np.linalg.lstsq(design, y, rcond=None)[0]
    phi = float(np.clip(phi, 1e-6, 0.999999))

    theta = -np.log(phi)
    mu = intercept / (1.0 - phi)
    residuals = y - (intercept + phi * x)
    sigma = float(np.std(residuals, ddof=1) * np.sqrt(2.0 * theta / (1.0 - phi**2)))
    half_life = float(np.log(2.0) / theta)
    return OUParams(theta=float(theta), mu=float(mu), sigma=sigma, half_life=half_life)
