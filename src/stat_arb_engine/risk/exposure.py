from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExposureReport:
    net: float
    gross: float
    beta: float
    largest_long: float
    largest_short: float
    long_gross: float = 0.0
    short_gross: float = 0.0
    largest_abs: float = 0.0
    concentration: float = 0.0


def measure_exposure(weights: np.ndarray, betas: np.ndarray) -> ExposureReport:
    """Measure portfolio leverage, beta exposure, and position concentration."""

    w = np.asarray(weights, dtype=float)
    b = np.asarray(betas, dtype=float)
    if w.shape != b.shape or w.ndim != 1:
        raise ValueError("weights and betas must be one-dimensional arrays with the same shape")

    gross = float(np.abs(w).sum())
    long_gross = float(w[w > 0.0].sum(initial=0.0))
    short_gross = float(np.abs(w[w < 0.0]).sum(initial=0.0))
    return ExposureReport(
        net=float(w.sum()),
        gross=gross,
        beta=float(w @ b),
        largest_long=float(w.max(initial=0.0)),
        largest_short=float(w.min(initial=0.0)),
        long_gross=long_gross,
        short_gross=short_gross,
        largest_abs=float(np.abs(w).max(initial=0.0)),
        concentration=_gross_concentration(w, gross),
    )


def _gross_concentration(weights: np.ndarray, gross: float) -> float:
    if gross == 0.0:
        return 0.0
    gross_shares = np.abs(weights) / gross
    return float(np.square(gross_shares).sum())
