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


def measure_exposure(weights: np.ndarray, betas: np.ndarray) -> ExposureReport:
    w = np.asarray(weights, dtype=float)
    b = np.asarray(betas, dtype=float)
    if w.shape != b.shape or w.ndim != 1:
        raise ValueError("weights and betas must be one-dimensional arrays with the same shape")

    return ExposureReport(
        net=float(w.sum()),
        gross=float(np.abs(w).sum()),
        beta=float(w @ b),
        largest_long=float(w.max(initial=0.0)),
        largest_short=float(w.min(initial=0.0)),
    )
