from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from statsmodels.tsa.stattools import coint

from stat_arb_engine.utils import ensure_1d_pair


@dataclass(frozen=True)
class CointegrationResult:
    score: float
    pvalue: float
    hedge_ratio: float
    spread: np.ndarray
    is_candidate: bool


def screen_pair(y: np.ndarray, x: np.ndarray, pvalue_threshold: float = 0.05) -> CointegrationResult:
    """Run Engle-Granger screening and estimate a static hedge ratio."""

    lhs, rhs = ensure_1d_pair(y, x, "y", "x")

    design = np.column_stack([np.ones_like(rhs), rhs])
    intercept, beta = np.linalg.lstsq(design, lhs, rcond=None)[0]
    spread = lhs - (intercept + beta * rhs)
    score, pvalue, _ = coint(lhs, rhs)

    return CointegrationResult(
        score=float(score),
        pvalue=float(pvalue),
        hedge_ratio=float(beta),
        spread=spread,
        is_candidate=bool(pvalue <= pvalue_threshold),
    )
