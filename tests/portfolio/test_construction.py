from __future__ import annotations

import numpy as np

from stat_arb_engine import build_market_neutral_weights
from stat_arb_engine.risk import measure_exposure


def test_market_neutral_weights_constraints() -> None:
    weights = build_market_neutral_weights(
        expected_returns=np.array([0.03, 0.01, -0.02]),
        betas=np.array([1.0, 1.0, 1.0]),
    )
    assert abs(weights.sum()) < 1e-8


def test_exposure_report() -> None:
    report = measure_exposure(np.array([0.1, -0.1]), np.array([1.0, 1.2]))
    assert report.gross == 0.2
    assert report.beta < 0
