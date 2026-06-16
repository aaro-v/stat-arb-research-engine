from __future__ import annotations

import numpy as np
import pytest

from stat_arb_engine.risk import measure_exposure


def test_measure_exposure_reports_beta_leverage_and_concentration() -> None:
    report = measure_exposure(
        weights=np.array([0.35, -0.25, 0.15, -0.05]),
        betas=np.array([1.1, 0.7, -0.2, 1.5]),
    )

    assert report.net == pytest.approx(0.2)
    assert report.gross == pytest.approx(0.8)
    assert report.beta == pytest.approx(0.105)
    assert report.largest_long == 0.35
    assert report.largest_short == -0.25
    assert report.long_gross == pytest.approx(0.5)
    assert report.short_gross == pytest.approx(0.3)
    assert report.largest_abs == 0.35
    assert report.concentration == pytest.approx(0.328125)


def test_measure_exposure_reports_zero_concentration_for_flat_book() -> None:
    report = measure_exposure(weights=np.zeros(3), betas=np.array([1.0, 0.5, -0.5]))

    assert report.net == 0.0
    assert report.gross == 0.0
    assert report.largest_abs == 0.0
    assert report.concentration == 0.0


def test_measure_exposure_validates_shapes() -> None:
    with pytest.raises(ValueError, match="same shape"):
        measure_exposure(weights=np.array([0.1, -0.1]), betas=np.array([1.0]))
