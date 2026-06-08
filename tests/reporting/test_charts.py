from __future__ import annotations

import pandas as pd
import pytest

from stat_arb_engine.reporting.charts import plot_mean_reversion_diagnostic


def test_plot_mean_reversion_diagnostic_writes_png(tmp_path) -> None:
    frame = pd.DataFrame(
        {
            "date": pd.bdate_range("2024-01-02", periods=4),
            "zscore": [0.0, 1.8, -1.7, 0.2],
            "position": [0.0, -1.0, 1.0, 0.0],
            "pnl": [0.0, 0.2, 0.1, -0.05],
        }
    )

    output = tmp_path / "chart.png"
    plot_mean_reversion_diagnostic(frame, output)

    assert output.exists()
    assert output.stat().st_size > 0


def test_plot_mean_reversion_diagnostic_requires_columns(tmp_path) -> None:
    with pytest.raises(ValueError, match="missing columns"):
        plot_mean_reversion_diagnostic(pd.DataFrame({"date": []}), tmp_path / "chart.png")
