from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TypeAlias

import pandas as pd


MarketDataFrame: TypeAlias = pd.DataFrame
REQUIRED_BAR_COLUMNS = {"timestamp", "symbol", "open", "high", "low", "close", "volume"}


@dataclass(frozen=True)
class Bar:
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float


def validate_market_frame(frame: MarketDataFrame) -> MarketDataFrame:
    missing = REQUIRED_BAR_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"market frame is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("market frame cannot be empty")
    return frame.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
