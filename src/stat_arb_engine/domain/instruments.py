from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class AssetClass(StrEnum):
    EQUITY = "equity"
    FUTURE = "future"
    ETF = "etf"
    FX = "fx"
    RATE = "rate"


@dataclass(frozen=True)
class Instrument:
    symbol: str
    asset_class: AssetClass
    currency: str = "USD"
    exchange: str | None = None
    point_value: float = 1.0

    @property
    def identifier(self) -> str:
        venue = self.exchange or "GLOBAL"
        return f"{venue}:{self.symbol}"
