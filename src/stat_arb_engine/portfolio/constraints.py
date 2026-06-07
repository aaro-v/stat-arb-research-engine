from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioLimits:
    gross_leverage: float = 2.0
    max_position_weight: float = 0.20
    beta_tolerance: float = 1e-6
    turnover_limit: float | None = None

    def validate(self) -> None:
        if self.gross_leverage <= 0:
            raise ValueError("gross_leverage must be positive")
        if not 0 < self.max_position_weight <= 1:
            raise ValueError("max_position_weight must be in (0, 1]")
        if self.turnover_limit is not None and self.turnover_limit <= 0:
            raise ValueError("turnover_limit must be positive when provided")
