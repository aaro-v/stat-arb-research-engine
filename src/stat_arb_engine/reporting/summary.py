from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResearchSummary:
    universe_size: int
    candidates: int
    median_half_life: float
    annualized_sharpe: float
    max_drawdown: float

    def as_lines(self) -> list[str]:
        return [
            f"Universe: {self.universe_size} instruments",
            f"Candidates: {self.candidates}",
            f"Median half-life: {self.median_half_life:.2f} days",
            f"Sharpe: {self.annualized_sharpe:.2f}",
            f"Max drawdown: {self.max_drawdown:.2%}",
        ]
