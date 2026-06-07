"""Statistical arbitrage research primitives."""

from .portfolio.construction import build_market_neutral_weights
from .research.cointegration import CointegrationResult, screen_pair
from .research.ou import OUParams, estimate_ou

__all__ = [
    "CointegrationResult",
    "OUParams",
    "build_market_neutral_weights",
    "estimate_ou",
    "screen_pair",
]
