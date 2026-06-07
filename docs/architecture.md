# Architecture

The engine is organized as a layered research platform.

## Layers

- `data`: market-data contracts, validation, and loader interfaces.
- `domain`: instruments, calendars, identifiers, and shared value objects.
- `research`: statistical routines such as cointegration tests, OU calibration, and dynamic hedge-ratio models.
- `signals`: entry, exit, sizing, and state-transition rules.
- `portfolio`: optimizers, constraints, and exposure-aware construction.
- `execution`: transaction-cost, borrow-cost, and slippage assumptions.
- `backtesting`: simulation, accounting, stress tests, and walk-forward evaluation.
- `risk`: drawdown, beta, leverage, concentration, and capacity checks.
- `reporting`: diagnostics and human-readable research summaries.

## Growth Plan

1. Expand data ingestion behind stable interfaces.
2. Add basket cointegration and cross-sectional candidate ranking.
3. Separate signal generation from position sizing.
4. Add walk-forward experiment manifests and reproducible result stores.
5. Add capacity analysis with liquidity and borrow availability.
6. Add report generation for candidate reviews and strategy decay monitoring.

The code should stay modular: large research ideas belong in several small modules with tests, not in one notebook-style file.
