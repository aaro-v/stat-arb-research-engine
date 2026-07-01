# Statistical Arbitrage Research Engine

A Python research platform for mean-reversion statistical arbitrage across equities and futures.

The project is organized around a production research workflow: market data normalization, candidate discovery, signal calibration, portfolio construction, risk controls, backtest accounting, and reporting. The current implementation focuses on reliable primitives and clear module boundaries so the engine can grow without turning into a collection of notebooks or oversized scripts.

## Scope

- Cointegration screening for pair and basket candidates.
- Ornstein-Uhlenbeck calibration for spread half-life, long-run mean, and volatility.
- Kalman-filtered dynamic hedge ratios for changing relationships.
- Convex portfolio construction with leverage, beta-neutrality, and turnover constraints.
- Entry and exit threshold objects for mean-reversion signals.
- Backtests with transaction costs, borrow costs, slippage, walk-forward validation, and bootstrap stress tests.
- Exposure diagnostics for net/gross leverage, beta, long/short books, and gross concentration.

## Design Principles

- Keep research methods isolated from IO and reporting.
- Prefer typed configuration objects over loose dictionaries.
- Model costs and constraints explicitly instead of hiding them in backtest loops.
- Make every layer testable with synthetic data before connecting live data sources.
- Keep public APIs small while allowing internal modules to evolve.

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Run the daily maintenance check:

```bash
python scripts/daily_research_check.py
```

Generate the deterministic sample report artifacts:

```bash
python scripts/generate_sample_report.py
```

## Sample Results

The chart below is generated from a deterministic synthetic mean-reverting
spread. It is included because it exercises real project logic: threshold
classification, position state, transaction-cost-aware PnL, and reporting.

![Mean-reversion diagnostic](reports/generated/mean_reversion_diagnostic.png)

The paired `reports/generated/mean_reversion_summary.csv` file records the
same deterministic engineering validation run with path diagnostics including
PnL, Sharpe, drawdown, trade count, hit rate, profit factor, turnover, and
average holding period. It also records active, long, short, and flat exposure
fractions so reviewers can separate path results from position occupancy. It
also includes gross exposure and return per gross exposure so reviewers can
normalize synthetic PnL by the amount of position deployed. It
also records drawdown-duration and time-under-water metrics so reviewers can
inspect path persistence, not just return magnitude.
It also includes underwater fraction, average drawdown, ulcer index, and
drawdown recovery ratio for a more complete view of drawdown persistence.
The companion `reports/generated/mean_reversion_walk_forward.csv` file breaks
the same synthetic path into deterministic out-of-sample folds with fold-level
PnL, drawdown, turnover, and hit-rate diagnostics. The summary also includes
fold consistency and positive-return concentration metrics to flag whether a
synthetic validation result is broadly persistent or dominated by a small
number of out-of-sample windows. It also reports the fold mean-return standard
error, t-statistic, and normal-approximation 95% confidence interval as
engineering diagnostics for out-of-sample stability.
The `reports/generated/mean_reversion_stress.csv` file adds deterministic
moving-block bootstrap stress diagnostics for the same synthetic path,
including simulated return quantiles, loss probability, drawdown quantile, and
expected shortfall.
The `reports/generated/mean_reversion_decay.csv` file records rolling 63-day
window diagnostics with overlapping 21-day steps, and the summary captures
early-vs-recent return and Sharpe decay checks for strategy monitoring.
The `reports/generated/mean_reversion_drawdowns.csv` file records each
contiguous drawdown episode, including start, trough, recovery endpoint,
depth, duration, and recovery duration for path-level review.
The `reports/generated/mean_reversion_regimes.csv` file groups the same
synthetic PnL path by deterministic low/high spread-volatility regimes so
reviewers can inspect whether validation output is concentrated in one
market-state proxy. It also reports gross exposure share and long/short/flat
occupancy by regime so concentration can be reviewed against deployed
positioning, not just realized synthetic PnL.
The `reports/generated/manifest.csv` file records byte sizes and SHA-256
checksums for every generated sample artifact so deterministic report changes
are easy to review.

## Project Layout

```text
src/stat_arb_engine/
  backtesting/        Engines, cost models, metrics, stress tests
  data/               Data contracts, loaders, validation
  domain/             Shared research dataclasses and instruments
  execution/          Slippage, borrow, and transaction cost assumptions
  portfolio/          Optimizers, constraints, and exposure accounting
  reporting/          Research summaries and diagnostics
  research/           Cointegration, OU calibration, Kalman filters
  risk/               Exposure, drawdown, and capacity controls
  signals/            Entry/exit rules and signal state
  utils/              Time, arrays, and validation helpers
scripts/
  daily_research_check.py
tests/
  backtesting/
  portfolio/
  research/
```

See `docs/architecture.md` for the intended growth path.

## Notes

This is research software, not investment advice. Backtest results are not guarantees of live performance.
