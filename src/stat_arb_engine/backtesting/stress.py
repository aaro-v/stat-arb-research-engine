from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BootstrapStressSummary:
    simulations: int
    horizon: int
    block_size: int
    mean_total_return: float
    median_total_return: float
    total_return_p05: float
    total_return_p95: float
    loss_probability: float
    max_drawdown_p05: float
    expected_shortfall_95: float


def block_bootstrap_stress(
    pnl: np.ndarray,
    *,
    simulations: int = 1_000,
    horizon: int | None = None,
    block_size: int = 5,
    seed: int | None = None,
) -> BootstrapStressSummary:
    """Estimate PnL path stability with moving-block bootstrap resamples."""

    values = np.asarray(pnl, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("pnl must be a non-empty one-dimensional array")
    if simulations <= 0:
        raise ValueError("simulations must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    target_horizon = values.size if horizon is None else horizon
    if target_horizon <= 0:
        raise ValueError("horizon must be positive")

    rng = np.random.default_rng(seed)
    total_returns = np.empty(simulations, dtype=float)
    max_drawdowns = np.empty(simulations, dtype=float)
    for index in range(simulations):
        sample = _moving_block_sample(values, target_horizon, block_size, rng)
        equity = np.cumsum(sample)
        drawdowns = equity - np.maximum.accumulate(equity)
        total_returns[index] = equity[-1]
        max_drawdowns[index] = drawdowns.min(initial=0.0)

    loss_tail = total_returns[total_returns <= np.quantile(total_returns, 0.05)]
    expected_shortfall = 0.0 if loss_tail.size == 0 else max(0.0, -float(np.mean(loss_tail)))
    return BootstrapStressSummary(
        simulations=simulations,
        horizon=target_horizon,
        block_size=block_size,
        mean_total_return=float(np.mean(total_returns)),
        median_total_return=float(np.median(total_returns)),
        total_return_p05=float(np.quantile(total_returns, 0.05)),
        total_return_p95=float(np.quantile(total_returns, 0.95)),
        loss_probability=float(np.count_nonzero(total_returns < 0.0) / simulations),
        max_drawdown_p05=float(np.quantile(max_drawdowns, 0.05)),
        expected_shortfall_95=expected_shortfall,
    )


def _moving_block_sample(
    values: np.ndarray,
    horizon: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    starts = np.arange(max(1, values.size - block_size + 1))
    chunks: list[np.ndarray] = []
    sampled = 0
    while sampled < horizon:
        start = int(rng.choice(starts))
        block = values[start : start + block_size]
        chunks.append(block)
        sampled += block.size
    return np.concatenate(chunks)[:horizon]
