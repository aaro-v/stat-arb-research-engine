from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ThresholdSignal(StrEnum):
    ENTER_LONG_SPREAD = "enter_long_spread"
    ENTER_SHORT_SPREAD = "enter_short_spread"
    EXIT = "exit"
    HOLD = "hold"


@dataclass(frozen=True)
class ThresholdPolicy:
    entry_z: float = 2.0
    exit_z: float = 0.5


def classify_zscore(zscore: float, policy: ThresholdPolicy | None = None) -> ThresholdSignal:
    active_policy = policy or ThresholdPolicy()
    if zscore <= -active_policy.entry_z:
        return ThresholdSignal.ENTER_LONG_SPREAD
    if zscore >= active_policy.entry_z:
        return ThresholdSignal.ENTER_SHORT_SPREAD
    if abs(zscore) <= active_policy.exit_z:
        return ThresholdSignal.EXIT
    return ThresholdSignal.HOLD
