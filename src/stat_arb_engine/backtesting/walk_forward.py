from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WalkForwardSplit:
    train_start: int
    train_end: int
    test_start: int
    test_end: int


def rolling_splits(length: int, train_size: int, test_size: int, step: int | None = None) -> list[WalkForwardSplit]:
    if min(length, train_size, test_size) <= 0:
        raise ValueError("length, train_size, and test_size must be positive")

    stride = step or test_size
    splits: list[WalkForwardSplit] = []
    start = 0
    while start + train_size + test_size <= length:
        train_end = start + train_size
        test_end = train_end + test_size
        splits.append(WalkForwardSplit(start, train_end, train_end, test_end))
        start += stride
    return splits
