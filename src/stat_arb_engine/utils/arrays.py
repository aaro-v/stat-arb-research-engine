from __future__ import annotations

import numpy as np


def ensure_1d_pair(left: np.ndarray, right: np.ndarray, left_name: str = "left", right_name: str = "right") -> tuple[np.ndarray, np.ndarray]:
    lhs = np.asarray(left, dtype=float)
    rhs = np.asarray(right, dtype=float)
    if lhs.shape != rhs.shape or lhs.ndim != 1:
        raise ValueError(f"{left_name} and {right_name} must be one-dimensional arrays with the same shape")
    return lhs, rhs
