"""Low-level 2×2 matrix ops (float64, vectorized-friendly)."""

from __future__ import annotations

import numpy as np

DTYPE_F = np.float64


def matmul_2x2(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """2×2 matrix product, result shape (2, 2)."""
    return A @ B


def power_T(a: int) -> np.ndarray:
    """T^a = [[1, a], [0, 1]]."""
    out = np.empty((2, 2), dtype=DTYPE_F)
    out[0, 0] = 1.0
    out[0, 1] = float(a)
    out[1, 0] = 0.0
    out[1, 1] = 1.0
    return out


def precompute_T_powers(max_power: int) -> tuple[np.ndarray, int]:
    """
    Stack T^a for a ∈ [-max_power, max_power].
    Index: a + offset, offset = max_power.
    Returns (stack, offset).
    """
    n = 2 * max_power + 1
    stack = np.empty((n, 2, 2), dtype=DTYPE_F)
    offset = max_power
    for i in range(n):
        a = i - offset
        stack[i, 0, 0] = 1.0
        stack[i, 0, 1] = float(a)
        stack[i, 1, 0] = 0.0
        stack[i, 1, 1] = 1.0
    return stack, offset


def matpow_2x2(A: np.ndarray, k: int) -> np.ndarray:
    """Integer power of a 2×2 matrix."""
    return np.linalg.matrix_power(np.asarray(A, dtype=DTYPE_F), int(k))
