from __future__ import annotations

"""Experimental DTES spectral-learning utilities.

These routines build a symmetric DTES graph operator and compare its spectrum
to known zeta-zero ordinates. This is a numerical learning signal, not a proof
of the Riemann hypothesis.
"""

import numpy as np


def _as_1d_float(values) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def _as_square_float(values) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("matrix input must be square")
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def build_operator(t_grid, zeta_abs, pheromone_matrix):
    t_grid = _as_1d_float(t_grid)
    zeta_abs = _as_1d_float(zeta_abs)
    W = _as_square_float(pheromone_matrix)
    if t_grid.size == 0:
        raise ValueError("t_grid must be non-empty")
    if zeta_abs.size != t_grid.size or W.shape[0] != t_grid.size:
        raise ValueError("t_grid, zeta_abs, and pheromone_matrix dimensions must agree")

    W = 0.5 * (W + W.T)
    W = np.maximum(W, 0.0)
    np.fill_diagonal(W, 0.0)

    D = np.diag(W.sum(axis=1))
    L = D - W

    zeta_abs = np.maximum(zeta_abs, 1e-12)
    V = -np.log(zeta_abs)
    V = (V - V.mean()) / (V.std() + 1e-12)

    H = L + np.diag(V)
    H = 0.5 * (H + H.T)
    return np.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)


def compute_spectrum(H, k=50):
    H = _as_square_float(H)
    H = 0.5 * (H + H.T)
    eigvals, _ = np.linalg.eigh(H)
    if k is None or k <= 0:
        return eigvals
    return eigvals[: min(int(k), eigvals.size)]


def normalize(x):
    x = _as_1d_float(x)
    if x.size == 0:
        return x
    return (x - np.mean(x)) / (np.std(x) + 1e-12)


def spectral_loss(eigvals, zeta_zeros):
    eigvals = _as_1d_float(eigvals)
    zeta_zeros = _as_1d_float(zeta_zeros)
    k = min(len(eigvals), len(zeta_zeros))
    if k == 0:
        return float("inf")
    e = normalize(eigvals[:k])
    z = normalize(zeta_zeros[:k])
    return float(np.mean((e - z) ** 2))


def spacing_loss(eigvals, zeta_zeros):
    eigvals = _as_1d_float(eigvals)
    zeta_zeros = _as_1d_float(zeta_zeros)
    e = np.diff(sorted(eigvals))
    z = np.diff(sorted(zeta_zeros))
    k = min(len(e), len(z))
    if k == 0:
        return 0.0
    e = normalize(e)
    z = normalize(z)
    return float(np.mean(np.abs(e[:k] - z[:k])))


def total_loss(eigvals, zeta_zeros):
    loss = spectral_loss(eigvals, zeta_zeros) + 0.5 * spacing_loss(eigvals, zeta_zeros)
    return float(loss)


def spectral_diagnostics(eigvals, zeta_zeros):
    eigvals = _as_1d_float(eigvals)
    zeta_zeros = _as_1d_float(zeta_zeros)
    k = min(len(eigvals), len(zeta_zeros))
    if k == 0:
        return {
            "best_alignment": None,
            "correlation": None,
            "n_compared": 0,
        }

    e = normalize(eigvals[:k])
    z = normalize(zeta_zeros[:k])
    alignment = float(np.min(np.abs(e - z)))
    if k > 1 and np.std(e) > 0.0 and np.std(z) > 0.0:
        correlation = float(np.corrcoef(e, z)[0, 1])
    else:
        correlation = None
    return {
        "best_alignment": alignment,
        "correlation": correlation,
        "n_compared": int(k),
    }
