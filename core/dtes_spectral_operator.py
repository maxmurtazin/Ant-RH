from __future__ import annotations

"""Self-adjoint DTES spectral operator utilities.

The operator is a numerical Hilbert-Polya inspired diagnostic built from a
DTES pheromone graph and a zeta-derived potential. It is not a proof tool.
"""

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np


def _as_1d_float(name: str, values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or inf")
    return arr


def _as_square_float(name: str, values: Any) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(f"{name} must be a square matrix")
    if arr.shape[0] == 0:
        raise ValueError(f"{name} must be non-empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or inf")
    return arr


def build_dtes_operator(
    t_grid,
    zeta_abs,
    pheromone,
    eps: float = 1e-12,
    potential_mode: str = "neglog",
    normalize_laplacian: bool = False,
) -> np.ndarray:
    """Build a symmetric DTES Schrödinger-style graph operator.

    H = L + diag(V), where L is either the standard or normalized graph
    Laplacian of the symmetrized pheromone graph.
    """
    t = _as_1d_float("t_grid", t_grid)
    z = _as_1d_float("zeta_abs", zeta_abs)
    W = _as_square_float("pheromone", pheromone)
    if z.size != t.size or W.shape[0] != t.size:
        raise ValueError(
            "t_grid, zeta_abs, and pheromone dimensions must agree "
            f"(got {t.size}, {z.size}, {W.shape})"
        )

    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)
    W = np.maximum(W, 0.0)

    degree = np.sum(W, axis=1)
    if normalize_laplacian:
        inv_sqrt_degree = np.zeros_like(degree)
        mask = degree > eps
        inv_sqrt_degree[mask] = 1.0 / np.sqrt(degree[mask])
        L = np.eye(W.shape[0]) - (inv_sqrt_degree[:, None] * W * inv_sqrt_degree[None, :])
    else:
        L = np.diag(degree) - W

    if potential_mode == "neglog":
        V = -np.log(z + eps)
    elif potential_mode == "log":
        V = np.log(z + eps)
    elif potential_mode == "inverse":
        V = 1.0 / (z + eps)
    else:
        raise ValueError(
            "potential_mode must be one of: 'neglog', 'log', or 'inverse'"
        )

    V = np.asarray(V, dtype=float)
    V = V - float(np.mean(V))
    std = float(np.std(V))
    if std > 0.0:
        V = V / std

    H = L + np.diag(V)
    H = 0.5 * (H + H.T)
    if not np.all(np.isfinite(H)):
        raise ValueError("DTES operator contains NaN or inf")
    return H


def compute_spectrum(H, k: int = 50) -> np.ndarray:
    H_arr = _as_square_float("H", H)
    H_arr = 0.5 * (H_arr + H_arr.T)
    eigvals = np.linalg.eigh(H_arr)[0]
    eigvals = np.sort(np.real(eigvals))
    if k is None or k <= 0:
        return eigvals
    return eigvals[: min(int(k), eigvals.size)]


def unfold_spectrum(x) -> np.ndarray:
    arr = np.sort(_as_1d_float("spectrum", x))
    if arr.size <= 1:
        return arr.astype(float)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if std <= 0.0:
        return arr - mean
    return (arr - mean) / std


def spacing_distribution(x) -> np.ndarray:
    unfolded = unfold_spectrum(x)
    if unfolded.size < 2:
        return np.array([], dtype=float)
    spacings = np.diff(np.sort(unfolded))
    mean_spacing = float(np.mean(spacings))
    if mean_spacing > 0.0:
        spacings = spacings / mean_spacing
    return spacings


def spectral_alignment_loss(eigvals, zeta_zeros) -> Dict[str, Any]:
    e = unfold_spectrum(eigvals)
    z = unfold_spectrum(zeta_zeros)
    n = min(e.size, z.size)
    if n == 0:
        return {
            "mae": None,
            "rmse": None,
            "correlation": None,
            "spacing_mae": None,
            "n_compared": 0,
        }

    e_cmp = e[:n]
    z_cmp = z[:n]
    diff = e_cmp - z_cmp
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    if n > 1 and float(np.std(e_cmp)) > 0.0 and float(np.std(z_cmp)) > 0.0:
        corr = float(np.corrcoef(e_cmp, z_cmp)[0, 1])
    else:
        corr = None

    e_spacing = spacing_distribution(e_cmp)
    z_spacing = spacing_distribution(z_cmp)
    ns = min(e_spacing.size, z_spacing.size)
    spacing_mae = (
        float(np.mean(np.abs(e_spacing[:ns] - z_spacing[:ns]))) if ns > 0 else None
    )
    return {
        "mae": mae,
        "rmse": rmse,
        "correlation": corr,
        "spacing_mae": spacing_mae,
        "n_compared": int(n),
    }


def compare_spectral_statistics(eigvals, zeta_zeros) -> Dict[str, Any]:
    report = spectral_alignment_loss(eigvals, zeta_zeros)
    e_spacing = spacing_distribution(eigvals)
    z_spacing = spacing_distribution(zeta_zeros)
    report.update(
        {
            "eigenvalue_count": int(np.asarray(eigvals).size),
            "zeta_zero_count": int(np.asarray(zeta_zeros).size),
            "eig_spacing_mean": float(np.mean(e_spacing)) if e_spacing.size else None,
            "eig_spacing_std": float(np.std(e_spacing)) if e_spacing.size else None,
            "zero_spacing_mean": float(np.mean(z_spacing)) if z_spacing.size else None,
            "zero_spacing_std": float(np.std(z_spacing)) if z_spacing.size else None,
        }
    )
    return report


def save_spectral_report(report: Dict[str, Any], out_path) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
