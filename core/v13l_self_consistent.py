#!/usr/bin/env python3
"""
V13L: fixed-point self-consistent operator H* aligning spectrum to rescaled zeta targets.

Builds H_base = Sym(λ_L L + λ_g K_w) + ε I (no sin potential), then iterates
diagonal spectral feedback V_diag[i] = -α (λ_sorted[i] - γ_scaled[i]).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.artin_operator_word_sensitive import EPS, _build_word_sensitive_core

DTYPE = np.float64


def geodesic_entry_for_word(word: List[int]) -> Dict[str, Any]:
    w = [int(x) for x in word]
    return {
        "a_list": w,
        "length": float(max(1.0, len(w))),
        "is_hyperbolic": True,
        "primitive": True,
    }


def build_h_base_no_potential(
    *,
    z_points: np.ndarray,
    geodesics: List[Dict[str, Any]],
    eps: float = 0.6,
    geo_sigma: float = 0.6,
    geo_weight: float = 10.0,
    laplacian_weight: float = 1.0,
    distances: Optional[np.ndarray] = None,
    diag_shift: float = 1e-6,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """H_base = Sym(λ_L L + λ_g K_norm) + ε I (potential_weight = 0)."""
    C = _build_word_sensitive_core(
        z_points=z_points,
        distances=distances,
        geodesics=geodesics,
        eps=float(eps),
        geo_sigma=float(geo_sigma),
        kernel_normalization="max",
        laplacian_weight=float(laplacian_weight),
        geo_weight=float(geo_weight),
        potential_weight=0.0,
        diag_shift=float(diag_shift),
    )
    H = np.asarray(C["H_final"], dtype=DTYPE, copy=True)
    meta = {
        "fro_H_base": float(np.linalg.norm(H, ord="fro")),
        "fro_H0": float(np.linalg.norm(C["H0"], ord="fro")),
        "fro_G": float(np.linalg.norm(C["G"], ord="fro")),
    }
    return H, meta


def rescale_gamma_to_range(gamma: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Sort positive zeros and linearly map to [lo, hi] (length preserved)."""
    g = np.asarray(gamma, dtype=DTYPE).reshape(-1)
    g = g[np.isfinite(g) & (g > 0.0)]
    g = np.sort(g)
    if g.size == 0:
        return np.zeros((0,), dtype=DTYPE)
    gmin, gmax = float(g[0]), float(g[-1])
    span = gmax - gmin + EPS
    return (lo + (g - gmin) / span * (hi - lo + EPS)).astype(DTYPE, copy=False)


def gaussian_smooth_1d(x: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0 or x.size <= 1:
        return np.asarray(x, dtype=DTYPE, copy=True)
    rad = int(3.0 * float(sigma) + 1.0)
    # Keep kernel length <= len(x) so ``convolve(..., mode="same")`` returns len(x) (NumPy otherwise may lengthen).
    max_rad = max(1, (int(x.size) - 1) // 2)
    rad = max(1, min(int(rad), max_rad))
    k = np.arange(-rad, rad + 1, dtype=DTYPE)
    w = np.exp(-0.5 * (k / (float(sigma) + EPS)) ** 2)
    w = w / (np.sum(w) + EPS)
    return np.convolve(np.asarray(x, dtype=DTYPE), w, mode="same").astype(DTYPE, copy=False)


def sym_eps(H_tilde: np.ndarray, diag_shift: float) -> np.ndarray:
    H_sym = 0.5 * (H_tilde + H_tilde.T)
    n = int(H_sym.shape[0])
    return H_sym + float(diag_shift) * np.eye(n, dtype=DTYPE)


def clip_v_diag(v: np.ndarray, cap: float) -> np.ndarray:
    c = max(float(cap), EPS)
    return np.clip(np.asarray(v, dtype=DTYPE), -c, c)


def spectral_metrics(
    eig: np.ndarray,
    zeros: np.ndarray,
    *,
    spacing_fn: Any,
    ks_fn: Any,
    norm_gaps_fn: Any,
) -> Dict[str, float]:
    """spectral_log_mse, spacing_mse_normalized, ks_wigner vs true zeros (first k)."""
    out = {
        "spectral_raw_mse": float("nan"),
        "spectral_log_mse": float("nan"),
        "spacing_mse_normalized": float("nan"),
        "ks_wigner": float("nan"),
    }
    z = np.asarray(zeros, dtype=DTYPE).reshape(-1)
    z = z[np.isfinite(z)]
    eig = np.sort(np.asarray(eig, dtype=DTYPE).reshape(-1))
    eig = eig[np.isfinite(eig)]
    k = int(min(int(eig.size), int(z.size)))
    if k < 1:
        return out
    e_k = np.sort(eig[:k])
    z_k = np.sort(z[:k])
    raw = float(np.mean((e_k - z_k) ** 2))
    out["spectral_raw_mse"] = raw
    out["spectral_log_mse"] = float(np.log1p(max(0.0, raw)))
    if k >= 2:
        out["spacing_mse_normalized"] = float(spacing_fn(eig[:k], z[:k]))
        nu = norm_gaps_fn(np.sort(eig[:k]))
        if nu.size >= 2:
            out["ks_wigner"] = float(ks_fn(nu))
    return out


def run_self_consistent_loop(
    *,
    H_base: np.ndarray,
    gamma: np.ndarray,
    alpha: float,
    lambda_p: float,
    beta: float,
    max_iter: int,
    tol: float,
    diag_shift: float,
    smooth_sigma: float,
    use_smooth: bool,
    zeros_eval: np.ndarray,
    spacing_fn: Any,
    ks_fn: Any,
    norm_gaps_fn: Any,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Returns (H_star, eigvals_star, metrics_list, run_meta).

    run_meta may include eig_error, converged, n_iter, monotonic_spectral_prefix.
    """
    H_base = np.asarray(H_base, dtype=DTYPE, copy=False)
    n = int(H_base.shape[0])
    metrics_list: List[Dict[str, Any]] = []
    run_meta: Dict[str, Any] = {
        "converged": False,
        "n_iter": 0,
        "eig_error": None,
        "monotonic_spectral_prefix": False,
        "monotonic_prefix_len": 0,
    }

    if not np.isfinite(H_base).all():
        run_meta["eig_error"] = "nonfinite_H_base"
        return H_base, np.array([], dtype=DTYPE), metrics_list, run_meta

    w0, err0 = _eigvals_sorted(H_base)
    if err0 or w0.size != n:
        run_meta["eig_error"] = err0 or "eigvalsh_failed_initial"
        return H_base, w0, metrics_list, run_meta

    lo, hi = float(w0[0]), float(w0[-1])
    g_sorted = rescale_gamma_to_range(gamma, lo, hi)
    if g_sorted.size < n:
        run_meta["eig_error"] = "insufficient_zeros"
        return H_base, w0, metrics_list, run_meta
    gamma_scaled = g_sorted[:n].astype(DTYPE, copy=False)
    gamma_scaled = np.sort(gamma_scaled)

    H_t = np.asarray(H_base, dtype=DTYPE, copy=True)
    spectral_log_series: List[float] = []

    eig_init = np.sort(np.linalg.eigvalsh(0.5 * (H_t + H_t.T)))
    m0 = spectral_metrics(eig_init, zeros_eval, spacing_fn=spacing_fn, ks_fn=ks_fn, norm_gaps_fn=norm_gaps_fn)

    cap = float(5.0 * (hi - lo + 1.0) / max(float(alpha) * float(lambda_p), EPS))

    for t in range(int(max_iter)):
        try:
            eigvals_t = np.linalg.eigvalsh(0.5 * (H_t + H_t.T))
        except Exception as ex:
            run_meta["eig_error"] = repr(ex)
            break
        if not np.isfinite(eigvals_t).all():
            run_meta["eig_error"] = "nonfinite_eigenvalues"
            break
        eigvals_t = np.sort(eigvals_t.astype(DTYPE, copy=False))

        delta = eigvals_t - gamma_scaled
        V_diag = -float(alpha) * delta
        if use_smooth and smooth_sigma > 0:
            V_diag = gaussian_smooth_1d(V_diag, float(smooth_sigma))
        V_diag = clip_v_diag(V_diag, cap)

        V_t = np.diag(V_diag.astype(DTYPE, copy=False))
        H_tilde_new = H_base + float(lambda_p) * V_t
        H_new = sym_eps(H_tilde_new, diag_shift)

        operator_diff = float(np.linalg.norm(H_new - H_t, ord="fro"))
        delta_norm = float(np.mean(np.abs(delta)))

        m = spectral_metrics(eigvals_t, zeros_eval, spacing_fn=spacing_fn, ks_fn=ks_fn, norm_gaps_fn=norm_gaps_fn)
        spectral_log_series.append(m["spectral_log_mse"])

        metrics_list.append(
            {
                "iter": int(t),
                "spectral_raw_mse": m["spectral_raw_mse"],
                "spectral_log_mse": m["spectral_log_mse"],
                "spacing_mse_normalized": m["spacing_mse_normalized"],
                "ks_wigner": m["ks_wigner"],
                "delta_norm": delta_norm,
                "operator_diff": operator_diff,
            }
        )

        if operator_diff < float(tol):
            H_t = (1.0 - float(beta)) * H_t + float(beta) * H_new
            H_t = 0.5 * (H_t + H_t.T)
            run_meta["converged"] = True
            run_meta["n_iter"] = int(t + 1)
            break

        H_t = (1.0 - float(beta)) * H_t + float(beta) * H_new
        H_t = 0.5 * (H_t + H_t.T)

        run_meta["n_iter"] = int(t + 1)

    eig_star = np.sort(np.linalg.eigvalsh(0.5 * (H_t + H_t.T)).astype(DTYPE, copy=False))
    prefix = _monotonic_improving_prefix(spectral_log_series, min_len=3)
    run_meta["monotonic_spectral_prefix"] = bool(prefix[0])
    run_meta["monotonic_prefix_len"] = int(prefix[1])

    return H_t, eig_star, metrics_list, run_meta


def _eigvals_sorted(H: np.ndarray) -> Tuple[np.ndarray, Optional[str]]:
    H = np.asarray(H, dtype=DTYPE, copy=False)
    if not np.isfinite(H).all():
        return np.array([], dtype=DTYPE), "nonfinite_matrix"
    try:
        w = np.linalg.eigvalsh(0.5 * (H + H.T))
    except Exception as ex:
        return np.array([], dtype=DTYPE), repr(ex)
    w = np.sort(w.astype(DTYPE, copy=False))
    if not np.isfinite(w).all():
        return np.array([], dtype=DTYPE), "nonfinite_eigenvalues"
    return w, None


def _monotonic_improving_prefix(series: List[float], min_len: int = 3) -> Tuple[bool, int]:
    """
    True if first ``min_len`` spectral_log_mse values are non-increasing (lower is better),
    with a net drop from index 0 to min_len-1 (allows short plateaus).
    """
    if len(series) < min_len:
        return False, len(series)
    for i in range(min_len - 1):
        a, b = float(series[i]), float(series[i + 1])
        if not (math.isfinite(a) and math.isfinite(b)) or a < b:
            return False, i + 1
    a0 = float(series[0])
    a2 = float(series[min_len - 1])
    if not (math.isfinite(a0) and math.isfinite(a2)) or a0 <= a2:
        return False, min_len
    return True, min_len
