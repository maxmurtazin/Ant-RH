#!/usr/bin/env python3
"""V13L.1: stabilized self-consistent fixed-point iteration (scheduled beta, clipping, delta smoothing)."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.v13l_self_consistent import (
    DTYPE,
    EPS,
    build_h_base_no_potential,
    gaussian_smooth_1d,
    rescale_gamma_to_range,
    spectral_metrics,
    sym_eps,
)


def beta_schedule(t: int, *, beta0: float = 0.3, tau: float = 50.0, beta_floor: float = 0.05) -> float:
    """beta_t = max(beta_floor, beta0 * exp(-t / tau))."""
    return max(float(beta_floor), float(beta0) * math.exp(-float(t) / max(float(tau), EPS)))


def clip_v_diag_percentile(v: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0) -> np.ndarray:
    """Clip vector to [percentile_lo, percentile_hi] of its own distribution."""
    a = np.asarray(v, dtype=DTYPE).reshape(-1)
    if a.size < 2:
        return a.copy()
    lo = float(np.percentile(a, float(lo_pct)))
    hi = float(np.percentile(a, float(hi_pct)))
    if not (math.isfinite(lo) and math.isfinite(hi)) or lo >= hi:
        return np.clip(a, float(np.min(a)), float(np.max(a)))
    return np.clip(a, lo, hi).astype(DTYPE, copy=False)


def smooth_delta_multi(delta: np.ndarray, sigmas: Tuple[float, ...]) -> np.ndarray:
    """Apply Gaussian smoothing with each sigma in sequence (e.g. 2, 4, 8)."""
    d = np.asarray(delta, dtype=DTYPE, copy=True).reshape(-1)
    for s in sigmas:
        d = gaussian_smooth_1d(d, float(s))
    return d.astype(DTYPE, copy=False)


def self_adjointness_fro(H: np.ndarray) -> float:
    """Frobenius norm of skew-symmetric part (0 for exact symmetry)."""
    A = np.asarray(H, dtype=DTYPE, copy=False)
    S = 0.5 * (A - A.T)
    return float(np.linalg.norm(S, ord="fro"))


def run_stabilized_self_consistent(
    *,
    H_base: np.ndarray,
    gamma: np.ndarray,
    alpha: float,
    lambda_p: float,
    max_iter: int,
    tol: float,
    diag_shift: float,
    beta0: float = 0.3,
    tau_beta: float = 50.0,
    beta_floor: float = 0.05,
    delta_smooth_sigmas: Tuple[float, ...] = (2.0, 4.0, 8.0),
    clip_percentiles: Tuple[float, float] = (1.0, 99.0),
    stagnation_window: int = 20,
    stagnation_eps: float = 1e-5,
    abs_cap_factor: float = 5.0,
    zeros_eval: np.ndarray,
    spacing_fn: Any,
    ks_fn: Any,
    norm_gaps_fn: Any,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Anderson-style map: H_new = Sym(H_base + λ_p diag(V)); update H ← (1-β_t) H + β_t H_new.

    Returns (H_star, eigvals_sorted, per-iter rows, run_meta).
    """
    H_base = np.asarray(H_base, dtype=DTYPE, copy=False)
    n = int(H_base.shape[0])
    rows: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {
        "converged_operator": False,
        "stopped_stagnation": False,
        "n_iter": 0,
        "eig_error": None,
    }

    if not np.isfinite(H_base).all():
        meta["eig_error"] = "nonfinite_H_base"
        return H_base, np.array([], dtype=DTYPE), rows, meta

    try:
        w0 = np.sort(
            np.asarray(np.linalg.eigvalsh(0.5 * (H_base + H_base.T)), dtype=DTYPE).reshape(-1)
        )
    except Exception as ex:
        meta["eig_error"] = repr(ex)
        return H_base, np.array([], dtype=DTYPE), rows, meta
    if w0.size != n or not np.isfinite(w0).all():
        meta["eig_error"] = "eigvalsh_failed_initial"
        return H_base, w0, rows, meta

    lo, hi = float(w0[0]), float(w0[-1])
    g_sorted = rescale_gamma_to_range(gamma, lo, hi)
    if g_sorted.size < n:
        meta["eig_error"] = "insufficient_zeros"
        return H_base, w0, rows, meta
    gamma_scaled = np.sort(np.asarray(g_sorted[:n], dtype=DTYPE).reshape(-1))
    if gamma_scaled.size != n:
        meta["eig_error"] = "gamma_scaled_init_mismatch"
        return H_base, w0, rows, meta

    cap = float(abs_cap_factor * (hi - lo + 1.0) / max(float(alpha) * float(lambda_p), EPS))

    H_t = np.asarray(H_base, dtype=DTYPE, copy=True)
    stag = 0
    prev_sl: Optional[float] = None

    for t in range(int(max_iter)):
        beta_t = beta_schedule(t, beta0=beta0, tau=tau_beta, beta_floor=beta_floor)

        try:
            eigvals_t = np.asarray(
                np.linalg.eigvalsh(0.5 * (H_t + H_t.T)), dtype=DTYPE, order="C"
            ).reshape(-1)
            eigvals_t = np.sort(eigvals_t)
        except Exception as ex:
            meta["eig_error"] = repr(ex)
            break
        if eigvals_t.size != n or not np.isfinite(eigvals_t).all():
            meta["eig_error"] = "bad_eigenvalue_count_or_nonfinite"
            break

        delta = eigvals_t - gamma_scaled
        delta_s = smooth_delta_multi(delta, delta_smooth_sigmas)
        V_diag = -float(alpha) * delta_s
        V_diag = clip_v_diag_percentile(V_diag, float(clip_percentiles[0]), float(clip_percentiles[1]))
        V_diag = np.clip(V_diag, -cap, cap)
        V_diag = np.asarray(V_diag, dtype=DTYPE).reshape(-1)
        if V_diag.size != n:
            meta["eig_error"] = "V_diag_shape_mismatch"
            break

        V_t = np.diag(V_diag.astype(DTYPE, copy=False))
        H_tilde_new = H_base + float(lambda_p) * V_t
        H_new = sym_eps(H_tilde_new, diag_shift)

        operator_diff = float(np.linalg.norm(H_new - H_t, ord="fro"))
        delta_norm = float(np.mean(np.abs(delta)))

        m = spectral_metrics(eigvals_t, zeros_eval, spacing_fn=spacing_fn, ks_fn=ks_fn, norm_gaps_fn=norm_gaps_fn)
        cur_sl = float(m["spectral_log_mse"])

        if prev_sl is not None:
            imp = prev_sl - cur_sl
            if imp >= 0.0 and imp < float(stagnation_eps):
                stag += 1
            else:
                stag = 0
        prev_sl = cur_sl

        rows.append(
            {
                "iter": int(t),
                "beta_t": float(beta_t),
                "spectral_log_mse": m["spectral_log_mse"],
                "spacing_mse_normalized": m["spacing_mse_normalized"],
                "ks_wigner": m["ks_wigner"],
                "delta_norm": delta_norm,
                "operator_diff": operator_diff,
                "eig_min": float(eigvals_t[0]),
                "eig_max": float(eigvals_t[-1]),
            }
        )

        if operator_diff < float(tol):
            H_t = (1.0 - beta_t) * H_t + beta_t * H_new
            H_t = 0.5 * (H_t + H_t.T)
            meta["converged_operator"] = True
            meta["n_iter"] = int(t + 1)
            break

        if stag >= int(stagnation_window):
            H_t = (1.0 - beta_t) * H_t + beta_t * H_new
            H_t = 0.5 * (H_t + H_t.T)
            meta["stopped_stagnation"] = True
            meta["n_iter"] = int(t + 1)
            break

        H_t = (1.0 - beta_t) * H_t + beta_t * H_new
        H_t = 0.5 * (H_t + H_t.T)
        meta["n_iter"] = int(t + 1)

    eig_star = np.sort(np.linalg.eigvalsh(0.5 * (H_t + H_t.T)).astype(DTYPE, copy=False))
    meta["self_adjointness_fro_final"] = self_adjointness_fro(H_t)
    return H_t, eig_star, rows, meta
