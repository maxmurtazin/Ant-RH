#!/usr/bin/env python3
"""V13M.1: renormalized self-consistent iteration with zeros_eff < dim and interpolated delta."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.v13l_self_consistent import DTYPE, EPS, rescale_gamma_preserve_order, rescale_gamma_to_range, sym_eps
from core.v13l1_stabilized import (
    beta_schedule,
    clip_v_diag_percentile,
    self_adjointness_fro,
    smooth_delta_multi,
)
from core.v13l2_pareto import pareto_objective_J


def checkpoint_pick_score(
    spectral_log_mse: float,
    spacing_mse_normalized: float,
    ks_wigner: float,
    operator_diff: float,
) -> float:
    """Uncapped Pareto-style score used only to pick ``H_best`` (avoids min(od,1) hiding huge updates)."""
    sl = float(spectral_log_mse) if math.isfinite(float(spectral_log_mse)) else 1e6
    sp = float(spacing_mse_normalized) if math.isfinite(float(spacing_mse_normalized)) else 1e6
    ks = float(ks_wigner) if math.isfinite(float(ks_wigner)) else 1.0
    od = float(operator_diff) if math.isfinite(float(operator_diff)) else 1e6
    return sl + 0.05 * sp + ks + 0.1 * od


def zeros_eff_for_family(family: str, dim: int) -> int:
    """Effective zero count for renormalization family name."""
    d = int(dim)
    fam = str(family).strip().lower()
    if fam == "linear":
        return d
    if fam == "fixed128":
        return int(min(d, 128))
    if fam == "sqrt":
        return int(round(128.0 * math.sqrt(d / 256.0)))
    if fam == "power075":
        return int(round(128.0 * ((d / 256.0) ** 0.75)))
    if fam == "manual":
        if d <= 64:
            return 64
        if d <= 128:
            return 96
        return 128
    raise ValueError(f"unknown zeros family={family!r}")


def lambda_p_scaled(lambda_p_base: float, dim: int, q: float) -> float:
    return float(lambda_p_base) * ((float(dim) / 128.0) ** float(q))


def geo_sigma_scaled(geo_sigma_base: float, dim: int, r: float) -> float:
    return float(geo_sigma_base) * ((float(dim) / 128.0) ** float(r))


def smooth_sigma_scaled(smooth_sigma_base: float, dim: int, s: float) -> float:
    return float(smooth_sigma_base) * ((float(dim) / 128.0) ** float(s))


def interpolate_delta_k_to_dim(delta_k: np.ndarray, dim: int) -> np.ndarray:
    """Map length-k delta to length-dim via linear index interpolation."""
    dk = np.asarray(delta_k, dtype=DTYPE).reshape(-1)
    n = int(dim)
    k = int(dk.size)
    if k <= 0:
        return np.zeros((n,), dtype=DTYPE)
    if k == 1:
        return np.full((n,), float(dk[0]), dtype=DTYPE)
    xp = np.linspace(0.0, float(n - 1), k, dtype=DTYPE)
    xi = np.arange(n, dtype=DTYPE)
    return np.interp(xi, xp, dk.astype(np.float64)).astype(DTYPE, copy=False)


def spectral_metrics_window(
    eig_sorted: np.ndarray,
    z_true_sorted: np.ndarray,
    k_align: int,
    *,
    spacing_fn: Any,
    ks_fn: Any,
    norm_gaps_fn: Any,
) -> Dict[str, float]:
    """Spectral / spacing / KS on first ``k_align`` sorted eig vs sorted true zeros."""
    from core.v13l_self_consistent import spectral_metrics

    e = np.sort(np.asarray(eig_sorted, dtype=DTYPE).reshape(-1))
    z = np.sort(np.asarray(z_true_sorted, dtype=DTYPE).reshape(-1))
    k = int(min(k_align, int(e.size), int(z.size)))
    if k < 1:
        return {
            "spectral_raw_mse": float("nan"),
            "spectral_log_mse": float("nan"),
            "spacing_mse_normalized": float("nan"),
            "ks_wigner": float("nan"),
        }
    return spectral_metrics(e[:k], z[:k], spacing_fn=spacing_fn, ks_fn=ks_fn, norm_gaps_fn=norm_gaps_fn)


def run_renormalized_cell(
    *,
    H_base: np.ndarray,
    z_pool_positive: np.ndarray,
    dim: int,
    zeros_eff: int,
    alpha: float,
    lambda_p_dim: float,
    beta0: float,
    tau_beta: float,
    beta_floor: float,
    smooth_sigma_dim: float,
    clip_percentiles: Tuple[float, float],
    diag_shift: float,
    abs_cap_factor: float,
    zeros_true_for_metrics: np.ndarray,
    k_align: int,
    spacing_fn: Any,
    ks_fn: Any,
    norm_gaps_fn: Any,
    stagnation_window: int = 20,
    stagnation_eps: float = 1e-5,
    stagnation_stop_only_if_operator_diff_below: Optional[float] = None,
    target_positive_ordered: Optional[np.ndarray] = None,
    max_iter: int,
    tol: float,
) -> Dict[str, Any]:
    """
    Self-consistent map with delta built on min(dim, zeros_eff) aligned pairs,
    interpolated to length dim, then triple-Gaussian smooth (σ,σ,σ).

    If ``stagnation_stop_only_if_operator_diff_below`` is set (e.g. ``1e-3`` for V13M.2),
    stagnation-based early exit runs only when ``operator_diff`` is already below that
    threshold; otherwise the stagnation counter is reset and iteration continues.

    If ``target_positive_ordered`` is set (length ``ze``), it replaces the first ``ze`` sorted
    pool zeros for building ``g_short``; values are **rescaled to** ``[lo, hi]`` **without**
    sorting so that ``g_short[i]`` pairs with the ``i``-th smallest eigenvalue (V13O shuffle
    and synthetic targets). Otherwise the pool branch sorts as in V13M.1.
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
    sigmas = (float(smooth_sigma_dim), float(smooth_sigma_dim), float(smooth_sigma_dim))
    clo, chi = float(clip_percentiles[0]), float(clip_percentiles[1])

    if not np.isfinite(H_base).all():
        meta["eig_error"] = "nonfinite_H_base"
        return _empty(H_base, meta)

    try:
        w0 = np.sort(
            np.asarray(np.linalg.eigvalsh(0.5 * (H_base + H_base.T)), dtype=DTYPE).reshape(-1)
        )
    except Exception as ex:
        meta["eig_error"] = repr(ex)
        return _empty(H_base, meta)
    if w0.size != n or not np.isfinite(w0).all():
        meta["eig_error"] = "eigvalsh_failed_initial"
        return _empty(H_base, meta, w0)

    lo, hi = float(w0[0]), float(w0[-1])
    zp = np.asarray(z_pool_positive, dtype=DTYPE).reshape(-1)
    zp = zp[np.isfinite(zp) & (zp > 0.0)]
    zp = np.sort(zp)

    if target_positive_ordered is not None:
        z_ord = np.asarray(target_positive_ordered, dtype=DTYPE).reshape(-1)
        z_ord = z_ord[np.isfinite(z_ord) & (z_ord > 0.0)]
        if z_ord.size < 1:
            meta["eig_error"] = "empty_target_positive_ordered"
            return _empty(H_base, meta, w0)
        ze = int(min(int(zeros_eff), int(z_ord.size)))
        z_ord = z_ord[:ze].astype(DTYPE, copy=False)
        g_short = rescale_gamma_preserve_order(z_ord, lo, hi)
        g_short = np.asarray(g_short, dtype=DTYPE).reshape(-1)
        k_use = int(min(n, int(g_short.size)))
    else:
        ze = int(min(int(zeros_eff), int(zp.size)))
        if ze < 1:
            meta["eig_error"] = "insufficient_zeros"
            return _empty(H_base, meta, w0)

        z_short = zp[:ze].astype(DTYPE, copy=False)
        g_short = rescale_gamma_to_range(z_short, lo, hi)
        g_short = np.sort(np.asarray(g_short, dtype=DTYPE).reshape(-1))
        k_use = int(min(n, ze, g_short.size))

    cap = float(abs_cap_factor * (hi - lo + 1.0) / max(float(alpha) * float(lambda_p_dim), EPS))

    H_t = np.asarray(H_base, dtype=DTYPE, copy=True)
    stag = 0
    prev_sl: Optional[float] = None
    best_J = float("inf")
    best_pick = float("inf")
    best_iter = 0
    best_H = H_t.copy()
    best_row: Dict[str, Any] = {}

    z_metric = np.sort(np.asarray(zeros_true_for_metrics, dtype=DTYPE).reshape(-1))
    z_metric = z_metric[np.isfinite(z_metric) & (z_metric > 0.0)]
    z_metric = np.sort(z_metric)

    for t in range(int(max_iter)):
        beta_t = beta_schedule(t, beta0=beta0, tau=tau_beta, beta_floor=beta_floor)
        try:
            eigvals_t = np.sort(
                np.asarray(np.linalg.eigvalsh(0.5 * (H_t + H_t.T)), dtype=DTYPE).reshape(-1)
            )
        except Exception as ex:
            meta["eig_error"] = repr(ex)
            break
        if eigvals_t.size != n or not np.isfinite(eigvals_t).all():
            meta["eig_error"] = "bad_eigenvalues"
            break

        eig_k = eigvals_t[:k_use]
        g_k = g_short[:k_use]
        delta_k = eig_k - g_k
        delta_dim = interpolate_delta_k_to_dim(delta_k, n)
        delta_s = smooth_delta_multi(delta_dim, sigmas)
        V_diag = -float(alpha) * delta_s
        V_diag = clip_v_diag_percentile(V_diag, clo, chi)
        V_diag = np.clip(V_diag, -cap, cap)
        V_diag = np.asarray(V_diag, dtype=DTYPE).reshape(-1)
        if V_diag.size != n:
            meta["eig_error"] = "V_diag_shape"
            break

        V_t = np.diag(V_diag)
        H_tilde_new = H_base + float(lambda_p_dim) * V_t
        H_new = sym_eps(H_tilde_new, diag_shift)
        operator_diff = float(np.linalg.norm(H_new - H_t, ord="fro"))
        delta_norm = float(np.mean(np.abs(delta_dim)))

        m = spectral_metrics_window(
            eigvals_t,
            z_metric,
            int(k_align),
            spacing_fn=spacing_fn,
            ks_fn=ks_fn,
            norm_gaps_fn=norm_gaps_fn,
        )
        cur_sl = float(m["spectral_log_mse"])
        J = pareto_objective_J(
            m["spectral_log_mse"],
            m["spacing_mse_normalized"],
            m["ks_wigner"],
            operator_diff,
        )
        pick = checkpoint_pick_score(
            m["spectral_log_mse"],
            m["spacing_mse_normalized"],
            m["ks_wigner"],
            operator_diff,
        )

        if math.isfinite(pick) and pick < best_pick:
            best_pick = float(pick)
            best_J = float(J)
            best_iter = int(t)
            best_H = np.asarray(H_t, dtype=DTYPE, copy=True)
            best_row = {
                "spectral_log_mse": float(m["spectral_log_mse"]),
                "spacing_mse_normalized": float(m["spacing_mse_normalized"]),
                "ks_wigner": float(m["ks_wigner"]),
                "operator_diff": float(operator_diff),
                "delta_norm": float(delta_norm),
                "pareto_objective": float(J),
            }

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
                "spectral_log_mse": float(m["spectral_log_mse"]),
                "spacing_mse_normalized": float(m["spacing_mse_normalized"]),
                "ks_wigner": float(m["ks_wigner"]),
                "operator_diff": float(operator_diff),
                "delta_norm": float(delta_norm),
                "eig_min": float(eigvals_t[0]),
                "eig_max": float(eigvals_t[-1]),
                "pareto_objective": float(J),
            }
        )

        if operator_diff < float(tol):
            H_t = (1.0 - beta_t) * H_t + beta_t * H_new
            H_t = 0.5 * (H_t + H_t.T)
            meta["converged_operator"] = True
            meta["n_iter"] = int(t + 1)
            break

        stagnation_break = stag >= int(stagnation_window)
        if stagnation_break and stagnation_stop_only_if_operator_diff_below is not None:
            if operator_diff >= float(stagnation_stop_only_if_operator_diff_below):
                stagnation_break = False
                stag = 0

        if stagnation_break:
            H_t = (1.0 - beta_t) * H_t + beta_t * H_new
            H_t = 0.5 * (H_t + H_t.T)
            meta["stopped_stagnation"] = True
            meta["n_iter"] = int(t + 1)
            break
        H_t = (1.0 - beta_t) * H_t + beta_t * H_new
        H_t = 0.5 * (H_t + H_t.T)
        meta["n_iter"] = int(t + 1)

    meta["self_adjointness_fro_final"] = self_adjointness_fro(H_t)
    final_od = float(rows[-1]["operator_diff"]) if rows else float("nan")
    best_sl = float(best_row.get("spectral_log_mse", float("nan")))
    best_sp = float(best_row.get("spacing_mse_normalized", float("nan")))
    best_ks = float(best_row.get("ks_wigner", float("nan")))
    best_od = float(best_row.get("operator_diff", float("nan")))
    best_dn = float(best_row.get("delta_norm", float("nan")))
    return {
        "H_final": H_t,
        "H_best_J": best_H,
        "rows": rows,
        "meta": meta,
        "best_iter_by_J": int(best_iter),
        "best_J": float(best_J) if math.isfinite(best_J) else float("nan"),
        "best_spectral_log_mse": best_sl,
        "best_spacing_mse_normalized": best_sp,
        "best_ks_wigner": best_ks,
        "best_operator_diff": best_od,
        "best_delta_norm": best_dn,
        "best_row": dict(best_row),
        "final_operator_diff": final_od,
    }


def _empty(H_base: np.ndarray, meta: Dict[str, Any], w0: Optional[np.ndarray] = None) -> Dict[str, Any]:
    nan = float("nan")
    return {
        "H_final": np.asarray(H_base, dtype=DTYPE, copy=True),
        "H_best_J": np.asarray(H_base, dtype=DTYPE, copy=True),
        "rows": [],
        "meta": meta,
        "best_iter_by_J": 0,
        "best_J": float("inf"),
        "best_spectral_log_mse": nan,
        "best_spacing_mse_normalized": nan,
        "best_ks_wigner": nan,
        "best_operator_diff": nan,
        "best_delta_norm": nan,
        "best_row": {},
        "final_operator_diff": float("nan"),
    }
