#!/usr/bin/env python3
"""V13O.1: train/test OOS helpers and renormalized self-consistent loop with potential variants."""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from core.v13l_self_consistent import DTYPE, EPS, rescale_gamma_preserve_order, rescale_gamma_to_range, sym_eps
from core.v13l1_stabilized import beta_schedule, clip_v_diag_percentile, smooth_delta_multi
from core.v13l2_pareto import pareto_objective_J
from core.v13m1_renormalized import interpolate_delta_k_to_dim, spectral_metrics_window


def pareto_objective_j(
    spectral_log_mse: float,
    spacing_mse_normalized: float,
    ks_wigner: float,
    operator_diff: float,
) -> float:
    return float(
        pareto_objective_J(
            spectral_log_mse,
            spacing_mse_normalized,
            ks_wigner,
            operator_diff,
        )
    )


def rescale_targets_to_range(z_pos: np.ndarray, lo: float, hi: float, *, ordered: bool) -> np.ndarray:
    z = np.asarray(z_pos, dtype=DTYPE).reshape(-1)
    z = z[np.isfinite(z) & (z > 0.0)]
    if z.size == 0:
        return np.zeros((0,), dtype=DTYPE)
    if ordered:
        return rescale_gamma_preserve_order(z, lo, hi)
    return rescale_gamma_to_range(z, lo, hi)


def project_v_lowfreq_first_m(v: np.ndarray, m: int) -> np.ndarray:
    """Project ``v`` onto first ``m`` DCT-II type cosine modes (least-squares), length preserved."""
    n = int(v.size)
    m = int(max(1, min(m, n)))
    x = np.arange(n, dtype=np.float64)
    A = np.zeros((n, m), dtype=np.float64)
    for j in range(m):
        A[:, j] = np.cos(math.pi * j * (x + 0.5) / n)
    coef, _, _, _ = np.linalg.lstsq(A, np.asarray(v, dtype=np.float64), rcond=None)
    return (A @ coef).astype(DTYPE, copy=False)


def target_blind_v_diag(H_base: np.ndarray, smooth_sigma: float, alpha: float) -> np.ndarray:
    """Experimental: smooth of absolute diagonal of ``H_base`` (no zeta), scaled like a potential."""
    d = np.abs(np.diag(np.asarray(H_base, dtype=DTYPE))).astype(DTYPE, copy=False)
    sigmas = (float(smooth_sigma), float(smooth_sigma), float(smooth_sigma))
    d_s = smooth_delta_multi(d, sigmas)
    return (-float(alpha) * d_s).astype(DTYPE, copy=False)


def train_renormalized_with_variant(
    *,
    H_base: np.ndarray,
    z_pool_sorted: np.ndarray,
    train_ordinates: np.ndarray,
    train_ordered: bool,
    dim: int,
    k_train: int,
    alpha: float,
    lambda_p_dim: float,
    beta0: float,
    tau_beta: float,
    beta_floor: float,
    smooth_sigma_dim: float,
    clip_lo: float,
    clip_hi: float,
    diag_shift: float,
    abs_cap_factor: float,
    zeros_train_metric: np.ndarray,
    spacing_fn: Any,
    ks_fn: Any,
    norm_gaps_fn: Any,
    max_iter: int,
    tol: float,
    variant: str,
    freeze_v_after: Optional[int],
    target_blind: bool,
    on_train_iter: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """
    Self-consistent train on train-window targets. Returns ``H_final``, ``H_base``, last ``V_diag``,
    per-iter rows, meta, and train-window metrics on final ``H``.
    """
    H_base = np.asarray(H_base, dtype=DTYPE, copy=False)
    n = int(H_base.shape[0])
    rows: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {"converged_operator": False, "n_iter": 0, "eig_error": None}
    sigmas = (float(smooth_sigma_dim), float(smooth_sigma_dim), float(smooth_sigma_dim))

    if not np.isfinite(H_base).all():
        meta["eig_error"] = "nonfinite_H_base"
        return _fail_train(H_base, meta)

    try:
        w0 = np.sort(np.asarray(np.linalg.eigvalsh(0.5 * (H_base + H_base.T)), dtype=DTYPE).reshape(-1))
    except Exception as ex:
        meta["eig_error"] = repr(ex)
        return _fail_train(H_base, meta)
    if w0.size != n or not np.isfinite(w0).all():
        meta["eig_error"] = "eigvalsh_failed_initial"
        return _fail_train(H_base, meta)

    lo, hi = float(w0[0]), float(w0[-1])
    zt = np.asarray(train_ordinates, dtype=DTYPE).reshape(-1)
    zt = zt[np.isfinite(zt) & (zt > 0.0)]
    if zt.size < int(k_train):
        meta["eig_error"] = "insufficient_train_ordinates"
        return _fail_train(H_base, meta)
    zt = zt[: int(k_train)].astype(DTYPE, copy=False)
    g_short = rescale_targets_to_range(zt, lo, hi, ordered=train_ordered)
    g_short = np.asarray(g_short, dtype=DTYPE).reshape(-1)
    k_use = int(min(n, int(g_short.size)))
    cap = float(abs_cap_factor * (hi - lo + 1.0) / max(float(alpha) * float(lambda_p_dim), EPS))

    H_t = np.asarray(H_base, dtype=DTYPE, copy=True)
    V_stash_for_freeze: Optional[np.ndarray] = None
    freeze_after = int(freeze_v_after) if freeze_v_after is not None else None
    V_last_applied = np.zeros((n,), dtype=DTYPE)

    z_metric_train = np.sort(np.asarray(zeros_train_metric, dtype=DTYPE).reshape(-1))
    z_metric_train = z_metric_train[np.isfinite(z_metric_train) & (z_metric_train > 0.0)]
    z_metric_train = np.sort(z_metric_train)
    k_align = int(min(int(dim), int(k_train), int(z_metric_train.size)))

    for t in range(int(max_iter)):
        beta_t = beta_schedule(t, beta0=beta0, tau=tau_beta, beta_floor=beta_floor)
        try:
            eigvals_t = np.sort(np.asarray(np.linalg.eigvalsh(0.5 * (H_t + H_t.T)), dtype=DTYPE).reshape(-1))
        except Exception as ex:
            meta["eig_error"] = repr(ex)
            break
        if eigvals_t.size != n or not np.isfinite(eigvals_t).all():
            meta["eig_error"] = "bad_eigenvalues"
            break

        if target_blind:
            V_diag = target_blind_v_diag(H_base, float(smooth_sigma_dim), float(alpha))
            V_diag = clip_v_diag_percentile(V_diag, float(clip_lo), float(clip_hi))
            V_diag = np.clip(V_diag, -cap, cap)
        else:
            eig_k = eigvals_t[:k_use]
            g_k = g_short[:k_use]
            if not train_ordered:
                g_k = np.sort(g_k)
            delta_k = eig_k - g_k
            delta_dim = interpolate_delta_k_to_dim(delta_k, n)
            if variant == "lowfreq_V":
                m = int(max(8, int(math.sqrt(n))))
                delta_s = smooth_delta_multi(delta_dim, sigmas)
                delta_s = project_v_lowfreq_first_m(delta_s, m)
            else:
                delta_s = smooth_delta_multi(delta_dim, sigmas)
            V_diag = -float(alpha) * delta_s
            V_diag = clip_v_diag_percentile(V_diag, float(clip_lo), float(clip_hi))
            V_diag = np.clip(V_diag, -cap, cap)
        V_diag = np.asarray(V_diag, dtype=DTYPE).reshape(-1)

        if freeze_after is not None and freeze_after > 0:
            if t == freeze_after - 1:
                V_stash_for_freeze = np.asarray(V_diag, dtype=DTYPE, copy=True)
            if t >= freeze_after and V_stash_for_freeze is not None:
                V_diag = np.asarray(V_stash_for_freeze, dtype=DTYPE, copy=True)

        V_last_applied = np.asarray(V_diag, dtype=DTYPE, copy=True)
        V_t = np.diag(V_diag)
        H_tilde_new = H_base + float(lambda_p_dim) * V_t
        H_new = sym_eps(H_tilde_new, diag_shift)
        operator_diff = float(np.linalg.norm(H_new - H_t, ord="fro"))

        m = spectral_metrics_window(
            eigvals_t,
            z_metric_train,
            int(k_align),
            spacing_fn=spacing_fn,
            ks_fn=ks_fn,
            norm_gaps_fn=norm_gaps_fn,
        )
        J = pareto_objective_j(
            m["spectral_log_mse"],
            m["spacing_mse_normalized"],
            m["ks_wigner"],
            operator_diff,
        )
        row_iter = {
            "iter": t,
            "spectral_log_mse": float(m["spectral_log_mse"]),
            "spacing_mse_normalized": float(m["spacing_mse_normalized"]),
            "ks_wigner": float(m["ks_wigner"]),
            "operator_diff": operator_diff,
            "pareto_objective": float(J),
        }
        rows.append(dict(row_iter))

        if on_train_iter is not None:
            on_train_iter(int(t), int(max_iter), dict(row_iter))

        if operator_diff < float(tol):
            H_t = (1.0 - beta_t) * H_t + beta_t * H_new
            H_t = 0.5 * (H_t + H_t.T)
            meta["converged_operator"] = True
            meta["n_iter"] = t + 1
            break

        H_t = (1.0 - beta_t) * H_t + beta_t * H_new
        H_t = 0.5 * (H_t + H_t.T)
        meta["n_iter"] = t + 1

    if not rows:
        rN = {}
        sl = sp = ks = float("nan")
        od = float("nan")
        J = float("nan")
    else:
        rN = rows[-1]
        sl = float(rN["spectral_log_mse"])
        sp = float(rN["spacing_mse_normalized"])
        ks = float(rN["ks_wigner"])
        od = float(rN["operator_diff"])
        J = float(rN["pareto_objective"])

    V_last = V_last_applied if rows else np.zeros((n,), dtype=DTYPE)

    return {
        "H_final": H_t,
        "H_base": np.asarray(H_base, dtype=DTYPE, copy=True),
        "V_diag_last": V_last,
        "rows": rows,
        "meta": meta,
        "spectral_log_mse_train": sl,
        "spacing_mse_normalized_train": sp,
        "ks_wigner_train": ks,
        "pareto_objective_train": J,
        "operator_diff_final": od,
        "final_row": dict(rN) if rows else {},
    }


def _fail_train(H_base: np.ndarray, meta: Dict[str, Any]) -> Dict[str, Any]:
    n = int(H_base.shape[0])
    return {
        "H_final": np.asarray(H_base, dtype=DTYPE, copy=True),
        "H_base": np.asarray(H_base, dtype=DTYPE, copy=True),
        "V_diag_last": np.zeros((n,), dtype=DTYPE),
        "rows": [],
        "meta": meta,
        "spectral_log_mse_train": float("nan"),
        "spacing_mse_normalized_train": float("nan"),
        "ks_wigner_train": float("nan"),
        "pareto_objective_train": float("nan"),
        "operator_diff_final": float("nan"),
        "final_row": {},
    }


def eval_on_window(
    *,
    H: np.ndarray,
    H_base: np.ndarray,
    z_window_sorted: np.ndarray,
    k_align: int,
    dim: int,
    k_train: int,
    alpha: float,
    lambda_p_dim: float,
    smooth_sigma_dim: float,
    clip_lo: float,
    clip_hi: float,
    diag_shift: float,
    abs_cap_factor: float,
    spacing_fn: Any,
    ks_fn: Any,
    norm_gaps_fn: Any,
    mode: str,
    V_fixed: Optional[np.ndarray],
) -> Tuple[Dict[str, float], float]:
    """
    ``mode``: ``full_H`` | ``fixed_V`` | ``fixed_K_update_V_test``.
    Returns (metrics dict, operator_diff vs previous H for update modes).
    """
    H = np.asarray(H, dtype=DTYPE, copy=False)
    n = int(H.shape[0])
    H_base = np.asarray(H_base, dtype=DTYPE, copy=False)
    zw = np.asarray(z_window_sorted, dtype=DTYPE).reshape(-1)
    zw = zw[np.isfinite(zw) & (zw > 0.0)]
    zw = np.sort(zw)[: int(k_train)]

    eig0 = np.sort(np.linalg.eigvalsh(0.5 * (H + H.T)).astype(DTYPE))
    lo, hi = float(eig0[0]), float(eig0[-1])
    g_short = rescale_gamma_to_range(zw, lo, hi)
    g_short = np.sort(np.asarray(g_short, dtype=DTYPE).reshape(-1))
    k_use = int(min(n, int(g_short.size), int(k_train)))
    cap = float(abs_cap_factor * (hi - lo + 1.0) / max(float(alpha) * float(lambda_p_dim), EPS))
    sigmas = (float(smooth_sigma_dim),) * 3

    od = float("nan")
    H_eval = H

    if mode == "full_H":
        H_eval = H
    elif mode == "fixed_V":
        if V_fixed is None or V_fixed.size != n:
            H_eval = H
        else:
            Vd = np.asarray(V_fixed, dtype=DTYPE).reshape(-1)
            H_eval = sym_eps(H_base + float(lambda_p_dim) * np.diag(Vd), diag_shift)
            od = float(np.linalg.norm(H_eval - H, ord="fro"))
    elif mode == "fixed_K_update_V_test":
        eigvals_t = np.sort(np.asarray(np.linalg.eigvalsh(0.5 * (H + H.T)), dtype=DTYPE).reshape(-1))
        eig_k = eigvals_t[:k_use]
        g_k = g_short[:k_use]
        delta_k = eig_k - g_k
        delta_dim = interpolate_delta_k_to_dim(delta_k, n)
        delta_s = smooth_delta_multi(delta_dim, sigmas)
        V_diag = -float(alpha) * delta_s
        V_diag = clip_v_diag_percentile(V_diag, float(clip_lo), float(clip_hi))
        V_diag = np.clip(V_diag, -cap, cap)
        H_new = sym_eps(H_base + float(lambda_p_dim) * np.diag(V_diag), diag_shift)
        od = float(np.linalg.norm(H_new - H, ord="fro"))
        H_eval = H_new
    else:
        H_eval = H

    m = spectral_metrics_window(
        np.sort(np.linalg.eigvalsh(0.5 * (H_eval + H_eval.T))).astype(DTYPE),
        zw,
        int(k_align),
        spacing_fn=spacing_fn,
        ks_fn=ks_fn,
        norm_gaps_fn=norm_gaps_fn,
    )
    return m, od


def self_adjointness_fro(H: np.ndarray) -> float:
    from core.v13l1_stabilized import self_adjointness_fro as saf

    return float(saf(H))
