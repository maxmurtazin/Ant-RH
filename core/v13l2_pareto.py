#!/usr/bin/env python3
"""V13L.2: Pareto-weighted stabilized self-consistent iteration with best-J checkpoint tracking."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.v13l_self_consistent import (
    DTYPE,
    EPS,
    rescale_gamma_to_range,
    spectral_metrics,
    sym_eps,
)
from core.v13l1_stabilized import (
    beta_schedule,
    clip_v_diag_percentile,
    self_adjointness_fro,
    smooth_delta_multi,
)


def pareto_objective_J(
    spectral_log_mse: float,
    spacing_mse_normalized: float,
    ks_wigner: float,
    operator_diff: float,
) -> float:
    """
    J = spectral_log_mse + 0.05 * spacing_mse_normalized + ks_wigner + 0.1 * min(operator_diff, 1.0).
    Non-finite pieces are penalized with large constants so J remains comparable for minimization.
    """
    sl = float(spectral_log_mse) if math.isfinite(float(spectral_log_mse)) else 1e6
    sp = float(spacing_mse_normalized) if math.isfinite(float(spacing_mse_normalized)) else 1e6
    ks = float(ks_wigner) if math.isfinite(float(ks_wigner)) else 1.0
    od = float(operator_diff) if math.isfinite(float(operator_diff)) else 1.0
    return sl + 0.05 * sp + ks + 0.1 * min(od, 1.0)


def meets_acceptance(
    *,
    operator_diff: float,
    spectral_log_mse: float,
    spacing_mse_normalized: float,
    ks_wigner: float,
    od_max: float = 1e-3,
    sl_max: float = 7.48,
    sp_max: float = 16.84,
    ks_max: float = 0.451,
) -> bool:
    return bool(
        math.isfinite(operator_diff)
        and math.isfinite(spectral_log_mse)
        and math.isfinite(spacing_mse_normalized)
        and math.isfinite(ks_wigner)
        and operator_diff <= od_max
        and spectral_log_mse <= sl_max
        and spacing_mse_normalized <= sp_max
        and ks_wigner <= ks_max
    )


def run_pareto_cell(
    *,
    H_base: np.ndarray,
    gamma: np.ndarray,
    alpha: float,
    lambda_p_eff: float,
    beta0: float,
    tau_beta: float,
    beta_floor: float,
    smooth_sigma: float,
    clip_percentiles: Tuple[float, float],
    max_iter: int,
    tol: float,
    diag_shift: float,
    abs_cap_factor: float,
    zeros_eval: np.ndarray,
    spacing_fn: Any,
    ks_fn: Any,
    norm_gaps_fn: Any,
    stagnation_window: int = 20,
    stagnation_eps: float = 1e-5,
) -> Dict[str, Any]:
    """
    Same map as V13L.1 with tunable schedule, triple delta smooth (σ,σ,σ), and percentile clip.
    Tracks operator snapshot ``H_t`` at the iteration that minimizes ``J`` (among finite J).
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
    sigmas = (float(smooth_sigma), float(smooth_sigma), float(smooth_sigma))
    clo, chi = float(clip_percentiles[0]), float(clip_percentiles[1])

    if not np.isfinite(H_base).all():
        meta["eig_error"] = "nonfinite_H_base"
        return _empty_result(H_base, meta)

    try:
        w0 = np.sort(
            np.asarray(np.linalg.eigvalsh(0.5 * (H_base + H_base.T)), dtype=DTYPE).reshape(-1)
        )
    except Exception as ex:
        meta["eig_error"] = repr(ex)
        return _empty_result(H_base, meta)
    if w0.size != n or not np.isfinite(w0).all():
        meta["eig_error"] = "eigvalsh_failed_initial"
        return _empty_result(H_base, meta, w0)

    lo, hi = float(w0[0]), float(w0[-1])
    g_sorted = rescale_gamma_to_range(gamma, lo, hi)
    if g_sorted.size < n:
        meta["eig_error"] = "insufficient_zeros"
        return _empty_result(H_base, meta, w0)
    gamma_scaled = np.sort(np.asarray(g_sorted[:n], dtype=DTYPE).reshape(-1))
    if gamma_scaled.size != n:
        meta["eig_error"] = "gamma_scaled_init_mismatch"
        return _empty_result(H_base, meta, w0)

    cap = float(abs_cap_factor * (hi - lo + 1.0) / max(float(alpha) * float(lambda_p_eff), EPS))

    H_t = np.asarray(H_base, dtype=DTYPE, copy=True)
    stag = 0
    prev_sl: Optional[float] = None

    best_J = float("inf")
    best_iter = 0
    best_H = H_t.copy()
    best_row: Dict[str, Any] = {}

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
        delta_s = smooth_delta_multi(delta, sigmas)
        V_diag = -float(alpha) * delta_s
        V_diag = clip_v_diag_percentile(V_diag, clo, chi)
        V_diag = np.clip(V_diag, -cap, cap)
        V_diag = np.asarray(V_diag, dtype=DTYPE).reshape(-1)
        if V_diag.size != n:
            meta["eig_error"] = "V_diag_shape_mismatch"
            break

        V_t = np.diag(V_diag.astype(DTYPE, copy=False))
        H_tilde_new = H_base + float(lambda_p_eff) * V_t
        H_new = sym_eps(H_tilde_new, diag_shift)

        operator_diff = float(np.linalg.norm(H_new - H_t, ord="fro"))
        delta_norm = float(np.mean(np.abs(delta)))

        m = spectral_metrics(eigvals_t, zeros_eval, spacing_fn=spacing_fn, ks_fn=ks_fn, norm_gaps_fn=norm_gaps_fn)
        cur_sl = float(m["spectral_log_mse"])
        J = pareto_objective_J(
            m["spectral_log_mse"],
            m["spacing_mse_normalized"],
            m["ks_wigner"],
            operator_diff,
        )

        if math.isfinite(J) and J < best_J:
            best_J = float(J)
            best_iter = int(t)
            best_H = np.asarray(H_t, dtype=DTYPE, copy=True)
            best_row = {
                "iter": int(t),
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

        if stag >= int(stagnation_window):
            H_t = (1.0 - beta_t) * H_t + beta_t * H_new
            H_t = 0.5 * (H_t + H_t.T)
            meta["stopped_stagnation"] = True
            meta["n_iter"] = int(t + 1)
            break

        H_t = (1.0 - beta_t) * H_t + beta_t * H_new
        H_t = 0.5 * (H_t + H_t.T)
        meta["n_iter"] = int(t + 1)

    final_od = float(rows[-1]["operator_diff"]) if rows else float("nan")
    meta["self_adjointness_fro_final"] = self_adjointness_fro(H_t)

    best_sl = float(best_row.get("spectral_log_mse", float("nan")))
    best_sp = float(best_row.get("spacing_mse_normalized", float("nan")))
    best_ks = float(best_row.get("ks_wigner", float("nan")))
    best_od = float(best_row.get("operator_diff", float("nan")))
    meets = meets_acceptance(
        operator_diff=best_od,
        spectral_log_mse=best_sl,
        spacing_mse_normalized=best_sp,
        ks_wigner=best_ks,
    )

    return {
        "H_final": np.asarray(H_t, dtype=DTYPE, copy=True),
        "H_best_J": best_H,
        "rows": rows,
        "meta": meta,
        "best_iter_by_J": int(best_iter),
        "best_J": float(best_J) if math.isfinite(best_J) else float("nan"),
        "best_spectral_log_mse": best_sl,
        "best_spacing_mse_normalized": best_sp,
        "best_ks_wigner": best_ks,
        "best_operator_diff": best_od,
        "best_delta_norm": float(best_row.get("delta_norm", float("nan"))),
        "final_operator_diff": final_od,
        "meets_all_at_best_checkpoint": bool(meets),
        "best_row": dict(best_row) if best_row else {},
    }


def _empty_result(H_base: np.ndarray, meta: Dict[str, Any], w0: Optional[np.ndarray] = None) -> Dict[str, Any]:
    w = w0 if w0 is not None else np.array([], dtype=DTYPE)
    return {
        "H_final": np.asarray(H_base, dtype=DTYPE, copy=True),
        "H_best_J": np.asarray(H_base, dtype=DTYPE, copy=True),
        "rows": [],
        "meta": meta,
        "best_iter_by_J": 0,
        "best_J": float("inf"),
        "best_spectral_log_mse": float("nan"),
        "best_spacing_mse_normalized": float("nan"),
        "best_ks_wigner": float("nan"),
        "best_operator_diff": float("nan"),
        "best_delta_norm": float("nan"),
        "final_operator_diff": float("nan"),
        "meets_all_at_best_checkpoint": False,
        "best_row": {},
    }
