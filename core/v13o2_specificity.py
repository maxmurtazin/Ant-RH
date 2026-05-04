#!/usr/bin/env python3
"""V13O.2: target-blind potentials, synthetic targets, long-range spectral statistics, Pareto J."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from core.v13l_self_consistent import DTYPE, EPS, rescale_gamma_preserve_order, rescale_gamma_to_range, sym_eps  # noqa: F401
from core.v13l1_stabilized import beta_schedule, clip_v_diag_percentile, smooth_delta_multi
from core.v13m1_renormalized import interpolate_delta_k_to_dim, spectral_metrics_window
from core.v13o1_oos import pareto_objective_j, target_blind_v_diag

LARGE_PEN = 1.0e6


def safe_float(x: Any, default: float = LARGE_PEN) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def stable_hash_phase(word: List[int]) -> float:
    h = hashlib.sha256(bytes([((b + 128) % 256) for b in [int(x) for x in word]])).digest()
    return float(int.from_bytes(h[:8], "little", signed=False) % (2**53)) / float(2**53) * (2.0 * math.pi)


def density_rho_vector(n: int) -> np.ndarray:
    """Approximate zero-counting density proxy rho(T) ~ (1/(2*pi))*log(T/(2*pi)), T by index."""
    i = np.arange(1, n + 1, dtype=np.float64)
    T = np.maximum(i, 1.0)
    rho = (1.0 / (2.0 * math.pi)) * np.log(np.maximum(T / (2.0 * math.pi), 1e-12))
    return rho.astype(DTYPE, copy=False)


def word_only_v_diag(n: int, word: List[int], smooth_sigma: float, alpha: float) -> np.ndarray:
    w = [int(x) for x in word]
    sum_w = float(sum(w))
    abs_sum = float(sum(abs(a) for a in w))
    sign_sum = float(sum(1 if a > 0 else -1 for a in w))
    alt = float(sum(((-1) ** i) * w[i] for i in range(len(w))))
    hp = stable_hash_phase(w)
    wl = float(max(len(w), 1))
    x = np.arange(n, dtype=np.float64)
    raw = (
        0.01 * sum_w * np.sin(x * hp / max(n, 1))
        + 0.02 * abs_sum * np.cos(x * 0.17 + sign_sum * 0.03)
        + 0.015 * alt * np.sin(x * 0.09 + hp)
        + 0.02 * wl * np.cos(x * 0.11)
    )
    sigmas = (float(smooth_sigma), float(smooth_sigma), float(smooth_sigma))
    raw_t = smooth_delta_multi(raw.astype(DTYPE, copy=False), sigmas)
    return (-float(alpha) * raw_t).astype(DTYPE, copy=False)


def phase_only_v_diag(z_points: np.ndarray, n: int, word: List[int], alpha: float) -> np.ndarray:
    Z = np.asarray(z_points, dtype=np.complex128).reshape(-1)
    if Z.size < n:
        Z = np.pad(Z, (0, max(0, n - Z.size)), mode="wrap")
    im = np.clip(np.imag(Z[:n]).astype(np.float64), 1e-6, 1e6)
    hp = stable_hash_phase([int(x) for x in word])
    v = np.sin(hp + im).astype(DTYPE, copy=False)
    return (-float(alpha) * v).astype(DTYPE, copy=False)


def unfold_cumulative(gaps: np.ndarray) -> np.ndarray:
    g = np.asarray(gaps, dtype=np.float64).reshape(-1)
    g = g[np.isfinite(g) & (g > 0.0)]
    if g.size == 0:
        return np.zeros((1,), dtype=np.float64)
    m = float(np.mean(g))
    if m <= 0.0:
        m = 1.0
    gn = g / m
    return np.concatenate([[0.0], np.cumsum(gn)]).astype(np.float64, copy=False)


def number_variance_l2(eig: np.ndarray, z: np.ndarray, k: int) -> float:
    try:
        e = np.sort(np.asarray(eig, dtype=np.float64).reshape(-1))[: int(k)]
        zz = np.sort(np.asarray(z, dtype=np.float64).reshape(-1))[: int(k)]
        if e.size < 4 or zz.size < 4:
            return LARGE_PEN
        ue = unfold_cumulative(np.diff(e))
        uz = unfold_cumulative(np.diff(zz))
        mx = float(max(ue[-1], uz[-1], 1e-6))
        Ls = [2, 4, 8, 16]
        errs: List[float] = []
        for L in Ls:
            L = int(min(L, int(ue.size), int(uz.size)))
            if L < 2:
                continue
            nwin = max(8, int(mx))
            var_e: List[float] = []
            var_z: List[float] = []
            for s in np.linspace(0.0, mx - L, num=min(24, max(2, nwin))):
                we = np.sum((ue >= s) & (ue < s + L))
                wz = np.sum((uz >= s) & (uz < s + L))
                var_e.append(float(we))
                var_z.append(float(wz))
            if len(var_e) < 2:
                continue
            ve = float(np.var(np.asarray(var_e, dtype=np.float64)))
            vz = float(np.var(np.asarray(var_z, dtype=np.float64)))
            errs.append((ve - vz) ** 2)
        if not errs:
            return LARGE_PEN
        return float(math.sqrt(float(np.mean(np.asarray(errs, dtype=np.float64)))))
    except Exception:
        return LARGE_PEN


def spectral_rigidity_proxy(eig: np.ndarray, k: int) -> float:
    try:
        e = np.sort(np.asarray(eig, dtype=np.float64).reshape(-1))[: int(k)]
        if e.size < 40:
            return LARGE_PEN
        ue = unfold_cumulative(np.diff(e))
        x = np.arange(ue.size, dtype=np.float64)
        resid_sum = 0.0
        count = 0
        for W in (8, 16, 32):
            W = int(min(W, ue.size - 2))
            if W < 4:
                continue
            for start in range(0, ue.size - W, max(1, W // 2)):
                seg = ue[start : start + W]
                xs = np.arange(seg.size, dtype=np.float64)
                coef = np.polyfit(xs, seg, 1)
                pred = np.polyval(coef, xs)
                resid_sum += float(np.mean((seg - pred) ** 2))
                count += 1
        if count == 0:
            return LARGE_PEN
        return float(resid_sum / count)
    except Exception:
        return LARGE_PEN


def two_point_correlation_l2(eig: np.ndarray, z: np.ndarray, k: int, *, max_lag: int = 10, bins: int = 32) -> float:
    try:
        e = np.sort(np.asarray(eig, dtype=np.float64).reshape(-1))[: min(int(k), 48)]
        zz = np.sort(np.asarray(z, dtype=np.float64).reshape(-1))[: min(int(k), 48)]
        if e.size < 5 or zz.size < 5:
            return LARGE_PEN

        def hist_pairwise(u: np.ndarray) -> np.ndarray:
            m = int(min(u.size, 40))
            dlist: List[float] = []
            for i in range(m):
                for j in range(i + 1, min(i + 1 + max_lag, m)):
                    dlist.append(abs(float(u[j]) - float(u[i])))
            if not dlist:
                return np.zeros((bins,), dtype=np.float64)
            d = np.asarray(dlist, dtype=np.float64)
            h, _ = np.histogram(d, bins=bins, range=(0.0, float(np.percentile(d, 99) + 1e-9)))
            s = float(np.sum(h)) + 1e-12
            return (h / s).astype(np.float64, copy=False)

        he = hist_pairwise(e)
        hz = hist_pairwise(zz)
        return float(np.linalg.norm(he - hz))
    except Exception:
        return LARGE_PEN


def run_length_score_fn(eig: np.ndarray, z: np.ndarray, k: int) -> float:
    try:
        e = np.sort(np.asarray(eig, dtype=np.float64).reshape(-1))[: int(k)]
        zz = np.sort(np.asarray(z, dtype=np.float64).reshape(-1))[: int(k)]
        if e.size < 6 or zz.size < 6:
            return LARGE_PEN
        ge = np.diff(e)
        gz = np.diff(zz)
        me = float(np.median(ge))
        mz = float(np.median(gz))
        be = (ge > me).astype(np.int32)
        bz = (gz > mz).astype(np.int32)

        def runs(b: np.ndarray) -> np.ndarray:
            if b.size == 0:
                return np.zeros((0,), dtype=np.int32)
            ch = np.diff(np.concatenate([[0], b, [0]]))
            idx = np.where(ch != 0)[0]
            if idx.size < 2:
                return np.ones((1,), dtype=np.int32)
            out: List[int] = []
            for i in range(0, len(idx) - 1, 2):
                out.append(int(idx[i + 1] - idx[i]))
            return np.asarray(out, dtype=np.int32) if out else np.ones((1,), dtype=np.int32)

        re = runs(be)
        rz = runs(bz)
        h1, edges = np.histogram(re.astype(np.float64), bins=min(12, max(3, int(re.max()))))
        h2, _ = np.histogram(rz.astype(np.float64), bins=edges)
        s1 = float(np.sum(h1)) + 1e-12
        s2 = float(np.sum(h2)) + 1e-12
        return float(np.sum(np.abs(h1 / s1 - h2 / s2)))
    except Exception:
        return LARGE_PEN


def nearest_neighbor_spacing_mse(eig: np.ndarray, z: np.ndarray, k: int, norm_gaps_fn: Any) -> float:
    try:
        e = np.sort(np.asarray(eig, dtype=DTYPE).reshape(-1))[: int(k)]
        zz = np.sort(np.asarray(z, dtype=DTYPE).reshape(-1))[: int(k)]
        kk = int(min(e.size, zz.size, k))
        if kk < 3:
            return LARGE_PEN
        ge = np.diff(e[:kk])
        gz = np.diff(zz[:kk])
        ne = norm_gaps_fn(ge.astype(DTYPE))
        nz = norm_gaps_fn(gz.astype(DTYPE))
        ne = np.asarray(ne, dtype=np.float64).reshape(-1)
        nz = np.asarray(nz, dtype=np.float64).reshape(-1)
        m = int(min(ne.size, nz.size))
        if m < 1:
            return LARGE_PEN
        return float(np.mean((ne[:m] - nz[:m]) ** 2))
    except Exception:
        return LARGE_PEN


def long_range_bundle(
    eig: np.ndarray,
    z: np.ndarray,
    k: int,
    *,
    spacing_fn: Any,
    ks_fn: Any,
    norm_gaps_fn: Any,
) -> Dict[str, float]:
    m = spectral_metrics_window(
        np.sort(np.asarray(eig, dtype=DTYPE).reshape(-1)),
        np.sort(np.asarray(z, dtype=DTYPE).reshape(-1)),
        int(k),
        spacing_fn=spacing_fn,
        ks_fn=ks_fn,
        norm_gaps_fn=norm_gaps_fn,
    )
    sp = safe_float(m.get("spacing_mse_normalized"), LARGE_PEN)
    ks = safe_float(m.get("ks_wigner"), LARGE_PEN)
    nn = nearest_neighbor_spacing_mse(eig, z, k, norm_gaps_fn)
    nv = number_variance_l2(eig, z, k)
    rig = spectral_rigidity_proxy(eig, k)
    rl = run_length_score_fn(eig, z, k)
    tp = two_point_correlation_l2(eig, z, k)
    return {
        "spectral_log_mse": safe_float(m.get("spectral_log_mse"), LARGE_PEN),
        "spacing_mse_normalized": sp,
        "ks_wigner": ks,
        "nearest_neighbor_spacing_mse": nn,
        "number_variance_l2": nv,
        "spectral_rigidity_proxy": rig,
        "run_length_score": rl,
        "two_point_correlation_l2": tp,
    }


def pareto_J_v13o2(lr: Dict[str, float]) -> float:
    return float(
        lr["spectral_log_mse"]
        + 0.25 * lr["spacing_mse_normalized"]
        + 0.50 * lr["ks_wigner"]
        + 0.20 * lr["number_variance_l2"]
        + 0.20 * lr["spectral_rigidity_proxy"]
        + 0.10 * lr["run_length_score"]
    )


def compute_v_diag_v13o2(
    *,
    v_mode: str,
    H_base: np.ndarray,
    eigvals_t: np.ndarray,
    g_short: np.ndarray,
    train_ordered: bool,
    n: int,
    k_use: int,
    alpha: float,
    smooth_sigma_dim: float,
    clip_lo: float,
    clip_hi: float,
    cap: float,
    sigmas: Tuple[float, float, float],
    z_points: np.ndarray,
    word: List[int],
) -> np.ndarray:
    if v_mode in ("target_blind_V",):
        V_diag = target_blind_v_diag(H_base, float(smooth_sigma_dim), float(alpha))
    elif v_mode == "density_only_V":
        rho = density_rho_vector(n)
        rho_s = smooth_delta_multi(rho, sigmas)
        V_diag = -float(alpha) * rho_s
    elif v_mode == "word_only_V":
        V_diag = word_only_v_diag(n, word, float(smooth_sigma_dim), float(alpha))
    elif v_mode == "phase_only_V":
        V_diag = phase_only_v_diag(z_points, n, word, float(alpha))
    else:
        eig_k = eigvals_t[:k_use]
        g_k = g_short[:k_use]
        if not train_ordered:
            g_k = np.sort(np.asarray(g_k, dtype=DTYPE).copy())
        delta_k = eig_k - g_k
        delta_dim = interpolate_delta_k_to_dim(delta_k, n)
        delta_s = smooth_delta_multi(delta_dim, sigmas)
        V_diag = -float(alpha) * delta_s
    V_diag = clip_v_diag_percentile(np.asarray(V_diag, dtype=DTYPE).reshape(-1), float(clip_lo), float(clip_hi))
    V_diag = np.clip(V_diag, -cap, cap)
    return np.asarray(V_diag, dtype=DTYPE).reshape(-1)


def train_v13o2_cell(
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
    v_mode: str,
    z_points: np.ndarray,
    word: List[int],
    on_train_iter: Optional[Callable[[int, int, Dict[str, Any]], None]] = None,
) -> Dict[str, Any]:
    """Self-consistent train; ``v_mode`` selects potential construction."""
    H_base = np.asarray(H_base, dtype=DTYPE, copy=False)
    n = int(H_base.shape[0])
    rows: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {"converged_operator": False, "n_iter": 0, "eig_error": None}
    sigmas = (float(smooth_sigma_dim), float(smooth_sigma_dim), float(smooth_sigma_dim))

    lp_use = float(lambda_p_dim)
    if v_mode == "very_weak_V":
        lp_use *= 0.05
    elif v_mode == "weak_V":
        lp_use *= 0.20

    freeze_after: Optional[int] = None
    if v_mode == "frozen_V_after_5":
        freeze_after = 5
    elif v_mode == "frozen_V_after_10":
        freeze_after = 10

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

    use_target_mismatch = v_mode in (
        "full_V",
        "frozen_V_after_5",
        "frozen_V_after_10",
        "very_weak_V",
        "weak_V",
        "smooth_V_strong",
    )
    if use_target_mismatch:
        g_short = (
            rescale_gamma_preserve_order(zt, lo, hi) if train_ordered else rescale_gamma_to_range(zt, lo, hi)
        )
    else:
        g_short = np.zeros_like(zt)
    g_short = np.asarray(g_short, dtype=DTYPE).reshape(-1)
    k_use = int(min(n, int(g_short.size))) if use_target_mismatch else int(min(n, int(k_train)))
    cap = float(abs_cap_factor * (hi - lo + 1.0) / max(float(alpha) * float(max(lp_use, EPS)), EPS))

    H_t = np.asarray(H_base, dtype=DTYPE, copy=True)
    V_stash_for_freeze: Optional[np.ndarray] = None
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

        if v_mode in ("target_blind_V", "density_only_V", "word_only_V", "phase_only_V"):
            V_diag = compute_v_diag_v13o2(
                v_mode=v_mode,
                H_base=H_base,
                eigvals_t=eigvals_t,
                g_short=g_short,
                train_ordered=train_ordered,
                n=n,
                k_use=k_use,
                alpha=alpha,
                smooth_sigma_dim=smooth_sigma_dim,
                clip_lo=clip_lo,
                clip_hi=clip_hi,
                cap=cap,
                sigmas=sigmas,
                z_points=z_points,
                word=word,
            )
        else:
            V_diag = compute_v_diag_v13o2(
                v_mode="full_V",
                H_base=H_base,
                eigvals_t=eigvals_t,
                g_short=g_short,
                train_ordered=train_ordered,
                n=n,
                k_use=k_use,
                alpha=alpha,
                smooth_sigma_dim=smooth_sigma_dim,
                clip_lo=clip_lo,
                clip_hi=clip_hi,
                cap=cap,
                sigmas=sigmas,
                z_points=z_points,
                word=word,
            )

        if freeze_after is not None and freeze_after > 0:
            if t == freeze_after - 1:
                V_stash_for_freeze = np.asarray(V_diag, dtype=DTYPE, copy=True)
            if t >= freeze_after and V_stash_for_freeze is not None:
                V_diag = np.asarray(V_stash_for_freeze, dtype=DTYPE, copy=True)

        V_last_applied = np.asarray(V_diag, dtype=DTYPE, copy=True)
        V_t = np.diag(V_diag)
        H_tilde_new = H_base + float(lp_use) * V_t
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
        rN: Dict[str, Any] = {}
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


def self_adjointness_fro(H: np.ndarray) -> float:
    H = np.asarray(H, dtype=DTYPE, copy=False)
    return float(np.linalg.norm(H - H.T, ord="fro"))
