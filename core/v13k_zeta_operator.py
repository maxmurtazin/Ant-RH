#!/usr/bin/env python3
"""
V13K: zeta-informed diagonal potentials on the V13J word-sensitive backbone.

Reuses ``_build_word_sensitive_core`` (L, K_raw, max-norm K, baseline sin-V) and only
replaces the diagonal potential factor ``V_norm`` for non-baseline modes.
Does not change ACO or ``build_word_sensitive_operator``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.artin_operator_word_sensitive import EPS, _build_word_sensitive_core

DTYPE = np.float64


def geodesic_entry_for_word(word: List[int]) -> Dict[str, Any]:
    """Single-geodesic dict aligned with validation / ACO word-sensitive path."""
    w = [int(x) for x in word]
    return {
        "a_list": w,
        "length": float(max(1.0, len(w))),
        "is_hyperbolic": True,
        "primitive": True,
    }


def _clip_exp_arg(x: np.ndarray) -> np.ndarray:
    return np.clip(x, -500.0, 500.0)


def _map_gamma_to_imag_range(gamma: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Linear map sorted positive zeros into [min(Im z), max(Im z)] for envelope alignment."""
    g = np.asarray(gamma, dtype=DTYPE).reshape(-1)
    g = g[np.isfinite(g) & (g > 0.0)]
    if g.size == 0:
        return np.zeros((0,), dtype=DTYPE)
    g = np.sort(g)
    ymin = float(np.min(y))
    ymax = float(np.max(y))
    gmin = float(g[0])
    gmax = float(g[-1])
    span = gmax - gmin + EPS
    return ymin + (g - gmin) / span * (ymax - ymin + EPS)


def potential_vec_zero_phase(y: np.ndarray, gamma: np.ndarray, tau: float) -> np.ndarray:
    """V_q = sum_n exp(-((Im z_q - gamma_n_scaled)^2)/(2 tau^2)) * cos(gamma_n_scaled * Im z_q)."""
    y = np.asarray(y, dtype=DTYPE).reshape(-1)
    gmap = _map_gamma_to_imag_range(gamma, y)
    if gmap.size == 0:
        return np.zeros_like(y, dtype=DTYPE)
    t = max(float(tau), EPS)
    diff = y[:, None] - gmap[None, :]
    env = np.exp(_clip_exp_arg(-0.5 * (diff / t) ** 2))
    ph = np.cos(gmap[None, :] * y[:, None])
    return np.sum(env * ph, axis=1).astype(DTYPE, copy=False)


def potential_vec_spacing_phase(n_dim: int, spacings: np.ndarray, tau: float) -> np.ndarray:
    """s_n = gamma_{n+1}-gamma_n; V_q = sum_n Gaussian(q/n_dim - n/N) * cos(s_n * q)."""
    s = np.asarray(spacings, dtype=DTYPE).reshape(-1)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return np.zeros((n_dim,), dtype=DTYPE)
    N = int(s.size)
    t = max(float(tau), EPS)
    q = np.arange(n_dim, dtype=DTYPE)
    idx = np.arange(N, dtype=DTYPE)
    # (n_dim, N)
    diff = (q[:, None] / max(float(n_dim), 1.0)) - (idx[None, :] / max(float(N), 1.0))
    env = np.exp(_clip_exp_arg(-0.5 * (diff / t) ** 2))
    ph = np.cos(s[None, :] * q[:, None])
    return np.sum(env * ph, axis=1).astype(DTYPE, copy=False)


def potential_vec_log_zero_phase(n_dim: int, gamma: np.ndarray, tau: float) -> np.ndarray:
    """u_n = log(1+gamma_n); same index Gaussian as spacing_phase; cos(u_n * q)."""
    g = np.asarray(gamma, dtype=DTYPE).reshape(-1)
    g = g[np.isfinite(g) & (g > 0.0)]
    g = np.sort(g)
    if g.size == 0:
        return np.zeros((n_dim,), dtype=DTYPE)
    u = np.log1p(g).astype(DTYPE, copy=False)
    N = int(u.size)
    t = max(float(tau), EPS)
    q = np.arange(n_dim, dtype=DTYPE)
    idx = np.arange(N, dtype=DTYPE)
    diff = (q[:, None] / max(float(n_dim), 1.0)) - (idx[None, :] / max(float(N), 1.0))
    env = np.exp(_clip_exp_arg(-0.5 * (diff / t) ** 2))
    ph = np.cos(u[None, :] * q[:, None])
    return np.sum(env * ph, axis=1).astype(DTYPE, copy=False)


def potential_vec_self_consistent(
    n_dim: int,
    eig_sorted: np.ndarray,
    gamma: np.ndarray,
    tau: float,
) -> np.ndarray:
    """
    V_q = sum_k exp(-((q/n_dim - k/N)^2)/(2 tau^2)) * cos((eig_k - gamma_k_scaled) * q)
    with gamma_k_scaled matched to eig scale via median ratio.
    """
    eig = np.sort(np.asarray(eig_sorted, dtype=DTYPE).reshape(-1))
    eig = eig[np.isfinite(eig)]
    g = np.asarray(gamma, dtype=DTYPE).reshape(-1)
    g = g[np.isfinite(g) & (g > 0.0)]
    g = np.sort(g)
    if eig.size == 0 or g.size == 0:
        return np.zeros((n_dim,), dtype=DTYPE)
    K = int(min(eig.size, g.size, n_dim))
    eig = eig[:K]
    g = g[:K]
    med_e = float(np.median(np.abs(eig)) + EPS)
    med_g = float(np.median(g) + EPS)
    scale = med_e / med_g
    g_scaled = (g * scale).astype(DTYPE, copy=False)
    N = K
    t = max(float(tau), EPS)
    q = np.arange(n_dim, dtype=DTYPE)
    kidx = np.arange(N, dtype=DTYPE)
    diff = (q[:, None] / max(float(n_dim), 1.0)) - (kidx[None, :] / max(float(N), 1.0))
    env = np.exp(_clip_exp_arg(-0.5 * (diff / t) ** 2))
    d = (eig[None, :] - g_scaled[None, :]).astype(DTYPE, copy=False)
    ph = np.cos(d * q[:, None])
    return np.sum(env * ph, axis=1).astype(DTYPE, copy=False)


def _assemble_h_custom_v(
    core: Dict[str, Any],
    *,
    potential_weight: float,
    v_custom_vec: np.ndarray,
    diag_shift: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """H = Sym(H0 + G_w + pw * diag(v_custom / max|v_custom|)) + diag_shift * I."""
    v = np.asarray(v_custom_vec, dtype=DTYPE).reshape(-1)
    n = int(v.size)
    if not np.isfinite(v).all():
        raise ValueError("nonfinite custom potential vector")
    vs = float(np.max(np.abs(v)))
    v_n = v / (vs + EPS)
    pw = float(potential_weight)
    V_w = pw * np.diag(v_n.astype(DTYPE, copy=False))
    H0 = np.asarray(core["H0"], dtype=DTYPE, copy=False)
    G_w = np.asarray(core["G"], dtype=DTYPE, copy=False)
    H_tilde = H0 + G_w + V_w
    H_sym = 0.5 * (H_tilde + H_tilde.T)
    H_final = H_sym + float(diag_shift) * np.eye(n, dtype=DTYPE)
    if not np.isfinite(H_final).all():
        raise ValueError("nonfinite assembled operator")
    norms = {
        "potential_norm": float(np.linalg.norm(V_w, ord="fro")),
        "geodesic_norm": float(np.linalg.norm(G_w, ord="fro")),
        "base_norm": float(np.linalg.norm(H0, ord="fro")),
        "operator_fro_norm": float(np.linalg.norm(H_final, ord="fro")),
    }
    return H_final, norms


def build_v13k_operator(
    *,
    z_points: np.ndarray,
    geodesics: List[Dict[str, Any]],
    zeros: np.ndarray,
    mode: str,
    eps: float = 0.6,
    geo_sigma: float = 0.6,
    geo_weight: float = 10.0,
    potential_weight: float = 0.25,
    distances: Optional[np.ndarray] = None,
    tau: Optional[float] = None,
    diag_shift: float = 1e-6,
    baseline_eig_for_self_consistent: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Build H_w for the given ``mode``.

    - ``baseline``: identical to ``_build_word_sensitive_core`` / V13J sin-V.
    - Other modes: same L and K normalization; replace diagonal V with zeta-informed vector.
    """
    mode = str(mode).strip().lower()
    z = np.asarray(z_points, dtype=np.complex128 if z_points.dtype.kind == "c" else DTYPE)
    y = np.imag(z).astype(DTYPE, copy=False).reshape(-1)
    n_dim = int(y.size)

    gam = np.asarray(zeros, dtype=DTYPE).reshape(-1)
    gam = gam[np.isfinite(gam) & (gam > 0.0)]
    gam = np.sort(gam)

    # Default taus (dimensionless for index modes; in Im(z) units for zero_phase)
    if tau is None:
        tau_z = float(0.25 * (float(np.max(y)) - float(np.min(y)) + EPS))
        tau_idx = float(0.12)
    else:
        tau_z = float(tau)
        tau_idx = float(tau)

    core = _build_word_sensitive_core(
        z_points=z,
        distances=distances,
        geodesics=geodesics,
        eps=float(eps),
        geo_sigma=float(geo_sigma),
        kernel_normalization="max",
        laplacian_weight=1.0,
        geo_weight=float(geo_weight),
        potential_weight=float(potential_weight),
        diag_shift=float(diag_shift),
    )

    meta: Dict[str, Any] = {
        "mode": mode,
        "n_dim": n_dim,
        "n_zeros_used": int(gam.size),
        "tau_zero_phase_default": float(tau_z),
        "tau_index_default": float(tau_idx),
    }

    if mode == "baseline":
        H = np.asarray(core["H_final"], dtype=DTYPE, copy=True)
        V_w = np.asarray(core["V"], dtype=DTYPE, copy=False)
        G_w = np.asarray(core["G"], dtype=DTYPE, copy=False)
        H0 = np.asarray(core["H0"], dtype=DTYPE, copy=False)
        meta.update(
            {
                "potential_norm": float(np.linalg.norm(V_w, ord="fro")),
                "geodesic_norm": float(np.linalg.norm(G_w, ord="fro")),
                "base_norm": float(np.linalg.norm(H0, ord="fro")),
                "operator_fro_norm": float(np.linalg.norm(H, ord="fro")),
            }
        )
        return H, meta

    if mode == "zero_phase":
        t_use = float(tau) if tau is not None else tau_z
        v_vec = potential_vec_zero_phase(y, gam, t_use)
        meta["tau_zero_phase"] = float(t_use)
    elif mode == "spacing_phase":
        if gam.size < 2:
            v_vec = np.zeros((n_dim,), dtype=DTYPE)
        else:
            t_use = float(tau) if tau is not None else tau_idx
            s = np.diff(gam)
            v_vec = potential_vec_spacing_phase(n_dim, s, t_use)
            meta["tau_index"] = float(t_use)
    elif mode == "log_zero_phase":
        t_use = float(tau) if tau is not None else tau_idx
        v_vec = potential_vec_log_zero_phase(n_dim, gam, t_use)
        meta["tau_index"] = float(t_use)
    elif mode == "self_consistent_phase":
        t_use = float(tau) if tau is not None else tau_idx
        if baseline_eig_for_self_consistent is not None:
            eig0 = np.asarray(baseline_eig_for_self_consistent, dtype=DTYPE).reshape(-1)
        else:
            Hb = np.asarray(core["H_final"], dtype=DTYPE, copy=False)
            eig0 = np.linalg.eigvalsh(0.5 * (Hb + Hb.T))
        v_vec = potential_vec_self_consistent(n_dim, eig0, gam, t_use)
        meta["tau_index"] = float(t_use)
    else:
        raise ValueError(f"unknown potential mode={mode!r}")

    H, norms = _assemble_h_custom_v(core, potential_weight=potential_weight, v_custom_vec=v_vec, diag_shift=diag_shift)
    meta.update({k: float(v) for k, v in norms.items()})
    return H, meta
