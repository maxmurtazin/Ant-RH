#!/usr/bin/env python3
from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

_DTF = np.float64

from core.spectral_stabilization import safe_eigh, stable_spectral_loss


def sample_domain(n_points: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-0.5, 0.5, size=n_points).astype(_DTF, copy=False)
    y = rng.uniform(0.5, 3.0, size=n_points).astype(_DTF, copy=False)
    z = (x + 1j * y).astype(np.complex64, copy=False)
    return z


def hyperbolic_distance_matrix(Z1: np.ndarray, Z2: np.ndarray) -> np.ndarray:
    """
    d(z,w) = arccosh(1 + |z-w|^2 / (2 Im(z) Im(w))).
    Returns matrix shape (N1, N2).
    """
    Z1 = np.asarray(Z1)
    Z2 = np.asarray(Z2)
    y1 = np.maximum(np.imag(Z1).astype(_DTF, copy=False), 1e-12)[:, None]
    y2 = np.maximum(np.imag(Z2).astype(_DTF, copy=False), 1e-12)[None, :]
    dz = (Z1[:, None] - Z2[None, :]).astype(np.complex128, copy=False)
    num = (dz.real * dz.real + dz.imag * dz.imag).astype(_DTF, copy=False)
    arg = 1.0 + num / (2.0 * y1 * y2)
    arg = np.maximum(arg, 1.0 + 1e-12)
    return np.arccosh(arg, dtype=_DTF)


def build_laplacian(Z: np.ndarray, eps: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = hyperbolic_distance_matrix(Z, Z)
    d2 = d * d
    A = np.exp(-d2 / (_DTF(eps) + 1e-12), dtype=_DTF)
    np.fill_diagonal(A, 0.0)
    D = np.sum(A, axis=1)
    L = np.diag(D) - A
    return L, A, d


def power_T(a: int) -> np.ndarray:
    out = np.empty((2, 2), dtype=_DTF)
    out[0, 0] = 1.0
    out[0, 1] = _DTF(a)
    out[1, 0] = 0.0
    out[1, 1] = 1.0
    return out


def precompute_T_powers(max_power: int) -> Tuple[np.ndarray, int]:
    n = 2 * max_power + 1
    stack = np.empty((n, 2, 2), dtype=_DTF)
    offset = max_power
    for i in range(n):
        a = i - offset
        stack[i, 0, 0] = 1.0
        stack[i, 0, 1] = _DTF(a)
        stack[i, 1, 0] = 0.0
        stack[i, 1, 1] = 1.0
    return stack, offset


S_SL2 = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=_DTF)
I2 = np.eye(2, dtype=_DTF)


def build_word(a_list: List[int], T_stack: np.ndarray, offset: int) -> np.ndarray:
    M = I2.copy()
    for a in a_list:
        ai = int(a)
        M = M @ S_SL2
        M = M @ T_stack[ai + offset]
    return M


def mobius_apply(M: np.ndarray, Z: np.ndarray, denom_eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply gamma(z) = (a z + b) / (c z + d) to an array of z.
    Returns (Zg, valid_mask) where invalid points are set to a dummy value.
    """
    a = _DTF(M[0, 0])
    b = _DTF(M[0, 1])
    c = _DTF(M[1, 0])
    d = _DTF(M[1, 1])

    Zc = np.asarray(Z, dtype=np.complex128)
    denom = c * Zc + d
    valid = np.abs(denom) >= denom_eps
    out = np.empty_like(Zc)
    out[valid] = (a * Zc[valid] + b) / denom[valid]
    out[~valid] = 1j  # dummy in upper half-plane; will be masked out
    return out.astype(np.complex64, copy=False), valid


def load_geodesics_words_json(path: str | Path) -> List[Dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"geodesics file not found: {p}")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("artin words json must be a list of feature dicts")
    return data


def select_top_k_geodesics(words: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    keep: List[Dict[str, Any]] = []
    for w in words:
        try:
            if not bool(w.get("is_hyperbolic", False)):
                continue
            if not bool(w.get("primitive", False)):
                continue
            ell = float(w.get("length", float("nan")))
            if not np.isfinite(ell) or ell <= 0.0:
                continue
            a_list = w.get("a_list", None)
            if not isinstance(a_list, list) or len(a_list) == 0:
                continue
            keep.append(w)
        except Exception:
            continue
    keep.sort(key=lambda d: float(d["length"]))
    return keep[: int(top_k)]


def build_geodesic_kernel(
    Z: np.ndarray,
    geodesics: List[Dict[str, Any]],
    sigma: float,
) -> Tuple[np.ndarray, int]:
    n = Z.shape[0]
    if len(geodesics) == 0:
        return np.zeros((n, n), dtype=_DTF), 0

    max_power = 0
    for g in geodesics:
        al = g["a_list"]
        if al:
            max_power = max(max_power, int(np.max(np.abs(np.asarray(al, dtype=np.int64)))))
    T_stack, offset = precompute_T_powers(max_power)

    sig2 = _DTF(sigma) * _DTF(sigma) + 1e-12
    Kmat = np.zeros((n, n), dtype=_DTF)
    used = 0

    for g in geodesics:
        a_list = [int(x) for x in g["a_list"]]
        ell = _DTF(g["length"])
        w = 1.0 / (1.0 + ell)

        M = build_word(a_list, T_stack, offset)
        Zg, valid = mobius_apply(M, Z)
        if not np.any(valid):
            continue

        d = hyperbolic_distance_matrix(Z, Zg)
        d2 = d * d
        contrib = np.exp(-d2 / sig2, dtype=_DTF)
        if not np.all(valid):
            contrib[:, ~valid] = 0.0
        Kmat += w * contrib
        used += 1

    return Kmat, used


def load_zeros(path: str | Path) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"zeros file not found: {p}")
    z = np.loadtxt(str(p), dtype=_DTF)
    z = np.asarray(z, dtype=_DTF).reshape(-1)
    z = z[np.isfinite(z)]
    return z


def main() -> None:
    ap = argparse.ArgumentParser(description="Build Hilbert–Pólya candidate operator from Artin billiard geometry")
    ap.add_argument("--n_points", type=int, default=256)
    ap.add_argument("--sigma", type=float, default=0.3)
    ap.add_argument("--top_k_geodesics", type=int, default=500)
    ap.add_argument("--eps", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--geodesics", type=str, default="runs/artin_words.json")
    ap.add_argument("--zeros", type=str, default="data/zeta_zeros.txt")
    ap.add_argument("--out_dir", type=str, default="runs/")
    args = ap.parse_args()

    n = int(args.n_points)
    sigma = float(args.sigma)
    top_k = int(args.top_k_geodesics)
    eps = float(args.eps)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    Z = sample_domain(n, seed=int(args.seed))

    L, A, _ = build_laplacian(Z, eps=eps)

    words = load_geodesics_words_json(args.geodesics)
    geodesics = select_top_k_geodesics(words, top_k=top_k)
    Kmat, used_geodesics = build_geodesic_kernel(Z, geodesics, sigma=sigma)

    H = -L + Kmat
    H = (H + H.T) * 0.5
    symmetry_error = float(np.linalg.norm(H - H.T, ord="fro"))
    t_build = time.perf_counter() - t0

    t1 = time.perf_counter()
    eigvals, _, erep = safe_eigh(H, stabilize=True, seed=int(args.seed))
    eigvals = np.asarray(eigvals, dtype=_DTF).reshape(-1)
    eigvals.sort()
    t_eig = time.perf_counter() - t1

    zeros = load_zeros(args.zeros)
    if zeros.size < n:
        raise ValueError(f"need at least {n} zeta zeros, got {zeros.size}")
    loss_total, loss_rep = stable_spectral_loss(H, zeros, k=n, normalize_spectrum=True, spacing_loss=True, seed=int(args.seed))
    spectral_loss = float(loss_rep.get("spectral_loss", float("inf")))

    print(f"build time (s): {t_build:.6g}", flush=True)
    print(f"eig time (s): {t_eig:.6g}", flush=True)
    print(f"symmetry error: {symmetry_error:.6g}", flush=True)
    print(f"spectral loss: {spectral_loss:.12g}", flush=True)

    np.save(out_dir / "artin_operator.npy", H.astype(_DTF, copy=False))

    with open(out_dir / "artin_operator_spectrum.csv", "w", encoding="utf-8") as f:
        f.write("idx,eigval\n")
        for i, ev in enumerate(eigvals):
            f.write(f"{i},{float(ev)}\n")

    report: Dict[str, Any] = {
        "n_points": n,
        "sigma": float(sigma),
        "eps": float(eps),
        "top_k_geodesics": top_k,
        "used_geodesics": int(used_geodesics),
        "symmetry_error": symmetry_error,
        "spectral_loss": spectral_loss,
        "build_time_s": float(t_build),
        "eig_time_s": float(t_eig),
        "eigh_report": erep,
        "spectral_loss_report": loss_rep,
        "spectral_loss_total": float(loss_total),
        "geodesics_path": str(args.geodesics),
        "zeros_path": str(args.zeros),
    }
    with open(out_dir / "operator_validation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()


