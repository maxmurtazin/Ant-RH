#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.artin_operator import (
    build_geodesic_kernel,
    build_laplacian,
    hyperbolic_distance_matrix,
    load_geodesics_words_json,
    sample_domain,
    select_top_k_geodesics,
)
from core.spectral_stabilization import safe_eigh


DTYPE_NP = np.float64
DTYPE_T = torch.float64


def _fro_norm_np(A: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(A, dtype=DTYPE_NP), ord="fro"))


def _normalize_basis_np(B: np.ndarray) -> np.ndarray:
    nrm = _fro_norm_np(B) + 1e-8
    return (B / nrm).astype(DTYPE_NP, copy=False)


def _smooth_random_potential(z: np.ndarray, seed: int = 42) -> np.ndarray:
    """
    Random smooth values over z (uses x coordinate + a few Gaussian bumps).
    """
    rng = np.random.default_rng(int(seed))
    x = np.real(z).astype(DTYPE_NP, copy=False)
    y = np.imag(z).astype(DTYPE_NP, copy=False)
    n = x.size
    centers = rng.uniform(-0.5, 0.5, size=6).astype(DTYPE_NP)
    widths = rng.uniform(0.10, 0.35, size=6).astype(DTYPE_NP)
    amps = rng.normal(size=6).astype(DTYPE_NP)
    v = np.zeros((n,), dtype=DTYPE_NP)
    for c, w, a in zip(centers, widths, amps):
        v += a * np.exp(-((x - c) ** 2) / (2.0 * w * w + 1e-12))
    v += 0.15 * rng.normal(size=n).astype(DTYPE_NP) * np.exp(-0.5 * (y - y.mean()) ** 2)
    v = (v - v.mean()) / (v.std() + 1e-8)
    return v


def build_operator_basis(
    z_points: np.ndarray,
    distances: np.ndarray,
    *,
    eps: float,
    geo_sigma: float,
    top_k_geodesics: int,
    geodesics_path: str,
    seed: int = 42,
) -> Tuple[List[np.ndarray], List[str], Dict[str, Any]]:
    """
    Returns:
        basis_matrices (list of NxN float64),
        basis_names,
        report
    """
    z = np.asarray(z_points)
    d = np.asarray(distances, dtype=DTYPE_NP)
    n = int(z.shape[0])

    rep: Dict[str, Any] = {"n_points": n}
    basis: List[np.ndarray] = []
    names: List[str] = []

    # 1) Identity
    B0 = np.eye(n, dtype=DTYPE_NP)
    basis.append(_normalize_basis_np(B0))
    names.append("I")

    # 2) Laplacian L = D - A, A_ij = exp(-d^2/eps)
    d2 = d * d
    A = np.exp(-d2 / (DTYPE_NP(eps) + 1e-12), dtype=DTYPE_NP)
    np.fill_diagonal(A, 0.0)
    D = np.diag(np.sum(A, axis=1).astype(DTYPE_NP, copy=False))
    L = D - A
    basis.append(_normalize_basis_np(L))
    names.append("L_graph")

    # 3) Distance kernels
    for kk in (0.1, 0.3, 1.0):
        K = np.exp(-d2 / (DTYPE_NP(kk * kk) + 1e-12), dtype=DTYPE_NP)
        basis.append(_normalize_basis_np(K))
        names.append(f"K_dist_{kk:g}")

    # 4) Diagonal potential
    v = _smooth_random_potential(z, seed=seed)
    Bdiag = np.diag(v.astype(DTYPE_NP, copy=False))
    basis.append(_normalize_basis_np(Bdiag))
    names.append("V_diag")

    # 5) Artin geodesic kernel
    words = load_geodesics_words_json(geodesics_path)
    geodesics = select_top_k_geodesics(words, top_k=int(top_k_geodesics))
    Bgeo, used = build_geodesic_kernel(z, geodesics, sigma=float(geo_sigma))
    rep["geo_used_geodesics"] = int(used)
    rep["geo_requested_geodesics"] = int(top_k_geodesics)
    basis.append(_normalize_basis_np(Bgeo))
    names.append("K_geo")

    rep["n_basis"] = int(len(basis))
    rep["basis_names"] = list(names)
    return basis, names, rep


def stabilize_H(H: torch.Tensor, diagonal_shift: float = 1e-3) -> torch.Tensor:
    H = 0.5 * (H + H.T)
    H = torch.nan_to_num(H, nan=0.0, posinf=1e6, neginf=-1e6)
    H = H / (torch.linalg.norm(H) + 1e-8)
    n = H.shape[0]
    H = H + float(diagonal_shift) * torch.eye(n, dtype=H.dtype, device=H.device)
    H = torch.tanh(H)
    return 0.5 * (H + H.T)


def _zscore_t(x: torch.Tensor) -> torch.Tensor:
    mu = torch.mean(x)
    sd = torch.std(x, unbiased=False)
    sd = torch.where(sd > 1e-12, sd, torch.ones_like(sd))
    return (x - mu) / sd


def _spacing_loss(e: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    if e.numel() < 3 or z.numel() < 3:
        return e.new_tensor(0.0)
    de = torch.diff(torch.sort(e).values)
    dz = torch.diff(torch.sort(z).values)
    me = torch.mean(de)
    mz = torch.mean(dz)
    me = torch.where(me.abs() > 1e-12, me, torch.ones_like(me))
    mz = torch.where(mz.abs() > 1e-12, mz, torch.ones_like(mz))
    de = de / me
    dz = dz / mz
    m = min(de.numel(), dz.numel())
    return torch.mean((de[:m] - dz[:m]) ** 2)


def _load_zeros_numeric(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"zeros file not found: {p}")
    z = np.loadtxt(str(p), dtype=DTYPE_NP)
    z = np.asarray(z, dtype=DTYPE_NP).reshape(-1)
    z = z[np.isfinite(z)]
    z = z[z > 0.0]
    return z


def main() -> None:
    ap = argparse.ArgumentParser(description="V12.6 Structured Artin operator with basis expansion + stable spectrum")
    ap.add_argument("--n_points", type=int, default=128)
    ap.add_argument("--top_k_geodesics", type=int, default=200)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--eps", type=float, default=0.6)
    ap.add_argument("--geo_sigma", type=float, default=0.3)
    ap.add_argument("--diagonal_shift", type=float, default=1e-3)
    ap.add_argument("--lambda_spec", type=float, default=1.0)
    ap.add_argument("--lambda_spacing", type=float, default=0.5)
    ap.add_argument("--lambda_reg", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--zeros", type=str, default="data/zeta_zeros.txt")
    ap.add_argument("--geodesics", type=str, default="runs/artin_words.json")
    ap.add_argument("--out_dir", type=str, default="runs/")
    args = ap.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    # sample domain + distances (numpy)
    z = sample_domain(int(args.n_points), seed=int(args.seed))
    d = hyperbolic_distance_matrix(z, z)

    basis_np, basis_names, basis_rep = build_operator_basis(
        z_points=z,
        distances=d,
        eps=float(args.eps),
        geo_sigma=float(args.geo_sigma),
        top_k_geodesics=int(args.top_k_geodesics),
        geodesics_path=str(args.geodesics),
        seed=int(args.seed),
    )

    # convert to torch basis tensors
    basis_t = [torch.tensor(B, dtype=DTYPE_T, device=device) for B in basis_np]
    n_basis = len(basis_t)

    w = torch.zeros((n_basis,), dtype=DTYPE_T, device=device, requires_grad=True)
    # mild bias toward graph Laplacian structure
    if n_basis >= 2:
        with torch.no_grad():
            w[1] = -1.0

    opt = torch.optim.Adam([w], lr=float(args.lr))

    zeros_np = _load_zeros_numeric(str(args.zeros))
    k = min(int(args.n_points), int(zeros_np.size))
    if k < 3:
        raise ValueError(f"need at least 3 zeta zeros; got {zeros_np.size}")
    zeros_t = torch.tensor(zeros_np[:k], dtype=DTYPE_T, device=device)
    zeros_tn = _zscore_t(zeros_t)

    last_print = -1
    for step in range(int(args.steps)):
        opt.zero_grad(set_to_none=True)

        H_raw = torch.zeros((int(args.n_points), int(args.n_points)), dtype=DTYPE_T, device=device)
        for i in range(n_basis):
            H_raw = H_raw + w[i] * basis_t[i]

        H = stabilize_H(H_raw, diagonal_shift=float(args.diagonal_shift))

        # Prefer torch eigvalsh for gradients; stabilized H should avoid failures.
        try:
            eig = torch.linalg.eigvalsh(H)
        except RuntimeError:
            eig = torch.linalg.eigvalsh(H.to("cpu")).to(device)

        eig = eig[:k]
        eig_n = _zscore_t(eig)

        L_spec = torch.mean((eig_n - zeros_tn) ** 2)
        L_sp = _spacing_loss(eig_n, zeros_tn)
        L_reg = torch.sum(w * w)

        L = float(args.lambda_spec) * L_spec + float(args.lambda_spacing) * L_sp + float(args.lambda_reg) * L_reg
        if not torch.isfinite(L):
            raise FloatingPointError("loss became NaN/Inf")

        L.backward()
        torch.nn.utils.clip_grad_norm_([w], 1.0)
        opt.step()

        if step % 20 == 0 or step == int(args.steps) - 1:
            with torch.no_grad():
                wn = float(torch.linalg.norm(w).detach().cpu())
                print(
                    f"[{step}] loss={float(L.detach().cpu()):.6g} "
                    f"spec={float(L_spec.detach().cpu()):.6g} "
                    f"spacing={float(L_sp.detach().cpu()):.6g} "
                    f"||w||={wn:.6g}",
                    flush=True,
                )
            last_print = step

    # final operator + robust eig (safe_eigh) for report
    with torch.no_grad():
        H_raw = torch.zeros((int(args.n_points), int(args.n_points)), dtype=DTYPE_T, device=device)
        for i in range(n_basis):
            H_raw = H_raw + w[i] * basis_t[i]
        H = stabilize_H(H_raw, diagonal_shift=float(args.diagonal_shift))
        H_np = H.detach().to("cpu").numpy().astype(DTYPE_NP, copy=False)

    eigvals_np, _, erep = safe_eigh(H_np, stabilize=True, seed=int(args.seed))
    eigvals_np = np.sort(np.asarray(eigvals_np, dtype=DTYPE_NP).reshape(-1))

    kk = min(int(k), int(eigvals_np.size), int(zeros_np.size))
    eig_use = eigvals_np[:kk]
    z_use = zeros_np[:kk]
    eig_n = (eig_use - eig_use.mean()) / (eig_use.std() + 1e-12)
    z_n = (z_use - z_use.mean()) / (z_use.std() + 1e-12)
    L_spec_np = float(np.mean((eig_n - z_n) ** 2))
    de = np.diff(eig_n)
    dz = np.diff(z_n)
    if de.size > 0:
        de = de / (de.mean() + 1e-12)
    if dz.size > 0:
        dz = dz / (dz.mean() + 1e-12)
    m = min(de.size, dz.size)
    L_sp_np = float(np.mean((de[:m] - dz[:m]) ** 2)) if m > 0 else 0.0
    w_np = w.detach().to("cpu").numpy().astype(DTYPE_NP, copy=False)
    w_norm = float(np.linalg.norm(w_np))
    top_idx = np.argsort(-np.abs(w_np))[: min(10, w_np.size)]
    top_weights = [{"name": basis_names[int(i)], "weight": float(w_np[int(i)])} for i in top_idx]

    report: Dict[str, Any] = {
        "final_loss": float(L_spec_np + float(args.lambda_spacing) * L_sp_np + float(args.lambda_reg) * float(np.sum(w_np * w_np))),
        "spectral_loss": float(L_spec_np),
        "spacing_loss": float(L_sp_np),
        "weight_norm": float(w_norm),
        "top_weights": top_weights,
        "basis": basis_rep,
        "eigh_report": erep,
        "device": str(device),
        "n_basis": int(n_basis),
        "timing_s": float(time.perf_counter() - t0),
    }

    np.save(out_dir / "artin_operator_structured.npy", H_np.astype(DTYPE_NP, copy=False))
    np.save(out_dir / "artin_structured_weights.npy", w_np.astype(DTYPE_NP, copy=False))

    with open(out_dir / "artin_structured_spectrum.csv", "w", encoding="utf-8") as f:
        f.write("idx,eigval\n")
        for i, ev in enumerate(eigvals_np):
            f.write(f"{i},{float(ev)}\n")

    with open(out_dir / "artin_structured_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()

