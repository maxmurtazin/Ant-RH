#!/usr/bin/env python3
from __future__ import annotations

import os
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.spectral_stabilization import safe_eigh, stabilize_operator, stable_spectral_loss


_DTF = np.float64


def _load_operator(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"operator not found: {p}")
    H = np.load(str(p))
    return np.asarray(H)


def _load_zeros(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"zeros not found: {p}")
    z = np.loadtxt(str(p), dtype=_DTF)
    z = np.asarray(z, dtype=_DTF).reshape(-1)
    z = z[np.isfinite(z)]
    return z


def _write_eigs_csv(path: Path, eigvals: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("idx,eigval\n")
        for i, v in enumerate(eigvals):
            f.write(f"{i},{float(v)}\n")


def _plot_before_after(out_png: Path, eig_before: np.ndarray, eig_after: np.ndarray, k: int) -> None:
    plot_cache = out_png.parent / ".matplotlib"
    plot_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(plot_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(plot_cache))
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt

    kb = min(int(k), eig_before.size)
    ka = min(int(k), eig_after.size)

    plt.figure(figsize=(9, 5))
    if kb > 0:
        plt.plot(np.arange(kb), np.sort(eig_before)[:kb], label="before (raw)")
    if ka > 0:
        plt.plot(np.arange(ka), np.sort(eig_after)[:ka], label="after (stable)")
    plt.xlabel("rank")
    plt.ylabel("eigenvalue")
    plt.title("Spectrum before/after stabilization")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def _synthetic_ill_conditioned(n: int = 256, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    # low-rank + near repeated eigenvalues + tiny diagonal
    U = rng.normal(size=(n, 8)).astype(_DTF)
    H = U @ U.T
    H = 0.5 * (H + H.T)
    # make repeated-ish eigenvalues
    H = H + 1e-12 * rng.normal(size=(n, n)).astype(_DTF)
    H = 0.5 * (H + H.T)
    # inject NaN/Inf sparsely
    H[0, 1] = np.nan
    H[2, 3] = np.inf
    H[3, 2] = -np.inf
    return H


def main() -> None:
    ap = argparse.ArgumentParser(description="Operator stabilization + robust eigen diagnostics")
    ap.add_argument("--operator", type=str, default="runs/artin_operator.npy")
    ap.add_argument("--zeros", type=str, default="data/zeta_zeros.txt")
    ap.add_argument("--k", type=int, default=128)
    ap.add_argument("--out", type=str, default="runs/operator_stability_report.json")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_json = Path(args.out)
    out_dir = out_json.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    if args.dry_run:
        H = _synthetic_ill_conditioned(n=max(64, int(args.k) * 2), seed=int(args.seed))
        zeros = (np.arange(1, 5000, dtype=_DTF) * 0.5 + 10.0).astype(_DTF)
    else:
        H = _load_operator(args.operator)
        zeros = _load_zeros(args.zeros)

    # raw spectrum attempt (diagnostic only)
    eig_before, _, rep_before = safe_eigh(H, stabilize=False, seed=int(args.seed))
    if eig_before.size == 0:
        eig_before = np.array([], dtype=_DTF)

    Hs, srep = stabilize_operator(H, seed=int(args.seed))
    eig_after, _, rep_after = safe_eigh(Hs, stabilize=False, seed=int(args.seed))

    loss, lrep = stable_spectral_loss(Hs, zeros, k=int(args.k), seed=int(args.seed))

    report: Dict[str, Any] = {}
    report.update(srep)
    report["eigh_before"] = rep_before
    report["eigh_after"] = rep_after
    report.update(lrep)
    report["spectral_loss"] = float(lrep.get("spectral_loss", float("inf")))
    report["total_loss"] = float(lrep.get("total_loss", float("inf")))

    # timing
    report["total_time_s"] = float(time.perf_counter() - t0)

    # outputs
    eig_csv = out_dir / "stable_eigenvalues.csv"
    png = out_dir / "spectrum_before_after.png"
    _write_eigs_csv(eig_csv, np.sort(eig_after))
    _plot_before_after(png, eig_before, eig_after, k=int(args.k))

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    if args.dry_run:
        if not out_json.exists() or not eig_csv.exists() or not png.exists():
            raise RuntimeError("dry_run failed to produce expected outputs")

    print(f"stabilization_time_s: {report.get('stabilization_time_s')}", flush=True)
    print(f"eig_time_s(after): {rep_after.get('eig_time_s')}", flush=True)
    print(f"total_time_s: {report.get('total_time_s')}", flush=True)


if __name__ == "__main__":
    main()

