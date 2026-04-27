from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PLOT_CACHE = ROOT / ".matplotlib"
PLOT_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(PLOT_CACHE))
os.environ.setdefault("XDG_CACHE_HOME", str(PLOT_CACHE))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt

from core.operator_health import validate_operator_health


def load_operator(path):
    path = str(path)
    if path.endswith(".pt"):
        return torch.load(path, map_location="cpu")
    arr = np.load(path)
    return torch.tensor(arr)


def _complex_dtype(dtype):
    return torch.complex128 if dtype == torch.float64 else torch.complex64


def make_position_observable(n, dtype):
    x = torch.linspace(-1.0, 1.0, n)
    return torch.diag(x).to(dtype)


def make_momentum_observable(n, dtype):
    P = torch.zeros(n, n)
    for i in range(n - 1):
        P[i, i + 1] = 1.0
        P[i + 1, i] = -1.0
    if not torch.is_complex(torch.empty((), dtype=dtype)):
        dtype = _complex_dtype(dtype)
    return (1j * P).to(dtype)


def compute_otoc(H, t_max=10.0, n_times=200):
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError("operator must be a square matrix")

    if not torch.is_complex(H):
        H = H.to(_complex_dtype(H.dtype))

    n = H.shape[0]
    eigvals, U = torch.linalg.eigh(H)

    dtype = H.dtype
    W = make_position_observable(n, dtype)
    V = make_momentum_observable(n, dtype)
    W_e = U.conj().T @ W @ U

    times = torch.linspace(0, float(t_max), int(n_times), device=H.device)
    C_vals = []

    with torch.no_grad():
        for t in times:
            phase = torch.exp(1j * eigvals * t)

            Wt_e = phase[:, None] * W_e * torch.conj(phase[None, :])
            Wt = U @ Wt_e @ U.conj().T

            comm = Wt @ V - V @ Wt
            C = torch.real(torch.trace(comm.conj().T @ comm)) / n

            C_vals.append(float(C.detach().cpu()))

    return times.detach().cpu().numpy(), np.array(C_vals, dtype=float)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compute V10.11 OTOC diagnostic.")
    parser.add_argument("--operator", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--t-max", type=float, default=10.0)
    parser.add_argument("--n-times", type=int, default=200)
    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    H_raw = load_operator(args.operator)

    H, eig, health = validate_operator_health(H_raw)
    if eig is None:
        reason = health.get("failure_reason") or "eigvalsh failed"
        raise RuntimeError(f"Operator failed spectral validation: {reason}")
    if not health["self_adjoint_pass"]:
        raise RuntimeError("Operator failed self-adjoint validation")
    if not health["finite_outputs"]:
        raise RuntimeError("Operator failed finite-output validation")

    times, C = compute_otoc(H, args.t_max, args.n_times)
    if not np.all(np.isfinite(C)):
        raise FloatingPointError("OTOC curve contains NaN or Inf")

    np.savetxt(
        out_dir / "otoc_curve.csv",
        np.stack([times, C], axis=1),
        delimiter=",",
        header="t,C(t)",
        comments="",
    )

    plt.figure(figsize=(6, 4))
    plt.plot(times, C)
    plt.xlabel("t")
    plt.ylabel("C(t)")
    plt.title("V10.11 OTOC diagnostic")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "otoc_curve.png", dpi=160)
    plt.close()

    C_safe = np.maximum(C, 1e-12)
    mask = times < 0.3 * float(args.t_max)
    if mask.sum() > 5:
        slope = float(np.polyfit(times[mask], np.log(C_safe[mask]), 1)[0])
    else:
        slope = None

    report = {
        "version": "V10.11 OTOC diagnostic",
        "operator": str(args.operator),
        "self_adjoint_pass": health["self_adjoint_pass"],
        "hermitian_error_projected": health["hermitian_error_projected"],
        "finite_outputs": health["finite_outputs"],
        "spectral_pass": health["spectral_pass"],
        "used_jitter": health["used_jitter"],
        "otoc_initial": float(C[0]),
        "otoc_max": float(C.max()),
        "otoc_final": float(C[-1]),
        "growth_rate_proxy": slope,
        "note": (
            "operator-level Hilbert–Pólya compatibility check; OTOC diagnostic "
            "for quantum-chaos-like behavior; not a proof of RH."
        ),
    }

    with (out_dir / "otoc_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("[V10.11] OTOC completed ->", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
