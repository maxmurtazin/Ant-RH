from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.dtes_operator_physics import locality_penalty, train_physics_operator


def _first_present(data: Dict[str, Any], keys):
    for key in keys:
        if key in data:
            return data[key]
    return None


def _load_inputs(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    t_grid = _first_present(data, ("t_grid", "grid", "t"))
    zeta_zeros = _first_present(data, ("zeta_zeros", "true_zeros", "known_zeros"))
    if t_grid is None:
        raise ValueError("t_grid required for physics-constrained operator learning")
    if zeta_zeros is None or len(zeta_zeros) == 0:
        raise ValueError("zeta_zeros required for physics-constrained operator learning")

    t_grid = np.asarray(t_grid, dtype=float).reshape(-1)
    zeta_zeros = np.asarray(zeta_zeros, dtype=float).reshape(-1)
    t_grid = t_grid[np.isfinite(t_grid)]
    zeta_zeros = np.sort(np.abs(zeta_zeros[np.isfinite(zeta_zeros)]))
    zeta_zeros = zeta_zeros[zeta_zeros > 0.0]
    if t_grid.size < 2:
        raise ValueError("t_grid must contain at least two finite values")
    if zeta_zeros.size == 0:
        raise ValueError("zeta_zeros required for physics-constrained operator learning")
    return t_grid, zeta_zeros


def _ensure_uniform_grid(t_grid: np.ndarray):
    t_grid = np.sort(np.asarray(t_grid, dtype=float).reshape(-1))
    diffs = np.diff(t_grid)
    if diffs.size == 0 or np.any(diffs <= 0.0):
        raise ValueError("t_grid must be strictly increasing")

    dx = float(np.median(diffs))
    if not np.allclose(diffs, dx, rtol=1e-4, atol=max(1e-10, abs(dx) * 1e-6)):
        print("[WARN] t_grid is not uniform; resampling to a uniform grid")
        return np.linspace(float(t_grid[0]), float(t_grid[-1]), t_grid.size), True
    return t_grid, False


def _write_eigenvalues(path: Path, eigvals):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "eigenvalue"])
        for i, val in enumerate(eigvals, start=1):
            writer.writerow([i, float(val)])


def _write_loss_curve(path: Path, history):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "loss",
                "spectral_loss",
                "spacing_loss",
                "smoothness_loss",
                "curvature_loss",
                "tv_loss",
                "amplitude_loss",
                "frequency_loss",
                "spectral_scale",
                "spectral_shift",
                "calibration_reg",
                "raw_eig_min",
                "raw_eig_max",
                "scaled_eig_min",
                "scaled_eig_max",
                "prime_loss",
            ],
        )
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def _maybe_write_plots(out_dir: Path, t_grid, potential, eigvals_raw, eigvals_scaled, zeta_zeros, history):
    plot_cache = out_dir / ".matplotlib"
    plot_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(plot_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(plot_cache))
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    plots = []
    steps = [row["step"] for row in history]
    losses = [row["loss"] for row in history]
    plt.figure()
    plt.plot(steps, losses)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("V8.3 Physics Operator Loss")
    path = out_dir / "loss_v83.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    plots.append(str(path))

    plt.figure()
    plt.plot(t_grid, potential)
    plt.xlabel("t")
    plt.ylabel("V(t)")
    plt.title("V8.3 Learned Schrödinger Potential")
    path = out_dir / "potential_v83.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    plots.append(str(path))

    k = min(len(eigvals_raw), len(zeta_zeros))
    plt.figure()
    plt.plot(range(k), np.sort(eigvals_raw)[:k], label="raw physics eigenvalues")
    plt.plot(range(k), np.sort(zeta_zeros)[:k], label="zeta zeros")
    plt.xlabel("rank")
    plt.ylabel("value")
    plt.title("V8.3 Raw Physics Spectrum vs Zeta Zeros")
    plt.legend()
    path = out_dir / "spectrum_v83_raw.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    plots.append(str(path))

    k = min(len(eigvals_scaled), len(zeta_zeros))
    plt.figure()
    plt.plot(range(k), np.sort(eigvals_scaled)[:k], label="scaled physics eigenvalues")
    plt.plot(range(k), np.sort(zeta_zeros)[:k], label="zeta zeros")
    plt.xlabel("rank")
    plt.ylabel("value")
    plt.title("V8.3 Scaled Physics Spectrum vs Zeta Zeros")
    plt.legend()
    path = out_dir / "spectrum_v83_scaled.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    plots.append(str(path))

    return plots


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train a physics-constrained Schrödinger DTES operator."
    )
    parser.add_argument("--input", default="runs/dtes_spectral_input.json")
    parser.add_argument("--out", default="runs/physics_operator_report.json")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--smooth-weight", type=float, default=0.01)
    parser.add_argument("--curvature-weight", type=float, default=0.1)
    parser.add_argument("--tv-weight", type=float, default=0.01)
    parser.add_argument("--amplitude-weight", type=float, default=0.001)
    parser.add_argument("--frequency-weight", type=float, default=0.1)
    parser.add_argument("--frequency-cutoff-ratio", type=float, default=0.20)
    parser.add_argument("--potential-type", type=str, default="free", choices=["free", "lowfreq"])
    parser.add_argument("--n-modes", type=int, default=12)
    parser.add_argument("--calibration-weight", type=float, default=1e-4)
    parser.add_argument("--spectral-calibration", action="store_true")
    parser.add_argument("--prime-weight", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    input_path = Path(args.input)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t_grid, zeta_zeros = _load_inputs(input_path)
    t_grid, resampled = _ensure_uniform_grid(t_grid)
    model, history = train_physics_operator(
        t_grid,
        zeta_zeros,
        steps=args.steps,
        lr=args.lr,
        k=args.k,
        prime_weight=args.prime_weight,
        smooth_weight=args.smooth_weight,
        curvature_weight=args.curvature_weight,
        tv_weight=args.tv_weight,
        amplitude_weight=args.amplitude_weight,
        frequency_weight=args.frequency_weight,
        frequency_cutoff_ratio=args.frequency_cutoff_ratio,
        potential_type=args.potential_type,
        n_modes=args.n_modes,
        spectral_calibration=args.spectral_calibration,
        calibration_weight=args.calibration_weight,
        device=args.device,
    )

    with torch.no_grad():
        H = model()
        eigvals_raw_t = torch.linalg.eigvalsh(H)
        calibration = getattr(model, "spectral_calibration", None)
        if calibration is not None:
            eigvals_scaled_t, spectral_scale_t, spectral_shift_t = calibration(eigvals_raw_t)
        else:
            eigvals_scaled_t = eigvals_raw_t
            spectral_scale_t = torch.ones(1, dtype=torch.float32, device=H.device)
            spectral_shift_t = torch.zeros(1, dtype=torch.float32, device=H.device)
        eigvals_raw = eigvals_raw_t.detach().cpu().numpy()
        eigvals_scaled = eigvals_scaled_t.detach().cpu().numpy()
        spectral_scale = float(spectral_scale_t.detach().cpu().reshape(-1)[0])
        spectral_shift = float(spectral_shift_t.detach().cpu().reshape(-1)[0])
        potential = model.V().detach().cpu().numpy()
        band_penalty = float(locality_penalty(H).detach().cpu())

    potential_path = out_path.parent / "potential_v83.npy"
    eigenvalues_raw_path = out_path.parent / "eigenvalues_v83_raw.csv"
    eigenvalues_scaled_path = out_path.parent / "eigenvalues_v83_scaled.csv"
    loss_curve_path = out_path.parent / "loss_curve_v83.csv"

    np.save(potential_path, potential)
    _write_eigenvalues(eigenvalues_raw_path, eigvals_raw)
    _write_eigenvalues(eigenvalues_scaled_path, eigvals_scaled)
    _write_loss_curve(loss_curve_path, history)
    plots = _maybe_write_plots(
        out_path.parent,
        t_grid,
        potential,
        eigvals_raw,
        eigvals_scaled,
        zeta_zeros,
        history,
    )

    report = {
        "version": "V8.3",
        "input": str(input_path),
        "operator": "SchrodingerOperator",
        "model": "H = -Delta + V(x)",
        "potential_type": str(args.potential_type),
        "n_modes": int(args.n_modes),
        "spectral_calibration": bool(args.spectral_calibration),
        "calibration_weight": float(args.calibration_weight),
        "steps": int(args.steps),
        "lr": float(args.lr),
        "k": int(args.k),
        "smooth_weight": float(args.smooth_weight),
        "curvature_weight": float(args.curvature_weight),
        "tv_weight": float(args.tv_weight),
        "amplitude_weight": float(args.amplitude_weight),
        "frequency_weight": float(args.frequency_weight),
        "frequency_cutoff_ratio": float(args.frequency_cutoff_ratio),
        "prime_weight": float(args.prime_weight),
        "grid_size": int(len(t_grid)),
        "dx": float(t_grid[1] - t_grid[0]),
        "resampled_uniform_grid": bool(resampled),
        "initial_loss": history[0]["loss"] if history else None,
        "final_loss": history[-1]["loss"] if history else None,
        "final_spectral_loss": history[-1]["spectral_loss"] if history else None,
        "final_spacing_loss": history[-1]["spacing_loss"] if history else None,
        "final_smoothness_loss": history[-1]["smoothness_loss"] if history else None,
        "final_curvature_loss": history[-1]["curvature_loss"] if history else None,
        "final_tv_loss": history[-1]["tv_loss"] if history else None,
        "final_amplitude_loss": history[-1]["amplitude_loss"] if history else None,
        "final_frequency_loss": history[-1]["frequency_loss"] if history else None,
        "final_spectral_scale": spectral_scale,
        "final_spectral_shift": spectral_shift,
        "final_calibration_reg": history[-1]["calibration_reg"] if history else None,
        "locality_penalty": band_penalty,
        "potential": str(potential_path),
        "eigenvalues_raw": str(eigenvalues_raw_path),
        "eigenvalues_scaled": str(eigenvalues_scaled_path),
        "loss_curve": str(loss_curve_path),
        "plots": plots,
        "history": history,
        "note": (
            "V8.3 calibrated low-frequency physics-constrained inverse spectral fitting. "
            "Numerical experiment, not proof of RH."
        ),
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)

    print(f"[V8.3] report saved: {out_path}")
    print(f"[V8.3] potential saved: {potential_path}")
    print(f"[V8.3] raw eigenvalues saved: {eigenvalues_raw_path}")
    print(f"[V8.3] scaled eigenvalues saved: {eigenvalues_scaled_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
