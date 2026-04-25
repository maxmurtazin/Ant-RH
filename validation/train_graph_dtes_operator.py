from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.dtes_graph_operator import train_graph_dtes_operator


def _first_present(data: Dict[str, Any], keys):
    for key in keys:
        if key in data:
            return data[key]
    return None


def _load_inputs(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    t_grid = _first_present(data, ("t_grid", "grid", "t"))
    zeta_abs = _first_present(data, ("zeta_abs", "abs_zeta", "abs_values"))
    zeta_zeros = _first_present(data, ("zeta_zeros", "true_zeros", "known_zeros"))

    missing = []
    if t_grid is None:
        missing.append("t_grid")
    if zeta_abs is None:
        missing.append("zeta_abs")
    if missing:
        raise ValueError("Missing required input fields: " + ", ".join(missing))
    if zeta_zeros is None or len(zeta_zeros) == 0:
        raise ValueError("zeta_zeros required for Graph-DTES operator learning")

    t_grid = np.asarray(t_grid, dtype=float).reshape(-1)
    zeta_abs = np.asarray(zeta_abs, dtype=float).reshape(-1)
    zeta_zeros = np.asarray(zeta_zeros, dtype=float).reshape(-1)
    finite = np.isfinite(t_grid) & np.isfinite(zeta_abs)
    t_grid = t_grid[finite]
    zeta_abs = zeta_abs[finite]
    zeta_zeros = np.sort(np.abs(zeta_zeros[np.isfinite(zeta_zeros)]))
    zeta_zeros = zeta_zeros[zeta_zeros > 0.0]

    if t_grid.size == 0 or zeta_abs.size != t_grid.size:
        raise ValueError("t_grid and zeta_abs must be finite arrays with matching length")
    if zeta_zeros.size == 0:
        raise ValueError("zeta_zeros required for Graph-DTES operator learning")
    return t_grid, zeta_abs, zeta_zeros


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
                "amplitude_loss",
                "edge_scale",
                "spectral_scale",
                "spectral_shift",
            ],
        )
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def _maybe_write_plots(out_dir: Path, t_grid, zeta_zeros, result, history):
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
    plt.title("Graph-DTES V9 Loss")
    path = out_dir / "graph_dtes_v9_loss.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    plots.append(str(path))

    plt.figure()
    plt.plot(t_grid, result["V"])
    plt.xlabel("t")
    plt.ylabel("V")
    plt.title("Graph-DTES V9 Potential")
    path = out_dir / "graph_dtes_v9_potential.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    plots.append(str(path))

    k = min(len(result["eig_scaled"]), len(zeta_zeros))
    plt.figure()
    plt.plot(range(k), np.sort(result["eig_scaled"])[:k], label="scaled graph eigenvalues")
    plt.plot(range(k), np.sort(zeta_zeros)[:k], label="zeta zeros")
    plt.xlabel("rank")
    plt.ylabel("value")
    plt.title("Graph-DTES V9 Scaled Spectrum")
    plt.legend()
    path = out_dir / "graph_dtes_v9_spectrum_scaled.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    plots.append(str(path))

    plt.figure()
    plt.imshow(result["W"], aspect="auto", interpolation="nearest")
    plt.colorbar(label="weight")
    plt.title("Graph-DTES V9 Adjacency")
    path = out_dir / "graph_dtes_v9_adjacency.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    plots.append(str(path))

    return plots


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train a V9 Graph-DTES operator.")
    parser.add_argument("--input", default="runs/dtes_spectral_input.json")
    parser.add_argument("--out", default="runs/graph_dtes_v9_report.json")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--sigma-t", type=float, default=5.0)
    parser.add_argument("--energy-beta", type=float, default=1.0)
    parser.add_argument("--k-neighbors", type=int, default=12)
    parser.add_argument("--laplacian-type", choices=("normalized", "standard"), default="normalized")
    parser.add_argument("--spectral-calibration", action="store_true")
    parser.add_argument("--smooth-weight", type=float, default=0.01)
    parser.add_argument("--amplitude-weight", type=float, default=0.001)
    parser.add_argument("--max-n", type=int, default=800)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    input_path = Path(args.input)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t_grid, zeta_abs, zeta_zeros = _load_inputs(input_path)
    if len(t_grid) > args.max_n:
        orig_n = len(t_grid)
        idx = np.linspace(0, len(t_grid) - 1, args.max_n).astype(int)
        t_grid = t_grid[idx]
        zeta_abs = zeta_abs[idx]
        print(f"[WARN] downsampled graph from {orig_n} nodes to {args.max_n}")

    result = train_graph_dtes_operator(
        t_grid,
        zeta_abs,
        zeta_zeros,
        steps=args.steps,
        lr=args.lr,
        k=args.k,
        sigma_t=args.sigma_t,
        energy_beta=args.energy_beta,
        k_neighbors=args.k_neighbors,
        laplacian_type=args.laplacian_type,
        spectral_calibration=args.spectral_calibration,
        smooth_weight=args.smooth_weight,
        amplitude_weight=args.amplitude_weight,
        device=args.device,
    )

    potential_path = out_path.parent / "graph_dtes_v9_potential.npy"
    raw_eig_path = out_path.parent / "graph_dtes_v9_eigenvalues_raw.csv"
    scaled_eig_path = out_path.parent / "graph_dtes_v9_eigenvalues_scaled.csv"
    adjacency_path = out_path.parent / "graph_dtes_v9_adjacency.npy"
    operator_path = out_path.parent / "graph_dtes_v9_operator.npy"
    loss_curve_path = out_path.parent / "graph_dtes_v9_loss_curve.csv"

    np.save(potential_path, result["V"])
    np.save(adjacency_path, result["W"])
    np.save(operator_path, result["H"])
    _write_eigenvalues(raw_eig_path, result["eig_raw"])
    _write_eigenvalues(scaled_eig_path, result["eig_scaled"])
    _write_loss_curve(loss_curve_path, result["history"])
    plots = _maybe_write_plots(out_path.parent, t_grid, zeta_zeros, result, result["history"])

    final = result["history"][-1] if result["history"] else {}
    report = {
        "version": "V9 Graph-DTES Operator",
        "input": str(input_path),
        "steps": int(args.steps),
        "k": int(args.k),
        "sigma_t": float(args.sigma_t),
        "energy_beta": float(args.energy_beta),
        "k_neighbors": int(args.k_neighbors),
        "max_n": int(args.max_n),
        "n_nodes": int(len(t_grid)),
        "laplacian_type": str(args.laplacian_type),
        "spectral_calibration": bool(args.spectral_calibration),
        "smooth_weight": float(args.smooth_weight),
        "amplitude_weight": float(args.amplitude_weight),
        "final_loss": final.get("loss"),
        "final_spectral_loss": final.get("spectral_loss"),
        "final_spacing_loss": final.get("spacing_loss"),
        "edge_scale": result["edge_scale"],
        "spectral_scale": result["spectral_scale"],
        "spectral_shift": result["spectral_shift"],
        "outputs": {
            "potential": str(potential_path),
            "eigenvalues_raw": str(raw_eig_path),
            "eigenvalues_scaled": str(scaled_eig_path),
            "adjacency": str(adjacency_path),
            "operator": str(operator_path),
            "loss_curve": str(loss_curve_path),
            "plots": plots,
        },
        "history": result["history"],
        "note": "Numerical DTES graph operator fitting experiment; not a proof of RH.",
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, allow_nan=False)

    print(f"[V9] report saved: {out_path}")
    print(f"[V9] potential saved: {potential_path}")
    print(f"[V9] raw eigenvalues saved: {raw_eig_path}")
    print(f"[V9] scaled eigenvalues saved: {scaled_eig_path}")
    print(f"[V9] adjacency saved: {adjacency_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
