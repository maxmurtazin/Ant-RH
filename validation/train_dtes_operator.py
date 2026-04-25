from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.dtes_operator_learning import train_operator
from core.dtes_spectral_operator import build_dtes_operator


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
    pheromone = _first_present(data, ("pheromone_matrix", "pheromone"))
    zeta_zeros = _first_present(data, ("zeta_zeros", "true_zeros", "known_zeros"))

    missing = []
    if t_grid is None:
        missing.append("t_grid")
    if zeta_abs is None:
        missing.append("zeta_abs")
    if pheromone is None:
        missing.append("pheromone_matrix")
    if missing:
        raise KeyError("Missing required input fields: " + ", ".join(missing))

    zeta_zeros = [] if zeta_zeros is None else zeta_zeros
    if len(zeta_zeros) == 0:
        raise ValueError("zeta_zeros required for direct operator learning")

    return (
        np.asarray(t_grid, dtype=float),
        np.asarray(zeta_abs, dtype=float),
        np.asarray(pheromone, dtype=float),
        np.asarray(zeta_zeros, dtype=float),
    )


def _write_loss_curve(path: Path, history):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "loss",
                "spectral_loss",
                "spacing_loss",
                "prime_loss",
                "regularization",
            ],
        )
        writer.writeheader()
        for row in history:
            writer.writerow(row)


def _write_eigenvalues(path: Path, eigvals):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "eigenvalue"])
        for i, val in enumerate(eigvals, start=1):
            writer.writerow([i, float(val)])


def _maybe_write_plots(out_dir: Path, history, eigvals, zeta_zeros):
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
    plt.title("DTES Operator Learning Loss")
    path = out_dir / "operator_learning_loss.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    plots.append(str(path))

    k = min(len(eigvals), len(zeta_zeros))
    plt.figure()
    plt.plot(range(k), np.sort(eigvals)[:k], label="learned eigenvalues")
    plt.plot(range(k), np.sort(zeta_zeros)[:k], label="zeta zeros")
    plt.xlabel("rank")
    plt.ylabel("value")
    plt.title("Learned Spectrum vs Zeta Zeros")
    plt.legend()
    path = out_dir / "learned_vs_true_spectrum.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    plots.append(str(path))

    if k > 1:
        plt.figure()
        plt.hist(np.diff(np.sort(eigvals)[:k]), alpha=0.6, label="learned spacing")
        plt.hist(np.diff(np.sort(zeta_zeros)[:k]), alpha=0.6, label="zero spacing")
        plt.xlabel("spacing")
        plt.ylabel("count")
        plt.title("Learned Spacing Distribution")
        plt.legend()
        path = out_dir / "learned_spacing_distribution.png"
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        plots.append(str(path))

    return plots


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train a learnable self-adjoint DTES operator from spectral input."
    )
    parser.add_argument("--input", default="runs/dtes_spectral_input.json")
    parser.add_argument("--out", default="runs/operator_learning_report.json")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--spacing-weight", type=float, default=0.5)
    parser.add_argument("--reg-weight", type=float, default=1e-3)
    parser.add_argument("--prime-aware", action="store_true")
    parser.add_argument("--prime-weight", type=float, default=0.1)
    parser.add_argument("--prime-x-max", type=int, default=500)
    parser.add_argument("--prime-temperature", type=float, default=0.25)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    input_path = Path(args.input)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        t_grid, zeta_abs, pheromone, zeta_zeros = _load_inputs(input_path)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    init_operator = build_dtes_operator(t_grid, zeta_abs, pheromone)
    H_final, eig_final, history = train_operator(
        init_operator,
        zeta_zeros,
        steps=args.steps,
        lr=args.lr,
        k=args.k,
        spacing_weight=args.spacing_weight,
        reg_weight=args.reg_weight,
        prime_weight=args.prime_weight if args.prime_aware else 0.0,
        prime_x_max=args.prime_x_max,
        prime_temperature=args.prime_temperature,
        device=args.device,
    )

    learned_operator_path = out_path.parent / "learned_operator.npy"
    learned_eig_path = out_path.parent / "learned_eigenvalues.csv"
    loss_curve_path = out_path.parent / "loss_curve.csv"

    np.save(learned_operator_path, H_final)
    _write_eigenvalues(learned_eig_path, eig_final)
    _write_loss_curve(loss_curve_path, history)
    plots = _maybe_write_plots(out_path.parent, history, eig_final, zeta_zeros)

    report = {
        "input": str(input_path),
        "initial_loss": history[0]["loss"] if history else None,
        "final_loss": history[-1]["loss"] if history else None,
        "steps": int(args.steps),
        "lr": float(args.lr),
        "k": int(args.k),
        "prime_aware": bool(args.prime_aware),
        "prime_weight": float(args.prime_weight if args.prime_aware else 0.0),
        "prime_x_max": int(args.prime_x_max),
        "prime_temperature": float(args.prime_temperature),
        "final_prime_loss": history[-1].get("prime_loss") if history else None,
        "operator_shape": list(H_final.shape),
        "learned_operator": str(learned_operator_path),
        "learned_eigenvalues": str(learned_eig_path),
        "loss_curve": str(loss_curve_path),
        "plots": plots,
        "history": history,
        "note": (
            "Experimental Hilbert-Polya-inspired numerical spectral fitting; "
            "not a proof of the Riemann hypothesis."
        ),
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[operator-learning] report saved: {out_path}")
    print(f"[operator-learning] learned eigenvalues saved: {learned_eig_path}")
    print(f"[operator-learning] learned operator saved: {learned_operator_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
