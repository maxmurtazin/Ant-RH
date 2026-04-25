from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.dtes_spectral_operator import (  # noqa: E402
    build_dtes_operator,
    compare_spectral_statistics,
    compute_spectrum,
    save_spectral_report,
    spacing_distribution,
    unfold_spectrum,
)


def _first_present(data: Dict[str, Any], names: Iterable[str]) -> Optional[Any]:
    for name in names:
        if name in data and data[name] is not None:
            return data[name]
    return None


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("input JSON must contain an object at the top level")
    return data


def _extract_inputs(data: Dict[str, Any]) -> Dict[str, Any]:
    t_grid = _first_present(data, ("t_grid", "grid", "t"))
    zeta_abs = _first_present(data, ("zeta_abs", "abs_zeta", "abs_values"))
    pheromone = _first_present(data, ("pheromone", "pheromone_matrix"))
    zeros = _first_present(data, ("true_zeros", "zeta_zeros", "known_zeros"))

    missing = []
    if t_grid is None:
        missing.append("t_grid")
    if zeta_abs is None:
        missing.append("zeta_abs or abs_zeta")
    if pheromone is None:
        missing.append("pheromone or pheromone_matrix")
    if zeros is None:
        missing.append("true_zeros or zeta_zeros")
    if missing:
        raise KeyError(
            "Input is not spectral-ready. Missing fields: "
            + ", ".join(missing)
            + ". Export these from the ACO run: t_grid, zeta_abs, "
            "pheromone_matrix, true_zeros, spectral_ready=true."
        )

    return {
        "t_grid": t_grid,
        "zeta_abs": zeta_abs,
        "pheromone": pheromone,
        "zeta_zeros": zeros,
    }


def _write_eigenvalues_csv(path: Path, eigvals: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "eigenvalue"])
        for i, value in enumerate(eigvals):
            writer.writerow([i, float(value)])


def _maybe_write_plots(out_path: Path, eigvals: np.ndarray, zeros: np.ndarray) -> None:
    plot_cache = out_path.parent / ".matplotlib"
    plot_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(plot_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(plot_cache))
    os.environ.setdefault("MPLBACKEND", "Agg")
    try:
        import matplotlib.pyplot as plt
    except BaseException:
        return

    out_dir = out_path.parent
    n = min(eigvals.size, zeros.size)
    if n > 0:
        eig_unfolded = unfold_spectrum(eigvals)[:n]
        zero_unfolded = unfold_spectrum(zeros)[:n]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(n), eig_unfolded, label="DTES eigenvalues")
        ax.plot(range(n), zero_unfolded, label="zeta zeros")
        ax.set_title("Unfolded Eigenvalues vs Zeta Zeros")
        ax.set_xlabel("index")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "eigenvalues_vs_zeros.png", dpi=180)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(n), eig_unfolded - zero_unfolded)
        ax.set_title("Unfolded Spectral Error Curve")
        ax.set_xlabel("index")
        ax.set_ylabel("unfolded eigenvalue - unfolded zero")
        fig.tight_layout()
        fig.savefig(out_dir / "spectral_error_curve.png", dpi=180)
        plt.close(fig)

    if eigvals.size > 1:
        eig_spacing = spacing_distribution(eigvals)
        zero_spacing = spacing_distribution(zeros)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(eig_spacing, bins=30, alpha=0.75, label="eigenvalues")
        if zero_spacing.size:
            ax.hist(zero_spacing, bins=30, alpha=0.5, label="zeta zeros")
        ax.set_title("Unfolded Spectral Spacing Histogram")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "spectral_spacing_hist.png", dpi=180)
        plt.close(fig)


def run_validation(args: argparse.Namespace) -> Dict[str, Any]:
    data = _load_json(Path(args.input))
    fields = _extract_inputs(data)
    H = build_dtes_operator(
        fields["t_grid"],
        fields["zeta_abs"],
        fields["pheromone"],
        potential_mode=args.potential_mode,
        normalize_laplacian=args.normalized_laplacian,
    )
    eigvals = compute_spectrum(H, k=args.k)
    zeros = np.asarray(fields["zeta_zeros"], dtype=float)
    if args.k and args.k > 0:
        zeros = np.sort(zeros)[: min(int(args.k), zeros.size)]

    spectral_metrics = compare_spectral_statistics(eigvals, zeros)
    symmetry_error = float(np.max(np.abs(H - H.T))) if H.size else 0.0
    report = {
        "eigvals": [float(x) for x in eigvals],
        "unfolded_eigvals": [float(x) for x in unfold_spectrum(eigvals)],
        "zeta_zeros": [float(x) for x in zeros],
        "spectral_metrics": spectral_metrics,
        "operator_shape": list(H.shape),
        "operator_symmetry_error": symmetry_error,
        "potential_mode": args.potential_mode,
        "normalized_laplacian": bool(args.normalized_laplacian),
        "input": str(args.input),
        "spectral_ready": bool(data.get("spectral_ready", False)),
        "pheromone_downsampled": bool(data.get("pheromone_downsampled", False)),
        "pheromone_original_n": data.get("pheromone_original_n"),
    }

    out_path = Path(args.out)
    save_spectral_report(report, out_path)
    _write_eigenvalues_csv(out_path.with_name(out_path.stem + "_eigenvalues.csv"), eigvals)
    _maybe_write_plots(out_path, eigvals, zeros)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate DTES spectral operator output.")
    parser.add_argument("--input", required=True, help="Spectral-ready ACO JSON result")
    parser.add_argument("--out", required=True, help="Output spectral report JSON")
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument(
        "--potential-mode",
        default="neglog",
        choices=("neglog", "log", "inverse"),
    )
    parser.add_argument("--normalized-laplacian", action="store_true")
    args = parser.parse_args()

    try:
        report = run_validation(args)
    except Exception as exc:
        print(f"[ERROR] spectral validation failed: {exc}", file=sys.stderr)
        raise SystemExit(2)

    print(
        "[OK] spectral report saved "
        f"to {args.out} | n={report.get('spectral_metrics', {}).get('n_compared')} "
        f"symmetry_error={report.get('operator_symmetry_error'):.3e}"
    )


if __name__ == "__main__":
    main()
