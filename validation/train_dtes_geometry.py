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

from core.dtes_geometry import train_dtes_geometry
from core.pauli import pauli_mask


def _first_present(data: Dict[str, Any], keys):
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def _load_inputs(path: Path, pauli_aware: bool = False):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("input JSON must contain an object at the top level")

    t_grid = _first_present(data, ("t_grid", "grid", "t"))
    zeta_zeros = _first_present(data, ("zeta_zeros", "true_zeros", "known_zeros"))
    particle_configs = _first_present(data, ("particle_configs",))

    missing = []
    if t_grid is None:
        missing.append("t_grid")
    if zeta_zeros is None:
        missing.append("zeta_zeros")
    if missing:
        raise ValueError("Missing required input fields: " + ", ".join(missing))

    t_grid = np.asarray(t_grid, dtype=float).reshape(-1)
    zeta_zeros = np.asarray(zeta_zeros, dtype=float).reshape(-1)
    particle_configs = list(particle_configs) if particle_configs is not None else None

    finite_t = np.isfinite(t_grid)
    if particle_configs is not None:
        if len(particle_configs) != t_grid.size:
            raise ValueError("particle_configs length must match t_grid length")
        particle_configs = [
            c for c, keep in zip(particle_configs, finite_t) if bool(keep)
        ]
    t_grid = t_grid[finite_t]
    zeta_zeros = np.sort(np.abs(zeta_zeros[np.isfinite(zeta_zeros)]))
    zeta_zeros = zeta_zeros[zeta_zeros > 0.0]

    pauli_stats = {
        "pauli_aware": bool(pauli_aware),
        "pauli_valid_count": None,
        "pauli_total_count": None,
        "pauli_removed_count": None,
        "particle_configs_present": particle_configs is not None,
        "zeta_zeros_filtered_by_pauli": False,
    }

    if pauli_aware:
        if particle_configs is None:
            print(
                "[Pauli] no particle_configs found; geometry training proceeds "
                "without Pauli filtering"
            )
        else:
            mask = pauli_mask(particle_configs)
            valid_count = int(sum(mask))
            total_count = int(len(mask))
            print(f"[Pauli] valid states: {valid_count} / {total_count}")
            t_grid = np.asarray(
                [t for t, keep in zip(t_grid, mask) if keep], dtype=float
            )
            particle_configs = [
                c for c, keep in zip(particle_configs, mask) if keep
            ]
            if zeta_zeros.size == total_count:
                zeta_zeros = np.asarray(
                    [z for z, keep in zip(zeta_zeros, mask) if keep], dtype=float
                )
                zeta_zeros = np.sort(np.abs(zeta_zeros[np.isfinite(zeta_zeros)]))
                zeta_zeros = zeta_zeros[zeta_zeros > 0.0]
                pauli_stats["zeta_zeros_filtered_by_pauli"] = True
            pauli_stats.update(
                {
                    "pauli_valid_count": valid_count,
                    "pauli_total_count": total_count,
                    "pauli_removed_count": int(total_count - valid_count),
                }
            )

    if t_grid.size == 0:
        raise ValueError("t_grid must contain at least one finite value")
    if zeta_zeros.size == 0:
        raise ValueError("zeta_zeros must contain at least one positive finite value")

    return t_grid, zeta_zeros, pauli_stats


def _write_eigenvalues(path: Path, eigvals):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "eigenvalue"])
        for i, val in enumerate(eigvals, start=1):
            writer.writerow([i, float(val)])


def _history_loss(row):
    return float(row.get("loss", 0.0)) if isinstance(row, dict) else float(row)


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if np.isfinite(value) else None
    return value


def _write_weights(path: Path, history):
    weight_names = []
    for row in history:
        if isinstance(row, dict) and isinstance(row.get("weights"), dict):
            for name in row["weights"]:
                if name not in weight_names:
                    weight_names.append(name)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", *weight_names])
        for i, row in enumerate(history):
            if not isinstance(row, dict):
                continue
            weights = row.get("weights", {})
            writer.writerow(
                [
                    int(row.get("step", i)),
                    *[float(weights.get(name, 0.0)) for name in weight_names],
                ]
            )


def _write_loss_curve(path: Path, history):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss"])
        for i, row in enumerate(history):
            if not isinstance(row, dict):
                continue
            writer.writerow([int(row.get("step", i)), float(row.get("loss", 0.0))])


def _maybe_write_plots(out_dir: Path, zeta_zeros, result):
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
    eig = np.asarray(result["eig"], dtype=float)
    Z = np.asarray(result["Z"], dtype=float)
    W = np.asarray(result["W"], dtype=float)
    history = result["history"]
    amplitude = np.asarray(result.get("amplitude", []), dtype=float)
    phase = np.asarray(result.get("phase", []), dtype=float)
    nodal = np.asarray(result.get("nodal_score", []), dtype=float)

    k = min(eig.size, zeta_zeros.size)
    if k > 0:
        plt.figure()
        plt.plot(range(k), np.sort(eig)[:k], label="DTES V10 eigenvalues")
        plt.plot(range(k), np.sort(zeta_zeros)[:k], label="zeta zeros")
        plt.xlabel("rank")
        plt.ylabel("value")
        plt.title("DTES V10 Spectrum")
        plt.legend()
        path = out_dir / "dtes_v10_spectrum.png"
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        plots.append(str(path))

    if Z.size:
        plt.figure()
        if Z.shape[1] >= 2:
            plt.scatter(Z[:, 0], Z[:, 1], s=12, alpha=0.8)
            plt.xlabel("Z0")
            plt.ylabel("Z1")
        else:
            plt.plot(np.arange(Z.shape[0]), Z[:, 0], marker=".", linestyle="none")
            plt.xlabel("node")
            plt.ylabel("Z0")
        plt.title("DTES V10 Learned Embedding")
        path = out_dir / "dtes_v10_embedding.png"
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        plots.append(str(path))

    plt.figure()
    plt.imshow(W, aspect="auto", interpolation="nearest")
    plt.colorbar(label="weight")
    plt.title("DTES V10 Adjacency")
    path = out_dir / "dtes_v10_adjacency.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    plots.append(str(path))

    if history:
        plt.figure()
        plt.plot(np.arange(len(history)), [_history_loss(row) for row in history])
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.title("DTES V10 Loss")
        path = out_dir / "dtes_v10_loss.png"
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        plots.append(str(path))

    if amplitude.size:
        plt.figure()
        plt.plot(np.arange(amplitude.size), amplitude)
        plt.xlabel("node")
        plt.ylabel("amplitude")
        plt.title("DTES V10.6 Wavefunction Amplitude")
        path = out_dir / "dtes_v10_6_amplitude.png"
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        plots.append(str(path))

    if phase.size:
        plt.figure()
        plt.plot(np.arange(phase.size), phase)
        plt.xlabel("node")
        plt.ylabel("phase")
        plt.title("DTES V10.6 Wavefunction Phase")
        path = out_dir / "dtes_v10_6_phase.png"
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        plots.append(str(path))

    if nodal.size:
        plt.figure()
        plt.plot(np.arange(nodal.size), nodal)
        plt.xlabel("node")
        plt.ylabel("nodal score")
        plt.title("DTES V10.6 Nodal Score")
        path = out_dir / "dtes_v10_6_nodal_score.png"
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        plots.append(str(path))

    weight_rows = [
        row for row in history if isinstance(row, dict) and isinstance(row.get("weights"), dict)
    ]
    if weight_rows:
        weight_names = list(weight_rows[-1]["weights"])
        plt.figure()
        for name in weight_names:
            plt.plot(
                [int(row.get("step", i)) for i, row in enumerate(weight_rows)],
                [float(row["weights"].get(name, 0.0)) for row in weight_rows],
                label=name,
            )
        plt.xlabel("step")
        plt.ylabel("weight")
        plt.title("DTES V10 Adaptive Loss Weights")
        plt.legend()
        path = out_dir / "dtes_v10_weight_curves.png"
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        plots.append(str(path))

    return plots


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train V10 DTES geometry from spectral input."
    )
    parser.add_argument("--input", default="runs/dtes_spectral_input.json")
    parser.add_argument("--out", default="runs/dtes_v10_report.json")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--dim", type=int, default=2)
    parser.add_argument("--max-n", type=int, default=400)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pauli-aware", action="store_true")
    parser.add_argument("--pauli-weight", type=float, default=1.0)
    parser.add_argument("--adaptive-weights", action="store_true")
    parser.add_argument("--adaptive-lr", type=float, default=0.05)
    parser.add_argument("--wavefunction-aware", action="store_true")
    parser.add_argument("--no-scaling", action="store_true")
    parser.add_argument("--generalization-split", action="store_true")
    parser.add_argument("--anchor-loss", action="store_true")
    parser.add_argument("--anchor-weight", type=float, default=10.0)
    parser.add_argument("--spacing-anchor-weight", type=float, default=5.0)
    parser.add_argument("--n-anchor", type=int, default=5)
    parser.add_argument("--line-geometry", action="store_true")
    parser.add_argument("--spacing-weight", type=float, default=1.0)
    parser.add_argument("--ordering-weight", type=float, default=0.5)
    parser.add_argument("--center-weight", type=float, default=0.1)
    parser.add_argument("--parametric-line", action="store_true")
    parser.add_argument("--curve-smooth-weight", type=float, default=1.0)
    parser.add_argument("--curve-amp-weight", type=float, default=0.1)
    parser.add_argument("--multiscale-geometry", action="store_true")
    parser.add_argument("--range-weight", type=float, default=10.0)
    parser.add_argument("--nested-geometry", action="store_true")
    parser.add_argument("--wavefunction-topology", action="store_true")
    parser.add_argument("--wave-amp-weight", type=float, default=0.2)
    parser.add_argument("--phase-weight", type=float, default=0.05)
    parser.add_argument("--nodal-weight", type=float, default=0.1)
    parser.add_argument("--amplitude-collapse-weight", type=float, default=0.1)
    parser.add_argument("--phase-activity-weight", type=float, default=1.0)
    parser.add_argument("--phase-target-std", type=float, default=0.5)
    parser.add_argument("--phase-curvature-weight", type=float, default=0.05)
    parser.add_argument("--phase-coupled-operator", action="store_true")
    parser.add_argument("--phase-coupling", type=float, default=1.0)
    parser.add_argument("--phase-coupling-positive", action="store_true")
    parser.add_argument("--magnetic-operator", action="store_true")
    parser.add_argument("--magnetic-beta", type=float, default=1.0)
    parser.add_argument("--fractional-operator", action="store_true")
    parser.add_argument("--fractional-alpha", type=float, default=0.7)
    parser.add_argument("--detach-fractional", action="store_true")
    parser.add_argument("--self-adjoint-tol", type=float, default=1e-8)
    parser.add_argument("--strict-self-adjoint", action="store_true")
    parser.add_argument("--jitter-eps", type=float, default=1e-8)
    parser.add_argument("--log-operator-health", action="store_true")
    parser.add_argument("--save-operator-health", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    input_path = Path(args.input)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    t_grid, zeta_zeros, pauli_stats = _load_inputs(
        input_path, pauli_aware=args.pauli_aware
    )
    n_input = int(t_grid.size)
    n_nodes = int(min(t_grid.size, args.max_n))
    detach_fractional = bool(args.detach_fractional or args.fractional_operator)
    v10_10_output = bool(
        args.log_operator_health
        or args.save_operator_health
        or "v10_10_self_adjoint" in str(out_path.parent)
    )

    result = train_dtes_geometry(
        t_grid,
        zeta_zeros,
        steps=args.steps,
        lr=args.lr,
        k=args.k,
        dim=args.dim,
        max_n=args.max_n,
        device=args.device,
        pauli_weight=args.pauli_weight,
        adaptive_weights=args.adaptive_weights,
        adaptive_lr=args.adaptive_lr,
        no_scaling=args.no_scaling,
        generalization_split=args.generalization_split,
        anchor_loss_enabled=args.anchor_loss,
        anchor_weight=args.anchor_weight,
        spacing_anchor_weight=args.spacing_anchor_weight,
        n_anchor=args.n_anchor,
        line_geometry=args.line_geometry,
        spacing_weight=args.spacing_weight,
        ordering_weight=args.ordering_weight,
        center_weight=args.center_weight,
        parametric_line=args.parametric_line,
        curve_smooth_weight=args.curve_smooth_weight,
        curve_amp_weight=args.curve_amp_weight,
        multiscale_geometry=args.multiscale_geometry,
        range_weight=args.range_weight,
        nested_geometry=args.nested_geometry,
        wavefunction_topology=args.wavefunction_topology,
        wave_amp_weight=args.wave_amp_weight,
        phase_weight=args.phase_weight,
        nodal_weight=args.nodal_weight,
        amplitude_collapse_weight=args.amplitude_collapse_weight,
        phase_activity_weight=args.phase_activity_weight,
        phase_target_std=args.phase_target_std,
        phase_curvature_weight=args.phase_curvature_weight,
        phase_coupled_operator=args.phase_coupled_operator,
        phase_coupling=args.phase_coupling,
        phase_coupling_positive=args.phase_coupling_positive,
        magnetic_operator=args.magnetic_operator,
        magnetic_beta=args.magnetic_beta,
        fractional_operator=args.fractional_operator,
        fractional_alpha=args.fractional_alpha,
        detach_fractional=detach_fractional,
        self_adjoint_tol=args.self_adjoint_tol,
        strict_self_adjoint=args.strict_self_adjoint,
        jitter_eps=args.jitter_eps,
        log_operator_health=args.log_operator_health,
    )

    embedding_path = out_path.parent / "dtes_v10_embedding.npy"
    adjacency_path = out_path.parent / "dtes_v10_adjacency.npy"
    potential_path = out_path.parent / "dtes_v10_potential.npy"
    eigenvalues_path = (
        out_path.parent / "eigenvalues.csv"
        if v10_10_output
        else out_path.parent / "dtes_v10_eigenvalues.csv"
    )
    weights_path = out_path.parent / "dtes_v10_weights.csv"
    loss_curve_path = out_path.parent / "loss_curve.csv"
    amplitude_path = out_path.parent / "dtes_v10_6_amplitude.npy"
    phase_path = out_path.parent / "dtes_v10_6_phase.npy"
    psi_real_path = out_path.parent / "dtes_v10_6_psi_real.npy"
    psi_imag_path = out_path.parent / "dtes_v10_6_psi_imag.npy"
    learned_operator_path = out_path.parent / "learned_operator.npy"
    operator_real_path = (
        out_path.parent / "learned_operator_real.npy"
        if v10_10_output
        else out_path.parent / "dtes_v10_8_operator_real.npy"
    )
    operator_imag_path = (
        out_path.parent / "learned_operator_imag.npy"
        if v10_10_output
        else out_path.parent / "dtes_v10_8_operator_imag.npy"
    )
    operator_health_path = out_path.parent / "operator_health.json"
    v10_10_dir = Path("runs/v10_10_self_adjoint")
    v10_10_dir.mkdir(parents=True, exist_ok=True)
    v10_10_operator_path = v10_10_dir / "learned_operator.npy"
    v10_10_operator_real_path = v10_10_dir / "learned_operator_real.npy"
    v10_10_operator_imag_path = v10_10_dir / "learned_operator_imag.npy"
    v10_10_eigenvalues_path = v10_10_dir / "eigenvalues.csv"
    v10_10_report_path = v10_10_dir / "report.json"

    np.save(embedding_path, result["Z"])
    np.save(adjacency_path, result["W"])
    np.save(potential_path, result["V"])
    H_result = np.asarray(result["H"])
    operator_is_complex = bool(np.iscomplexobj(H_result))
    np.save(v10_10_operator_path, H_result)
    if operator_is_complex:
        np.save(v10_10_operator_real_path, H_result.real)
        np.save(v10_10_operator_imag_path, H_result.imag)
    np.savetxt(v10_10_eigenvalues_path, result["eig"], delimiter=",")
    print(f"[V10.10] Operator saved to {v10_10_dir}")
    if args.wavefunction_topology:
        np.save(amplitude_path, result["amplitude"])
        np.save(phase_path, result["phase"])
        np.save(psi_real_path, result["psi_real"])
        np.save(psi_imag_path, result["psi_imag"])
    if args.magnetic_operator:
        np.save(operator_real_path, H_result.real)
        np.save(operator_imag_path, H_result.imag)
    if v10_10_output:
        np.save(learned_operator_path, H_result)
        if operator_is_complex:
            np.save(operator_real_path, H_result.real)
            np.save(operator_imag_path, H_result.imag)
        np.savetxt(eigenvalues_path, result["eig"], delimiter=",")
        _write_loss_curve(loss_curve_path, result["history"])
    else:
        _write_eigenvalues(eigenvalues_path, result["eig"])
    _write_weights(weights_path, result["history"])
    plots = _maybe_write_plots(out_path.parent, zeta_zeros, result)
    if args.save_operator_health:
        with operator_health_path.open("w", encoding="utf-8") as f:
            json.dump(_json_safe(result.get("operator_health", {})), f, indent=2)

    eig = np.asarray(result["eig"], dtype=float)
    W = np.asarray(result["W"], dtype=float)
    Z = np.asarray(result["Z"], dtype=float)
    history = result["history"]
    history_losses = [_history_loss(row) for row in history]
    final_history = history[-1] if history and isinstance(history[-1], dict) else {}
    operator_health = result.get("operator_health") or {}
    finite_outputs = all(
        np.all(np.isfinite(x)) for x in (eig, W, Z, np.asarray(history_losses, dtype=float))
    )

    report = {
        "version": (
            "V10.3 No-Scaling DTES Geometry Learning"
            if args.no_scaling
            else (
                "V10.1 Pauli-Aware DTES Geometry Learning"
                if args.pauli_aware
                else "V10 DTES Geometry Learning"
            )
        ),
        "input": str(input_path),
        "steps": int(args.steps),
        "lr": float(args.lr),
        "k": int(args.k),
        "dim": int(args.dim),
        "max_n": int(args.max_n),
        "pauli_weight": float(args.pauli_weight),
        "adaptive_weights": bool(args.adaptive_weights),
        "adaptive_lr": float(args.adaptive_lr),
        "wavefunction_aware": bool(args.wavefunction_aware),
        "no_scaling": bool(args.no_scaling),
        "generalization_split": bool(args.generalization_split),
        "anchor_loss_enabled": bool(args.anchor_loss),
        "anchor_weight": float(args.anchor_weight),
        "spacing_anchor_weight": float(args.spacing_anchor_weight),
        "n_anchor": int(args.n_anchor),
        "line_geometry": bool(args.line_geometry),
        "spacing_weight": float(args.spacing_weight),
        "ordering_weight": float(args.ordering_weight),
        "center_weight": float(args.center_weight),
        "parametric_line": bool(args.parametric_line),
        "curve_smooth_weight": float(args.curve_smooth_weight),
        "curve_amp_weight": float(args.curve_amp_weight),
        "multiscale_geometry": bool(args.multiscale_geometry),
        "range_weight": float(args.range_weight),
        "nested_geometry": bool(args.nested_geometry),
        "wavefunction_topology": bool(args.wavefunction_topology),
        "wave_amp_weight": float(args.wave_amp_weight),
        "phase_weight": float(args.phase_weight),
        "nodal_weight": float(args.nodal_weight),
        "amplitude_collapse_weight": float(args.amplitude_collapse_weight),
        "phase_activity_weight": float(args.phase_activity_weight),
        "phase_target_std": float(args.phase_target_std),
        "phase_curvature_weight": float(args.phase_curvature_weight),
        "phase_coupled_operator": bool(args.phase_coupled_operator),
        "phase_coupling": float(args.phase_coupling),
        "phase_coupling_positive": bool(args.phase_coupling_positive),
        "magnetic_operator": bool(args.magnetic_operator),
        "magnetic_beta": float(args.magnetic_beta),
        "fractional_operator": bool(args.fractional_operator),
        "fractional_alpha": float(args.fractional_alpha),
        "detach_fractional": bool(detach_fractional),
        "self_adjoint_tol": float(args.self_adjoint_tol),
        "strict_self_adjoint": bool(args.strict_self_adjoint),
        "jitter_eps": float(args.jitter_eps),
        "log_operator_health": bool(args.log_operator_health),
        "save_operator_health": bool(args.save_operator_health),
        "operator_saved": True,
        "is_complex": operator_is_complex,
        "operator_type": (
            "fractional_complex_hermitian_magnetic_dtes"
            if args.fractional_operator and args.magnetic_operator
            else (
                "fractional_real_dtes"
                if args.fractional_operator
                else (
                    "complex_hermitian_magnetic_dtes"
                    if args.magnetic_operator
                    else "real_dtes"
                )
            )
        ),
        "final_anchor_loss": final_history.get("anchor_loss"),
        "final_spacing_anchor_loss": final_history.get("spacing_anchor_loss"),
        "final_line_spacing_loss": final_history.get("line_spacing_loss"),
        "final_ordering_loss": final_history.get("ordering_loss"),
        "final_center_loss": final_history.get("center_loss"),
        "final_curve_smoothness_loss": final_history.get("curve_smoothness_loss"),
        "final_curve_amplitude_loss": final_history.get("curve_amplitude_loss"),
        "final_range_loss": final_history.get("range_loss"),
        "final_multiscale_weights": result.get("multiscale_weights"),
        "final_nested_weights": result.get("nested_weights"),
        "final_eig_range": final_history.get("eig_range"),
        "final_zeta_range": final_history.get("zeta_range"),
        "final_eig_min": operator_health.get("eig_min"),
        "final_eig_max": operator_health.get("eig_max"),
        "final_eig_std": operator_health.get("eig_std"),
        "eig_range": final_history.get("eig_range"),
        "zeta_range": final_history.get("zeta_range"),
        "eig_min": operator_health.get("eig_min"),
        "eig_max": operator_health.get("eig_max"),
        "eig_std": operator_health.get("eig_std"),
        "final_wave_amp_loss": final_history.get("wave_amp_loss"),
        "final_phase_loss": final_history.get("phase_loss"),
        "final_phase_activity_loss": final_history.get("phase_activity_loss"),
        "final_phase_activity_target_loss": final_history.get(
            "phase_activity_target_loss"
        ),
        "final_phase_curvature_loss": final_history.get("phase_curvature_loss"),
        "final_phase_grad_std": final_history.get("phase_grad_std"),
        "final_phase_weight_min": final_history.get("phase_weight_min"),
        "final_phase_weight_max": final_history.get("phase_weight_max"),
        "final_hermitian_error": final_history.get(
            "hermitian_error", result.get("hermitian_error")
        ),
        "hermitian_error_raw": operator_health.get("hermitian_error_raw"),
        "hermitian_error_projected": operator_health.get(
            "hermitian_error_projected"
        ),
        "condition_proxy": operator_health.get("condition_proxy"),
        "used_jitter": operator_health.get("used_jitter"),
        "self_adjoint_pass": operator_health.get("self_adjoint_pass"),
        "spectral_pass": operator_health.get("spectral_pass"),
        "failure_reason": operator_health.get("failure_reason"),
        "final_nodal_loss": final_history.get("nodal_loss"),
        "final_amplitude_collapse_loss": final_history.get("amplitude_collapse_loss"),
        "final_amplitude_min": final_history.get("amplitude_min"),
        "final_amplitude_max": final_history.get("amplitude_max"),
        "final_amplitude_std": final_history.get("amplitude_std"),
        "final_weights": result.get("final_weights", {}),
        "train_error": result.get("train_error"),
        "test_error": result.get("test_error"),
        "affine_penalty": result.get("affine_penalty"),
        "n_input": n_input,
        "n_nodes": n_nodes,
        "zeta_zero_count": int(zeta_zeros.size),
        "initial_loss": history_losses[0] if history_losses else None,
        "final_loss": history_losses[-1] if history_losses else None,
        "edge_scale": result.get("edge_scale"),
        "finite_outputs": bool(
            finite_outputs and operator_health.get("finite_outputs", True)
        ),
        "graph_weight_sum": float(np.sum(W)),
        "graph_weight_mean": float(np.mean(W)) if W.size else None,
        "embedding_std": [float(x) for x in np.std(Z, axis=0)] if Z.size else [],
        **pauli_stats,
        "outputs": {
            "embedding": str(embedding_path),
            "adjacency": str(adjacency_path),
            "potential": str(potential_path),
            "eigenvalues": str(eigenvalues_path),
            "weights": str(weights_path),
            "loss_curve": str(loss_curve_path) if v10_10_output else None,
            "amplitude": str(amplitude_path) if args.wavefunction_topology else None,
            "phase": str(phase_path) if args.wavefunction_topology else None,
            "psi_real": str(psi_real_path) if args.wavefunction_topology else None,
            "psi_imag": str(psi_imag_path) if args.wavefunction_topology else None,
            "operator_real": str(operator_real_path)
            if args.magnetic_operator or (v10_10_output and operator_is_complex)
            else None,
            "operator_imag": str(operator_imag_path)
            if args.magnetic_operator or (v10_10_output and operator_is_complex)
            else None,
            "learned_operator": str(learned_operator_path) if v10_10_output else None,
            "v10_10_operator_dir": str(v10_10_dir),
            "v10_10_learned_operator": str(v10_10_operator_path),
            "v10_10_learned_operator_real": str(v10_10_operator_real_path)
            if operator_is_complex
            else None,
            "v10_10_learned_operator_imag": str(v10_10_operator_imag_path)
            if operator_is_complex
            else None,
            "v10_10_eigenvalues": str(v10_10_eigenvalues_path),
            "v10_10_report": str(v10_10_report_path),
            "operator_health": str(operator_health_path)
            if args.save_operator_health
            else None,
            "plots": plots,
        },
        "operator_health": operator_health,
        "history": history,
        "note": (
            "operator-level Hilbert–Pólya compatibility check; not a proof of RH"
            if v10_10_output
            else (
                "V10.3 no-scaling DTES operator identification experiment. "
                "Eigenvalues are evaluated in raw scale with a held-out split when "
                "enabled; numerical experiment, not proof of RH."
                if args.no_scaling
                else (
                    "Pauli-aware DTES geometry learning. Pauli nodal states are removed "
                    "as hard constraints; numerical experiment, not proof of RH."
                    if args.pauli_aware
                    else (
                        "Numerical DTES geometry learning experiment; not a proof of the "
                        "Riemann hypothesis."
                    )
                )
            )
        ),
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(_json_safe(report), f, indent=2, allow_nan=False)
    if v10_10_report_path.resolve() != out_path.resolve():
        with v10_10_report_path.open("w", encoding="utf-8") as f:
            json.dump(_json_safe(report), f, indent=2, allow_nan=False)

    print(f"[V10] report saved: {out_path}")
    print(f"[V10] embedding saved: {embedding_path}")
    print(f"[V10] eigenvalues saved: {eigenvalues_path}")
    print(f"[V10] adjacency saved: {adjacency_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
