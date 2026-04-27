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
VALIDATION_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(VALIDATION_DIR) not in sys.path:
    sys.path.insert(0, str(VALIDATION_DIR))

from compute_otoc_v10_11 import compute_otoc
from core.artin_billiard import (
    ArtinDTESOperator,
    potential_smoothness_loss,
    primitive_geodesic_length_proxy,
    sample_artin_domain,
    selberg_trace_proxy_loss,
)
from core.gue_losses import (
    gue_wigner_surmise_loss,
    log_gas_loss,
    range_loss,
    raw_spectral_loss,
    spacing_variance_loss,
)
from core.operator_health import validate_operator_health


def _first_present(data: Dict[str, Any], keys):
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def _load_zeta_zeros(path: Path):
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("input JSON must contain an object")

    zeta = _first_present(data, ("zeta_zeros", "true_zeros", "known_zeros"))
    if zeta is None:
        raise ValueError("input JSON missing zeta_zeros")

    zeta = np.asarray(zeta, dtype=float).reshape(-1)
    zeta = np.sort(np.abs(zeta[np.isfinite(zeta)]))
    zeta = zeta[zeta > 0.0]
    if zeta.size == 0:
        raise ValueError("zeta_zeros must contain positive finite values")
    return zeta


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


def _setup_matplotlib(out_dir: Path):
    plot_cache = out_dir / ".matplotlib"
    plot_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(plot_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(plot_cache))
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt

    return plt


def _write_eigenvalues(path: Path, eigvals):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "eigenvalue"])
        for i, val in enumerate(eigvals, start=1):
            writer.writerow([i, float(val)])


def _write_loss_curve(path: Path, history):
    fields = [
        "step",
        "loss",
        "raw_loss",
        "range_loss",
        "gue_loss",
        "loggas_loss",
        "rigidity_loss",
        "trace_proxy_loss",
        "smooth_loss",
        "eig_range",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in history:
            writer.writerow({field: row.get(field) for field in fields})


def _write_artin_plots(out_dir: Path, points, W, V, eig, zeta, history):
    plt = _setup_matplotlib(out_dir)
    plots = []

    k = min(eig.size, zeta.size)
    if k > 0:
        plt.figure()
        plt.plot(np.arange(k), np.sort(eig)[:k], label="Artin-GUE DTES eigenvalues")
        plt.plot(np.arange(k), np.sort(zeta)[:k], label="zeta zeros")
        plt.xlabel("rank")
        plt.ylabel("value")
        plt.title("V11.1 Artin-GUE-OTOC Spectrum")
        plt.legend()
        path = out_dir / "artin_spectrum.png"
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        plots.append(str(path))

    plt.figure()
    scatter = plt.scatter(points[:, 0], points[:, 1], c=V, s=16, cmap="viridis")
    plt.colorbar(scatter, label="V_theta")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("V11.1 Artin Potential")
    path = out_dir / "artin_potential.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    plots.append(str(path))

    plt.figure()
    plt.scatter(points[:, 0], points[:, 1], s=12)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Sampled PSL(2,Z) Fundamental Domain")
    path = out_dir / "artin_domain.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    plots.append(str(path))

    plt.figure()
    plt.imshow(W, aspect="auto", interpolation="nearest")
    plt.colorbar(label="weight")
    plt.title("V11.1 Artin Hyperbolic Adjacency")
    path = out_dir / "artin_adjacency.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    plots.append(str(path))

    if history:
        plt.figure()
        plt.plot([row["step"] for row in history], [row["loss"] for row in history])
        plt.xlabel("step")
        plt.ylabel("loss")
        plt.title("V11.1 Artin-GUE-OTOC Loss")
        path = out_dir / "loss_curve.png"
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        plots.append(str(path))

    return plots


def _write_otoc_outputs(out_dir: Path, H, t_max: float, n_times: int, health):
    plt = _setup_matplotlib(out_dir)
    times, C = compute_otoc(H, t_max=t_max, n_times=n_times)
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
    plt.title("V11.1 OTOC diagnostic")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "otoc_curve.png", dpi=160)
    plt.close()

    C_safe = np.maximum(C, 1e-12)
    mask = times < 0.3 * float(t_max)
    if mask.sum() > 5:
        slope = float(np.polyfit(times[mask], np.log(C_safe[mask]), 1)[0])
    else:
        slope = None

    report = {
        "version": "V11.1 OTOC diagnostic",
        "self_adjoint_pass": health["self_adjoint_pass"],
        "hermitian_error_projected": health["hermitian_error_projected"],
        "finite_outputs": health["finite_outputs"],
        "otoc_initial": float(C[0]),
        "otoc_max": float(C.max()),
        "otoc_final": float(C[-1]),
        "growth_rate_proxy": slope,
        "note": "Operator-level Hilbert-Pólya compatibility check; not proof of RH.",
    }
    with (out_dir / "otoc_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train V11.1 Artin-GUE-OTOC DTES.")
    parser.add_argument("--input", default="runs/dtes_spectral_input.json")
    parser.add_argument("--out-dir", default="runs/v11_1_artin_gue_otoc")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--nx", type=int, default=48)
    parser.add_argument("--ny", type=int, default=48)
    parser.add_argument("--y-min", type=float, default=0.15)
    parser.add_argument("--y-max", type=float, default=4.0)
    parser.add_argument("--sigma", type=float, default=0.35)
    parser.add_argument("--k-neighbors", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--raw-weight", type=float, default=1.0)
    parser.add_argument("--range-weight", type=float, default=5.0)
    parser.add_argument("--gue-weight", type=float, default=2.0)
    parser.add_argument("--loggas-weight", type=float, default=1.0)
    parser.add_argument("--rigidity-weight", type=float, default=0.2)
    parser.add_argument("--trace-weight", type=float, default=0.1)
    parser.add_argument("--smooth-weight", type=float, default=0.01)
    parser.add_argument("--run-otoc", action="store_true")
    parser.add_argument("--otoc-t-max", type=float, default=10.0)
    parser.add_argument("--otoc-n-times", type=int, default=200)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    zeta_np = _load_zeta_zeros(input_path)
    zeta = torch.tensor(zeta_np, dtype=torch.float32, device=args.device)

    points = sample_artin_domain(
        n_x=args.nx,
        n_y=args.ny,
        y_min=args.y_min,
        y_max=args.y_max,
    ).to(args.device)
    model = ArtinDTESOperator(
        points,
        sigma=args.sigma,
        k_neighbors=args.k_neighbors,
    ).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lengths = primitive_geodesic_length_proxy(points).detach()

    history = []
    for step in range(int(args.steps)):
        opt.zero_grad(set_to_none=True)

        H, W, points_t, cL, cV, b = model()
        eig = torch.linalg.eigvalsh(H)

        loss_raw = raw_spectral_loss(eig, zeta, args.k)
        loss_range = range_loss(eig, zeta, args.k)
        loss_gue = gue_wigner_surmise_loss(eig, k=args.k)
        loss_loggas = log_gas_loss(eig, k=args.k)
        loss_rigidity = spacing_variance_loss(eig, k=args.k)
        loss_trace = selberg_trace_proxy_loss(eig, lengths, k=args.k, tau=0.1)
        loss_smooth = potential_smoothness_loss(model.V, W)

        loss = (
            float(args.raw_weight) * loss_raw
            + float(args.range_weight) * loss_range
            + float(args.gue_weight) * loss_gue
            + float(args.loggas_weight) * loss_loggas
            + float(args.rigidity_weight) * loss_rigidity
            + float(args.trace_weight) * loss_trace
            + float(args.smooth_weight) * loss_smooth
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("V11.1 Artin-GUE-OTOC loss became NaN or Inf")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        kk = min(int(args.k), eig.numel())
        eig_range = eig[:kk].max() - eig[:kk].min() if kk > 0 else eig.new_tensor(0.0)

        row = {
            "step": int(step),
            "loss": float(loss.detach().cpu()),
            "raw_loss": float(loss_raw.detach().cpu()),
            "range_loss": float(loss_range.detach().cpu()),
            "gue_loss": float(loss_gue.detach().cpu()),
            "loggas_loss": float(loss_loggas.detach().cpu()),
            "rigidity_loss": float(loss_rigidity.detach().cpu()),
            "trace_proxy_loss": float(loss_trace.detach().cpu()),
            "smooth_loss": float(loss_smooth.detach().cpu()),
            "eig_range": float(eig_range.detach().cpu()),
            "cL": float(cL.detach().cpu()),
            "cV": float(cV.detach().cpu()),
            "energy_shift": float(b.detach().cpu()),
        }
        history.append(row)

        if step % 50 == 0:
            print(
                f"[V11.1-Artin-GUE] step={step} "
                f"loss={loss.item():.4f} "
                f"raw={loss_raw.item():.4f} "
                f"range={loss_range.item():.4f} "
                f"gue={loss_gue.item():.4f} "
                f"loggas={loss_loggas.item():.4f} "
                f"eig_range={eig_range.item():.2f} "
                f"cL={cL.item():.2f} b={b.item():.2f}"
            )

    with torch.no_grad():
        H, W, points_t, cL, cV, b = model()
        H_projected, eig_valid, health = validate_operator_health(H)
        if eig_valid is None:
            reason = health.get("failure_reason") or "eigvalsh failed"
            raise RuntimeError(f"final operator failed self-adjoint validation: {reason}")
        if not health["self_adjoint_pass"]:
            raise RuntimeError("final operator failed self-adjoint validation")
        eig = eig_valid
        loss_raw = raw_spectral_loss(eig, zeta, args.k)
        loss_range = range_loss(eig, zeta, args.k)
        loss_gue = gue_wigner_surmise_loss(eig, k=args.k)
        loss_loggas = log_gas_loss(eig, k=args.k)
        loss_rigidity = spacing_variance_loss(eig, k=args.k)
        loss_trace = selberg_trace_proxy_loss(eig, lengths, k=args.k, tau=0.1)
        loss_smooth = potential_smoothness_loss(model.V, W)

    H_np = H_projected.detach().cpu().numpy()
    W_np = W.detach().cpu().numpy()
    points_np = points_t.detach().cpu().numpy()
    V_np = model.V.detach().cpu().numpy()
    eig_np = eig.detach().cpu().numpy()
    zeta_k_np = zeta_np[: min(int(args.k), zeta_np.size)]

    np.save(out_dir / "learned_operator.npy", H_np)
    _write_eigenvalues(out_dir / "eigenvalues.csv", eig_np)
    _write_loss_curve(out_dir / "loss_curve.csv", history)
    plots = _write_artin_plots(out_dir, points_np, W_np, V_np, eig_np, zeta_np, history)

    otoc_report = None
    if args.run_otoc:
        otoc_report = _write_otoc_outputs(
            out_dir,
            H_projected,
            t_max=args.otoc_t_max,
            n_times=args.otoc_n_times,
            health=health,
        )

    kk = min(int(args.k), eig.numel(), zeta.numel())
    eig_range = float((eig[:kk].max() - eig[:kk].min()).detach().cpu()) if kk else 0.0
    zeta_range = float((zeta[:kk].max() - zeta[:kk].min()).detach().cpu()) if kk else 0.0

    report = {
        "version": "V11.1 Artin-GUE-OTOC DTES",
        "operator": "H_theta = -Delta_Artin + V_theta",
        "input": str(input_path),
        "n_points": int(points.shape[0]),
        "k": int(args.k),
        "nx": int(args.nx),
        "ny": int(args.ny),
        "sigma": float(args.sigma),
        "k_neighbors": int(args.k_neighbors),
        "steps": int(args.steps),
        "eig_range": eig_range,
        "zeta_range": zeta_range,
        "raw_loss": float(loss_raw.detach().cpu()),
        "range_loss": float(loss_range.detach().cpu()),
        "gue_loss": float(loss_gue.detach().cpu()),
        "loggas_loss": float(loss_loggas.detach().cpu()),
        "rigidity_loss": float(loss_rigidity.detach().cpu()),
        "trace_proxy_loss": float(loss_trace.detach().cpu()),
        "smooth_loss": float(loss_smooth.detach().cpu()),
        "hermitian_error_projected": health["hermitian_error_projected"],
        "self_adjoint_pass": health["self_adjoint_pass"],
        "finite_outputs": bool(
            health["finite_outputs"]
            and np.all(np.isfinite(eig_np))
            and np.all(np.isfinite(H_np))
            and np.all(np.isfinite(V_np))
        ),
        "cL": float(cL.detach().cpu()),
        "cV": float(cV.detach().cpu()),
        "energy_shift": float(b.detach().cpu()),
        "otoc_initial": otoc_report.get("otoc_initial") if otoc_report else None,
        "otoc_max": otoc_report.get("otoc_max") if otoc_report else None,
        "otoc_final": otoc_report.get("otoc_final") if otoc_report else None,
        "growth_rate_proxy": otoc_report.get("growth_rate_proxy")
        if otoc_report
        else None,
        "outputs": {
            "report": str(out_dir / "report.json"),
            "eigenvalues": str(out_dir / "eigenvalues.csv"),
            "learned_operator": str(out_dir / "learned_operator.npy"),
            "loss_curve": str(out_dir / "loss_curve.csv"),
            "otoc_curve": str(out_dir / "otoc_curve.csv") if args.run_otoc else None,
            "otoc_report": str(out_dir / "otoc_report.json") if args.run_otoc else None,
            "plots": plots
            + ([str(out_dir / "otoc_curve.png")] if args.run_otoc else []),
        },
        "operator_health": health,
        "otoc_report": otoc_report,
        "history": history,
        "note": "Operator-level Hilbert-Pólya compatibility check; not proof of RH.",
    }
    if zeta_k_np.size:
        report["zeta_min"] = float(zeta_k_np.min())
        report["zeta_max"] = float(zeta_k_np.max())

    with (out_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(_json_safe(report), f, indent=2, allow_nan=False)

    print(f"[V11.1-Artin-GUE] report saved: {out_dir / 'report.json'}")
    print(f"[V11.1-Artin-GUE] eigenvalues saved: {out_dir / 'eigenvalues.csv'}")
    print(f"[V11.1-Artin-GUE] operator saved: {out_dir / 'learned_operator.npy'}")
    if args.run_otoc:
        print(f"[V11.1-Artin-GUE] OTOC saved: {out_dir / 'otoc_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
