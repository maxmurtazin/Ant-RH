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

from core.artin_billiard import (
    ArtinDTESOperator,
    potential_smoothness_loss,
    primitive_geodesic_length_proxy,
    range_loss,
    raw_spectral_loss,
    sample_artin_domain,
    selberg_trace_proxy_loss,
    spacing_loss,
)


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

    zeta_zeros = _first_present(data, ("zeta_zeros", "true_zeros", "known_zeros"))
    if zeta_zeros is None:
        raise ValueError("input JSON missing zeta_zeros")

    zeta_zeros = np.asarray(zeta_zeros, dtype=float).reshape(-1)
    zeta_zeros = np.sort(np.abs(zeta_zeros[np.isfinite(zeta_zeros)]))
    zeta_zeros = zeta_zeros[zeta_zeros > 0.0]
    if zeta_zeros.size == 0:
        raise ValueError("zeta_zeros must contain at least one positive finite value")

    return zeta_zeros


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


def _write_eigenvalues(path: Path, eigvals):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["rank", "eigenvalue"])
        for i, val in enumerate(eigvals, start=1):
            writer.writerow([i, float(val)])


def _write_loss_curve(path: Path, history):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "step",
                "loss",
                "raw_loss",
                "spacing_loss",
                "range_loss",
                "trace_proxy_loss",
                "smooth_loss",
            ]
        )
        for row in history:
            writer.writerow(
                [
                    int(row["step"]),
                    float(row["loss"]),
                    float(row["raw_loss"]),
                    float(row["spacing_loss"]),
                    float(row["range_loss"]),
                    float(row["trace_proxy_loss"]),
                    float(row["smooth_loss"]),
                ]
            )


def _setup_matplotlib(out_dir: Path):
    plot_cache = out_dir / ".matplotlib"
    plot_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(plot_cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(plot_cache))
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib.pyplot as plt

    return plt


def _write_plots(out_dir: Path, points, W, V, eig, zeta, history):
    plt = _setup_matplotlib(out_dir)
    plots = []

    k = min(eig.size, zeta.size)
    if k > 0:
        plt.figure()
        plt.plot(np.arange(k), np.sort(eig)[:k], label="Artin DTES eigenvalues")
        plt.plot(np.arange(k), np.sort(zeta)[:k], label="zeta zeros")
        plt.xlabel("rank")
        plt.ylabel("value")
        plt.title("V11 Artin DTES Spectrum")
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
    plt.title("V11 Artin Potential")
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
    plt.title("V11 Artin Hyperbolic Adjacency")
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
        plt.title("V11 Artin DTES Loss")
        path = out_dir / "loss_curve.png"
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        plots.append(str(path))

    return plots


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train V11 Selberg-Artin DTES.")
    parser.add_argument("--input", default="runs/dtes_spectral_input.json")
    parser.add_argument("--out-dir", default="runs/v11_artin_dtes")
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--k", type=int, default=50)
    parser.add_argument("--nx", type=int, default=48)
    parser.add_argument("--ny", type=int, default=48)
    parser.add_argument("--y-min", type=float, default=0.15)
    parser.add_argument("--y-max", type=float, default=4.0)
    parser.add_argument("--sigma", type=float, default=0.35)
    parser.add_argument("--k-neighbors", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--trace-weight", type=float, default=0.1)
    parser.add_argument("--range-weight", type=float, default=5.0)
    parser.add_argument("--smooth-weight", type=float, default=0.01)
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
        loss_spacing = spacing_loss(eig, zeta, args.k)
        loss_range = range_loss(eig, zeta, args.k)
        loss_smooth = potential_smoothness_loss(model.V, W)
        loss_trace = selberg_trace_proxy_loss(eig, lengths, k=args.k, tau=0.1)

        loss = (
            loss_raw
            + loss_spacing
            + float(args.range_weight) * loss_range
            + float(args.trace_weight) * loss_trace
            + float(args.smooth_weight) * loss_smooth
        )

        if not torch.isfinite(loss):
            raise FloatingPointError("V11 Artin DTES loss became NaN or Inf")

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        kk = min(int(args.k), eig.numel())
        eig_range = eig[:kk].max() - eig[:kk].min() if kk > 0 else eig.new_tensor(0.0)

        row = {
            "step": int(step),
            "loss": float(loss.detach().cpu()),
            "raw_loss": float(loss_raw.detach().cpu()),
            "spacing_loss": float(loss_spacing.detach().cpu()),
            "range_loss": float(loss_range.detach().cpu()),
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
                f"[V11-Artin] step={step} "
                f"loss={loss.item():.4f} "
                f"raw={loss_raw.item():.4f} "
                f"range={loss_range.item():.4f} "
                f"eig_range={eig_range.item():.2f} "
                f"cL={cL.item():.2f} b={b.item():.2f}"
            )

    with torch.no_grad():
        H, W, points_t, cL, cV, b = model()
        eig = torch.linalg.eigvalsh(H)
        loss_raw = raw_spectral_loss(eig, zeta, args.k)
        loss_spacing = spacing_loss(eig, zeta, args.k)
        loss_range = range_loss(eig, zeta, args.k)
        loss_smooth = potential_smoothness_loss(model.V, W)
        loss_trace = selberg_trace_proxy_loss(eig, lengths, k=args.k, tau=0.1)

    H_np = H.detach().cpu().numpy()
    W_np = W.detach().cpu().numpy()
    points_np = points_t.detach().cpu().numpy()
    V_np = model.V.detach().cpu().numpy()
    eig_np = eig.detach().cpu().numpy()

    np.save(out_dir / "learned_operator.npy", H_np)
    _write_eigenvalues(out_dir / "eigenvalues.csv", eig_np)
    _write_loss_curve(out_dir / "loss_curve.csv", history)
    plots = _write_plots(out_dir, points_np, W_np, V_np, eig_np, zeta_np, history)

    kk = min(int(args.k), eig.numel(), zeta.numel())
    eig_range = float((eig[:kk].max() - eig[:kk].min()).detach().cpu()) if kk else 0.0
    zeta_range = float((zeta[:kk].max() - zeta[:kk].min()).detach().cpu()) if kk else 0.0

    report = {
        "version": "V11 Selberg-Artin DTES",
        "operator": "H_theta = -Delta_Artin + V_theta",
        "input": str(input_path),
        "n_points": int(points.shape[0]),
        "k": int(args.k),
        "nx": int(args.nx),
        "ny": int(args.ny),
        "y_min": float(args.y_min),
        "y_max": float(args.y_max),
        "sigma": float(args.sigma),
        "k_neighbors": int(args.k_neighbors),
        "steps": int(args.steps),
        "eig_range": eig_range,
        "zeta_range": zeta_range,
        "raw_loss": float(loss_raw.detach().cpu()),
        "spacing_loss": float(loss_spacing.detach().cpu()),
        "range_loss": float(loss_range.detach().cpu()),
        "trace_proxy_loss": float(loss_trace.detach().cpu()),
        "smooth_loss": float(loss_smooth.detach().cpu()),
        "cL": float(cL.detach().cpu()),
        "cV": float(cV.detach().cpu()),
        "energy_shift": float(b.detach().cpu()),
        "finite_outputs": bool(
            np.all(np.isfinite(eig_np))
            and np.all(np.isfinite(H_np))
            and np.all(np.isfinite(V_np))
        ),
        "outputs": {
            "report": str(out_dir / "report.json"),
            "eigenvalues": str(out_dir / "eigenvalues.csv"),
            "loss_curve": str(out_dir / "loss_curve.csv"),
            "learned_operator": str(out_dir / "learned_operator.npy"),
            "plots": plots,
        },
        "history": history,
        "note": "Trainable Selberg-Artin DTES system; not proof of RH.",
    }

    with (out_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(_json_safe(report), f, indent=2, allow_nan=False)

    print(f"[V11-Artin] report saved: {out_dir / 'report.json'}")
    print(f"[V11-Artin] eigenvalues saved: {out_dir / 'eigenvalues.csv'}")
    print(f"[V11-Artin] operator saved: {out_dir / 'learned_operator.npy'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
