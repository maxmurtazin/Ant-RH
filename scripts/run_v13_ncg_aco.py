#!/usr/bin/env python3
"""Convenience launcher for V13 NCG-braid ACO (experimental spectral search; not an RH proof)."""
from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(ROOT)


def main() -> int:
    cmd = [
        sys.executable,
        "-m",
        "core.artin_aco",
        "--num_ants",
        "32",
        "--num_iters",
        "50",
        "--max_length",
        "8",
        "--out_dir",
        "runs/v13_ncg_aco",
        "--use_ncg_braid",
        "--lambda_ncg",
        "0.25",
        "--ncg_dim",
        "128",
        "--ncg_growth_alpha",
        "0.35",
        "--ncg_edge_scale",
        "0.15",
        "--ncg_spectrum_scale",
        "20.0",
        "--device",
        "cpu",
        "--ncg_target_zeros",
        "14.134725,21.022039,25.010857,30.424876,32.935062,37.586178",
    ]
    print("Running:", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
