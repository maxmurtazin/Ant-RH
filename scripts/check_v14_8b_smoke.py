#!/usr/bin/env python3
"""
Smoke checker for V14.8b outputs.

Prints:
- missing files
- gate_rows
- all_gate_pass count
- support_ok / argument_ok / residue_ok / trace_ok / hadamard_ok / not_poisson_like counts
- best J per dim
- best support_overlap per dim
- best active_argument_error_med per dim
- best hadamard_error_med per dim
- whether V14.8b improves over V14.8 on ZERO_SUPPORT_REJECT frequency (best-effort)
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict, List


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def read_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def main() -> None:
    ap = argparse.ArgumentParser(description="V14.8b smoke checker.")
    ap.add_argument("--out_dir", type=str, default="runs/v14_8b_spectral_support_rescaling")
    ap.add_argument("--v14_8_dir", type=str, default="runs/v14_8_braid_graph_laplacian_hadamard")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    required = [
        "v14_8b_scaled_spectra.csv",
        "v14_8b_transport_maps.csv",
        "v14_8b_active_argument_counts.csv",
        "v14_8b_residue_scores.csv",
        "v14_8b_trace_proxy.csv",
        "v14_8b_hadamard_scores.csv",
        "v14_8b_nv_diagnostics.csv",
        "v14_8b_candidate_ranking.csv",
        "v14_8b_best_candidates.csv",
        "v14_8b_gate_summary.csv",
        "v14_8b_null_comparisons.csv",
        "v14_8b_results.json",
        "v14_8b_report.md",
        "v14_8b_report.tex",
    ]
    missing = [f for f in required if not (out_dir / f).is_file()]

    gate = read_rows(out_dir / "v14_8b_gate_summary.csv")
    rank = read_rows(out_dir / "v14_8b_candidate_ranking.csv")

    all_gate_pass = sum(1 for r in gate if str(r.get("G14_all_gate_pass", "")).lower() == "true")
    support_ok = sum(1 for r in gate if str(r.get("G2_support_overlap_ok", "")).lower() == "true")
    argument_ok = sum(1 for r in gate if str(r.get("G3_active_argument_ok", "")).lower() == "true")
    residue_ok = sum(1 for r in gate if str(r.get("G4_residue_error_ok", "")).lower() == "true")
    trace_ok = sum(1 for r in gate if str(r.get("G5_trace_proxy_ok", "")).lower() == "true")
    hadamard_ok = sum(1 for r in gate if str(r.get("G6_hadamard_determinant_ok", "")).lower() == "true")
    not_poisson_like = sum(1 for r in gate if str(r.get("G8_not_poisson_like", "")).lower() == "true")

    # best per dim
    best_by_dim: Dict[int, Dict[str, float]] = {}
    for r in rank:
        d = int(float(r.get("dim", 0) or 0))
        J = safe_float(r.get("J_v14_8b", float("nan")))
        sup = safe_float(r.get("support_overlap", float("nan")))
        arg = safe_float(r.get("active_argument_error_med", float("nan")))
        had = safe_float(r.get("hadamard_error_med", float("nan")))
        cur = best_by_dim.get(d)
        if cur is None or (math.isfinite(J) and J < cur["J"]):
            best_by_dim[d] = {
                "J": float(J) if math.isfinite(J) else float("inf"),
                "support": float(sup),
                "arg": float(arg),
                "had": float(had),
            }

    # compare ZERO_SUPPORT_REJECT frequency vs v14_8
    v14_8_dir = Path(args.v14_8_dir)
    v148 = read_rows(v14_8_dir / "v14_8_candidate_ranking.csv") if v14_8_dir.is_dir() else []
    zero_support_v148 = sum(1 for r in v148 if str(r.get("classification", "")).strip() == "ZERO_SUPPORT_REJECT")
    zero_support_v148b = sum(1 for r in rank if str(r.get("classification", "")).strip() == "ZERO_SUPPORT_REJECT")
    improves = None
    if v148:
        improves = zero_support_v148b < zero_support_v148

    print("=== V14.8b SMOKE ===")
    print("missing_files:", missing if missing else "none")
    print("gate_rows:", len(gate))
    print("all_gate_pass:", all_gate_pass)
    print("support_ok:", support_ok)
    print("argument_ok:", argument_ok)
    print("residue_ok:", residue_ok)
    print("trace_ok:", trace_ok)
    print("hadamard_ok:", hadamard_ok)
    print("not_poisson_like:", not_poisson_like)
    print("zero_support_v14_8:", zero_support_v148 if v148 else "unknown")
    print("zero_support_v14_8b:", zero_support_v148b)
    print("improves_zero_support_reject_frequency:", improves if improves is not None else "unknown")
    print("\n=== BEST PER DIM ===")
    for d in sorted(best_by_dim.keys()):
        b = best_by_dim[d]
        print("dim", d, "best_J", b["J"], "best_support", b["support"], "best_arg", b["arg"], "best_had", b["had"])


if __name__ == "__main__":
    main()

