#!/usr/bin/env python3
"""
V14.7c smoke test helper.

Checks that the V14.7c run directory contains required outputs and that the
candidate ranking/gates have sane non-empty content. Does NOT require all-gate pass.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Optional


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
    ap = argparse.ArgumentParser(description="Smoke test for V14.7c outputs.")
    ap.add_argument("--out_dir", type=str, default="runs/v14_7c_argument_trace_repair")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    required = [
        "v14_7c_seed_pool.csv",
        "v14_7c_repair_history.csv",
        "v14_7c_candidate_ranking.csv",
        "v14_7c_best_candidates.csv",
        "v14_7c_gate_summary.csv",
        "v14_7c_null_comparisons.csv",
        "v14_7c_active_argument_counts.csv",
        "v14_7c_trace_proxy.csv",
        "v14_7c_residue_scores.csv",
        "v14_7c_nv_diagnostics.csv",
        "v14_7c_results.json",
        "v14_7c_report.md",
        "v14_7c_report.tex",
    ]

    missing = [name for name in required if not (out_dir / name).is_file()]

    ranking = read_rows(out_dir / "v14_7c_candidate_ranking.csv")
    gates = read_rows(out_dir / "v14_7c_gate_summary.csv")
    arg = read_rows(out_dir / "v14_7c_active_argument_counts.csv")
    tr = read_rows(out_dir / "v14_7c_trace_proxy.csv")

    candidate_rows = len(ranking)
    gate_rows = len(gates)
    all_gate_pass = sum(1 for r in gates if str(r.get("G14_all_gate_pass", "")).lower() == "true")

    best_J: Optional[float] = None
    best_reward: Optional[float] = None
    support_ok = False
    not_poisson_like = False
    argument_ok = False
    trace_ok = False
    final_reward_finite = True
    any_support_pos = False
    any_poisson_lt1 = False

    for r in ranking:
        J = safe_float(r.get("J_v14_7c", float("nan")))
        rew = safe_float(r.get("final_reward", float("nan")))
        sup = safe_float(r.get("support_overlap", float("nan")), 0.0)
        pois = safe_float(r.get("poisson_like_fraction", float("nan")), 1.0)
        if not math.isfinite(rew):
            final_reward_finite = False
        if sup > 0.0:
            any_support_pos = True
        if pois < 1.0:
            any_poisson_lt1 = True
        if best_J is None or (math.isfinite(J) and J < best_J):
            best_J = J
            best_reward = rew

    for r in gates:
        support_ok = support_ok or (str(r.get("G2_support_overlap_ok", "")).lower() == "true")
        not_poisson_like = not_poisson_like or (str(r.get("G5_not_poisson_like", "")).lower() == "true")
        argument_ok = argument_ok or (str(r.get("G3_active_argument_ok", "")).lower() == "true")
        trace_ok = trace_ok or (str(r.get("G7_trace_proxy_ok", "")).lower() == "true")

    verdict = True
    verdict = verdict and (len(missing) == 0)
    verdict = verdict and (candidate_rows > 0)
    verdict = verdict and final_reward_finite
    verdict = verdict and (gate_rows > 0)
    verdict = verdict and any_support_pos
    verdict = verdict and any_poisson_lt1
    verdict = verdict and (len(arg) > 0)
    verdict = verdict and (len(tr) > 0)

    print("=== V14.7c SMOKE VERDICT ===")
    print("missing:", ",".join(missing) if missing else "(none)")
    print("candidate_rows:", candidate_rows)
    print("gate_rows:", gate_rows)
    print("all_gate_pass:", all_gate_pass)
    print("best_J:", best_J)
    print("best_reward:", best_reward)
    print("support_ok:", support_ok)
    print("not_poisson_like:", not_poisson_like)
    print("argument_ok:", argument_ok)
    print("trace_ok:", trace_ok)
    print("verdict:", "PASS" if verdict else "FAIL")


if __name__ == "__main__":
    main()

