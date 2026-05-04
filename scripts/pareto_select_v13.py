#!/usr/bin/env python3
"""
V13D: ensemble score + Pareto selection from V13C ``validation_results.json``.

Run from repo root:

  python3 scripts/pareto_select_v13.py \\
    --input_json runs/v13_candidate_validation_word_sensitive/validation_results.json \\
    --out_dir runs/v13_pareto_selection
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

if not os.environ.get("MPLCONFIGDIR"):
    _mpl_cfg = Path(ROOT) / ".mpl_cache"
    try:
        _mpl_cfg.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(_mpl_cfg)
    except OSError:
        pass


def _f(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def min_max_normalize(values: np.ndarray) -> np.ndarray:
    """Map finite values to [0, 1]; non-finite left as nan."""
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    fin = np.isfinite(v)
    if not fin.any():
        return np.full_like(v, np.nan)
    lo = float(np.min(v[fin]))
    hi = float(np.max(v[fin]))
    if hi - lo < 1e-15:
        out = np.zeros_like(v)
        out[~fin] = np.nan
        return out
    out = (v - lo) / (hi - lo)
    out[~fin] = np.nan
    return out


def competition_rank_ascending(values: np.ndarray) -> np.ndarray:
    """Lower value is better; rank 1 = best. Ties share the same rank (competition ranking)."""
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    n = int(v.size)
    for i in range(n):
        if not math.isfinite(float(v[i])):
            v[i] = np.inf
    order = np.argsort(v, kind="mergesort")
    ranks = np.empty(n, dtype=np.int32)
    r = 1
    k = 0
    while k < n:
        cur = float(v[order[k]])
        j = k
        while j < n and float(v[order[j]]) == cur:
            j += 1
        for t in range(k, j):
            ranks[order[t]] = r
        r += j - k
        k = j
    return ranks


def pareto_front_mask(objectives: np.ndarray) -> np.ndarray:
    """
    ``objectives`` shape (n, m), all to be minimized.
    Returns boolean mask length n: True if on Pareto front (not strictly dominated).
    """
    X = np.asarray(objectives, dtype=np.float64)
    n, m = X.shape
    mask = np.zeros(n, dtype=bool)
    for i in range(n):
        xi = X[i]
        if not np.all(np.isfinite(xi)):
            mask[i] = False
            continue
        dominated = False
        for j in range(n):
            if i == j:
                continue
            xj = X[j]
            if not np.all(np.isfinite(xj)):
                continue
            le = np.all(xj <= xi)
            lt = np.any(xj < xi)
            if le and lt:
                dominated = True
                break
        mask[i] = not dominated
    return mask


def word_length(c: Dict[str, Any]) -> float:
    w = c.get("word_used") or c.get("word")
    if isinstance(w, list):
        return float(len(w))
    return float("nan")


def enrich_candidates(
    raw: List[Dict[str, Any]],
    *,
    target_length: float,
    length_weight: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    n = len(raw)
    spec = np.array([_f(c.get("spectral_log_mse")) for c in raw])
    ks = np.array([_f(c.get("ks_wigner")) for c in raw])
    nij = np.array([_f(c.get("nijenhuis_defect")) for c in raw])
    comm = np.array([_f(c.get("comm_norm_proxy")) for c in raw])
    ram = np.array([_f(c.get("ramsey_score")) for c in raw])
    lengths = np.array([word_length(c) for c in raw])

    spectral_norm = min_max_normalize(spec)
    ks_norm = min_max_normalize(ks)
    nijenhuis_norm = min_max_normalize(nij)
    comm_norm = min_max_normalize(comm)
    ram_n = min_max_normalize(ram)
    ramsey_norm = 1.0 - ram_n
    # If all ramsey identical, ram_n is zeros → ramsey_norm all 1; acceptable.

    tgt = max(float(target_length), 1e-12)
    length_target_penalty = np.abs(lengths - float(target_length)) / tgt

    def fill_nan_worst(a: np.ndarray, worst: float = 1.0) -> np.ndarray:
        out = np.asarray(a, dtype=np.float64).copy()
        out[~np.isfinite(out)] = worst
        return out

    sn = fill_nan_worst(spectral_norm)
    kn = fill_nan_worst(ks_norm)
    nn = fill_nan_worst(nijenhuis_norm)
    cn = fill_nan_worst(comm_norm)
    rn = fill_nan_worst(ramsey_norm)
    ltp = fill_nan_worst(length_target_penalty, worst=np.nanmax(length_target_penalty) if np.any(np.isfinite(length_target_penalty)) else 0.0)

    score = (
        0.35 * sn
        + 0.20 * kn
        + 0.20 * nn
        + 0.15 * cn
        + 0.10 * rn
        + float(length_weight) * ltp
    )

    neg_ram = -ram
    obj = np.column_stack([spec, ks, nij, comm, neg_ram])
    is_pf = pareto_front_mask(obj)

    geom_sum = ks + nij + comm
    rank_score = competition_rank_ascending(score)
    rank_spec = competition_rank_ascending(spec)
    rank_geom = competition_rank_ascending(geom_sum)

    out_rows: List[Dict[str, Any]] = []
    for i, c in enumerate(raw):
        row = dict(c)
        row["word_length"] = float(lengths[i]) if np.isfinite(lengths[i]) else None
        row["spectral_norm"] = float(sn[i])
        row["ks_norm"] = float(kn[i])
        row["nijenhuis_norm"] = float(nn[i])
        row["comm_norm"] = float(cn[i])
        row["ramsey_norm"] = float(rn[i])
        row["length_target_penalty"] = float(ltp[i]) if np.isfinite(ltp[i]) else None
        row["ensemble_score"] = float(score[i])
        row["is_pareto_front"] = bool(is_pf[i])
        row["rank_by_score"] = int(rank_score[i])
        row["rank_by_spectral"] = int(rank_spec[i])
        row["rank_by_geometry"] = int(rank_geom[i])
        row["geometry_sum"] = float(geom_sum[i]) if np.isfinite(geom_sum[i]) else None
        out_rows.append(row)

    sorted_by_score = sorted(out_rows, key=lambda r: (float(r.get("ensemble_score", np.inf)), str(r.get("id", ""))))

    best_spec_idx = int(np.nanargmin(spec)) if np.any(np.isfinite(spec)) else 0
    best_geom_idx = int(np.nanargmin(geom_sum)) if np.any(np.isfinite(geom_sum)) else 0
    best_bal_idx = int(np.nanargmin(score))

    rec = {
        "spectral_best": str(raw[best_spec_idx].get("id", "")),
        "geometry_best": str(raw[best_geom_idx].get("id", "")),
        "balanced_best": str(raw[best_bal_idx].get("id", "")),
        "pareto_front_ids": [str(raw[i].get("id", "")) for i in range(n) if is_pf[i]],
    }

    meta = {
        "ensemble_weights": {
            "spectral_norm": 0.35,
            "ks_norm": 0.20,
            "nijenhuis_norm": 0.20,
            "comm_norm": 0.15,
            "ramsey_norm": 0.10,
            "length_target_penalty_coeff": float(length_weight),
        },
        "target_length": float(target_length),
        "recommendation": rec,
    }
    return sorted_by_score, meta


def plot_spectral_vs_nijenhuis(rows: List[Dict[str, Any]], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(rows))))
    for i, r in enumerate(rows):
        cid = str(r.get("id", "?"))
        sx = _f(r.get("spectral_log_mse"))
        sy = _f(r.get("nijenhuis_defect"))
        if not (math.isfinite(sx) and math.isfinite(sy)):
            continue
        c = colors[i % len(colors)]
        ax.scatter(sx, sy, s=80, c=[c], edgecolors="k", linewidths=0.5, zorder=3)
        ax.annotate(cid, (sx, sy), textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.set_xlabel("spectral_log_mse (lower better)")
    ax.set_ylabel("nijenhuis_defect (lower better)")
    ax.set_title("V13D: spectral vs Nijenhuis (Pareto context)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="V13D Pareto + ensemble selection from V13C validation JSON.")
    ap.add_argument(
        "--input_json",
        type=str,
        default="runs/v13_candidate_validation_word_sensitive/validation_results.json",
    )
    ap.add_argument("--out_dir", type=str, default="runs/v13_pareto_selection")
    ap.add_argument("--target_length", type=float, default=6.0)
    ap.add_argument("--length_weight", type=float, default=0.1)
    args = ap.parse_args()

    in_path = Path(args.input_json)
    if not in_path.is_file():
        raise FileNotFoundError(f"input_json not found: {in_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(in_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("validation JSON must contain a non-empty 'candidates' list")

    enriched, meta = enrich_candidates(
        candidates,
        target_length=float(args.target_length),
        length_weight=float(args.length_weight),
    )

    out_json = {
        "input_json": str(in_path.resolve()),
        "meta": meta,
        "candidates": enriched,
    }
    with open(out_dir / "pareto_results.json", "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2, allow_nan=True)

    csv_cols: List[str] = [
        "id",
        "ensemble_score",
        "is_pareto_front",
        "rank_by_score",
        "rank_by_spectral",
        "rank_by_geometry",
        "spectral_log_mse",
        "ks_wigner",
        "nijenhuis_defect",
        "comm_norm_proxy",
        "ramsey_score",
        "word_length",
        "geometry_sum",
        "spectral_norm",
        "ks_norm",
        "nijenhuis_norm",
        "comm_norm",
        "ramsey_norm",
        "length_target_penalty",
    ]
    with open(out_dir / "pareto_results.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_cols, extrasaction="ignore")
        w.writeheader()
        for r in enriched:
            w.writerow({k: r.get(k) for k in csv_cols})

    plot_spectral_vs_nijenhuis(enriched, out_dir / "pareto_spectral_vs_nijenhuis.png")

    print("V13D Pareto + ensemble — sorted by ensemble_score (ascending; lower is better)\n")
    for r in enriched:
        pf = "Pareto" if r.get("is_pareto_front") else "     "
        print(
            f"  [{pf}] {r['id']}: score={r['ensemble_score']!s} "
            f"r_score={r['rank_by_score']} r_spec={r['rank_by_spectral']} r_geom={r['rank_by_geometry']} "
            f"log_mse={r.get('spectral_log_mse')!s} ks={r.get('ks_wigner')!s} nij={r.get('nijenhuis_defect')!s}"
        )

    rec = meta["recommendation"]
    print("\nRecommendations:")
    print(f"  spectral_best:    {rec['spectral_best']}")
    print(f"  geometry_best:    {rec['geometry_best']}")
    print(f"  balanced_best:    {rec['balanced_best']}")
    print(f"  pareto_front_ids: {rec['pareto_front_ids']}")
    print(f"\nWrote {out_dir / 'pareto_results.json'}")
    print(f"Wrote {out_dir / 'pareto_results.csv'}")
    print(f"Wrote {out_dir / 'pareto_spectral_vs_nijenhuis.png'}")


if __name__ == "__main__":
    main()
