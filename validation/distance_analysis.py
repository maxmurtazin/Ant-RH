#!/usr/bin/env python3
"""
distance_analysis.py

Analyze DTES candidate distance-to-truth-zero errors.

Supported inputs:

A) Nearest-distance CSV from compare_full_scan_vs_hybrid.py:
   full_vs_hybrid_100_400_truth_nearest_candidate.csv
   columns:
       truth_index, truth_t, nearest_candidate_t, nearest_distance

B) Candidate-nearest CSV:
   full_vs_hybrid_100_400_candidate_nearest.csv
   columns:
       candidate_index, candidate_t, nearest_truth_t, nearest_distance

C) JSON pair:
   --truth zeros_100_400_precise.json
   --dtes dtes_candidates.json

Outputs:
   <out>_summary.md
   <out>_stats.json
   <out>_distances.csv
   <out>_histogram.png
   <out>_cdf.png
   <out>_window_recall.png

Ant-RH:
    Implements the “distance / recall analysis” step after DTES or hybrid runs
    (see repository README pipeline).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception as exc:
    plt = None
    _MATPLOTLIB_ERROR = exc
else:
    _MATPLOTLIB_ERROR = None


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_sequence(data: Any, keys: Tuple[str, ...]) -> List[Any]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in keys:
            if key in data and isinstance(data[key], list):
                return data[key]
        for val in data.values():
            if isinstance(val, list):
                return val
    raise ValueError("Could not find list in JSON.")


def extract_t(item: Any) -> float:
    if isinstance(item, (int, float)):
        return float(item)
    if isinstance(item, dict):
        for key in ("t", "time", "candidate", "zero", "value"):
            if key in item:
                return float(item[key])
    raise ValueError(f"Cannot extract t from item={item!r}")


def load_truth_json(path: Path, t_min: Optional[float], t_max: Optional[float]) -> np.ndarray:
    data = load_json(path)
    seq = extract_sequence(data, ("zeros", "truth", "ground_truth", "candidates"))
    vals = []
    for item in seq:
        t = extract_t(item)
        if t_min is not None and t < t_min:
            continue
        if t_max is not None and t > t_max:
            continue
        vals.append(t)
    return np.array(sorted(vals), dtype=float)


def load_dtes_json(path: Path, t_min: Optional[float], t_max: Optional[float]) -> np.ndarray:
    data = load_json(path)
    seq = extract_sequence(data, ("candidates", "dtes_candidates", "zeros", "results"))
    vals = []
    for item in seq:
        t = extract_t(item)
        if t_min is not None and t < t_min:
            continue
        if t_max is not None and t > t_max:
            continue
        vals.append(t)
    return np.array(sorted(vals), dtype=float)


def nearest_distances_from_arrays(truth: np.ndarray, dtes: np.ndarray) -> List[Dict[str, Any]]:
    rows = []
    for i, t in enumerate(truth):
        if dtes.size == 0:
            rows.append({
                "truth_index": i,
                "truth_t": float(t),
                "nearest_candidate_t": None,
                "nearest_distance": None,
            })
            continue
        diffs = np.abs(dtes - t)
        j = int(np.argmin(diffs))
        rows.append({
            "truth_index": i,
            "truth_t": float(t),
            "nearest_candidate_t": float(dtes[j]),
            "nearest_distance": float(diffs[j]),
        })
    return rows


def load_distances_csv(path: Path, t_min: Optional[float], t_max: Optional[float]) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        rd = csv.DictReader(f)
        for r in rd:
            if "nearest_distance" not in r:
                continue
            dist_raw = r.get("nearest_distance")
            if dist_raw in (None, "", "None", "nan"):
                continue
            # Prefer truth_t if present, else candidate_t.
            t_ref = None
            for key in ("truth_t", "candidate_t", "t"):
                if key in r and r[key] not in ("", "None", None):
                    try:
                        t_ref = float(r[key])
                        break
                    except Exception:
                        pass
            if t_ref is not None:
                if t_min is not None and t_ref < t_min:
                    continue
                if t_max is not None and t_ref > t_max:
                    continue
            row = dict(r)
            row["nearest_distance"] = float(dist_raw)
            if t_ref is not None:
                row["t_ref"] = t_ref
            rows.append(row)
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    keys = []
    for row in rows:
        for k in row.keys():
            if isinstance(row[k], (dict, list)):
                continue
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for row in rows:
            wr.writerow({k: row.get(k) for k in keys})


def ensure_plotting() -> None:
    if plt is None:
        raise RuntimeError(f"matplotlib unavailable: {_MATPLOTLIB_ERROR}")


def compute_stats(distances: np.ndarray, target_recalls: List[float]) -> Dict[str, Any]:
    if distances.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "max": None,
            "std": None,
            "percentiles": {},
            "recommended_windows": {},
        }

    percentiles = {
        "p50": float(np.percentile(distances, 50)),
        "p75": float(np.percentile(distances, 75)),
        "p90": float(np.percentile(distances, 90)),
        "p95": float(np.percentile(distances, 95)),
        "p99": float(np.percentile(distances, 99)),
        "p100": float(np.max(distances)),
    }

    recommended = {}
    for r in target_recalls:
        q = max(0.0, min(100.0, 100.0 * r))
        recommended[f"recall_{r:.4f}"] = float(np.percentile(distances, q))

    return {
        "count": int(distances.size),
        "mean": float(np.mean(distances)),
        "median": float(np.median(distances)),
        "std": float(np.std(distances)),
        "min": float(np.min(distances)),
        "max": float(np.max(distances)),
        "percentiles": percentiles,
        "recommended_windows": recommended,
    }


def recall_at_windows(distances: np.ndarray, windows: np.ndarray) -> List[Dict[str, float]]:
    rows = []
    n = max(1, int(distances.size))
    for w in windows:
        rows.append({
            "window": float(w),
            "recall": float(np.sum(distances <= w) / n),
            "missed": int(np.sum(distances > w)),
        })
    return rows


def plot_histogram(path: Path, distances: np.ndarray, stats: Dict[str, Any]) -> None:
    ensure_plotting()
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    if distances.size:
        ax.hist(distances, bins=40)
        ax.axvline(stats["percentiles"]["p95"], linestyle="--", label="p95")
        ax.axvline(stats["percentiles"]["p99"], linestyle="--", label="p99")
        ax.axvline(stats["percentiles"]["p100"], linestyle=":", label="max")
    ax.set_xlabel("distance to nearest DTES candidate")
    ax.set_ylabel("count")
    ax.set_title("DTES distance-to-zero error histogram")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_cdf(path: Path, distances: np.ndarray, stats: Dict[str, Any]) -> None:
    ensure_plotting()
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    if distances.size:
        x = np.sort(distances)
        y = np.arange(1, len(x) + 1) / len(x)
        ax.plot(x, y, linewidth=2)
        for name, val in stats["percentiles"].items():
            if name in ("p95", "p99", "p100"):
                ax.axvline(val, linestyle="--", label=f"{name}={val:.5g}")
    ax.set_xlabel("window half-width")
    ax.set_ylabel("recall / CDF")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Recall as a function of hybrid window")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_window_recall(path: Path, sweep: List[Dict[str, float]]) -> None:
    ensure_plotting()
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    x = [r["window"] for r in sweep]
    y = [r["recall"] for r in sweep]
    ax.plot(x, y, marker="o")
    ax.set_xlabel("window half-width")
    ax.set_ylabel("recall")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Hybrid window tuning curve")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_summary(path: Path, stats: Dict[str, Any], sweep: List[Dict[str, float]], source: str) -> None:
    lines = [
        "# DTES distance-to-zero analysis",
        "",
        f"- source: `{source}`",
        f"- count: `{stats['count']}`",
        f"- mean distance: `{stats['mean']}`",
        f"- median distance: `{stats['median']}`",
        f"- max distance: `{stats['max']}`",
        "",
        "## Percentiles",
        "",
    ]

    for k, v in (stats.get("percentiles") or {}).items():
        lines.append(f"- {k}: `{v}`")

    lines.extend(["", "## Recommended windows", ""])
    for k, v in (stats.get("recommended_windows") or {}).items():
        lines.append(f"- {k}: `window >= {v}`")

    lines.extend([
        "",
        "## Window recall sweep",
        "",
        "| window | recall | missed |",
        "|---:|---:|---:|",
    ])
    for r in sweep:
        lines.append(f"| {r['window']:.6g} | {r['recall']:.6f} | {r['missed']} |")

    lines.extend([
        "",
        "## Interpretation",
        "",
        "The minimal hybrid half-window required for a desired recall is the corresponding",
        "quantile of the distance distribution. If the maximum distance is small and stable",
        "across intervals, DTES candidates form a compact cover of the true zero set.",
    ])

    path.write_text("\n".join(lines), encoding="utf-8")


def parse_float_list(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze DTES candidate distance-to-zero errors.")
    ap.add_argument("--distances_csv", default=None, help="CSV with nearest_distance column.")
    ap.add_argument("--truth", default=None, help="Truth zeros JSON.")
    ap.add_argument("--dtes", default=None, help="DTES candidates JSON.")
    ap.add_argument("--t_min", type=float, default=None)
    ap.add_argument("--t_max", type=float, default=None)
    ap.add_argument("--target_recalls", default="0.9,0.95,0.99,1.0")
    ap.add_argument("--window_sweep", default="0.005,0.01,0.02,0.03,0.04,0.05,0.06,0.08,0.1,0.12,0.15,0.2")
    ap.add_argument("--out", default="distance_analysis")
    ap.add_argument("--no_plots", action="store_true")
    args = ap.parse_args()

    source = ""
    if args.distances_csv:
        rows = load_distances_csv(Path(args.distances_csv), args.t_min, args.t_max)
        source = args.distances_csv
    elif args.truth and args.dtes:
        truth = load_truth_json(Path(args.truth), args.t_min, args.t_max)
        dtes = load_dtes_json(Path(args.dtes), args.t_min, args.t_max)
        rows = nearest_distances_from_arrays(truth, dtes)
        source = f"truth={args.truth}; dtes={args.dtes}"
    else:
        raise SystemExit("Provide either --distances_csv OR both --truth and --dtes.")

    distances = np.array(
        [float(r["nearest_distance"]) for r in rows if r.get("nearest_distance") is not None],
        dtype=float,
    )

    target_recalls = parse_float_list(args.target_recalls)
    windows = np.array(parse_float_list(args.window_sweep), dtype=float)

    stats = compute_stats(distances, target_recalls)
    sweep = recall_at_windows(distances, windows)

    prefix = Path(args.out)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    out_stats = {
        "source": source,
        "stats": stats,
        "sweep": sweep,
    }

    with prefix.with_name(prefix.name + "_stats.json").open("w", encoding="utf-8") as f:
        json.dump(out_stats, f, indent=2, ensure_ascii=False)

    write_csv(prefix.with_name(prefix.name + "_distances.csv"), rows)
    write_csv(prefix.with_name(prefix.name + "_window_recall.csv"), sweep)
    write_summary(prefix.with_name(prefix.name + "_summary.md"), stats, sweep, source)

    if not args.no_plots:
        plot_histogram(prefix.with_name(prefix.name + "_histogram.png"), distances, stats)
        plot_cdf(prefix.with_name(prefix.name + "_cdf.png"), distances, stats)
        plot_window_recall(prefix.with_name(prefix.name + "_window_recall.png"), sweep)

    print("=== Distance analysis ===")
    print(f"count: {stats['count']}")
    print(f"mean: {stats['mean']}")
    print(f"median: {stats['median']}")
    print(f"max: {stats['max']}")
    print("percentiles:")
    for k, v in stats["percentiles"].items():
        print(f"  {k}: {v}")
    print("recommended windows:")
    for k, v in stats["recommended_windows"].items():
        print(f"  {k}: {v}")
    print(f"summary: {prefix.with_name(prefix.name + '_summary.md')}")


if __name__ == "__main__":
    main()
