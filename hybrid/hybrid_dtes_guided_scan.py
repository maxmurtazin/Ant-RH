#!/usr/bin/env python3
"""
hybrid_dtes_guided_scan.py

Hybrid DTES-guided zero scan for Riemann zeta zeros.

Pipeline:
    1. Load DTES candidates from JSON.
    2. Build windows [candidate - window, candidate + window].
    3. Clip to [t_min, t_max].
    4. Merge overlapping windows.
    5. Scan Hardy Z only inside these windows.
    6. Refine sign-change brackets by bisection.
    7. Save zeros + hybrid statistics.

Why:
    Full scan evaluates every point in [t_min, t_max].
    Hybrid scan evaluates only DTES-selected windows.

Input DTES JSON supported:
    [101.3, 103.7, ...]
    {"candidates": [101.3, ...]}
    {"candidates": [{"t": 101.3, "score": ...}, ...]}
    {"zeros": [...]}

Output:
    <out>.json
    <out>.csv
    <out>.txt
    <out>_stats.json
    <out>_windows.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import mpmath as mp
import numpy as np


# ------------------------------
# IO helpers
# ------------------------------
# Throughout this script, ``t`` is the Gram/Hardy axis: s = 1/2 + i t on the critical line.

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_sequence(data: Any) -> List[Any]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("candidates", "dtes_candidates", "zeros", "results"):
            if key in data and isinstance(data[key], list):
                return data[key]
        for val in data.values():
            if isinstance(val, list):
                return val
    raise ValueError("Could not find candidates list in DTES JSON.")


def extract_t(item: Any) -> float:
    if isinstance(item, (int, float)):
        return float(item)
    if isinstance(item, dict):
        for key in ("t", "time", "candidate", "zero", "value"):
            if key in item:
                return float(item[key])
    raise ValueError(f"Cannot extract t from item={item!r}")


def extract_score(item: Any) -> Optional[float]:
    if isinstance(item, dict):
        for key in ("score", "dtes_score", "energy_score", "rank_score", "pheromone_score"):
            if key in item and item[key] is not None:
                return float(item[key])
    return None


def load_dtes_candidates(path: Path, t_min: float, t_max: float, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
    data = load_json(path)
    seq = extract_sequence(data)
    rows = []
    for item in seq:
        t = extract_t(item)
        if t < t_min or t > t_max:
            continue
        rows.append({
            "candidate_index": len(rows),
            "t": t,
            "score": extract_score(item),
        })

    if top_k is not None:
        if any(r.get("score") is not None for r in rows):
            rows.sort(key=lambda r: (-(r.get("score") or -math.inf), r["t"]))
        rows = rows[:top_k]

    rows.sort(key=lambda r: r["t"])
    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    keys = []
    for row in rows:
        for k, v in row.items():
            if isinstance(v, (dict, list)):
                continue
            if k not in keys:
                keys.append(k)

    with path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for row in rows:
            wr.writerow({k: row.get(k) for k in keys})


# ------------------------------
# Windows
# ------------------------------

def merge_intervals(intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [[intervals[0][0], intervals[0][1]]]
    for a, b in intervals[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(float(a), float(b)) for a, b in merged]


def build_windows(
    candidates: List[Dict[str, Any]],
    t_min: float,
    t_max: float,
    window: float,
) -> List[Dict[str, Any]]:
    raw = []
    for c in candidates:
        a = max(t_min, c["t"] - window)
        b = min(t_max, c["t"] + window)
        if b >= a:
            raw.append((a, b))
    merged = merge_intervals(raw)
    return [
        {"window_index": i + 1, "start": a, "end": b, "length": b - a}
        for i, (a, b) in enumerate(merged)
    ]


def total_window_length(windows: List[Dict[str, Any]]) -> float:
    return float(sum(max(0.0, w["end"] - w["start"]) for w in windows))


# ------------------------------
# Hardy Z / root refinement
# ------------------------------

def hardy_z(t: float) -> mp.mpf:
    """Real Hardy Z value on the critical line.

    Z(t) = exp(i theta(t)) zeta(1/2 + it), real for real t.
    mpmath.siegelz is used when available because it is faster/stable.
    """
    try:
        return mp.siegelz(t)
    except Exception:
        s = mp.mpf("0.5") + 1j * mp.mpf(str(float(t)))
        return mp.re(mp.zeta(s))


def abs_zeta(t: float) -> mp.mpf:
    s = mp.mpf("0.5") + 1j * mp.mpf(str(float(t)))
    return abs(mp.zeta(s))


def bisect_root(a: float, b: float, fa: mp.mpf, fb: mp.mpf, max_iter: int, value_tol: mp.mpf) -> Tuple[float, mp.mpf, int]:
    lo = mp.mpf(str(float(a)))
    hi = mp.mpf(str(float(b)))
    flo = mp.mpf(fa)
    fhi = mp.mpf(fb)

    if abs(flo) <= value_tol:
        return float(lo), flo, 0
    if abs(fhi) <= value_tol:
        return float(hi), fhi, 0

    for it in range(1, max_iter + 1):
        mid = (lo + hi) / 2
        fm = hardy_z(mid)

        if abs(fm) <= value_tol:
            return float(mid), fm, it

        if flo == 0:
            return float(lo), flo, it
        if fhi == 0:
            return float(hi), fhi, it

        if mp.sign(flo) == mp.sign(fm):
            lo = mid
            flo = fm
        else:
            hi = mid
            fhi = fm

    mid = (lo + hi) / 2
    fm = hardy_z(mid)
    return float(mid), fm, max_iter


def scan_window(
    window: Dict[str, Any],
    step: float,
    max_bisect_iter: int,
    zero_value_tol: mp.mpf,
    progress: bool = False,
) -> List[Dict[str, Any]]:
    a = float(window["start"])
    b = float(window["end"])
    if b < a:
        return []

    zeros = []

    # Build stable grid including endpoint.
    n_steps = max(1, int(math.ceil((b - a) / step)))
    ts = [a + i * step for i in range(n_steps)]
    if not ts or ts[-1] < b:
        ts.append(b)

    prev_t = ts[0]
    prev_z = hardy_z(prev_t)

    for t in ts[1:]:
        z = hardy_z(t)

        # Sign change or exact endpoint hit.
        hit = False
        if abs(prev_z) <= zero_value_tol:
            root_t, root_z, iters = prev_t, prev_z, 0
            hit = True
        elif abs(z) <= zero_value_tol:
            root_t, root_z, iters = t, z, 0
            hit = True
        elif mp.sign(prev_z) != mp.sign(z):
            root_t, root_z, iters = bisect_root(prev_t, t, prev_z, z, max_bisect_iter, zero_value_tol)
            hit = True

        if hit:
            az = abs_zeta(root_t)
            zeros.append({
                "t": float(root_t),
                "abs_zeta": float(az),
                "hardy_z": float(root_z),
                "bracket_left": float(prev_t),
                "bracket_right": float(t),
                "bracket_width": float(t - prev_t),
                "iterations": int(iters),
                "window_index": int(window["window_index"]),
                "method": "hybrid_dtes_guided_hardy_z_bisection",
            })

        prev_t, prev_z = t, z

    return zeros


def merge_close_zeros(zeros: List[Dict[str, Any]], tol: float = 1e-7) -> List[Dict[str, Any]]:
    if not zeros:
        return []
    zeros = sorted(zeros, key=lambda z: z["t"])
    merged = [zeros[0]]
    for z in zeros[1:]:
        if abs(z["t"] - merged[-1]["t"]) <= tol:
            # Keep better residual.
            if z.get("abs_zeta", math.inf) < merged[-1].get("abs_zeta", math.inf):
                merged[-1] = z
        else:
            merged.append(z)
    for i, z in enumerate(merged, start=1):
        z["index_in_run"] = i
    return merged


# ------------------------------
# ETA
# ------------------------------

class ETA:
    def __init__(self, total: int, every: int = 1, alpha: float = 0.2, label: str = "SCAN"):
        self.total = max(1, int(total))
        self.every = max(1, int(every))
        self.alpha = alpha
        self.label = label
        self.start = time.time()
        self.last = self.start
        self.ema_dt = None

    def update(self, i: int, extra: str = "") -> None:
        now = time.time()
        dt = now - self.last
        self.last = now

        if self.ema_dt is None:
            self.ema_dt = dt
        else:
            self.ema_dt = self.alpha * dt + (1 - self.alpha) * self.ema_dt

        done = i + 1
        remain = self.total - done
        eta = max(0.0, remain * (self.ema_dt or 0.0))
        elapsed = now - self.start

        if done % self.every == 0 or done == self.total:
            speed = done / elapsed if elapsed > 0 else 0.0
            print(
                f"[{self.label}] {done}/{self.total} "
                f"| elapsed={fmt_time(elapsed)} "
                f"| ETA={fmt_time(eta)} "
                f"| speed={speed:.2f} windows/s"
                + (f" | {extra}" if extra else ""),
                flush=True,
            )


def fmt_time(sec: float) -> str:
    sec = max(0.0, float(sec))
    if sec < 60:
        return f"{sec:.1f}s"
    if sec < 3600:
        return f"{sec/60:.1f}m"
    return f"{sec/3600:.2f}h"


# ------------------------------
# Main
# ------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Hybrid DTES-guided Hardy-Z scan.")
    ap.add_argument("--dtes", required=True, help="DTES candidates JSON.")
    ap.add_argument("--t_min", type=float, required=True)
    ap.add_argument("--t_max", type=float, required=True)
    ap.add_argument("--window", type=float, default=0.2, help="Half-window around each DTES candidate.")
    ap.add_argument("--top_k", type=int, default=None, help="Use only top-k candidates.")
    ap.add_argument("--step", type=float, default=0.01, help="Scan step inside hybrid windows.")
    ap.add_argument("--dps", type=int, default=80)
    ap.add_argument("--zero_value_tol", type=float, default=1e-30)
    ap.add_argument("--verify_abs_zeta", type=float, default=1e-8)
    ap.add_argument("--max_bisect_iter", type=int, default=120)
    ap.add_argument("--merge_tol", type=float, default=1e-7)
    ap.add_argument("--progress_every", type=int, default=1, help="Print ETA every N windows.")
    ap.add_argument("--out", default="hybrid_zeros")
    args = ap.parse_args()

    mp.mp.dps = int(args.dps)

    started = time.time()

    candidates = load_dtes_candidates(Path(args.dtes), args.t_min, args.t_max, top_k=args.top_k)
    windows = build_windows(candidates, args.t_min, args.t_max, args.window)

    full_length = float(args.t_max - args.t_min)
    hybrid_length = total_window_length(windows)
    full_points_est = int(math.ceil(full_length / args.step)) + 1
    hybrid_points_est = int(sum(max(1, math.ceil(w["length"] / args.step)) + 1 for w in windows))

    print("=== Hybrid DTES-guided scan ===")
    print(f"DTES candidates in range: {len(candidates)}")
    print(f"windows after merge: {len(windows)}")
    print(f"full length: {full_length:.6f}")
    print(f"hybrid length: {hybrid_length:.6f}")
    print(f"scanned fraction: {hybrid_length/full_length if full_length > 0 else 0:.6f}")
    print(f"estimated speedup by length: {full_length/hybrid_length if hybrid_length > 0 else math.inf:.3f}x")
    print(f"full points est: {full_points_est}")
    print(f"hybrid points est: {hybrid_points_est}")
    print(f"estimated speedup by points: {full_points_est/hybrid_points_est if hybrid_points_est > 0 else math.inf:.3f}x")

    all_zeros = []
    eta = ETA(total=len(windows), every=args.progress_every, label="HYBRID")
    for i, w in enumerate(windows):
        z = scan_window(
            w,
            step=args.step,
            max_bisect_iter=args.max_bisect_iter,
            zero_value_tol=mp.mpf(str(args.zero_value_tol)),
        )
        all_zeros.extend(z)
        eta.update(i, extra=f"zeros={len(all_zeros)} last_window=[{w['start']:.3f},{w['end']:.3f}]")

    zeros = merge_close_zeros(all_zeros, tol=args.merge_tol)

    # Verification count.
    verified = [z for z in zeros if z.get("abs_zeta", math.inf) <= args.verify_abs_zeta]

    prefix = Path(args.out)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "config": vars(args),
        "count": len(zeros),
        "verified_count": len(verified),
        "zeros": zeros,
    }

    stats = {
        "config": vars(args),
        "runtime_s": time.time() - started,
        "dtes_candidates_in_range": len(candidates),
        "windows_after_merge": len(windows),
        "full_length": full_length,
        "hybrid_length": hybrid_length,
        "scanned_fraction": hybrid_length / full_length if full_length > 0 else None,
        "estimated_speedup_by_length": full_length / hybrid_length if hybrid_length > 0 else math.inf,
        "full_points_est": full_points_est,
        "hybrid_points_est": hybrid_points_est,
        "estimated_speedup_by_points": full_points_est / hybrid_points_est if hybrid_points_est > 0 else math.inf,
        "found_zeros": len(zeros),
        "verified_zeros": len(verified),
        "min_abs_zeta": min((z["abs_zeta"] for z in zeros), default=None),
        "max_abs_zeta": max((z["abs_zeta"] for z in zeros), default=None),
    }

    json_path = prefix.with_suffix(".json")
    csv_path = prefix.with_suffix(".csv")
    txt_path = prefix.with_suffix(".txt")
    stats_path = prefix.with_name(prefix.name + "_stats.json")
    windows_path = prefix.with_name(prefix.name + "_windows.csv")
    candidates_path = prefix.with_name(prefix.name + "_used_candidates.csv")

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    write_csv(csv_path, zeros)
    write_csv(windows_path, windows)
    write_csv(candidates_path, candidates)

    with txt_path.open("w", encoding="utf-8") as f:
        for z in zeros:
            f.write(f"{z['index_in_run']:5d}  t={z['t']:.15f}  |zeta|={z['abs_zeta']:.3e}  window={z['window_index']}\n")

    print("\n=== Done ===")
    print(f"found zeros: {len(zeros)}")
    print(f"verified zeros: {len(verified)}")
    print(f"runtime: {fmt_time(time.time() - started)}")
    print(f"saved: {json_path}")
    print(f"saved: {stats_path}")
    print(f"saved: {csv_path}")
    print(f"saved: {txt_path}")
    print(f"saved: {windows_path}")


if __name__ == "__main__":
    main()
