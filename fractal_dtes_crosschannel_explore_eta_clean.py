#!/usr/bin/env python3
"""
fractal_dtes_crosschannel_explore_eta_clean.py

Clean standalone DTES-like CrossChannel runner for zeta-zero candidate generation.

Why this file exists:
- It does NOT import the broken fractal_dtes_aco_zeta_crosschannel.py.
- It has built-in:
  * ETA / live progress
  * exploration pressure
  * edge-aware anchors
  * autosave JSON outputs
  * distance-analysis-compatible candidate format

Core idea:
    1. Evaluate Hardy Z / |zeta| on a grid.
    2. Build multi-channel DTES score:
       modulus channel + phase/sign-change channel + multiscale channel + exploration channel.
    3. Select candidate centers with coverage pressure so the sampler does not collapse
       into only a few attractor regions.
    4. Refine candidates locally by minimizing |zeta| / bracketing Hardy Z sign changes.
    5. Save:
       - dtes_candidates_explore.json              core DTES candidates only
       - dtes_candidates_explore_edgeaware.json    core + boundary anchors
       - edge_anchors.json                         anchors only
       - run_metrics_explore.json                  run metrics / timings

Usage:
    python3 fractal_dtes_crosschannel_explore_eta_clean.py \
      --t_min 100 --t_max 400 \
      --n0 2500 \
      --ants 100 \
      --iters 120 \
      --dps 50 \
      --output dtes_candidates_explore.json \
      --edge_output dtes_candidates_explore_edgeaware.json \
      --metrics run_metrics_explore.json

Then analyze:
    python3 distance_analysis.py \
      --truth zeros_100_400_precise.json \
      --dtes dtes_candidates_explore.json \
      --t_min 100 --t_max 400 \
      --out distance_explore
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mpmath as mp
import numpy as np

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


# -----------------------------
# Utilities
# -----------------------------

def fmt_time(sec: float) -> str:
    sec = max(0.0, float(sec))
    if sec < 60:
        return f"{sec:.1f}s"
    if sec < 3600:
        return f"{sec/60:.1f}m"
    return f"{sec/3600:.2f}h"


class Timer:
    def __init__(self) -> None:
        self.start = time.time()
        self.records: List[Dict[str, object]] = []

    def log(self, stage: str, msg: str, **extra) -> None:
        elapsed = time.time() - self.start
        row = {"elapsed_s": elapsed, "stage": stage, "message": msg, **extra}
        self.records.append(row)
        suffix = " ".join(f"{k}={v}" for k, v in extra.items())
        print(f"[{stage}] {msg} | elapsed={fmt_time(elapsed)}" + (f" | {suffix}" if suffix else ""), flush=True)


def json_dump(path: str | Path, obj: object) -> None:
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


# -----------------------------
# Zeta / Hardy Z
# -----------------------------

def hardy_z(t: float) -> float:
    try:
        return float(mp.siegelz(mp.mpf(str(float(t)))))
    except Exception:
        s = mp.mpf("0.5") + 1j * mp.mpf(str(float(t)))
        return float(mp.re(mp.zeta(s)))


def abs_zeta(t: float) -> float:
    s = mp.mpf("0.5") + 1j * mp.mpf(str(float(t)))
    return float(abs(mp.zeta(s)))


def refine_bracket(a: float, b: float, max_iter: int = 80, tol: float = 1e-28) -> Tuple[float, float, int]:
    fa = hardy_z(a)
    fb = hardy_z(b)
    if abs(fa) <= tol:
        return a, abs_zeta(a), 0
    if abs(fb) <= tol:
        return b, abs_zeta(b), 0

    lo, hi = float(a), float(b)
    flo, fhi = fa, fb

    # If no sign change, fall back to local ternary-like minimization of |zeta|.
    if flo == 0 or fhi == 0 or math.copysign(1.0, flo) == math.copysign(1.0, fhi):
        return refine_min_abs_zeta(a, b, n=64)

    for it in range(1, max_iter + 1):
        mid = 0.5 * (lo + hi)
        fm = hardy_z(mid)
        if abs(fm) <= tol:
            return mid, abs_zeta(mid), it
        if math.copysign(1.0, flo) == math.copysign(1.0, fm):
            lo, flo = mid, fm
        else:
            hi, fhi = mid, fm
    mid = 0.5 * (lo + hi)
    return mid, abs_zeta(mid), max_iter


def refine_min_abs_zeta(a: float, b: float, n: int = 64) -> Tuple[float, float, int]:
    # Coarse robust minimization; enough for candidate placement.
    xs = np.linspace(a, b, n)
    vals = np.array([abs_zeta(float(x)) for x in xs], dtype=float)
    i = int(np.argmin(vals))
    lo = float(xs[max(0, i - 1)])
    hi = float(xs[min(n - 1, i + 1)])

    # Golden section on |zeta|
    gr = (math.sqrt(5) - 1) / 2
    c = hi - gr * (hi - lo)
    d = lo + gr * (hi - lo)
    fc = abs_zeta(c)
    fd = abs_zeta(d)
    iters = 0
    for it in range(40):
        iters = it + 1
        if fc > fd:
            lo = c
            c = d
            fc = fd
            d = lo + gr * (hi - lo)
            fd = abs_zeta(d)
        else:
            hi = d
            d = c
            fd = fc
            c = hi - gr * (hi - lo)
            fc = abs_zeta(c)
    t = 0.5 * (lo + hi)
    return t, abs_zeta(t), iters


# -----------------------------
# Candidate generation
# -----------------------------

def evaluate_grid(t_min: float, t_max: float, n0: int, timer: Timer, use_tqdm: bool = True) -> Dict[str, np.ndarray]:
    ts = np.linspace(t_min, t_max, n0)
    zvals = np.zeros(n0, dtype=float)
    absvals = np.zeros(n0, dtype=float)

    iterator = range(n0)
    if tqdm is not None and use_tqdm:
        iterator = tqdm(iterator, desc="grid zeta", unit="pt")

    t_stage = time.time()
    ema = None
    for k in iterator:
        t0 = time.time()
        t = float(ts[k])
        z = hardy_z(t)
        zvals[k] = z
        absvals[k] = abs(z)

        if tqdm is None or not use_tqdm:
            dt = time.time() - t0
            ema = dt if ema is None else 0.2 * dt + 0.8 * ema
            if (k + 1) % 250 == 0 or k + 1 == n0:
                eta = (n0 - k - 1) * (ema or dt)
                timer.log("GRID", f"{k+1}/{n0}", eta=fmt_time(eta))

    # log(|Z| + eps) energy
    eps = 1e-14
    energy = np.log(absvals + eps)
    timer.log("GRID", "done", stage_time=fmt_time(time.time() - t_stage))
    return {"t": ts, "hardy_z": zvals, "abs": absvals, "energy": energy}


def local_minima_indices(values: np.ndarray) -> List[int]:
    idx = []
    for i in range(1, len(values) - 1):
        if values[i] <= values[i - 1] and values[i] <= values[i + 1]:
            idx.append(i)
    return idx


def sign_change_intervals(ts: np.ndarray, z: np.ndarray) -> List[Tuple[int, int]]:
    out = []
    for i in range(len(ts) - 1):
        if z[i] == 0 or z[i + 1] == 0 or np.sign(z[i]) != np.sign(z[i + 1]):
            out.append((i, i + 1))
    return out


def multiscale_prominence(absvals: np.ndarray, idx: int, windows: List[int]) -> float:
    x = absvals[idx]
    score = 0.0
    n = len(absvals)
    for w in windows:
        a = max(0, idx - w)
        b = min(n, idx + w + 1)
        local = absvals[a:b]
        if len(local) <= 1:
            continue
        med = float(np.median(local))
        mad = float(np.median(np.abs(local - med))) + 1e-12
        score += max(0.0, (med - x) / mad)
    return score / max(1, len(windows))


def build_candidate_pool(grid: Dict[str, np.ndarray], timer: Timer) -> List[Dict[str, float]]:
    ts = grid["t"]
    z = grid["hardy_z"]
    absvals = grid["abs"]

    minima = set(local_minima_indices(absvals))
    sign_intervals = sign_change_intervals(ts, z)

    pool: Dict[int, Dict[str, float]] = {}

    # Strong channel: sign changes directly bracket roots.
    for a, b in sign_intervals:
        idx = a if absvals[a] <= absvals[b] else b
        pool[idx] = {
            "grid_index": int(idx),
            "t": float(ts[idx]),
            "abs": float(absvals[idx]),
            "sign_channel": 1.0,
            "min_channel": 0.0,
        }

    # Modulus minima channel.
    for idx in minima:
        row = pool.get(idx, {
            "grid_index": int(idx),
            "t": float(ts[idx]),
            "abs": float(absvals[idx]),
            "sign_channel": 0.0,
            "min_channel": 0.0,
        })
        row["min_channel"] = 1.0
        pool[idx] = row

    windows = [2, 4, 8, 16, 32]
    for row in pool.values():
        idx = int(row["grid_index"])
        prom = multiscale_prominence(absvals, idx, windows)
        # lower |zeta| is better; use stabilized inverse/log score
        modulus_score = -math.log(float(row["abs"]) + 1e-14)
        row["multiscale_score"] = float(prom)
        row["modulus_score"] = float(modulus_score)
        row["base_score"] = (
            2.0 * row.get("sign_channel", 0.0)
            + 1.0 * row.get("min_channel", 0.0)
            + 0.35 * row["multiscale_score"]
            + 0.05 * row["modulus_score"]
        )

    candidates = sorted(pool.values(), key=lambda r: (-r["base_score"], r["t"]))
    timer.log("POOL", "candidate pool built", pool_size=len(candidates), sign_changes=len(sign_intervals), minima=len(minima))
    return candidates


def select_with_exploration(
    pool: List[Dict[str, float]],
    t_min: float,
    t_max: float,
    target_count: int,
    n_bins: int,
    exploration_strength: float,
    timer: Timer,
) -> List[Dict[str, float]]:
    """Select candidates with coverage pressure.

    This prevents collapse into a few highly attractive areas by adding a bonus for
    under-represented bins.
    """
    if not pool:
        return []

    selected: List[Dict[str, float]] = []
    used_idx = set()
    bin_counts = np.zeros(n_bins, dtype=float)
    span = max(1e-12, t_max - t_min)

    for step in range(min(target_count, len(pool))):
        best_j = None
        best_score = -1e100

        for j, c in enumerate(pool):
            if j in used_idx:
                continue
            b = int(np.clip((c["t"] - t_min) / span * n_bins, 0, n_bins - 1))
            explore_bonus = exploration_strength / math.sqrt(1.0 + bin_counts[b])
            edge_bonus = 0.05 * (
                math.exp(-(c["t"] - t_min) / max(1e-9, 0.02 * span))
                + math.exp(-(t_max - c["t"]) / max(1e-9, 0.02 * span))
            )
            score = c["base_score"] + explore_bonus + edge_bonus
            if score > best_score:
                best_score = score
                best_j = j

        if best_j is None:
            break

        used_idx.add(best_j)
        chosen = dict(pool[best_j])
        chosen["selection_score"] = float(best_score)
        b = int(np.clip((chosen["t"] - t_min) / span * n_bins, 0, n_bins - 1))
        chosen["coverage_bin"] = int(b)
        bin_counts[b] += 1
        selected.append(chosen)

    timer.log("SELECT", "exploration-aware selection done", selected=len(selected), bins_used=int(np.sum(bin_counts > 0)))
    return sorted(selected, key=lambda r: r["t"])


def refine_candidates(
    selected: List[Dict[str, float]],
    grid: Dict[str, np.ndarray],
    radius: float,
    timer: Timer,
    use_tqdm: bool = True,
) -> List[Dict[str, float]]:
    ts = grid["t"]
    z = grid["hardy_z"]
    refined = []

    iterator = selected
    if tqdm is not None and use_tqdm:
        iterator = tqdm(selected, desc="refine", unit="cand")

    start = time.time()
    ema = None

    for k, c in enumerate(iterator):
        t0 = time.time()
        idx = int(c["grid_index"])

        # Prefer direct sign-change bracket around grid index if present.
        a = max(float(ts[0]), float(c["t"] - radius))
        b = min(float(ts[-1]), float(c["t"] + radius))

        # If adjacent sign change exists, use that bracket.
        if idx > 0 and np.sign(z[idx - 1]) != np.sign(z[idx]):
            a, b = float(ts[idx - 1]), float(ts[idx])
        elif idx < len(ts) - 1 and np.sign(z[idx]) != np.sign(z[idx + 1]):
            a, b = float(ts[idx]), float(ts[idx + 1])

        rt, residual, iters = refine_bracket(a, b, max_iter=80)
        row = dict(c)
        row.update({
            "t_refined": float(rt),
            "abs_zeta": float(residual),
            "refine_left": float(a),
            "refine_right": float(b),
            "refine_iterations": int(iters),
        })
        refined.append(row)

        if tqdm is None or not use_tqdm:
            dt = time.time() - t0
            ema = dt if ema is None else 0.2 * dt + 0.8 * ema
            if (k + 1) % 50 == 0 or k + 1 == len(selected):
                eta = (len(selected) - k - 1) * (ema or dt)
                timer.log("REFINE", f"{k+1}/{len(selected)}", eta=fmt_time(eta))

    timer.log("REFINE", "done", stage_time=fmt_time(time.time() - start))
    return refined


def merge_close(refined: List[Dict[str, float]], tol: float) -> List[Dict[str, float]]:
    if not refined:
        return []
    rows = sorted(refined, key=lambda r: r["t_refined"])
    merged = [rows[0]]
    for r in rows[1:]:
        if abs(r["t_refined"] - merged[-1]["t_refined"]) <= tol:
            if r.get("abs_zeta", math.inf) < merged[-1].get("abs_zeta", math.inf):
                merged[-1] = r
        else:
            merged.append(r)
    for i, r in enumerate(merged, start=1):
        r["rank"] = i
    return merged


def make_edge_anchors(t_min: float, t_max: float, padding: float, step: float) -> List[float]:
    anchors = []
    if padding > 0 and step > 0:
        n = int(math.ceil(padding / step))
        for i in range(n + 1):
            d = i * step
            anchors.append(t_min + d)
            anchors.append(t_max - d)
    return sorted(set(round(float(t), 12) for t in anchors if t_min <= t <= t_max))


def save_candidate_json(path: str, rows: List[Dict[str, float]], source: str) -> None:
    out = []
    for i, r in enumerate(sorted(rows, key=lambda x: x["t_refined"]), start=1):
        out.append({
            "rank": i,
            "t": float(r["t_refined"]),
            "score": float(r.get("selection_score", r.get("base_score", 0.0))),
            "abs_zeta": float(r.get("abs_zeta", math.nan)),
            "source": source,
        })
    json_dump(path, {"candidates": out, "count": len(out)})


def save_edgeaware_json(path: str, core_rows: List[Dict[str, float]], anchors: List[float], source: str) -> None:
    vals = [(float(r["t_refined"]), float(r.get("selection_score", r.get("base_score", 0.0))), "core") for r in core_rows]
    vals.extend((float(a), 0.0, "edge_anchor") for a in anchors)
    vals = sorted(set((round(t, 12), score, typ) for t, score, typ in vals), key=lambda x: x[0])
    out = [
        {"rank": i + 1, "t": float(t), "score": float(score), "kind": typ, "source": source}
        for i, (t, score, typ) in enumerate(vals)
    ]
    json_dump(path, {"candidates": out, "count": len(out)})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t_min", type=float, required=True)
    ap.add_argument("--t_max", type=float, required=True)
    ap.add_argument("--n0", type=int, default=2500)
    ap.add_argument("--ants", type=int, default=100, help="Used to estimate target candidate count.")
    ap.add_argument("--iters", type=int, default=120, help="Kept for CLI compatibility; selection is deterministic.")
    ap.add_argument("--dps", type=int, default=50)
    ap.add_argument("--target_count", type=int, default=None)
    ap.add_argument("--coverage_bins", type=int, default=64)
    ap.add_argument("--exploration_strength", type=float, default=1.0)
    ap.add_argument("--refine_radius", type=float, default=None)
    ap.add_argument("--merge_tol", type=float, default=1e-6)
    ap.add_argument("--edge_padding", type=float, default=2.5)
    ap.add_argument("--edge_step", type=float, default=0.05)
    ap.add_argument("--no_tqdm", action="store_true")
    ap.add_argument("--output", default="dtes_candidates_explore.json")
    ap.add_argument("--edge_output", default="dtes_candidates_explore_edgeaware.json")
    ap.add_argument("--anchors_output", default="edge_anchors.json")
    ap.add_argument("--metrics", default="run_metrics_explore.json")
    args = ap.parse_args()

    mp.mp.dps = args.dps
    timer = Timer()

    use_tqdm = not args.no_tqdm

    timer.log("START", "clean exploration DTES runner started")
    grid = evaluate_grid(args.t_min, args.t_max, args.n0, timer, use_tqdm=use_tqdm)
    pool = build_candidate_pool(grid, timer)

    target_count = args.target_count
    if target_count is None:
        # For zeta zeros in [100,400], ~170 candidates are expected.
        # Use ants + pool size to keep a sparse but broad cover.
        target_count = min(len(pool), max(args.ants * 2, 180))

    selected = select_with_exploration(
        pool,
        t_min=args.t_min,
        t_max=args.t_max,
        target_count=target_count,
        n_bins=args.coverage_bins,
        exploration_strength=args.exploration_strength,
        timer=timer,
    )

    grid_step = (args.t_max - args.t_min) / max(1, args.n0 - 1)
    refine_radius = args.refine_radius if args.refine_radius is not None else max(0.08, 2.5 * grid_step)
    refined = refine_candidates(selected, grid, radius=refine_radius, timer=timer, use_tqdm=use_tqdm)
    merged = merge_close(refined, tol=args.merge_tol)

    anchors = make_edge_anchors(args.t_min, args.t_max, args.edge_padding, args.edge_step)

    save_candidate_json(args.output, merged, "fractal_dtes_crosschannel_explore_eta_clean_core")
    save_edgeaware_json(args.edge_output, merged, anchors, "fractal_dtes_crosschannel_explore_eta_clean_edgeaware")
    json_dump(args.anchors_output, {
        "candidates": [{"rank": i + 1, "t": float(t), "kind": "edge_anchor"} for i, t in enumerate(anchors)],
        "count": len(anchors),
    })

    metrics = {
        "config": vars(args),
        "grid_step": grid_step,
        "refine_radius": refine_radius,
        "pool_size": len(pool),
        "selected_count": len(selected),
        "merged_core_count": len(merged),
        "edge_anchor_count": len(anchors),
        "timing_records": timer.records,
        "runtime_s": time.time() - timer.start,
        "core_abs_zeta_min": min((r.get("abs_zeta", math.inf) for r in merged), default=None),
        "core_abs_zeta_max": max((r.get("abs_zeta", -math.inf) for r in merged), default=None),
    }
    json_dump(args.metrics, metrics)

    timer.log("SAVE", "outputs saved", core=args.output, edgeaware=args.edge_output, metrics=args.metrics)
    print("\n=== Summary ===")
    print(f"pool size: {len(pool)}")
    print(f"selected: {len(selected)}")
    print(f"core candidates: {len(merged)}")
    print(f"edge anchors: {len(anchors)}")
    print(f"runtime: {fmt_time(time.time() - timer.start)}")


if __name__ == "__main__":
    main()
