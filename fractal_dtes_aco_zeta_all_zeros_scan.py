#!/usr/bin/env python3
"""
fractal_dtes_aco_zeta_all_zeros_scan.py

Ground-truth generator for Riemann zeta zeros on the critical line.

Compatible with run_full_pipeline.sh:
python3 validation/fractal_dtes_aco_zeta_all_zeros_scan.py --t_min 100 --t_max 400 --step 0.01 --dps 80 --output runs/zeros_100_400_precise

Ant-RH:
    First stage in the README pipeline (“ground truth Hardy-Z scan”). Feeds
    ``distance_analysis.py`` / hybrid evaluation as ``--truth`` JSON.
    Identical logic lives under ``validation/`` for shell paths that ``cd`` there.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import mpmath as mp


def fmt_time(sec: float) -> str:
    sec = max(0.0, float(sec))
    if sec < 60:
        return f"{sec:.1f}s"
    if sec < 3600:
        return f"{sec/60:.1f}m"
    return f"{sec/3600:.2f}h"


def hardy_z(t) -> mp.mpf:
    try:
        return mp.siegelz(t)
    except Exception:
        s = mp.mpf("0.5") + 1j * mp.mpf(str(float(t)))
        return mp.re(mp.zeta(s))


def abs_zeta(t) -> mp.mpf:
    s = mp.mpf("0.5") + 1j * mp.mpf(str(float(t)))
    return abs(mp.zeta(s))


def bisect_root(a: mp.mpf, b: mp.mpf, fa: mp.mpf, fb: mp.mpf, max_iter: int, value_tol: mp.mpf) -> Tuple[mp.mpf, mp.mpf, int]:
    lo, hi = mp.mpf(a), mp.mpf(b)
    flo, fhi = mp.mpf(fa), mp.mpf(fb)

    if abs(flo) <= value_tol:
        return lo, flo, 0
    if abs(fhi) <= value_tol:
        return hi, fhi, 0

    for it in range(1, max_iter + 1):
        mid = (lo + hi) / 2
        fm = hardy_z(mid)
        if abs(fm) <= value_tol:
            return mid, fm, it
        if mp.sign(flo) == mp.sign(fm):
            lo, flo = mid, fm
        else:
            hi, fhi = mid, fm

    mid = (lo + hi) / 2
    return mid, hardy_z(mid), max_iter


def merge_close_zeros(zeros: List[Dict], tol_t: float) -> List[Dict]:
    if not zeros:
        return []
    zeros = sorted(zeros, key=lambda z: z["t"])
    merged = [zeros[0]]
    for z in zeros[1:]:
        if abs(z["t"] - merged[-1]["t"]) <= tol_t:
            if z.get("abs_zeta", math.inf) < merged[-1].get("abs_zeta", math.inf):
                merged[-1] = z
        else:
            merged.append(z)
    for i, z in enumerate(merged, start=1):
        z["index_in_run"] = i
    return merged


def scan_zeros(t_min: float, t_max: float, step: float, max_bisect_iter: int, zero_value_tol: float, tol_t: float, progress_every: int) -> List[Dict]:
    # Uniform t-grid: sign changes of Z(t) bracket zeros; bisection refines each bracket.
    start = time.time()

    a0 = mp.mpf(str(t_min))
    b0 = mp.mpf(str(t_max))
    h = mp.mpf(str(step))
    value_tol = mp.mpf(str(zero_value_tol))

    n_steps = int(mp.ceil((b0 - a0) / h))
    total_points = n_steps + 1
    zeros: List[Dict] = []

    prev_t = a0
    prev_z = hardy_z(prev_t)

    ema = None
    last = time.time()

    for i in range(1, total_points + 1):
        t = a0 + i * h
        if t > b0:
            t = b0

        z = hardy_z(t)
        has_hit = False
        root_t = None
        root_z = None
        iters = 0

        if abs(prev_z) <= value_tol:
            root_t, root_z, iters = prev_t, prev_z, 0
            has_hit = True
        elif abs(z) <= value_tol:
            root_t, root_z, iters = t, z, 0
            has_hit = True
        elif mp.sign(prev_z) != mp.sign(z):
            root_t, root_z, iters = bisect_root(prev_t, t, prev_z, z, max_bisect_iter, value_tol)
            has_hit = True

        if has_hit and root_t is not None:
            az = abs_zeta(root_t)
            zeros.append({
                "index_in_run": len(zeros) + 1,
                "t": float(root_t),
                "abs_zeta": float(az),
                "hardy_z": float(root_z),
                "bracket_left": float(prev_t),
                "bracket_right": float(t),
                "bracket_width": float(t - prev_t),
                "iterations": int(iters),
                "method": "hardy_z_sign_scan_bisection",
            })

        now = time.time()
        dt = now - last
        last = now
        ema = dt if ema is None else 0.2 * dt + 0.8 * ema

        if progress_every > 0 and (i % progress_every == 0 or i == total_points):
            remain = total_points - i
            eta = remain * (ema or dt)
            elapsed = now - start
            speed = i / elapsed if elapsed > 0 else 0.0
            print(
                f"[SCAN] {i}/{total_points} | zeros={len(zeros)} "
                f"| elapsed={fmt_time(elapsed)} | ETA={fmt_time(eta)} "
                f"| speed={speed:.1f} pts/s",
                flush=True,
            )

        prev_t, prev_z = t, z
        if t >= b0:
            break

    return merge_close_zeros(zeros, tol_t=tol_t)


def write_csv(path: Path, rows: List[Dict]) -> None:
    keys = ["index_in_run", "t", "abs_zeta", "hardy_z", "bracket_left", "bracket_right", "bracket_width", "iterations", "method"]
    with path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k) for k in keys})


def write_txt(path: Path, rows: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for z in rows:
            f.write(
                f"{z['index_in_run']:5d} "
                f"t={z['t']:.15f} "
                f"|zeta|={z['abs_zeta']:.3e} "
                f"bracket=[{z['bracket_left']:.6f},{z['bracket_right']:.6f}]\n"
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t_min", type=float, required=True)
    ap.add_argument("--t_max", type=float, required=True)
    ap.add_argument("--step", type=float, default=0.01)
    ap.add_argument("--dps", type=int, default=80)
    ap.add_argument("--tol_t", type=float, default=1e-12)
    ap.add_argument("--zero_value_tol", type=float, default=1e-30)
    ap.add_argument("--max_bisect_iter", type=int, default=120)
    ap.add_argument("--progress_every", type=int, default=500)
    ap.add_argument("--output", required=True, help="Output prefix, without extension.")
    args = ap.parse_args()

    mp.mp.dps = args.dps
    started = time.time()

    print("=== Ground truth Hardy-Z zero scan ===")
    print(f"range=[{args.t_min}, {args.t_max}] step={args.step} dps={args.dps}")

    zeros = scan_zeros(
        t_min=args.t_min,
        t_max=args.t_max,
        step=args.step,
        max_bisect_iter=args.max_bisect_iter,
        zero_value_tol=args.zero_value_tol,
        tol_t=args.tol_t,
        progress_every=args.progress_every,
    )

    prefix = Path(args.output)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "config": vars(args),
        "count": len(zeros),
        "zeros": zeros,
        "runtime_s": time.time() - started,
    }

    json_path = prefix.with_suffix(".json")
    csv_path = prefix.with_suffix(".csv")
    txt_path = prefix.with_suffix(".txt")

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    write_csv(csv_path, zeros)
    write_txt(txt_path, zeros)

    print("=== Done ===")
    print(f"count: {len(zeros)}")
    print(f"runtime: {fmt_time(time.time() - started)}")
    print(f"saved: {json_path}")
    print(f"saved: {csv_path}")
    print(f"saved: {txt_path}")


if __name__ == "__main__":
    main()
