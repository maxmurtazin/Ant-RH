#!/usr/bin/env python3
"""
run_crosschannel_fixed.py

Robust CLI runner for fractal_dtes_aco_zeta_crosschannel.py.

Why this exists:
    Some generated crosschannel files have default_demo() hardcoded and ignore CLI args.
    This runner imports the actual FractalDTESACOZeta / ZetaSearchConfig classes,
    sets t_min/t_max/n_grid/ants/iters explicitly, filters candidates by interval,
    adds optional edge anchors, and saves dtes_candidates.json.

Usage:
    python3 run_crosschannel_fixed.py \
      --t_min 100 --t_max 400 \
      --n0 2500 \
      --ants 60 \
      --iters 80 \
      --output dtes_candidates.json

Then:
    python3 distance_analysis.py \
      --truth zeros_100_400_precise.json \
      --dtes dtes_candidates.json \
      --t_min 100 --t_max 400 \
      --out distance_edge_fixed

Ant-RH:
    Thin CLI over ``fractal_dtes_aco_zeta_crosschannel`` when you need explicit
    ``ZetaSearchConfig`` without the interactive demo path.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import List

import mpmath as mp

from fractal_dtes_aco_zeta_crosschannel import FractalDTESACOZeta, ZetaSearchConfig


def add_edge_anchors(
    candidates: List[float],
    t_min: float,
    t_max: float,
    padding: float,
    step: float,
) -> List[float]:
    cleaned = []
    for t in candidates:
        try:
            tt = float(t)
        except Exception:
            continue
        if t_min <= tt <= t_max:
            cleaned.append(tt)

    anchors = []
    if padding > 0 and step > 0:
        n = int(math.ceil(padding / step))
        for i in range(n + 1):
            d = i * step
            anchors.append(t_min + d)
            anchors.append(t_max - d)

    merged = cleaned + [t for t in anchors if t_min <= t <= t_max]
    return sorted(set(round(float(t), 12) for t in merged))


def save_candidates(candidates: List[float], output: str) -> None:
    rows = [
        {
            "rank": i + 1,
            "t": float(t),
            "source": "run_crosschannel_fixed",
        }
        for i, t in enumerate(candidates)
    ]
    with open(output, "w", encoding="utf-8") as f:
        json.dump({"candidates": rows, "count": len(rows)}, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] {output} | count={len(rows)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t_min", type=float, required=True)
    ap.add_argument("--t_max", type=float, required=True)
    ap.add_argument("--n0", type=int, default=2500)
    ap.add_argument("--levels", type=int, default=8)
    ap.add_argument("--feature_levels", type=int, default=5)
    ap.add_argument("--ants", type=int, default=60)
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--max_ant_steps", type=int, default=20)
    ap.add_argument("--top_candidate_nodes", type=int, default=256)
    ap.add_argument("--r0", type=float, default=6.0)
    ap.add_argument("--dps", type=int, default=50)
    ap.add_argument("--edge_padding", type=float, default=2.5)
    ap.add_argument("--edge_step", type=float, default=0.05)
    ap.add_argument("--output", default="dtes_candidates.json")
    args = ap.parse_args()

    mp.mp.dps = args.dps

    cfg = ZetaSearchConfig(
        t_min=args.t_min,
        t_max=args.t_max,
        n_grid=args.n0,
        tree_depth=args.levels,
        feature_levels=args.feature_levels,
        n_ants=args.ants,
        n_iterations=args.iters,
        max_ant_steps=args.max_ant_steps,
        top_candidate_nodes=args.top_candidate_nodes,
        r0=args.r0,
        mp_dps=args.dps,
    )

    print("=== Fixed CrossChannel runner ===")
    print(f"range: [{cfg.t_min}, {cfg.t_max}]")
    print(f"n_grid: {cfg.n_grid}")
    print(f"ants: {cfg.n_ants}")
    print(f"iters: {cfg.n_iterations}")

    searcher = FractalDTESACOZeta(cfg)
    raw_candidates = searcher.run()

    interval_candidates = [
        float(t) for t in raw_candidates
        if args.t_min <= float(t) <= args.t_max
    ]

    candidates = add_edge_anchors(
        interval_candidates,
        args.t_min,
        args.t_max,
        padding=args.edge_padding,
        step=args.edge_step,
    )

    print("\nRaw candidates:", len(raw_candidates))
    print("Interval-filtered candidates:", len(interval_candidates))
    print("Edge-aware candidates:", len(candidates))

    print("\nCandidates preview:")
    for t in candidates[:20]:
        try:
            z = mp.zeta(mp.mpf("0.5") + 1j * mp.mpf(str(float(t))))
            print(f"t = {t:.12f} |zeta| = {abs(complex(z)):.3e}")
        except Exception:
            print(f"t = {t:.12f}")
    if len(candidates) > 20:
        print(f"... ({len(candidates)} total)")

    save_candidates(candidates, args.output)


if __name__ == "__main__":
    main()
