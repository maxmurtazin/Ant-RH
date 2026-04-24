#!/usr/bin/env python3
"""
run_crosschannel_live_v2.py

Safer CrossChannel DTES runner.

Important fix vs previous runner:
- dtes_candidates.json contains ONLY core DTES candidates.
- dtes_candidates_edgeaware.json contains core DTES candidates + edge anchors.
- edge_anchors.json contains only boundary anchors.
- run_metrics.json records raw/filtered/anchor counts.

Why:
    Edge anchors are useful for hybrid boundary coverage, but they must not replace
    the real DTES candidate set when analyzing DTES distance-to-zero quality.

Usage:
    python3 run_crosschannel_live_v2.py \
      --t_min 100 --t_max 400 \
      --n0 2500 \
      --ants 60 \
      --iters 80 \
      --output dtes_candidates.json \
      --edge_output dtes_candidates_edgeaware.json \
      --metrics run_metrics.json

For pure DTES analysis:
    use dtes_candidates.json

For hybrid scan with boundary protection:
    use dtes_candidates_edgeaware.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional

import mpmath as mp
import numpy as np

from fractal_dtes_aco_zeta_crosschannel import (
    AntPath,
    FractalDTESACOZeta,
    ZetaSearchConfig,
)

try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def fmt_time(sec: float) -> str:
    sec = max(0.0, float(sec))
    if sec < 60:
        return f"{sec:.1f}s"
    if sec < 3600:
        return f"{sec/60:.1f}m"
    return f"{sec/3600:.2f}h"


class StageTimer:
    def __init__(self) -> None:
        self.t0 = time.time()
        self.records: List[Dict[str, object]] = []

    def log(self, stage: str, message: str, **extra) -> None:
        elapsed = time.time() - self.t0
        row = {"elapsed_s": elapsed, "stage": stage, "message": message, **extra}
        self.records.append(row)
        suffix = " ".join(f"{k}={v}" for k, v in extra.items())
        print(f"[{stage}] {message} | elapsed={fmt_time(elapsed)}" + (f" | {suffix}" if suffix else ""), flush=True)


class LiveFractalDTESACOZeta(FractalDTESACOZeta):
    def __init__(self, cfg: ZetaSearchConfig, timer: StageTimer, progress_every: int = 1, use_tqdm: bool = True):
        super().__init__(cfg)
        self.timer = timer
        self.progress_every = max(1, int(progress_every))
        self.use_tqdm = bool(use_tqdm)
        self.aco_history: List[Dict[str, object]] = []
        self.stage_timings: Dict[str, float] = {}
        self.last_aco_paths: List[AntPath] = []

    def time_stage(self, name: str, fn, *args, **kwargs):
        t0 = time.time()
        self.timer.log("STAGE", f"{name} started")
        result = fn(*args, **kwargs)
        dt = time.time() - t0
        self.stage_timings[name] = dt
        self.timer.log("STAGE", f"{name} done", stage_time=fmt_time(dt))
        return result

    def run(self) -> List[float]:
        self.timer.log("START", "live CrossChannel DTES run started")
        self.time_stage("evaluate_grid", self.evaluate_grid)
        self.time_stage("compute_multiscale_features", self.compute_multiscale_features)
        self.time_stage("build_dyadic_tree", self.build_dyadic_tree)
        self.time_stage("aggregate_node_statistics", self.aggregate_node_statistics)
        self.time_stage("build_graph", self.build_graph)

        if hasattr(self, "compute_node_stability"):
            self.time_stage("compute_node_stability", self.compute_node_stability)

        self.time_stage("initialize_pheromones", self.initialize_pheromones)
        self.time_stage("run_aco", self.run_aco)

        candidate_nodes = self.time_stage("rank_candidate_nodes", self.rank_candidate_nodes)
        candidates = self.time_stage("refine_candidates", self.refine_candidates, candidate_nodes)
        candidates = self.merge_close_candidates(candidates)
        self.timer.log("DONE", "core DTES run done", candidates=len(candidates))
        return candidates

    def run_aco(self) -> None:
        if self.root_id is None:
            raise RuntimeError("Tree root is not built.")

        ants = [self.make_ant(k) for k in range(self.cfg.n_ants)]
        ema_iter: Optional[float] = None
        alpha = 0.2
        start = time.time()

        iterator = range(self.cfg.n_iterations)
        if tqdm is not None and self.use_tqdm:
            iterator = tqdm(iterator, desc="ACO", unit="iter")

        for it in iterator:
            t_iter = time.time()
            paths: List[AntPath] = []

            for ant in ants:
                path_nodes = self.sample_ant_path(self.root_id, ant)
                score = self.evaluate_path(path_nodes, ant.agent_type)
                paths.append(AntPath(node_ids=path_nodes, score=score, agent_type=ant.agent_type))
                self.update_ant_memory(ant, path_nodes)
                for nid in path_nodes:
                    self.nodes[nid].visit_count += 1

            self.last_aco_paths.extend(paths)
            self.evaporate_pheromones()
            self.reinforce_pheromones(paths)

            dt = time.time() - t_iter
            ema_iter = dt if ema_iter is None else alpha * dt + (1 - alpha) * ema_iter
            done = it + 1
            remain = self.cfg.n_iterations - done
            eta = remain * (ema_iter or dt)

            scores = np.array([p.score for p in paths], dtype=float)
            max_score = float(np.max(scores)) if scores.size else None
            mean_score = float(np.mean(scores)) if scores.size else None
            terminal_nodes = [p.node_ids[-1] for p in paths if p.node_ids]
            diversity = len(set(terminal_nodes)) / max(1, len(terminal_nodes))
            best_energy = self.current_best_energy()
            pheromone_mass = float(sum(self.pheromones.values())) if self.pheromones else 0.0

            rec = {
                "iteration": done,
                "n_iterations": self.cfg.n_iterations,
                "iter_time_s": dt,
                "ema_iter_time_s": ema_iter,
                "elapsed_s": time.time() - start,
                "eta_s": eta,
                "max_score": max_score,
                "mean_score": mean_score,
                "terminal_diversity": diversity,
                "best_energy": best_energy,
                "shared_pheromone_mass": pheromone_mass,
            }
            self.aco_history.append(rec)

            postfix = {
                "ETA": fmt_time(eta),
                "max": f"{max_score:.3g}" if max_score is not None else "na",
                "div": f"{diversity:.2f}",
                "E": f"{best_energy:.3g}" if best_energy is not None else "na",
            }

            if tqdm is not None and self.use_tqdm:
                iterator.set_postfix(postfix)
            elif done % self.progress_every == 0 or done == self.cfg.n_iterations:
                self.timer.log(
                    "ACO",
                    f"iter {done}/{self.cfg.n_iterations}",
                    iter_time=fmt_time(dt),
                    eta=fmt_time(eta),
                    max_score=postfix["max"],
                    diversity=postfix["div"],
                    best_E=postfix["E"],
                )

    def current_best_energy(self) -> Optional[float]:
        target_level = self.cfg.target_level or self.cfg.tree_depth
        ids = self.nodes_by_level.get(target_level, [])
        if not ids:
            return None
        return float(min(self.nodes[nid].energy for nid in ids))


def make_edge_anchors(t_min: float, t_max: float, padding: float, step: float) -> List[float]:
    anchors = []
    if padding > 0 and step > 0:
        n = int(math.ceil(padding / step))
        for i in range(n + 1):
            d = i * step
            anchors.append(t_min + d)
            anchors.append(t_max - d)
    return sorted(set(round(float(t), 12) for t in anchors if t_min <= t <= t_max))


def clean_candidates(candidates: List[float], t_min: float, t_max: float) -> List[float]:
    out = []
    for t in candidates:
        try:
            tt = float(t)
        except Exception:
            continue
        if t_min <= tt <= t_max:
            out.append(tt)
    return sorted(set(round(float(t), 12) for t in out))


def save_candidates(candidates: List[float], output: str, source: str) -> None:
    rows = [
        {"rank": i + 1, "t": float(t), "source": source}
        for i, t in enumerate(candidates)
    ]
    with open(output, "w", encoding="utf-8") as f:
        json.dump({"candidates": rows, "count": len(rows)}, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] {output} | count={len(rows)}")


def save_metrics(path: str, runner: LiveFractalDTESACOZeta, raw_candidates, core_candidates, anchors, edgeaware_candidates, args) -> None:
    data = {
        "config": vars(args),
        "raw_candidate_count": len(raw_candidates),
        "core_candidate_count": len(core_candidates),
        "edge_anchor_count": len(anchors),
        "edgeaware_candidate_count": len(edgeaware_candidates),
        "stage_timings_s": runner.stage_timings,
        "aco_history": runner.aco_history,
        "timer_records": runner.timer.records,
        "warning": None,
    }
    if len(core_candidates) == 0:
        data["warning"] = "No core DTES candidates found in requested interval. Edge-aware output would contain only anchors."
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] {path}")


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
    ap.add_argument("--progress_every", type=int, default=1)
    ap.add_argument("--no_tqdm", action="store_true")
    ap.add_argument("--output", default="dtes_candidates.json", help="Core DTES candidates only.")
    ap.add_argument("--edge_output", default="dtes_candidates_edgeaware.json", help="Core candidates + edge anchors.")
    ap.add_argument("--anchors_output", default="edge_anchors.json", help="Only boundary anchors.")
    ap.add_argument("--metrics", default="run_metrics.json")
    args = ap.parse_args()

    mp.mp.dps = args.dps
    timer = StageTimer()

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

    print("=== Live CrossChannel DTES runner v2 ===")
    print(f"range=[{args.t_min}, {args.t_max}] n0={args.n0} ants={args.ants} iters={args.iters}")

    runner = LiveFractalDTESACOZeta(
        cfg,
        timer=timer,
        progress_every=args.progress_every,
        use_tqdm=not args.no_tqdm,
    )

    raw = runner.run()
    core = clean_candidates(raw, args.t_min, args.t_max)
    anchors = make_edge_anchors(args.t_min, args.t_max, args.edge_padding, args.edge_step)
    edgeaware = sorted(set(core + anchors))

    print("\n=== Candidate summary ===")
    print(f"raw candidates: {len(raw)}")
    print(f"core interval candidates: {len(core)}")
    print(f"edge anchors: {len(anchors)}")
    print(f"edge-aware candidates: {len(edgeaware)}")

    if len(core) == 0:
        print("[WARN] No core DTES candidates in interval. Do NOT use edge-aware file for DTES quality analysis.")

    save_candidates(core, args.output, "run_crosschannel_live_v2_core_dtes")
    save_candidates(anchors, args.anchors_output, "run_crosschannel_live_v2_edge_anchor")
    save_candidates(edgeaware, args.edge_output, "run_crosschannel_live_v2_core_plus_edge_anchor")
    save_metrics(args.metrics, runner, raw, core, anchors, edgeaware, args)


if __name__ == "__main__":
    main()
