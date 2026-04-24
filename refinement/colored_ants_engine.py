#!/usr/bin/env python3
"""
colored_ants_engine.py

Colored grouped-ant DTES candidate selector.

This is a lightweight, dependency-minimal module that implements:
- grouped ants;
- ant colors / policies;
- color-specific pheromone maps;
- shared dynamic roads;
- gap-seeking behavior;
- live ETA.

It can be used standalone with a precomputed candidate pool JSON, or imported
from another DTES runner.

Color semantics:
    red     = exploitation / low-energy attractors
    blue    = exploration / low-visit regions
    green   = boundary-aware ants
    violet  = bridge / gap-closing ants

Ant-RH:
    Optional refinement after a coarse candidate pool exists; pairs with
    ``gap_detector.py`` / ``dynamic_roads.py`` for coverage-aware selection.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional


COLORS = ("red", "blue", "green", "violet")


@dataclass
class ColoredAnt:
    ant_id: int
    color: str
    memory: List[int] = field(default_factory=list)


@dataclass
class RoadEdge:
    i: int
    j: int
    distance: float
    barrier: float = 0.0


@dataclass
class ColoredAntConfig:
    groups: int = 4
    ants_per_group: int = 24
    iterations_per_group: int = 20
    max_steps: int = 12
    seed: int = 7
    alpha_pheromone: float = 1.0
    beta_barrier: float = 1.0
    gamma_explore: float = 1.0
    delta_color: float = 1.0
    evaporation: float = 0.08
    deposit_scale: float = 1.0
    k_neighbors: int = 8
    target_count: int = 180


def fmt_time(sec: float) -> str:
    sec = max(0.0, sec)
    if sec < 60:
        return f"{sec:.1f}s"
    if sec < 3600:
        return f"{sec/60:.1f}m"
    return f"{sec/3600:.2f}h"


class ColoredGroupedAntEngine:
    def __init__(self, points: List[dict], cfg: ColoredAntConfig):
        self.points = sorted(points, key=lambda x: float(x["t"]))
        self.cfg = cfg
        self.rng = random.Random(cfg.seed)
        self.edges: Dict[Tuple[int, int], RoadEdge] = {}
        self.neighbors: Dict[int, List[int]] = {}
        self.shared_pheromone: Dict[Tuple[int, int], float] = {}
        self.color_pheromone: Dict[str, Dict[Tuple[int, int], float]] = {c: {} for c in COLORS}
        self.visit_count: List[int] = [0 for _ in self.points]
        self.selected: List[int] = []
        self.history: List[dict] = []
        self._build_roads()

    def _point_score(self, idx: int) -> float:
        p = self.points[idx]
        score = float(p.get("score", p.get("base_score", 0.0)))
        abs_zeta = p.get("abs_zeta")
        if abs_zeta is not None:
            try:
                score += -0.05 * math.log(float(abs_zeta) + 1e-14)
            except Exception:
                pass
        return score

    def _build_roads(self) -> None:
        n = len(self.points)
        for i in range(n):
            dists = []
            ti = float(self.points[i]["t"])
            for j in range(n):
                if i == j:
                    continue
                tj = float(self.points[j]["t"])
                dists.append((abs(tj - ti), j))
            dists.sort()
            self.neighbors[i] = [j for _, j in dists[: self.cfg.k_neighbors]]
            for dist, j in dists[: self.cfg.k_neighbors]:
                e = (i, j)
                barrier = dist
                self.edges[e] = RoadEdge(i=i, j=j, distance=dist, barrier=barrier)
                self.shared_pheromone[e] = 1.0
                for c in COLORS:
                    self.color_pheromone[c][e] = 1.0

    def color_bonus(self, color: str, current: int, nxt: int) -> float:
        p = self.points[nxt]
        t = float(p["t"])
        t_min = float(self.points[0]["t"])
        t_max = float(self.points[-1]["t"])
        span = max(1e-9, t_max - t_min)

        if color == "red":
            return self._point_score(nxt)

        if color == "blue":
            return 1.0 / math.sqrt(1.0 + self.visit_count[nxt])

        if color == "green":
            edge_dist = min(abs(t - t_min), abs(t_max - t))
            return math.exp(-edge_dist / max(1e-9, 0.04 * span))

        if color == "violet":
            return self.gap_bonus(t)

        return 0.0

    def gap_bonus(self, t: float) -> float:
        if len(self.selected) < 2:
            return 1.0
        ts = sorted(float(self.points[i]["t"]) for i in self.selected)
        # Bonus is high in largest uncovered intervals.
        best = 0.0
        prev = float(self.points[0]["t"])
        for x in ts:
            mid = 0.5 * (prev + x)
            gap = x - prev
            best = max(best, gap / (1.0 + abs(t - mid)))
            prev = x
        last = float(self.points[-1]["t"])
        mid = 0.5 * (prev + last)
        gap = last - prev
        best = max(best, gap / (1.0 + abs(t - mid)))
        return best

    def road_value(self, color: str, current: int, nxt: int) -> float:
        e = (current, nxt)
        edge = self.edges[e]
        shared = self.shared_pheromone.get(e, 1.0)
        own = self.color_pheromone[color].get(e, 1.0)
        tau = 0.55 * shared + 0.45 * own
        explore = 1.0 / math.sqrt(1.0 + self.visit_count[nxt])
        cb = self.color_bonus(color, current, nxt)
        return (
            self.cfg.alpha_pheromone * math.log(max(tau, 1e-12))
            - self.cfg.beta_barrier * edge.barrier
            + self.cfg.gamma_explore * explore
            + self.cfg.delta_color * cb
        )

    def choose_next(self, ant: ColoredAnt, current: int) -> int:
        nbrs = self.neighbors.get(current, [])
        if not nbrs:
            return current
        vals = [self.road_value(ant.color, current, j) for j in nbrs]
        m = max(vals)
        weights = [math.exp(v - m) for v in vals]
        s = sum(weights)
        r = self.rng.random() * s
        acc = 0.0
        for j, w in zip(nbrs, weights):
            acc += w
            if acc >= r:
                return j
        return nbrs[-1]

    def path_score(self, path: List[int], color: str) -> float:
        if not path:
            return 0.0
        scores = [self._point_score(i) for i in path]
        novelty = sum(1.0 / math.sqrt(1.0 + self.visit_count[i]) for i in path)
        color_term = sum(self.color_bonus(color, path[max(0, k - 1)], i) for k, i in enumerate(path))
        return sum(scores) / len(scores) + 0.25 * novelty + 0.15 * color_term

    def reinforce(self, path: List[int], score: float, color: str) -> None:
        dep = self.cfg.deposit_scale * max(0.0, score)
        for a, b in zip(path[:-1], path[1:]):
            e = (a, b)
            if e in self.shared_pheromone:
                self.shared_pheromone[e] += dep
                self.color_pheromone[color][e] += 1.25 * dep
                # Weak cross-color reinforcement.
                for c in COLORS:
                    if c != color and e in self.color_pheromone[c]:
                        self.color_pheromone[c][e] += 0.05 * dep

    def evaporate(self) -> None:
        keep = 1.0 - self.cfg.evaporation
        for d in [self.shared_pheromone, *self.color_pheromone.values()]:
            for k in list(d.keys()):
                d[k] = max(1e-6, d[k] * keep)

    def run_group(self, color: str, group_idx: int) -> None:
        ants = [ColoredAnt(ant_id=group_idx * self.cfg.ants_per_group + i, color=color) for i in range(self.cfg.ants_per_group)]
        start_time = time.time()
        ema = None

        for it in range(self.cfg.iterations_per_group):
            iter_start = time.time()
            paths = []
            for ant in ants:
                # Start: violet/blue prefer sparse regions, green boundaries, red high score.
                start = self.pick_start(color)
                path = [start]
                cur = start
                for _ in range(self.cfg.max_steps):
                    nxt = self.choose_next(ant, cur)
                    if nxt == cur:
                        break
                    path.append(nxt)
                    cur = nxt
                score = self.path_score(path, color)
                paths.append((path, score))
                for idx in path:
                    self.visit_count[idx] += 1
                best = max(path, key=lambda i: self._point_score(i))
                if best not in self.selected:
                    self.selected.append(best)

            self.evaporate()
            for path, score in paths:
                self.reinforce(path, score, color)

            dt = time.time() - iter_start
            ema = dt if ema is None else 0.2 * dt + 0.8 * ema
            eta = (self.cfg.iterations_per_group - it - 1) * (ema or dt)
            self.history.append({
                "group": group_idx,
                "color": color,
                "iteration": it + 1,
                "selected_count": len(set(self.selected)),
                "iter_time_s": dt,
                "eta_s": eta,
                "mean_path_score": sum(s for _, s in paths) / max(1, len(paths)),
            })
            print(
                f"[GROUP {group_idx}:{color}] {it+1}/{self.cfg.iterations_per_group} "
                f"| selected={len(set(self.selected))} | ETA={fmt_time(eta)}",
                flush=True,
            )

    def pick_start(self, color: str) -> int:
        n = len(self.points)
        if color == "red":
            return max(range(n), key=self._point_score)
        if color == "green":
            return self.rng.choice([0, max(0, n - 1)])
        if color == "blue":
            return min(range(n), key=lambda i: self.visit_count[i])
        if color == "violet":
            if not self.selected:
                return self.rng.randrange(n)
            return max(range(n), key=lambda i: self.gap_bonus(float(self.points[i]["t"])))
        return self.rng.randrange(n)

    def run(self) -> List[dict]:
        colors = list(COLORS)
        for g in range(self.cfg.groups):
            color = colors[g % len(colors)]
            self.run_group(color, g)
            if len(set(self.selected)) >= self.cfg.target_count:
                break
        unique = sorted(set(self.selected), key=lambda i: float(self.points[i]["t"]))
        return [self.points[i] | {"selected_by_colored_ants": True} for i in unique[: self.cfg.target_count]]


def load_points(path: Path) -> List[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    seq = data.get("candidates", data if isinstance(data, list) else [])
    points = []
    for i, x in enumerate(seq):
        if isinstance(x, dict):
            t = float(x["t"])
            points.append(dict(x, pool_index=i, t=t))
        else:
            points.append({"pool_index": i, "t": float(x), "score": 0.0})
    return points


def save_candidates(path: Path, rows: List[dict]) -> None:
    out = []
    for i, r in enumerate(sorted(rows, key=lambda x: float(x["t"])), start=1):
        out.append({
            "rank": i,
            "t": float(r["t"]),
            "score": float(r.get("score", r.get("selection_score", 0.0))),
            "color_selected": True,
            "source": "colored_group_ants",
        })
    path.write_text(json.dumps({"candidates": out, "count": len(out)}, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, help="Candidate pool JSON, e.g. dtes_candidates_explore.json")
    ap.add_argument("--output", default="colored_group_candidates.json")
    ap.add_argument("--metrics", default="colored_group_metrics.json")
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--ants_per_group", type=int, default=24)
    ap.add_argument("--iterations_per_group", type=int, default=20)
    ap.add_argument("--max_steps", type=int, default=12)
    ap.add_argument("--target_count", type=int, default=180)
    args = ap.parse_args()

    cfg = ColoredAntConfig(
        groups=args.groups,
        ants_per_group=args.ants_per_group,
        iterations_per_group=args.iterations_per_group,
        max_steps=args.max_steps,
        target_count=args.target_count,
    )
    points = load_points(Path(args.pool))
    engine = ColoredGroupedAntEngine(points, cfg)
    selected = engine.run()
    save_candidates(Path(args.output), selected)
    Path(args.metrics).write_text(json.dumps({
        "config": vars(args),
        "history": engine.history,
        "selected_count": len(selected),
    }, indent=2), encoding="utf-8")
    print(f"[SAVE] {args.output} | count={len(selected)}")
    print(f"[SAVE] {args.metrics}")


if __name__ == "__main__":
    main()
