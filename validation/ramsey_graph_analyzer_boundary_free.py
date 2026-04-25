#!/usr/bin/env python3
"""
ramsey_graph_analyzer_boundary_free.py

Boundary-free / cyclic Ramsey analyzer for DTES candidate graphs.

Goal:
    Remove finite-interval boundary artifacts.

Difference from standard ramsey_graph_analyzer.py:
    - t-axis can be treated as a circle / periodic domain.
    - distance is cyclic:
          d(t_i,t_j) = min(|t_i-t_j|, L - |t_i-t_j|)
      where L = t_max - t_min.
    - no boundary color is used by default.
    - edge colors:
          red    = energy/score-similar
          blue   = short local road
          violet = long bridge/gap road
      optionally green can be disabled entirely.

Usage:
    python3 validation/ramsey_graph_analyzer_boundary_free.py \
      --candidates runs/run_20260425_112046/dtes_candidates.json \
      --t_min 150 \
      --t_max 450 \
      --k_neighbors 8 \
      --n_perm 500 \
      --out runs/run_20260425_112046/ramsey_boundary_free

Outputs:
    *_summary.md
    *_report.json
    *_edges.csv
    *_cliques.csv
    *_graph.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


COLORS = ("red", "blue", "violet")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_sequence(data: Any) -> List[Any]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("candidates", "zeros", "dtes_candidates", "results"):
            if key in data and isinstance(data[key], list):
                return data[key]
        for val in data.values():
            if isinstance(val, list):
                return val
    raise ValueError("Could not find candidates list in JSON")


def extract_candidate(item: Any, idx: int) -> Dict[str, Any]:
    if isinstance(item, (int, float)):
        return {"idx": idx, "t": float(item), "score": None, "abs_zeta": None}
    if isinstance(item, dict):
        t = None
        for key in ("t", "time", "candidate", "zero", "value"):
            if key in item:
                t = float(item[key])
                break
        if t is None:
            raise ValueError(f"Cannot extract t from item keys={list(item.keys())}")

        score = None
        for key in ("score", "selection_score", "base_score", "dtes_score", "rank_score"):
            if key in item and item[key] is not None:
                try:
                    score = float(item[key])
                    break
                except Exception:
                    pass

        abs_zeta = None
        for key in ("abs_zeta", "abs", "residual"):
            if key in item and item[key] is not None:
                try:
                    abs_zeta = float(item[key])
                    break
                except Exception:
                    pass

        return {
            "idx": idx,
            "t": t,
            "score": score,
            "abs_zeta": abs_zeta,
            "source": item.get("source"),
        }
    raise ValueError(f"Unsupported item type: {type(item)}")


def load_candidates(path: Path, t_min: Optional[float], t_max: Optional[float]) -> List[Dict[str, Any]]:
    rows = []
    for item in extract_sequence(load_json(path)):
        c = extract_candidate(item, len(rows))
        if t_min is not None and c["t"] < t_min:
            continue
        if t_max is not None and c["t"] > t_max:
            continue
        rows.append(c)
    rows.sort(key=lambda x: x["t"])
    for i, r in enumerate(rows):
        r["idx"] = i
    return rows


def robust_score(c: Dict[str, Any]) -> float:
    if c.get("score") is not None:
        return float(c["score"])
    if c.get("abs_zeta") is not None:
        return -math.log(float(c["abs_zeta"]) + 1e-14)
    return 0.0


def cyclic_distance(a: float, b: float, t_min: float, t_max: float, cyclic: bool) -> float:
    d = abs(b - a)
    if not cyclic:
        return d
    L = max(1e-12, t_max - t_min)
    return min(d, L - d)


def build_edges(
    nodes: List[Dict[str, Any]],
    t_min: float,
    t_max: float,
    k_neighbors: int,
    radius: Optional[float],
    cyclic: bool,
) -> List[Dict[str, Any]]:
    edges = []
    seen = set()
    ts = np.array([x["t"] for x in nodes], dtype=float)
    scores = np.array([robust_score(x) for x in nodes], dtype=float)
    n = len(nodes)

    for i in range(n):
        dists = []
        for j in range(n):
            if i == j:
                continue
            dist = cyclic_distance(float(ts[i]), float(ts[j]), t_min, t_max, cyclic)
            if radius is not None and dist > radius:
                continue
            dists.append((dist, j))
        dists.sort()

        for dist, j in dists[:k_neighbors]:
            a, b = min(i, j), max(i, j)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            raw_dist = abs(float(ts[b] - ts[a]))
            edges.append({
                "u": a,
                "v": b,
                "t_u": float(ts[a]),
                "t_v": float(ts[b]),
                "distance": float(dist),
                "raw_distance": float(raw_dist),
                "wrap_edge": bool(cyclic and dist < raw_dist),
                "score_diff": float(abs(scores[b] - scores[a])),
                "score_mean": float(0.5 * (scores[a] + scores[b])),
            })

    return edges


def color_edges(edges: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not edges:
        return edges

    distances = np.array([e["distance"] for e in edges], dtype=float)
    score_diffs = np.array([e["score_diff"] for e in edges], dtype=float)

    d_short = float(np.percentile(distances, 35))
    d_long = float(np.percentile(distances, 80))
    s_similar = float(np.percentile(score_diffs, 35))

    for e in edges:
        if e["distance"] >= d_long:
            e["color"] = "violet"
            e["color_reason"] = "cyclic-bridge/gap-road"
        elif e["score_diff"] <= s_similar:
            e["color"] = "red"
            e["color_reason"] = "energy-similar"
        else:
            e["color"] = "blue"
            e["color_reason"] = "local-road"
    return edges


def adjacency(n: int, edges: List[Dict[str, Any]], color: Optional[str] = None) -> List[Set[int]]:
    adj = [set() for _ in range(n)]
    for e in edges:
        if color is not None and e.get("color") != color:
            continue
        u, v = int(e["u"]), int(e["v"])
        adj[u].add(v)
        adj[v].add(u)
    return adj


def greedy_cliques(adj: List[Set[int]]) -> List[List[int]]:
    order = sorted(range(len(adj)), key=lambda i: len(adj[i]), reverse=True)
    out = []
    seen = set()
    for start in order:
        clique = [start]
        candidates = set(adj[start])
        while candidates:
            v = max(candidates, key=lambda x: len(adj[x]))
            if all(v in adj[u] for u in clique):
                clique.append(v)
                candidates &= adj[v]
            else:
                candidates.remove(v)
        key = tuple(sorted(clique))
        if key not in seen:
            seen.add(key)
            out.append(list(key))
    return sorted(out, key=lambda c: (-len(c), c))


def bron_kerbosch(adj: List[Set[int]], max_nodes_for_exact: int = 80) -> List[List[int]]:
    if len(adj) > max_nodes_for_exact:
        return greedy_cliques(adj)

    cliques = []

    def bk(r: Set[int], p: Set[int], x: Set[int]):
        if not p and not x:
            cliques.append(sorted(r))
            return

        union = p | x
        pivot = max(union, key=lambda u: len(p & adj[u])) if union else None
        candidates = p - (adj[pivot] if pivot is not None else set())

        for v in list(candidates):
            bk(r | {v}, p & adj[v], x & adj[v])
            p.remove(v)
            x.add(v)

    bk(set(), set(range(len(adj))), set())
    return sorted(cliques, key=lambda c: (-len(c), c))


def triangle_count(adj: List[Set[int]]) -> int:
    count = 0
    for i in range(len(adj)):
        for j in adj[i]:
            if j <= i:
                continue
            count += sum(1 for k in (adj[i] & adj[j]) if k > j)
    return count


def clustering(adj: List[Set[int]]) -> float:
    vals = []
    for i, nbrs in enumerate(adj):
        d = len(nbrs)
        if d < 2:
            continue
        ns = list(nbrs)
        actual = 0
        for a in range(len(ns)):
            for b in range(a + 1, len(ns)):
                if ns[b] in adj[ns[a]]:
                    actual += 1
        vals.append(actual / (d * (d - 1) / 2))
    return float(np.mean(vals)) if vals else 0.0


def clique_rows(cliques: List[List[int]], nodes: List[Dict[str, Any]], color: str) -> List[Dict[str, Any]]:
    rows = []
    for k, c in enumerate(cliques):
        ts = [nodes[i]["t"] for i in c]
        rows.append({
            "rank": k + 1,
            "color": color,
            "size": len(c),
            "node_indices": " ".join(map(str, c)),
            "t_min": min(ts) if ts else None,
            "t_max": max(ts) if ts else None,
            "diameter_t": (max(ts) - min(ts)) if len(ts) > 1 else 0.0,
            "t_values": " ".join(f"{t:.12f}" for t in ts),
        })
    return rows


def shuffled_baseline(nodes, edges, n_perm, seed, max_nodes_for_exact):
    rng = random.Random(seed)
    colors = [e["color"] for e in edges]
    vals = []
    color_vals = {c: [] for c in COLORS}

    for _ in range(n_perm):
        sh = list(colors)
        rng.shuffle(sh)
        e2 = []
        for e, c in zip(edges, sh):
            ee = dict(e)
            ee["color"] = c
            e2.append(ee)

        max_any = 0
        for c in COLORS:
            cl = bron_kerbosch(adjacency(len(nodes), e2, c), max_nodes_for_exact)
            m = len(cl[0]) if cl else 0
            color_vals[c].append(m)
            max_any = max(max_any, m)
        vals.append(max_any)

    arr = np.array(vals, dtype=float)
    return {
        "n_perm": n_perm,
        "mean_max_mono_clique": float(np.mean(arr)) if len(arr) else None,
        "std_max_mono_clique": float(np.std(arr)) if len(arr) else None,
        "p95_max_mono_clique": float(np.percentile(arr, 95)) if len(arr) else None,
        "color_p95": {c: float(np.percentile(np.array(v, dtype=float), 95)) if v else None for c, v in color_vals.items()},
    }


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    keys = []
    for row in rows:
        for k, v in row.items():
            if isinstance(v, (list, dict, set)):
                continue
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k) for k in keys})


def plot_graph(path: Path, nodes, edges, cyclic: bool):
    if plt is None:
        return

    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(111)
    y = {"red": 0.3, "blue": 0.0, "violet": -0.3}
    color_map = {"red": "red", "blue": "blue", "violet": "purple"}

    for e in edges:
        c = e.get("color", "blue")
        alpha = 0.45 if not e.get("wrap_edge") else 0.75
        lw = 1.2 if e.get("wrap_edge") else 0.8
        ax.plot([e["t_u"], e["t_v"]], [y[c], y[c]], color=color_map[c], alpha=alpha, linewidth=lw)

    ax.scatter([n["t"] for n in nodes], [0.55] * len(nodes), s=12, label="candidates")
    ax.set_xlabel("t")
    ax.set_yticks([-0.3, 0.0, 0.3, 0.55])
    ax.set_yticklabels(["violet", "blue", "red", "nodes"])
    ax.set_title("Boundary-free Ramsey graph" + (" (cyclic)" if cyclic else ""))
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_cliques(path: Path, color_stats):
    if plt is None:
        return
    labels = list(COLORS)
    vals = [color_stats[c]["max_clique_size"] for c in labels]
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.bar(labels, vals)
    ax.set_ylabel("max monochromatic clique size")
    ax.set_title("Boundary-free max clique size by color")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_summary(path, nodes, edges, color_stats, global_stats, baseline, args):
    lines = [
        "# Boundary-free Ramsey graph analysis",
        "",
        f"- nodes: `{len(nodes)}`",
        f"- edges: `{len(edges)}`",
        f"- cyclic topology: `{args.cyclic}`",
        f"- max clique size: `{global_stats['max_clique_size']}`",
        f"- max monochromatic clique size: `{global_stats['max_mono_clique_size']}`",
        f"- Ramsey score: `{global_stats['ramsey_score']:.6f}`",
        "",
        "## Color stats",
        "",
        "| color | edges | max clique | triangles | clustering |",
        "|---|---:|---:|---:|---:|",
    ]

    for c in COLORS:
        s = color_stats[c]
        lines.append(f"| {c} | {s['edge_count']} | {s['max_clique_size']} | {s['triangles']} | {s['clustering_coefficient']:.4f} |")

    if baseline:
        lines += [
            "",
            "## Shuffled coloring baseline",
            "",
            f"- permutations: `{baseline['n_perm']}`",
            f"- mean max mono clique: `{baseline['mean_max_mono_clique']}`",
            f"- std max mono clique: `{baseline['std_max_mono_clique']}`",
            f"- p95 max mono clique: `{baseline['p95_max_mono_clique']}`",
            f"- color p95: `{baseline['color_p95']}`",
        ]

    lines += [
        "",
        "## Interpretation",
        "",
        "This analysis removes explicit boundary coloring. If the previous green K9 disappears,",
        "the earlier Ramsey signal was mostly finite-window boundary structure. If a large",
        "red/blue/violet clique remains above shuffled baseline, it is stronger evidence for",
        "intrinsic DTES/Ramsey organization in the bulk.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Boundary-free Ramsey graph analyzer.")
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--t_min", type=float, required=True)
    ap.add_argument("--t_max", type=float, required=True)
    ap.add_argument("--k_neighbors", type=int, default=8)
    ap.add_argument("--radius", type=float, default=None)
    ap.add_argument("--n_perm", type=int, default=500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max_nodes_for_exact", type=int, default=80)
    ap.add_argument("--cyclic", action="store_true", help="Use periodic/cyclic distance.")
    ap.add_argument("--out", default="ramsey_boundary_free")
    ap.add_argument("--no_plots", action="store_true")
    args = ap.parse_args()

    nodes = load_candidates(Path(args.candidates), args.t_min, args.t_max)
    edges = build_edges(nodes, args.t_min, args.t_max, args.k_neighbors, args.radius, args.cyclic)
    edges = color_edges(edges)

    global_cliques = bron_kerbosch(adjacency(len(nodes), edges, None), args.max_nodes_for_exact)
    global_max = len(global_cliques[0]) if global_cliques else 0

    color_stats = {}
    all_cliques = []
    for c in COLORS:
        adj = adjacency(len(nodes), edges, c)
        cliques = bron_kerbosch(adj, args.max_nodes_for_exact)
        rows = clique_rows(cliques[:20], nodes, c)
        color_stats[c] = {
            "edge_count": sum(1 for e in edges if e.get("color") == c),
            "max_clique_size": len(cliques[0]) if cliques else 0,
            "triangles": triangle_count(adj),
            "clustering_coefficient": clustering(adj),
            "top_cliques": rows,
        }
        all_cliques.extend(rows)

    max_mono = max(color_stats[c]["max_clique_size"] for c in COLORS)
    global_stats = {
        "max_clique_size": global_max,
        "max_mono_clique_size": max_mono,
        "ramsey_score": max_mono / max(1, len(nodes)),
        "n_nodes": len(nodes),
        "n_edges": len(edges),
    }

    baseline = shuffled_baseline(nodes, edges, args.n_perm, args.seed, args.max_nodes_for_exact) if args.n_perm > 0 and edges else None

    prefix = Path(args.out)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "config": vars(args),
        "global_stats": global_stats,
        "color_stats": color_stats,
        "baseline": baseline,
    }
    (prefix.with_name(prefix.name + "_report.json")).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(prefix.with_name(prefix.name + "_edges.csv"), edges)
    write_csv(prefix.with_name(prefix.name + "_cliques.csv"), all_cliques)
    write_csv(prefix.with_name(prefix.name + "_color_counts.csv"), [{"color": c, **{k: v for k, v in color_stats[c].items() if k != "top_cliques"}} for c in COLORS])
    write_summary(prefix.with_name(prefix.name + "_summary.md"), nodes, edges, color_stats, global_stats, baseline, args)

    if not args.no_plots:
        plot_graph(prefix.with_name(prefix.name + "_graph.png"), nodes, edges, args.cyclic)
        plot_cliques(prefix.with_name(prefix.name + "_clique_sizes.png"), color_stats)

    print("=== Boundary-free Ramsey graph analysis ===")
    print(f"nodes: {len(nodes)}")
    print(f"edges: {len(edges)}")
    print(f"cyclic: {args.cyclic}")
    print(f"max clique size: {global_max}")
    print(f"max mono clique size: {max_mono}")
    print(f"ramsey score: {global_stats['ramsey_score']:.6f}")
    for c in COLORS:
        s = color_stats[c]
        print(f"{c}: edges={s['edge_count']} max_clique={s['max_clique_size']} triangles={s['triangles']}")
    print(f"summary: {prefix.with_name(prefix.name + '_summary.md')}")


if __name__ == "__main__":
    main()
