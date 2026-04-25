#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

COLORS = ("red", "blue", "green", "violet")


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
        return {
            "idx": idx,
            "t": float(item),
            "score": None,
            "abs_zeta": None,
            "kind": None,
        }
    if isinstance(item, dict):
        t = None
        for key in ("t", "time", "candidate", "zero", "value"):
            if key in item:
                t = float(item[key])
                break
        if t is None:
            raise ValueError(f"Cannot extract t from item keys={list(item.keys())}")
        score = None
        for key in (
            "score",
            "selection_score",
            "base_score",
            "dtes_score",
            "rank_score",
        ):
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
            "kind": item.get("kind"),
            "source": item.get("source"),
        }
    raise ValueError(f"Unsupported item type: {type(item)}")


def load_candidates(
    path: Path, t_min: Optional[float], t_max: Optional[float]
) -> List[Dict[str, Any]]:
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


def build_edges(
    nodes: List[Dict[str, Any]], k_neighbors: int, radius: Optional[float]
) -> List[Dict[str, Any]]:
    edges, seen = [], set()
    ts = np.array([x["t"] for x in nodes], dtype=float)
    scores = np.array([robust_score(x) for x in nodes], dtype=float)
    n = len(nodes)
    for i in range(n):
        dists = []
        for j in range(n):
            if i == j:
                continue
            dist = abs(float(ts[j] - ts[i]))
            if radius is not None and dist > radius:
                continue
            dists.append((dist, j))
        dists.sort()
        for dist, j in dists[:k_neighbors]:
            a, b = min(i, j), max(i, j)
            if (a, b) in seen:
                continue
            seen.add((a, b))
            edges.append(
                {
                    "u": a,
                    "v": b,
                    "t_u": float(ts[a]),
                    "t_v": float(ts[b]),
                    "distance": float(abs(ts[b] - ts[a])),
                    "score_diff": float(abs(scores[b] - scores[a])),
                    "score_mean": float(0.5 * (scores[a] + scores[b])),
                }
            )
    return edges


def color_edges(
    nodes: List[Dict[str, Any]],
    edges: List[Dict[str, Any]],
    boundary_fraction: float = 0.08,
) -> List[Dict[str, Any]]:
    if not nodes:
        return edges
    t_min, t_max = min(x["t"] for x in nodes), max(x["t"] for x in nodes)
    span = max(1e-12, t_max - t_min)
    distances = (
        np.array([e["distance"] for e in edges], dtype=float) if edges else np.array([])
    )
    score_diffs = (
        np.array([e["score_diff"] for e in edges], dtype=float)
        if edges
        else np.array([])
    )
    d_long = float(np.percentile(distances, 80)) if distances.size else 0.0
    s_similar = float(np.percentile(score_diffs, 35)) if score_diffs.size else 0.0
    for e in edges:
        mid = 0.5 * (e["t_u"] + e["t_v"])
        edge_dist = min(mid - t_min, t_max - mid)
        if edge_dist <= boundary_fraction * span:
            e["color"], e["color_reason"] = "green", "boundary-road"
        elif e["distance"] >= d_long:
            e["color"], e["color_reason"] = "violet", "bridge/gap-road"
        elif e["score_diff"] <= s_similar:
            e["color"], e["color_reason"] = "red", "energy-similar"
        else:
            e["color"], e["color_reason"] = "blue", "local-road"
    return edges


def adjacency(
    n: int, edges: List[Dict[str, Any]], color: Optional[str] = None
) -> List[Set[int]]:
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
    out, seen = [], set()
    for s in order:
        clique = [s]
        cand = set(adj[s])
        while cand:
            v = max(cand, key=lambda x: len(adj[x]))
            if all(v in adj[u] for u in clique):
                clique.append(v)
                cand &= adj[v]
            else:
                cand.remove(v)
        key = tuple(sorted(clique))
        if key not in seen:
            seen.add(key)
            out.append(list(key))
    return sorted(out, key=lambda c: (-len(c), c))


def bron_kerbosch(
    adj: List[Set[int]], max_nodes_for_exact: int = 80
) -> List[List[int]]:
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
        actual = 0
        ns = list(nbrs)
        for a in range(len(ns)):
            for b in range(a + 1, len(ns)):
                if ns[b] in adj[ns[a]]:
                    actual += 1
        vals.append(actual / (d * (d - 1) / 2))
    return float(np.mean(vals)) if vals else 0.0


def clique_rows(
    cliques: List[List[int]], nodes: List[Dict[str, Any]], color: str
) -> List[Dict[str, Any]]:
    rows = []
    for k, c in enumerate(cliques):
        ts = [nodes[i]["t"] for i in c]
        rows.append(
            {
                "rank": k + 1,
                "color": color,
                "size": len(c),
                "node_indices": " ".join(map(str, c)),
                "t_min": min(ts) if ts else None,
                "t_max": max(ts) if ts else None,
                "diameter_t": (max(ts) - min(ts)) if len(ts) > 1 else 0.0,
            }
        )
    return rows


def shuffled_baseline(nodes, edges, n_perm, seed, max_nodes_for_exact):
    rng = random.Random(seed)
    colors = [e["color"] for e in edges]
    vals = []
    for _ in range(n_perm):
        sh = list(colors)
        rng.shuffle(sh)
        e2 = []
        for e, c in zip(edges, sh):
            ee = dict(e)
            ee["color"] = c
            e2.append(ee)
        m = 0
        for color in COLORS:
            cl = bron_kerbosch(adjacency(len(nodes), e2, color), max_nodes_for_exact)
            m = max(m, len(cl[0]) if cl else 0)
        vals.append(m)
    arr = np.array(vals, dtype=float)
    return {
        "n_perm": n_perm,
        "mean_max_mono_clique": float(np.mean(arr)) if len(arr) else None,
        "std_max_mono_clique": float(np.std(arr)) if len(arr) else None,
        "p95_max_mono_clique": float(np.percentile(arr, 95)) if len(arr) else None,
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
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


def plot_graph(path: Path, nodes, edges):
    if plt is None:
        return
    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(111)
    y = {"red": 0.4, "blue": 0.2, "green": -0.2, "violet": -0.4}
    col = {"red": "red", "blue": "blue", "green": "green", "violet": "purple"}
    for e in edges:
        c = e.get("color", "blue")
        ax.plot(
            [e["t_u"], e["t_v"]], [y[c], y[c]], alpha=0.35, linewidth=0.9, color=col[c]
        )
    ax.scatter([n["t"] for n in nodes], [0] * len(nodes), s=14)
    ax.set_xlabel("t")
    ax.set_title("Ramsey colored candidate graph")
    ax.grid(True, alpha=0.25)
    ax.set_yticks([-0.4, -0.2, 0, 0.2, 0.4])
    ax.set_yticklabels(["violet", "green", "nodes", "blue", "red"])
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
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_summary(path, nodes, edges, color_stats, global_stats, baseline):
    lines = [
        "# Ramsey graph analysis",
        "",
        f"- nodes: `{len(nodes)}`",
        f"- edges: `{len(edges)}`",
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
        lines.append(
            f"| {c} | {s['edge_count']} | {s['max_clique_size']} | {s['triangles']} | {s['clustering_coefficient']:.4f} |"
        )
    if baseline:
        lines += [
            "",
            "## Shuffled coloring baseline",
            "",
            f"- permutations: `{baseline['n_perm']}`",
            f"- mean max mono clique: `{baseline['mean_max_mono_clique']}`",
            f"- std max mono clique: `{baseline['std_max_mono_clique']}`",
            f"- p95 max mono clique: `{baseline['p95_max_mono_clique']}`",
        ]
    lines += [
        "",
        "## Interpretation",
        "",
        "- Red clique: energetically coherent region.",
        "- Blue clique: local-road coherence.",
        "- Green clique: boundary concentration.",
        "- Violet clique: bridge/gap structure.",
        "- If real max monochromatic clique exceeds shuffled baseline, coloring is non-random.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(
        description="Ramsey graph analyzer for DTES candidates."
    )
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--t_min", type=float, default=None)
    ap.add_argument("--t_max", type=float, default=None)
    ap.add_argument("--k_neighbors", type=int, default=6)
    ap.add_argument("--radius", type=float, default=None)
    ap.add_argument("--boundary_fraction", type=float, default=0.08)
    ap.add_argument("--n_perm", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max_nodes_for_exact", type=int, default=80)
    ap.add_argument("--out", default="ramsey")
    ap.add_argument("--no_plots", action="store_true")
    args = ap.parse_args()

    nodes = load_candidates(Path(args.candidates), args.t_min, args.t_max)
    edges = color_edges(
        nodes, build_edges(nodes, args.k_neighbors, args.radius), args.boundary_fraction
    )

    global_cliques = bron_kerbosch(
        adjacency(len(nodes), edges, None), args.max_nodes_for_exact
    )
    global_max = len(global_cliques[0]) if global_cliques else 0
    color_stats, all_cliques = {}, []
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
    baseline = (
        shuffled_baseline(
            nodes, edges, args.n_perm, args.seed, args.max_nodes_for_exact
        )
        if args.n_perm > 0 and edges
        else None
    )

    prefix = Path(args.out)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "config": vars(args),
        "global_stats": global_stats,
        "color_stats": color_stats,
        "baseline": baseline,
    }
    (prefix.with_name(prefix.name + "_report.json")).write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_csv(prefix.with_name(prefix.name + "_edges.csv"), edges)
    write_csv(prefix.with_name(prefix.name + "_cliques.csv"), all_cliques)
    write_csv(
        prefix.with_name(prefix.name + "_color_counts.csv"),
        [
            {
                "color": c,
                **{k: v for k, v in color_stats[c].items() if k != "top_cliques"},
            }
            for c in COLORS
        ],
    )
    write_summary(
        prefix.with_name(prefix.name + "_summary.md"),
        nodes,
        edges,
        color_stats,
        global_stats,
        baseline,
    )
    if not args.no_plots:
        plot_graph(prefix.with_name(prefix.name + "_graph.png"), nodes, edges)
        plot_cliques(prefix.with_name(prefix.name + "_clique_sizes.png"), color_stats)

    print("=== Ramsey graph analysis ===")
    print(f"nodes: {len(nodes)}")
    print(f"edges: {len(edges)}")
    print(f"max clique size: {global_max}")
    print(f"max mono clique size: {max_mono}")
    print(f"ramsey score: {global_stats['ramsey_score']:.6f}")
    for c in COLORS:
        s = color_stats[c]
        print(
            f"{c}: edges={s['edge_count']} max_clique={s['max_clique_size']} triangles={s['triangles']}"
        )
    print(f"summary: {prefix.with_name(prefix.name + '_summary.md')}")


if __name__ == "__main__":
    main()
