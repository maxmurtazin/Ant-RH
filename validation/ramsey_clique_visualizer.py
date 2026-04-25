#!/usr/bin/env python3
from __future__ import annotations

import argparse, csv, json
from pathlib import Path
from typing import Any, Dict, List

try:
    import matplotlib.pyplot as plt
except Exception as exc:
    plt = None
    _PLOT_ERROR = exc
else:
    _PLOT_ERROR = None

COLOR_MAP = {
    "red": "red",
    "blue": "blue",
    "green": "green",
    "violet": "purple",
}


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
    raise ValueError("Could not find candidate list")


def load_candidates(path: Path) -> List[Dict[str, Any]]:
    seq = extract_sequence(load_json(path))
    rows = []
    for item in seq:
        if isinstance(item, dict):
            t = None
            for key in ("t", "time", "candidate", "zero", "value"):
                if key in item:
                    t = float(item[key])
                    break
            if t is None:
                continue
            rows.append({
                "idx": len(rows),
                "t": t,
                "score": item.get("score"),
                "abs_zeta": item.get("abs_zeta"),
                "kind": item.get("kind"),
                "source": item.get("source"),
            })
        else:
            rows.append({"idx": len(rows), "t": float(item), "score": None, "abs_zeta": None})
    rows.sort(key=lambda x: x["t"])
    for i, r in enumerate(rows):
        r["idx"] = i
    return rows


def parse_indices(s: str) -> List[int]:
    if not s:
        return []
    return [int(x) for x in str(s).replace(",", " ").split() if x.strip()]


def collect_top_cliques(report: Dict[str, Any], top_k: int, color_filter: str | None) -> List[Dict[str, Any]]:
    rows = []
    for color, stat in report.get("color_stats", {}).items():
        if color_filter and color != color_filter:
            continue
        for c in stat.get("top_cliques", []):
            row = dict(c)
            row["color"] = color
            rows.append(row)
    rows.sort(key=lambda r: (-int(r.get("size", 0)), str(r.get("color", "")), int(r.get("rank", 999999))))
    return rows[:top_k]


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    keys = []
    for r in rows:
        for k, v in r.items():
            if isinstance(v, (list, dict)):
                continue
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k) for k in keys})


def ensure_plotting() -> None:
    if plt is None:
        raise RuntimeError(f"matplotlib unavailable: {_PLOT_ERROR}")


def plot_timeline(path: Path, nodes: List[Dict[str, Any]], cliques: List[Dict[str, Any]]) -> None:
    ensure_plotting()
    fig = plt.figure(figsize=(13, 5))
    ax = fig.add_subplot(111)

    ts = [n["t"] for n in nodes]
    ax.scatter(ts, [0] * len(ts), s=12, label="all candidates")

    y0 = 0.35
    for k, clique in enumerate(cliques):
        color = clique["color"]
        idxs = parse_indices(clique.get("node_indices", ""))
        cts = [nodes[i]["t"] for i in idxs if 0 <= i < len(nodes)]
        y = y0 + 0.25 * k
        ax.scatter(cts, [y] * len(cts), s=45, color=COLOR_MAP.get(color, "gray"), label=f"#{k+1} {color} K{len(cts)}")
        if cts:
            ax.plot([min(cts), max(cts)], [y, y], color=COLOR_MAP.get(color, "gray"), alpha=0.6)

    ax.set_xlabel("t")
    ax.set_yticks([0] + [y0 + 0.25 * k for k in range(len(cliques))])
    ax.set_yticklabels(["all"] + [f"#{k+1}" for k in range(len(cliques))])
    ax.set_title("Top monochromatic Ramsey cliques")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def plot_zoom(path: Path, nodes: List[Dict[str, Any]], clique: Dict[str, Any], pad: float) -> None:
    ensure_plotting()
    idxs = parse_indices(clique.get("node_indices", ""))
    cts = [nodes[i]["t"] for i in idxs if 0 <= i < len(nodes)]
    if not cts:
        return

    t_min = min(cts) - pad
    t_max = max(cts) + pad
    local = [n for n in nodes if t_min <= n["t"] <= t_max]
    color = clique["color"]

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    ax.scatter([n["t"] for n in local], [0] * len(local), s=18, label="local candidates")
    ax.scatter(cts, [0.25] * len(cts), s=70, color=COLOR_MAP.get(color, "gray"), label=f"{color} clique K{len(cts)}")

    for t in cts:
        ax.axvline(t, color=COLOR_MAP.get(color, "gray"), alpha=0.25)

    ax.set_xlim(t_min, t_max)
    ax.set_xlabel("t")
    ax.set_yticks([0, 0.25])
    ax.set_yticklabels(["local", "clique"])
    ax.set_title(f"Ramsey clique zoom: {color} K{len(cts)}")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Visualize top Ramsey cliques.")
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--ramsey_report", required=True)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--color", default=None, choices=["red", "blue", "green", "violet"])
    ap.add_argument("--zoom_pad", type=float, default=2.0)
    ap.add_argument("--out", default="ramsey_cliques")
    ap.add_argument("--no_plots", action="store_true")
    args = ap.parse_args()

    nodes = load_candidates(Path(args.candidates))
    report = load_json(Path(args.ramsey_report))
    cliques = collect_top_cliques(report, args.top_k, args.color)

    prefix = Path(args.out)
    prefix.parent.mkdir(parents=True, exist_ok=True)

    clique_rows = []
    node_rows = []

    for rank, c in enumerate(cliques, start=1):
        idxs = parse_indices(c.get("node_indices", ""))
        ts = [nodes[i]["t"] for i in idxs if 0 <= i < len(nodes)]
        row = dict(c)
        row["global_rank"] = rank
        row["n_nodes"] = len(idxs)
        row["t_values"] = " ".join(f"{t:.12f}" for t in ts)
        clique_rows.append(row)

        for local_rank, i in enumerate(idxs, start=1):
            if 0 <= i < len(nodes):
                n = nodes[i]
                node_rows.append({
                    "clique_rank": rank,
                    "local_rank": local_rank,
                    "color": c["color"],
                    "clique_size": len(idxs),
                    "node_idx": i,
                    "t": n["t"],
                    "score": n.get("score"),
                    "abs_zeta": n.get("abs_zeta"),
                    "kind": n.get("kind"),
                })

    write_csv(prefix.with_name(prefix.name + "_top_cliques.csv"), clique_rows)
    write_csv(prefix.with_name(prefix.name + "_clique_nodes.csv"), node_rows)

    summary = [
        "# Ramsey clique visualization",
        "",
        f"- candidates: `{args.candidates}`",
        f"- report: `{args.ramsey_report}`",
        f"- top_k: `{args.top_k}`",
        "",
        "## Top cliques",
        "",
        "| rank | color | size | t_min | t_max | diameter |",
        "|---:|---|---:|---:|---:|---:|",
    ]

    for r in clique_rows:
        summary.append(
            f"| {r['global_rank']} | {r['color']} | {r.get('size')} | "
            f"{float(r.get('t_min', 0)):.6f} | {float(r.get('t_max', 0)):.6f} | "
            f"{float(r.get('diameter_t', 0)):.6f} |"
        )

    prefix.with_name(prefix.name + "_summary.md").write_text("\n".join(summary), encoding="utf-8")

    if not args.no_plots:
        plot_timeline(prefix.with_name(prefix.name + "_timeline.png"), nodes, cliques)
        for rank, c in enumerate(cliques, start=1):
            color = c["color"]
            plot_zoom(prefix.with_name(prefix.name + f"_zoom_{rank}_{color}.png"), nodes, c, args.zoom_pad)

    print("=== Ramsey clique visualizer ===")
    print(f"nodes: {len(nodes)}")
    print(f"cliques visualized: {len(cliques)}")
    for r in clique_rows:
        print(f"#{r['global_rank']} {r['color']} K{r.get('size')} t=[{float(r.get('t_min', 0)):.6f}, {float(r.get('t_max', 0)):.6f}]")
    print(f"summary: {prefix.with_name(prefix.name + '_summary.md')}")


if __name__ == "__main__":
    main()
