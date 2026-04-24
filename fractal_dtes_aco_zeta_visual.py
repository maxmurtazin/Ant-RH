from __future__ import annotations

"""Matplotlib diagnostics for Fractal-DTES-ACO-Zeta (Ant-RH).

Plots energy landscapes, ACO-related views, and writes ``save_metrics_json`` helpers.
Depends on ``FractalDTESACOZeta`` from ``fractal_dtes_aco_zeta_metrics``; does not add
new search logic beyond visualization I/O.
"""

import json
import os
from typing import Dict, List, Optional

import numpy as np
import mpmath as mp

from fractal_dtes_aco_zeta_metrics import (
    FractalDTESACOZeta,
    ZetaSearchConfig,
    _run_lengths,
)


def _ensure_output_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _json_default(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


def save_metrics_json(path: str, data) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_json_default)


class VisualFractalDTESACOZeta(FractalDTESACOZeta):
    """Fractal-DTES-ACO-Zeta with matplotlib visual diagnostics."""

    def plot_energy_landscape(self, candidates: Optional[List[float]], out_dir: str) -> str:
        import matplotlib.pyplot as plt

        path = os.path.join(out_dir, "energy_landscape_candidates.png")
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        ax.plot(self.t_grid, self.log_abs_values, linewidth=1.0, label="log(|zeta| + eps)")
        if candidates:
            ys = []
            for t in candidates:
                idx = int(np.argmin(np.abs(self.t_grid - t)))
                ys.append(float(self.log_abs_values[idx]))
            ax.scatter(candidates, ys, marker="x", s=60, label="candidates")
        ax.set_title("DTES energy landscape on the critical line")
        ax.set_xlabel("t")
        ax.set_ylabel("log(|zeta(1/2+it)| + eps)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path

    def plot_tree_energy_by_level(self, out_dir: str) -> str:
        import matplotlib.pyplot as plt

        path = os.path.join(out_dir, "tree_energy_by_level.png")
        xs = [node.center() for node in self.nodes]
        ys = [node.level for node in self.nodes]
        energies = [node.energy for node in self.nodes]
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(111)
        sc = ax.scatter(xs, ys, c=energies, s=18)
        ax.set_title("Fractal tree nodes: energy by level")
        ax.set_xlabel("t-center")
        ax.set_ylabel("tree level")
        fig.colorbar(sc, ax=ax, label="node energy")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path

    def plot_pheromone_distribution(self, out_dir: str) -> str:
        import matplotlib.pyplot as plt

        path = os.path.join(out_dir, "pheromone_distribution.png")
        vals = np.array(list(self.pheromones.values()), dtype=float) if self.pheromones else np.array([])
        fig = plt.figure(figsize=(9, 5))
        ax = fig.add_subplot(111)
        if vals.size:
            ax.hist(np.log10(np.maximum(vals, self.cfg.tau_min)), bins=40)
        ax.set_title("Shared pheromone distribution")
        ax.set_xlabel("log10(pheromone)")
        ax.set_ylabel("edge count")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path

    def plot_channel_summary(self, out_dir: str) -> str:
        import matplotlib.pyplot as plt

        path = os.path.join(out_dir, "pheromone_channels_summary.png")
        names, means, maxs = [], [], []
        for name, channel in self.pheromone_channels.items():
            vals = np.array(list(channel.values()), dtype=float) if channel else np.array([])
            names.append(name)
            means.append(float(np.mean(vals)) if vals.size else 0.0)
            maxs.append(float(np.max(vals)) if vals.size else 0.0)
        x = np.arange(len(names))
        width = 0.35
        fig = plt.figure(figsize=(9, 5))
        ax = fig.add_subplot(111)
        ax.bar(x - width / 2, means, width, label="mean")
        ax.bar(x + width / 2, maxs, width, label="max")
        ax.set_title("Pheromone channels summary")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20)
        ax.set_ylabel("pheromone")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path

    def plot_ramsey_coloring(self, out_dir: str) -> Dict[str, str]:
        import matplotlib.pyplot as plt

        level = self.cfg.target_level or self.cfg.tree_depth
        node_ids = sorted(self.nodes_by_level.get(level, []), key=lambda nid: self.nodes[nid].center())
        centers = [self.nodes[nid].center() for nid in node_ids]
        colors = [self.node_color(nid) for nid in node_ids]
        runs = _run_lengths(colors)

        path1 = os.path.join(out_dir, "ramsey_coloring.png")
        fig = plt.figure(figsize=(10, 4))
        ax = fig.add_subplot(111)
        if centers:
            ax.scatter(centers, colors, s=18)
        ax.set_title(f"Ramsey coloring at tree level {level}")
        ax.set_xlabel("t-center")
        ax.set_ylabel("dominant transition color")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path1, dpi=180)
        plt.close(fig)

        path2 = os.path.join(out_dir, "ramsey_run_lengths.png")
        fig2 = plt.figure(figsize=(8, 4))
        ax2 = fig2.add_subplot(111)
        if runs:
            ax2.hist(runs, bins=range(1, max(runs) + 2))
        ax2.set_title("Ramsey monochromatic run lengths")
        ax2.set_xlabel("run length")
        ax2.set_ylabel("count")
        ax2.grid(True, alpha=0.3)
        fig2.tight_layout()
        fig2.savefig(path2, dpi=180)
        plt.close(fig2)
        return {"ramsey_coloring": path1, "ramsey_run_lengths": path2}

    def plot_agent_path_scores(self, out_dir: str) -> str:
        import matplotlib.pyplot as plt

        path = os.path.join(out_dir, "agent_path_scores.png")
        paths = getattr(self, "last_aco_paths", [])
        grouped = {t: [] for t in self.ant_types}
        for p in paths:
            grouped.setdefault(p.agent_type, []).append(p.score)
        names = list(grouped.keys())
        data = [grouped[n] if grouped[n] else [0.0] for n in names]
        fig = plt.figure(figsize=(9, 5))
        ax = fig.add_subplot(111)
        ax.boxplot(data, labels=names, showfliers=False)
        ax.set_title("Agent path score distribution")
        ax.set_ylabel("path score")
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        return path

    def generate_visualizations(self, candidates: Optional[List[float]] = None, out_dir: str = "zeta_visuals") -> Dict[str, str]:
        out_dir = _ensure_output_dir(out_dir)
        if candidates is None:
            candidates = getattr(self, "last_candidates", [])
        outputs: Dict[str, str] = {}
        outputs["energy_landscape"] = self.plot_energy_landscape(candidates, out_dir)
        outputs["tree_energy"] = self.plot_tree_energy_by_level(out_dir)
        outputs["pheromone_distribution"] = self.plot_pheromone_distribution(out_dir)
        outputs["channel_summary"] = self.plot_channel_summary(out_dir)
        outputs.update(self.plot_ramsey_coloring(out_dir))
        outputs["agent_scores"] = self.plot_agent_path_scores(out_dir)
        metrics = self.compute_metrics(candidates)
        metrics_path = os.path.join(out_dir, "metrics_summary.json")
        save_metrics_json(metrics_path, metrics)
        outputs["metrics_json"] = metrics_path
        return outputs

    def visual_run(self, out_dir: str = "zeta_visuals"):
        candidates = self.run()
        outputs = self.generate_visualizations(candidates, out_dir=out_dir)
        return candidates, outputs


def visual_demo() -> None:
    cfg = ZetaSearchConfig(
        t_min=10.0,
        t_max=40.0,
        n_grid=2048,
        tree_depth=8,
        feature_levels=5,
        n_ants=48,
        n_iterations=60,
        max_ant_steps=20,
        top_candidate_nodes=16,
        verification_abs_tol=1e-6,
        refinement_subgrid=128,
        r0=6.0,
        mp_dps=50,
    )
    searcher = VisualFractalDTESACOZeta(cfg)
    candidates, outputs = searcher.visual_run(out_dir="zeta_visuals")
    print("Candidates:")
    for t in candidates:
        z = mp.zeta(mp.mpf("0.5") + 1j * mp.mpf(str(float(t))))
        print(f"t = {t:.12f} |zeta| = {abs(complex(z)):.3e}")
    print("\nVisualization files:")
    for name, path in outputs.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    visual_demo()
