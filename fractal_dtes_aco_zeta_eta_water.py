from __future__ import annotations

"""
Water-field / rain extension for Ant-RH ETA runner.

Put this file next to:
    fractal_dtes_aco_zeta_eta.py
    fractal_dtes_aco_zeta_metrics.py
    fractal_dtes_aco_zeta_visual.py

Run:
    python3 fractal_dtes_aco_zeta_eta_water.py

This keeps the ETA pipeline, but adds a slow hydrodynamic field W over tree nodes:
    rain  -> adds water everywhere
    flow  -> moves water from high-energy nodes to lower-energy children/neighbors
    evap  -> prevents global flooding
    score -> rewards ant paths ending in / passing through water-rich basins

The idea: zeta zeros are low-energy wells in E(t)=log(|zeta(1/2+it)|+eps),
so stable wells should accumulate water, and ants should be attracted to water.
"""

import math
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from fractal_dtes_aco_zeta_eta import ETAFractalDTESACOZeta, _format_seconds
from fractal_dtes_aco_zeta_metrics import AntPath, ZetaSearchConfig


class WaterETAFractalDTESACOZeta(ETAFractalDTESACOZeta):
    """ETA runner with DTES rain/water-field bias.

    This subclass is intentionally conservative: it does not depend on the
    exact internals of sample_ant_path(). Instead it injects water through:
      1) path score bonus, so pheromone reinforcement prefers wet basins;
      2) optional light pheromone pre-bias on edges entering wet nodes.
    """

    def __init__(self, cfg: ZetaSearchConfig):
        super().__init__(cfg)
        self.water_by_node: Dict[int, float] = {}
        self.water_history: List[dict] = []
        self.water_diag_history = []

        # Safe defaults. Override by setting cfg.water_* after creating cfg.
        self.water_rain_rate = float(getattr(cfg, "water_rain_rate", 0.012))
        self.water_evap_rate = float(getattr(cfg, "water_evap_rate", 0.985))
        self.water_flow_rate = float(getattr(cfg, "water_flow_rate", 0.35))
        self.water_alpha = float(getattr(cfg, "water_alpha", 2.0))
        self.water_score_weight = float(getattr(cfg, "water_score_weight", 0.35))
        self.water_pheromone_bias = float(getattr(cfg, "water_pheromone_bias", 0.08))

    # ---------- water field -------------------------------------------------

    def _node_energy(self, nid: int) -> float:
        node = self.nodes[nid]
        e = getattr(node, "energy", None)
        if e is None or not np.isfinite(e):
            return 0.0
        return float(e)

    def initialize_water(self) -> None:
        """Initialize W with larger mass in lower-energy nodes."""
        ids = list(self.nodes.keys())
        if not ids:
            self.water_by_node = {}
            return

        energies = np.array([self._node_energy(nid) for nid in ids], dtype=float)
        e_med = float(np.nanmedian(energies))
        e_scale = float(np.nanstd(energies) + 1e-9)

        # Low energy -> larger initial water, but bounded.
        init = np.exp(-(energies - e_med) / e_scale)
        init = init / (float(np.max(init)) + 1e-12)
        self.water_by_node = {nid: float(w) for nid, w in zip(ids, init)}

    def update_water(self) -> None:
        """One rain/evap/flow step on the existing dyadic tree graph."""
        if not self.water_by_node:
            self.initialize_water()
        if not self.water_by_node:
            return

        # Evaporation + rain.
        w = {
            nid: 0.94 * float(val) + 0.001 * np.exp(-self._node_energy(nid))
            for nid, val in self.water_by_node.items()
        }
        w_new = dict(w)

        # Flow down the energy landscape. Prefer children if available,
        # otherwise use graph neighbors if the base implementation exposes them.
        for nid, val in w.items():
            if val <= 0.0:
                continue
            e0 = self._node_energy(nid)
            node = self.nodes[nid]

            neigh = []
            children = getattr(node, "children", None)
            if children:
                neigh.extend(list(children))

            graph = getattr(self, "graph", None)
            if graph is not None:
                try:
                    neigh.extend(list(graph.get(nid, [])))
                except Exception:
                    pass

            neigh = [m for m in set(neigh) if m in self.nodes]
            downhill = [m for m in neigh if self._node_energy(m) < e0]
            if not downhill:
                continue

            # Send water to the lowest-energy reachable neighbor.
            j = min(downhill, key=self._node_energy)
            q = 0.05 * val
            w_new[nid] = w_new.get(nid, 0.0) - q
            w_new[j] = w_new.get(j, 0.0) + q

        # Clip to [0, 1].
        self.water_by_node = {
            nid: float(np.clip(float(v), 0.0, 1.0)) for nid, v in w_new.items()
        }

    def record_water_diagnostics(self, iteration: int) -> None:
        water_node_ids = list(self.water_by_node.keys())
        if not water_node_ids:
            return

        water_vals = np.array(
            [self.water_by_node[nid] for nid in water_node_ids], dtype=float
        )
        energy_vals = np.array(
            [self._node_energy(nid) for nid in water_node_ids], dtype=float
        )
        try:
            corr = float(np.corrcoef(water_vals, -energy_vals)[0, 1])
        except Exception:
            corr = float("nan")

        k = 10 if len(water_vals) >= 10 else len(water_vals)
        top_w_idx = np.argsort(water_vals)[-k:]
        low_e_idx = np.argsort(energy_vals)[:k]
        overlap_at_k = len(
            set(map(int, top_w_idx)).intersection(set(map(int, low_e_idx)))
        ) / max(1, k)

        top_w_t = [float(self.nodes[water_node_ids[i]].center()) for i in top_w_idx]
        low_e_t = [float(self.nodes[water_node_ids[i]].center()) for i in low_e_idx]

        self.water_diag_history.append(
            {
                "iteration": int(iteration),
                "corr_water_neg_energy": corr,
                "water_mean": float(water_vals.mean()),
                "water_max": float(water_vals.max()),
                "wet_nodes_075": int(np.sum(water_vals >= 0.75)),
                "overlap_at_k": float(overlap_at_k),
                "top_water": top_w_t,
                "low_energy": low_e_t,
            }
        )

    def water_score(self, path_nodes: List[int]) -> float:
        if not path_nodes or not self.water_by_node:
            return 0.0
        vals = np.array(
            [self.water_by_node.get(nid, 0.0) for nid in path_nodes], dtype=float
        )
        terminal = float(vals[-1])
        mean = float(np.mean(vals))
        return (terminal + mean) * 0.5

    def evaluate_path(
        self, path_nodes: List[int], agent_type: str = "default"
    ) -> float:
        base_score = float(super().evaluate_path(path_nodes, agent_type))
        w = self.water_score(path_nodes)
        # Multiplicative-ish but safe for arbitrary score signs.
        return base_score + self.water_score_weight * (w**self.water_alpha)

    def inject_water_pheromone_bias(self) -> None:
        """Light pre-bias: edges into wet nodes get a tiny pheromone lift.

        If the parent implementation uses self.pheromones[(a,b)], this helps
        ants choose water-rich branches immediately. If not, it is harmless.
        """
        if not getattr(self, "pheromones", None) or not self.water_by_node:
            return
        for edge in list(self.pheromones.keys()):
            try:
                _, dst = edge
            except Exception:
                continue
            wet = self.water_by_node.get(dst, 0.0)
            if wet > 0.0:
                self.pheromones[edge] = float(self.pheromones[edge]) * (
                    1.0 + self.water_pheromone_bias * (wet**self.water_alpha)
                )

    # ---------- ACO loop with ETA + water logs ------------------------------

    def run_aco(self) -> None:
        if self.root_id is None:
            raise RuntimeError("Tree root is not built.")

        self.initialize_water()
        self.logger.emit(
            "WATER",
            "water field initialized",
            rain=self.water_rain_rate,
            evap=self.water_evap_rate,
            flow=self.water_flow_rate,
            alpha=self.water_alpha,
        )

        ants = [self.make_ant(k) for k in range(self.cfg.n_ants)]
        ema_iter_time: Optional[float] = None
        ema_alpha = float(getattr(self.cfg, "eta_ema_alpha", 0.20))
        log_every = int(getattr(self.cfg, "eta_log_every", 1))
        early_patience = int(getattr(self.cfg, "early_stop_patience", 0))
        min_delta = float(getattr(self.cfg, "early_stop_min_delta", 1e-9))
        best_score = -math.inf
        stale = 0
        start = time.time()

        for it in range(self.cfg.n_iterations):
            iter_start = time.time()

            # Rain before sampling: ants see water accumulated from previous iterations.
            self.update_water()
            self.record_water_diagnostics(it + 1)
            self.inject_water_pheromone_bias()

            paths: List[AntPath] = []
            for ant in ants:
                path_nodes = self.sample_ant_path(self.root_id, ant)
                score = self.evaluate_path(path_nodes, ant.agent_type)
                paths.append(
                    AntPath(node_ids=path_nodes, score=score, agent_type=ant.agent_type)
                )
                self.update_ant_memory(ant, path_nodes)
                for nid in path_nodes:
                    self.nodes[nid].visit_count += 1

            self.evaporate_pheromones()
            self.reinforce_pheromones(paths)

            iter_time = time.time() - iter_start
            ema_iter_time = (
                iter_time
                if ema_iter_time is None
                else (ema_alpha * iter_time + (1.0 - ema_alpha) * ema_iter_time)
            )
            done = it + 1
            remaining = self.cfg.n_iterations - done
            eta = ema_iter_time * remaining

            scores = np.array([p.score for p in paths], dtype=float)
            mean_score = float(np.mean(scores)) if scores.size else None
            max_score = float(np.max(scores)) if scores.size else None
            best_energy = self._current_best_leaf_energy()
            pheromone_mass = (
                float(sum(self.pheromones.values())) if self.pheromones else 0.0
            )
            diversity = self._terminal_diversity(paths)

            water_vals = np.array(list(self.water_by_node.values()), dtype=float)
            water_max = float(np.max(water_vals)) if water_vals.size else 0.0
            water_mean = float(np.mean(water_vals)) if water_vals.size else 0.0
            wet_nodes = int(np.sum(water_vals > 0.75)) if water_vals.size else 0

            rec = {
                "iteration": done,
                "n_iterations": self.cfg.n_iterations,
                "iter_time_s": iter_time,
                "ema_iter_time_s": ema_iter_time,
                "elapsed_s": time.time() - start,
                "eta_s": eta,
                "mean_score": mean_score,
                "max_score": max_score,
                "best_leaf_energy": best_energy,
                "shared_pheromone_mass": pheromone_mass,
                "terminal_diversity": diversity,
                "water_max": water_max,
                "water_mean": water_mean,
                "wet_nodes_075": wet_nodes,
            }
            self.aco_history.append(rec)
            self.water_history.append(
                {
                    "iteration": done,
                    "water_max": water_max,
                    "water_mean": water_mean,
                    "wet_nodes_075": wet_nodes,
                }
            )

            if done % max(1, log_every) == 0 or done == self.cfg.n_iterations:
                self.logger.emit(
                    "ACO+WATER",
                    f"iter {done}/{self.cfg.n_iterations}",
                    iter_time=_format_seconds(iter_time),
                    eta=_format_seconds(eta),
                    best_E=f"{best_energy:.5g}" if best_energy is not None else None,
                    max_score=f"{max_score:.5g}" if max_score is not None else None,
                    diversity=f"{diversity:.3f}",
                    water_mean=f"{water_mean:.3f}",
                    wet_nodes=wet_nodes,
                )

            if max_score is not None and max_score > best_score + min_delta:
                best_score = max_score
                stale = 0
            else:
                stale += 1

            if early_patience > 0 and stale >= early_patience:
                self.logger.emit(
                    "EARLY_STOP",
                    f"no max_score improvement for {early_patience} iterations",
                    iteration=done,
                )
                break

    def run(self) -> List[float]:
        candidates = super().run()
        try:
            from fractal_dtes_aco_zeta_visual import save_metrics_json

            self.metrics["water_history"] = self.water_history
            self.metrics["water_config"] = {
                "rain_rate": self.water_rain_rate,
                "evap_rate": self.water_evap_rate,
                "flow_rate": self.water_flow_rate,
                "alpha": self.water_alpha,
                "score_weight": self.water_score_weight,
                "pheromone_bias": self.water_pheromone_bias,
            }
            save_metrics_json(
                os.path.join(self.out_dir, "water_history.json"), self.water_history
            )
            out_dir = Path("fractal_dtes_aco_eta_water_output")
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "water_diag_history.json", "w", encoding="utf-8") as f:
                json.dump(self.water_diag_history, f, indent=2)
            save_metrics_json(
                os.path.join(self.out_dir, "metrics_summary_water.json"), self.metrics
            )
        except Exception as exc:
            self.logger.emit("WATER", "failed to save water metrics", error=str(exc))
        return candidates


def default_water_eta_demo() -> None:
    out_dir = "fractal_dtes_aco_eta_water_output"
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

    cfg.out_dir = out_dir
    cfg.verbose_eta = True
    cfg.eta_log_every = 1
    cfg.eta_ema_alpha = 0.20
    cfg.early_stop_patience = 0
    cfg.early_stop_min_delta = 1e-9
    cfg.metrics_out_path = os.path.join(out_dir, "metrics_summary_water.json")

    # Water knobs. Increase water_score_weight to make ants more water-seeking.
    cfg.water_rain_rate = 0.012
    cfg.water_evap_rate = 0.985
    cfg.water_flow_rate = 0.35
    cfg.water_alpha = 2.0
    cfg.water_score_weight = 0.35
    cfg.water_pheromone_bias = 0.08

    searcher = WaterETAFractalDTESACOZeta(cfg)
    candidates = searcher.run()

    print("\nCandidates:")
    import mpmath as mp

    for t in candidates:
        z = mp.zeta(mp.mpf("0.5") + 1j * mp.mpf(str(float(t))))
        print(f"t = {t:.12f} |zeta| = {abs(complex(z)):.3e}")
    print(f"\nOutputs: {out_dir}")


if __name__ == "__main__":
    default_water_eta_demo()
