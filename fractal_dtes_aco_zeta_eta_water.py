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
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import mpmath as mp
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
        self.rl_actions = [
            {"mix": 0.15, "gamma": 1.0, "alpha": 1.5},
            {"mix": 0.30, "gamma": 0.5, "alpha": 2.0},
            {"mix": 0.45, "gamma": 0.35, "alpha": 2.5},
        ]
        self.rl_values = [0.0 for _ in self.rl_actions]
        self.rl_counts = [1 for _ in self.rl_actions]
        self.rl_epsilon = 0.2
        self.current_action_idx = 0

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

    def update_water(self, mix: float = 0.15, gamma: float = 1.0) -> None:
        """One rain/evap/flow step on the existing dyadic tree graph."""
        if not self.water_by_node:
            self.initialize_water()
        if not self.water_by_node:
            return

        # Evaporation + rain.
        water_node_ids = list(self.water_by_node.keys())
        energy_rain = np.exp(
            -np.array([self._node_energy(nid) for nid in water_node_ids], dtype=float)
        )
        energy_rain = energy_rain / (energy_rain.max() + 1e-12)
        w = {
            nid: (1.0 - mix) * float(self.water_by_node[nid]) + mix * float(rain)
            for nid, rain in zip(water_node_ids, energy_rain)
        }
        w = {
            nid: float(np.power(np.clip(float(val), 0.0, 1.0), gamma))
            for nid, val in w.items()
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
            q = 0.0 * val
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

    def zeta_on_critical_line(self, t: float):
        return mp.zeta(mp.mpf("0.5") + 1j * mp.mpf(str(float(t))))

    def evaluate_path(
        self, path_nodes: List[int], agent_type: str = "default"
    ) -> float:
        base_score = float(super().evaluate_path(path_nodes, agent_type))
        w = self.water_score(path_nodes)
        # Multiplicative-ish but safe for arbitrary score signs.
        return base_score + self.water_score_weight * (w**self.water_alpha)

    def sample_ant_path(self, start_node_id: int, ant) -> List[int]:
        path = [start_node_id]
        current = start_node_id
        target_level = self.cfg.target_level or self.cfg.tree_depth

        for _step in range(self.cfg.max_ant_steps):
            if self.nodes[current].level >= target_level:
                break

            nbrs = self.neighbors.get(current, [])
            if not nbrs:
                break

            weights = []
            for nxt in nbrs:
                barrier = self.compute_barrier(current, nxt)
                tau = self.mixed_pheromone(ant.agent_type, current, nxt)
                pheromone = self.node_pheromone_mass(nxt)
                energy = self.nodes[nxt].energy
                effective_energy = energy - 0.3 * np.log(pheromone + 1e-12)
                barrier += self.cfg.lambda_energy * (effective_energy - energy)
                agent_score = self.compute_agent_value(ant, current, nxt)
                log_w = (
                    self.cfg.alpha * math.log(max(tau, self.cfg.tau_min))
                    - self.cfg.beta * barrier
                    + self.cfg.agent_gamma * agent_score
                )
                w = math.exp(min(700.0, max(-700.0, log_w)))
                weights.append(max(w, 1e-300))

            total = sum(weights)
            if total <= 0:
                break

            r = self.rng.random() * total
            cumsum = 0.0
            chosen = nbrs[-1]
            for nxt, w in zip(nbrs, weights):
                cumsum += w
                if r <= cumsum:
                    chosen = nxt
                    break

            path.append(chosen)
            current = chosen

        return path

    def reinforce_pheromones(self, paths: List[AntPath]) -> None:
        for ant_path in paths:
            ids = ant_path.node_ids
            channel_scale = self.agent_type_channel_scale(ant_path.agent_type)
            deposit = ant_path.score * channel_scale
            deposit = getattr(ant_path, "mass", 1.0) * deposit
            agent_channel = self.pheromone_channels[ant_path.agent_type]
            cross_weights = (
                self.channel_agreement_weights(ant_path.agent_type)
                if self.cfg.enable_cross_channel_agreement
                else {}
            )

            for a, b in zip(ids[:-1], ids[1:]):
                # Shared memory: global colony consensus.
                self.pheromones[(a, b)] = min(
                    self.cfg.tau_max, self.pheromones[(a, b)] + deposit
                )
                self.pheromones[(b, a)] = min(
                    self.cfg.tau_max, self.pheromones[(b, a)] + deposit
                )

                # Own channel: preserves specialization.
                agent_channel[(a, b)] = min(
                    self.cfg.tau_max, agent_channel[(a, b)] + 1.25 * deposit
                )
                agent_channel[(b, a)] = min(
                    self.cfg.tau_max, agent_channel[(b, a)] + 1.25 * deposit
                )

                # Cross-channel agreement: weakly reinforce related channels.
                for channel_name, weight in cross_weights.items():
                    channel = self.pheromone_channels.get(channel_name)
                    if channel is None:
                        continue
                    cross_deposit = (
                        self.cfg.cross_reinforcement_strength * weight * deposit
                    )
                    channel[(a, b)] = min(
                        self.cfg.tau_max, channel[(a, b)] + cross_deposit
                    )
                    channel[(b, a)] = min(
                        self.cfg.tau_max, channel[(b, a)] + cross_deposit
                    )

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

            if random.random() < self.rl_epsilon:
                self.current_action_idx = random.randint(0, len(self.rl_actions) - 1)
            else:
                self.current_action_idx = int(np.argmax(self.rl_values))

            action = self.rl_actions[self.current_action_idx]
            mix = action["mix"]
            gamma = action["gamma"]
            self.water_alpha = action["alpha"]

            # Rain before sampling: ants see water accumulated from previous iterations.
            self.update_water(mix=mix, gamma=gamma)
            self.record_water_diagnostics(it + 1)
            self.inject_water_pheromone_bias()

            paths: List[AntPath] = []
            for ant in ants:
                path_nodes = self.sample_ant_path(self.root_id, ant)
                score = self.evaluate_path(path_nodes, ant.agent_type)
                terminal_t = self.nodes[path_nodes[-1]].center()
                ant.zeta_value = complex(
                    mp.zeta(mp.mpf("0.5") + 1j * mp.mpf(str(float(terminal_t))))
                )
                ant.mass = 1.0 / (abs(ant.zeta_value) + 1e-8)
                ant_path = AntPath(
                    node_ids=path_nodes, score=score, agent_type=ant.agent_type
                )
                ant_path.mass = ant.mass
                paths.append(ant_path)
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

            reward = 0.0
            if "best_val" in locals():
                reward += -np.log(best_val + 1e-12)
            reward += 0.1 * float(water_vals.mean()) if water_vals.size else 0.0
            if hasattr(self, "water_diag_history") and len(self.water_diag_history) > 0:
                reward += 0.5 * self.water_diag_history[-1].get("overlap_at_k", 0.0)

            idx = self.current_action_idx
            self.rl_counts[idx] += 1
            self.rl_values[idx] += (reward - self.rl_values[idx]) / self.rl_counts[idx]
            print(
                f"[RL] action={action} value={self.rl_values[self.current_action_idx]:.4f}"
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
        from fractal_dtes_aco_zeta_visual import save_metrics_json

        self.logger.emit("START", "Fractal-DTES-ACO-Zeta ETA run started")
        self._time_stage("evaluate_grid", self.evaluate_grid)
        self._time_stage(
            "compute_multiscale_features", self.compute_multiscale_features
        )
        self._time_stage("build_dyadic_tree", self.build_dyadic_tree)
        self._time_stage("aggregate_node_statistics", self.aggregate_node_statistics)
        self._time_stage("build_graph", self.build_graph)
        self._time_stage("compute_node_stability", self.compute_node_stability)
        self._time_stage("initialize_pheromones", self.initialize_pheromones)
        self._time_stage("run_aco", self.run_aco)
        candidate_nodes = self._time_stage(
            "rank_candidate_nodes", self.rank_candidate_nodes
        )
        candidates = self._time_stage(
            "refine_candidates", self.refine_candidates, candidate_nodes
        )

        water_k = 12
        water_node_ids = list(self.water_by_node.keys())
        water_vals = np.array(
            [self.water_by_node[nid] for nid in water_node_ids], dtype=float
        )
        top_water_idx = np.argsort(water_vals)[-water_k:] if water_vals.size else []
        water_candidate_ts = [
            float(self.nodes[water_node_ids[i]].center()) for i in top_water_idx
        ]
        try:
            refined_water_ts = []
            for i, t0 in zip(top_water_idx, water_candidate_ts):
                try:
                    node = self.nodes[water_node_ids[i]]
                    refined = self.local_refinement(node.interval)
                    refined_water_ts.append(
                        float(refined) if refined is not None else float(t0)
                    )
                except Exception:
                    refined_water_ts.append(float(t0))
            water_candidate_ts = refined_water_ts
        except Exception:
            pass

        def dedup_ts(ts, tol=1e-3):
            out = []
            for t in sorted(ts):
                if not out or abs(t - out[-1]) > tol:
                    out.append(float(t))
            return out

        def dedup_ts_by_zeta(ts, tol=0.1):
            cleaned = []
            for t in sorted(float(x) for x in ts):
                if not cleaned or abs(t - cleaned[-1]) > tol:
                    cleaned.append(t)
                else:
                    old = cleaned[-1]
                    try:
                        old_val = abs(complex(self.zeta_on_critical_line(old)))
                        new_val = abs(complex(self.zeta_on_critical_line(t)))
                    except Exception:
                        try:
                            old_val = self.eval_abs_zeta(old)
                            new_val = self.eval_abs_zeta(t)
                        except Exception:
                            old_val = abs(
                                complex(
                                    mp.zeta(
                                        mp.mpf("0.5") + 1j * mp.mpf(str(float(old)))
                                    )
                                )
                            )
                            new_val = abs(
                                complex(
                                    mp.zeta(
                                        mp.mpf("0.5") + 1j * mp.mpf(str(float(t)))
                                    )
                                )
                            )

                    if new_val < old_val:
                        cleaned[-1] = t
            return cleaned

        water_candidate_ts = dedup_ts(water_candidate_ts, tol=1e-4)

        try:
            candidates = list(candidates) + water_candidate_ts
        except NameError:
            candidates = water_candidate_ts

        candidates = dedup_ts_by_zeta(candidates, tol=0.1)

        def local_refine(t0, f, steps=5, grid=25, window=0.1):
            t_best = float(t0)

            for _ in range(steps):
                ts = np.linspace(t_best - window, t_best + window, grid)
                vals = [abs(complex(f(float(t)))) for t in ts]
                idx = int(np.argmin(vals))
                t_best = float(ts[idx])
                window *= 0.3

            return t_best

        try:
            refined = []
            for t0 in candidates:
                try:
                    t1 = local_refine(
                        float(t0),
                        lambda x: self.zeta_on_critical_line(float(x)),
                        steps=8,
                        grid=41,
                        window=0.08,
                    )
                    refined.append(float(t1))
                except Exception:
                    refined.append(float(t0))

            candidates = dedup_ts_by_zeta(refined, tol=0.1)

        except Exception as e:
            print(f"[WARN] local refine failed: {e}")

        out_dir = Path(self.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "water_candidates.json", "w", encoding="utf-8") as f:
            json.dump(water_candidate_ts, f, indent=2)

        candidates = self.merge_close_candidates(candidates)
        self.metrics = self.compute_metrics(candidate_nodes, candidates)
        self.metrics["stage_timings_s"] = self.stage_timings
        self.metrics["aco_history"] = self.aco_history
        self.metrics["config"] = self._safe_config_dict()

        try:
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
                os.path.join(self.out_dir, "metrics_summary.json"), self.metrics
            )
            save_metrics_json(
                os.path.join(self.out_dir, "aco_history.json"), self.aco_history
            )
            save_metrics_json(
                os.path.join(self.out_dir, "water_history.json"), self.water_history
            )
            with open(out_dir / "water_diag_history.json", "w", encoding="utf-8") as f:
                json.dump(self.water_diag_history, f, indent=2)
            save_metrics_json(
                os.path.join(self.out_dir, "metrics_summary_water.json"), self.metrics
            )
        except Exception as exc:
            self.logger.emit("WATER", "failed to save water metrics", error=str(exc))

        self.plot_all(candidates, self.out_dir)
        self.logger.emit(
            "DONE",
            "run completed",
            candidates=len(candidates),
            total_time=_format_seconds(time.time() - self.logger.global_start),
        )
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
