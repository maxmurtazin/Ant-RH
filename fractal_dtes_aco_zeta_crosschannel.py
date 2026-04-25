from __future__ import annotations

# Small I/O helpers defined first so external patch scripts can import them without
# pulling the full NumPy/mpmath stack if only JSON merging is needed.

def _edge_aware_candidates(candidates, t_min, t_max, edge_padding=2.5, edge_step=0.05):
    import math

    cleaned = []
    for t in candidates:
        try:
            tt = float(t)
        except:
            continue
        if t_min <= tt <= t_max:
            cleaned.append(tt)

    anchors = []
    n = int(math.ceil(edge_padding / edge_step))
    for i in range(n + 1):
        d = i * edge_step
        anchors.append(t_min + d)
        anchors.append(t_max - d)

    merged = cleaned + anchors
    merged = sorted(set(round(float(t), 12) for t in merged))
    return merged


def _autosave_dtes_candidates(candidates, path="dtes_candidates.json"):
    import json
    rows = []
    for i, t in enumerate(candidates, start=1):
        try:
            tt = float(t)
        except:
            continue
        rows.append({
            "rank": i,
            "t": tt
        })

    with open(path, "w", encoding="utf-8") as f:
        json.dump({"candidates": rows}, f, indent=2)

    print(f"[SAVE] {path} | count={len(rows)}")


import cmath
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np
import mpmath as mp


KNOWN_ZETA_ZEROS: Tuple[float, ...] = (
    14.134725141734693,
    21.022039638771554,
    25.010857580145688,
    30.424876125859513,
    32.935061587739189,
    37.586178158825671,
    40.918719012147495,
    43.327073280914999,
    48.005150881167159,
    49.773832477672302,
    52.970321477714460,
    56.446247697063394,
    59.347044002602353,
    60.831778524609809,
    65.112544048081606,
    67.079810529494173,
    69.546401711173979,
    72.067157674481907,
    75.704690699083933,
    77.144840068874805,
    79.337375020249367,
    82.910380854086030,
    84.735492980517050,
    87.425274613125229,
    88.809111207634465,
    92.491899270558484,
    94.651344040519886,
    95.870634228245309,
    98.831194218193692,
    101.31785100573139,
)


# ------------------------------------------------------------
# Fractal-DTES-ACO-Zeta: research skeleton / MVP
# ------------------------------------------------------------
# Self-contained CrossChannel variant (duplicated config/core vs metrics.py) so this
# module can be patched or run in isolation; for a dependency-light grid-only path see
# ``core/fractal_dtes_crosschannel_explore_eta_clean.py``.
# This file implements a practical first version of the scheme:
#   1) sample zeta(1/2 + i t) on a grid,
#   2) build multi-scale local features,
#   3) build a dyadic interval tree,
#   4) define node energies,
#   5) run ACO on tree + same-level neighbor graph,
#   6) rank promising intervals,
#   7) locally refine candidates.
#
# It is intentionally modular so that you can replace any part:
# - the energy,
# - the features,
# - the graph topology,
# - the path score,
# - the refinement routine.
# ------------------------------------------------------------


@dataclass
class ZetaSearchConfig:
    t_min: float
    t_max: float
    n_grid: int = 4096
    tree_depth: int = 8
    max_ant_steps: int = 24
    n_ants: int = 64
    n_iterations: int = 80
    target_level: Optional[int] = None

    # Numerical stability
    eps: float = 1e-14
    mp_dps: int = 50
    random_seed: int = 42

    # Feature windows: r0 / 2^(ell-1)
    feature_levels: int = 5
    r0: float = 8.0

    # Edge / transition parameters
    alpha: float = 1.0  # pheromone importance
    beta: float = 2.0   # barrier importance
    rho: float = 0.12   # evaporation
    tau0: float = 1.0
    tau_min: float = 1e-6
    tau_max: float = 1e6

    # Multi-agent DTES ants
    ant_types: Tuple[str, ...] = ("modulus", "phase", "multiscale", "stability")
    agent_gamma: float = 1.0
    agent_memory_decay: float = 0.85
    agent_phase_weight: float = 0.90
    agent_multiscale_weight: float = 0.85
    agent_stability_weight: float = 1.10
    agent_modulus_weight: float = 1.00

    # Cross-channel pheromone agreement
    enable_cross_channel_agreement: bool = True
    shared_pheromone_mix: float = 0.45
    own_channel_mix: float = 0.35
    cross_channel_mix: float = 0.20
    cross_reinforcement_strength: float = 0.20

    # Energy weights
    w_min_logabs: float = 1.0
    w_mean_logabs: float = 0.30
    w_var_phase: float = 0.15
    w_roughness: float = 0.20
    w_sign_penalty: float = 0.40

    # Barrier weights
    lambda_energy: float = 1.0
    lambda_signature: float = 0.15
    lambda_tree: float = 0.20
    lambda_phase: float = 0.10

    # Path score weights
    gamma_depth: float = 0.30
    gamma_coherence: float = 0.30
    gamma_osc: float = 0.20

    # Graph construction
    same_level_neighbor_k: int = 2
    feature_neighbor_radius: Optional[float] = None

    # Candidate extraction
    top_candidate_nodes: int = 24
    merge_tol: float = 1e-5
    verification_abs_tol: float = 1e-8
    refinement_subgrid: int = 128

    # Experimental spectral-learning feedback
    spectral_learning: bool = False
    spectral_weight: float = 0.1
    spectral_k: int = 30
    spectral_max_points: int = 256
    zeta_zeros: Tuple[float, ...] = KNOWN_ZETA_ZEROS

    def __post_init__(self) -> None:
        if self.target_level is None:
            self.target_level = self.tree_depth


@dataclass
class FractalNode:
    node_id: int
    level: int
    interval: Tuple[float, float]
    point_ids: List[int]
    parent_id: Optional[int] = None
    child_ids: List[int] = field(default_factory=list)

    signature: np.ndarray = field(default_factory=lambda: np.zeros(1, dtype=float))
    energy: float = math.inf
    min_log_abs: float = math.inf
    mean_log_abs: float = math.inf
    var_phase: float = math.inf
    roughness: float = math.inf
    sign_changes: int = 0

    visit_count: int = 0

    def center(self) -> float:
        return 0.5 * (self.interval[0] + self.interval[1])

    def width(self) -> float:
        return self.interval[1] - self.interval[0]


@dataclass
class AntPath:
    node_ids: List[int]
    score: float
    agent_type: str


@dataclass
class DTESAnt:
    agent_id: int
    agent_type: str
    memory_node_id: Optional[int] = None
    memory_energy_drop: float = 0.0
    memory_phase_pref: float = 0.0
    memory_stability_pref: float = 0.0


class FractalDTESACOZeta:
    def __init__(self, cfg: ZetaSearchConfig):
        self.cfg = cfg
        self.rng = random.Random(cfg.random_seed)
        np.random.seed(cfg.random_seed)
        mp.mp.dps = cfg.mp_dps

        self.t_grid = np.linspace(cfg.t_min, cfg.t_max, cfg.n_grid)
        self.values: List[complex] = []
        self.abs_values: np.ndarray = np.array([])
        self.log_abs_values: np.ndarray = np.array([])
        self.phase_values: np.ndarray = np.array([])
        self.features: Dict[int, np.ndarray] = {}

        self.nodes: Dict[int, FractalNode] = {}
        self.root_id: Optional[int] = None
        self.nodes_by_level: Dict[int, List[int]] = {}
        self.neighbors: Dict[int, List[int]] = {}
        self.pheromones: Dict[Tuple[int, int], float] = {}
        self.pheromone_channels: Dict[str, Dict[Tuple[int, int], float]] = {}
        self.node_stability: Dict[int, float] = {}
        self.ant_types = tuple(cfg.ant_types)

    # ------------------------------
    # Public API
    # ------------------------------
    def zeta(self, s):
        return mp.zeta(s)

    def run(self) -> List[float]:
        self.evaluate_grid()
        self.compute_multiscale_features()
        self.build_dyadic_tree()
        self.aggregate_node_statistics()
        self.build_graph()
        self.compute_node_stability()
        self.initialize_pheromones()
        self.run_aco()
        candidate_nodes = self.rank_candidate_nodes()
        candidates = self.refine_candidates(candidate_nodes)
        merged = self.merge_close_candidates(candidates)
        return self.filter_candidates_to_interval(merged)

    # ------------------------------
    # Step 1. Evaluate zeta on grid
    # ------------------------------
    def evaluate_grid(self) -> None:
        vals: List[complex] = []
        for t in self.t_grid:
            z = mp.zeta(mp.mpf('0.5') + 1j * mp.mpf(str(float(t))))
            vals.append(complex(z))

        self.values = vals
        self.abs_values = np.array([abs(z) for z in vals], dtype=float)
        self.log_abs_values = np.log(self.abs_values + self.cfg.eps)

        raw_phase = np.array([cmath.phase(z) for z in vals], dtype=float)
        self.phase_values = np.unwrap(raw_phase)

    # ------------------------------
    # Step 2. Multi-scale features
    # ------------------------------
    def compute_multiscale_features(self) -> None:
        for i, t in enumerate(self.t_grid):
            feat: List[float] = []
            for ell in range(1, self.cfg.feature_levels + 1):
                radius = self.cfg.r0 / (2 ** (ell - 1))
                lo = t - radius
                hi = t + radius
                idx = self.indices_in_window(lo, hi)
                feat.extend(self.compute_window_features(idx))
            self.features[i] = np.array(feat, dtype=float)

    def compute_window_features(self, idx: np.ndarray) -> List[float]:
        if idx.size == 0:
            return [math.inf, math.inf, math.inf, math.inf, math.inf]

        local_log_abs = self.log_abs_values[idx]
        local_phase = self.phase_values[idx]
        local_vals = [self.values[j] for j in idx]

        min_log_abs = float(np.min(local_log_abs))
        mean_log_abs = float(np.mean(local_log_abs))
        var_phase = float(np.var(local_phase))
        roughness = self.local_roughness(local_log_abs)
        sign_changes = float(self.count_sign_changes(local_vals))
        return [min_log_abs, mean_log_abs, var_phase, roughness, sign_changes]

    @staticmethod
    def local_roughness(series: np.ndarray) -> float:
        if len(series) < 3:
            return 0.0
        second_diff = np.diff(series, n=2)
        return float(np.mean(np.abs(second_diff)))

    @staticmethod
    def count_sign_changes(vals: List[complex]) -> int:
        def count_component_sign_changes(arr: Iterable[float]) -> int:
            last = 0
            count = 0
            for x in arr:
                s = 1 if x > 0 else (-1 if x < 0 else 0)
                if s == 0:
                    continue
                if last != 0 and s != last:
                    count += 1
                last = s
            return count

        real_changes = count_component_sign_changes((z.real for z in vals))
        imag_changes = count_component_sign_changes((z.imag for z in vals))
        return real_changes + imag_changes

    def indices_in_window(self, lo: float, hi: float) -> np.ndarray:
        left = int(np.searchsorted(self.t_grid, lo, side="left"))
        right = int(np.searchsorted(self.t_grid, hi, side="right"))
        if right <= left:
            return np.array([], dtype=int)
        return np.arange(left, right, dtype=int)

    # ------------------------------
    # Step 3. Dyadic tree
    # ------------------------------
    def build_dyadic_tree(self) -> None:
        self.nodes.clear()
        self.nodes_by_level.clear()

        all_point_ids = list(range(len(self.t_grid)))
        root = self._make_node(level=0, interval=(self.cfg.t_min, self.cfg.t_max), point_ids=all_point_ids)
        self.root_id = root.node_id
        self._split_node_recursive(root.node_id, depth=self.cfg.tree_depth)

    def _make_node(self, level: int, interval: Tuple[float, float], point_ids: List[int], parent_id: Optional[int] = None) -> FractalNode:
        node_id = len(self.nodes)
        node = FractalNode(node_id=node_id, level=level, interval=interval, point_ids=point_ids, parent_id=parent_id)
        self.nodes[node_id] = node
        self.nodes_by_level.setdefault(level, []).append(node_id)
        return node

    def _split_node_recursive(self, node_id: int, depth: int) -> None:
        node = self.nodes[node_id]
        if node.level >= depth:
            return
        if len(node.point_ids) <= 2:
            return

        a, b = node.interval
        mid = 0.5 * (a + b)
        left_ids = [idx for idx in node.point_ids if self.t_grid[idx] <= mid]
        right_ids = [idx for idx in node.point_ids if self.t_grid[idx] > mid]

        if not left_ids or not right_ids:
            return

        left = self._make_node(node.level + 1, (a, mid), left_ids, parent_id=node_id)
        right = self._make_node(node.level + 1, (mid, b), right_ids, parent_id=node_id)
        node.child_ids = [left.node_id, right.node_id]

        self._split_node_recursive(left.node_id, depth)
        self._split_node_recursive(right.node_id, depth)

    # ------------------------------
    # Step 4. Aggregate node stats / energy
    # ------------------------------
    def aggregate_node_statistics(self) -> None:
        for node in self.nodes.values():
            feats = np.array([self.features[idx] for idx in node.point_ids], dtype=float)
            node.signature = np.mean(feats, axis=0)

            log_abs = self.log_abs_values[node.point_ids]
            phases = self.phase_values[node.point_ids]
            vals = [self.values[idx] for idx in node.point_ids]

            node.min_log_abs = float(np.min(log_abs))
            node.mean_log_abs = float(np.mean(log_abs))
            node.var_phase = float(np.var(phases)) if len(phases) > 1 else 0.0
            node.roughness = self.local_roughness(log_abs)
            node.sign_changes = self.count_sign_changes(vals)
            node.energy = self.compute_node_energy(node)

    def compute_node_energy(self, node: FractalNode) -> float:
        sign_penalty = 0.0 if node.sign_changes >= 1 else 1.0
        energy = (
            self.cfg.w_min_logabs * node.min_log_abs
            + self.cfg.w_mean_logabs * node.mean_log_abs
            + self.cfg.w_var_phase * node.var_phase
            + self.cfg.w_roughness * node.roughness
            + self.cfg.w_sign_penalty * sign_penalty
        )
        return float(energy)

    # ------------------------------
    # Step 5. Graph
    # ------------------------------
    def build_graph(self) -> None:
        self.neighbors = {node_id: [] for node_id in self.nodes}

        # Tree edges: parent-child
        for node in self.nodes.values():
            if node.parent_id is not None:
                self.add_undirected_edge(node.node_id, node.parent_id)
            for child_id in node.child_ids:
                self.add_undirected_edge(node.node_id, child_id)

        # Same-level local adjacency in interval order
        for level, node_ids in self.nodes_by_level.items():
            ordered = sorted(node_ids, key=lambda nid: self.nodes[nid].interval[0])
            k = self.cfg.same_level_neighbor_k
            for i, nid in enumerate(ordered):
                for d in range(1, k + 1):
                    if i - d >= 0:
                        self.add_undirected_edge(nid, ordered[i - d])
                    if i + d < len(ordered):
                        self.add_undirected_edge(nid, ordered[i + d])

        # Optional feature-neighbor edges on same level
        radius = self.cfg.feature_neighbor_radius
        if radius is not None:
            for level, node_ids in self.nodes_by_level.items():
                for i in range(len(node_ids)):
                    for j in range(i + 1, len(node_ids)):
                        ni = self.nodes[node_ids[i]]
                        nj = self.nodes[node_ids[j]]
                        dist = self.signature_distance(ni, nj)
                        if dist <= radius:
                            self.add_undirected_edge(ni.node_id, nj.node_id)

    def add_undirected_edge(self, a: int, b: int) -> None:
        if b not in self.neighbors[a]:
            self.neighbors[a].append(b)
        if a not in self.neighbors[b]:
            self.neighbors[b].append(a)

    # ------------------------------
    # Step 6. Pheromones
    # ------------------------------
    def initialize_pheromones(self) -> None:
        self.pheromones.clear()
        self.pheromone_channels = {agent_type: {} for agent_type in self.ant_types}
        for a, nbrs in self.neighbors.items():
            for b in nbrs:
                self.pheromones[(a, b)] = self.cfg.tau0
                for agent_type in self.ant_types:
                    self.pheromone_channels[agent_type][(a, b)] = self.cfg.tau0

    # ------------------------------
    # Step 7. ACO main loop
    # ------------------------------
    def run_aco(self) -> None:
        if self.root_id is None:
            raise RuntimeError("Tree root is not built.")

        ants = [self.make_ant(k) for k in range(self.cfg.n_ants)]
        for _it in range(self.cfg.n_iterations):
            paths: List[AntPath] = []
            for ant in ants:
                path_nodes = self.sample_ant_path(self.root_id, ant)
                score = self.evaluate_path(path_nodes, ant.agent_type)
                paths.append(AntPath(node_ids=path_nodes, score=score, agent_type=ant.agent_type))
                self.update_ant_memory(ant, path_nodes)
                for nid in path_nodes:
                    self.nodes[nid].visit_count += 1

            self.evaporate_pheromones()
            self.reinforce_pheromones(paths)
            if self.cfg.spectral_learning:
                self.run_spectral_learning_step(_it)

    def spectral_learning_inputs(self) -> Tuple[np.ndarray, np.ndarray, List[Optional[int]]]:
        max_points = max(2, int(self.cfg.spectral_max_points))
        stride = max(1, int(math.ceil(len(self.t_grid) / max_points)))
        sample_indices = np.arange(0, len(self.t_grid), stride, dtype=int)
        t_grid = np.asarray(self.t_grid[sample_indices], dtype=float)
        zeta_abs = np.asarray(self.abs_values[sample_indices], dtype=float)

        target_level = self.cfg.target_level or self.cfg.tree_depth
        point_to_node: Dict[int, int] = {}
        for node_id in self.nodes_by_level.get(target_level, []):
            for point_id in self.nodes[node_id].point_ids:
                point_to_node[point_id] = node_id

        node_ids: List[Optional[int]] = [
            point_to_node.get(int(point_id)) for point_id in sample_indices
        ]
        return t_grid, zeta_abs, node_ids

    def build_spectral_pheromone_matrix(
        self,
        t_grid: np.ndarray,
        zeta_abs: np.ndarray,
        node_ids: List[Optional[int]],
    ) -> np.ndarray:
        N = len(t_grid)
        pheromone_matrix = np.zeros((N, N), dtype=np.float64)

        for i, ti in enumerate(t_grid):
            for j, tj in enumerate(t_grid):
                if i == j:
                    continue

                dt = abs(float(ti) - float(tj))
                if dt > 20:
                    continue

                local_weight = math.exp(-dt / 5.0)
                zi = max(float(zeta_abs[i]), 1e-12)
                zj = max(float(zeta_abs[j]), 1e-12)
                energy_weight = 1.0 / (1.0 + zi + zj)
                weight = local_weight * energy_weight

                node_i = node_ids[i]
                node_j = node_ids[j]
                if node_i is not None and node_j is not None and node_i != node_j:
                    tau = self.mixed_pheromone("modulus", node_i, node_j)
                    weight *= max(float(tau), 1e-6)

                pheromone_matrix[i, j] = weight

        pheromone_matrix = 0.5 * (pheromone_matrix + pheromone_matrix.T)
        row_sums = pheromone_matrix.sum(axis=1)
        for i in range(N):
            if row_sums[i] < 1e-12:
                pheromone_matrix[i, i] = 1.0
        return np.nan_to_num(pheromone_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    def apply_spectral_feedback(self, loss: float) -> None:
        decay = math.exp(-max(0.0, float(self.cfg.spectral_weight)) * min(float(loss), 100.0))
        for edge in list(self.pheromones):
            self.pheromones[edge] = min(
                self.cfg.tau_max,
                max(self.cfg.tau_min, self.pheromones[edge] * decay),
            )
        for channel in self.pheromone_channels.values():
            for edge in list(channel):
                channel[edge] = min(
                    self.cfg.tau_max,
                    max(self.cfg.tau_min, channel[edge] * decay),
                )

    def run_spectral_learning_step(self, iteration: int) -> None:
        from core.dtes_spectral_learning import (
            build_operator,
            compute_spectrum,
            spectral_diagnostics,
        )
        from core.dtes_trace_tools import v5_loss_components

        t_grid, zeta_abs, node_ids = self.spectral_learning_inputs()
        zeta_zeros = np.asarray(self.cfg.zeta_zeros, dtype=float)
        pheromone_matrix = self.build_spectral_pheromone_matrix(t_grid, zeta_abs, node_ids)
        H = build_operator(t_grid, zeta_abs, pheromone_matrix)
        eigvals = compute_spectrum(H, k=min(max(1, int(self.cfg.spectral_k)), 100))
        losses = v5_loss_components(eigvals, zeta_zeros, t_grid, self.zeta)
        loss = losses["total"]

        if np.isnan(loss) or np.isinf(loss):
            print("[WARN] invalid spectral loss, skipping update")
            return

        diagnostics = spectral_diagnostics(eigvals, zeta_zeros)
        corr = diagnostics["correlation"]
        corr_text = "nan" if corr is None else f"{corr:.6f}"
        align = diagnostics["best_alignment"]
        align_text = "nan" if align is None else f"{align:.6f}"
        print(
            f"[SPECTRAL LOSS] iter={iteration + 1} "
            f"spectral={losses['spectral']:.6f} "
            f"counting={losses['counting']:.6f} "
            f"determinant={losses['determinant']:.6f} "
            f"total={loss:.6f} "
            f"best_alignment={align_text} correlation={corr_text}"
        )
        self.apply_spectral_feedback(loss)

    def sample_ant_path(self, start_node_id: int, ant: DTESAnt) -> List[int]:
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
                agent_score = self.compute_agent_value(ant, current, nxt)

                # --- DTES exploration boost ---
                visit_bonus = 1.0 / (1.0 + self.nodes[nxt].visit_count) ** 0.5
                level_bonus = 0.15 * (self.nodes[nxt].level / max(1, self.cfg.tree_depth))
                agent_score += 0.75 * visit_bonus + level_bonus
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


    def make_ant(self, ant_id: int) -> DTESAnt:
        agent_type = self.ant_types[ant_id % max(1, len(self.ant_types))]
        return DTESAnt(agent_id=ant_id, agent_type=agent_type)

    def update_ant_memory(self, ant: DTESAnt, node_ids: List[int]) -> None:
        if not node_ids:
            return
        last_node = self.nodes[node_ids[-1]]
        first_node = self.nodes[node_ids[0]]
        energy_drop = first_node.energy - last_node.energy
        ant.memory_energy_drop = (
            self.cfg.agent_memory_decay * ant.memory_energy_drop
            + (1.0 - self.cfg.agent_memory_decay) * energy_drop
        )
        ant.memory_phase_pref = (
            self.cfg.agent_memory_decay * ant.memory_phase_pref
            + (1.0 - self.cfg.agent_memory_decay) * last_node.var_phase
        )
        ant.memory_stability_pref = (
            self.cfg.agent_memory_decay * ant.memory_stability_pref
            + (1.0 - self.cfg.agent_memory_decay) * self.node_stability.get(last_node.node_id, 1.0)
        )
        ant.memory_node_id = last_node.node_id

    def compute_agent_value(self, ant: DTESAnt, a_id: int, b_id: int) -> float:
        a = self.nodes[a_id]
        b = self.nodes[b_id]
        if ant.agent_type == "modulus":
            return self.cfg.agent_modulus_weight * (a.min_log_abs - b.min_log_abs)
        if ant.agent_type == "phase":
            phase_gain = -(b.var_phase + abs(a.var_phase - b.var_phase))
            memory_pull = -abs(ant.memory_phase_pref - b.var_phase)
            return self.cfg.agent_phase_weight * (phase_gain + 0.25 * memory_pull)
        if ant.agent_type == "multiscale":
            coherence = 1.0 / (1.0 + self.signature_distance(a, b))
            narrowing = 1.0 if b.width() <= a.width() else -0.5
            return self.cfg.agent_multiscale_weight * (coherence + narrowing)
        if ant.agent_type == "stability":
            stability = self.node_stability.get(b_id, 1.0)
            memory_term = 0.0 if ant.memory_node_id is None else self.interval_overlap_ratio(
                self.nodes[ant.memory_node_id].interval, b.interval
            )
            return self.cfg.agent_stability_weight * (stability + memory_term)
        return -self.compute_barrier(a_id, b_id)

    def compute_agent_path_bonus(self, nodes: List[FractalNode], agent_type: str) -> float:
        if not nodes:
            return 0.0
        if agent_type == "modulus":
            return 0.15 * (nodes[0].min_log_abs - min(n.min_log_abs for n in nodes))
        if agent_type == "phase":
            return 0.10 / (1.0 + float(np.mean([n.var_phase for n in nodes])))
        if agent_type == "multiscale":
            return 0.12 * self.path_coherence(nodes)
        if agent_type == "stability":
            stability_vals = [self.node_stability.get(n.node_id, 1.0) for n in nodes]
            return 0.18 * float(np.mean(stability_vals))
        return 0.0

    def agent_type_channel_scale(self, agent_type: str) -> float:
        scales = {
            "modulus": 1.00,
            "phase": 0.95,
            "multiscale": 1.05,
            "stability": 1.10,
        }
        return scales.get(agent_type, 1.0)

    def channel_agreement_weights(self, agent_type: str) -> Dict[str, float]:
        """Cross-channel listening weights for each DTES-agent type."""
        matrix: Dict[str, Dict[str, float]] = {
            "modulus": {"phase": 0.15, "multiscale": 0.25, "stability": 0.10},
            "phase": {"modulus": 0.10, "multiscale": 0.35, "stability": 0.15},
            "multiscale": {"modulus": 0.20, "phase": 0.25, "stability": 0.25},
            "stability": {"modulus": 0.20, "phase": 0.10, "multiscale": 0.35},
        }
        return matrix.get(agent_type, {})

    def cross_channel_pheromone(self, agent_type: str, a: int, b: int) -> float:
        weights = self.channel_agreement_weights(agent_type)
        if not weights:
            return self.cfg.tau0
        weighted_sum = 0.0
        total_weight = 0.0
        for channel_name, weight in weights.items():
            channel = self.pheromone_channels.get(channel_name)
            if channel is None:
                continue
            weighted_sum += weight * channel.get((a, b), self.cfg.tau0)
            total_weight += weight
        return weighted_sum / total_weight if total_weight > 0 else self.cfg.tau0

    def mixed_pheromone(self, agent_type: str, a: int, b: int) -> float:
        shared = self.pheromones.get((a, b), self.cfg.tau0)
        own = self.pheromone_channels.get(agent_type, {}).get((a, b), self.cfg.tau0)
        if not self.cfg.enable_cross_channel_agreement:
            return 0.55 * shared + 0.45 * own
        cross = self.cross_channel_pheromone(agent_type, a, b)
        total = self.cfg.shared_pheromone_mix + self.cfg.own_channel_mix + self.cfg.cross_channel_mix
        if total <= 0:
            return own
        return (
            self.cfg.shared_pheromone_mix * shared
            + self.cfg.own_channel_mix * own
            + self.cfg.cross_channel_mix * cross
        ) / total

    def compute_node_stability(self) -> None:
        self.node_stability = {}
        for node_id, node in self.nodes.items():
            nbrs = self.neighbors.get(node_id, [])
            if not nbrs:
                self.node_stability[node_id] = 1.0
                continue
            local_overlaps = []
            for nbr in nbrs:
                local_overlaps.append(self.interval_overlap_ratio(node.interval, self.nodes[nbr].interval))
            self.node_stability[node_id] = 1.0 / (1.0 + float(np.mean(local_overlaps)))

    @staticmethod
    def interval_overlap_ratio(a: Tuple[float, float], b: Tuple[float, float]) -> float:
        left = max(a[0], b[0])
        right = min(a[1], b[1])
        inter = max(0.0, right - left)
        union = max(a[1], b[1]) - min(a[0], b[0])
        if union <= 0:
            return 0.0
        return inter / union

    def compute_barrier(self, a_id: int, b_id: int) -> float:
        a = self.nodes[a_id]
        b = self.nodes[b_id]
        energy_term = self.cfg.lambda_energy * b.energy
        signature_term = self.cfg.lambda_signature * self.signature_distance(a, b)
        tree_term = self.cfg.lambda_tree * self.tree_distance_proxy(a, b)
        phase_term = self.cfg.lambda_phase * abs(a.var_phase - b.var_phase)
        return float(energy_term + signature_term + tree_term + phase_term)

    @staticmethod
    def tree_distance_proxy(a: FractalNode, b: FractalNode) -> float:
        # Simple local proxy: width change + center shift.
        return abs(math.log((a.width() + 1e-15) / (b.width() + 1e-15))) + abs(a.center() - b.center())

    @staticmethod
    def signature_distance(a: FractalNode, b: FractalNode) -> float:
        x = a.signature
        y = b.signature
        if x.shape != y.shape:
            raise ValueError("Signatures must have the same shape.")
        diff = np.nan_to_num(x - y, nan=0.0, posinf=1e6, neginf=-1e6)
        return float(np.linalg.norm(diff))

    def evaluate_path(self, node_ids: List[int], agent_type: str = "modulus") -> float:
        nodes = [self.nodes[nid] for nid in node_ids]
        min_energy = min(node.energy for node in nodes)
        depth_bonus = max(node.level for node in nodes) / max(1, self.cfg.target_level)
        coherence = self.path_coherence(nodes)
        oscillation_penalty = self.path_oscillation_penalty(nodes)

        agent_bonus = self.compute_agent_path_bonus(nodes, agent_type)
        score = (
            1.0 / (self.cfg.eps + math.exp(min_energy))
            + self.cfg.gamma_depth * depth_bonus
            + self.cfg.gamma_coherence * coherence
            - self.cfg.gamma_osc * oscillation_penalty
            + agent_bonus
        )
        return float(score)

    @staticmethod
    def path_coherence(nodes: List[FractalNode]) -> float:
        if len(nodes) < 2:
            return 1.0
        dists = []
        for i in range(len(nodes) - 1):
            a = nodes[i]
            b = nodes[i + 1]
            dists.append(np.linalg.norm(np.nan_to_num(a.signature - b.signature, nan=0.0)))
        mean_dist = float(np.mean(dists)) if dists else 0.0
        return 1.0 / (1.0 + mean_dist)

    @staticmethod
    def path_oscillation_penalty(nodes: List[FractalNode]) -> float:
        if len(nodes) < 3:
            return 0.0
        centers = [n.center() for n in nodes]
        changes = 0
        for i in range(1, len(centers) - 1):
            d1 = centers[i] - centers[i - 1]
            d2 = centers[i + 1] - centers[i]
            if d1 * d2 < 0:
                changes += 1
        return changes / max(1, len(nodes) - 2)

    def evaporate_pheromones(self) -> None:
        for edge in list(self.pheromones.keys()):
            self.pheromones[edge] *= (1.0 - self.cfg.rho)
            self.pheromones[edge] = min(max(self.pheromones[edge], self.cfg.tau_min), self.cfg.tau_max)
        for channel in self.pheromone_channels.values():
            for edge in list(channel.keys()):
                channel[edge] *= (1.0 - self.cfg.rho)
                channel[edge] = min(max(channel[edge], self.cfg.tau_min), self.cfg.tau_max)

    def reinforce_pheromones(self, paths: List[AntPath]) -> None:
        for ant_path in paths:
            ids = ant_path.node_ids
            channel_scale = self.agent_type_channel_scale(ant_path.agent_type)
            deposit = ant_path.score * channel_scale
            agent_channel = self.pheromone_channels[ant_path.agent_type]
            cross_weights = self.channel_agreement_weights(ant_path.agent_type) if self.cfg.enable_cross_channel_agreement else {}

            for a, b in zip(ids[:-1], ids[1:]):
                # Shared memory: global colony consensus.
                self.pheromones[(a, b)] = min(self.cfg.tau_max, self.pheromones[(a, b)] + deposit)
                self.pheromones[(b, a)] = min(self.cfg.tau_max, self.pheromones[(b, a)] + deposit)

                # Own channel: preserves specialization.
                agent_channel[(a, b)] = min(self.cfg.tau_max, agent_channel[(a, b)] + 1.25 * deposit)
                agent_channel[(b, a)] = min(self.cfg.tau_max, agent_channel[(b, a)] + 1.25 * deposit)

                # Cross-channel agreement: weakly reinforce related channels.
                for channel_name, weight in cross_weights.items():
                    channel = self.pheromone_channels.get(channel_name)
                    if channel is None:
                        continue
                    cross_deposit = self.cfg.cross_reinforcement_strength * weight * deposit
                    channel[(a, b)] = min(self.cfg.tau_max, channel[(a, b)] + cross_deposit)
                    channel[(b, a)] = min(self.cfg.tau_max, channel[(b, a)] + cross_deposit)

    # ------------------------------
    # Step 8. Rank nodes / intervals
    # ------------------------------
    def node_pheromone_mass(self, node_id: int) -> float:
        nbrs = self.neighbors.get(node_id, [])
        return float(sum(self.pheromones[(node_id, nbr)] for nbr in nbrs))

    def rank_candidate_nodes(self) -> List[int]:
        scored: List[Tuple[float, int]] = []
        for level, node_ids in self.nodes_by_level.items():
            if level < max(1, self.cfg.target_level - 1):
                continue
            for nid in node_ids:
                node = self.nodes[nid]
                pheromone_mass = self.node_pheromone_mass(nid)
                score = -node.energy + math.log(1.0 + pheromone_mass)
                scored.append((score, nid))

        scored.sort(reverse=True)
        return [nid for _, nid in scored[: self.cfg.top_candidate_nodes]]

    # ------------------------------
    # Step 9. Local refinement
    # ------------------------------
    def refine_candidates(self, candidate_nodes: List[int]) -> List[float]:
        refined: List[float] = []
        for nid in candidate_nodes:
            node = self.nodes[nid]
            t_star = self.local_refinement(node.interval)
            if t_star is not None and self.verify_zero_candidate(t_star):
                refined.append(t_star)
        return refined

    def local_refinement(self, interval: Tuple[float, float]) -> Optional[float]:
        a, b = interval
        subgrid = np.linspace(a, b, self.cfg.refinement_subgrid)
        vals = []
        for t in subgrid:
            z = mp.zeta(mp.mpf('0.5') + 1j * mp.mpf(str(float(t))))
            vals.append(abs(complex(z)))

        vals_arr = np.array(vals, dtype=float)
        i0 = int(np.argmin(vals_arr))
        t0 = float(subgrid[i0])

        # Refine by minimizing |zeta|^2 numerically via a local quadratic fit fallback.
        # Simple and robust for a skeleton; can be replaced by Brent/Newton variants.
        left = max(0, i0 - 1)
        right = min(len(subgrid) - 1, i0 + 1)
        bracket = subgrid[left:right + 1]

        best_t = t0
        best_val = float(vals_arr[i0])

        dense = np.linspace(float(bracket[0]), float(bracket[-1]), 64)
        for t in dense:
            z = mp.zeta(mp.mpf('0.5') + 1j * mp.mpf(str(float(t))))
            score = abs(complex(z))
            if score < best_val:
                best_val = float(score)
                best_t = float(t)

        if self.cfg.t_min <= best_t <= self.cfg.t_max:
            return best_t
        return None

    def verify_zero_candidate(self, t: float) -> bool:
        z = mp.zeta(mp.mpf('0.5') + 1j * mp.mpf(str(float(t))))
        abs_z = abs(complex(z))
        if abs_z >= self.cfg.verification_abs_tol:
            return False

        # Local consistency: the point should be near a local minimum of |zeta|.
        delta = max((self.cfg.t_max - self.cfg.t_min) / self.cfg.n_grid, 1e-6)
        z_left = abs(complex(mp.zeta(mp.mpf('0.5') + 1j * mp.mpf(str(float(t - delta))))))
        z_right = abs(complex(mp.zeta(mp.mpf('0.5') + 1j * mp.mpf(str(float(t + delta))))))
        return abs_z <= z_left and abs_z <= z_right

    @staticmethod
    def merge_close_candidates(candidates: List[float], tol: float = 1e-5) -> List[float]:
        if not candidates:
            return []
        xs = sorted(candidates)
        merged = [xs[0]]
        for x in xs[1:]:
            if abs(x - merged[-1]) <= tol:
                merged[-1] = 0.5 * (merged[-1] + x)
            else:
                merged.append(x)
        return merged


    def filter_candidates_to_interval(self, candidates):
        """Keep only refined candidates inside the configured search interval.

        This prevents local refinement from returning zeros outside [t_min, t_max],
        e.g. returning t≈21 when the requested interval is [100, 400].
        """
        lo = float(self.cfg.t_min)
        hi = float(self.cfg.t_max)
        filtered = []
        rejected = []
        for t in candidates:
            tf = float(t)
            if lo <= tf <= hi:
                filtered.append(tf)
            else:
                rejected.append(tf)
        if rejected:
            print(
                f"[FILTER] rejected {len(rejected)} out-of-interval candidates "
                f"outside [{lo}, {hi}]: "
                + ", ".join(f"{x:.12f}" for x in rejected[:8])
                + (" ..." if len(rejected) > 8 else "")
            )
        return filtered

def parse_args(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Run the experimental Fractal-DTES-ACO zeta search."
    )
    parser.add_argument(
        "--spectral-learning",
        action="store_true",
        help="Enable experimental DTES spectral-loss feedback into ACO pheromones.",
    )
    parser.add_argument(
        "--spectral-weight",
        type=float,
        default=0.1,
        help="Feedback strength for spectral-learning pheromone decay.",
    )
    args, _unknown = parser.parse_known_args(argv)
    return args


def default_demo(argv=None) -> None:
    args = parse_args(argv)

    # First ten nontrivial zeros are near:
    # 14.1347, 21.0220, 25.0108, 30.4248, 32.9351, ...
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
        spectral_learning=args.spectral_learning,
        spectral_weight=args.spectral_weight,
    )

    searcher = FractalDTESACOZeta(cfg)
    candidates = searcher.run()

    print("Candidates:")
    for t in candidates:
        z = mp.zeta(mp.mpf('0.5') + 1j * mp.mpf(str(float(t))))
        print(f"t = {t:.12f} |zeta| = {abs(complex(z)):.3e}")

    # === DTES spectral export (SAFE PATCH) ===
    try:
        import json
        import os
        import numpy as np

        run_dir = "runs"
        os.makedirs(run_dir, exist_ok=True)

        # --- t_grid ---
        if hasattr(searcher, "t_grid"):
            t_grid = list(map(float, searcher.t_grid))
        else:
            t_grid = []
            if hasattr(searcher, "nodes"):
                for n in searcher.nodes.values():
                    if hasattr(n, "t"):
                        t_grid.append(float(n.t))

        if len(t_grid) > 1500:
            t_grid = t_grid[:1500]

        # --- zeta_abs ---
        zeta_abs = []
        for t in t_grid:
            try:
                val = abs(mp.zeta(mp.mpf('0.5') + 1j * mp.mpf(str(float(t)))))
            except Exception:
                val = 0.0
            zeta_abs.append(float(val))

        # --- pheromone_matrix ---
        N = len(t_grid)
        pheromone_matrix = np.zeros((N, N), dtype=float)

        for i, ti in enumerate(t_grid):
            for j, tj in enumerate(t_grid):
                if i == j:
                    continue

                dt = abs(ti - tj)

                # locality kernel (critical for structure)
                local_weight = np.exp(-dt / 5.0)

                # cutoff to sparsify graph
                if dt > 20:
                    continue

                zi = zeta_abs[i]
                zj = zeta_abs[j]

                # avoid division issues
                zi = max(zi, 1e-12)
                zj = max(zj, 1e-12)

                # energy-based weighting (enhances near zeros)
                energy_weight = 1.0 / (1.0 + zi + zj)

                # combine
                pheromone_matrix[i, j] = local_weight * energy_weight

        # enforce symmetry (CRITICAL for self-adjoint operator)
        pheromone_matrix = 0.5 * (pheromone_matrix + pheromone_matrix.T)

        # ensure no empty rows (fallback safety)
        row_sums = pheromone_matrix.sum(axis=1)
        for i in range(N):
            if row_sums[i] < 1e-12:
                pheromone_matrix[i, i] = 1.0

        # --- optional zeros ---
        zeros = []
        if hasattr(searcher, "zeros"):
            zeros = [float(z) for z in searcher.zeros]
        if not zeros and hasattr(searcher.cfg, "zeta_zeros"):
            zeros = [float(z) for z in searcher.cfg.zeta_zeros]

        spectral_data = {
            "t_grid": t_grid,
            "zeta_abs": zeta_abs,
            "pheromone_matrix": pheromone_matrix.tolist(),
            "zeta_zeros": zeros,
            "spectral_ready": True
        }

        out_file = os.path.join(run_dir, "dtes_spectral_input.json")

        with open(out_file, "w") as f:
            json.dump(spectral_data, f)

        print(f"[OK] DTES spectral input saved -> {out_file}")

    except Exception as e:
        print(f"[WARN] DTES spectral export failed: {e}")


if __name__ == "__main__":
    default_demo()
