from __future__ import annotations

import cmath
import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Iterable

import numpy as np
import mpmath as mp

from core.pauli import pauli_valid


# ------------------------------------------------------------
# Fractal-DTES-ACO-Zeta: research skeleton / MVP
# ------------------------------------------------------------
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
# ``fractal_dtes_aco_zeta_crosschannel.py`` embeds a parallel copy of this pipeline for
# cross-channel experiments; keep behavioral changes in sync or consolidate via import.

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
        self.pauli_rejections = 0
        self.pauli_action_checks = 0
        self.pauli_valid_action_checks = 0

    # ------------------------------
    # Public API
    # ------------------------------
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
        return self.merge_close_candidates(candidates)

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

    def is_valid_action(self, path: List[int], action: int) -> bool:
        return pauli_valid(path + [action])

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
            legal_nbrs = []
            for nxt in nbrs:
                self.pauli_action_checks += 1
                if self.is_valid_action(path, nxt):
                    self.pauli_valid_action_checks += 1
                    legal_nbrs.append(nxt)
                else:
                    self.pauli_rejections += 1
            if not legal_nbrs:
                break

            weights = []
            for nxt in legal_nbrs:
                barrier = self.compute_barrier(current, nxt)
                tau = self.mixed_pheromone(ant.agent_type, current, nxt)
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
            chosen = legal_nbrs[-1]
            for nxt, w in zip(legal_nbrs, weights):
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

        return best_t

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


def default_demo() -> None:
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
    )

    searcher = FractalDTESACOZeta(cfg)
    candidates = searcher.run()

    print("Candidates:")
    for t in candidates:
        z = mp.zeta(mp.mpf('0.5') + 1j * mp.mpf(str(float(t))))
        print(f"t = {t:.12f} |zeta| = {abs(complex(z)):.3e}")



# ------------------------------------------------------------
# Metrics extension: monkey-patched diagnostics layer
# ------------------------------------------------------------
import json as _json

_DEFAULT_KNOWN_ZEROS = (
    14.134725141734693, 21.022039638771554,
    25.010857580145688, 30.424876125859513,
    32.935061587739189, 37.586178158825671,
)

_ORIGINAL_REINFORCE_PHEROMONES = FractalDTESACOZeta.reinforce_pheromones
_ORIGINAL_RUN = FractalDTESACOZeta.run


def _metrics_reinforce_pheromones(self, paths):
    if not hasattr(self, "last_aco_paths"):
        self.last_aco_paths = []
    self.last_aco_paths.extend(paths)
    return _ORIGINAL_REINFORCE_PHEROMONES(self, paths)


def _metrics_run(self):
    self.last_aco_paths = []
    candidates = _ORIGINAL_RUN(self)
    candidate_nodes = self.rank_candidate_nodes()
    self.metrics = self.compute_metrics(candidate_nodes, candidates)
    out_path = getattr(self.cfg, "metrics_out_path", "fractal_dtes_aco_metrics.json")
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            _json.dump(self.metrics, f, indent=2, ensure_ascii=False)
    return candidates


def _compute_metrics(self, candidate_nodes, candidates):
    zeta_abs = []
    for t in candidates:
        z = mp.zeta(mp.mpf('0.5') + 1j * mp.mpf(str(float(t))))
        zeta_abs.append(float(abs(complex(z))))
    candidate_node_stability = [self.node_stability.get(nid, 1.0) for nid in candidate_nodes]
    pauli_checks = int(getattr(self, "pauli_action_checks", 0))
    pauli_valid = int(getattr(self, "pauli_valid_action_checks", 0))
    return {
        "interval": [self.cfg.t_min, self.cfg.t_max],
        "n_grid": self.cfg.n_grid,
        "tree_depth": self.cfg.tree_depth,
        "n_iterations": self.cfg.n_iterations,
        "n_ants": self.cfg.n_ants,
        "candidate_count": len(candidates),
        "candidate_nodes_ranked": len(candidate_nodes),
        "best_candidate_t": float(candidates[int(np.argmin(zeta_abs))]) if zeta_abs else None,
        "best_candidate_abs_zeta": float(min(zeta_abs)) if zeta_abs else None,
        "mean_candidate_abs_zeta": float(np.mean(zeta_abs)) if zeta_abs else None,
        "mean_candidate_node_stability": float(np.mean(candidate_node_stability)) if candidate_node_stability else None,
        "hit_stats": self.compute_known_zero_hits(candidates),
        "agent_stats": self.compute_agent_metrics(),
        "channel_stats": self.compute_channel_metrics(),
        "ramsey_stats": self.compute_ramsey_metrics(),
        "pauli_rejections": int(getattr(self, "pauli_rejections", 0)),
        "valid_action_ratio": float(pauli_valid / pauli_checks) if pauli_checks else 1.0,
    }


def _compute_known_zero_hits(self, candidates):
    known_zeros = getattr(self.cfg, "known_zeros", _DEFAULT_KNOWN_ZEROS)
    tol = getattr(self.cfg, "known_zero_tol", 1e-3)
    known = [z for z in known_zeros if self.cfg.t_min <= z <= self.cfg.t_max]
    hits = []
    for z0 in known:
        nearest = min(candidates, key=lambda x: abs(x - z0)) if candidates else None
        err = abs(nearest - z0) if nearest is not None else None
        if err is not None and err <= tol:
            hits.append({"known_zero": float(z0), "candidate": float(nearest), "abs_error": float(err)})
    return {"known_zeros_in_interval": len(known), "hits": len(hits), "hit_rate": float(len(hits) / len(known)) if known else None, "hit_details": hits}


def _compute_agent_metrics(self):
    paths = getattr(self, "last_aco_paths", [])
    by_type = {t: [] for t in self.ant_types}
    for path in paths:
        by_type.setdefault(path.agent_type, []).append(path)
    out = {}
    terminal_nodes = []
    for agent_type, type_paths in by_type.items():
        scores = [p.score for p in type_paths]
        terminals = [p.node_ids[-1] for p in type_paths if p.node_ids]
        terminal_nodes.extend(terminals)
        out[agent_type] = {
            "paths": len(type_paths),
            "mean_score": float(np.mean(scores)) if scores else None,
            "max_score": float(np.max(scores)) if scores else None,
            "unique_terminal_nodes": len(set(terminals)),
            "mean_terminal_level": float(np.mean([self.nodes[n].level for n in terminals])) if terminals else None,
        }
    out["global_path_diversity"] = float(len(set(terminal_nodes)) / max(1, len(terminal_nodes)))
    return out


def _compute_channel_metrics(self):
    shared_values = np.array(list(self.pheromones.values()), dtype=float) if self.pheromones else np.array([])
    out = {"shared_gini": _gini(shared_values), "shared_top_10_percent_mass": _top_fraction_mass(shared_values, 0.10), "shared_entropy": _normalized_entropy(shared_values)}
    channel_vectors = {}
    all_edges = sorted(self.pheromones.keys())
    for name, channel in self.pheromone_channels.items():
        vals = np.array([channel.get(e, self.cfg.tau0) for e in all_edges], dtype=float)
        channel_vectors[name] = vals
        out[name] = {"gini": _gini(vals), "top_10_percent_mass": _top_fraction_mass(vals, 0.10), "entropy": _normalized_entropy(vals), "mean": float(np.mean(vals)) if vals.size else None, "max": float(np.max(vals)) if vals.size else None}
    alignments = {}
    names = list(channel_vectors.keys())
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            alignments[f"{a}__{b}"] = _safe_corr(channel_vectors[a], channel_vectors[b])
    out["channel_alignment_corr"] = alignments
    return out


def _compute_ramsey_metrics(self):
    target_level = self.cfg.target_level or self.cfg.tree_depth
    node_ids = sorted(self.nodes_by_level.get(target_level, []), key=lambda nid: self.nodes[nid].center())
    if not node_ids:
        return {"mean_run_length": None, "max_run_length": None, "shuffled_mean_run_length": None, "ramsey_score": None}
    colors = [self.node_color(nid) for nid in node_ids]
    runs = _run_lengths(colors)
    shuffled_means = []
    for _ in range(32):
        shuffled = list(colors)
        self.rng.shuffle(shuffled)
        shuffled_means.append(float(np.mean(_run_lengths(shuffled))))
    baseline = float(np.mean(shuffled_means)) if shuffled_means else None
    mean_run = float(np.mean(runs)) if runs else None
    return {"n_colored_nodes": len(colors), "mean_run_length": mean_run, "max_run_length": int(max(runs)) if runs else None, "shuffled_mean_run_length": baseline, "ramsey_score": float(mean_run / baseline) if mean_run is not None and baseline and baseline > 0 else None}


def _node_color(self, node_id):
    nbrs = self.neighbors.get(node_id, [])
    if not nbrs:
        return -1
    best = max(nbrs, key=lambda n: self.pheromones.get((node_id, n), self.cfg.tau0) - self.compute_barrier(node_id, n))
    if self.nodes[best].level > self.nodes[node_id].level:
        return 1
    if self.nodes[best].level < self.nodes[node_id].level:
        return -1
    return 0 if self.nodes[best].center() < self.nodes[node_id].center() else 2


def _run_lengths(colors):
    if not colors:
        return []
    runs = []
    cur = colors[0]
    length = 1
    for c in colors[1:]:
        if c == cur:
            length += 1
        else:
            runs.append(length)
            cur = c
            length = 1
    runs.append(length)
    return runs


def _gini(values):
    if values.size == 0:
        return None
    x = np.sort(np.maximum(values.astype(float), 0.0))
    total = float(np.sum(x))
    if total <= 0:
        return 0.0
    n = x.size
    return float((2.0 * np.sum((np.arange(1, n + 1)) * x) / (n * total)) - (n + 1.0) / n)


def _top_fraction_mass(values, frac):
    if values.size == 0:
        return None
    x = np.sort(np.maximum(values.astype(float), 0.0))[::-1]
    total = float(np.sum(x))
    if total <= 0:
        return 0.0
    k = max(1, int(math.ceil(frac * x.size)))
    return float(np.sum(x[:k]) / total)


def _normalized_entropy(values):
    if values.size == 0:
        return None
    x = np.maximum(values.astype(float), 0.0)
    total = float(np.sum(x))
    if total <= 0:
        return 0.0
    p = x / total
    p = p[p > 0]
    h = -float(np.sum(p * np.log(p)))
    return float(h / math.log(values.size)) if values.size > 1 else 0.0


def _safe_corr(a, b):
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return None
    if float(np.std(a)) <= 1e-15 or float(np.std(b)) <= 1e-15:
        return None
    return float(np.corrcoef(a, b)[0, 1])


FractalDTESACOZeta.reinforce_pheromones = _metrics_reinforce_pheromones
FractalDTESACOZeta.run = _metrics_run
FractalDTESACOZeta.compute_metrics = _compute_metrics
FractalDTESACOZeta.compute_known_zero_hits = _compute_known_zero_hits
FractalDTESACOZeta.compute_agent_metrics = _compute_agent_metrics
FractalDTESACOZeta.compute_channel_metrics = _compute_channel_metrics
FractalDTESACOZeta.compute_ramsey_metrics = _compute_ramsey_metrics
FractalDTESACOZeta.node_color = _node_color


if __name__ == "__main__":
    default_demo()
