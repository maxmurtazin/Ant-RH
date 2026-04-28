from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.artin_symbolic_billiard import (
    build_word,
    hyperbolic_length_from_trace,
    is_hyperbolic_matrix,
    is_primitive_heuristic,
    precompute_T_powers,
    trace_2x2,
)


_DTF = np.float64


def _safe_sinh(x: np.ndarray | float) -> np.ndarray | float:
    xa = np.asarray(x, dtype=_DTF)
    out = np.where(xa < 20.0, np.sinh(xa), 0.5 * np.exp(xa))
    return float(out) if np.ndim(x) == 0 else out


def _inv_2sinh(x: _DTF) -> _DTF:
    if x < 20.0:
        return _DTF(1.0) / (_DTF(2.0) * _DTF(math.sinh(float(x))))
    return _DTF(math.exp(-float(x)))


@dataclass
class StepInfo:
    a_list: List[int]
    trace: float
    length: float
    is_hyperbolic: bool
    primitive: bool
    terminated_reason: str


class ArtinWordEnv:
    """
    Lightweight symbolic word environment.
    Exact Selberg/operator eval is handled by training loop; env provides fast proxies.
    """

    def __init__(
        self,
        max_length: int = 8,
        max_power: int = 6,
        target_zeros_path: str = "data/zeta_zeros.txt",
        seed: int = 42,
        *,
        length_cap: float = 50.0,
        stop_probability: float = 0.1,
    ):
        self.max_length = int(max_length)
        self.max_power = int(max_power)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        self.length_cap = float(length_cap)
        self.stop_probability = float(stop_probability)

        # action space excludes 0
        neg = np.arange(-self.max_power, 0, dtype=np.int32)
        pos = np.arange(1, self.max_power + 1, dtype=np.int32)
        self.actions = np.concatenate([neg, pos], axis=0)
        self.action_dim = int(self.actions.size)

        self.T_stack, self.T_offset = precompute_T_powers(self.max_power)

        self.target_r0 = 14.134725141734693  # default first zero if file missing
        self.target_mass_scale = 0.35
        self._load_target_zeros(target_zeros_path)

        self.planner_action_prior: Optional[np.ndarray] = None
        self.planner_prior_strength: float = 0.0

        self.reset()

    def _load_target_zeros(self, path: str) -> None:
        try:
            z = np.loadtxt(path, dtype=_DTF)
            z = np.asarray(z, dtype=_DTF).reshape(-1)
            z = z[np.isfinite(z)]
            if z.size > 0:
                self.target_r0 = float(z[0])
        except Exception:
            pass

    def reset(self) -> np.ndarray:
        self.step_idx = 0
        self.prev_a = 0
        self.a_list: List[int] = []
        self.M = np.eye(2, dtype=_DTF)

        self.trace = 2.0
        self.is_hyperbolic = False
        self.length = 0.0
        self.primitive = False

        self.rolling_selberg_proxy = 0.0
        self.rolling_reward_est = 0.0
        self.done = False
        return self.get_observation()

    def set_planner_action_prior(self, prior: Optional[np.ndarray], strength: float = 0.0) -> None:
        """
        prior: optional logits-like vector shape (action_dim,)
        strength: scalar multiplier applied by caller (e.g., policy logits += strength * prior)
        """
        if prior is None:
            self.planner_action_prior = None
            self.planner_prior_strength = float(strength)
            return
        p = np.asarray(prior, dtype=np.float32).reshape(-1)
        if p.size != self.action_dim:
            self.planner_action_prior = None
            self.planner_prior_strength = float(strength)
            return
        self.planner_action_prior = p
        self.planner_prior_strength = float(strength)

    def get_planner_logits(self) -> Optional[np.ndarray]:
        return None if self.planner_action_prior is None else self.planner_action_prior.copy()

    def current_word(self) -> List[int]:
        return list(self.a_list)

    def _update_matrix(self, a: int) -> None:
        # M <- M * S * T^a
        S = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=_DTF)
        self.M = self.M @ S
        self.M = self.M @ self.T_stack[int(a) + self.T_offset]

    def _compute_flags(self) -> None:
        self.trace = float(trace_2x2(self.M))
        self.is_hyperbolic = bool(is_hyperbolic_matrix(self.M))
        if self.is_hyperbolic:
            ell = float(hyperbolic_length_from_trace(abs(self.trace)))
            self.length = ell if (np.isfinite(ell) and ell > 0.0) else 0.0
            self.primitive = bool(is_primitive_heuristic(self.a_list, self.M, self.T_stack, self.T_offset))
        else:
            self.length = 0.0
            self.primitive = False

    def _selberg_proxy(self) -> float:
        # hypothetical_geo_contribution(ℓ): ℓ/(2*sinh(ℓ/2)) * exp(-ℓ^2/(4σ^2)) * cos(r0 ℓ)
        ell = _DTF(self.length)
        if not self.is_hyperbolic or ell <= 0:
            return 1.0
        sigma = _DTF(0.6)
        inv2s = _inv_2sinh(_DTF(0.5) * ell)
        geo = ell * inv2s * _DTF(math.exp(-float((ell * ell) / (_DTF(4.0) * sigma * sigma + 1e-12))))
        geo *= _DTF(math.cos(float(self.target_r0 * float(ell))))
        target = _DTF(math.exp(-float(self.target_mass_scale * (float(ell) - float(self.target_r0)) ** 2)))
        return float(abs(float(geo) - float(target)))

    def evaluate_word(self) -> StepInfo:
        return StepInfo(
            a_list=list(self.a_list),
            trace=float(self.trace),
            length=float(self.length),
            is_hyperbolic=bool(self.is_hyperbolic),
            primitive=bool(self.primitive),
            terminated_reason="",
        )

    def get_observation(self) -> np.ndarray:
        j_norm = float(self.step_idx) / float(max(1, self.max_length))
        prev_norm = float(self.prev_a) / float(max(1, self.max_power))

        tr = float(self.trace)
        tr_norm = float(np.tanh(tr / 10.0))
        word_len = float(len(self.a_list)) / float(max(1, self.max_length))
        ell = float(self.length)
        ell_norm = float(np.tanh(ell / 20.0))
        hyp = 1.0 if self.is_hyperbolic else 0.0
        prim = 1.0 if self.primitive else 0.0
        selp = float(np.tanh(self.rolling_selberg_proxy))
        rew = float(np.tanh(self.rolling_reward_est))

        obs = np.array(
            [j_norm, prev_norm, tr_norm, word_len, ell_norm, hyp, prim, selp, rew],
            dtype=np.float32,
        )
        return obs

    def step(self, action_index: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        if self.done:
            return self.get_observation(), 0.0, True, {"reason": "already_done"}

        if action_index < 0 or action_index >= self.action_dim:
            self.done = True
            return self.get_observation(), -10.0, True, {"reason": "invalid_action_index"}

        a = int(self.actions[int(action_index)])
        self.prev_a = a
        self.a_list.append(a)
        self.step_idx += 1

        terminated_reason = ""

        try:
            self._update_matrix(a)
            self._compute_flags()
        except Exception:
            self.done = True
            terminated_reason = "numerical_error"
            info = self.evaluate_word()
            info.terminated_reason = terminated_reason
            return self.get_observation(), -10.0, True, {"info": info.__dict__}

        # penalties / bonuses (proxy-only; exact eval done externally)
        lam_sel = 1.0
        lam_len = 0.15
        lam_nonhyp = 1.0
        lam_rep = 0.25
        prim_bonus = 0.5
        stab_bonus = 0.25

        sel_proxy = self._selberg_proxy()
        self.rolling_selberg_proxy = 0.9 * self.rolling_selberg_proxy + 0.1 * sel_proxy

        nonhyp_pen = 1.0 if not self.is_hyperbolic else 0.0
        rep_pen = 1.0 if (len(self.a_list) >= 4 and self.a_list[: len(self.a_list) // 2] == self.a_list[len(self.a_list) // 2 :]) else 0.0
        len_pen = 0.0
        if self.is_hyperbolic and self.length > self.length_cap:
            len_pen = float((self.length - self.length_cap) ** 2)

        stability = 0.0
        if self.is_hyperbolic and np.isfinite(self.length) and abs(self.trace) < 1e6:
            stability = 1.0

        reward = (
            -lam_sel * float(sel_proxy)
            -lam_len * float(len_pen)
            -lam_nonhyp * float(nonhyp_pen)
            -lam_rep * float(rep_pen)
            +prim_bonus * (1.0 if self.primitive else 0.0)
            +stab_bonus * float(stability)
        )

        if not np.isfinite(reward):
            reward = -10.0
            terminated_reason = "nan_reward"
            self.done = True

        reward = float(np.clip(reward, -100.0, 100.0))
        self.rolling_reward_est = 0.9 * self.rolling_reward_est + 0.1 * reward

        # termination conditions
        done = False
        if self.step_idx >= self.max_length:
            done = True
            terminated_reason = "max_length"
        elif self.is_hyperbolic and self.primitive:
            if float(self.rng.random()) < self.stop_probability:
                done = True
                terminated_reason = "early_stop"

        if terminated_reason == "nan_reward":
            done = True

        self.done = done
        info = self.evaluate_word()
        info.terminated_reason = terminated_reason
        return self.get_observation(), reward, done, {"info": info.__dict__}

