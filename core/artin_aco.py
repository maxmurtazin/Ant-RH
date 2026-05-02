#!/usr/bin/env python3
from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse
import json
import math
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from core.dtes_spectral_triple import DTESSpectralTriple
from core.ncg_braid_spectral import (
    build_braid_operator,
    compute_heat_trace,
    compute_ncg_braid_losses,
    compute_zeta_loss,
    parse_braid_word,
    spectral_entropy,
)
from core.selberg_braid_trace import compute_selberg_braid_trace_metrics
from core.spectral_stabilization import safe_eigh
from core.artin_symbolic_billiard import (
    build_word as build_word_matrix,
    hyperbolic_length_from_trace,
    is_hyperbolic_matrix,
    precompute_T_powers,
    trace_2x2,
)
from core.artin_operator import build_geodesic_kernel, build_laplacian, sample_domain
from validation.selberg_trace_loss import compute_selberg_loss

try:
    from core.artin_operator_word_sensitive import build_word_sensitive_operator

    _HAVE_WORD_SENSITIVE = True
except Exception:
    build_word_sensitive_operator = None
    _HAVE_WORD_SENSITIVE = False


def load_zeros(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"zeros file not found: {p}")
    z = np.loadtxt(str(p), dtype=_DTF)
    z = np.asarray(z, dtype=_DTF).reshape(-1)
    z = z[np.isfinite(z)]
    return z


def compute_scheduler_effective(
    *,
    adaptive_scheduler: bool,
    stagnation_iters: int,
    scheduler_window: int,
    exploration_floor: float,
    exploration_floor_max: float,
    lambda_ncg: float,
    lambda_ncg_max: float,
    lambda_diversity: float,
    lambda_diversity_max: float,
    restart_patience: int,
    restart_patience_min: int,
) -> Tuple[float, float, float, int, float]:
    """
    V12.8 adaptive stagnation ramp (diagnostic; not an RH proof).
    Returns (exploration_floor_eff, lambda_ncg_eff, lambda_diversity_eff, restart_patience_eff, ramp_p).
    """
    if not adaptive_scheduler or int(scheduler_window) <= 0:
        return (
            float(exploration_floor),
            float(lambda_ncg),
            float(lambda_diversity),
            int(restart_patience),
            0.0,
        )
    if stagnation_iters < int(scheduler_window):
        return (
            float(exploration_floor),
            float(lambda_ncg),
            float(lambda_diversity),
            int(restart_patience),
            0.0,
        )
    excess = int(stagnation_iters) - int(scheduler_window)
    denom = float(max(1, int(scheduler_window)))
    p = min(1.0, float(excess) / denom)
    ef = min(
        float(exploration_floor_max),
        float(exploration_floor) + p * (float(exploration_floor_max) - float(exploration_floor)),
    )
    ln_cg = min(float(lambda_ncg_max), float(lambda_ncg) + p * (float(lambda_ncg_max) - float(lambda_ncg)))
    ld = min(
        float(lambda_diversity_max),
        float(lambda_diversity) + p * (float(lambda_diversity_max) - float(lambda_diversity)),
    )
    rp_f = float(restart_patience) - p * (float(restart_patience) - float(restart_patience_min))
    rp = int(max(int(restart_patience_min), int(round(rp_f))))
    return (ef, ln_cg, ld, rp, p)


def resolve_zeros_cli(z: str) -> np.ndarray:
    """
    If ``z`` is an integer string (e.g. ``32``), build synthetic ordinates for smoke tests.
    Otherwise load from path via ``load_zeros``.
    """
    s = str(z).strip()
    if s.isdigit() or (s.startswith("-") and len(s) > 1 and s[1:].isdigit()):
        n = max(1, abs(int(s)))
        base = 14.134725141734693
        return np.array([base + float(i) * 4.2 for i in range(n)], dtype=_DTF)
    return load_zeros(s)


_DTF = np.float64

# Commutator-like braid fragments from prior ACO runs (algebraic prior; not an RH proof).
SEED_MOTIFS = [
    [1, -1, 1, -1],
    [-1, 1, -1, 1],
    [-2, 2, 1, -1],
    [2, -2, 1, -1],
    [3, -3, -1, 1],
    [-4, 1, -1, 1],
    [4, -1, 1, -1],
]


def count_motif_occurrences(word: List[int], motifs: Optional[List[List[int]]] = None) -> int:
    """Count contiguous matches of any seed motif as a sublist of ``word`` (overlaps allowed)."""
    if motifs is None:
        motifs = SEED_MOTIFS
    w = [int(x) for x in word]
    total = 0
    for m in motifs:
        if not m:
            continue
        k = len(m)
        n = len(w)
        if n < k:
            continue
        for i in range(n - k + 1):
            if w[i : i + k] == m:
                total += 1
    return int(total)


def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


def clamp_pheromone(obj: Any, min_val: float, max_val: float) -> None:
    """
    In-place clamp of pheromone deposits (dict keyed edges, list, ndarray, or torch.Tensor).
    """
    lo = float(min_val)
    hi = float(max_val)
    if isinstance(obj, dict):
        for k in list(obj.keys()):
            try:
                obj[k] = float(min(max(float(obj[k]), lo), hi))
            except (TypeError, ValueError):
                continue
        return
    if isinstance(obj, list):
        for i in range(len(obj)):
            try:
                obj[i] = float(min(max(float(obj[i]), lo), hi))
            except (TypeError, ValueError):
                continue
        return
    try:
        import torch

        if isinstance(obj, torch.Tensor):
            with torch.no_grad():
                obj.clamp_(min=lo, max=hi)
            return
    except Exception:
        pass
    if isinstance(obj, np.ndarray):
        np.clip(obj, lo, hi, out=obj)


def _stable_softmax(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=_DTF)
    x = x - np.max(x)
    ex = np.exp(np.clip(x, -60.0, 60.0))
    s = ex.sum()
    if s <= 0.0 or not np.isfinite(s):
        return np.full_like(ex, 1.0 / ex.size)
    return ex / s


def _word_key(a_list: List[int]) -> Tuple[int, ...]:
    return tuple(int(a) for a in a_list)


# ---------------------------------------------------------------------------
# NCG-inspired anti-collapse regularizers (NOT a proof of RH; experimental operator
# search prior to discourage geodesic/spectrum collapse in the ACO loop).
# ---------------------------------------------------------------------------


def _build_diagonal_probe(a_list: List[int], n: int, max_power: int) -> np.ndarray:
    """Diagonal algebra probe A from candidate Artin letters (normalized)."""
    if n <= 0:
        return np.zeros((0, 0), dtype=_DTF)
    mp = float(max(1, int(max_power)))
    vals = np.zeros(n, dtype=_DTF)
    Lw = len(a_list)
    for i in range(n):
        if Lw > 0:
            j = i % Lw
            vals[i] = abs(float(int(a_list[j]))) / mp
        else:
            vals[i] = float(i + 1) / float(max(n, 1))
    nv = float(np.linalg.norm(vals))
    if nv > 1e-15:
        vals = vals / nv
    return np.diag(vals).astype(_DTF, copy=False)


def _commutator_collapse_loss(
    H: np.ndarray,
    a_list: List[int],
    *,
    comm_eps: float,
    max_power: int,
    warned: List[bool],
) -> Tuple[float, float]:
    """
    comm = H @ A - A @ H with diagonal probe A; loss = 1 / (comm_eps + ||comm||_F).
    Returns (loss, comm_norm). On failure returns (0.0, 0.0) and warns once.
    """
    try:
        H = np.asarray(H, dtype=_DTF)
        if H.ndim != 2 or H.shape[0] != H.shape[1]:
            raise ValueError("H not square")
        n = int(H.shape[0])
        A = _build_diagonal_probe(list(a_list), n, max_power)
        if A.shape != H.shape:
            raise ValueError("shape mismatch")
        comm = H @ A - A @ H
        comm_norm = float(np.linalg.norm(comm, ord="fro"))
        loss = 1.0 / (float(comm_eps) + comm_norm)
        return (float(loss), comm_norm)
    except Exception:
        if not warned[0]:
            warnings.warn(
                "NCG commutator probe skipped (operator unavailable or incompatible); using 0.",
                stacklevel=2,
            )
            warned[0] = True
        return (0.0, 0.0)


def _spectral_diversity_penalty(
    eig_sorted: np.ndarray,
    prior_spectra: List[np.ndarray],
    *,
    diversity_sigma: float,
) -> float:
    """Mean Gaussian kernel similarity to prior spectra (penalize overlap)."""
    if eig_sorted.size == 0 or len(prior_spectra) < 1:
        return 0.0
    sig = float(max(float(diversity_sigma), 1e-12))
    acc: List[float] = []
    s = np.asarray(eig_sorted, dtype=_DTF).reshape(-1)
    for sj in prior_spectra:
        t = np.asarray(sj, dtype=_DTF).reshape(-1)
        m = int(min(s.size, t.size))
        if m <= 0:
            continue
        d = s[:m] - t[:m]
        acc.append(math.exp(-float(np.dot(d, d)) / (sig * sig)))
    if not acc:
        return 0.0
    return float(np.mean(acc))


def _length_collapse_penalty(ell: float, target_length: float) -> float:
    return float(max(0.0, float(target_length) - float(ell)) ** 2)


def normalize_braid_token(tok: str) -> str:
    tok = str(tok).strip()
    mapping = {
        "sigma1": "s1+",
        "sigma1^-1": "s1-",
        "sigma2": "s2+",
        "sigma2^-1": "s2-",
        "sigma3": "s3+",
        "sigma3^-1": "s3-",
        "s1+": "s1+",
        "s1-": "s1-",
        "s2+": "s2+",
        "s2-": "s2-",
        "s3+": "s3+",
        "s3-": "s3-",
    }
    return mapping.get(tok, tok)


def candidate_to_braid_words(candidate: Any, *, n_strands: int = 4) -> List[str]:
    """
    Map an ACO candidate (Artin ints or braid-like tokens) to NCG parser tokens (s1+, s2-, ...).
    """
    if isinstance(candidate, str):
        return [normalize_braid_token(t) for t in candidate.split() if str(t).strip()]
    if isinstance(candidate, (list, tuple)):
        if not candidate:
            return ["e"]
        if isinstance(candidate[0], str):
            return [normalize_braid_token(t) for t in candidate]
        if isinstance(candidate[0], int):
            m = max(1, int(n_strands) - 1)
            out: List[str] = []
            for a in candidate:
                try:
                    ai = int(a)
                except Exception:
                    continue
                if ai == 0:
                    continue
                sign = 1 if ai > 0 else -1
                idx = 1 + (abs(ai) - 1) % m
                idx = min(idx, m)
                out.append(f"s{idx}{'+' if sign > 0 else '-'}")
            return out if out else ["e"]
    return ["e"]


def _is_valid_action_word(word: List[int], max_length: int, max_power: int) -> bool:
    if not isinstance(word, list):
        return False
    if len(word) < 3 or len(word) > int(max_length):
        return False
    vals: List[int] = []
    for a in word:
        try:
            ai = int(a)
        except Exception:
            return False
        if ai == 0 or abs(ai) > int(max_power):
            return False
        vals.append(ai)
    return True


@dataclass(frozen=True)
class Candidate:
    a_list: Tuple[int, ...]
    length: float
    trace: float
    reward: float
    loss: float


class ArtinACO:
    def __init__(
        self,
        num_ants: int,
        max_length: int,
        max_power: int,
        alpha: float,
        beta: float,
        rho: float,
        seed: int,
        *,
        length_threshold: float = 50.0,
        tau_min: float = 1e-6,
        tau_max: float = 1e6,
        bank_size: int = 1000,
        best_k_ants: int = 8,
        q: float = 1.0,
    ):
        self.num_ants = int(num_ants)
        self.max_length = int(max_length)
        self.max_power = int(max_power)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.rho = float(rho)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        self.length_threshold = float(length_threshold)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.bank_size = int(bank_size)
        self.best_k_ants = int(max(1, best_k_ants))
        self.q = float(q)

        self.a_vals = np.array(
            [a for a in range(-self.max_power, self.max_power + 1) if a != 0],
            dtype=np.int32,
        )
        self.heuristic_vals = (1.0 / (1.0 + np.abs(self.a_vals).astype(_DTF))).astype(_DTF)
        self._heur_log = np.log(np.maximum(self.heuristic_vals, 1e-18))

        self.pheromone: Dict[Tuple[int, int], float] = {}

        self.best_words: List[List[int]] = []
        self.best_loss: float = float("inf")

        self._T_stack, self._T_offset = precompute_T_powers(self.max_power)

        self._word_cache: Dict[Tuple[int, ...], Tuple[bool, float, float]] = {}
        self._bank: Dict[Tuple[int, ...], Candidate] = {}
        self._op_cache: Dict[Tuple[Tuple[int, ...], ...], float] = {}
        self.best_ncg_export: Optional[Dict[str, Any]] = None

    def _assign_rewards(
        self,
        scored: List[Candidate],
        *,
        reward_mode: str,
        loss_clip: float,
        rank_temperature: float,
    ) -> List[Candidate]:
        mode = str(reward_mode).lower()
        clip_value = float(loss_clip)
        temp = max(float(rank_temperature), 1e-8)
        updated: List[Candidate] = []
        for idx, c in enumerate(scored, start=1):
            safe_loss = float(c.loss)
            if not np.isfinite(safe_loss):
                safe_loss = clip_value
            safe_loss = min(safe_loss, clip_value)
            if mode == "inverse":
                reward = 1.0 / (safe_loss + 1e-8)
            elif mode == "raw":
                # Continuous reward directly from loss (no normalization).
                reward = -safe_loss
            elif mode == "soft_rank":
                reward = float(math.exp(-float(idx) / temp))
            else:
                reward = 1.0 / float(idx)
            updated.append(
                Candidate(
                    a_list=c.a_list,
                    length=float(c.length),
                    trace=float(c.trace),
                    reward=float(reward),
                    loss=float(safe_loss),
                )
            )
        return updated

    def _assign_adaptive_rewards(
        self,
        scored: List[Candidate],
        *,
        adaptive_reward_window: int,
        adaptive_reward_eps: float,
        adaptive_reward_clip: float,
    ) -> Tuple[List[Candidate], Dict[str, float]]:
        """
        Compute rewards over the whole candidate population (not per-candidate independently),
        using robust scaling based on the current iteration's loss distribution.
        """
        losses = np.asarray([float(c.loss) for c in scored], dtype=_DTF)
        finite_losses = losses[np.isfinite(losses)]

        # NOTE: adaptive_reward_window is accepted for CLI/API stability, but the scaling
        # is intentionally computed per-iteration from the candidate population only.
        _ = int(adaptive_reward_window)

        if finite_losses.size < 2:
            rewards = np.zeros_like(losses, dtype=_DTF)
            med = float("nan")
            iqr = float("nan")
            std = float("nan")
        else:
            med = float(np.median(finite_losses))
            q25 = float(np.percentile(finite_losses, 25))
            q75 = float(np.percentile(finite_losses, 75))
            iqr = float(q75 - q25)
            std = float(np.std(finite_losses))
            scale = float(max(iqr, std, float(adaptive_reward_eps)))
            rewards = -(losses - med) / scale
            rewards = np.clip(rewards, -float(adaptive_reward_clip), float(adaptive_reward_clip))

        updated: List[Candidate] = []
        for c, r in zip(scored, rewards.tolist()):
            updated.append(
                Candidate(
                    a_list=c.a_list,
                    length=float(c.length),
                    trace=float(c.trace),
                    reward=float(r),
                    loss=float(c.loss),
                )
            )

        rep = {
            "loss_median": float(med),
            "loss_iqr": float(iqr),
            "loss_std": float(std),
            "reward_min": float(np.min(rewards)) if rewards.size else float("nan"),
            "reward_max": float(np.max(rewards)) if rewards.size else float("nan"),
            "reward_mean": float(np.mean(rewards)) if rewards.size else float("nan"),
        }
        return updated, rep

    def _tau(self, prev_a: int, next_a: int) -> float:
        return float(self.pheromone.get((int(prev_a), int(next_a)), 1.0))

    def _set_tau(self, prev_a: int, next_a: int, value: float) -> None:
        self.pheromone[(int(prev_a), int(next_a))] = _clip(value, self.tau_min, self.tau_max)

    def _sample_random_word_of_length(self, L: int) -> List[int]:
        L = int(L)
        if L <= 0:
            return []
        L = min(L, self.max_length)
        alpha_use = float(getattr(self, "_alpha_eff", self.alpha))
        beta_use = float(getattr(self, "_beta_eff", self.beta))
        ef = float(getattr(self, "_exploration_floor", 0.0))
        a1 = int(self.rng.choice(self.a_vals))
        word = [a1]
        prev = a1
        for _ in range(L - 1):
            taus = np.empty(self.a_vals.size, dtype=_DTF)
            for i, a in enumerate(self.a_vals):
                taus[i] = self._tau(prev, int(a))
            taus = np.clip(taus, self.tau_min, self.tau_max)
            logits = alpha_use * np.log(taus) + beta_use * self._heur_log
            probs = _stable_softmax(logits)
            if ef > 0.0:
                probs = np.asarray(probs, dtype=_DTF) + ef
                s = float(np.sum(probs))
                if s > 0.0 and np.isfinite(s):
                    probs = probs / s
            idx = int(self.rng.choice(self.a_vals.size, p=probs))
            nxt = int(self.a_vals[idx])
            word.append(nxt)
            prev = nxt
        return word

    def sample_word(self) -> List[int]:
        L = int(self.rng.integers(3, self.max_length + 1))
        return self._sample_random_word_of_length(L)

    def _maybe_seed_mutate(
        self,
        word: List[int],
        *,
        use_seed_motifs: bool,
        seed_mutation_prob: float,
    ) -> List[int]:
        if not use_seed_motifs or self.rng.random() >= float(seed_mutation_prob):
            return list(word)
        w = list(word)
        motif = list(SEED_MOTIFS[int(self.rng.integers(0, len(SEED_MOTIFS)))])
        m2 = motif[:2]
        if not m2:
            return w
        if len(w) == 0:
            idx = 0
        else:
            idx = int(self.rng.integers(0, len(w)))
        w[idx:idx] = m2
        return w[: self.max_length]

    def _boost_motif_pheromone(self) -> None:
        for motif in SEED_MOTIFS:
            for i in range(len(motif) - 1):
                key = (int(motif[i]), int(motif[i + 1]))
                v = float(self.pheromone.get(key, 1.0)) * 1.05
                self.pheromone[key] = _clip(v, self.tau_min, self.tau_max)

    def _pheromone_restart_blend(self, restart_fraction: float) -> None:
        """Blend pheromone toward uniform baseline (mean edge strength)."""
        if not self.pheromone:
            return
        rf = _clip(float(restart_fraction), 0.0, 1.0)
        vals = [float(v) for v in self.pheromone.values() if np.isfinite(float(v))]
        u = float(np.mean(vals)) if vals else 1.0
        if not np.isfinite(u):
            u = 1.0
        for k in list(self.pheromone.keys()):
            try:
                old = float(self.pheromone[k])
            except (TypeError, ValueError):
                continue
            self.pheromone[k] = (1.0 - rf) * old + rf * u
        clamp_pheromone(self.pheromone, self.tau_min, self.tau_max)

    def _prune_bank(self, bank_prune_fraction: float) -> Tuple[int, int]:
        """Keep lowest-loss fraction of bank (reward-aware tie-break)."""
        items = list(self._bank.values())
        old_n = len(items)
        if old_n == 0:
            return (0, 0)
        frac = float(bank_prune_fraction)
        target = max(1, int(math.floor(old_n * frac)))
        items.sort(key=lambda c: (float(c.loss), -float(c.reward)))
        self._bank = {c.a_list: c for c in items[:target]}
        return (old_n, len(self._bank))

    def _validate_and_length(self, a_list: List[int]) -> Tuple[bool, float, float]:
        if not _is_valid_action_word(a_list, self.max_length, self.max_power):
            return (False, float("nan"), float("nan"))
        key = _word_key(a_list)
        cached = self._word_cache.get(key)
        if cached is not None:
            return cached

        M = build_word_matrix(list(key), self._T_stack, self._T_offset)
        tr = trace_2x2(M)
        if not is_hyperbolic_matrix(M):
            out = (False, float("nan"), float(tr))
            self._word_cache[key] = out
            return out

        ell = hyperbolic_length_from_trace(abs(tr))
        if not np.isfinite(ell) or ell <= 0.0 or ell > self.length_threshold:
            out = (False, float("nan"), float(tr))
            self._word_cache[key] = out
            return out

        out = (True, float(ell), float(tr))
        self._word_cache[key] = out
        return out

    def _bank_top_geodesics(self, top_k: int) -> List[Dict[str, Any]]:
        if not self._bank:
            return []
        items = list(self._bank.values())
        items.sort(key=lambda c: (-c.reward, c.loss))
        out: List[Dict[str, Any]] = []
        for c in items[: int(top_k)]:
            out.append({"a_list": list(c.a_list), "length": float(c.length), "is_hyperbolic": True, "primitive": True})
        return out

    def _operator_spectral_loss(
        self,
        zeros: np.ndarray,
        *,
        n_points: int,
        op_sigma: float,
        op_eps: float,
        top_k_geodesics: int,
        operator_builder: str,
        geo_weight: float,
        geo_sigma: float,
        potential_weight: float,
        seed: int,
    ) -> float:
        geodesics = self._bank_top_geodesics(top_k_geodesics)
        if not geodesics:
            return float("inf")

        sig = (
            str(operator_builder).lower(),
            float(op_sigma),
            float(op_eps),
            float(geo_weight),
            float(geo_sigma),
            float(potential_weight),
            tuple(tuple(int(a) for a in g["a_list"]) for g in geodesics),
        )
        cached = self._op_cache.get(sig)
        if cached is not None:
            return cached

        Z = sample_domain(int(n_points), seed=int(seed))
        builder = str(operator_builder).lower()
        if builder == "word_sensitive" and _HAVE_WORD_SENSITIVE and build_word_sensitive_operator is not None:
            # Word-sensitive builder explicitly depends on word entries/signs/etc.
            H, _ = build_word_sensitive_operator(
                z_points=Z,
                distances=None,
                geodesics=geodesics,
                eps=float(op_eps),
                geo_sigma=float(geo_sigma),
                kernel_normalization="max",
                laplacian_weight=1.0,
                geo_weight=float(geo_weight),
                potential_weight=float(potential_weight),
            )
            H = np.asarray(H, dtype=_DTF, copy=False)
        else:
            # Legacy baseline: graph Laplacian + geodesic kernel.
            L, _, _ = build_laplacian(Z, eps=float(op_eps))
            Kmat, used = build_geodesic_kernel(Z, geodesics, sigma=float(op_sigma))
            if used <= 0:
                self._op_cache[sig] = float("inf")
                return float("inf")
            H = -L + Kmat
            H = (H + H.T) * 0.5

        eigvals = np.linalg.eigh(H, UPLO="U")[0].astype(_DTF, copy=False)
        eigvals.sort()

        if zeros.size < eigvals.size:
            loss = float("inf")
        else:
            target = zeros[: eigvals.size].astype(_DTF, copy=False)
            loss = float(np.mean((eigvals - target) ** 2))

        self._op_cache[sig] = loss
        if len(self._op_cache) > 64:
            for k in list(self._op_cache.keys())[:16]:
                self._op_cache.pop(k, None)
        return loss

    def _selberg_loss(self, zeros: np.ndarray, *, sigma: float, m_max: int, bank_top_n: int) -> float:
        if not self._bank:
            return float("inf")
        items = list(self._bank.values())
        items.sort(key=lambda c: (-c.reward, c.loss))
        lengths = np.array([c.length for c in items[: int(bank_top_n)]], dtype=_DTF)
        if lengths.size == 0:
            return float("inf")
        return float(compute_selberg_loss(lengths, zeros, sigma=float(sigma), m_max=int(m_max)))

    def evaluate_iteration(
        self,
        candidates: List[Tuple[List[int], float, float]],
        zeros: np.ndarray,
        *,
        lambda_selberg: float,
        lambda_spec: float,
        lambda_spacing: float,
        lambda_zeta: float = 0.0,
        selberg_sigma: float,
        selberg_m_max: int,
        selberg_bank_top_n: int,
        n_points: int,
        op_sigma: float,
        op_eps: float,
        op_top_k_geodesics: int,
        operator_builder: str = "word_sensitive",
        geo_weight: float = 10.0,
        geo_sigma: float = 0.6,
        potential_weight: float = 0.25,
        ncg_dim: int = 128,
        ncg_n_strands: int = 4,
        ncg_max_word_len: int = 8,
        ncg_diagonal_growth: str = "exp",
        ncg_growth_alpha: float = 0.15,
        ncg_edge_scale: float = 0.05,
        ncg_spectrum_scale: float = 1.0,
        ncg_dtype: str = "float64",
        ncg_device: str = "cpu",
        use_ncg_braid: bool = False,
        lambda_ncg: float = 0.0,
        lambda_ncg_spectral: float = 1.0,
        lambda_ncg_selfadjoint: float = 0.01,
        lambda_ncg_commutator: float = 0.001,
        lambda_ncg_spacing: float = 0.05,
        lambda_ncg_zeta: float = 0.01,
        ncg_target_zeros: Optional[np.ndarray] = None,
        planner_braid_words: Optional[List[Any]] = None,
        reward_mode: str,
        loss_clip: float,
        rank_temperature: float,
        adaptive_reward_window: int = 50,
        adaptive_reward_eps: float = 1e-8,
        adaptive_reward_clip: float = 5.0,
        normalize_loss_components: bool = True,
        component_clip: float = 100.0,
        component_log_scale: bool = True,
        use_selberg_trace: bool = False,
        lambda_trace: float = 0.0,
        trace_sigma: float = 0.75,
        trace_max_cycles: int = 16,
        lambda_comm: float = 0.0,
        lambda_diversity: float = 0.0,
        lambda_length: float = 0.0,
        comm_eps: float = 1e-6,
        diversity_sigma: float = 1.0,
        target_length: float = 4.0,
        lambda_dtes_triple: float = 0.0,
        lambda_spectral_div: float = 0.0,
        ncg_eps: float = 1e-6,
        spec_clip: float = 100.0,
        reject_nonfinite_spec: bool = False,
    ) -> Tuple[List[Candidate], Dict[str, float]]:
        # Update bank with newly found valid geodesics (placeholder losses/rewards for ranking)
        for a_list, ell, tr in candidates:
            key = _word_key(a_list)
            if key in self._bank:
                continue
            self._bank[key] = Candidate(a_list=key, length=float(ell), trace=float(tr), reward=0.0, loss=float("inf"))

        # Keep bank bounded (by shortest lengths initially)
        if len(self._bank) > self.bank_size * 2:
            bank_items = list(self._bank.values())
            bank_items.sort(key=lambda c: c.length)
            self._bank = {c.a_list: c for c in bank_items[: self.bank_size]}

        # Compute global losses using current bank (raw components)
        L_selberg = self._selberg_loss(zeros, sigma=selberg_sigma, m_max=selberg_m_max, bank_top_n=selberg_bank_top_n)
        L_spec_raw = self._operator_spectral_loss(
            zeros,
            n_points=n_points,
            op_sigma=op_sigma,
            op_eps=op_eps,
            top_k_geodesics=op_top_k_geodesics,
            operator_builder=str(operator_builder),
            geo_weight=float(geo_weight),
            geo_sigma=float(geo_sigma),
            potential_weight=float(potential_weight),
            seed=self.seed,
        )
        spec_cf = float(spec_clip)
        spec_clip_iteration = 0
        if not np.isfinite(L_spec_raw):
            if bool(reject_nonfinite_spec):
                nan_v = float("nan")
                return [], {
                    "L_spec_raw": float(L_spec_raw),
                    "L_spec_eff": nan_v,
                    "L_spec_clipped": 0,
                    "spec_clip_iteration": 0,
                    "spec_nonfinite_rejected": 1,
                    "L_selberg_raw": float(L_selberg),
                    "L_spacing_raw": 0.0,
                    "L_selberg_norm": nan_v,
                    "L_spec_norm": nan_v,
                    "L_spacing_norm": nan_v,
                    "L_zeta_raw": 0.0,
                    "L_zeta_norm": nan_v,
                    "L_total_global": nan_v,
                    "heat_trace": nan_v,
                    "spectral_entropy": nan_v,
                }
            L_spec = float(spec_cf)
            spec_clip_iteration = 1
        else:
            L_spec = min(float(L_spec_raw), float(spec_cf))
            if float(L_spec_raw) > float(spec_cf):
                spec_clip_iteration = 1

        L_spacing = 0.0
        if lambda_spacing != 0.0:
            L_spacing = 0.0

        L_zeta_raw = 0.0
        heat_trace_ncg = float("nan")
        spectral_entropy_ncg = float("nan")
        if float(lambda_zeta) != 0.0 and candidates:
            try:
                word_lists = [list(w) for w, _, _ in candidates]
                if planner_braid_words:
                    word_lists = word_lists + [w for w in planner_braid_words if w is not None]
                H_ncg = build_braid_operator(
                    words=word_lists,
                    dim=int(ncg_dim),
                    device=str(ncg_device),
                    n_strands=int(ncg_n_strands),
                    max_word_len=int(ncg_max_word_len),
                    diagonal_growth=str(ncg_diagonal_growth),
                    growth_alpha=float(ncg_growth_alpha),
                    edge_scale=float(ncg_edge_scale),
                    spectrum_scale=float(ncg_spectrum_scale),
                    dtype=str(ncg_dtype),
                )
                H_np = H_ncg.detach().cpu().numpy()
                eigs_np, _, _ = safe_eigh(H_np, return_eigenvectors=False)
                zt = compute_zeta_loss(eigs_np, zeros)
                L_zeta_raw = float(zt.detach().cpu().item())
                heat_trace_ncg = float(compute_heat_trace(H_ncg, t=0.1).detach().cpu().item())
                e_t = torch.from_numpy(np.asarray(eigs_np, dtype=np.float64)).reshape(-1)
                spectral_entropy_ncg = float(spectral_entropy(e_t).detach().cpu().item())
            except Exception:
                L_zeta_raw = float("inf")
                heat_trace_ncg = float("nan")
                spectral_entropy_ncg = float("nan")

        def _to_bool(x: Any) -> bool:
            return str(x).lower() in ["1", "true", "yes", "y"]

        def _norm_component(x: Any, clip: float = 100.0, log_scale: Any = True) -> float:
            if x is None or not np.isfinite(float(x)):
                return float(clip)
            v = min(abs(float(x)), float(clip))
            if _to_bool(log_scale):
                return float(np.log1p(v))
            return float(v)

        if _to_bool(normalize_loss_components):
            L_selberg_used = _norm_component(L_selberg, float(component_clip), component_log_scale)
            L_spec_used = _norm_component(L_spec, float(component_clip), component_log_scale)
            L_spacing_used = _norm_component(L_spacing, float(component_clip), component_log_scale)
            L_zeta_used = _norm_component(L_zeta_raw, float(component_clip), component_log_scale)
        else:
            L_selberg_used = float(L_selberg)
            L_spec_used = float(L_spec)
            L_spacing_used = float(L_spacing)
            L_zeta_used = float(L_zeta_raw)

        L_total_global = (
            float(lambda_selberg) * float(L_selberg_used)
            + float(lambda_spec) * float(L_spec_used)
            + float(lambda_spacing) * float(L_spacing_used)
            + float(lambda_zeta) * float(L_zeta_used)
        )

        # For logging: "norm" fields reflect what was actually used in the sum.
        L_sel_norm = float(L_selberg_used)
        L_spec_norm = float(L_spec_used)
        L_spacing_norm = float(L_spacing_used)
        L_zeta_norm = float(L_zeta_used)

        if not np.isfinite(float(L_total_global)):
            L_total_global = float(loss_clip)

        tz_ncg = ncg_target_zeros
        if tz_ncg is None or np.asarray(tz_ncg).size == 0:
            tz_ncg = zeros
        tz_ncg = np.asarray(tz_ncg, dtype=_DTF).reshape(-1)
        tz_ncg = tz_ncg[np.isfinite(tz_ncg)]

        per_ncg: Dict[Tuple[int, ...], Dict[str, float]] = {}
        per_trace: Dict[Tuple[int, ...], Dict[str, float]] = {}
        scored: List[Candidate] = []

        lc = float(lambda_comm)
        # Spectral-diversity weight: --lambda_diversity + --lambda_spectral_div (same penalty).
        ld = float(lambda_diversity) + float(lambda_spectral_div)
        ll = float(lambda_length)
        ldt = float(lambda_dtes_triple)
        ln = float(lambda_ncg)
        # DTES collapse weight: explicit --lambda_dtes_triple, else --lambda_ncg (NCG-inspired; not RH).
        dtes_w = float(ldt) if float(ldt) != 0.0 else float(ln)
        # DTES (A,H,D) triple regularizer: finite-dim NCG-inspired anti-collapse; not an RH proof.
        build_h_reg = (lc > 0.0 or ld > 0.0 or ldt > 0.0 or ln > 0.0)
        warned_comm: List[bool] = [False]
        warned_dtes: List[bool] = [False]
        spectra_accum: List[np.ndarray] = []
        L_comm_terms: List[float] = []
        L_div_terms: List[float] = []
        L_len_terms: List[float] = []
        comm_norms: List[float] = []
        div_pen_raw: List[float] = []
        len_pen_raw: List[float] = []
        dtes_comm_norms: List[float] = []
        dtes_ncg_losses: List[float] = []
        dtes: Optional[DTESSpectralTriple] = None
        if dtes_w > 0.0:
            dtes = DTESSpectralTriple(int(ncg_dim), str(ncg_device), dtype=str(ncg_dtype))

        # Per-candidate total (when regularizers enabled):
        #   total_loss = base_loss
        #     + lambda_comm * commutator_collapse_loss
        #     + lambda_diversity * spectral_diversity_penalty
        #     + lambda_length * length_collapse_penalty
        # (plus existing NCG braid / Selberg-trace / DTES terms when those flags are on).
        for cand_idx, (a_list, ell, tr) in enumerate(candidates):
            reg = 1e-3 * float(ell)
            base_loss = float(L_total_global + reg)
            key_w = _word_key(a_list)
            logged_comm = False
            logged_div = False

            len_pen_val = 0.0
            if ll > 0.0:
                len_pen_val = _length_collapse_penalty(float(ell), float(target_length))
                len_pen_raw.append(len_pen_val)
                L_len_terms.append(ll * len_pen_val)
            else:
                len_pen_raw.append(0.0)

            loss = base_loss + (ll * len_pen_val if ll > 0.0 else 0.0)

            build_h = bool(use_ncg_braid) or bool(use_selberg_trace) or build_h_reg
            if build_h:
                try:
                    toks = candidate_to_braid_words(list(a_list), n_strands=int(ncg_n_strands))
                    phrase = " ".join(toks) if toks else "e"
                    if phrase.strip() and phrase.strip() != "e":
                        try:
                            bw = parse_braid_word(phrase)
                        except Exception:
                            bw = tuple()
                    else:
                        bw = tuple()

                    H_ncg, braid_basis, cfg_ncg = build_braid_operator(
                        words=[phrase],
                        n_strands=int(ncg_n_strands),
                        dim=int(ncg_dim),
                        max_word_len=int(ncg_max_word_len),
                        diagonal_growth=str(ncg_diagonal_growth),
                        growth_alpha=float(ncg_growth_alpha),
                        edge_scale=float(ncg_edge_scale),
                        spectrum_scale=float(ncg_spectrum_scale),
                        dtype=str(ncg_dtype),
                        device=str(ncg_device),
                        return_basis=True,
                    )
                    if bool(use_ncg_braid):
                        ncg_d = compute_ncg_braid_losses(
                            H_ncg,
                            tz_ncg,
                            braid_words=braid_basis,
                            cfg=cfg_ncg,
                            spectral_weight=float(lambda_ncg_spectral),
                            selfadjoint_weight=float(lambda_ncg_selfadjoint),
                            commutator_weight=float(lambda_ncg_commutator),
                            spacing_weight=float(lambda_ncg_spacing),
                            zeta_weight=float(lambda_ncg_zeta),
                        )
                        per_ncg[key_w] = ncg_d
                        loss = float(loss) + float(lambda_ncg) * float(ncg_d["loss"])
                    if bool(use_selberg_trace):
                        tr_d = compute_selberg_braid_trace_metrics(
                            H_ncg,
                            bw,
                            float(ell),
                            sigma=float(trace_sigma),
                            max_cycles=int(trace_max_cycles),
                        )
                        per_trace[key_w] = tr_d
                        loss = float(loss) + float(lambda_trace) * float(tr_d["trace_loss"])

                    if dtes is not None:
                        try:
                            D = dtes.dirac_from_operator(H_ncg)
                            A = dtes.algebra_probe_from_word(list(a_list))
                            if tuple(A.shape) != tuple(D.shape):
                                raise ValueError("DTESSpectralTriple: A and D shape mismatch")
                            cn = float(dtes.commutator_norm(D, A).detach().cpu().item())
                            ncg_loss_val = float(
                                dtes.ncg_collapse_loss(D, A, float(ncg_eps)).detach().cpu().item()
                            )
                            loss = float(loss) + float(dtes_w) * ncg_loss_val
                            dtes_comm_norms.append(cn)
                            dtes_ncg_losses.append(ncg_loss_val)
                        except Exception:
                            if not warned_dtes[0]:
                                warnings.warn(
                                    "DTES spectral triple: operator unavailable or incompatible; "
                                    "skipping NCG collapse term.",
                                    UserWarning,
                                    stacklevel=2,
                                )
                                warned_dtes[0] = True

                    if build_h_reg:
                        H_np = np.asarray(H_ncg.detach().cpu().numpy(), dtype=_DTF)
                        if lc > 0.0:
                            comm_loss_val, comm_norm_val = _commutator_collapse_loss(
                                H_np,
                                list(a_list),
                                comm_eps=float(comm_eps),
                                max_power=int(self.max_power),
                                warned=warned_comm,
                            )
                            loss = float(loss) + lc * comm_loss_val
                            L_comm_terms.append(lc * comm_loss_val)
                            comm_norms.append(comm_norm_val)
                            logged_comm = True
                        if ld > 0.0 and cand_idx < 64:
                            eigs = np.linalg.eigh(H_np, UPLO="U")[0].astype(_DTF)
                            eigs.sort()
                            div_pen_val = 0.0
                            if len(spectra_accum) >= 1:
                                div_pen_val = _spectral_diversity_penalty(
                                    eigs,
                                    spectra_accum,
                                    diversity_sigma=float(diversity_sigma),
                                )
                            spectra_accum.append(np.asarray(eigs, copy=True))
                            div_pen_raw.append(div_pen_val)
                            loss = float(loss) + ld * div_pen_val
                            L_div_terms.append(ld * div_pen_val)
                            logged_div = True
                        elif ld > 0.0:
                            div_pen_raw.append(0.0)
                            L_div_terms.append(0.0)
                            logged_div = True
                except Exception:
                    pen = (
                        (float(lambda_ncg) != 0.0 and bool(use_ncg_braid))
                        or (float(lambda_trace) != 0.0 and bool(use_selberg_trace))
                        or build_h_reg
                    )
                    len_extra = ll * len_pen_val if ll > 0.0 else 0.0
                    loss = float(loss_clip) if pen else base_loss + len_extra
                if lc > 0.0 and not logged_comm:
                    L_comm_terms.append(0.0)
                    comm_norms.append(0.0)
                if ld > 0.0 and not logged_div:
                    div_pen_raw.append(0.0)
                    L_div_terms.append(0.0)
            if not np.isfinite(loss):
                loss = float(loss_clip)
            c = Candidate(a_list=key_w, length=float(ell), trace=float(tr), reward=0.0, loss=float(loss))
            scored.append(c)

        scored.sort(key=lambda c: c.loss)
        mode = str(reward_mode).lower()
        adaptive_rep: Dict[str, float] = {}
        if mode == "adaptive":
            scored, adaptive_rep = self._assign_adaptive_rewards(
                scored,
                adaptive_reward_window=int(adaptive_reward_window),
                adaptive_reward_eps=float(adaptive_reward_eps),
                adaptive_reward_clip=float(adaptive_reward_clip),
            )
        else:
            scored = self._assign_rewards(
                scored,
                reward_mode=reward_mode,
                loss_clip=float(loss_clip),
                rank_temperature=float(rank_temperature),
            )
        for c in scored:
            self._bank[c.a_list] = c

        # Prune bank by reward then loss
        bank_items2 = list(self._bank.values())
        bank_items2.sort(key=lambda c: (-c.reward, c.loss))
        self._bank = {c.a_list: c for c in bank_items2[: self.bank_size]}

        bm: Dict[str, float] = {}
        bm_trace: Dict[str, float] = {}
        if scored:
            bm = dict(per_ncg.get(scored[0].a_list, {}))
            bm_trace = dict(per_trace.get(scored[0].a_list, {}))
        nan = float("nan")

        def _gf(key: str, default: float = nan) -> float:
            v = bm.get(key)
            if v is None:
                return float(default)
            return float(v)

        stats = {
            "L_selberg_raw": float(L_selberg),
            "L_spec_raw": float(L_spec_raw),
            "L_spec_eff": float(L_spec),
            "L_spec_clipped": int(spec_clip_iteration > 0),
            "spec_clip_iteration": int(spec_clip_iteration),
            "L_spacing_raw": float(L_spacing),
            "L_selberg_norm": float(L_sel_norm),
            "L_spec_norm": float(L_spec_norm),
            "L_spacing_norm": float(L_spacing_norm),
            "L_zeta_raw": float(L_zeta_raw),
            "L_zeta_norm": float(L_zeta_norm),
            "zeta_loss": float(L_zeta_raw),
            "heat_trace": float(heat_trace_ncg),
            "spectral_entropy": float(spectral_entropy_ncg),
            "L_total_global": float(L_total_global),
            "ncg_loss": _gf("loss") if bool(use_ncg_braid) else nan,
            "ncg_spectral_loss": _gf("spectral_loss") if bool(use_ncg_braid) else nan,
            "ncg_selfadjoint_loss": _gf("selfadjoint_loss") if bool(use_ncg_braid) else nan,
            "ncg_commutator_loss": _gf("commutator_loss") if bool(use_ncg_braid) else nan,
            "ncg_heat_trace_log": _gf("heat_trace_log") if bool(use_ncg_braid) else nan,
            "ncg_zeta_D_s2": _gf("zeta_D_s2") if bool(use_ncg_braid) else nan,
            "ncg_spacing_mean": _gf("spacing_mean") if bool(use_ncg_braid) else nan,
            "ncg_spacing_std": _gf("spacing_std") if bool(use_ncg_braid) else nan,
            "ncg_r_stat_mean": _gf("r_stat_mean") if bool(use_ncg_braid) else nan,
            "ncg_eig_min": _gf("eig_min") if bool(use_ncg_braid) else nan,
            "ncg_eig_max": _gf("eig_max") if bool(use_ncg_braid) else nan,
            "best_candidate_base_loss": float(L_total_global + 1e-3 * float(scored[0].length))
            if scored
            else nan,
            "best_candidate_total_loss": float(scored[0].loss) if scored else nan,
            "trace_loss": float(bm_trace.get("trace_loss", nan)) if bool(use_selberg_trace) else nan,
            "spectral_side": float(bm_trace.get("spectral_side", nan)) if bool(use_selberg_trace) else nan,
            "geometric_side": float(bm_trace.get("geometric_side", nan)) if bool(use_selberg_trace) else nan,
            "primitive_braid_count": float(bm_trace.get("primitive_braid_count", nan))
            if bool(use_selberg_trace)
            else nan,
            "mean_braid_length": float(bm_trace.get("mean_braid_length", nan)) if bool(use_selberg_trace) else nan,
            "L_comm": float(np.mean(L_comm_terms)) if L_comm_terms else nan,
            "L_div": float(np.mean(L_div_terms)) if L_div_terms else nan,
            "L_len": float(np.mean(L_len_terms)) if L_len_terms else nan,
            "comm_norm_mean": float(np.mean(comm_norms)) if comm_norms else nan,
            "diversity_penalty_mean": float(np.mean(div_pen_raw)) if div_pen_raw else nan,
            "length_penalty_mean": float(np.mean(len_pen_raw)) if len_pen_raw else nan,
            "dtes_comm_norm_mean": float(np.mean(dtes_comm_norms)) if dtes_comm_norms else nan,
            "dtes_comm_norm_min": float(np.min(dtes_comm_norms)) if dtes_comm_norms else nan,
            "dtes_ncg_loss_mean": float(np.mean(dtes_ncg_losses)) if dtes_ncg_losses else nan,
            "dtes_ncg_loss_max": float(np.max(dtes_ncg_losses)) if dtes_ncg_losses else nan,
            # DTES triple aliases (same series as dtes_*; not the lambda_comm probe).
            "triple_comm_norm_mean": float(np.mean(dtes_comm_norms)) if dtes_comm_norms else nan,
            "triple_comm_norm_min": float(np.min(dtes_comm_norms)) if dtes_comm_norms else nan,
            "ncg_collapse_loss_mean": float(np.mean(dtes_ncg_losses)) if dtes_ncg_losses else nan,
            "ncg_collapse_loss_max": float(np.max(dtes_ncg_losses)) if dtes_ncg_losses else nan,
            "dtes_effective_weight": float(dtes_w),
            **adaptive_rep,
        }
        return scored, stats

    def evaporate(self) -> None:
        if not self.pheromone:
            return
        rho = _clip(self.rho, 0.0, 1.0)
        for k, v in list(self.pheromone.items()):
            nv = (1.0 - rho) * float(v)
            self.pheromone[k] = _clip(nv, self.tau_min, self.tau_max)

    def reinforce(self, best: List[Candidate]) -> None:
        for c in best:
            a = list(c.a_list)
            if len(a) < 2:
                continue
            delta = self.q * float(c.reward)
            for i in range(len(a) - 1):
                key = (int(a[i]), int(a[i + 1]))
                self.pheromone[key] = _clip(self.pheromone.get(key, 1.0) + delta, self.tau_min, self.tau_max)

    def reinforce_candidate(self, c: Candidate, *, weight: float = 1.0) -> None:
        """Deposit pheromone along one candidate; ``weight`` scales deposit (elite/global-best reinforcement)."""
        if weight <= 0.0:
            return
        a = list(c.a_list)
        if len(a) < 2:
            return
        delta = float(weight) * self.q * float(c.reward)
        for i in range(len(a) - 1):
            key = (int(a[i]), int(a[i + 1]))
            self.pheromone[key] = _clip(self.pheromone.get(key, 1.0) + delta, self.tau_min, self.tau_max)

    def run(
        self,
        num_iters: int,
        zeros: np.ndarray,
        *,
        lambda_selberg: float,
        lambda_spec: float,
        lambda_spacing: float,
        selberg_sigma: float,
        selberg_m_max: int,
        selberg_bank_top_n: int,
        n_points: int,
        op_sigma: float,
        op_eps: float,
        op_top_k_geodesics: int,
        operator_builder: str = "word_sensitive",
        geo_weight: float = 10.0,
        geo_sigma: float = 0.6,
        potential_weight: float = 0.25,
        log_top_words: int = 5,
        use_planner: bool = False,
        planner_backend: str = "llama_cpp",
        llama_cli: str = "/Users/machome/llama.cpp/llama-cli",
        planner_model: str = "/Users/machome/models/gemma/gemma-3-1b-it-Q4_K_M.gguf",
        planner_inject_frac: float = 0.2,
        planner_log_path: str = "runs/gemma_planner_log.jsonl",
        planner_replace_frac: float = 0.2,
        reward_mode: str = "rank",
        loss_clip: float = 1000.0,
        rank_temperature: float = 1.0,
        adaptive_reward_window: int = 50,
        adaptive_reward_eps: float = 1e-8,
        adaptive_reward_clip: float = 5.0,
        normalize_loss_components: bool = True,
        component_clip: float = 100.0,
        component_log_scale: bool = True,
        lambda_zeta: float = 0.0,
        ncg_dim: int = 128,
        ncg_n_strands: int = 4,
        ncg_max_word_len: int = 8,
        ncg_diagonal_growth: str = "exp",
        ncg_growth_alpha: float = 0.15,
        ncg_edge_scale: float = 0.05,
        ncg_spectrum_scale: float = 1.0,
        ncg_dtype: str = "float64",
        ncg_device: str = "cpu",
        use_ncg_braid: bool = False,
        lambda_ncg: float = 0.0,
        lambda_ncg_spectral: float = 1.0,
        lambda_ncg_selfadjoint: float = 0.01,
        lambda_ncg_commutator: float = 0.001,
        lambda_ncg_spacing: float = 0.05,
        lambda_ncg_zeta: float = 0.01,
        ncg_target_zeros: Optional[np.ndarray] = None,
        use_selberg_trace: bool = False,
        lambda_trace: float = 0.0,
        trace_sigma: float = 0.75,
        trace_max_cycles: int = 16,
        use_seed_motifs: bool = False,
        seed_motif_frac: float = 0.25,
        seed_mutation_prob: float = 0.5,
        elite_weight: float = 1.0,
        lambda_comm: float = 0.0,
        lambda_diversity: float = 0.0,
        lambda_length: float = 0.0,
        comm_eps: float = 1e-6,
        diversity_sigma: float = 1.0,
        target_length: float = 4.0,
        lambda_dtes_triple: float = 0.0,
        lambda_spectral_div: float = 0.0,
        ncg_eps: float = 1e-6,
        restart_patience: int = 0,
        restart_fraction: float = 0.5,
        exploration_floor: float = 0.0,
        pheromone_min: float = 1e-6,
        pheromone_max: float = 1e6,
        bank_prune_fraction: float = 0.5,
        alpha_anneal: float = 0.0,
        beta_anneal: float = 0.0,
        adaptive_scheduler: bool = False,
        scheduler_window: int = 10,
        exploration_floor_max: float = 0.3,
        lambda_ncg_max: float = 0.1,
        lambda_diversity_max: float = 0.2,
        restart_patience_min: int = 4,
        spec_clip: float = 100.0,
        reject_nonfinite_spec: bool = False,
    ) -> Tuple[Dict[str, Any], List[Tuple[Any, ...]]]:
        history: List[Tuple[Any, ...]] = []
        best_snapshot: Dict[str, Any] = {"best_loss": float("inf"), "best_words": [], "best_lengths": []}
        recent_losses: List[float] = []
        last_eval_stats: Dict[str, Any] = {}
        spec_was_clipped_count = 0

        alpha_base = float(self.alpha)
        beta_base = float(self.beta)

        global_best_loss = float("inf")
        global_best_candidate: Optional[Candidate] = None
        global_best_iter = -1
        final_iter_best_loss = float("nan")
        last_improvement_iter = -1
        restart_count = 0
        bank_saturation_iter: Optional[int] = None
        global_best_history: List[float] = []
        iter_best_history: List[float] = []
        exploration_floor_eff_final = float(exploration_floor)
        lambda_ncg_eff_final = float(lambda_ncg)
        lambda_diversity_eff_final = float(lambda_diversity)
        restart_patience_eff_final = int(restart_patience)
        stagnation_iters_final = 0
        sched_ramp_p_final = 0.0
        sched_note_ncg: Optional[str] = None
        sched_note_div: Optional[str] = None
        _sched_warned_ncg_cap: List[bool] = [False]
        _sched_warned_div_cap: List[bool] = [False]

        planner = None
        if use_planner:
            try:
                from core.gemma_planner import GemmaPlanner

                planner = GemmaPlanner(
                    model_path=str(planner_model),
                    llama_cli=str(llama_cli),
                    backend=str(planner_backend),
                    max_length=int(self.max_length),
                    max_power=int(self.max_power),
                )
            except Exception:
                planner = None

        last_valid_rate = 0.0
        last_mean_loss = float("inf")
        last_best_loss = float("inf")

        for it in range(int(num_iters)):
            self.tau_min = float(pheromone_min)
            self.tau_max = float(pheromone_max)
            ni = max(1, int(num_iters) - 1)
            t_scale = float(it) / float(ni)
            alpha_eff = alpha_base * (1.0 + float(alpha_anneal) * t_scale)
            beta_eff = beta_base * (1.0 + float(beta_anneal) * t_scale)
            self._alpha_eff = alpha_eff
            self._beta_eff = beta_eff

            stagnation_iters = int(it) - int(global_best_iter)
            (
                exploration_floor_eff,
                lambda_ncg_eff,
                lambda_diversity_eff,
                restart_patience_eff,
                sched_ramp_p,
            ) = compute_scheduler_effective(
                adaptive_scheduler=bool(adaptive_scheduler),
                stagnation_iters=stagnation_iters,
                scheduler_window=int(scheduler_window),
                exploration_floor=float(exploration_floor),
                exploration_floor_max=float(exploration_floor_max),
                lambda_ncg=float(lambda_ncg),
                lambda_ncg_max=float(lambda_ncg_max),
                lambda_diversity=float(lambda_diversity),
                lambda_diversity_max=float(lambda_diversity_max),
                restart_patience=int(restart_patience),
                restart_patience_min=int(restart_patience_min),
            )
            self._exploration_floor = float(exploration_floor_eff)
            exploration_floor_eff_final = float(exploration_floor_eff)
            lambda_ncg_eff_final = float(lambda_ncg_eff)
            lambda_diversity_eff_final = float(lambda_diversity_eff)
            restart_patience_eff_final = int(restart_patience_eff)
            stagnation_iters_final = int(stagnation_iters)
            sched_ramp_p_final = float(sched_ramp_p)

            scheduler_active = int(
                bool(adaptive_scheduler) and stagnation_iters >= int(scheduler_window)
            )
            ncg_headroom = float(lambda_ncg_max) - float(lambda_ncg)
            div_headroom = float(lambda_diversity_max) - float(lambda_diversity)
            if (
                bool(adaptive_scheduler)
                and stagnation_iters >= int(scheduler_window)
                and float(sched_ramp_p) > 0.0
            ):
                if ncg_headroom <= 0.0 and not _sched_warned_ncg_cap[0]:
                    _sched_warned_ncg_cap[0] = True
                    sched_note_ncg = "lambda_ncg_at_cap"
                    warnings.warn(
                        "V12.8 adaptive scheduler: lambda_ncg already at or above lambda_ncg_max; "
                        "NCG weight ramp skipped.",
                        UserWarning,
                        stacklevel=2,
                    )
                if div_headroom <= 0.0 and not _sched_warned_div_cap[0]:
                    _sched_warned_div_cap[0] = True
                    sched_note_div = "lambda_diversity_at_cap"
                    warnings.warn(
                        "V12.8 adaptive scheduler: lambda_diversity already at or above lambda_diversity_max; "
                        "diversity weight ramp skipped.",
                        UserWarning,
                        stacklevel=2,
                    )

            rp = int(restart_patience_eff)
            if rp > 0 and it >= rp:
                gap = int(it) - int(last_improvement_iter)
                if gap >= rp:
                    gb_l = float(global_best_loss) if global_best_candidate is not None else float("nan")
                    self._pheromone_restart_blend(float(restart_fraction))
                    restart_count += 1
                    last_improvement_iter = int(it)
                    o_b, n_b = self._prune_bank(float(bank_prune_fraction))
                    print(
                        f"[restart] iter={it} reason=no_global_improvement global_best={gb_l:.6g} "
                        f"restart_count={restart_count}",
                        flush=True,
                    )
                    print(f"[bank-prune] iter={it} old={o_b} new={n_b}", flush=True)

            t_it0 = time.perf_counter()
            # build raw population (optional seed motifs + local motif splice mutation)
            population = []
            seed_init_count = 0
            for _ in range(self.num_ants):
                if use_seed_motifs and self.rng.random() < float(seed_motif_frac):
                    base = list(SEED_MOTIFS[int(self.rng.integers(0, len(SEED_MOTIFS)))])
                    seed_init_count += 1
                    if len(base) < self.max_length:
                        room = self.max_length - len(base)
                        extra_len = (
                            int(self.rng.integers(0, room)) if room > 0 else 0
                        )
                        extra = (
                            self._sample_random_word_of_length(extra_len) if extra_len > 0 else []
                        )
                        word = base + extra
                    else:
                        word = base[: self.max_length]
                else:
                    word = self.sample_word()
                word = self._maybe_seed_mutate(
                    word,
                    use_seed_motifs=bool(use_seed_motifs),
                    seed_mutation_prob=float(seed_mutation_prob),
                )
                population.append(word)

            # planner injection
            if planner is not None:
                try:
                    planner_best_words = [
                        w
                        for w in self.best_words
                        if _is_valid_action_word(w, self.max_length, self.max_power)
                    ]
                    ctx = {
                        "best_words": planner_best_words,
                        "recent_losses": recent_losses[-20:],
                        "iteration": int(it),
                        "stats": {
                            "valid_rate": float(last_valid_rate),
                            "mean_loss": float(last_mean_loss),
                            "best_loss": float(last_best_loss),
                        },
                    }
                    raw_proposals = planner.suggest_words(ctx)
                    parsed_words = [
                        list(w) for w in raw_proposals if isinstance(w, list)
                    ]
                    proposals = [
                        list(w)
                        for w in parsed_words
                        if _is_valid_action_word(list(w), self.max_length, self.max_power)
                    ]
                    valid_seen = set()
                    valid_words: List[List[int]] = []
                    for w in proposals:
                        key = tuple(int(a) for a in w)
                        if key in valid_seen:
                            continue
                        valid_seen.add(key)
                        valid_words.append(w)
                    proposals = valid_words
                    rejected_count = max(0, len(parsed_words) - len(valid_words))
                    used = False
                    if proposals:
                        n_replace = int(min(len(population), max(1, int(float(self.num_ants) * float(planner_inject_frac)))))
                        n_use = min(n_replace, len(proposals))
                        for j in range(n_use):
                            if _is_valid_action_word(proposals[j], self.max_length, self.max_power):
                                population[j] = proposals[j]
                        used = n_use > 0

                    try:
                        Path(planner_log_path).parent.mkdir(parents=True, exist_ok=True)
                        with open(planner_log_path, "a", encoding="utf-8") as f:
                            f.write(
                                json.dumps(
                                    {
                                        "iteration": int(it),
                                        "raw_response": getattr(planner, "last_raw_response", ""),
                                        "parsed_words": getattr(planner, "last_parsed_words", parsed_words),
                                        "valid_words": getattr(planner, "last_valid_words", valid_words),
                                        "rejected_count": int(getattr(planner, "last_rejected_count", rejected_count)),
                                        "planner_words": proposals,
                                        "used": bool(used),
                                    }
                                )
                                + "\n"
                            )
                    except Exception:
                        pass
                except Exception:
                    pass

            # validate
            valids: List[Tuple[List[int], float, float]] = []
            for w in population:
                if not _is_valid_action_word(w, self.max_length, self.max_power):
                    continue
                ok, ell, tr = self._validate_and_length(w)
                if ok:
                    valids.append((w, ell, tr))

            if not valids:
                self.evaporate()
                history.append(
                    (
                        it,
                        float("inf"),
                        float("inf"),
                        0.0,
                        0.0,
                        float("-inf"),
                        float("-inf"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        0,
                        int(spec_was_clipped_count),
                        str(reward_mode),
                    )
                )
                gb_h = float(global_best_loss) if global_best_candidate is not None else float("inf")
                global_best_history.append(gb_h)
                iter_best_history.append(float("inf"))
                print(
                    f"[{it}] best=inf mean=inf valid=0 avg_ell=nan bank={len(self._bank)} "
                    f"scheduler_active={int(bool(adaptive_scheduler) and int(stagnation_iters) >= int(scheduler_window))} "
                    f"exploration_floor_eff={float(exploration_floor_eff):.6g} "
                    f"lambda_ncg_eff={float(lambda_ncg_eff):.6g} "
                    f"lambda_diversity_eff={float(lambda_diversity_eff):.6g} "
                    f"restart_patience_eff={int(restart_patience_eff)} "
                    f"stagnation_iters={int(stagnation_iters)}",
                    flush=True,
                )
                continue

            scored, stats = self.evaluate_iteration(
                candidates=valids,
                zeros=zeros,
                lambda_selberg=float(lambda_selberg),
                lambda_spec=float(lambda_spec),
                lambda_spacing=float(lambda_spacing),
                lambda_zeta=float(lambda_zeta),
                selberg_sigma=float(selberg_sigma),
                selberg_m_max=int(selberg_m_max),
                selberg_bank_top_n=int(selberg_bank_top_n),
                n_points=int(n_points),
                op_sigma=float(op_sigma),
                op_eps=float(op_eps),
                op_top_k_geodesics=int(op_top_k_geodesics),
                operator_builder=str(operator_builder),
                geo_weight=float(geo_weight),
                geo_sigma=float(geo_sigma),
                potential_weight=float(potential_weight),
                ncg_dim=int(ncg_dim),
                ncg_n_strands=int(ncg_n_strands),
                ncg_max_word_len=int(ncg_max_word_len),
                ncg_diagonal_growth=str(ncg_diagonal_growth),
                ncg_growth_alpha=float(ncg_growth_alpha),
                ncg_edge_scale=float(ncg_edge_scale),
                ncg_spectrum_scale=float(ncg_spectrum_scale),
                ncg_dtype=str(ncg_dtype),
                ncg_device=str(ncg_device),
                use_ncg_braid=bool(use_ncg_braid),
                lambda_ncg=float(lambda_ncg_eff),
                lambda_ncg_spectral=float(lambda_ncg_spectral),
                lambda_ncg_selfadjoint=float(lambda_ncg_selfadjoint),
                lambda_ncg_commutator=float(lambda_ncg_commutator),
                lambda_ncg_spacing=float(lambda_ncg_spacing),
                lambda_ncg_zeta=float(lambda_ncg_zeta),
                ncg_target_zeros=ncg_target_zeros,
                reward_mode=str(reward_mode),
                loss_clip=float(loss_clip),
                rank_temperature=float(rank_temperature),
                adaptive_reward_window=int(adaptive_reward_window),
                adaptive_reward_eps=float(adaptive_reward_eps),
                adaptive_reward_clip=float(adaptive_reward_clip),
                normalize_loss_components=bool(normalize_loss_components),
                component_clip=float(component_clip),
                component_log_scale=bool(component_log_scale),
                planner_braid_words=(
                    list(getattr(planner, "last_candidate_braid_words", []) or [])
                    if planner is not None
                    else None
                ),
                use_selberg_trace=bool(use_selberg_trace),
                lambda_trace=float(lambda_trace),
                trace_sigma=float(trace_sigma),
                trace_max_cycles=int(trace_max_cycles),
                lambda_comm=float(lambda_comm),
                lambda_diversity=float(lambda_diversity_eff),
                lambda_length=float(lambda_length),
                comm_eps=float(comm_eps),
                diversity_sigma=float(diversity_sigma),
                target_length=float(target_length),
                lambda_dtes_triple=float(lambda_dtes_triple),
                lambda_spectral_div=float(lambda_spectral_div),
                ncg_eps=float(ncg_eps),
                spec_clip=float(spec_clip),
                reject_nonfinite_spec=bool(reject_nonfinite_spec),
            )

            spec_was_clipped_count += int(stats.get("spec_clip_iteration", 0))

            if not scored:
                last_eval_stats = dict(stats)
                self.evaporate()
                history.append(
                    (
                        it,
                        float("inf"),
                        float("inf"),
                        0.0,
                        0.0,
                        float("-inf"),
                        float("-inf"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        float("nan"),
                        int(stats.get("L_spec_clipped", 0)),
                        int(spec_was_clipped_count),
                        str(reward_mode),
                    )
                )
                gb_h = float(global_best_loss) if global_best_candidate is not None else float("inf")
                global_best_history.append(gb_h)
                iter_best_history.append(float("inf"))
                lsr = float(stats.get("L_spec_raw", float("nan")))
                print(
                    f"[{it}] spec_nonfinite_rejected valid={len(valids)} bank={len(self._bank)} "
                    f"L_spec_raw={lsr} L_spec_clipped={int(stats.get('L_spec_clipped', 0))} "
                    f"spec_clip_count={spec_was_clipped_count} "
                    f"scheduler_active={int(bool(adaptive_scheduler) and int(stagnation_iters) >= int(scheduler_window))} "
                    f"exploration_floor_eff={float(exploration_floor_eff):.6g} "
                    f"lambda_ncg_eff={float(lambda_ncg_eff):.6g} "
                    f"lambda_diversity_eff={float(lambda_diversity_eff):.6g} "
                    f"restart_patience_eff={int(restart_patience_eff)} "
                    f"stagnation_iters={int(stagnation_iters)}",
                    flush=True,
                )
                continue

            stats["seed_used"] = float(seed_init_count)
            stats["seed_frac"] = float(seed_init_count) / float(max(1, self.num_ants))
            stats["motif_hits"] = (
                float(count_motif_occurrences(list(scored[0].a_list))) if scored else float("nan")
            )
            last_eval_stats = dict(stats)

            best = scored[0]
            iter_best_loss = float(best.loss)
            iter_best_candidate = best
            if iter_best_loss < global_best_loss:
                global_best_loss = iter_best_loss
                global_best_candidate = iter_best_candidate
                global_best_iter = int(it)
                last_improvement_iter = int(it)
            final_iter_best_loss = iter_best_loss
            global_best_history.append(
                float(global_best_loss) if global_best_candidate is not None else float("inf")
            )
            iter_best_history.append(float(iter_best_loss))

            mean_loss = float(np.mean([c.loss for c in scored])) if scored else float("inf")
            best_reward = float(scored[0].reward) if scored else 0.0
            mean_reward = float(np.mean([c.reward for c in scored])) if scored else 0.0
            # Raw reward is always defined as -loss (un-normalized), for logging/debugging.
            best_raw_reward = -float(best.loss)
            mean_raw_reward = -float(mean_loss)
            loss_median = float(stats.get("loss_median", float("nan")))
            loss_iqr = float(stats.get("loss_iqr", float("nan")))
            loss_std = float(stats.get("loss_std", float("nan")))
            reward_min = float(stats.get("reward_min", float("nan")))
            reward_max = float(stats.get("reward_max", float("nan")))
            reward_mean = float(stats.get("reward_mean", float("nan")))
            L_selberg_raw = float(stats.get("L_selberg_raw", float("nan")))
            L_spec_raw = float(stats.get("L_spec_raw", float("nan")))
            L_spec_eff_v = float(stats.get("L_spec_eff", float("nan")))
            L_spec_clipped_v = int(stats.get("L_spec_clipped", 0))
            L_spacing_raw = float(stats.get("L_spacing_raw", float("nan")))
            L_selberg_norm = float(stats.get("L_selberg_norm", float("nan")))
            L_spec_norm = float(stats.get("L_spec_norm", float("nan")))
            L_spacing_norm = float(stats.get("L_spacing_norm", float("nan")))
            L_zeta_raw = float(stats.get("L_zeta_raw", float("nan")))
            heat_trace_ncg = float(stats.get("heat_trace", float("nan")))
            spectral_entropy_ncg = float(stats.get("spectral_entropy", float("nan")))
            ncg_loss = float(stats.get("ncg_loss", float("nan")))
            ncg_spectral_loss = float(stats.get("ncg_spectral_loss", float("nan")))
            ncg_selfadjoint_loss = float(stats.get("ncg_selfadjoint_loss", float("nan")))
            ncg_commutator_loss = float(stats.get("ncg_commutator_loss", float("nan")))
            ncg_heat_trace_log = float(stats.get("ncg_heat_trace_log", float("nan")))
            ncg_zeta_D_s2 = float(stats.get("ncg_zeta_D_s2", float("nan")))
            ncg_spacing_mean = float(stats.get("ncg_spacing_mean", float("nan")))
            ncg_spacing_std = float(stats.get("ncg_spacing_std", float("nan")))
            ncg_r_stat_mean = float(stats.get("ncg_r_stat_mean", float("nan")))
            ncg_eig_min = float(stats.get("ncg_eig_min", float("nan")))
            ncg_eig_max = float(stats.get("ncg_eig_max", float("nan")))
            trace_loss_v = float(stats.get("trace_loss", float("nan")))
            spectral_side_v = float(stats.get("spectral_side", float("nan")))
            geometric_side_v = float(stats.get("geometric_side", float("nan")))
            primitive_braid_count_v = float(stats.get("primitive_braid_count", float("nan")))
            mean_braid_length_v = float(stats.get("mean_braid_length", float("nan")))
            seed_used_v = float(stats.get("seed_used", float("nan")))
            seed_frac_v = float(stats.get("seed_frac", float("nan")))
            motif_hits_v = float(stats.get("motif_hits", float("nan")))
            L_comm_v = float(stats.get("L_comm", float("nan")))
            L_div_v = float(stats.get("L_div", float("nan")))
            L_len_v = float(stats.get("L_len", float("nan")))
            comm_norm_mean_v = float(stats.get("comm_norm_mean", float("nan")))
            div_pen_mean_v = float(stats.get("diversity_penalty_mean", float("nan")))
            len_pen_mean_v = float(stats.get("length_penalty_mean", float("nan")))
            dtes_comm_mean_v = float(stats.get("dtes_comm_norm_mean", float("nan")))
            dtes_comm_min_v = float(stats.get("dtes_comm_norm_min", float("nan")))
            dtes_ncg_lm_v = float(stats.get("dtes_ncg_loss_mean", float("nan")))
            dtes_ncg_lx_v = float(stats.get("dtes_ncg_loss_max", float("nan")))
            triple_cm_v = float(stats.get("triple_comm_norm_mean", float("nan")))
            triple_cmin_v = float(stats.get("triple_comm_norm_min", float("nan")))
            ncg_clm_v = float(stats.get("ncg_collapse_loss_mean", float("nan")))
            ncg_clx_v = float(stats.get("ncg_collapse_loss_max", float("nan")))
            avg_ell = float(np.mean([c.length for c in scored])) if scored else float("nan")
            valid_rate = float(len(scored)) / float(max(1, self.num_ants))

            if best.loss < self.best_loss:
                self.best_loss = float(best.loss)
                self.best_words = [list(best.a_list)]
                best_snapshot = {
                    "best_loss": float(self.best_loss),
                    "best_words": [list(best.a_list)],
                    "best_lengths": [float(best.length)],
                }
                if bool(use_ncg_braid) or bool(use_selberg_trace):
                    ncg_m: Optional[Dict[str, Any]] = None
                    if bool(use_ncg_braid):
                        ncg_m = {
                            "loss": float(stats.get("ncg_loss", float("nan"))),
                            "spectral_loss": float(stats.get("ncg_spectral_loss", float("nan"))),
                            "selfadjoint_loss": float(stats.get("ncg_selfadjoint_loss", float("nan"))),
                            "commutator_loss": float(stats.get("ncg_commutator_loss", float("nan"))),
                            "heat_trace_log": float(stats.get("ncg_heat_trace_log", float("nan"))),
                            "zeta_D_s2": float(stats.get("ncg_zeta_D_s2", float("nan"))),
                            "spacing_mean": float(stats.get("ncg_spacing_mean", float("nan"))),
                            "spacing_std": float(stats.get("ncg_spacing_std", float("nan"))),
                            "r_stat_mean": float(stats.get("ncg_r_stat_mean", float("nan"))),
                            "eig_min": float(stats.get("ncg_eig_min", float("nan"))),
                            "eig_max": float(stats.get("ncg_eig_max", float("nan"))),
                        }
                    trace_m: Optional[Dict[str, float]] = None
                    if bool(use_selberg_trace):
                        trace_m = {
                            "trace_loss": float(stats.get("trace_loss", float("nan"))),
                            "spectral_side": float(stats.get("spectral_side", float("nan"))),
                            "geometric_side": float(stats.get("geometric_side", float("nan"))),
                            "primitive_braid_count": float(stats.get("primitive_braid_count", float("nan"))),
                            "mean_braid_length": float(stats.get("mean_braid_length", float("nan"))),
                        }
                    self.best_ncg_export = {
                        "best_words": [list(best.a_list)],
                        "base_loss": float(stats.get("best_candidate_base_loss", float("nan"))),
                        "ncg_loss": float(stats.get("ncg_loss", float("nan"))) if bool(use_ncg_braid) else float("nan"),
                        "total_loss": float(best.loss),
                        "ncg_metrics": ncg_m,
                        "selberg_trace_metrics": trace_m,
                    }

            self.evaporate()
            self.reinforce(scored[: self.best_k_ants])
            if float(elite_weight) > 0.0 and global_best_candidate is not None:
                self.reinforce_candidate(global_best_candidate, weight=float(elite_weight))
            if use_seed_motifs:
                self._boost_motif_pheromone()
            clamp_pheromone(self.pheromone, self.tau_min, self.tau_max)

            if bank_saturation_iter is None and (
                len(self._bank) >= int(self.bank_size) or len(self._bank) >= 1000
            ):
                bank_saturation_iter = int(it)

            history.append(
                (
                    it,
                    float(best.loss),
                    float(mean_loss),
                    float(best_reward),
                    float(mean_reward),
                    float(best_raw_reward),
                    float(mean_raw_reward),
                    float(loss_median),
                    float(loss_iqr),
                    float(loss_std),
                    float(reward_min),
                    float(reward_max),
                    float(reward_mean),
                    float(L_selberg_raw),
                    float(L_spec_raw),
                    float(L_spacing_raw),
                    float(L_selberg_norm),
                    float(L_spec_norm),
                    float(L_spacing_norm),
                    float(L_zeta_raw),
                    float(heat_trace_ncg),
                    float(spectral_entropy_ncg),
                    float(ncg_loss),
                    float(ncg_spectral_loss),
                    float(ncg_selfadjoint_loss),
                    float(ncg_commutator_loss),
                    float(ncg_heat_trace_log),
                    float(ncg_zeta_D_s2),
                    float(ncg_spacing_mean),
                    float(ncg_spacing_std),
                    float(ncg_r_stat_mean),
                    float(ncg_eig_min),
                    float(ncg_eig_max),
                    float(trace_loss_v),
                    float(spectral_side_v),
                    float(geometric_side_v),
                    float(primitive_braid_count_v),
                    float(mean_braid_length_v),
                    float(seed_used_v),
                    float(seed_frac_v),
                    float(motif_hits_v),
                    int(L_spec_clipped_v),
                    int(spec_was_clipped_count),
                    str(reward_mode),
                )
            )
            recent_losses.append(float(best.loss))
            last_valid_rate = float(valid_rate)
            last_mean_loss = float(mean_loss)
            last_best_loss = float(best.loss)

            top_words = scored[: max(1, int(log_top_words))]
            top_words_s = " | ".join(
                [f"{list(c.a_list)} ℓ={c.length:.3g} L={c.loss:.3g} r={c.reward:.3g}" for c in top_words]
            )
            dt = time.perf_counter() - t_it0
            gb_loss_log = (
                float(global_best_loss) if global_best_candidate is not None else float("nan")
            )
            ae_v = float(getattr(self, "_alpha_eff", self.alpha))
            be_v = float(getattr(self, "_beta_eff", self.beta))
            sched_log = (
                f"scheduler_active={scheduler_active} "
                f"exploration_floor_eff={float(exploration_floor_eff):.6g} "
                f"lambda_ncg_eff={float(lambda_ncg_eff):.6g} "
                f"lambda_diversity_eff={float(lambda_diversity_eff):.6g} "
                f"restart_patience_eff={int(restart_patience_eff)} "
                f"stagnation_iters={int(stagnation_iters)} "
                f"L_spec_raw={L_spec_raw:.6g} L_spec_eff={L_spec_eff_v:.6g} "
                f"L_spec_clipped={L_spec_clipped_v} spec_clip_count={spec_was_clipped_count}"
            )
            if it % 10 == 0:
                print(
                    f"[{it}] best={gb_loss_log:.6g} iter_best={iter_best_loss:.6g} "
                    f"global_best={gb_loss_log:.6g} global_best_iter={global_best_iter} "
                    f"mean={mean_loss:.6g} valid={len(scored)} "
                    f"avg_ell={avg_ell:.6g} bank={len(self._bank)} "
                    f"{sched_log} "
                    f"alpha_eff={ae_v:.4g} beta_eff={be_v:.4g} "
                    f"L_sel={L_selberg_raw:.3g} L_spec_raw={L_spec_raw:.3g} L_spec_eff={L_spec_eff_v:.3g} "
                    f"L_comm={L_comm_v:.3g} L_div={L_div_v:.3g} L_len={L_len_v:.3g} "
                    f"comm_norm_mean={comm_norm_mean_v:.3g} diversity_penalty_mean={div_pen_mean_v:.3g} "
                    f"length_penalty_mean={len_pen_mean_v:.3g} "
                    f"triple_comm_norm_mean={triple_cm_v:.3g} triple_comm_norm_min={triple_cmin_v:.3g} "
                    f"ncg_loss_mean={ncg_clm_v:.3g} ncg_loss_max={ncg_clx_v:.3g} "
                    f"dtes_comm_norm_mean={dtes_comm_mean_v:.3g} dtes_comm_norm_min={dtes_comm_min_v:.3g} "
                    f"dtes_ncg_loss_mean={dtes_ncg_lm_v:.3g} dtes_ncg_loss_max={dtes_ncg_lx_v:.3g} "
                    f"loss_std={loss_std:.3g} loss_iqr={loss_iqr:.3g} "
                    f"r_min={reward_min:.3g} r_max={reward_max:.3g} r_mean={reward_mean:.3g} "
                    f"best_r={best_reward:.3g} mean_r={mean_reward:.3g} mode={reward_mode} "
                    f"builder={operator_builder} dt={dt:.3g}s :: {top_words_s}",
                    flush=True,
                )
            else:
                print(
                    f"[{it}] best={gb_loss_log:.6g} iter_best={iter_best_loss:.6g} "
                    f"global_best={gb_loss_log:.6g} global_best_iter={global_best_iter} "
                    f"mean={mean_loss:.6g} valid={len(scored)} "
                    f"avg_ell={avg_ell:.6g} bank={len(self._bank)} "
                    f"{sched_log} "
                    f"alpha_eff={ae_v:.4g} beta_eff={be_v:.4g} "
                    f"L_sel={L_selberg_raw:.3g} L_spec_raw={L_spec_raw:.3g} L_spec_eff={L_spec_eff_v:.3g} "
                    f"L_comm={L_comm_v:.3g} L_div={L_div_v:.3g} L_len={L_len_v:.3g} "
                    f"comm_norm_mean={comm_norm_mean_v:.3g} diversity_penalty_mean={div_pen_mean_v:.3g} "
                    f"length_penalty_mean={len_pen_mean_v:.3g} "
                    f"triple_comm_norm_mean={triple_cm_v:.3g} triple_comm_norm_min={triple_cmin_v:.3g} "
                    f"ncg_loss_mean={ncg_clm_v:.3g} ncg_loss_max={ncg_clx_v:.3g} "
                    f"dtes_comm_norm_mean={dtes_comm_mean_v:.3g} dtes_comm_norm_min={dtes_comm_min_v:.3g} "
                    f"dtes_ncg_loss_mean={dtes_ncg_lm_v:.3g} dtes_ncg_loss_max={dtes_ncg_lx_v:.3g} "
                    f"best_r={best_reward:.3g} mean_r={mean_reward:.3g} mode={reward_mode} "
                    f"builder={operator_builder} dt={dt:.3g}s :: {top_words_s}",
                    flush=True,
                )

        best_snapshot["global_best_loss"] = (
            float(global_best_loss) if global_best_candidate is not None else float("nan")
        )
        best_snapshot["global_best_candidate"] = (
            {
                "a_list": list(global_best_candidate.a_list),
                "loss": float(global_best_candidate.loss),
                "length": float(global_best_candidate.length),
                "trace": float(global_best_candidate.trace),
                "reward": float(global_best_candidate.reward),
            }
            if global_best_candidate is not None
            else None
        )
        best_snapshot["global_best_iter"] = int(global_best_iter)
        best_snapshot["final_iter_best_loss"] = float(final_iter_best_loss)

        nan_f = float("nan")
        best_snapshot["lambda_ncg"] = float(lambda_ncg)
        best_snapshot["lambda_comm"] = float(lambda_comm)
        best_snapshot["lambda_diversity"] = float(lambda_diversity)
        best_snapshot["lambda_length"] = float(lambda_length)
        best_snapshot["regularizer_comm_norm_mean_final"] = float(
            last_eval_stats.get("comm_norm_mean", nan_f)
        )
        best_snapshot["diversity_penalty_mean_final"] = float(
            last_eval_stats.get("diversity_penalty_mean", nan_f)
        )
        best_snapshot["length_penalty_mean_final"] = float(last_eval_stats.get("length_penalty_mean", nan_f))
        # DTES (A,H,D) spectral triple — NCG-inspired anti-collapse; not an RH proof.
        best_snapshot["lambda_dtes_triple"] = float(lambda_dtes_triple)
        best_snapshot["lambda_spectral_div"] = float(lambda_spectral_div)
        best_snapshot["dtes_effective_weight_final"] = float(
            last_eval_stats.get("dtes_effective_weight", nan_f)
        )
        best_snapshot["dtes_comm_norm_mean_final"] = float(
            last_eval_stats.get("dtes_comm_norm_mean", nan_f)
        )
        best_snapshot["dtes_comm_norm_min_final"] = float(
            last_eval_stats.get("dtes_comm_norm_min", nan_f)
        )
        best_snapshot["dtes_ncg_loss_mean_final"] = float(
            last_eval_stats.get("dtes_ncg_loss_mean", nan_f)
        )
        best_snapshot["dtes_ncg_loss_max_final"] = float(
            last_eval_stats.get("dtes_ncg_loss_max", nan_f)
        )
        # Flat aliases for DTES triple (distinct from lambda_comm probe comm_norm_mean_final).
        best_snapshot["comm_norm_mean_final"] = float(
            last_eval_stats.get("dtes_comm_norm_mean", nan_f)
        )
        best_snapshot["comm_norm_min_final"] = float(
            last_eval_stats.get("dtes_comm_norm_min", nan_f)
        )
        best_snapshot["ncg_loss_mean_final"] = float(
            last_eval_stats.get("dtes_ncg_loss_mean", nan_f)
        )
        best_snapshot["dtes_triple"] = {
            "lambda_ncg": float(lambda_ncg),
            "lambda_dtes_triple": float(lambda_dtes_triple),
            "effective_weight": float(last_eval_stats.get("dtes_effective_weight", nan_f)),
            "comm_norm_mean_final": float(last_eval_stats.get("dtes_comm_norm_mean", nan_f)),
            "comm_norm_min_final": float(last_eval_stats.get("dtes_comm_norm_min", nan_f)),
            "ncg_loss_mean_final": float(last_eval_stats.get("dtes_ncg_loss_mean", nan_f)),
            "ncg_loss_max_final": float(last_eval_stats.get("dtes_ncg_loss_max", nan_f)),
        }

        # V12.7 stagnation escape / NCG-ready diagnostics (not an RH proof).
        best_snapshot["restart_count"] = int(restart_count)
        best_snapshot["bank_saturation_iter"] = (
            int(bank_saturation_iter) if bank_saturation_iter is not None else None
        )
        best_snapshot["last_improvement_iter"] = int(last_improvement_iter)
        best_snapshot["restart_patience"] = int(restart_patience)
        best_snapshot["restart_fraction"] = float(restart_fraction)
        best_snapshot["exploration_floor"] = float(exploration_floor)
        best_snapshot["pheromone_min"] = float(pheromone_min)
        best_snapshot["pheromone_max"] = float(pheromone_max)
        best_snapshot["bank_prune_fraction"] = float(bank_prune_fraction)
        best_snapshot["elite_weight"] = float(elite_weight)
        best_snapshot["alpha_anneal"] = float(alpha_anneal)
        best_snapshot["beta_anneal"] = float(beta_anneal)
        best_snapshot["elite_reinforcement_enabled"] = bool(float(elite_weight) > 0.0)

        best_snapshot["spec_clip"] = float(spec_clip)
        best_snapshot["spec_was_clipped_count"] = int(spec_was_clipped_count)
        best_snapshot["reject_nonfinite_spec"] = bool(reject_nonfinite_spec)

        # V12.8 adaptive stagnation scheduler (not an RH proof).
        best_snapshot["adaptive_scheduler"] = bool(adaptive_scheduler)
        best_snapshot["scheduler_window"] = int(scheduler_window)
        best_snapshot["exploration_floor_final"] = float(exploration_floor_eff_final)
        best_snapshot["lambda_ncg_final"] = float(lambda_ncg_eff_final)
        best_snapshot["lambda_diversity_final"] = float(lambda_diversity_eff_final)
        best_snapshot["restart_patience_final"] = int(restart_patience_eff_final)
        best_snapshot["stagnation_iters_final"] = int(stagnation_iters_final)
        best_snapshot["global_best_history"] = list(global_best_history)
        best_snapshot["iter_best_history"] = list(iter_best_history)
        if sched_note_ncg is not None:
            best_snapshot["scheduler_note_ncg"] = str(sched_note_ncg)
        if sched_note_div is not None:
            best_snapshot["scheduler_note_diversity"] = str(sched_note_div)

        return best_snapshot, history


def _csv_history_cell(v: Any) -> str:
    """Single CSV field (comma-separated; minimal escaping)."""
    if v is None:
        return ""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        if math.isinf(v):
            return "inf" if v > 0 else "-inf"
        return repr(v)
    if isinstance(v, str):
        if any(c in v for c in ",\n\r\""):
            return '"' + v.replace('"', '""') + '"'
        return v
    return str(v)


def _normalize_history_csv_row(row: Tuple[Any, ...], *, target_len: int) -> List[Any]:
    """Pad/trim history tuples so CSV columns stay aligned with the header."""
    r = list(row)
    n = len(r)
    if n == target_len:
        return r
    # Pre–V12.8B CSV (42 data columns ending with reward_mode): insert spectral columns before mode.
    if n == target_len - 2 and n >= 2:
        print(
            f"[history-warning] row_len={n} header_len={target_len} legacy_42_expanding",
            flush=True,
        )
        return r[:-1] + [float("nan"), 0, r[-1]]
    if n < target_len:
        print(
            f"[history-warning] row_len={n} header_len={target_len} padding_short_row",
            flush=True,
        )
        return r + [None] * (target_len - n)
    print(
        f"[history-warning] row_len={n} header_len={target_len} trimming_long_row",
        flush=True,
    )
    return r[:target_len]


def _write_history_csv(
    path: Path,
    history: Iterable[Tuple[Any, ...]],
    *,
    operator_builder: str,
) -> None:
    header_line = (
        "iter,best_loss,mean_loss,best_reward,mean_reward,"
        "best_raw_reward,mean_raw_reward,"
        "loss_median,loss_iqr,loss_std,reward_min,reward_max,reward_mean,"
        "L_selberg_raw,L_spec_raw,L_spacing_raw,L_selberg_norm,L_spec_norm,L_spacing_norm,"
        "L_zeta_raw,heat_trace,spectral_entropy,"
        "ncg_loss,ncg_spectral_loss,ncg_selfadjoint_loss,ncg_commutator_loss,"
        "ncg_heat_trace_log,ncg_zeta_D_s2,ncg_spacing_mean,ncg_spacing_std,ncg_r_stat_mean,"
        "ncg_eig_min,ncg_eig_max,"
        "trace_loss,spectral_side,geometric_side,primitive_braid_count,mean_braid_length,"
        "seed_used,seed_frac,motif_hits,"
        "L_spec_clipped,spec_clip_count,reward_mode,"
        "operator_builder\n"
    )
    header_cols = header_line.strip().split(",")
    target_len = len(header_cols) - 1
    with open(path, "w", encoding="utf-8") as f:
        f.write(header_line)
        for row in history:
            cells = _normalize_history_csv_row(tuple(row), target_len=target_len)
            if len(cells) != target_len:
                print(
                    f"[history-warning] row_len={len(cells)} header_len={target_len} after_normalize",
                    flush=True,
                )
                if len(cells) < target_len:
                    cells = cells + [None] * (target_len - len(cells))
                else:
                    cells = cells[:target_len]
            line = ",".join(_csv_history_cell(c) for c in cells) + "," + _csv_history_cell(operator_builder)
            f.write(line + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="V12.7 ACO (Selberg + operator + stagnation-escape); not an RH proof.")
    ap.add_argument("--num_ants", type=int, default=64)
    ap.add_argument("--num_iters", type=int, default=100)
    ap.add_argument("--max_length", type=int, default=8)
    ap.add_argument("--max_power", type=int, default=6)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--rho", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--lambda_selberg", type=float, default=1.0)
    ap.add_argument("--lambda_spec", type=float, default=1.0)
    ap.add_argument(
        "--spec_clip",
        type=float,
        default=100.0,
        help="Cap raw operator spectral loss (finite values above this are clipped; non-finite uses this value unless rejected).",
    )
    ap.add_argument(
        "--reject_nonfinite_spec",
        type=str,
        default="false",
        help="If true, skip scoring for the iteration when operator spectral loss is not finite.",
    )
    ap.add_argument("--lambda_spacing", type=float, default=0.0)
    ap.add_argument("--lambda_zeta", type=float, default=0.0)
    ap.add_argument("--use_ncg_braid", action="store_true", help="Enable per-ant NCG braid spectral losses (experimental).")
    ap.add_argument(
        "--use_selberg_trace",
        action="store_true",
        help="Selberg-style braid trace alignment vs spectrum (experimental; not an RH proof).",
    )
    ap.add_argument("--lambda_trace", type=float, default=0.0)
    ap.add_argument("--trace_sigma", type=float, default=0.75)
    ap.add_argument("--trace_max_cycles", type=int, default=16)
    ap.add_argument(
        "--lambda_ncg",
        type=float,
        default=0.0,
        help=(
            "With --use_ncg_braid: scales the NCG braid bundle loss. "
            "Otherwise (or in addition if --lambda_dtes_triple is 0): scales DTES spectral-triple "
            "ncg_collapse_loss = 1/(ncg_eps+||[D,A]||_F) (finite-dim NCG-inspired regularizer; not an RH proof)."
        ),
    )
    ap.add_argument("--lambda_ncg_spectral", type=float, default=1.0)
    ap.add_argument("--lambda_ncg_selfadjoint", type=float, default=0.01)
    ap.add_argument("--lambda_ncg_commutator", type=float, default=0.001)
    ap.add_argument("--lambda_ncg_spacing", type=float, default=0.05)
    ap.add_argument("--lambda_ncg_zeta", type=float, default=0.01)
    ap.add_argument("--ncg_dim", type=int, default=128)
    ap.add_argument("--ncg_n_strands", type=int, default=4)
    ap.add_argument("--ncg_max_word_len", type=int, default=8)
    ap.add_argument("--ncg_diagonal_growth", type=str, default="exp")
    ap.add_argument("--ncg_growth_alpha", type=float, default=0.15)
    ap.add_argument("--ncg_edge_scale", type=float, default=0.05)
    ap.add_argument("--ncg_spectrum_scale", type=float, default=1.0)
    ap.add_argument("--ncg_dtype", type=str, default="float64", choices=["float32", "float64"])
    ap.add_argument("--device", type=str, default="cpu", help="Torch device for NCG builds (e.g. cpu, cuda:0).")
    ap.add_argument(
        "--ncg_target_zeros",
        type=str,
        default="",
        help="Comma-separated zeta ordinates for NCG loss; empty uses --zeros file.",
    )

    ap.add_argument("--selberg_sigma", type=float, default=0.5)
    ap.add_argument("--selberg_m_max", type=int, default=6)
    ap.add_argument("--selberg_bank_top_n", type=int, default=250)

    ap.add_argument("--n_points", type=int, default=128)
    ap.add_argument("--op_sigma", type=float, default=0.3)
    ap.add_argument("--op_eps", type=float, default=0.6)
    ap.add_argument("--op_top_k_geodesics", type=int, default=250)
    ap.add_argument("--operator_builder", type=str, default="word_sensitive", choices=["structured", "word_sensitive"])
    ap.add_argument("--geo_weight", type=float, default=10.0)
    ap.add_argument("--geo_sigma", type=float, default=0.6)
    ap.add_argument("--potential_weight", type=float, default=0.25)

    ap.add_argument("--zeros", type=str, default="data/zeta_zeros.txt")
    ap.add_argument("--out_dir", type=str, default="runs/")
    ap.add_argument("--best_k_ants", type=int, default=8)
    ap.add_argument("--bank_size", type=int, default=1000)
    ap.add_argument("--length_threshold", type=float, default=50.0)
    ap.add_argument("--use_planner", type=str, default="False")
    ap.add_argument("--planner_backend", type=str, default="llama_cpp")
    ap.add_argument("--llama_cli", type=str, default="/Users/machome/llama.cpp/llama-cli")
    ap.add_argument("--planner_model", type=str, default="/Users/machome/models/gemma/gemma-3-1b-it-Q4_K_M.gguf")
    ap.add_argument("--planner_inject_frac", type=float, default=0.2)
    ap.add_argument("--planner_replace_frac", type=float, default=0.2)
    ap.add_argument("--reward_mode", type=str, default="rank", choices=["inverse", "rank", "soft_rank", "raw", "adaptive"])
    ap.add_argument("--adaptive_reward_window", type=int, default=50)
    ap.add_argument("--adaptive_reward_eps", type=float, default=1e-8)
    ap.add_argument("--adaptive_reward_clip", type=float, default=5.0)
    ap.add_argument("--normalize_loss_components", type=str, default="True")
    ap.add_argument("--component_clip", type=float, default=100.0)
    ap.add_argument("--component_log_scale", type=str, default="True")
    ap.add_argument("--loss_clip", type=float, default=1000.0)
    ap.add_argument("--rank_temperature", type=float, default=1.0)
    ap.add_argument(
        "--use_seed_motifs",
        action="store_true",
        help="Inject structured braid motifs into initialization and mutation (experimental search prior).",
    )
    ap.add_argument("--seed_motif_frac", type=float, default=0.25)
    ap.add_argument("--seed_mutation_prob", type=float, default=0.5)
    ap.add_argument(
        "--elite_weight",
        type=float,
        default=1.0,
        help="Extra pheromone reinforcement scale for the global-best candidate each iteration (0 disables).",
    )
    # V12.7 ACO stagnation escape (diagnostic; not an RH proof).
    ap.add_argument("--restart_patience", type=int, default=0, help="Iterations without global-best improvement before pheromone restart (0 disables).")
    ap.add_argument("--restart_fraction", type=float, default=0.5, help="Blend toward uniform pheromone on restart.")
    ap.add_argument("--exploration_floor", type=float, default=0.0, help="Additive uniform floor on action probabilities before normalize.")
    ap.add_argument("--pheromone_min", type=float, default=1e-6, help="Clamp lower bound for edge pheromone.")
    ap.add_argument("--pheromone_max", type=float, default=1e6, help="Clamp upper bound for edge pheromone.")
    ap.add_argument(
        "--bank_prune_fraction",
        type=float,
        default=0.5,
        help="On restart, retain this fraction of bank by lowest loss.",
    )
    ap.add_argument("--alpha_anneal", type=float, default=0.0, help="Linear alpha multiplier ramp across iterations.")
    ap.add_argument("--beta_anneal", type=float, default=0.0, help="Linear beta multiplier ramp across iterations.")
    # V12.8 adaptive stagnation scheduler (diagnostic; not an RH proof).
    ap.add_argument(
        "--adaptive_scheduler",
        type=str,
        default="false",
        help="When true, ramp exploration_floor / lambda_ncg / lambda_diversity and tighten restart patience after stagnation.",
    )
    ap.add_argument("--scheduler_window", type=int, default=10, help="Iterations without global_best improvement before ramping.")
    ap.add_argument("--exploration_floor_max", type=float, default=0.3, help="Upper cap for exploration_floor under adaptive scheduler.")
    ap.add_argument("--lambda_ncg_max", type=float, default=0.1, help="Upper cap for lambda_ncg under adaptive scheduler.")
    ap.add_argument("--lambda_diversity_max", type=float, default=0.2, help="Upper cap for lambda_diversity under adaptive scheduler.")
    ap.add_argument(
        "--restart_patience_min",
        type=int,
        default=4,
        help="Lower floor for effective restart patience under adaptive scheduler.",
    )
    # NCG-inspired anti-collapse regularizers (NOT an RH proof); defaults preserve legacy behavior.
    ap.add_argument("--lambda_comm", type=float, default=0.0)
    ap.add_argument("--lambda_diversity", type=float, default=0.0)
    ap.add_argument("--lambda_length", type=float, default=0.0)
    ap.add_argument("--comm_eps", type=float, default=1e-6)
    ap.add_argument("--diversity_sigma", type=float, default=1.0)
    ap.add_argument("--target_length", type=float, default=4.0)
    ap.add_argument(
        "--lambda_dtes_triple",
        type=float,
        default=0.0,
        help=(
            "Weight for DTES (A,H,D) commutator-collapse via core.dtes_spectral_triple "
            "(distinct from --lambda_ncg which scales the NCG braid bundle loss)."
        ),
    )
    ap.add_argument(
        "--lambda_spectral_div",
        type=float,
        default=0.0,
        help="Added to --lambda_diversity for the same spectral-diversity penalty.",
    )
    ap.add_argument(
        "--ncg_eps",
        type=float,
        default=1e-6,
        help="Epsilon in DTES triple ncg_collapse_loss = 1/(eps+||[D,A]||_F).",
    )
    args = ap.parse_args()

    print(
        f"[loss-normalization] enabled={str(args.normalize_loss_components).lower() in ['1','true','yes','y']} "
        f"component_clip={float(args.component_clip)} "
        f"log_scale={str(args.component_log_scale).lower() in ['1','true','yes','y']}",
        flush=True,
    )

    zeros = resolve_zeros_cli(str(args.zeros))
    if zeros.size == 0:
        raise ValueError("zeros file is empty or unreadable")

    ncg_target_zeros: Optional[np.ndarray] = None
    if str(args.ncg_target_zeros).strip():
        ncg_target_zeros = np.array(
            [float(x) for x in str(args.ncg_target_zeros).split(",") if str(x).strip()],
            dtype=_DTF,
        )

    aco = ArtinACO(
        num_ants=int(args.num_ants),
        max_length=int(args.max_length),
        max_power=int(args.max_power),
        alpha=float(args.alpha),
        beta=float(args.beta),
        rho=float(args.rho),
        seed=int(args.seed),
        length_threshold=float(args.length_threshold),
        tau_min=float(args.pheromone_min),
        tau_max=float(args.pheromone_max),
        bank_size=int(args.bank_size),
        best_k_ants=int(args.best_k_ants),
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    best, history = aco.run(
        num_iters=int(args.num_iters),
        zeros=zeros,
        lambda_selberg=float(args.lambda_selberg),
        lambda_spec=float(args.lambda_spec),
        lambda_spacing=float(args.lambda_spacing),
        selberg_sigma=float(args.selberg_sigma),
        selberg_m_max=int(args.selberg_m_max),
        selberg_bank_top_n=int(args.selberg_bank_top_n),
        n_points=int(args.n_points),
        op_sigma=float(args.op_sigma),
        op_eps=float(args.op_eps),
        op_top_k_geodesics=int(args.op_top_k_geodesics),
        operator_builder=str(args.operator_builder),
        geo_weight=float(args.geo_weight),
        geo_sigma=float(args.geo_sigma),
        potential_weight=float(args.potential_weight),
        log_top_words=5,
        use_planner=str(args.use_planner).lower() in ["1", "true", "yes", "y"],
        planner_backend=str(args.planner_backend),
        llama_cli=str(args.llama_cli),
        planner_model=str(args.planner_model),
        planner_inject_frac=float(args.planner_inject_frac),
        planner_log_path=str(Path("runs") / "gemma_planner_log.jsonl"),
        planner_replace_frac=float(args.planner_replace_frac),
        reward_mode=str(args.reward_mode),
        adaptive_reward_window=int(args.adaptive_reward_window),
        adaptive_reward_eps=float(args.adaptive_reward_eps),
        adaptive_reward_clip=float(args.adaptive_reward_clip),
        normalize_loss_components=str(args.normalize_loss_components).lower() in ["1", "true", "yes", "y"],
        component_clip=float(args.component_clip),
        component_log_scale=str(args.component_log_scale).lower() in ["1", "true", "yes", "y"],
        loss_clip=float(args.loss_clip),
        rank_temperature=float(args.rank_temperature),
        lambda_zeta=float(args.lambda_zeta),
        ncg_dim=int(args.ncg_dim),
        ncg_n_strands=int(args.ncg_n_strands),
        ncg_max_word_len=int(args.ncg_max_word_len),
        ncg_diagonal_growth=str(args.ncg_diagonal_growth),
        ncg_growth_alpha=float(args.ncg_growth_alpha),
        ncg_edge_scale=float(args.ncg_edge_scale),
        ncg_spectrum_scale=float(args.ncg_spectrum_scale),
        ncg_dtype=str(args.ncg_dtype),
        ncg_device=str(args.device),
        use_ncg_braid=bool(args.use_ncg_braid),
        lambda_ncg=float(args.lambda_ncg),
        lambda_ncg_spectral=float(args.lambda_ncg_spectral),
        lambda_ncg_selfadjoint=float(args.lambda_ncg_selfadjoint),
        lambda_ncg_commutator=float(args.lambda_ncg_commutator),
        lambda_ncg_spacing=float(args.lambda_ncg_spacing),
        lambda_ncg_zeta=float(args.lambda_ncg_zeta),
        ncg_target_zeros=ncg_target_zeros,
        use_selberg_trace=bool(args.use_selberg_trace),
        lambda_trace=float(args.lambda_trace),
        trace_sigma=float(args.trace_sigma),
        trace_max_cycles=int(args.trace_max_cycles),
        use_seed_motifs=bool(args.use_seed_motifs),
        seed_motif_frac=float(args.seed_motif_frac),
        seed_mutation_prob=float(args.seed_mutation_prob),
        elite_weight=float(args.elite_weight),
        lambda_comm=float(args.lambda_comm),
        lambda_diversity=float(args.lambda_diversity),
        lambda_length=float(args.lambda_length),
        comm_eps=float(args.comm_eps),
        diversity_sigma=float(args.diversity_sigma),
        target_length=float(args.target_length),
        lambda_dtes_triple=float(args.lambda_dtes_triple),
        lambda_spectral_div=float(args.lambda_spectral_div),
        ncg_eps=float(args.ncg_eps),
        restart_patience=int(args.restart_patience),
        restart_fraction=float(args.restart_fraction),
        exploration_floor=float(args.exploration_floor),
        pheromone_min=float(args.pheromone_min),
        pheromone_max=float(args.pheromone_max),
        bank_prune_fraction=float(args.bank_prune_fraction),
        alpha_anneal=float(args.alpha_anneal),
        beta_anneal=float(args.beta_anneal),
        adaptive_scheduler=str(args.adaptive_scheduler).lower() in ["1", "true", "yes", "y"],
        scheduler_window=int(args.scheduler_window),
        exploration_floor_max=float(args.exploration_floor_max),
        lambda_ncg_max=float(args.lambda_ncg_max),
        lambda_diversity_max=float(args.lambda_diversity_max),
        restart_patience_min=int(args.restart_patience_min),
        spec_clip=float(args.spec_clip),
        reject_nonfinite_spec=str(args.reject_nonfinite_spec).lower() in ["1", "true", "yes", "y"],
    )
    dt = time.perf_counter() - t0

    with open(out_dir / "artin_aco_best.json", "w", encoding="utf-8") as f:
        payload = dict(best)
        payload["wall_time_s"] = float(dt)
        payload["operator_builder"] = str(args.operator_builder)
        json.dump(payload, f, indent=2)

    if (bool(args.use_ncg_braid) or bool(args.use_selberg_trace)) and aco.best_ncg_export is not None:
        with open(out_dir / "artin_aco_best_ncg.json", "w", encoding="utf-8") as f:
            json.dump(aco.best_ncg_export, f, indent=2, allow_nan=True)

    _write_history_csv(out_dir / "artin_aco_history.csv", history, operator_builder=str(args.operator_builder))


if __name__ == "__main__":
    main()

