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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from core.artin_symbolic_billiard import (
    build_word as build_word_matrix,
    hyperbolic_length_from_trace,
    is_hyperbolic_matrix,
    precompute_T_powers,
    trace_2x2,
)
from core.artin_operator import build_geodesic_kernel, build_laplacian, sample_domain
from validation.selberg_trace_loss import compute_selberg_loss


def load_zeros(path: str) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"zeros file not found: {p}")
    z = np.loadtxt(str(p), dtype=_DTF)
    z = np.asarray(z, dtype=_DTF).reshape(-1)
    z = z[np.isfinite(z)]
    return z


_DTF = np.float64


def _clip(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


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

        self.a_vals = np.arange(-self.max_power, self.max_power + 1, dtype=np.int32)
        self.heuristic_vals = (1.0 / (1.0 + np.abs(self.a_vals).astype(_DTF))).astype(_DTF)
        self._heur_log = np.log(np.maximum(self.heuristic_vals, 1e-18))

        self.pheromone: Dict[Tuple[int, int], float] = {}

        self.best_words: List[List[int]] = []
        self.best_loss: float = float("inf")

        self._T_stack, self._T_offset = precompute_T_powers(self.max_power)

        self._word_cache: Dict[Tuple[int, ...], Tuple[bool, float, float]] = {}
        self._bank: Dict[Tuple[int, ...], Candidate] = {}
        self._op_cache: Dict[Tuple[Tuple[int, ...], ...], float] = {}

    def _tau(self, prev_a: int, next_a: int) -> float:
        return float(self.pheromone.get((int(prev_a), int(next_a)), 1.0))

    def _set_tau(self, prev_a: int, next_a: int, value: float) -> None:
        self.pheromone[(int(prev_a), int(next_a))] = _clip(value, self.tau_min, self.tau_max)

    def sample_word(self) -> List[int]:
        L = int(self.rng.integers(2, self.max_length + 1))
        a1 = int(self.rng.integers(-self.max_power, self.max_power + 1))
        word = [a1]
        prev = a1

        for _ in range(L - 1):
            taus = np.empty(self.a_vals.size, dtype=_DTF)
            for i, a in enumerate(self.a_vals):
                taus[i] = self._tau(prev, int(a))
            taus = np.clip(taus, self.tau_min, self.tau_max)

            logits = self.alpha * np.log(taus) + self.beta * self._heur_log
            probs = _stable_softmax(logits)
            idx = int(self.rng.choice(self.a_vals.size, p=probs))
            nxt = int(self.a_vals[idx])
            word.append(nxt)
            prev = nxt

        return word

    def _validate_and_length(self, a_list: List[int]) -> Tuple[bool, float, float]:
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
        seed: int,
    ) -> float:
        geodesics = self._bank_top_geodesics(top_k_geodesics)
        if not geodesics:
            return float("inf")

        sig = tuple(tuple(int(a) for a in g["a_list"]) for g in geodesics)
        cached = self._op_cache.get(sig)
        if cached is not None:
            return cached

        Z = sample_domain(int(n_points), seed=int(seed))
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
        selberg_sigma: float,
        selberg_m_max: int,
        selberg_bank_top_n: int,
        n_points: int,
        op_sigma: float,
        op_eps: float,
        op_top_k_geodesics: int,
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

        # Compute global losses using current bank
        L_sel = self._selberg_loss(zeros, sigma=selberg_sigma, m_max=selberg_m_max, bank_top_n=selberg_bank_top_n)
        L_spec = self._operator_spectral_loss(
            zeros,
            n_points=n_points,
            op_sigma=op_sigma,
            op_eps=op_eps,
            top_k_geodesics=op_top_k_geodesics,
            seed=self.seed,
        )
        L_spacing = 0.0
        if lambda_spacing != 0.0:
            L_spacing = 0.0

        L_total_global = lambda_selberg * L_sel + lambda_spec * L_spec + lambda_spacing * L_spacing
        if not np.isfinite(L_total_global):
            L_total_global = float("inf")

        # Candidate-specific score: global + tiny length regularizer
        scored: List[Candidate] = []
        for a_list, ell, tr in candidates:
            reg = 1e-3 * float(ell)
            loss = float(L_total_global + reg)
            reward = 1.0 / (loss + 1e-8) if np.isfinite(loss) else 0.0
            reward = min(float(reward), 1e6)
            c = Candidate(a_list=_word_key(a_list), length=float(ell), trace=float(tr), reward=float(reward), loss=float(loss))
            scored.append(c)
            self._bank[c.a_list] = c

        # Prune bank by reward then loss
        bank_items2 = list(self._bank.values())
        bank_items2.sort(key=lambda c: (-c.reward, c.loss))
        self._bank = {c.a_list: c for c in bank_items2[: self.bank_size]}

        stats = {
            "L_selberg": float(L_sel),
            "L_spec": float(L_spec),
            "L_spacing": float(L_spacing),
            "L_total_global": float(L_total_global),
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
        log_top_words: int = 5,
        use_planner: bool = False,
        planner_backend: str = "llama_cpp",
        llama_cli: str = "/Users/machome/llama.cpp/llama-cli",
        planner_model: str = "/Users/machome/models/gemma/gemma-3-1b-it-Q4_K_M.gguf",
        planner_inject_frac: float = 0.2,
        planner_log_path: str = "runs/gemma_planner_log.jsonl",
        planner_replace_frac: float = 0.2,
    ) -> Tuple[Dict[str, Any], List[Tuple[int, float, float]]]:
        history: List[Tuple[int, float, float]] = []
        best_snapshot: Dict[str, Any] = {"best_loss": float("inf"), "best_words": [], "best_lengths": []}
        recent_losses: List[float] = []

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
            t_it0 = time.perf_counter()
            # build raw population
            population: List[List[int]] = [self.sample_word() for _ in range(self.num_ants)]

            # planner injection
            if planner is not None:
                try:
                    ctx = {
                        "best_words": self.best_words,
                        "recent_losses": recent_losses[-20:],
                        "iteration": int(it),
                        "stats": {
                            "valid_rate": float(last_valid_rate),
                            "mean_loss": float(last_mean_loss),
                            "best_loss": float(last_best_loss),
                        },
                    }
                    proposals = planner.suggest_words(ctx)
                    used = False
                    if proposals:
                        n_replace = int(min(len(population), max(1, int(float(self.num_ants) * float(planner_inject_frac)))))
                        n_use = min(n_replace, len(proposals))
                        for j in range(n_use):
                            population[j] = proposals[j]
                        used = n_use > 0

                    try:
                        Path(planner_log_path).parent.mkdir(parents=True, exist_ok=True)
                        with open(planner_log_path, "a", encoding="utf-8") as f:
                            f.write(
                                json.dumps(
                                    {
                                        "iteration": int(it),
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
                ok, ell, tr = self._validate_and_length(w)
                if ok:
                    valids.append((w, ell, tr))

            if not valids:
                self.evaporate()
                history.append((it, float("inf"), float("inf")))
                print(
                    f"[{it}] best=inf mean=inf valid=0 avg_ell=nan bank={len(self._bank)}",
                    flush=True,
                )
                continue

            scored, stats = self.evaluate_iteration(
                candidates=valids,
                zeros=zeros,
                lambda_selberg=float(lambda_selberg),
                lambda_spec=float(lambda_spec),
                lambda_spacing=float(lambda_spacing),
                selberg_sigma=float(selberg_sigma),
                selberg_m_max=int(selberg_m_max),
                selberg_bank_top_n=int(selberg_bank_top_n),
                n_points=int(n_points),
                op_sigma=float(op_sigma),
                op_eps=float(op_eps),
                op_top_k_geodesics=int(op_top_k_geodesics),
            )

            scored.sort(key=lambda c: c.loss)
            best = scored[0]
            mean_loss = float(np.mean([c.loss for c in scored])) if scored else float("inf")
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

            self.evaporate()
            self.reinforce(scored[: self.best_k_ants])

            history.append((it, float(best.loss), float(mean_loss)))
            recent_losses.append(float(best.loss))
            last_valid_rate = float(valid_rate)
            last_mean_loss = float(mean_loss)
            last_best_loss = float(best.loss)

            top_words = scored[: max(1, int(log_top_words))]
            top_words_s = " | ".join(
                [f"{list(c.a_list)} ℓ={c.length:.3g} L={c.loss:.3g} r={c.reward:.3g}" for c in top_words]
            )
            dt = time.perf_counter() - t_it0
            print(
                f"[{it}] best={best.loss:.6g} mean={mean_loss:.6g} valid={len(scored)} "
                f"avg_ell={avg_ell:.6g} bank={len(self._bank)} "
                f"L_sel={stats['L_selberg']:.3g} L_spec={stats['L_spec']:.3g} dt={dt:.3g}s :: {top_words_s}",
                flush=True,
            )

        return best_snapshot, history


def _write_history_csv(path: Path, history: Iterable[Tuple[int, float, float]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("iter,best_loss,mean_loss\n")
        for it, b, m in history:
            f.write(f"{int(it)},{float(b)},{float(m)}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="V12.3 ACO over Artin symbolic words (Selberg + operator losses)")
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
    ap.add_argument("--lambda_spacing", type=float, default=0.0)

    ap.add_argument("--selberg_sigma", type=float, default=0.5)
    ap.add_argument("--selberg_m_max", type=int, default=6)
    ap.add_argument("--selberg_bank_top_n", type=int, default=250)

    ap.add_argument("--n_points", type=int, default=128)
    ap.add_argument("--op_sigma", type=float, default=0.3)
    ap.add_argument("--op_eps", type=float, default=0.6)
    ap.add_argument("--op_top_k_geodesics", type=int, default=250)

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
    args = ap.parse_args()

    zeros = load_zeros(args.zeros)
    if zeros.size == 0:
        raise ValueError("zeros file is empty or unreadable")

    aco = ArtinACO(
        num_ants=int(args.num_ants),
        max_length=int(args.max_length),
        max_power=int(args.max_power),
        alpha=float(args.alpha),
        beta=float(args.beta),
        rho=float(args.rho),
        seed=int(args.seed),
        length_threshold=float(args.length_threshold),
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
        log_top_words=5,
        use_planner=str(args.use_planner).lower() in ["1", "true", "yes", "y"],
        planner_backend=str(args.planner_backend),
        llama_cli=str(args.llama_cli),
        planner_model=str(args.planner_model),
        planner_inject_frac=float(args.planner_inject_frac),
        planner_log_path=str(Path("runs") / "gemma_planner_log.jsonl"),
        planner_replace_frac=float(args.planner_replace_frac),
    )
    dt = time.perf_counter() - t0

    with open(out_dir / "artin_aco_best.json", "w", encoding="utf-8") as f:
        payload = dict(best)
        payload["wall_time_s"] = float(dt)
        json.dump(payload, f, indent=2)

    _write_history_csv(out_dir / "artin_aco_history.csv", history)


if __name__ == "__main__":
    main()

