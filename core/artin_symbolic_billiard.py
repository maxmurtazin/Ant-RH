#!/usr/bin/env python3
"""
Artin symbolic billiard: PSL(2,Z) words γ = S T^{a1} S T^{a2} ... S T^{ak},
hyperbolic length from trace, ACO-style sampling.

Run:
  python3 -m core.artin_symbolic_billiard --num_samples 10000 --max_length 10 --max_power 7 --out_dir runs/
"""

from __future__ import annotations

import os
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.matrix_2x2 import (
    DTYPE_F,
    matmul_2x2,
    matpow_2x2,
    precompute_T_powers,
    power_T,
)

# γ = S T^{a1} S T^{a2} ... S T^{ak} (left-to-right product)
S_SL2 = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=DTYPE_F)
I2 = np.eye(2, dtype=DTYPE_F)


def trace_2x2(M: np.ndarray) -> float:
    return float(M[0, 0] + M[1, 1])


def safe_arccosh(x: np.ndarray | float) -> np.ndarray | float:
    xa = np.asarray(x, dtype=DTYPE_F)
    out = np.log(xa + np.sqrt(np.maximum(xa * xa - 1.0, 0.0) + 1e-12))
    return float(out) if np.ndim(x) == 0 else out


def build_word(
    a_list: List[int],
    T_stack: np.ndarray | None = None,
    offset: int = 0,
) -> np.ndarray:
    """
    M = S T^{a1} S T^{a2} ... S T^{ak}.
    """
    M = I2.copy()
    use_stack = T_stack is not None
    for a in a_list:
        ai = int(a)
        M = matmul_2x2(M, S_SL2)
        if use_stack:
            Ta = T_stack[ai + offset]
        else:
            Ta = power_T(ai)
        M = matmul_2x2(M, Ta)
    return M


def is_hyperbolic_matrix(M: np.ndarray) -> bool:
    return abs(trace_2x2(M)) > 2.0


def hyperbolic_length_from_trace(tr_abs: float) -> float:
    x = max(tr_abs * 0.5, 1.0 + 1e-12)
    return float(2.0 * safe_arccosh(x))


def _minimal_period(a_list: List[int]) -> int:
    L = len(a_list)
    if L <= 1:
        return L
    for p in range(1, L):
        if L % p != 0:
            continue
        ok = True
        for i in range(L):
            if a_list[i] != a_list[i % p]:
                ok = False
                break
        if ok:
            return p
    return L


def _repeated_square_pattern(a_list: List[int]) -> bool:
    L = len(a_list)
    if L < 2 or L % 2 != 0:
        return False
    h = L // 2
    return a_list[:h] == a_list[h:]


def _trace_scaling_power_heuristic(
    M: np.ndarray,
    a_list: List[int],
    T_stack: np.ndarray,
    offset: int,
    atol: float = 1e-8,
) -> bool:
    """
    True => reject primitive (matrix equals k-fold power of block word; trace/length scaling).
    """
    L = len(a_list)
    tr_full = abs(trace_2x2(M))
    ell_full = hyperbolic_length_from_trace(tr_full)

    for k in range(2, min(L, 64) + 1):
        if L % k != 0:
            continue
        plen = L // k
        chunk = a_list[:plen]
        if len(chunk) * k != L or (chunk * k) != a_list:
            continue
        P = build_word(chunk, T_stack, offset)
        try:
            Pk = matpow_2x2(P, k)
        except ValueError:
            continue
        if not np.allclose(M, Pk, atol=atol, rtol=1e-9):
            continue
        tr_p = abs(trace_2x2(P))
        if tr_p <= 2.0:
            continue
        ell_p = hyperbolic_length_from_trace(tr_p)
        if ell_full > 1e-10 and abs(ell_full - k * ell_p) / ell_full < 0.05:
            return True
        if abs(tr_full - abs(trace_2x2(Pk))) < 1e-6 * max(1.0, abs(tr_full)):
            return True
    return False


def is_primitive_heuristic(
    a_list: List[int],
    M: np.ndarray,
    T_stack: np.ndarray,
    offset: int,
) -> bool:
    if _repeated_square_pattern(a_list):
        return False
    mp = _minimal_period(a_list)
    if mp < len(a_list) and len(a_list) // mp >= 2:
        return False
    if _trace_scaling_power_heuristic(M, a_list, T_stack, offset):
        return False
    return True


def extract_feature_dict(
    a_list: List[int],
    M: np.ndarray,
    T_stack: np.ndarray,
    offset: int,
) -> Dict[str, Any]:
    tr = trace_2x2(M)
    tr_abs = abs(tr)
    is_hyp = is_hyperbolic_matrix(M)
    length = hyperbolic_length_from_trace(tr_abs) if is_hyp else float("nan")
    prim = is_primitive_heuristic(a_list, M, T_stack, offset) if is_hyp else False
    return {
        "a_list": list(a_list),
        "trace": float(tr),
        "length": float(length),
        "is_hyperbolic": bool(is_hyp),
        "primitive": bool(prim),
    }


class ArtinWordSampler:
    def __init__(
        self,
        max_length: int,
        max_power: int,
        seed: int = 42,
        rho: float = 0.1,
        temperature: float = 1.0,
    ):
        self.max_length = max_length
        self.max_power = max_power
        self.rng = np.random.default_rng(seed)
        self.rho = rho
        self.temperature = max(temperature, 1e-12)
        self.pheromone: Dict[Tuple[int, int], float] = {}

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        z = logits - np.max(logits)
        ex = np.exp(z / self.temperature)
        s = ex.sum()
        if s <= 0.0:
            return np.full_like(ex, 1.0 / len(ex))
        return ex / s

    def sample_word(self) -> List[int]:
        lo = -self.max_power
        hi = self.max_power + 1
        vals = np.arange(lo, hi, dtype=np.int32)

        a1 = int(self.rng.integers(lo, hi))
        L = int(self.rng.integers(2, self.max_length + 1))

        path = [a1]
        prev = a1
        for _ in range(L - 1):
            logits = np.empty(len(vals), dtype=np.float64)
            for i, na in enumerate(vals):
                logits[i] = self.pheromone.get((prev, int(na)), 1.0)
            probs = self._softmax(logits)
            idx = int(self.rng.choice(len(vals), p=probs))
            next_a = int(vals[idx])
            path.append(next_a)
            prev = next_a
        return path

    def update_pheromone(self, path: List[int], reward: float) -> None:
        rho = self.rho
        for i in range(len(path) - 1):
            key = (path[i], path[i + 1])
            tau = self.pheromone.get(key, 1.0)
            self.pheromone[key] = (1.0 - rho) * tau + reward


def generate_dataset(
    num_samples: int,
    max_length: int,
    max_power: int,
    seed: int = 42,
    log_every: int = 500,
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    T_stack, offset = precompute_T_powers(max_power)
    sampler = ArtinWordSampler(
        max_length=max_length,
        max_power=max_power,
        seed=seed,
    )

    valid: List[Dict[str, Any]] = []
    valid_count = 0
    rejected_count = 0
    sum_ell = 0.0
    max_ell = 0.0
    lengths_for_avg: List[float] = []

    pbar = tqdm(range(num_samples), desc="artin words", unit="it")
    for it in pbar:
        a_list = sampler.sample_word()
        M = build_word(a_list, T_stack, offset)

        feat = extract_feature_dict(a_list, M, T_stack, offset)
        if feat["is_hyperbolic"] and feat["primitive"]:
            valid.append(feat)
            valid_count += 1
            ell = feat["length"]
            if np.isfinite(ell):
                sum_ell += ell
                lengths_for_avg.append(ell)
                max_ell = max(max_ell, ell)
        else:
            rejected_count += 1

        if log_every > 0 and (it + 1) % log_every == 0:
            total = valid_count + rejected_count
            pct = (100.0 * valid_count / total) if total else 0.0
            avg_l = float(sum_ell / len(lengths_for_avg)) if lengths_for_avg else 0.0
            print(
                f"[{it + 1}] valid={pct:.2f}% avg_ℓ={avg_l:.6g} max_ℓ={max_ell:.6g}",
                flush=True,
            )

    stats = {
        "valid_count": float(valid_count),
        "rejected_count": float(rejected_count),
        "avg_length": float(sum_ell / len(lengths_for_avg)) if lengths_for_avg else 0.0,
        "max_length": float(max_ell),
    }
    return valid, stats


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_outputs(out_dir: Path, features: List[Dict[str, Any]]) -> None:
    _ensure_dir(out_dir)

    words_path = out_dir / "artin_words.json"
    with open(words_path, "w", encoding="utf-8") as f:
        json.dump(features, f, indent=2)

    csv_path = out_dir / "artin_lengths.csv"
    lines = ["length,trace,word_len"]
    for feat in features:
        lines.append(
            f"{feat['length']},{feat['trace']},{len(feat['a_list'])}",
        )
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    n = len(features)
    if n == 0:
        np.save(out_dir / "artin_features.npy", np.zeros((0,), dtype=np.float64))
        return

    dt = np.dtype(
        [
            ("trace", np.float64),
            ("length", np.float64),
            ("word_len", np.int32),
            ("is_hyperbolic", np.bool_),
            ("primitive", np.bool_),
            ("a_list", object),
        ]
    )
    arr = np.empty(n, dtype=dt)
    for i, feat in enumerate(features):
        arr[i]["trace"] = feat["trace"]
        arr[i]["length"] = feat["length"]
        arr[i]["word_len"] = len(feat["a_list"])
        arr[i]["is_hyperbolic"] = feat["is_hyperbolic"]
        arr[i]["primitive"] = feat["primitive"]
        arr[i]["a_list"] = np.asarray(feat["a_list"], dtype=np.int32)

    np.save(out_dir / "artin_features.npy", arr)


def main() -> None:
    parser = argparse.ArgumentParser(description="Artin symbolic billiard PSL(2,Z)")
    parser.add_argument("--num_samples", type=int, default=10_000)
    parser.add_argument("--max_length", type=int, default=10)
    parser.add_argument("--max_power", type=int, default=7)
    parser.add_argument("--out_dir", type=str, default="runs/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=500)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    features, st = generate_dataset(
        num_samples=args.num_samples,
        max_length=args.max_length,
        max_power=args.max_power,
        seed=args.seed,
        log_every=args.log_every,
    )
    _write_outputs(out_dir, features)

    total = int(st["valid_count"] + st["rejected_count"])
    pct = (100.0 * st["valid_count"] / total) if total else 0.0
    print(
        f"done: valid={int(st['valid_count'])} rejected={int(st['rejected_count'])} "
        f"valid%={pct:.2f} avg_ℓ={st['avg_length']:.6g} max_ℓ={st['max_length']:.6g}",
        flush=True,
    )


if __name__ == "__main__":
    main()
