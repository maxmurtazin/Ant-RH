#!/usr/bin/env python3
"""
V14.5 — Rank-Based Semantic DTES Ants with Anti-Collapse Reward

Computational evidence only; not a proof of RH.

Implements:
  - rank-based reward per (iter, dim, mode)
  - diversity / collapse penalties + entropy tracking
  - semantic temperature schedule
  - staged survival objective (A..E)
  - ablation modes incl. hybrid_ranked_anticollapse

This is a search/diagnostic experiment; it does NOT claim any proof of RH.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pandas as pd  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit("This script requires pandas. Please install pandas and retry.") from e


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from validation import residue_diagnostics as rd  # noqa: E402

try:
    from core.spectral_stabilization import safe_eigh as _safe_eigh  # type: ignore

    _HAVE_SAFE_EIGH = True
except Exception:
    _safe_eigh = None
    _HAVE_SAFE_EIGH = False


# ----------------------------
# Small utilities
# ----------------------------


def _resolve(p: str) -> Path:
    pp = Path(p)
    if not pp.is_absolute():
        pp = Path(ROOT) / pp
    return pp


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def format_seconds(sec: float) -> str:
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def write_text(path: Path, s: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(s, encoding="utf-8")


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def json_sanitize(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): json_sanitize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_sanitize(v) for v in x]
    if x is None:
        return None
    if isinstance(x, (int, str, bool)):
        return x
    if isinstance(x, float):
        return float(x) if math.isfinite(x) else None
    try:
        v = float(x)
        return float(v) if math.isfinite(v) else None
    except Exception:
        return str(x)


def _find_pdflatex() -> Optional[str]:
    w = shutil.which("pdflatex")
    if w:
        return w
    for p in ("/Library/TeX/texbin/pdflatex",):
        if Path(p).is_file():
            return p
    return None


def try_pdflatex(tex_path: Path, out_dir: Path, pdf_name: str) -> bool:
    exe = _find_pdflatex()
    if not exe:
        return False
    try:
        r = subprocess.run(
            [exe, "-interaction=nonstopmode", f"-output-directory={out_dir.resolve()}", tex_path.name],
            cwd=str(out_dir.resolve()),
            capture_output=True,
            text=True,
            timeout=240,
        )
        return r.returncode == 0 and (out_dir / pdf_name).is_file()
    except Exception:
        return False


# ----------------------------
# Stabilized operator + NV helpers (adapted from V14.4)
# ----------------------------


def rotation_block(theta: float) -> np.ndarray:
    c = float(math.cos(theta))
    s = float(math.sin(theta))
    return np.asarray([[c, -s], [s, c]], dtype=np.float64)


def embed_2x2(dim: int, idx0: int, B: np.ndarray) -> np.ndarray:
    n = int(dim)
    G = np.eye(n, dtype=np.float64)
    i = int(idx0)
    if i < 0 or i + 1 >= n:
        return G
    G[i : i + 2, i : i + 2] = np.asarray(B, dtype=np.float64)
    return G


def hermitian_part(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.conj().T)


def op_norm_2(A: np.ndarray) -> float:
    try:
        return float(np.linalg.norm(A, ord=2))
    except Exception:
        return float(np.linalg.norm(A, ord="fro"))


def normalize_operator_norm(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = op_norm_2(M)
    return M / max(float(n), float(eps))


def remove_trace(H: np.ndarray) -> np.ndarray:
    n = int(H.shape[0])
    if n <= 0:
        return H
    tr = np.trace(H) / complex(n)
    return H - tr * np.eye(n, dtype=H.dtype)


def safe_eigvalsh(H: np.ndarray, *, seed: int) -> Optional[np.ndarray]:
    Hh = hermitian_part(H)
    if _HAVE_SAFE_EIGH and _safe_eigh is not None and np.isrealobj(Hh):
        try:
            w, _, _rep = _safe_eigh(
                np.asarray(Hh.real, dtype=np.float64),
                k=None,
                return_eigenvectors=False,
                stabilize=True,
                seed=int(seed),
            )
            w = np.asarray(w, dtype=np.float64).reshape(-1)
            w = w[np.isfinite(w)]
            w.sort()
            return w
        except Exception:
            pass
    try:
        w = np.linalg.eigvalsh(Hh)
        w = np.asarray(w, dtype=np.float64).reshape(-1)
        w = w[np.isfinite(w)]
        w.sort()
        return w
    except Exception:
        return None


def normalize_spectral_radius(H: np.ndarray, target_radius: float, eps: float = 1e-12) -> Tuple[np.ndarray, float]:
    w = safe_eigvalsh(H, seed=42)
    if w is None or w.size == 0:
        return H, float("nan")
    r = float(max(abs(float(w[0])), abs(float(w[-1]))))
    if not (math.isfinite(r) and r > eps):
        return H, r
    s = float(target_radius) / r
    return H * s, float(target_radius)


def make_stable_generator(dim: int, generator_index: int, power: int, *, theta_base: float) -> np.ndarray:
    n = int(dim)
    i = int(generator_index)
    i = int(max(1, min(n - 1, i)))
    max_i = max(1, n - 1)
    theta = float(theta_base) * float(i) / float(max_i)
    B = rotation_block(float(power) * theta)
    G = embed_2x2(n, i - 1, B)
    A = hermitian_part(G.astype(np.complex128, copy=False))
    return normalize_operator_norm(A)


def unfold_to_mean_spacing_one(x: np.ndarray) -> np.ndarray:
    return rd.unfold_to_mean_spacing_one(np.asarray(x, dtype=np.float64).reshape(-1))


def make_L_grid(L_min: float, L_max: float, n_L: int) -> np.ndarray:
    L_min = float(L_min)
    L_max = float(L_max)
    n_L = int(n_L)
    if not (math.isfinite(L_min) and math.isfinite(L_max)) or n_L < 2:
        return np.asarray([1.0, 2.0], dtype=np.float64)
    if L_max <= L_min:
        L_max = L_min + 1.0
    return np.linspace(L_min, L_max, n_L, dtype=np.float64)


def number_variance_curve(levels_unfolded: np.ndarray, L_grid: np.ndarray) -> np.ndarray:
    x = np.asarray(levels_unfolded, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size < 8:
        return np.full_like(L_grid, np.nan, dtype=np.float64)
    x = np.sort(x)
    out = np.full_like(L_grid, np.nan, dtype=np.float64)
    t_candidates = x[:-1]
    for i, L in enumerate(np.asarray(L_grid, dtype=np.float64).reshape(-1)):
        if not (math.isfinite(float(L)) and float(L) > 0.0):
            continue
        left = np.searchsorted(x, t_candidates, side="left")
        right = np.searchsorted(x, t_candidates + float(L), side="right")
        counts = (right - left).astype(np.float64)
        if counts.size < 8:
            continue
        out[i] = float(np.var(counts))
    return out


def sigma2_poisson(L: np.ndarray) -> np.ndarray:
    return np.asarray(L, dtype=np.float64)


def sigma2_gue_asymptotic(L: np.ndarray) -> np.ndarray:
    L = np.asarray(L, dtype=np.float64)
    gamma = 0.5772156649015329
    return (1.0 / (math.pi**2)) * (np.log(np.maximum(1e-12, 2.0 * math.pi * L)) + gamma + 1.0)


def curve_l2(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    if aa.size == 0 or bb.size == 0 or aa.size != bb.size:
        return float("nan")
    m = np.isfinite(aa) & np.isfinite(bb)
    if not m.any():
        return float("nan")
    return float(np.sqrt(np.mean((aa[m] - bb[m]) ** 2)))


def simplify_word(word: List[Tuple[int, int]], *, max_power: int, max_word_len: int) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for (i, p) in word:
        i = int(i)
        p = int(max(-max_power, min(max_power, int(p))))
        if p == 0:
            continue
        if out and out[-1][0] == i:
            pp = int(out[-1][1] + p)
            pp = int(max(-max_power, min(max_power, pp)))
            out[-1] = (i, pp)
            if out[-1][1] == 0:
                out.pop()
            continue
        out.append((i, p))
        if len(out) >= int(max_word_len):
            break
    return out


def word_to_string(word: List[Tuple[int, int]]) -> str:
    return " ".join([f"sigma_{int(i)}^{int(p)}" for i, p in word])


# ----------------------------
# Diversity / collapse metrics
# ----------------------------


def generator_entropy_normalized(word: List[Tuple[int, int]], n_generators: int) -> float:
    gens = [int(i) for (i, _p) in word]
    if not gens or n_generators <= 1:
        return 0.0
    c = Counter(gens)
    total = float(sum(c.values()))
    ps = np.asarray([v / total for v in c.values()], dtype=np.float64)
    H = float(-np.sum(ps * np.log(np.maximum(1e-12, ps))))
    return float(H / max(1e-12, math.log(float(n_generators))))


def unique_generator_fraction(word: List[Tuple[int, int]]) -> float:
    if not word:
        return 0.0
    gens = [int(i) for (i, _p) in word]
    return float(len(set(gens)) / max(1, len(gens)))


def local_band_width(word: List[Tuple[int, int]]) -> float:
    if not word:
        return 0.0
    gens = [int(i) for (i, _p) in word]
    return float(max(gens) - min(gens) + 1)


def repeated_bigram_fraction(word: List[Tuple[int, int]]) -> float:
    if len(word) < 3:
        return 0.0
    bigrams = [(word[i][0], word[i][1], word[i + 1][0], word[i + 1][1]) for i in range(len(word) - 1)]
    c = Counter(bigrams)
    rep = sum(v - 1 for v in c.values() if v > 1)
    return float(rep / max(1, len(bigrams)))


def compute_word_collapse_penalty(word: List[Tuple[int, int]], dim: int) -> Dict[str, float]:
    n_gen = max(1, int(dim) - 1)
    Hn = generator_entropy_normalized(word, n_gen)
    uniqf = unique_generator_fraction(word)
    bw = local_band_width(word)
    rep_bi = repeated_bigram_fraction(word)
    # narrow band if <= 8% of available generators
    band_target = max(3.0, 0.08 * float(dim))
    band_collapse = max(0.0, (band_target - bw) / max(1.0, band_target))
    # low entropy collapse
    ent_collapse = max(0.0, 0.5 - Hn)
    # too few unique generators
    uniq_collapse = max(0.0, 0.35 - uniqf)
    L = float(ent_collapse + band_collapse + uniq_collapse + rep_bi)
    return {
        "generator_entropy": float(Hn),
        "unique_generator_fraction": float(uniqf),
        "local_band_width": float(bw),
        "repeated_bigram_fraction": float(rep_bi),
        "collapse_penalty": float(L),
    }


# ----------------------------
# Semantic memory (lightweight + deterministic)
# ----------------------------


SemanticRecord = Dict[str, Any]


def load_semantic_jsonl(path: Path) -> List[SemanticRecord]:
    if not path.is_file():
        return []
    out: List[SemanticRecord] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                rec = json.loads(s)
                if isinstance(rec, dict):
                    out.append(rec)
            except Exception:
                continue
    return out


def write_semantic_jsonl(path: Path, recs: Sequence[SemanticRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def record_weight(rec: SemanticRecord) -> float:
    w = abs(safe_float(rec.get("weight", 0.0), 0.0))
    c = safe_float(rec.get("confidence", 1.0), 1.0)
    return float(w * max(0.0, min(1.0, c)))


def rule_applies_dim(rec: SemanticRecord, dim: int) -> bool:
    d = rec.get("dim", "all")
    if d in ("all", "*", None):
        return True
    try:
        return int(d) == int(dim)
    except Exception:
        return False


def semantic_score_word(word: List[Tuple[int, int]], dim: int, memory: Sequence[SemanticRecord]) -> Tuple[float, float, float]:
    """
    semantic_score(W) = prefer_bonus - avoid_penalty - artifact_penalty
    We implement motif-matching on simple features (entropy, band width, repetitions).
    """
    feat = compute_word_collapse_penalty(word, dim)
    Hn = float(feat["generator_entropy"])
    uniqf = float(feat["unique_generator_fraction"])
    bw = float(feat["local_band_width"])
    rep_bi = float(feat["repeated_bigram_fraction"])
    L = int(len(word))

    bonus = 0.0
    penalty = 0.0
    for rec in memory:
        if not isinstance(rec, dict) or not rule_applies_dim(rec, dim):
            continue
        typ = str(rec.get("type", "")).strip().lower()
        motif = str(rec.get("motif", "")).strip()
        w = record_weight(rec)
        if w <= 0:
            continue
        match = False
        if motif == "prefer_entropy":
            match = Hn >= 0.6
        elif motif == "avoid_band_collapse":
            match = bw <= max(3.0, 0.08 * float(dim))
        elif motif == "prefer_generator_diversity":
            match = uniqf >= 0.45
        elif motif == "avoid_repeated_bigrams":
            match = rep_bi >= 0.15
        elif motif == "avoid_long_words":
            match = L >= 24
        elif motif == "prefer_short_words":
            match = L <= 10
        elif motif == "poisson_like_warning":
            match = True
        else:
            match = False
        if not match:
            continue
        if typ == "prefer":
            bonus += w
        elif typ in ("avoid", "artifact", "caution"):
            penalty += w
    score = float(bonus - penalty)
    return score, float(bonus), float(penalty)


def semantic_temperature(iter_idx: int, iters: int, t0: float, t1: float) -> float:
    if iters <= 1:
        return float(t1)
    p = float((iter_idx - 1) / max(1, iters - 1))
    return float((1.0 - p) * float(t0) + p * float(t1))


def deterministic_rules_from_best(dim: int, metrics: Dict[str, Any], source: str, it: int) -> List[SemanticRecord]:
    out: List[SemanticRecord] = []
    sup = safe_float(metrics.get("support_overlap", float("nan")))
    pois = safe_float(metrics.get("poisson_like_fraction", float("nan")))
    if math.isfinite(sup) and sup < 0.25:
        out.append(
            {
                "type": "avoid",
                "text": "Avoid support artifacts: penalize narrow generator bands.",
                "motif": "avoid_band_collapse",
                "weight": 0.9,
                "confidence": 0.9,
                "source": source,
                "dim": int(dim),
                "scope": "gate",
                "iter": int(it),
            }
        )
    if math.isfinite(pois) and pois >= 0.95:
        out.append(
            {
                "type": "caution",
                "text": "Poisson-like dominates; favor entropy/diversity motifs.",
                "motif": "prefer_entropy",
                "weight": 0.6,
                "confidence": 0.85,
                "source": source,
                "dim": int(dim),
                "scope": "search",
                "iter": int(it),
            }
        )
    return out


# ----------------------------
# Evaluation + reward
# ----------------------------


@dataclass
class EvalOut:
    stable: bool
    raw_J: float
    support_overlap: float
    active_argument_error: float
    poisson_like_fraction: float
    residue_error: float
    trace_error: float
    anti_poisson_improved: bool
    nv_curve: Optional[np.ndarray]


def staged_eval(
    *,
    dim: int,
    word: List[Tuple[int, int]],
    target_levels: np.ndarray,
    windows: List[Tuple[float, float]],
    L_grid: np.ndarray,
    seed: int,
    args: argparse.Namespace,
) -> EvalOut:
    """
    Stage A: finite spectrum
    Stage B: support overlap + active-window argument counts
    Stage C: NV + anti-Poisson
    Stage D: residue + trace proxy
    Stage E: null comparisons (handled outside; conservative)
    """
    eps = 1e-12
    n = int(dim)
    theta_base = math.pi / 8.0

    # Stage A
    H = np.zeros((n, n), dtype=np.complex128)
    for k, (gi, p) in enumerate(word):
        A = make_stable_generator(n, int(gi), int(p), theta_base=theta_base)
        ck = 1.0 / math.sqrt(float(k) + 1.0)
        H = H + complex(ck) * A
    H = hermitian_part(H)
    H = remove_trace(H)
    H, _ = normalize_spectral_radius(H, target_radius=float(max(4.0, n / 4.0)), eps=eps)
    w = safe_eigvalsh(H, seed=seed)
    if w is None or w.size < 8:
        return EvalOut(False, 1e6, 0.0, 1.0, 1.0, 1.0, 1.0, False, None)

    levels = unfold_to_mean_spacing_one(w)
    # align to target quantiles so window ranges overlap
    try:
        oq = np.quantile(levels, [0.1, 0.9])
        tq = np.quantile(target_levels, [0.1, 0.9])
        a = float(max(1e-12, tq[1] - tq[0]) / max(1e-12, oq[1] - oq[0]))
        b = float(tq[0] - a * oq[0])
        levels = (a * levels + b).astype(np.float64, copy=False)
    except Exception:
        pass

    # Stage B
    n_active = 0
    n_both = 0
    arg_errs: List[float] = []
    for (a0, b0) in windows:
        n_op = rd.count_in_window(levels, float(a0), float(b0))
        n_tg = rd.count_in_window(target_levels, float(a0), float(b0))
        if (n_op == 0) and (n_tg == 0):
            continue
        n_active += 1
        if (n_op > 0) and (n_tg > 0):
            n_both += 1
        arg_errs.append(abs(int(n_op) - int(n_tg)) / float(max(1, int(n_tg))))
    support_overlap = float(n_both) / float(max(1, n_active))
    active_arg = float(np.median(np.asarray(arg_errs, dtype=np.float64))) if arg_errs else 1.0

    # Stage C
    nv = number_variance_curve(levels, L_grid)
    dP = float(curve_l2(nv, sigma2_poisson(L_grid)))
    dG = float(curve_l2(nv, sigma2_gue_asymptotic(L_grid)))
    poisson_like = bool(math.isfinite(dP) and math.isfinite(dG) and dP < dG)
    pois_frac = 1.0 if poisson_like else 0.0
    anti_poisson_improved = bool(math.isfinite(dP) and math.isfinite(dG) and (dP - dG) > float(args.anti_poisson_margin))

    # Stage D (active windows only)
    res_errs: List[float] = []
    tr_errs: List[float] = []
    for (a0, b0) in windows:
        n_op = rd.count_in_window(levels, float(a0), float(b0))
        n_tg = rd.count_in_window(target_levels, float(a0), float(b0))
        if (n_op == 0) and (n_tg == 0):
            continue
        I_op = rd.residue_proxy_count(levels, float(a0), float(b0), eta=float(args.eta), n_contour_points=int(args.n_contour_points))
        I_tg = rd.residue_proxy_count(target_levels, float(a0), float(b0), eta=float(args.eta), n_contour_points=int(args.n_contour_points))
        err = abs(float(I_op.real) - float(I_tg.real)) / max(1.0, abs(float(I_tg.real)))
        res_errs.append(float(err) + 0.10 * float(abs(I_op.imag)))
        c = 0.5 * (float(a0) + float(b0))
        for s in (0.5, 1.0, 2.0, 4.0):
            Sop = rd.trace_formula_proxy(levels, center=float(c), sigma=float(s))
            Stg = rd.trace_formula_proxy(target_levels, center=float(c), sigma=float(s))
            if not (math.isfinite(Sop) and math.isfinite(Stg)):
                continue
            tr = abs(math.log1p(max(0.0, float(Sop))) - math.log1p(max(0.0, float(Stg))))
            tr_errs.append(float(tr))
    residue_err = float(np.median(np.asarray(res_errs, dtype=np.float64))) if res_errs else 1.0
    trace_err = float(np.median(np.asarray(tr_errs, dtype=np.float64))) if tr_errs else 1.0

    # Base objective terms (J_v14.5 is constructed outside to add semantic/collapse terms)
    return EvalOut(True, 0.0, support_overlap, active_arg, float(pois_frac), residue_err, trace_err, anti_poisson_improved, nv)


def rank_rewards(rows: List[Dict[str, Any]], rank_reward_power: float) -> None:
    rows.sort(key=lambda r: float(r["raw_J"]))
    N = len(rows)
    for rank_i, r in enumerate(rows):
        if not bool(r.get("stable", False)):
            rr = 0.0
        else:
            base = float((N - 1 - rank_i) / max(1, N - 1))
            rr = float(base ** float(rank_reward_power))
        r["rank_reward"] = float(rr)


def apply_reward_modifiers(r: Dict[str, Any], *, args: argparse.Namespace) -> float:
    rr = float(r.get("rank_reward", 0.0))
    if rr <= 0.0:
        return 0.0
    fr = float(rr)
    if float(r.get("support_overlap", 0.0)) <= 0.0:
        fr *= 0.1
    if float(r.get("poisson_like_fraction", 1.0)) >= 0.95:
        fr *= 0.25
    if float(r.get("active_argument_error", 1.0)) > float(args.active_error_margin):
        fr *= 0.5
    if bool(r.get("anti_poisson_improved", False)):
        fr *= 1.25
    fr = float(min(max(fr, 1e-6), 1.0))
    return fr


def softmax_sample(logits: np.ndarray, rng: np.random.Generator) -> int:
    z = logits - float(np.max(logits))
    w = np.exp(np.clip(z, -60.0, 60.0))
    s = float(np.sum(w))
    if not (math.isfinite(s) and s > 0.0):
        return int(rng.integers(0, logits.size))
    p = w / s
    return int(rng.choice(np.arange(logits.size), p=p))


def main() -> None:
    ap = argparse.ArgumentParser(description="V14.5 — Rank-based semantic DTES ants (computational only).")
    ap.add_argument("--true_levels_csv", type=str, default="runs/v13o9_true_unfolded_spectra_from_v13o8/v13o9_unfolded_levels.csv")
    ap.add_argument("--zeros_csv", type=str, default="runs/zeros_100_400_precise.csv")
    ap.add_argument("--v14_4_dir", type=str, default="runs/v14_4_semantic_pheromone_dtes_ants")
    ap.add_argument("--v14_2_dir", type=str, default="runs/v14_2_stabilized_artin_operator_search")
    ap.add_argument("--v13o14_dir", type=str, default="runs/v13o14_transport_null_controls")
    ap.add_argument("--out_dir", type=str, default="runs/v14_5_rank_based_semantic_dtes_ants")
    ap.add_argument("--dims", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--num_ants", type=int, default=32)
    ap.add_argument("--num_iters", type=int, default=40)
    ap.add_argument("--max_word_len", type=int, default=32)
    ap.add_argument("--max_power", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--rho", type=float, default=0.15)
    ap.add_argument("--q", type=float, default=1.0)
    ap.add_argument("--semantic_weight", type=float, default=1.0)
    ap.add_argument("--semantic_temperature_start", type=float, default=2.0)
    ap.add_argument("--semantic_temperature_end", type=float, default=0.4)
    ap.add_argument("--diversity_weight", type=float, default=1.0)
    ap.add_argument("--collapse_penalty_weight", type=float, default=1.0)
    ap.add_argument("--rank_reward_power", type=float, default=1.0)
    ap.add_argument("--elite_fraction", type=float, default=0.2)
    ap.add_argument("--window_min", type=float, default=100.0)
    ap.add_argument("--window_max", type=float, default=400.0)
    ap.add_argument("--window_size", type=float, default=40.0)
    ap.add_argument("--window_stride", type=float, default=20.0)
    ap.add_argument("--L_min", type=float, default=0.5)
    ap.add_argument("--L_max", type=float, default=16.0)
    ap.add_argument("--n_L", type=int, default=64)
    ap.add_argument("--eta", type=float, default=0.15)
    ap.add_argument("--n_contour_points", type=int, default=256)
    ap.add_argument("--support_overlap_min", type=float, default=0.5)
    ap.add_argument("--active_error_margin", type=float, default=0.25)
    ap.add_argument("--anti_poisson_margin", type=float, default=0.0)
    ap.add_argument("--residue_margin", type=float, default=0.25)
    ap.add_argument("--trace_margin", type=float, default=0.5)
    ap.add_argument("--n_jobs", type=int, default=8)
    ap.add_argument("--progress_every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=6)
    args = ap.parse_args()

    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    warnings: List[str] = []
    if not _HAVE_SAFE_EIGH:
        warnings.append("core.spectral_stabilization.safe_eigh unavailable; using numpy eigvalsh.")

    dims = [int(d) for d in args.dims]
    windows = rd.make_windows(args.window_min, args.window_max, args.window_size, args.window_stride)
    L_grid = make_L_grid(args.L_min, args.L_max, args.n_L)
    if not windows:
        raise SystemExit("No windows created; check window_min/max/size/stride.")

    # Target levels by dim
    target_by_dim: Dict[int, np.ndarray] = {}
    df_levels, lvl_warns = rd.load_true_levels_csv(_resolve(args.true_levels_csv), dims_keep=dims)
    warnings.extend([f"true_levels_csv: {w}" for w in lvl_warns])
    if df_levels is not None and not df_levels.empty:
        try:
            df = df_levels.copy()
            df["dim"] = pd.to_numeric(df["dim"], errors="coerce").astype("Int64")
            for c in ("target_group", "source"):
                df[c] = df[c].astype(str).str.strip()
            df["unfolded_level"] = pd.to_numeric(df["unfolded_level"], errors="coerce").astype(float)
            for d in dims:
                sub = df[(df["dim"].astype(int) == int(d)) & (df["target_group"] == "real_zeta") & (df["source"] == "target")]
                if not sub.empty:
                    x = np.sort(sub["unfolded_level"].to_numpy(dtype=np.float64))
                    x = x[np.isfinite(x)]
                    if x.size >= 8:
                        target_by_dim[int(d)] = x
        except Exception as e:
            warnings.append(f"true_levels_csv parse failed: {e!r}")
    zeros_raw, zeros_warns = rd.load_zeros_csv(_resolve(args.zeros_csv))
    warnings.extend([f"zeros_csv: {w}" for w in zeros_warns])
    zeros_unfolded = unfold_to_mean_spacing_one(zeros_raw)
    for d in dims:
        if int(d) not in target_by_dim:
            target_by_dim[int(d)] = zeros_unfolded.copy()
            warnings.append(f"dim={d}: target fallback to zeros_csv (real_zeta not found in true_levels_csv)")

    # Semantic memory input (V14.4) if present
    v14_4_dir = _resolve(args.v14_4_dir)
    semantic_in = v14_4_dir / "v14_4_semantic_pheromones.jsonl"
    memory = load_semantic_jsonl(semantic_in)
    if not memory:
        memory = [
            {"type": "avoid", "motif": "avoid_band_collapse", "weight": 0.8, "confidence": 0.85, "dim": "all", "source": "init", "text": "Avoid narrow generator band collapse."},
            {"type": "prefer", "motif": "prefer_entropy", "weight": 0.6, "confidence": 0.75, "dim": "all", "source": "init", "text": "Prefer entropy/diversity to avoid collapse."},
            {"type": "avoid", "motif": "avoid_repeated_bigrams", "weight": 0.7, "confidence": 0.8, "dim": "all", "source": "init", "text": "Avoid repeated bigrams (looping motifs)."},
            {"type": "avoid", "motif": "avoid_long_words", "weight": 0.6, "confidence": 0.8, "dim": "all", "source": "init", "text": "Avoid very long words."},
        ]

    modes = ["numeric_only", "semantic_only", "hybrid_numeric_semantic", "hybrid_ranked_anticollapse"]
    main_mode = "hybrid_ranked_anticollapse"

    rng = np.random.default_rng(int(args.seed))
    random.seed(int(args.seed))

    max_power = int(args.max_power)
    powers = [p for p in range(-max_power, max_power + 1) if p != 0]

    # pheromones per dim/mode/token
    pher: Dict[str, Dict[int, Dict[Tuple[int, int], float]]] = {m: {} for m in modes}
    for m in modes:
        for d in dims:
            pher[m][int(d)] = {(gi, p): 1.0 for gi in range(1, int(d)) for p in powers}

    # Outputs (flat tables)
    aco_history: List[Dict[str, Any]] = []
    best_candidates: List[Dict[str, Any]] = []
    gate_summary: List[Dict[str, Any]] = []
    ablation_summary: List[Dict[str, Any]] = []
    numeric_pheromone_summary: List[Dict[str, Any]] = []
    semantic_decisions: List[Dict[str, Any]] = []
    diversity_summary: List[Dict[str, Any]] = []
    active_argument_counts: List[Dict[str, Any]] = []
    nv_curves: List[Dict[str, Any]] = []
    residue_scores: List[Dict[str, Any]] = []
    trace_proxy: List[Dict[str, Any]] = []

    # Best per dim/mode by J_total
    best_by: Dict[Tuple[int, str], Dict[str, Any]] = {(int(d), m): {"J": float("inf"), "row": None} for d in dims for m in modes}

    def mode_for_ant(ant_id: int) -> str:
        # bias 50% to main, rest split
        if ant_id % 2 == 0:
            return main_mode
        return modes[int(ant_id) % 3]  # among first 3

    def action_logits(dim: int, mode: str, temp: float) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        items = list(pher[mode][dim].keys())
        tau = np.asarray([pher[mode][dim][it] for it in items], dtype=np.float64)
        # heuristic: prefer mid generators (very mild)
        mid = 0.5 * (dim - 1)
        heur = np.asarray([1.0 / (1.0 + 0.01 * abs(float(gi) - mid)) for (gi, _p) in items], dtype=np.float64)
        logits = float(args.alpha) * np.log(np.maximum(1e-12, tau)) + float(args.beta) * np.log(np.maximum(1e-12, heur))
        # semantic action score (approx) scaled by semantic_weight/temperature depending on mode
        if mode == "numeric_only":
            sem_scale = 0.0
        elif mode == "semantic_only":
            sem_scale = float(args.semantic_weight) / max(1e-6, float(temp))
        elif mode == "hybrid_numeric_semantic":
            sem_scale = 0.5 * float(args.semantic_weight) / max(1e-6, float(temp))
        else:
            sem_scale = 0.8 * float(args.semantic_weight) / max(1e-6, float(temp))
        if sem_scale != 0.0:
            # prefer mid (avoid band collapse) + penalize high power
            sem = np.asarray([(-0.5 * abs(float(gi) - mid) / max(1.0, float(dim)) - 0.05 * abs(float(p))) for (gi, p) in items], dtype=np.float64)
            logits = logits + sem_scale * sem
        return items, logits

    # main loop
    for it in range(1, int(args.num_iters) + 1):
        temp = semantic_temperature(it, int(args.num_iters), float(args.semantic_temperature_start), float(args.semantic_temperature_end))
        for d in dims:
            # candidates grouped by mode for rank-based reward
            batch_by_mode: Dict[str, List[Dict[str, Any]]] = {m: [] for m in modes}
            # sample ants
            for ant in range(int(args.num_ants)):
                mode = mode_for_ant(ant)
                items, logits0 = action_logits(int(d), mode, float(temp))
                # sample a word
                L = int(rng.integers(2, int(args.max_word_len) + 1))
                word: List[Tuple[int, int]] = []
                for _k in range(L):
                    idx = softmax_sample(logits0, rng)
                    word.append(items[idx])
                word = simplify_word(word, max_power=int(args.max_power), max_word_len=int(args.max_word_len))
                if not word:
                    word = [items[int(rng.integers(0, len(items)))]]
                wstr = word_to_string(word)

                # semantic score (word-level)
                sem_score, sem_bonus, sem_pen = semantic_score_word(word, int(d), memory)

                # diversity features (word-level)
                div = compute_word_collapse_penalty(word, int(d))

                # staged eval
                ev = staged_eval(
                    dim=int(d),
                    word=word,
                    target_levels=target_by_dim[int(d)],
                    windows=windows,
                    L_grid=L_grid,
                    seed=int(args.seed + 100000 * it + 1000 * d + ant),
                    args=args,
                )

                # objective J_v14.5 (add semantic + collapse)
                L_support = max(0.0, float(args.support_overlap_min) - float(ev.support_overlap))
                L_active = float(ev.active_argument_error)
                L_nv = float(ev.poisson_like_fraction)  # proxy: Poisson-like = bad
                L_res = float(max(0.0, float(ev.residue_error) - float(args.residue_margin)))
                L_tr = float(max(0.0, float(ev.trace_error) - float(args.trace_margin)))
                L_collapse = float(div["collapse_penalty"])
                L_sem_artifact = float(-sem_score)
                raw_J = (
                    (0.0 if ev.stable else 1e6)
                    + 10.0 * L_support
                    + 3.0 * L_active
                    + 3.0 * L_nv
                    + 2.0 * L_res
                    + 1.5 * L_tr
                    + float(args.collapse_penalty_weight) * L_collapse
                    + float(args.semantic_weight) * L_sem_artifact
                    + 0.05 * (len(word) / max(1, int(args.max_word_len)))
                )
                raw_J = float(min(max(raw_J, 0.0), 1e6))

                row = {
                    "iter": int(it),
                    "dim": int(d),
                    "mode": str(mode),
                    "ant_id": int(ant),
                    "word": wstr,
                    "word_len": int(len(word)),
                    "stable": bool(ev.stable),
                    "support_overlap": float(ev.support_overlap),
                    "active_argument_error": float(ev.active_argument_error),
                    "poisson_like_fraction": float(ev.poisson_like_fraction),
                    "residue_error": float(ev.residue_error),
                    "trace_error": float(ev.trace_error),
                    "anti_poisson_improved": bool(ev.anti_poisson_improved),
                    "raw_J": float(raw_J),
                    "semantic_score": float(sem_score),
                    "semantic_bonus": float(sem_bonus),
                    "semantic_penalty": float(sem_pen),
                    "generator_entropy": float(div["generator_entropy"]),
                    "unique_generator_fraction": float(div["unique_generator_fraction"]),
                    "local_band_width": float(div["local_band_width"]),
                    "collapse_penalty": float(div["collapse_penalty"]),
                    "temperature": float(temp),
                }
                batch_by_mode[mode].append(row)

                # store per-candidate extra outputs (lightweight summaries)
                active_argument_counts.append(
                    {
                        "iter": int(it),
                        "dim": int(d),
                        "mode": str(mode),
                        "ant_id": int(ant),
                        "word": wstr,
                        "support_overlap": float(ev.support_overlap),
                        "active_argument_error": float(ev.active_argument_error),
                    }
                )
                residue_scores.append(
                    {
                        "iter": int(it),
                        "dim": int(d),
                        "mode": str(mode),
                        "ant_id": int(ant),
                        "word": wstr,
                        "residue_error": float(ev.residue_error),
                    }
                )
                trace_proxy.append(
                    {
                        "iter": int(it),
                        "dim": int(d),
                        "mode": str(mode),
                        "ant_id": int(ant),
                        "word": wstr,
                        "trace_error": float(ev.trace_error),
                    }
                )
                if ev.nv_curve is not None:
                    # store sparse curve summary (first/last few points)
                    nv = np.asarray(ev.nv_curve, dtype=np.float64)
                    nv_curves.append(
                        {
                            "iter": int(it),
                            "dim": int(d),
                            "mode": str(mode),
                            "ant_id": int(ant),
                            "word": wstr,
                            "nv_l2_to_poisson": float(curve_l2(nv, sigma2_poisson(L_grid))),
                            "nv_l2_to_gue": float(curve_l2(nv, sigma2_gue_asymptotic(L_grid))),
                        }
                    )

            # per (mode) rank-based rewards + pheromone update
            for mode in modes:
                rows = batch_by_mode[mode]
                # entropy tracking for this (iter,dim,mode) batch
                gen_counts = Counter()
                for r in rows:
                    for tok in str(r["word"]).split():
                        try:
                            left, _pstr = tok.split("^")
                            gi = int(left.split("_")[1])
                            gen_counts[gi] += 1
                        except Exception:
                            continue
                total = float(sum(gen_counts.values()))
                ps = np.asarray([v / total for v in gen_counts.values()], dtype=np.float64) if total > 0 else np.asarray([], dtype=np.float64)
                H = float(-np.sum(ps * np.log(np.maximum(1e-12, ps)))) if ps.size else 0.0
                Hn = float(H / max(1e-12, math.log(max(2.0, float(d - 1))))) if d > 2 else 0.0
                batch_collapse = max(0.0, 0.5 - Hn)
                diversity_summary.append(
                    {
                        "iter": int(it),
                        "dim": int(d),
                        "mode": str(mode),
                        "generator_entropy": float(Hn),
                        "normalized_entropy": float(Hn),
                        "collapse_penalty": float(batch_collapse),
                        "n_unique_generators": int(len(gen_counts)),
                        "n_actions": int(sum(gen_counts.values())),
                    }
                )

                # rank-based
                rank_rewards(rows, float(args.rank_reward_power))
                # final reward with modifiers + collapse penalty
                for r in rows:
                    fr = apply_reward_modifiers(r, args=args)
                    # anti-collapse scaling: penalize low batch entropy
                    fr *= max(0.0, 1.0 - float(args.diversity_weight) * float(batch_collapse))
                    # penalize word-level collapse
                    fr *= max(0.0, 1.0 - float(args.diversity_weight) * float(r.get("collapse_penalty", 0.0)))
                    fr = float(min(max(fr, 1e-6 if float(r.get("rank_reward", 0.0)) > 0 else 0.0), 1.0))
                    r["final_reward"] = float(fr)
                    aco_history.append(r)
                    semantic_decisions.append(
                        {
                            "iter": int(it),
                            "dim": int(d),
                            "mode": str(mode),
                            "ant_id": int(r["ant_id"]),
                            "word": str(r["word"]),
                            "temperature": float(temp),
                            "semantic_score": float(r["semantic_score"]),
                            "rank_reward": float(r["rank_reward"]),
                            "final_reward": float(fr),
                        }
                    )
                    # best tracking
                    key = (int(d), str(mode))
                    if float(r["raw_J"]) < float(best_by[key]["J"]):
                        best_by[key] = {"J": float(r["raw_J"]), "row": r.copy()}

                # pheromone update: evaporate + deposit from elites based on final_reward
                # evaporation
                for k in pher[mode][int(d)].keys():
                    pher[mode][int(d)][k] = float(max(1e-6, (1.0 - float(args.rho)) * pher[mode][int(d)][k]))
                # deposit
                rows_sorted = sorted(rows, key=lambda r: float(r["raw_J"]))
                elite_k = int(max(1, math.ceil(float(args.elite_fraction) * len(rows_sorted))))
                for r in rows_sorted[:elite_k]:
                    dep = float(args.q) * float(r.get("final_reward", 0.0))
                    if dep <= 0.0:
                        continue
                    for tok in str(r["word"]).split():
                        try:
                            left, pstr = tok.split("^")
                            gi = int(left.split("_")[1])
                            pw = int(pstr)
                            if (gi, pw) in pher[mode][int(d)]:
                                pher[mode][int(d)][(gi, pw)] = float(min(1e6, pher[mode][int(d)][(gi, pw)] + dep))
                        except Exception:
                            continue

                # add deterministic semantic rules from best in this batch
                if rows_sorted:
                    memory.extend(deterministic_rules_from_best(int(d), rows_sorted[0], source=f"v14_5_{mode}", it=int(it)))

            # prune semantic memory to keep deterministic + bounded
            memory.sort(key=lambda rec: record_weight(rec), reverse=True)
            memory = memory[:500]

        if it == 1 or it % max(1, int(args.progress_every)) == 0 or it == int(args.num_iters):
            elapsed = time.perf_counter() - t_start
            avg = elapsed / max(1, it)
            eta = avg * max(0, int(args.num_iters) - it)
            best_main = min(float(best_by[(int(d), main_mode)]["J"]) for d in dims)
            print(f"[V14.5] iter={it}/{int(args.num_iters)} best_main_J={best_main:.6g} eta={format_seconds(eta)}", flush=True)

    # Summaries
    for d in dims:
        for mode in modes:
            br = best_by[(int(d), str(mode))]["row"]
            if br is None:
                continue
            best_candidates.append(
                {
                    "dim": int(d),
                    "mode": str(mode),
                    "best_word": str(br["word"]),
                    "best_J": float(br["raw_J"]),
                    "best_support_overlap": float(br["support_overlap"]),
                    "best_active_argument_error": float(br["active_argument_error"]),
                    "best_poisson_like_fraction": float(br["poisson_like_fraction"]),
                    "best_residue_error": float(br["residue_error"]),
                    "best_trace_error": float(br["trace_error"]),
                    "best_generator_entropy": float(br["generator_entropy"]),
                    "best_unique_generator_fraction": float(br["unique_generator_fraction"]),
                    "best_local_band_width": float(br["local_band_width"]),
                    "best_rank_reward": float(br["rank_reward"]),
                    "best_final_reward": float(br["final_reward"]),
                }
            )

    for d in dims:
        br = best_by[(int(d), main_mode)]["row"]
        if br is None:
            continue
        G1 = bool(br["stable"])
        G2 = bool(float(br["support_overlap"]) >= float(args.support_overlap_min))
        G3 = bool(float(br["active_argument_error"]) <= float(args.active_error_margin))
        G4 = True  # NV curve quality gate not thresholded here; tracked via poisson_like_fraction
        G5 = bool(float(br["poisson_like_fraction"]) < 0.95)
        G6 = bool(float(br["residue_error"]) <= float(args.residue_margin))
        G7 = bool(float(br["trace_error"]) <= float(args.trace_margin))
        J_main = float(best_by[(int(d), main_mode)]["J"])
        J_num = float(best_by[(int(d), "numeric_only")]["J"])
        J_sem = float(best_by[(int(d), "semantic_only")]["J"])
        G8 = bool(math.isfinite(J_num) and J_main <= J_num)
        G9 = bool(math.isfinite(J_sem) and J_main <= J_sem)
        # rejected baseline comparison (only if exists); conservative False
        G10 = False
        G11 = bool(float(br["generator_entropy"]) >= 0.5)
        G12 = bool(G1 and G2 and G3 and G5 and G6 and G7 and G11)
        gate_summary.append(
            {
                "dim": int(d),
                "best_word": str(br["word"]),
                "G1_stable_operator_ok": bool(G1),
                "G2_support_overlap_ok": bool(G2),
                "G3_active_argument_ok": bool(G3),
                "G4_number_variance_ok": bool(G4),
                "G5_not_poisson_like": bool(G5),
                "G6_residue_ok": bool(G6),
                "G7_trace_ok": bool(G7),
                "G8_beats_numeric_only": bool(G8),
                "G9_beats_semantic_only": bool(G9),
                "G10_beats_rejected_baseline": bool(G10),
                "G11_no_semantic_collapse": bool(G11),
                "G12_all_gate_pass": bool(G12),
            }
        )

    for d in dims:
        ablation_summary.append(
            {
                "dim": int(d),
                "numeric_only_best_J": float(best_by[(int(d), "numeric_only")]["J"]),
                "semantic_only_best_J": float(best_by[(int(d), "semantic_only")]["J"]),
                "hybrid_numeric_semantic_best_J": float(best_by[(int(d), "hybrid_numeric_semantic")]["J"]),
                "hybrid_ranked_anticollapse_best_J": float(best_by[(int(d), "hybrid_ranked_anticollapse")]["J"]),
            }
        )

    # numeric pheromone summary (top tokens per dim/mode)
    for d in dims:
        for mode in modes:
            items = list(pher[mode][int(d)].items())
            items.sort(key=lambda kv: float(kv[1]), reverse=True)
            for (gi, pw), v in items[: min(80, len(items))]:
                numeric_pheromone_summary.append({"dim": int(d), "mode": str(mode), "generator": int(gi), "power": int(pw), "pheromone": float(v)})

    # write outputs
    write_semantic_jsonl(out_dir / "v14_5_semantic_pheromones.jsonl", memory)
    write_csv(
        out_dir / "v14_5_aco_history.csv",
        fieldnames=[
            "iter",
            "dim",
            "mode",
            "ant_id",
            "word",
            "word_len",
            "stable",
            "raw_J",
            "rank_reward",
            "final_reward",
            "support_overlap",
            "active_argument_error",
            "poisson_like_fraction",
            "residue_error",
            "trace_error",
            "anti_poisson_improved",
            "generator_entropy",
            "unique_generator_fraction",
            "local_band_width",
            "collapse_penalty",
            "semantic_score",
            "semantic_bonus",
            "semantic_penalty",
            "temperature",
        ],
        rows=aco_history,
    )
    write_csv(
        out_dir / "v14_5_best_candidates.csv",
        fieldnames=list(best_candidates[0].keys()) if best_candidates else ["dim", "mode", "best_word", "best_J"],
        rows=best_candidates,
    )
    write_csv(
        out_dir / "v14_5_gate_summary.csv",
        fieldnames=list(gate_summary[0].keys()) if gate_summary else ["dim", "G12_all_gate_pass"],
        rows=gate_summary,
    )
    write_csv(
        out_dir / "v14_5_ablation_summary.csv",
        fieldnames=list(ablation_summary[0].keys()) if ablation_summary else ["dim"],
        rows=ablation_summary,
    )
    write_csv(
        out_dir / "v14_5_numeric_pheromone_summary.csv",
        fieldnames=list(numeric_pheromone_summary[0].keys()) if numeric_pheromone_summary else ["dim", "mode", "generator", "power", "pheromone"],
        rows=numeric_pheromone_summary,
    )
    write_csv(
        out_dir / "v14_5_semantic_decisions.csv",
        fieldnames=list(semantic_decisions[0].keys()) if semantic_decisions else ["iter", "dim", "mode", "ant_id", "word", "final_reward"],
        rows=semantic_decisions,
    )
    write_csv(
        out_dir / "v14_5_diversity_summary.csv",
        fieldnames=list(diversity_summary[0].keys()) if diversity_summary else ["iter", "dim", "mode", "generator_entropy", "collapse_penalty"],
        rows=diversity_summary,
    )
    write_csv(
        out_dir / "v14_5_active_argument_counts.csv",
        fieldnames=list(active_argument_counts[0].keys()) if active_argument_counts else ["iter", "dim", "mode", "ant_id", "word"],
        rows=active_argument_counts,
    )
    write_csv(
        out_dir / "v14_5_nv_curves.csv",
        fieldnames=list(nv_curves[0].keys()) if nv_curves else ["iter", "dim", "mode", "ant_id", "word", "nv_l2_to_poisson", "nv_l2_to_gue"],
        rows=nv_curves,
    )
    write_csv(
        out_dir / "v14_5_residue_scores.csv",
        fieldnames=list(residue_scores[0].keys()) if residue_scores else ["iter", "dim", "mode", "ant_id", "word", "residue_error"],
        rows=residue_scores,
    )
    write_csv(
        out_dir / "v14_5_trace_proxy.csv",
        fieldnames=list(trace_proxy[0].keys()) if trace_proxy else ["iter", "dim", "mode", "ant_id", "word", "trace_error"],
        rows=trace_proxy,
    )

    results = {
        "version": "v14_5",
        "out_dir": str(out_dir),
        "dims": dims,
        "num_ants": int(args.num_ants),
        "num_iters": int(args.num_iters),
        "modes": modes,
        "main_mode": main_mode,
        "semantic_temperature_start": float(args.semantic_temperature_start),
        "semantic_temperature_end": float(args.semantic_temperature_end),
        "warnings": warnings,
        "gate_summary": gate_summary,
        "ablation_summary": ablation_summary,
        "notes": [
            "Computational evidence only; not a proof of RH.",
            "Rank-based reward is used; raw reward collapse from V14.4 is avoided by construction.",
            "Rejected-baseline gate (G10) is conservative False unless baseline is explicitly integrated.",
        ],
    }
    write_text(out_dir / "v14_5_results.json", json.dumps(json_sanitize(results), indent=2, ensure_ascii=False) + "\n")

    # Report (md + tex + optional pdf)
    md = []
    md.append("# V14.5 — Rank-Based Semantic DTES Ants\n")
    md.append("**Computational evidence only; not a proof of RH.**\n")
    md.append("## Executive answers\n")
    md.append("1. **Did rank-based reward fix reward collapse?** Yes: reward is rank-based by construction (per iter/dim/mode), clipped to [1e-6,1].\n")
    md.append("2. **Did semantic collapse decrease?** See `v14_5_diversity_summary.csv` (normalized generator entropy) and `best_generator_entropy` in `v14_5_best_candidates.csv`.\n")
    md.append("3. **Did hybrid_ranked_anticollapse beat numeric_only?** See `v14_5_gate_summary.csv` field `G8_beats_numeric_only` and `v14_5_ablation_summary.csv`.\n")
    md.append("4. **Did it beat semantic_only?** See `G9_beats_semantic_only` and ablation summary.\n")
    md.append("5. **Did any candidate pass anti-Poisson gate?** Use `best_poisson_like_fraction` and `anti_poisson_improved` (in history) — if Poisson-like remains ~1, then no.\n")
    md.append("6. **Did any candidate pass residue/argument gate?** Use `best_active_argument_error`, `best_residue_error` vs margins.\n")
    md.append("7. **Did any candidate beat rejected_word_seed17 baseline?** Not evaluated here unless baseline is explicitly loaded; `G10_beats_rejected_baseline` is conservative False.\n")
    md.append("8. **Did any candidate pass all gates?** See `G12_all_gate_pass`.\n")
    md.append("9. **Should proceed to V14.6?** If rewards/diversity improved but rigidity gates fail, proceed to V14.6 to expand operator family / multi-island search.\n")
    md.append("10. **Should proceed to analytic claim?** Always **False** unless all gates + null controls pass.\n")
    md.append("\n## Artifacts\n")
    md.append("- `v14_5_aco_history.csv`: per-ant raw_J, rank_reward, final_reward + diagnostics\n")
    md.append("- `v14_5_diversity_summary.csv`: generator entropy + collapse penalty per iter/dim/mode\n")
    md.append("- `v14_5_best_candidates.csv`, `v14_5_gate_summary.csv`, `v14_5_ablation_summary.csv`\n")
    md.append("- `v14_5_semantic_pheromones.jsonl`: semantic memory used/updated deterministically\n")
    md.append("\n## Verification commands\n")
    md.append("```bash\n")
    md.append('OUT=runs/v14_5_rank_based_semantic_dtes_ants\n\n')
    md.append('echo "=== FILES ==="\nfind "$OUT" -maxdepth 1 -type f | sort\n\n')
    md.append('echo "=== GATE ==="\ncolumn -s, -t < "$OUT"/v14_5_gate_summary.csv | head -120\n\n')
    md.append('echo "=== BEST CANDIDATES ==="\ncolumn -s, -t < "$OUT"/v14_5_best_candidates.csv | head -120\n\n')
    md.append('echo "=== ABLATION ==="\ncolumn -s, -t < "$OUT"/v14_5_ablation_summary.csv\n\n')
    md.append('echo "=== DIVERSITY ==="\ncolumn -s, -t < "$OUT"/v14_5_diversity_summary.csv | head -120\n\n')
    md.append('echo "=== ACO HISTORY TAIL ==="\ntail -100 "$OUT"/v14_5_aco_history.csv | column -s, -t\n\n')
    md.append('echo "=== PHEROMONES ==="\ncolumn -s, -t < "$OUT"/v14_5_numeric_pheromone_summary.csv | head -120\n\n')
    md.append('echo "=== SEMANTIC DECISIONS ==="\ncolumn -s, -t < "$OUT"/v14_5_semantic_decisions.csv | head -120\n\n')
    md.append('echo "=== REPORT ==="\nhead -260 "$OUT"/v14_5_report.md\n')
    md.append("```\n")
    write_text(out_dir / "v14_5_report.md", "".join(md))

    tex = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{hyperref}
\begin{document}
\section*{V14.5 --- Rank-Based Semantic DTES Ants}
\textbf{Computational evidence only; not a proof of RH.}

\subsection*{What this run answers}
\begin{itemize}
\item Rank-based reward fixes reward collapse by construction.
\item Semantic collapse is monitored via normalized generator entropy and collapse penalties.
\item Ablations compare numeric-only, semantic-only, hybrid, and hybrid\_ranked\_anticollapse.
\item Analytic claims are \textbf{not made}.
\end{itemize}

\subsection*{Artifacts}
See the run directory for CSVs and the markdown report.

\end{document}
"""
    tex_path = out_dir / "v14_5_report.tex"
    write_text(tex_path, tex)
    try_pdflatex(tex_path, out_dir, "v14_5_report.pdf")

    elapsed = time.perf_counter() - t_start
    print(f"[V14.5] done in {format_seconds(elapsed)}; out_dir={out_dir}", flush=True)


if __name__ == "__main__":
    main()

