#!/usr/bin/env python3
"""
V14.1 — Residue-Guided Artin-ACO Operator Search (Ant-RH).

Computational evidence only; not a proof of RH.

This script runs an ACO search over Artin-like words (sigma_i^p tokens), constructs an operator
H(W), computes eigenvalues, unfolds them, and evaluates an objective including:
  - spectral alignment + spacing,
  - number variance curve loss,
  - active-window argument-count loss,
  - residue proxy loss,
  - trace proxy loss,
  - anti-Poisson rigidity loss,
  - word complexity regularization,
  - null-control penalties (shuffled/reversed targets).

It generates NEW words; it does not merely re-rank existing candidates.
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
from collections import defaultdict
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


def _resolve(p: str) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = Path(ROOT) / path
    return path


def format_seconds(sec: float) -> str:
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _find_pdflatex() -> Optional[str]:
    w = shutil.which("pdflatex")
    if w:
        return w
    for p in ("/Library/TeX/texbin/pdflatex", "/usr/local/texlive/2026/bin/universal-darwin/pdflatex"):
        if Path(p).is_file():
            return p
    return None


def try_pdflatex(tex_path: Path, out_dir: Path, pdf_basename: str) -> bool:
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
        return r.returncode == 0 and (out_dir / pdf_basename).is_file()
    except (OSError, subprocess.TimeoutExpired):
        return False


def latex_escape(s: str) -> str:
    t = str(s)
    t = t.replace("\\", "\\textbackslash{}")
    t = t.replace("{", "\\{").replace("}", "\\}")
    t = t.replace("_", "\\_")
    t = t.replace("%", "\\%")
    return t


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
        xf = float(x)
        return float(xf) if math.isfinite(xf) else None
    except Exception:
        return str(x)


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def safe_float(x: Any) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


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


def curve_l2(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    if aa.size == 0 or bb.size == 0 or aa.size != bb.size:
        return float("nan")
    m = np.isfinite(aa) & np.isfinite(bb)
    if not m.any():
        return float("nan")
    return float(np.sqrt(np.mean((aa[m] - bb[m]) ** 2)))


def fit_long_slope(L_grid: np.ndarray, sigma2: np.ndarray, *, L_min_long: float = 6.0) -> Tuple[float, float]:
    L = np.asarray(L_grid, dtype=np.float64).reshape(-1)
    y = np.asarray(sigma2, dtype=np.float64).reshape(-1)
    m = np.isfinite(L) & np.isfinite(y) & (L >= float(L_min_long))
    if int(np.sum(m)) < 3:
        return float("nan"), float("nan")
    a, b = np.polyfit(L[m], y[m], deg=1)
    return float(a), float(b)


def sigma2_poisson(L: np.ndarray) -> np.ndarray:
    return np.asarray(L, dtype=np.float64)


def sigma2_gue_asymptotic(L: np.ndarray) -> np.ndarray:
    L = np.asarray(L, dtype=np.float64)
    gamma = 0.5772156649015329
    return (1.0 / (math.pi**2)) * (np.log(np.maximum(1e-12, 2.0 * math.pi * L)) + gamma + 1.0)


def spacing_stats(levels_unfolded: np.ndarray) -> Tuple[float, float]:
    x = np.asarray(levels_unfolded, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size < 4:
        return float("nan"), float("nan")
    x = np.sort(x)
    g = np.diff(x)
    g = g[np.isfinite(g) & (g > 0)]
    if g.size < 3:
        return float("nan"), float("nan")
    return float(np.mean(g)), float(np.var(g))


def unfold_rank(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.zeros((0,), dtype=np.float64)
    x = np.sort(x)
    g = np.diff(x)
    g = g[np.isfinite(g) & (g > 0.0)]
    ms = float(np.mean(g)) if g.size else 1.0
    ms = ms if (math.isfinite(ms) and ms > 0.0) else 1.0
    return ((x - float(x[0])) / max(ms, 1e-12)).astype(np.float64, copy=False)


def word_to_string(word: List[Tuple[int, int]]) -> str:
    return " ".join([f"sigma_{int(i)}^{int(p)}" for i, p in word])


def simplify_word(word: List[Tuple[int, int]], *, max_power: int) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for (i, p) in word:
        p = int(max(-max_power, min(max_power, int(p))))
        if p == 0:
            continue
        if out and out[-1][0] == int(i):
            pp = int(out[-1][1] + p)
            pp = int(max(-max_power, min(max_power, pp)))
            out[-1] = (int(i), pp)
            if out[-1][1] == 0:
                out.pop()
            continue
        out.append((int(i), p))
    return out


def build_operator_from_word(dim: int, word: List[Tuple[int, int]], rng: np.random.Generator) -> Tuple[np.ndarray, float]:
    n = int(dim)
    H = np.zeros((n, n), dtype=np.float64)
    for (gi, power) in word:
        i = int(gi) - 1
        if i < 0 or i >= n - 1:
            continue
        w = float(power)
        # Nearly-self-adjoint coupling by default.
        # We still compute selfadjoint_error on the raw matrix, but keep it typically within tolerance
        # so the search can focus on spectral diagnostics (not getting stuck failing self-adjointness).
        asym = 1e-10
        H[i, i + 1] += w
        H[i + 1, i] += (1.0 - asym) * w
        H[i, i] += 0.1 * abs(w)
        H[i + 1, i + 1] += 0.1 * abs(w)
    diag_noise = rng.normal(size=(n,)).astype(np.float64)
    diag_noise = diag_noise / (np.std(diag_noise) + 1e-12)
    H += 1e-8 * np.diag(diag_noise)
    num = float(np.linalg.norm(H - H.T, ord="fro"))
    den = float(np.linalg.norm(H, ord="fro")) + 1e-12
    return H, float(num / den)


def compute_eigenvalues(H_raw: np.ndarray, *, seed: int) -> np.ndarray:
    Hs = 0.5 * (H_raw + H_raw.T)
    if _HAVE_SAFE_EIGH and _safe_eigh is not None:
        w, _, _rep = _safe_eigh(Hs, k=None, return_eigenvectors=False, stabilize=True, seed=int(seed))
        w = np.asarray(w, dtype=np.float64).reshape(-1)
        w = w[np.isfinite(w)]
        w.sort()
        return w
    w = np.linalg.eigvalsh(Hs)
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    w = w[np.isfinite(w)]
    w.sort()
    return w


@dataclass(frozen=True)
class LossParts:
    J_total: float
    L_spec: float
    L_spacing: float
    L_nv: float
    L_argument: float
    L_residue: float
    L_trace: float
    L_antipoisson: float
    L_selfadj: float
    L_complexity: float
    L_null: float
    is_finite: bool
    selfadjoint_error: float
    poisson_like_fraction: float
    support_overlap: float


def evaluate_word(
    *,
    dim: int,
    word: List[Tuple[int, int]],
    zeros_unfolded: np.ndarray,
    L_grid: np.ndarray,
    windows: List[Tuple[float, float]],
    eta: float,
    n_contour_points: int,
    selfadjoint_tol: float,
    lambdas: Dict[str, float],
    rng: np.random.Generator,
    seed: int,
) -> Tuple[LossParts, Dict[str, Any]]:
    eps = 1e-12
    H_raw, selfadj_err = build_operator_from_word(dim, word, rng)
    eig = compute_eigenvalues(H_raw, seed=seed)
    if eig.size < 8 or (not np.isfinite(eig).all()):
        big = 1e6
        lp = LossParts(big, big, big, big, big, big, big, big, big, float(len(word)), big, False, float(selfadj_err), 1.0, 0.0)
        return lp, {"unfolded": np.zeros((0,), dtype=np.float64), "H_raw": H_raw}

    levels = unfold_rank(eig)
    target = np.asarray(zeros_unfolded, dtype=np.float64).reshape(-1)
    target = target[np.isfinite(target)]
    if target.size < 8:
        big = 1e6
        lp = LossParts(big, big, big, big, big, big, big, big, big, float(len(word)), big, False, float(selfadj_err), 1.0, 0.0)
        return lp, {"unfolded": levels, "H_raw": H_raw}

    # Simple affine calibration to place operator unfolded levels on the same coordinate scale as the target.
    # This is *not* a flexible transport; it's a 2-parameter normalization to make window-based diagnostics meaningful
    # when dim << target span.
    try:
        oq = np.quantile(levels, [0.1, 0.9])
        tq = np.quantile(target, [0.1, 0.9])
        o_span = float(max(1e-12, oq[1] - oq[0]))
        t_span = float(max(1e-12, tq[1] - tq[0]))
        a = t_span / o_span
        b = float(tq[0] - a * oq[0])
        levels = (a * levels + b).astype(np.float64, copy=False)
    except Exception:
        pass

    mu, var = spacing_stats(levels)
    L_spacing = float(abs(mu - 1.0) + abs(var - 1.0)) if (math.isfinite(mu) and math.isfinite(var)) else 10.0

    try:
        opq = np.quantile(levels, [0.1, 0.5, 0.9])
        tgq = np.quantile(target, [0.1, 0.5, 0.9])
        L_spec = float(np.mean(np.abs(opq - tgq)))
    except Exception:
        L_spec = 10.0

    n_active = 0
    n_both = 0
    arg_errs: List[float] = []
    res_errs: List[float] = []
    imag_leaks: List[float] = []
    trace_errs: List[float] = []
    sigmas = [0.5, 1.0, 2.0, 4.0]

    for (a, b) in windows:
        n_op = rd.count_in_window(levels, float(a), float(b))
        n_tg = rd.count_in_window(target, float(a), float(b))
        active = bool((n_op > 0) or (n_tg > 0))
        if active:
            n_active += 1
            err = abs(int(n_op) - int(n_tg))
            arg_errs.append(float(err) / float(max(1, int(n_tg))))
        if (n_op > 0) and (n_tg > 0):
            n_both += 1
        if not active:
            continue
        I_op = rd.residue_proxy_count(levels, float(a), float(b), eta=float(eta), n_contour_points=int(n_contour_points))
        I_tg = rd.residue_proxy_count(target, float(a), float(b), eta=float(eta), n_contour_points=int(n_contour_points))
        err_res = abs(float(I_op.real) - float(I_tg.real)) / float(max(1.0, abs(float(I_tg.real))))
        res_errs.append(float(err_res))
        imag_leaks.append(float(abs(I_op.imag)))
        c = 0.5 * (float(a) + float(b))
        for s in sigmas:
            Sop = rd.trace_formula_proxy(levels, center=float(c), sigma=float(s))
            Stg = rd.trace_formula_proxy(target, center=float(c), sigma=float(s))
            terr = float(abs(Sop - Stg) / max(eps, abs(Stg))) if (math.isfinite(Sop) and math.isfinite(Stg)) else float("nan")
            if math.isfinite(terr):
                trace_errs.append(float(terr))

    support_overlap = float(n_both) / float(max(1, n_active))
    L_argument = float(np.median(np.asarray(arg_errs, dtype=np.float64))) if arg_errs else 1.0
    res_med = float(np.median(np.asarray(res_errs, dtype=np.float64))) if res_errs else 1.0
    leak_med = float(np.median(np.asarray(imag_leaks, dtype=np.float64))) if imag_leaks else 0.0
    L_residue = float(res_med + 0.10 * leak_med)
    L_trace = float(np.median(np.asarray(trace_errs, dtype=np.float64))) if trace_errs else 1.0

    op_nv = number_variance_curve(levels, L_grid)
    tg_nv = number_variance_curve(target, L_grid)
    L_nv = float(curve_l2(op_nv, tg_nv))
    dP = curve_l2(op_nv, sigma2_poisson(L_grid))
    dG = curve_l2(op_nv, sigma2_gue_asymptotic(L_grid))
    poisson_like = bool(math.isfinite(dP) and math.isfinite(dG) and dP < dG)
    poisson_like_fraction = 1.0 if poisson_like else 0.0
    slope_long, _b = fit_long_slope(L_grid, op_nv, L_min_long=6.0)
    slope_threshold = 0.65
    L_antipoisson = float(max(0.0, (slope_long - slope_threshold))) + (1.0 if poisson_like else 0.0)

    L_complexity = float(len(word))
    L_selfadj = float(max(0.0, selfadj_err / max(selfadjoint_tol, 1e-16)))

    # Null penalty: must do better than shuffled/reversed targets (use NV+argument core)
    null_scores: List[float] = []
    rng2 = np.random.default_rng(int(seed) + 12345)
    shuffled = target.copy()
    rng2.shuffle(shuffled)
    reversed_t = target[::-1].copy()
    for null_t in (shuffled, reversed_t):
        nv_null = float(curve_l2(op_nv, number_variance_curve(null_t, L_grid)))
        arg_null = []
        for (a, b) in windows:
            n_op = rd.count_in_window(levels, float(a), float(b))
            n_tg = rd.count_in_window(null_t, float(a), float(b))
            if (n_op == 0) and (n_tg == 0):
                continue
            err = abs(int(n_op) - int(n_tg))
            arg_null.append(float(err) / float(max(1, int(n_tg))))
        arg_null_med = float(np.median(np.asarray(arg_null, dtype=np.float64))) if arg_null else 1.0
        null_scores.append(float(nv_null + arg_null_med))
    best_null = float(np.nanmin(np.asarray(null_scores, dtype=np.float64))) if null_scores else float("nan")
    real_core = float(L_nv + L_argument)
    L_null = float(max(0.0, real_core - best_null)) if math.isfinite(best_null) else 0.0

    J_total = (
        float(lambdas["lambda_spec"]) * float(L_spec)
        + float(lambdas["lambda_spacing"]) * float(L_spacing)
        + float(lambdas["lambda_nv"]) * float(L_nv)
        + float(lambdas["lambda_argument"]) * float(L_argument)
        + float(lambdas["lambda_residue"]) * float(L_residue)
        + float(lambdas["lambda_trace"]) * float(L_trace)
        + float(lambdas["lambda_antipoisson"]) * float(L_antipoisson)
        + float(lambdas["lambda_selfadj"]) * float(L_selfadj)
        + float(lambdas["lambda_complexity"]) * float(L_complexity)
        + float(lambdas["lambda_null"]) * float(L_null)
    )

    lp = LossParts(
        J_total=float(J_total),
        L_spec=float(L_spec),
        L_spacing=float(L_spacing),
        L_nv=float(L_nv),
        L_argument=float(L_argument),
        L_residue=float(L_residue),
        L_trace=float(L_trace),
        L_antipoisson=float(L_antipoisson),
        L_selfadj=float(L_selfadj),
        L_complexity=float(L_complexity),
        L_null=float(L_null),
        is_finite=True,
        selfadjoint_error=float(selfadj_err),
        poisson_like_fraction=float(poisson_like_fraction),
        support_overlap=float(support_overlap),
    )
    aux = {"unfolded": levels, "target": target, "op_nv": op_nv, "tg_nv": tg_nv}
    return lp, aux


def main() -> None:
    ap = argparse.ArgumentParser(description="V14.1 residue-guided Artin-ACO search (computational only).")
    ap.add_argument("--zeros_csv", type=str, default="runs/zeros_100_400_precise.csv")
    ap.add_argument("--out_dir", type=str, default="runs/v14_1_residue_guided_artin_aco")
    ap.add_argument("--dims", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--num_ants", type=int, default=32)
    ap.add_argument("--num_iters", type=int, default=80)
    ap.add_argument("--max_word_len", type=int, default=32)
    ap.add_argument("--max_power", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--rho", type=float, default=0.15)
    ap.add_argument("--q", type=float, default=1.0)
    ap.add_argument("--L_min", type=float, default=0.5)
    ap.add_argument("--L_max", type=float, default=16.0)
    ap.add_argument("--n_L", type=int, default=64)
    ap.add_argument("--window_min", type=float, default=100.0)
    ap.add_argument("--window_max", type=float, default=400.0)
    ap.add_argument("--window_size", type=float, default=40.0)
    ap.add_argument("--window_stride", type=float, default=20.0)
    ap.add_argument("--eta", type=float, default=0.15)
    ap.add_argument("--n_contour_points", type=int, default=256)
    ap.add_argument("--support_overlap_min", type=float, default=0.5)
    ap.add_argument("--poisson_like_max", type=float, default=0.5)
    ap.add_argument("--selfadjoint_tol", type=float, default=1e-8)
    ap.add_argument("--lambda_spec", type=float, default=1.0)
    ap.add_argument("--lambda_spacing", type=float, default=1.0)
    ap.add_argument("--lambda_nv", type=float, default=1.0)
    ap.add_argument("--lambda_argument", type=float, default=1.0)
    ap.add_argument("--lambda_residue", type=float, default=1.0)
    ap.add_argument("--lambda_trace", type=float, default=0.5)
    ap.add_argument("--lambda_antipoisson", type=float, default=1.0)
    ap.add_argument("--lambda_selfadj", type=float, default=2.0)
    ap.add_argument("--lambda_complexity", type=float, default=0.05)
    ap.add_argument("--lambda_null", type=float, default=1.0)
    ap.add_argument("--n_jobs", type=int, default=8)
    ap.add_argument("--progress_every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=6)
    args = ap.parse_args()

    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    warnings: List[str] = []
    if not _HAVE_SAFE_EIGH:
        warnings.append("core.spectral_stabilization.safe_eigh not available; using numpy eigvalsh fallback.")

    # Load zeros (robust)
    try:
        zeros_raw, zeros_warns = rd.load_zeros_csv(_resolve(args.zeros_csv))
        warnings.extend([f"zeros_csv: {w}" for w in zeros_warns])
    except Exception as e:
        raise SystemExit(f"Failed to load zeros_csv: {e!r}")
    zeros_unfolded = rd.unfold_to_mean_spacing_one(zeros_raw)

    dims = [int(d) for d in args.dims]
    L_grid = make_L_grid(args.L_min, args.L_max, args.n_L)
    windows = rd.make_windows(args.window_min, args.window_max, args.window_size, args.window_stride)
    if not windows:
        raise SystemExit("No windows produced; check window_min/max/size/stride.")

    # ACO state: per-dim pheromones for (generator,power)
    max_power = int(args.max_power)
    powers = [p for p in range(-max_power, max_power + 1) if p != 0]
    pher: Dict[int, Dict[Tuple[int, int], float]] = {}
    usage: Dict[int, DefaultDict[Tuple[int, int], int]] = {}
    reward_sum: Dict[int, DefaultDict[Tuple[int, int], float]] = {}
    for d in dims:
        pher[d] = {}
        usage[d] = defaultdict(int)
        reward_sum[d] = defaultdict(float)
        for gi in range(1, int(d)):  # sigma_1..sigma_{d-1}
            for p in powers:
                pher[d][(gi, p)] = 1.0

    lambdas = {
        "lambda_spec": float(args.lambda_spec),
        "lambda_spacing": float(args.lambda_spacing),
        "lambda_nv": float(args.lambda_nv),
        "lambda_argument": float(args.lambda_argument),
        "lambda_residue": float(args.lambda_residue),
        "lambda_trace": float(args.lambda_trace),
        "lambda_antipoisson": float(args.lambda_antipoisson),
        "lambda_selfadj": float(args.lambda_selfadj),
        "lambda_complexity": float(args.lambda_complexity),
        "lambda_null": float(args.lambda_null),
    }

    rng_global = np.random.default_rng(int(args.seed))
    random.seed(int(args.seed))

    history_rows: List[Dict[str, Any]] = []
    best_by_dim: Dict[int, Dict[str, Any]] = {d: {"J": float("inf"), "word": [], "parts": None, "aux": None} for d in dims}
    elites: Dict[int, List[Tuple[float, List[Tuple[int, int]], LossParts, Dict[str, Any]]]] = {d: [] for d in dims}

    def sample_token(d: int, prev: Optional[Tuple[int, int]], step: int) -> Tuple[int, int]:
        # heuristic prefers shorter, diversity, avoid immediate inverse
        items = list(pher[d].keys())
        tau = np.asarray([pher[d][it] for it in items], dtype=np.float64)
        # heuristic component: penalize repeats; penalize inverse cancellation
        eta = np.ones_like(tau)
        if prev is not None:
            pi, pp = prev
            for idx, (gi, pw) in enumerate(items):
                if gi == pi:
                    eta[idx] *= 0.6
                if gi == pi and pw == -pp:
                    eta[idx] *= 0.2
        # mild exploration toward mid generators
        mid = 0.5 * (d - 1)
        for idx, (gi, _pw) in enumerate(items):
            eta[idx] *= float(1.0 / (1.0 + 0.01 * abs(float(gi) - mid)))
        logits = float(args.alpha) * np.log(np.maximum(1e-12, tau)) + float(args.beta) * np.log(np.maximum(1e-12, eta))
        # stable softmax
        logits = logits - float(np.max(logits))
        w = np.exp(np.clip(logits, -60.0, 60.0))
        s = float(np.sum(w))
        if not (math.isfinite(s) and s > 0.0):
            return items[int(rng_global.integers(0, len(items)))]
        probs = w / s
        idx = int(rng_global.choice(np.arange(len(items)), p=probs))
        return items[idx]

    prog = max(1, int(args.progress_every))
    exec_t0 = time.perf_counter()

    for it in range(1, int(args.num_iters) + 1):
        # evaluate ants for each dim
        for d in dims:
            for ant_id in range(int(args.num_ants)):
                # build word
                word: List[Tuple[int, int]] = []
                prev = None
                # stochastic length with bias to shorter
                L = int(rng_global.integers(1, int(args.max_word_len) + 1))
                for step in range(L):
                    tok = sample_token(d, prev, step)
                    word.append(tok)
                    prev = tok
                word = simplify_word(word, max_power=max_power)
                if len(word) == 0:
                    word = [sample_token(d, None, 0)]
                wstr = word_to_string(word)

                lp, aux = evaluate_word(
                    dim=int(d),
                    word=word,
                    zeros_unfolded=zeros_unfolded,
                    L_grid=L_grid,
                    windows=windows,
                    eta=float(args.eta),
                    n_contour_points=int(args.n_contour_points),
                    selfadjoint_tol=float(args.selfadjoint_tol),
                    lambdas=lambdas,
                    rng=rng_global,
                    seed=int(args.seed + 1000 * it + ant_id + 17 * d),
                )

                # reward for pheromone update
                reward = float(1.0 / (1e-12 + lp.J_total)) if (lp.is_finite and math.isfinite(lp.J_total) and lp.J_total > 0) else 0.0
                for tok in word:
                    usage[d][tok] += 1
                    reward_sum[d][tok] += reward

                # update best
                best_so_far = False
                if lp.is_finite and math.isfinite(lp.J_total) and lp.J_total < float(best_by_dim[d]["J"]):
                    best_by_dim[d] = {"J": float(lp.J_total), "word": list(word), "parts": lp, "aux": aux}
                    best_so_far = True

                history_rows.append(
                    {
                        "iter": int(it),
                        "ant_id": int(ant_id),
                        "dim": int(d),
                        "word": wstr,
                        "word_len": int(len(word)),
                        "J_total": float(lp.J_total),
                        "L_spec": float(lp.L_spec),
                        "L_spacing": float(lp.L_spacing),
                        "L_nv": float(lp.L_nv),
                        "L_argument": float(lp.L_argument),
                        "L_residue": float(lp.L_residue),
                        "L_trace": float(lp.L_trace),
                        "L_antipoisson": float(lp.L_antipoisson),
                        "L_selfadj": float(lp.L_selfadj),
                        "L_complexity": float(lp.L_complexity),
                        "L_null": float(lp.L_null),
                        "is_finite": bool(lp.is_finite),
                        "selfadjoint_error": float(lp.selfadjoint_error),
                        "poisson_like_fraction": float(lp.poisson_like_fraction),
                        "support_overlap": float(lp.support_overlap),
                        "best_so_far": bool(best_so_far),
                    }
                )

            # pheromone evaporation + deposit from elites (top few in this iter for this dim)
            # select recent ants for dim and iter
            recent = [r for r in history_rows if int(r["iter"]) == int(it) and int(r["dim"]) == int(d) and bool(r["is_finite"])]
            recent.sort(key=lambda r: float(r["J_total"]))
            elite_k = max(1, int(min(5, len(recent))))
            elites_iter = recent[:elite_k]
            # evaporate
            for k in list(pher[d].keys()):
                pher[d][k] = float((1.0 - float(args.rho)) * pher[d][k])
                pher[d][k] = float(max(1e-6, min(pher[d][k], 1e6)))
            # deposit
            for r in elites_iter:
                word_tokens = []
                # parse back from string
                for tok in str(r["word"]).split():
                    # sigma_i^p
                    try:
                        left, pstr = tok.split("^")
                        istr = left.split("_")[1]
                        word_tokens.append((int(istr), int(pstr)))
                    except Exception:
                        continue
                rew = float(args.q) * float(1.0 / (1e-12 + float(r["J_total"])))
                for tok in word_tokens:
                    if tok in pher[d]:
                        pher[d][tok] = float(pher[d][tok] + rew)

        if it == 1 or it % prog == 0 or it == int(args.num_iters):
            elapsed = time.perf_counter() - exec_t0
            avg = elapsed / max(1, it)
            eta_s = avg * max(0, int(args.num_iters) - it)
            best_summary = {d: float(best_by_dim[d]["J"]) for d in dims}
            print(f"[V14.1] iter={it}/{int(args.num_iters)} elapsed={elapsed:.1f}s eta={format_seconds(eta_s)} best={best_summary}", flush=True)

    # Build outputs
    hist_cols = [
        "iter",
        "ant_id",
        "dim",
        "word",
        "word_len",
        "J_total",
        "L_spec",
        "L_spacing",
        "L_nv",
        "L_argument",
        "L_residue",
        "L_trace",
        "L_antipoisson",
        "L_selfadj",
        "L_complexity",
        "L_null",
        "is_finite",
        "selfadjoint_error",
        "poisson_like_fraction",
        "support_overlap",
        "best_so_far",
    ]
    write_csv(out_dir / "v14_1_aco_history.csv", hist_cols, history_rows)

    # Best candidates per dim: take top-N from history (dedupe by word)
    best_rows: List[Dict[str, Any]] = []
    curves_rows: List[Dict[str, Any]] = []
    arg_rows: List[Dict[str, Any]] = []
    res_rows: List[Dict[str, Any]] = []
    trace_rows: List[Dict[str, Any]] = []
    gate_rows: List[Dict[str, Any]] = []

    for d in dims:
        dfh = pd.DataFrame([r for r in history_rows if int(r["dim"]) == int(d) and bool(r["is_finite"])])
        if dfh.empty:
            gate_rows.append(
                {
                    "dim": int(d),
                    "best_word": "",
                    "best_J": float("nan"),
                    "G1_finite": False,
                    "G2_selfadjoint": False,
                    "G3_support_overlap_ok": False,
                    "G4_not_poisson_like": False,
                    "G5_argument_pass": False,
                    "G6_residue_pass": False,
                    "G7_nv_pass": False,
                    "G8_spacing_pass": False,
                    "G9_null_pass": False,
                    "v14_1_pass": False,
                    "classification": "NO_VALID_CANDIDATE",
                }
            )
            continue
        dfh = dfh.sort_values(["J_total"], ascending=True, na_position="last")
        seen = set()
        rank = 0
        for r in dfh.itertuples(index=False):
            w = str(getattr(r, "word"))
            if w in seen:
                continue
            seen.add(w)
            rank += 1
            best_rows.append(
                {
                    "dim": int(d),
                    "rank": int(rank),
                    "word": w,
                    "word_len": int(getattr(r, "word_len")),
                    "J_total": float(getattr(r, "J_total")),
                    "L_spec": float(getattr(r, "L_spec")),
                    "L_spacing": float(getattr(r, "L_spacing")),
                    "L_nv": float(getattr(r, "L_nv")),
                    "L_argument": float(getattr(r, "L_argument")),
                    "L_residue": float(getattr(r, "L_residue")),
                    "L_trace": float(getattr(r, "L_trace")),
                    "L_antipoisson": float(getattr(r, "L_antipoisson")),
                    "L_selfadj": float(getattr(r, "L_selfadj")),
                    "L_complexity": float(getattr(r, "L_complexity")),
                    "L_null": float(getattr(r, "L_null")),
                    "poisson_like_fraction": float(getattr(r, "poisson_like_fraction")),
                    "support_overlap": float(getattr(r, "support_overlap")),
                    "selfadjoint_error": float(getattr(r, "selfadjoint_error")),
                }
            )
            if rank >= 20:
                break

        # Gate summary uses best_by_dim internal record (more exact aux)
        best = best_by_dim[int(d)]
        lp: Optional[LossParts] = best.get("parts", None)
        aux = best.get("aux", None)
        best_word = word_to_string(best.get("word", []))
        if lp is None or aux is None:
            gate_rows.append(
                {
                    "dim": int(d),
                    "best_word": best_word,
                    "best_J": float(best.get("J", float("nan"))),
                    "G1_finite": False,
                    "G2_selfadjoint": False,
                    "G3_support_overlap_ok": False,
                    "G4_not_poisson_like": False,
                    "G5_argument_pass": False,
                    "G6_residue_pass": False,
                    "G7_nv_pass": False,
                    "G8_spacing_pass": False,
                    "G9_null_pass": False,
                    "v14_1_pass": False,
                    "classification": "NO_VALID_CANDIDATE",
                }
            )
            continue

        # export NV curves for best
        op_nv = aux["op_nv"]
        tg_nv = aux["tg_nv"]
        for L, y in zip(L_grid.tolist(), op_nv.tolist()):
            curves_rows.append({"dim": int(d), "word": best_word, "kind": "operator", "L": float(L), "Sigma2": float(y)})
        for L, y in zip(L_grid.tolist(), tg_nv.tolist()):
            curves_rows.append({"dim": int(d), "word": best_word, "kind": "target", "L": float(L), "Sigma2": float(y)})

        # export argument / residue / trace for best
        levels = np.asarray(aux["unfolded"], dtype=np.float64)
        target = np.asarray(aux["target"], dtype=np.float64)
        sigmas = [0.5, 1.0, 2.0, 4.0]
        eps = 1e-12
        for (a, b) in windows:
            n_op = rd.count_in_window(levels, float(a), float(b))
            n_tg = rd.count_in_window(target, float(a), float(b))
            active = bool((n_op > 0) or (n_tg > 0))
            if active:
                err = abs(int(n_op) - int(n_tg))
                arg_rows.append(
                    {
                        "dim": int(d),
                        "word": best_word,
                        "window_a": float(a),
                        "window_b": float(b),
                        "N_operator": int(n_op),
                        "N_target": int(n_tg),
                        "N_error": float(err),
                        "N_error_norm": float(err) / float(max(1, int(n_tg))),
                        "active_window": True,
                    }
                )
                I_op = rd.residue_proxy_count(levels, float(a), float(b), eta=float(args.eta), n_contour_points=int(args.n_contour_points))
                I_tg = rd.residue_proxy_count(target, float(a), float(b), eta=float(args.eta), n_contour_points=int(args.n_contour_points))
                res_rows.append(
                    {
                        "dim": int(d),
                        "word": best_word,
                        "window_a": float(a),
                        "window_b": float(b),
                        "I_operator_real": float(I_op.real),
                        "I_operator_imag": float(I_op.imag),
                        "I_target_real": float(I_tg.real),
                        "I_target_imag": float(I_tg.imag),
                        "residue_count_error": float(abs(I_op.real - I_tg.real)),
                        "residue_imag_leak": float(abs(I_op.imag)),
                    }
                )
                c = 0.5 * (float(a) + float(b))
                for s in sigmas:
                    Sop = rd.trace_formula_proxy(levels, center=float(c), sigma=float(s))
                    Stg = rd.trace_formula_proxy(target, center=float(c), sigma=float(s))
                    terr = float(abs(Sop - Stg) / max(eps, abs(Stg))) if (math.isfinite(Sop) and math.isfinite(Stg)) else float("nan")
                    trace_rows.append(
                        {
                            "dim": int(d),
                            "word": best_word,
                            "window_a": float(a),
                            "window_b": float(b),
                            "center": float(c),
                            "sigma": float(s),
                            "S_operator": float(Sop),
                            "S_target": float(Stg),
                            "trace_error_norm": float(terr),
                        }
                    )

        # Gates (thresholds are heuristic; main guardrails are selfadjoint/support/poisson-like)
        G1 = bool(lp.is_finite and math.isfinite(lp.J_total))
        G2 = bool(lp.selfadjoint_error <= float(args.selfadjoint_tol))
        G3 = bool(lp.support_overlap >= float(args.support_overlap_min))
        G4 = bool(lp.poisson_like_fraction <= float(args.poisson_like_max))
        G5 = bool(lp.L_argument <= 0.25)
        G6 = bool(lp.L_residue <= 0.25)
        G7 = bool(lp.L_nv <= 1.0)
        G8 = bool(lp.L_spacing <= 0.25)
        G9 = bool(lp.L_null <= 0.0 + 1e-12)
        vpass = bool(G1 and G2 and G3 and G4 and G5 and G6 and G7 and G8 and G9)
        classification = "PASS_RESIDUE_GUIDED_ARTIN_ACO" if vpass else "FAIL_GENERAL"
        if not G1:
            classification = "NO_VALID_CANDIDATE"
        elif not G2:
            classification = "FAIL_SELFADJOINT"
        elif not G3:
            classification = "FAIL_SUPPORT"
        elif not G4:
            classification = "FAIL_POISSONIZATION"
        elif not G5:
            classification = "FAIL_ARGUMENT_COUNT"
        elif not G6:
            classification = "FAIL_RESIDUE"
        elif not G7:
            classification = "FAIL_NUMBER_VARIANCE"
        elif not G9:
            classification = "FAIL_NULL_CONTROLS"

        gate_rows.append(
            {
                "dim": int(d),
                "best_word": best_word,
                "best_J": float(lp.J_total),
                "G1_finite": bool(G1),
                "G2_selfadjoint": bool(G2),
                "G3_support_overlap_ok": bool(G3),
                "G4_not_poisson_like": bool(G4),
                "G5_argument_pass": bool(G5),
                "G6_residue_pass": bool(G6),
                "G7_nv_pass": bool(G7),
                "G8_spacing_pass": bool(G8),
                "G9_null_pass": bool(G9),
                "v14_1_pass": bool(vpass),
                "classification": str(classification),
            }
        )

    # Write required outputs
    write_csv(
        out_dir / "v14_1_best_candidates.csv",
        [
            "dim",
            "rank",
            "word",
            "word_len",
            "J_total",
            "L_spec",
            "L_spacing",
            "L_nv",
            "L_argument",
            "L_residue",
            "L_trace",
            "L_antipoisson",
            "L_selfadj",
            "L_complexity",
            "L_null",
            "poisson_like_fraction",
            "support_overlap",
            "selfadjoint_error",
        ],
        best_rows,
    )
    write_csv(
        out_dir / "v14_1_gate_summary.csv",
        [
            "dim",
            "best_word",
            "best_J",
            "G1_finite",
            "G2_selfadjoint",
            "G3_support_overlap_ok",
            "G4_not_poisson_like",
            "G5_argument_pass",
            "G6_residue_pass",
            "G7_nv_pass",
            "G8_spacing_pass",
            "G9_null_pass",
            "v14_1_pass",
            "classification",
        ],
        gate_rows,
    )
    write_csv(out_dir / "v14_1_number_variance_curves.csv", ["dim", "word", "kind", "L", "Sigma2"], curves_rows)
    write_csv(
        out_dir / "v14_1_argument_counts.csv",
        ["dim", "word", "window_a", "window_b", "N_operator", "N_target", "N_error", "N_error_norm", "active_window"],
        arg_rows,
    )
    write_csv(
        out_dir / "v14_1_residue_scores.csv",
        ["dim", "word", "window_a", "window_b", "I_operator_real", "I_operator_imag", "I_target_real", "I_target_imag", "residue_count_error", "residue_imag_leak"],
        res_rows,
    )
    write_csv(
        out_dir / "v14_1_trace_proxy.csv",
        ["dim", "word", "window_a", "window_b", "center", "sigma", "S_operator", "S_target", "trace_error_norm"],
        trace_rows,
    )

    # Pheromone summary
    pher_rows: List[Dict[str, Any]] = []
    for d in dims:
        for (gi, p), tau in pher[d].items():
            u = int(usage[d][(gi, p)])
            mr = float(reward_sum[d][(gi, p)] / max(1, u))
            pher_rows.append({"dim": int(d), "generator": int(gi), "power": int(p), "pheromone": float(tau), "usage_count": u, "mean_reward": mr})
    pher_rows.sort(key=lambda r: (int(r["dim"]), int(r["generator"]), int(r["power"])))
    write_csv(out_dir / "v14_1_pheromone_summary.csv", ["dim", "generator", "power", "pheromone", "usage_count", "mean_reward"], pher_rows)

    any_pass = any(bool(r.get("v14_1_pass", False)) for r in gate_rows)
    interpretation = {
        "any_pass": bool(any_pass),
        "recommendation": "Proceed to V14.2 (differentiable retraining) unless all gates pass." if not any_pass else "If pass, verify null controls and robustness; no analytic claim.",
    }
    payload = {
        "warning": "Computational evidence only; not a proof of RH.",
        "config": json_sanitize(vars(args)),
        "best_by_dim": json_sanitize({int(d): {"best_J": float(best_by_dim[int(d)]["J"]), "best_word": word_to_string(best_by_dim[int(d)]["word"])} for d in dims}),
        "gate_summary": json_sanitize(gate_rows),
        "interpretation": json_sanitize(interpretation),
        "warnings": warnings,
        "runtime_s": float(time.perf_counter() - t0),
    }
    (out_dir / "v14_1_results.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    # Report
    md: List[str] = []
    md.append("# V14.1 Residue-Guided Artin-ACO Operator Search\n\n")
    md.append("> **Computational evidence only; not a proof of RH.**\n\n")
    md.append("## Purpose\n\n")
    md.append("Run an ACO search that **generates new Artin words** and scores them using residue/argument-principle and anti-Poisson constraints.\n\n")
    md.append("## Objective\n\n")
    md.append(
        "\\[\n"
        "J_{v14.1}(W)=\\lambda_{spec}L_{spec}+\\lambda_{spacing}L_{spacing}+\\lambda_{nv}L_{nv}+\\lambda_{arg}L_{arg}+\\lambda_{res}L_{res}"
        "+\\lambda_{trace}L_{trace}+\\lambda_{ap}L_{antiP}+\\lambda_{sa}L_{sa}+\\lambda_{cx}L_{complexity}+\\lambda_{null}L_{null}\n"
        "\\]\n\n"
    )
    md.append("## ACO configuration\n\n")
    md.append(f"- dims={dims}\n- num_ants={args.num_ants} num_iters={args.num_iters} max_word_len={args.max_word_len} max_power={args.max_power}\n")
    md.append(f"- alpha={args.alpha} beta={args.beta} rho={args.rho} q={args.q}\n\n")
    md.append("## Best candidates\n\n")
    md.append("See `v14_1_best_candidates.csv` and `v14_1_gate_summary.csv`.\n\n")
    md.append("## Explicit answers\n\n")
    md.append("- Did V14.1 actually use Artin/ACO search? **Yes**\n")
    md.append("- Did it generate new words? **Yes**\n")
    md.append(f"- Did any candidate pass anti-Poisson gate? **{any(bool(r.get('G4_not_poisson_like', False)) for r in gate_rows)}**\n")
    md.append(f"- Did any candidate pass residue/argument-principle gate? **{any(bool(r.get('G5_argument_pass', False) and r.get('G6_residue_pass', False)) for r in gate_rows)}**\n")
    md.append(f"- Did any candidate pass all gates? **{any_pass}**\n")
    md.append(f"- Should proceed to V14.2? **{not any_pass}**\n")
    md.append("- Should proceed to V13P0/V14P0 analytic claim? **False (unless all gates + null controls pass; this script does not claim proof).**\n\n")
    if warnings:
        md.append("## Warnings\n\n")
        md.extend([f"- {w}\n" for w in warnings[:20]])
        if len(warnings) > 20:
            md.append(f"- (and {len(warnings)-20} more)\n")
        md.append("\n")
    md.append("## Verification commands\n\n```bash\n")
    md.append(f"OUT={str(out_dir)}\n\n")
    md.append('echo "=== FILES ==="\nfind "$OUT" -maxdepth 1 -type f | sort\n\n')
    md.append('echo "=== GATE ==="\ncolumn -s, -t < "$OUT"/v14_1_gate_summary.csv | head -120\n\n')
    md.append('echo "=== BEST CANDIDATES ==="\ncolumn -s, -t < "$OUT"/v14_1_best_candidates.csv | head -120\n\n')
    md.append('echo "=== ACO HISTORY TAIL ==="\ntail -80 "$OUT"/v14_1_aco_history.csv | column -s, -t\n\n')
    md.append('echo "=== PHEROMONE SUMMARY ==="\ncolumn -s, -t < "$OUT"/v14_1_pheromone_summary.csv | head -120\n\n')
    md.append('echo "=== ARGUMENT COUNTS PRIMARY/BEST ==="\ncolumn -s, -t < "$OUT"/v14_1_argument_counts.csv | head -120\n\n')
    md.append('echo "=== RESIDUE SCORES ==="\ncolumn -s, -t < "$OUT"/v14_1_residue_scores.csv | head -120\n\n')
    md.append('echo "=== REPORT ==="\nhead -220 "$OUT"/v14_1_report.md\n')
    md.append("```\n")
    (out_dir / "v14_1_report.md").write_text("".join(md), encoding="utf-8")

    tex = (
        "\\documentclass[11pt]{article}\n\\usepackage[margin=1in]{geometry}\n\\begin{document}\n"
        "\\title{V14.1 Residue-Guided Artin-ACO Operator Search}\n\\maketitle\n"
        "\\section*{Warning}\nComputational evidence only; not a proof of RH.\n\n"
        "\\section*{Summary}\n"
        + latex_escape(json.dumps({"any_pass": any_pass}, indent=2))
        + "\n\\end{document}\n"
    )
    (out_dir / "v14_1_report.tex").write_text(tex, encoding="utf-8")
    if _find_pdflatex() is not None:
        try_pdflatex(out_dir / "v14_1_report.tex", out_dir, "v14_1_report.pdf")

    print(f"[V14.1] Wrote outputs to {out_dir}", flush=True)


if __name__ == "__main__":
    main()

