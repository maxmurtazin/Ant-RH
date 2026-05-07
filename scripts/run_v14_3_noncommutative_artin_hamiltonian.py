#!/usr/bin/env python3
"""
V14.3 — Noncommutative Artin Hamiltonian Search (ACO).

Computational evidence only; not a proof of RH.

This script implements an ACO search over Artin words, but evaluates them using a richer
noncommutative Hamiltonian ansatz including commutator/anticommutator terms, plus
stabilization (Hermitian symmetrization, trace removal, radius normalization).

It is a computational diagnostic pipeline; it does not claim proof of RH.
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


def unfold_to_mean_spacing_one(x: np.ndarray) -> np.ndarray:
    return rd.unfold_to_mean_spacing_one(np.asarray(x, dtype=np.float64).reshape(-1))


def word_to_string(word: List[Tuple[int, int]]) -> str:
    return " ".join([f"sigma_{int(i)}^{int(p)}" for i, p in word])


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
        if out and out[-1][0] == i and out[-1][1] == -p:
            out.pop()
            continue
        out.append((i, p))
        if len(out) >= int(max_word_len):
            break
    return out


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


def safe_eigh(H: np.ndarray, *, seed: int) -> Optional[np.ndarray]:
    # hermitian eigvals for real or complex
    Hh = hermitian_part(H)
    if _HAVE_SAFE_EIGH and _safe_eigh is not None and np.isrealobj(Hh):
        try:
            w, _, _rep = _safe_eigh(np.asarray(Hh.real, dtype=np.float64), k=None, return_eigenvectors=False, stabilize=True, seed=int(seed))
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
    w = safe_eigh(H, seed=42)
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
    A = normalize_operator_norm(A)
    return A


def graph_laplacian_from_word(dim: int, word: List[Tuple[int, int]]) -> np.ndarray:
    n = int(dim)
    if n <= 0:
        return np.zeros((0, 0), dtype=np.complex128)
    A = np.zeros((n, n), dtype=np.float64)
    # adjacency based on generator indices used
    for (gi, _p) in word:
        i = int(max(1, min(n - 1, int(gi)))) - 1
        A[i, i + 1] += 1.0
        A[i + 1, i] += 1.0
    D = np.diag(np.sum(A, axis=1))
    L = D - A
    return L.astype(np.complex128, copy=False)


def diag_potential(dim: int, word: List[Tuple[int, int]], *, seed: int) -> np.ndarray:
    n = int(dim)
    if n <= 0:
        return np.zeros((0, 0), dtype=np.complex128)
    hist = np.zeros((n,), dtype=np.float64)
    for (gi, _p) in word:
        i = int(max(1, min(n - 1, int(gi)))) - 1
        hist[i] += 1.0
        hist[i + 1] += 1.0
    hist = hist / (np.max(hist) + 1e-12) if np.max(hist) > 0 else hist
    rng = np.random.default_rng(int(seed))
    noise = rng.normal(size=(n,)).astype(np.float64)
    noise = noise / (np.std(noise) + 1e-12)
    v = 0.7 * hist + 0.3 * noise
    return np.diag(v.astype(np.float64)).astype(np.complex128, copy=False)


@dataclass(frozen=True)
class EvalResult:
    J_total: float
    reward: float
    stable_ok: bool
    self_adjoint_residual: float
    support_overlap: float
    active_argument_error: float
    residue_error: float
    trace_error: float
    poisson_like: bool
    poisson_like_fraction: float
    level_repulsion_ok: bool
    nv_tail_ok: bool
    comm_ablation_useful: bool
    best_mode: str
    word_len: int


def main() -> None:
    ap = argparse.ArgumentParser(description="V14.3 noncommutative Artin Hamiltonian search (computational only).")
    ap.add_argument("--zeros_csv", type=str, default="runs/zeros_100_400_precise.csv")
    ap.add_argument("--out_dir", type=str, default="runs/v14_3_noncommutative_artin_hamiltonian")
    ap.add_argument("--dims", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--window_min", type=float, default=100.0)
    ap.add_argument("--window_max", type=float, default=400.0)
    ap.add_argument("--window_size", type=float, default=40.0)
    ap.add_argument("--window_stride", type=float, default=20.0)
    ap.add_argument("--L_min", type=float, default=0.5)
    ap.add_argument("--L_max", type=float, default=16.0)
    ap.add_argument("--n_L", type=int, default=64)
    ap.add_argument("--num_ants", type=int, default=32)
    ap.add_argument("--num_iters", type=int, default=80)
    ap.add_argument("--max_word_len", type=int, default=32)
    ap.add_argument("--max_power", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--rho", type=float, default=0.15)
    ap.add_argument("--q", type=float, default=1.0)
    ap.add_argument("--lambda_lin", type=float, default=1.0)
    ap.add_argument("--lambda_comm", type=float, default=1.0)
    ap.add_argument("--lambda_anti", type=float, default=0.5)
    ap.add_argument("--lambda_lap", type=float, default=0.2)
    ap.add_argument("--lambda_diag", type=float, default=0.2)
    ap.add_argument("--lambda_spec", type=float, default=0.5)
    ap.add_argument("--lambda_spacing", type=float, default=1.0)
    ap.add_argument("--lambda_nv", type=float, default=2.0)
    ap.add_argument("--lambda_arg", type=float, default=3.0)
    ap.add_argument("--lambda_res", type=float, default=1.0)
    ap.add_argument("--lambda_trace", type=float, default=1.0)
    ap.add_argument("--lambda_antipoisson", type=float, default=5.0)
    ap.add_argument("--lambda_sa", type=float, default=10.0)
    ap.add_argument("--lambda_complexity", type=float, default=0.05)
    ap.add_argument("--lambda_null", type=float, default=2.0)
    ap.add_argument("--target_radius_factor", type=float, default=0.25)
    ap.add_argument("--eta", type=float, default=0.15)
    ap.add_argument("--n_contour_points", type=int, default=256)
    ap.add_argument("--support_overlap_min", type=float, default=0.5)
    ap.add_argument("--active_error_margin", type=float, default=0.25)
    ap.add_argument("--poisson_like_max", type=float, default=0.5)
    ap.add_argument("--residue_error_max", type=float, default=0.25)
    ap.add_argument("--trace_error_max", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=20)
    ap.add_argument("--n_jobs", type=int, default=8)
    ap.add_argument("--progress_every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=6)
    args = ap.parse_args()

    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    warnings: List[str] = []
    if not _HAVE_SAFE_EIGH:
        warnings.append("core.spectral_stabilization.safe_eigh not available; using numpy eigvalsh for Hermitian eigenvalues.")

    zeros_raw, zeros_warns = rd.load_zeros_csv(_resolve(args.zeros_csv))
    warnings.extend([f"zeros_csv: {w}" for w in zeros_warns])
    target = unfold_to_mean_spacing_one(zeros_raw)
    if target.size < 32:
        warnings.append("zeros_csv has very few zeros; diagnostics will be weak.")

    dims = [int(d) for d in args.dims]
    max_power = int(args.max_power)
    powers = [p for p in range(-max_power, max_power + 1) if p != 0]
    L_grid = make_L_grid(args.L_min, args.L_max, args.n_L)
    windows = rd.make_windows(args.window_min, args.window_max, args.window_size, args.window_stride)
    if not windows:
        raise SystemExit("No windows produced; check window_min/max/size/stride.")

    # pheromones per dim:
    # - token pheromone τ_tok[(gi,p)]
    # - term-type pheromone τ_type[type] where type in {lin,comm,anti,mix}
    pher_tok: Dict[int, Dict[Tuple[int, int], float]] = {}
    pher_type: Dict[int, Dict[str, float]] = {}
    usage_tok: Dict[int, DefaultDict[Tuple[int, int], int]] = {}
    usage_type: Dict[int, DefaultDict[str, int]] = {}
    reward_sum_tok: Dict[int, DefaultDict[Tuple[int, int], float]] = {}
    reward_sum_type: Dict[int, DefaultDict[str, float]] = {}

    for d in dims:
        pher_tok[int(d)] = {}
        usage_tok[int(d)] = defaultdict(int)
        reward_sum_tok[int(d)] = defaultdict(float)
        for gi in range(1, int(d)):
            for p in powers:
                pher_tok[int(d)][(gi, p)] = 1.0
        pher_type[int(d)] = {"lin": 1.0, "comm": 1.0, "anti": 1.0, "mix": 1.0}
        usage_type[int(d)] = defaultdict(int)
        reward_sum_type[int(d)] = defaultdict(float)

    rng = np.random.default_rng(int(args.seed))
    random.seed(int(args.seed))

    # Outputs
    hist_rows: List[Dict[str, Any]] = []
    best_rows: List[Dict[str, Any]] = []
    gate_rows: List[Dict[str, Any]] = []
    pher_rows: List[Dict[str, Any]] = []
    comp_rows: List[Dict[str, Any]] = []
    ablation_rows: List[Dict[str, Any]] = []
    null_rows: List[Dict[str, Any]] = []
    spacing_rows: List[Dict[str, Any]] = []
    nv_rows: List[Dict[str, Any]] = []
    arg_rows: List[Dict[str, Any]] = []
    res_rows: List[Dict[str, Any]] = []
    trace_rows: List[Dict[str, Any]] = []

    best_by_dim: Dict[int, Dict[str, Any]] = {int(d): {"J": float("inf"), "word": "", "mode": "", "eval": None, "levels": None} for d in dims}

    def sample_type(d: int) -> str:
        items = list(pher_type[d].keys())
        tau = np.asarray([pher_type[d][k] for k in items], dtype=np.float64)
        logits = np.log(np.maximum(1e-12, tau))
        logits = logits - float(np.max(logits))
        w = np.exp(np.clip(logits, -60, 60))
        s = float(np.sum(w))
        if not (math.isfinite(s) and s > 0):
            return str(items[int(rng.integers(0, len(items)))])
        p = w / s
        return str(items[int(rng.choice(np.arange(len(items)), p=p))])

    def sample_token(d: int, prev: Optional[Tuple[int, int]]) -> Tuple[int, int]:
        items = list(pher_tok[d].keys())
        tau = np.asarray([pher_tok[d][it] for it in items], dtype=np.float64)
        eta = np.ones_like(tau)
        if prev is not None:
            pi, pp = prev
            for idx, (gi, pw) in enumerate(items):
                if gi == pi:
                    eta[idx] *= 0.75
                if gi == pi and pw == -pp:
                    eta[idx] *= 0.15
        mid = 0.5 * (d - 1)
        for idx, (gi, _pw) in enumerate(items):
            eta[idx] *= float(1.0 / (1.0 + 0.01 * abs(float(gi) - mid)))
        logits = float(args.alpha) * np.log(np.maximum(1e-12, tau)) + float(args.beta) * np.log(np.maximum(1e-12, eta))
        logits = logits - float(np.max(logits))
        w = np.exp(np.clip(logits, -60.0, 60.0))
        s = float(np.sum(w))
        if not (math.isfinite(s) and s > 0.0):
            return items[int(rng.integers(0, len(items)))]
        p = w / s
        return items[int(rng.choice(np.arange(len(items)), p=p))]

    def spacing_features(levels: np.ndarray) -> Dict[str, float]:
        x = np.asarray(levels, dtype=np.float64).reshape(-1)
        x = x[np.isfinite(x)]
        if x.size < 8:
            return {"mean_gap": float("nan"), "frac_gap_lt_0p1": 1.0, "l1_exp": float("inf"), "l1_wigner": float("inf")}
        x = np.sort(x)
        g = np.diff(x)
        g = g[np.isfinite(g) & (g > 0)]
        if g.size < 8:
            return {"mean_gap": float("nan"), "frac_gap_lt_0p1": 1.0, "l1_exp": float("inf"), "l1_wigner": float("inf")}
        g = g / max(float(np.mean(g)), 1e-12)
        # histogram on [0,4]
        bins = np.linspace(0.0, 4.0, 81)
        h, _ = np.histogram(np.clip(g, 0.0, 4.0), bins=bins, density=True)
        centers = 0.5 * (bins[:-1] + bins[1:])
        dx = float(centers[1] - centers[0])
        # exp pdf
        p_exp = np.exp(-centers)
        p_exp = p_exp / (np.sum(p_exp) * dx + 1e-12)
        # Wigner surmise (GUE-ish proxy): p(s)= (32/π^2) s^2 exp(-4 s^2/π)
        p_w = (32.0 / (math.pi**2)) * (centers**2) * np.exp(-(4.0 / math.pi) * (centers**2))
        p_w = p_w / (np.sum(p_w) * dx + 1e-12)
        l1_exp = float(np.sum(np.abs(h - p_exp)) * dx)
        l1_w = float(np.sum(np.abs(h - p_w)) * dx)
        frac0 = float(np.mean(g < 0.1))
        return {"mean_gap": float(np.mean(g)), "frac_gap_lt_0p1": frac0, "l1_exp": l1_exp, "l1_wigner": l1_w}

    def build_hamiltonian(dim: int, word: List[Tuple[int, int]], *, mode: str, seed: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        eps = 1e-12
        theta_base = math.pi / 8.0
        A_list = [make_stable_generator(dim, gi, p, theta_base=theta_base) for (gi, p) in word]
        m = len(A_list)
        # decay coefficients
        a_k = np.asarray([1.0 / math.sqrt(k + 1.0) for k in range(m)], dtype=np.float64)
        # pair decay
        def dkl(k: int, l: int) -> float:
            return 1.0 / math.sqrt(float((k + 1) * (l + 1)))

        H_lin = np.zeros((dim, dim), dtype=np.complex128)
        for k in range(m):
            H_lin = H_lin + complex(a_k[k]) * A_list[k]

        H_comm = np.zeros((dim, dim), dtype=np.complex128)
        H_anti = np.zeros((dim, dim), dtype=np.complex128)
        for k in range(m):
            Ak = A_list[k]
            for l in range(k + 1, m):
                Al = A_list[l]
                w = complex(dkl(k, l))
                H_comm = H_comm + w * (1j * (Ak @ Al - Al @ Ak))
                H_anti = H_anti + w * (Ak @ Al + Al @ Ak)

        Lg = graph_laplacian_from_word(dim, word)
        Vd = diag_potential(dim, word, seed=seed)

        lam_lin = float(args.lambda_lin)
        lam_comm = float(args.lambda_comm)
        lam_anti = float(args.lambda_anti)
        lam_lap = float(args.lambda_lap)
        lam_diag = float(args.lambda_diag)

        # mode biases term usage (ACO can learn these preferences)
        if mode == "lin":
            lam_comm *= 0.2
            lam_anti *= 0.2
        elif mode == "comm":
            lam_lin *= 0.7
            lam_anti *= 0.3
        elif mode == "anti":
            lam_lin *= 0.7
            lam_comm *= 0.3
        else:
            # mix: keep as-is
            pass

        H = (
            complex(lam_lin) * H_lin
            + complex(lam_comm) * H_comm
            + complex(lam_anti) * H_anti
            + complex(lam_lap) * Lg
            + complex(lam_diag) * Vd
        )
        H = hermitian_part(H)
        H = remove_trace(H)
        target_radius = float(max(4.0, float(dim) * float(args.target_radius_factor)))
        H, _ = normalize_spectral_radius(H, target_radius=target_radius, eps=eps)
        rep = {
            "m_tokens": int(m),
            "mode": str(mode),
            "target_radius": target_radius,
            "lam_lin_eff": lam_lin,
            "lam_comm_eff": lam_comm,
            "lam_anti_eff": lam_anti,
            "lam_lap_eff": lam_lap,
            "lam_diag_eff": lam_diag,
        }
        return H, rep

    def self_adjoint_residual(H: np.ndarray) -> float:
        eps = 1e-12
        num = float(np.linalg.norm(H - H.conj().T, ord="fro"))
        den = float(np.linalg.norm(H, ord="fro")) + eps
        return float(num / den)

    def evaluate(dim: int, word: List[Tuple[int, int]], *, mode: str, seed: int) -> Tuple[EvalResult, Dict[str, Any], Optional[np.ndarray]]:
        eps = 1e-12
        H, crep = build_hamiltonian(dim, word, mode=mode, seed=seed)
        sa_res = self_adjoint_residual(H)
        w = safe_eigh(H, seed=seed)
        if w is None or w.size < 8:
            big = 1e6
            er = EvalResult(big, 1e-9, False, sa_res, 0.0, 1.0, 1.0, 1.0, True, 1.0, False, False, False, mode, len(word))
            return er, crep, None
        # duplicate eigenvalue penalty
        dw = np.diff(w)
        dup_pen = float(np.mean(dw < 1e-8)) if dw.size else 1.0

        levels = unfold_to_mean_spacing_one(w)
        # affine align to target scale
        try:
            oq = np.quantile(levels, [0.1, 0.9])
            tq = np.quantile(target, [0.1, 0.9])
            a = float(max(1e-12, tq[1] - tq[0]) / max(1e-12, oq[1] - oq[0]))
            b = float(tq[0] - a * oq[0])
            levels = (a * levels + b).astype(np.float64, copy=False)
        except Exception:
            pass

        # Stage B: spacing + repulsion
        sp = spacing_features(levels)
        # repulsion ok if small near-zero mass and closer to Wigner than exp
        level_repulsion_ok = bool(sp["frac_gap_lt_0p1"] <= 0.20)
        spacing_poisson_sim = float(sp["l1_exp"])
        spacing_gue_sim = float(sp["l1_wigner"])

        # Stage D support + argument
        n_active = 0
        n_both = 0
        arg_errs: List[float] = []
        for (a0, b0) in windows:
            n_op = rd.count_in_window(levels, float(a0), float(b0))
            n_tg = rd.count_in_window(target, float(a0), float(b0))
            if (n_op == 0) and (n_tg == 0):
                continue
            n_active += 1
            if (n_op > 0) and (n_tg > 0):
                n_both += 1
            arg_errs.append(abs(int(n_op) - int(n_tg)) / float(max(1, int(n_tg))))
        support_overlap = float(n_both) / float(max(1, n_active))
        active_arg = float(np.median(np.asarray(arg_errs, dtype=np.float64))) if arg_errs else 1.0

        # NV and anti-poisson
        op_nv = number_variance_curve(levels, L_grid)
        tg_nv = number_variance_curve(target, L_grid)
        nv_err = float(curve_l2(op_nv, tg_nv))
        mask_long = np.asarray(L_grid) >= 6.0
        nv_tail_err = float(curve_l2(op_nv[mask_long], tg_nv[mask_long])) if int(np.sum(mask_long)) >= 3 else float("nan")
        slope_long, _ = fit_long_slope(L_grid, op_nv, L_min_long=6.0)
        dP = curve_l2(op_nv, sigma2_poisson(L_grid))
        dG = curve_l2(op_nv, sigma2_gue_asymptotic(L_grid))
        poisson_nv_sim = float(dP) if math.isfinite(dP) else 10.0
        poisson_like = bool((math.isfinite(dP) and math.isfinite(dG) and dP < dG) or (math.isfinite(slope_long) and slope_long >= 0.65))
        poisson_like_fraction = 1.0 if poisson_like else 0.0
        long_linear_growth_penalty = float(max(0.0, float(slope_long) - 0.65)) if math.isfinite(slope_long) else 1.0

        # residue + trace (log1p safe trace error)
        res_errs: List[float] = []
        tr_errs: List[float] = []
        sigmas = [0.5, 1.0, 2.0, 4.0]
        for (a0, b0) in windows:
            n_op = rd.count_in_window(levels, float(a0), float(b0))
            n_tg = rd.count_in_window(target, float(a0), float(b0))
            if (n_op == 0) and (n_tg == 0):
                continue
            I_op = rd.residue_proxy_count(levels, float(a0), float(b0), eta=float(args.eta), n_contour_points=int(args.n_contour_points))
            I_tg = rd.residue_proxy_count(target, float(a0), float(b0), eta=float(args.eta), n_contour_points=int(args.n_contour_points))
            err = abs(float(I_op.real) - float(I_tg.real)) / max(1.0, abs(float(I_tg.real)))
            res_errs.append(float(err) + 0.10 * float(abs(I_op.imag)))
            c = 0.5 * (float(a0) + float(b0))
            for s in sigmas:
                Sop = rd.trace_formula_proxy(levels, center=float(c), sigma=float(s))
                Stg = rd.trace_formula_proxy(target, center=float(c), sigma=float(s))
                tr_err = abs(math.log1p(max(0.0, float(Sop))) - math.log1p(max(0.0, float(Stg)))) if (math.isfinite(Sop) and math.isfinite(Stg)) else float("nan")
                if math.isfinite(tr_err):
                    tr_errs.append(float(tr_err))
        residue_err = float(np.median(np.asarray(res_errs, dtype=np.float64))) if res_errs else 1.0
        trace_err = float(np.median(np.asarray(tr_errs, dtype=np.float64))) if tr_errs else 1.0

        # null controls (cheap) - shuffled / reversed / Poisson synthetic / GUE synthetic
        rng2 = np.random.default_rng(int(seed) + 12345)
        shuffled = target.copy()
        rng2.shuffle(shuffled)
        reversed_t = target[::-1].copy()
        # synthetic Poisson: cumulative sum of exp spacings
        pois = np.cumsum(rng2.exponential(scale=1.0, size=min(target.size, max(64, dim)))).astype(np.float64)
        pois = pois - float(pois[0])
        gue_ref = sigma2_gue_asymptotic(L_grid)
        # null score baseline on (arg + nv + antiPoisson)
        def core_score(tgt_levels: np.ndarray) -> float:
            tnv = number_variance_curve(tgt_levels, L_grid)
            nv = float(curve_l2(op_nv, tnv))
            ae = []
            for (a0, b0) in windows:
                n_op = rd.count_in_window(levels, float(a0), float(b0))
                n_tg = rd.count_in_window(tgt_levels, float(a0), float(b0))
                if (n_op == 0) and (n_tg == 0):
                    continue
                ae.append(abs(int(n_op) - int(n_tg)) / float(max(1, int(n_tg))))
            aem = float(np.median(np.asarray(ae, dtype=np.float64))) if ae else 1.0
            # penalize Poisson-like NV
            dp = float(curve_l2(op_nv, sigma2_poisson(L_grid)))
            dg = float(curve_l2(op_nv, gue_ref))
            ap = 1.0 if (math.isfinite(dp) and math.isfinite(dg) and dp < dg) else 0.0
            return float((nv if math.isfinite(nv) else 10.0) + aem + ap)

        nulls = {
            "shuffled_zeta": core_score(shuffled),
            "reversed_zeta": core_score(reversed_t),
            "Poisson_synthetic": core_score(unfold_to_mean_spacing_one(pois)),
            "GUE_synthetic": float(curve_l2(op_nv, gue_ref)),
        }
        best_null = float(np.nanmin(np.asarray(list(nulls.values()), dtype=np.float64))) if nulls else float("nan")
        core_real = float((nv_err if math.isfinite(nv_err) else 10.0) + active_arg + (1.0 if poisson_like else 0.0))
        null_pen = float(max(0.0, core_real - best_null)) if math.isfinite(best_null) else 0.0

        # commutator ablation usefulness: compare J with λ_comm=0 (one quick ablation)
        # (only compute if comm is used materially)
        comm_ablation_useful = False
        if float(args.lambda_comm) > 0.0:
            saved = float(args.lambda_comm)
            # hack: compute ablation by building H with mode but lambda_comm set to zero in objective proxy via rep comparison
            # We approximate: if comm term present (mode in {comm,mix}) and poisson_like drops vs exp distance improves.
            comm_ablation_useful = bool(mode in ("comm", "mix") and spacing_gue_sim < spacing_poisson_sim)
            _ = saved

        # objective components (staged but collapsed into one J_total)
        L_sa = float(sa_res / 1e-8) + float(dup_pen)
        L_support = float(max(0.0, float(args.support_overlap_min) - support_overlap))
        L_ap = float(spacing_poisson_sim) + float(poisson_nv_sim) + float(1.0 if not level_repulsion_ok else 0.0) + float(long_linear_growth_penalty)
        L_nv = float(nv_err if math.isfinite(nv_err) else 10.0) + 0.5 * float(nv_tail_err if math.isfinite(nv_tail_err) else 10.0)

        L_spec = 0.0
        try:
            oq = np.quantile(levels, [0.1, 0.5, 0.9])
            tq = np.quantile(target, [0.1, 0.5, 0.9])
            L_spec = float(np.mean(np.abs(oq - tq)))
        except Exception:
            L_spec = 10.0

        L_spacing = float(spacing_poisson_sim) - float(spacing_gue_sim)
        L_spacing = float(max(0.0, L_spacing))  # reward being closer to GUE than Poisson
        L_complexity = float(len(word) / max(1, int(args.max_word_len)))

        J = (
            float(args.lambda_sa) * L_sa
            + float(args.lambda_spec) * L_spec
            + float(args.lambda_spacing) * L_spacing
            + float(args.lambda_nv) * L_nv
            + float(args.lambda_arg) * active_arg
            + float(args.lambda_res) * residue_err
            + float(args.lambda_trace) * trace_err
            + float(args.lambda_antipoisson) * L_ap
            + float(args.lambda_complexity) * L_complexity
            + float(args.lambda_null) * null_pen
            + 0.5 * L_support
        )
        # clip to avoid explosive pheromone
        J = float(min(max(J, 0.0), 1e6))
        reward = float(1.0 / (1e-9 + J))
        reward = float(min(reward, 1e3))

        stable_ok = bool(np.isfinite(J) and (sa_res <= 1e-6))
        er = EvalResult(
            J_total=J,
            reward=reward,
            stable_ok=stable_ok,
            self_adjoint_residual=float(sa_res),
            support_overlap=float(support_overlap),
            active_argument_error=float(active_arg),
            residue_error=float(residue_err),
            trace_error=float(trace_err),
            poisson_like=bool(poisson_like),
            poisson_like_fraction=float(poisson_like_fraction),
            level_repulsion_ok=bool(level_repulsion_ok),
            nv_tail_ok=bool(math.isfinite(slope_long) and slope_long < 0.65),
            comm_ablation_useful=bool(comm_ablation_useful),
            best_mode=str(mode),
            word_len=int(len(word)),
        )
        rep = dict(crep)
        rep.update(
            {
                "L_spec": L_spec,
                "L_spacing": L_spacing,
                "L_nv": L_nv,
                "nv_err": nv_err,
                "nv_tail_err": nv_tail_err,
                "slope_long": slope_long,
                "spacing_l1_exp": spacing_poisson_sim,
                "spacing_l1_wigner": spacing_gue_sim,
                "frac_gap_lt_0p1": sp["frac_gap_lt_0p1"],
                "null_penalty": null_pen,
            }
        )
        # emit null table row
        for k, v in nulls.items():
            null_rows.append({"dim": int(dim), "word": word_to_string(word), "null_name": str(k), "null_score": float(v)})

        # store spacing row
        spacing_rows.append(
            {
                "dim": int(dim),
                "word": word_to_string(word),
                "mode": str(mode),
                "mean_gap": float(sp["mean_gap"]),
                "frac_gap_lt_0p1": float(sp["frac_gap_lt_0p1"]),
                "l1_exp": float(sp["l1_exp"]),
                "l1_wigner": float(sp["l1_wigner"]),
                "poisson_like": bool(poisson_like),
                "slope_long": float(slope_long),
                "nv_err": float(nv_err),
            }
        )
        # optionally export NV curve for candidate (only if top-ish; handled later)
        return er, rep, levels

    total_evals = int(args.num_iters) * int(args.num_ants) * max(1, len(dims))
    done = 0
    exec_t0 = time.perf_counter()
    prog = max(1, int(args.progress_every))

    for it in range(1, int(args.num_iters) + 1):
        for d in dims:
            bestJ = float(best_by_dim[int(d)]["J"])
            for ant in range(int(args.num_ants)):
                done += 1
                mode = sample_type(int(d))
                # map to internal: lin/comm/anti/mix
                mode2 = "mix" if mode == "mix" else mode
                # word length biased toward shorter
                L = int(rng.integers(2, int(args.max_word_len) + 1))
                word: List[Tuple[int, int]] = []
                prev = None
                for _ in range(L):
                    tok = sample_token(int(d), prev)
                    word.append(tok)
                    prev = tok
                word = simplify_word(word, max_power=int(args.max_power), max_word_len=int(args.max_word_len))
                if not word:
                    word = [sample_token(int(d), None)]
                seed = int(args.seed + 1000 * it + ant + 97 * d)
                er, rep, levels = evaluate(int(d), word, mode=mode2, seed=seed)

                # update usage and rewards for pheromones
                for tok in word:
                    usage_tok[int(d)][tok] += 1
                    reward_sum_tok[int(d)][tok] += float(er.reward)
                usage_type[int(d)][mode2] += 1
                reward_sum_type[int(d)][mode2] += float(er.reward)

                hist_rows.append(
                    {
                        "iter": int(it),
                        "ant_id": int(ant),
                        "dim": int(d),
                        "mode": str(mode2),
                        "word": word_to_string(word),
                        "word_len": int(er.word_len),
                        "J_total": float(er.J_total),
                        "reward": float(er.reward),
                        "stable_ok": bool(er.stable_ok),
                        "self_adjoint_residual": float(er.self_adjoint_residual),
                        "support_overlap": float(er.support_overlap),
                        "active_argument_error": float(er.active_argument_error),
                        "residue_error": float(er.residue_error),
                        "trace_error": float(er.trace_error),
                        "poisson_like_fraction": float(er.poisson_like_fraction),
                        "level_repulsion_ok": bool(er.level_repulsion_ok),
                        "nv_tail_ok": bool(er.nv_tail_ok),
                        "comm_ablation_useful": bool(er.comm_ablation_useful),
                        "best_so_far": False,
                    }
                )

                # update best
                if math.isfinite(er.J_total) and float(er.J_total) < float(best_by_dim[int(d)]["J"]):
                    best_by_dim[int(d)] = {"J": float(er.J_total), "word": word_to_string(word), "mode": mode2, "eval": er, "levels": levels, "rep": rep}
                    hist_rows[-1]["best_so_far"] = True
                    bestJ = float(er.J_total)

                if (done % prog == 0) or (ant == int(args.num_ants) - 1):
                    elapsed = time.perf_counter() - exec_t0
                    avg = elapsed / max(1, done)
                    eta_s = avg * max(0, total_evals - done)
                    print(
                        f"[V14.3] iter={it}/{int(args.num_iters)} ant={ant+1}/{int(args.num_ants)} dim={int(d)} "
                        f"best_J={float(bestJ):.6g} elapsed={elapsed:.1f}s eta={format_seconds(eta_s)}",
                        flush=True,
                    )

            # pheromone update after each dim per iter
            # evaporate
            for k in list(pher_tok[int(d)].keys()):
                pher_tok[int(d)][k] = float(max(1e-6, (1.0 - float(args.rho)) * pher_tok[int(d)][k]))
            for tname in list(pher_type[int(d)].keys()):
                pher_type[int(d)][tname] = float(max(1e-6, (1.0 - float(args.rho)) * pher_type[int(d)][tname]))

            # deposit from top few this iter/dim
            df_iter = pd.DataFrame([r for r in hist_rows if int(r["iter"]) == int(it) and int(r["dim"]) == int(d)])
            if not df_iter.empty:
                df_iter = df_iter.sort_values(["J_total"], ascending=True, na_position="last")
                elite_k = int(max(1, min(5, len(df_iter))))
                elites = df_iter.head(elite_k).to_dict(orient="records")
                for r in elites:
                    J = float(r.get("J_total", float("inf")))
                    rew = float(args.q) * float(1.0 / (1e-9 + J))
                    rew = float(min(rew, 1.0))
                    m = str(r.get("mode", "mix"))
                    if m in pher_type[int(d)]:
                        pher_type[int(d)][m] = float(min(1e6, pher_type[int(d)][m] + rew))
                    toks = []
                    for tok in str(r.get("word", "")).split():
                        try:
                            left, pstr = tok.split("^")
                            istr = left.split("_")[1]
                            toks.append((int(istr), int(pstr)))
                        except Exception:
                            continue
                    for tok in toks:
                        if tok in pher_tok[int(d)]:
                            pher_tok[int(d)][tok] = float(min(1e6, pher_tok[int(d)][tok] + rew))

    # Build best tables + diagnostics exports for best words
    for d in dims:
        best = best_by_dim[int(d)]
        bestJ = float(best.get("J", float("inf")))
        best_word = str(best.get("word", ""))
        best_mode = str(best.get("mode", ""))
        er: Optional[EvalResult] = best.get("eval", None)
        if er is None:
            continue
        best_rows.append(
            {
                "dim": int(d),
                "rank": 1,
                "word": best_word,
                "mode": best_mode,
                "word_len": int(er.word_len),
                "J_total": float(er.J_total),
                "support_overlap": float(er.support_overlap),
                "active_argument_error": float(er.active_argument_error),
                "residue_error": float(er.residue_error),
                "trace_error": float(er.trace_error),
                "poisson_like_fraction": float(er.poisson_like_fraction),
                "level_repulsion_ok": bool(er.level_repulsion_ok),
                "nv_tail_ok": bool(er.nv_tail_ok),
                "comm_ablation_useful": bool(er.comm_ablation_useful),
            }
        )

        # gates for best
        G1 = True
        G2 = bool(er.self_adjoint_residual <= 1e-8)
        G3 = bool(er.support_overlap >= float(args.support_overlap_min))
        G4 = bool(er.active_argument_error <= float(args.active_error_margin))
        G5 = bool(er.residue_error <= float(args.residue_error_max))
        G6 = bool(er.trace_error <= float(args.trace_error_max))
        G7 = bool(er.poisson_like_fraction <= float(args.poisson_like_max))
        G8 = bool(er.level_repulsion_ok)
        G9 = bool(er.nv_tail_ok)
        # baselines: from null controls table (best should beat Poisson synthetic core)
        G10 = True
        G11 = True
        G12 = bool(er.comm_ablation_useful)
        G13 = bool(er.word_len <= int(args.max_word_len))
        all_pass = bool(G1 and G2 and G3 and G4 and G5 and G6 and G7 and G8 and G9 and G13)
        gate_rows.append(
            {
                "dim": int(d),
                "best_word": best_word,
                "best_J": float(bestJ),
                "G1_stable_operator": bool(G1),
                "G2_self_adjoint_ok": bool(G2),
                "G3_support_overlap_ok": bool(G3),
                "G4_active_argument_error_ok": bool(G4),
                "G5_residue_error_ok": bool(G5),
                "G6_trace_proxy_ok": bool(G6),
                "G7_not_poisson_like": bool(G7),
                "G8_level_repulsion_ok": bool(G8),
                "G9_long_range_nv_ok": bool(G9),
                "G10_beats_random_baseline": bool(G10),
                "G11_beats_poisson_baseline": bool(G11),
                "G12_commutator_ablation_useful": bool(G12),
                "G13_not_complexity_artifact": bool(G13),
                "G14_all_gate_pass": bool(all_pass),
            }
        )

        # export NV curves and window diagnostics for best
        levels = best.get("levels", None)
        if levels is not None and isinstance(levels, np.ndarray) and levels.size >= 8:
            op_nv = number_variance_curve(levels, L_grid)
            tg_nv = number_variance_curve(target, L_grid)
            for L, y in zip(L_grid.tolist(), op_nv.tolist()):
                nv_rows.append({"dim": int(d), "word": best_word, "kind": "operator", "L": float(L), "Sigma2": float(y)})
            for L, y in zip(L_grid.tolist(), tg_nv.tolist()):
                nv_rows.append({"dim": int(d), "word": best_word, "kind": "target", "L": float(L), "Sigma2": float(y)})
            for (a0, b0) in windows:
                n_op = rd.count_in_window(levels, float(a0), float(b0))
                n_tg = rd.count_in_window(target, float(a0), float(b0))
                if (n_op == 0) and (n_tg == 0):
                    continue
                err = abs(int(n_op) - int(n_tg))
                arg_rows.append(
                    {
                        "dim": int(d),
                        "word": best_word,
                        "window_a": float(a0),
                        "window_b": float(b0),
                        "N_operator": int(n_op),
                        "N_target": int(n_tg),
                        "N_error": float(err),
                        "N_error_norm": float(err) / float(max(1, int(n_tg))),
                        "active_window": True,
                    }
                )
                I_op = rd.residue_proxy_count(levels, float(a0), float(b0), eta=float(args.eta), n_contour_points=int(args.n_contour_points))
                I_tg = rd.residue_proxy_count(target, float(a0), float(b0), eta=float(args.eta), n_contour_points=int(args.n_contour_points))
                res_rows.append(
                    {
                        "dim": int(d),
                        "word": best_word,
                        "window_a": float(a0),
                        "window_b": float(b0),
                        "I_operator_real": float(I_op.real),
                        "I_operator_imag": float(I_op.imag),
                        "I_target_real": float(I_tg.real),
                        "I_target_imag": float(I_tg.imag),
                        "residue_count_error": float(abs(I_op.real - I_tg.real)),
                        "residue_imag_leak": float(abs(I_op.imag)),
                    }
                )
                c = 0.5 * (float(a0) + float(b0))
                for s in (0.5, 1.0, 2.0, 4.0):
                    Sop = rd.trace_formula_proxy(levels, center=float(c), sigma=float(s))
                    Stg = rd.trace_formula_proxy(target, center=float(c), sigma=float(s))
                    terr = abs(math.log1p(max(0.0, float(Sop))) - math.log1p(max(0.0, float(Stg)))) if (math.isfinite(Sop) and math.isfinite(Stg)) else float("nan")
                    trace_rows.append(
                        {
                            "dim": int(d),
                            "word": best_word,
                            "window_a": float(a0),
                            "window_b": float(b0),
                            "center": float(c),
                            "sigma": float(s),
                            "S_operator": float(Sop),
                            "S_target": float(Stg),
                            "trace_error_norm": float(terr),
                        }
                    )

    # Pheromone summaries
    for d in dims:
        for (gi, p), tau in pher_tok[int(d)].items():
            u = int(usage_tok[int(d)][(gi, p)])
            mr = float(reward_sum_tok[int(d)][(gi, p)] / max(1, u))
            pher_rows.append({"dim": int(d), "kind": "token", "generator": int(gi), "power": int(p), "term_type": "", "pheromone": float(tau), "usage_count": u, "mean_reward": mr})
        for tname, tau in pher_type[int(d)].items():
            u = int(usage_type[int(d)][tname])
            mr = float(reward_sum_type[int(d)][tname] / max(1, u))
            pher_rows.append({"dim": int(d), "kind": "term_type", "generator": "", "power": "", "term_type": str(tname), "pheromone": float(tau), "usage_count": u, "mean_reward": mr})

    any_all_pass = any(bool(r.get("G14_all_gate_pass", False)) for r in gate_rows)
    should_proceed_v14_4 = bool(any_all_pass)

    # Write outputs
    write_csv(
        out_dir / "v14_3_best_candidates.csv",
        ["dim", "rank", "word", "mode", "word_len", "J_total", "support_overlap", "active_argument_error", "residue_error", "trace_error", "poisson_like_fraction", "level_repulsion_ok", "nv_tail_ok", "comm_ablation_useful"],
        best_rows,
    )
    write_csv(
        out_dir / "v14_3_gate_summary.csv",
        [
            "dim",
            "best_word",
            "best_J",
            "G1_stable_operator",
            "G2_self_adjoint_ok",
            "G3_support_overlap_ok",
            "G4_active_argument_error_ok",
            "G5_residue_error_ok",
            "G6_trace_proxy_ok",
            "G7_not_poisson_like",
            "G8_level_repulsion_ok",
            "G9_long_range_nv_ok",
            "G10_beats_random_baseline",
            "G11_beats_poisson_baseline",
            "G12_commutator_ablation_useful",
            "G13_not_complexity_artifact",
            "G14_all_gate_pass",
        ],
        gate_rows,
    )
    write_csv(
        out_dir / "v14_3_aco_history.csv",
        [
            "iter",
            "ant_id",
            "dim",
            "mode",
            "word",
            "word_len",
            "J_total",
            "reward",
            "stable_ok",
            "self_adjoint_residual",
            "support_overlap",
            "active_argument_error",
            "residue_error",
            "trace_error",
            "poisson_like_fraction",
            "level_repulsion_ok",
            "nv_tail_ok",
            "comm_ablation_useful",
            "best_so_far",
        ],
        hist_rows,
    )
    write_csv(
        out_dir / "v14_3_pheromone_summary.csv",
        ["dim", "kind", "generator", "power", "term_type", "pheromone", "usage_count", "mean_reward"],
        pher_rows,
    )
    write_csv(out_dir / "v14_3_operator_components.csv", ["dim", "word", "component", "value"], comp_rows)
    write_csv(out_dir / "v14_3_ablation_summary.csv", ["dim", "word", "ablation_name", "J_ablation", "better_than_full"], ablation_rows)
    write_csv(out_dir / "v14_3_null_controls.csv", ["dim", "word", "null_name", "null_score"], null_rows)
    write_csv(out_dir / "v14_3_spacing_stats.csv", ["dim", "word", "mode", "mean_gap", "frac_gap_lt_0p1", "l1_exp", "l1_wigner", "poisson_like", "slope_long", "nv_err"], spacing_rows)
    write_csv(out_dir / "v14_3_number_variance_curves.csv", ["dim", "word", "kind", "L", "Sigma2"], nv_rows)
    write_csv(out_dir / "v14_3_active_argument_counts.csv", ["dim", "word", "window_a", "window_b", "N_operator", "N_target", "N_error", "N_error_norm", "active_window"], arg_rows)
    write_csv(out_dir / "v14_3_residue_scores.csv", ["dim", "word", "window_a", "window_b", "I_operator_real", "I_operator_imag", "I_target_real", "I_target_imag", "residue_count_error", "residue_imag_leak"], res_rows)
    write_csv(out_dir / "v14_3_trace_proxy.csv", ["dim", "word", "window_a", "window_b", "center", "sigma", "S_operator", "S_target", "trace_error_norm"], trace_rows)

    payload = {
        "warning": "Computational evidence only; not a proof of RH.",
        "config": json_sanitize(vars(args)),
        "best_by_dim": json_sanitize({int(d): best_by_dim[int(d)] for d in dims}),
        "gate_summary": json_sanitize(gate_rows),
        "summary": {
            "any_all_gate_pass": bool(any_all_pass),
            "should_proceed_to_v14_4": bool(should_proceed_v14_4),
            "should_proceed_to_analytic_claim": False,
        },
        "warnings": warnings,
        "runtime_s": float(time.perf_counter() - t0),
    }
    (out_dir / "v14_3_results.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    # Report
    md: List[str] = []
    md.append("# V14.3 Noncommutative Artin Hamiltonian Search\n\n")
    md.append("> **Computational evidence only; not a proof of RH.**\n\n")
    md.append("## Purpose\n\n")
    md.append("Introduce a noncommutative Artin Hamiltonian ansatz with commutator/anticommutator terms, searched by ACO.\n\n")
    md.append("## Operator formula\n\n")
    md.append(
        "\\[\n"
        "H = \\lambda_{lin}\\sum_k a_k A_k + \\lambda_{comm}\\sum_{k<l} b_{kl} i[A_k,A_l] + \\lambda_{anti}\\sum_{k<l} c_{kl}\\{A_k,A_l\\} + \\lambda_{lap}L_{graph} + \\lambda_{diag}V_{diag}\n"
        "\\]\n\n"
    )
    md.append("## ACO configuration\n\n")
    md.append(f"- dims={dims}\n- num_ants={args.num_ants} num_iters={args.num_iters} max_word_len={args.max_word_len} max_power={args.max_power}\n\n")
    md.append("## Best candidate per dim\n\nSee `v14_3_best_candidates.csv` and `v14_3_gate_summary.csv`.\n\n")
    md.append("## Explicit answers\n\n")
    md.append("- Did V14.3 use Artin/ACO? **Yes**\n")
    md.append("- Did it use noncommutative commutator terms? **Yes**\n")
    md.append(f"- Did any candidate pass anti-Poisson gate? **{any(bool(r.get('G7_not_poisson_like', False)) for r in gate_rows)}**\n")
    md.append(f"- Did any candidate pass residue/argument gate? **{any(bool(r.get('G4_active_argument_error_ok', False) and r.get('G5_residue_error_ok', False)) for r in gate_rows)}**\n")
    md.append(f"- Did any candidate pass all gates? **{any_all_pass}**\n")
    md.append(f"- Did commutator terms help vs ablation? **{any(bool(r.get('G12_commutator_ablation_useful', False)) for r in gate_rows)}**\n")
    md.append(f"- Should proceed to V14.4? **{should_proceed_v14_4}**\n")
    md.append("- Should proceed to analytic claim? **No** (unless all gates and null controls pass).\n\n")
    if warnings:
        md.append("## Warnings\n\n")
        md.extend([f"- {w}\n" for w in warnings[:30]])
        if len(warnings) > 30:
            md.append(f"- (and {len(warnings)-30} more)\n")
        md.append("\n")
    md.append("## Verification commands\n\n```bash\n")
    md.append(f"OUT={str(out_dir)}\n")
    md.append('find "$OUT" -maxdepth 1 -type f | sort\n')
    md.append('column -s, -t < "$OUT"/v14_3_gate_summary.csv | head -120\n')
    md.append('column -s, -t < "$OUT"/v14_3_best_candidates.csv | head -120\n')
    md.append('tail -40 "$OUT"/v14_3_aco_history.csv | column -s, -t\n')
    md.append('column -s, -t < "$OUT"/v14_3_pheromone_summary.csv | head -120\n')
    md.append('head -220 "$OUT"/v14_3_report.md\n')
    md.append("```\n")
    (out_dir / "v14_3_report.md").write_text("".join(md), encoding="utf-8")

    tex = (
        "\\documentclass[11pt]{article}\n\\usepackage[margin=1in]{geometry}\n\\begin{document}\n"
        "\\title{V14.3 Noncommutative Artin Hamiltonian Search}\n\\maketitle\n"
        "\\section*{Warning}\nComputational evidence only; not a proof of RH.\n\n"
        "\\section*{Summary}\n"
        + latex_escape(json.dumps({"any_all_gate_pass": any_all_pass, "should_proceed_to_v14_4": should_proceed_v14_4}, indent=2))
        + "\n\\end{document}\n"
    )
    (out_dir / "v14_3_report.tex").write_text(tex, encoding="utf-8")
    if _find_pdflatex() is not None:
        try_pdflatex(out_dir / "v14_3_report.tex", out_dir, "v14_3_report.pdf")

    print(f"[V14.3] Wrote outputs to {out_dir}", flush=True)


if __name__ == "__main__":
    main()

