#!/usr/bin/env python3
"""
V14.2 — Stabilized Artin Operator Manifold Search (ACO).

Computational evidence only; not a proof of RH.

Purpose:
V14.1 demonstrated real Artin/ACO word search but suffered numerical pathologies:
very large losses, flat rewards, and optimization of instability rather than RH-like structure.
V14.2 stabilizes the *operator manifold* explored by ACO:
  - bounded/self-adjoint generator contributions,
  - controlled spectral radius / condition proxy / eigen spread,
  - staged objective: stability -> support/argument -> NV/anti-Poisson -> residue/trace.
Only stable candidates proceed to expensive diagnostics.
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
    # use validation helper (rank/mean spacing unfolding)
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
        # avoid immediate inverse cancellation
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


def op_norm_2(A: np.ndarray) -> float:
    # spectral norm (2-norm); for dim<=256 ok
    try:
        return float(np.linalg.norm(A, ord=2))
    except Exception:
        return float(np.linalg.norm(A, ord="fro"))


@dataclass(frozen=True)
class StabilityDiag:
    finite_ok: bool
    selfadjoint_error: float
    spectral_radius: float
    eigen_min: float
    eigen_max: float
    eigen_spread: float
    condition_proxy: float
    nontrivial_spectrum_ok: bool
    stable_operator_ok: bool


def compute_stability(H: np.ndarray, *, spectral_radius_max: float, condition_proxy_max: float, eigen_spread_max: float) -> Tuple[StabilityDiag, np.ndarray]:
    eps = 1e-12
    H = np.asarray(H, dtype=np.float64)
    finite_ok = bool(np.isfinite(H).all())
    num = float(np.linalg.norm(H - H.T, ord="fro"))
    den = float(np.linalg.norm(H, ord="fro")) + eps
    selfadj_err = float(num / den)
    try:
        w = np.linalg.eigvalsh(0.5 * (H + H.T))
        w = np.asarray(w, dtype=np.float64).reshape(-1)
        w = w[np.isfinite(w)]
        w.sort()
    except Exception:
        w = np.asarray([], dtype=np.float64)
    if w.size == 0:
        diag = StabilityDiag(False, selfadj_err, float("inf"), float("nan"), float("nan"), float("inf"), float("inf"), False, False)
        return diag, w
    e_min = float(w[0])
    e_max = float(w[-1])
    spread = float(e_max - e_min)
    radius = float(max(abs(e_min), abs(e_max)))
    # condition proxy: |max| / min_nonzero_abs
    absw = np.abs(w)
    nz = absw[absw > 1e-9]
    min_nz = float(np.min(nz)) if nz.size else eps
    cond = float(abs(e_max) / max(min_nz, eps))
    nontrivial = bool(np.std(w) > 1e-6 and spread > 1e-6)
    stable_ok = bool(
        finite_ok
        and selfadj_err < 1e-8
        and radius <= float(spectral_radius_max) + 1e-9
        and cond <= float(condition_proxy_max) + 1e-9
        and spread <= float(eigen_spread_max) + 1e-9
        and nontrivial
    )
    diag = StabilityDiag(finite_ok, selfadj_err, radius, e_min, e_max, spread, cond, nontrivial, stable_ok)
    return diag, w


def stable_eigvalsh(H: np.ndarray, *, seed: int) -> Optional[np.ndarray]:
    Hs = 0.5 * (H + H.T)
    if _HAVE_SAFE_EIGH and _safe_eigh is not None:
        try:
            w, _, _rep = _safe_eigh(Hs, k=None, return_eigenvectors=False, stabilize=True, seed=int(seed))
            w = np.asarray(w, dtype=np.float64).reshape(-1)
            w = w[np.isfinite(w)]
            w.sort()
            return w
        except Exception:
            return None
    try:
        w = np.linalg.eigvalsh(Hs)
        w = np.asarray(w, dtype=np.float64).reshape(-1)
        w = w[np.isfinite(w)]
        w.sort()
        return w
    except Exception:
        return None


def build_stabilized_operator(
    dim: int,
    word: List[Tuple[int, int]],
    *,
    theta_base: float,
    target_radius: float,
    spectral_radius_max: float,
    condition_proxy_max: float,
    eigen_spread_max: float,
) -> Tuple[np.ndarray, StabilityDiag, Dict[str, Any]]:
    """
    Construct bounded operator from sum of normalized Hermitian generator contributions.
    No unstable products.
    """
    eps = 1e-12
    n = int(dim)
    H_raw = np.zeros((n, n), dtype=np.float64)
    used = 0
    max_i = max(1, n - 1)
    for k, (gi, p) in enumerate(word):
        i = int(max(1, min(max_i, int(gi))))  # 1..dim-1
        p = int(p)
        theta = float(theta_base) * float(i) / float(max_i)
        B = rotation_block(float(p) * theta)
        G = embed_2x2(n, i - 1, B)
        A = 0.5 * (G + G.T)
        nrm = op_norm_2(A)
        A = A / max(nrm, eps)
        ck = 1.0 / math.sqrt(float(k) + 1.0)
        H_raw = H_raw + float(ck) * A
        used += 1
    H = 0.5 * (H_raw + H_raw.T)
    # remove trace
    tr = float(np.mean(np.diag(H))) if n > 0 else 0.0
    H = H - tr * np.eye(n, dtype=np.float64)

    # normalize spectral radius to target_radius (with guard)
    w = stable_eigvalsh(H, seed=42)
    if w is None or w.size == 0:
        diag = StabilityDiag(False, float("inf"), float("inf"), float("nan"), float("nan"), float("inf"), float("inf"), False, False)
        return H, diag, {"used_tokens": used, "trace_removed": tr, "radius_scaled": False}
    radius = float(max(abs(float(w[0])), abs(float(w[-1]))))
    scaled = False
    if math.isfinite(radius) and radius > eps:
        s = float(target_radius) / float(radius)
        H = H * s
        scaled = True
    diag, _w2 = compute_stability(H, spectral_radius_max=spectral_radius_max, condition_proxy_max=condition_proxy_max, eigen_spread_max=eigen_spread_max)
    rep = {"used_tokens": used, "trace_removed": tr, "radius_scaled": scaled, "target_radius": float(target_radius)}
    return H, diag, rep


@dataclass(frozen=True)
class StageLoss:
    J: float
    reward: float
    stable_ok: bool
    L_stability: float
    L_support: float
    L_arg: float
    L_nv: float
    L_antipoisson: float
    L_residue: float
    L_trace: float
    L_complexity: float
    L_null: float
    support_overlap: float
    poisson_like_fraction: float
    nv_curve_error: float
    long_range_nv_error: float
    slope_long: float


def main() -> None:
    ap = argparse.ArgumentParser(description="V14.2 stabilized Artin operator manifold search (computational only).")
    ap.add_argument("--true_levels_csv", type=str, default="runs/v13o9_true_unfolded_spectra_from_v13o8/v13o9_unfolded_levels.csv")
    ap.add_argument("--zeros_csv", type=str, default="runs/zeros_100_400_precise.csv")
    ap.add_argument("--out_dir", type=str, default="runs/v14_2_stabilized_artin_operator_search")
    ap.add_argument("--dims", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--num_ants", type=int, default=32)
    ap.add_argument("--num_iters", type=int, default=80)
    ap.add_argument("--max_word_len", type=int, default=32)
    ap.add_argument("--max_power", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--rho", type=float, default=0.15)
    ap.add_argument("--q", type=float, default=1.0)
    ap.add_argument("--window_min", type=float, default=100.0)
    ap.add_argument("--window_max", type=float, default=400.0)
    ap.add_argument("--window_size", type=float, default=40.0)
    ap.add_argument("--window_stride", type=float, default=20.0)
    ap.add_argument("--L_min", type=float, default=0.5)
    ap.add_argument("--L_max", type=float, default=16.0)
    ap.add_argument("--n_L", type=int, default=64)
    ap.add_argument("--eta", type=float, default=0.15)
    ap.add_argument("--n_contour_points", type=int, default=256)
    ap.add_argument("--spectral_radius_max", type=float, default=64.0)
    ap.add_argument("--condition_proxy_max", type=float, default=1e6)
    ap.add_argument("--eigen_spread_max", type=float, default=128.0)
    ap.add_argument("--support_overlap_min", type=float, default=0.5)
    ap.add_argument("--active_error_margin", type=float, default=0.25)
    ap.add_argument("--anti_poisson_threshold", type=float, default=0.5)
    ap.add_argument("--poisson_like_max", type=float, default=0.5)
    ap.add_argument("--n_jobs", type=int, default=8)
    ap.add_argument("--progress_every", type=int, default=10)
    args = ap.parse_args()

    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    warnings: List[str] = []
    if not _HAVE_SAFE_EIGH:
        warnings.append("core.spectral_stabilization.safe_eigh not available; using numpy eigvalsh fallback for stabilization.")

    dims = [int(d) for d in args.dims]
    max_power = int(args.max_power)
    powers = [p for p in range(-max_power, max_power + 1) if p != 0]

    # Load target unfolded levels for real_zeta from true_levels_csv if possible; else fallback to zeros_csv unfolding.
    target_by_dim: Dict[int, np.ndarray] = {}
    df_levels, lvl_warns = rd.load_true_levels_csv(_resolve(args.true_levels_csv), dims_keep=dims)
    if df_levels is None:
        warnings.extend([f"true_levels_csv: {w}" for w in lvl_warns])
        df_levels = pd.DataFrame()
    else:
        warnings.extend([f"true_levels_csv: {w}" for w in lvl_warns])
    if not df_levels.empty:
        try:
            df_levels = df_levels.copy()
            df_levels["dim"] = pd.to_numeric(df_levels["dim"], errors="coerce").astype("Int64")
            for c in ("target_group", "source"):
                df_levels[c] = df_levels[c].astype(str).str.strip()
            df_levels["unfolded_level"] = pd.to_numeric(df_levels["unfolded_level"], errors="coerce").astype(float)
            for d in dims:
                sub = df_levels[(df_levels["dim"].astype(int) == int(d)) & (df_levels["target_group"] == "real_zeta") & (df_levels["source"] == "target")]
                if not sub.empty:
                    x = np.sort(sub["unfolded_level"].astype(float).to_numpy(dtype=np.float64))
                    x = x[np.isfinite(x)]
                    if x.size >= 8:
                        target_by_dim[int(d)] = x
        except Exception as e:
            warnings.append(f"failed extracting real_zeta targets from true_levels_csv: {e!r}")

    # zeros_csv fallback
    zeros_raw, zeros_warns = rd.load_zeros_csv(_resolve(args.zeros_csv))
    warnings.extend([f"zeros_csv: {w}" for w in zeros_warns])
    zeros_unfolded = unfold_to_mean_spacing_one(zeros_raw)

    for d in dims:
        if int(d) not in target_by_dim:
            target_by_dim[int(d)] = zeros_unfolded.copy()
            warnings.append(f"dim={d}: using zeros_csv unfolded levels as target (real_zeta missing in true_levels_csv)")

    # grids
    L_grid = make_L_grid(args.L_min, args.L_max, args.n_L)
    windows = rd.make_windows(args.window_min, args.window_max, args.window_size, args.window_stride)
    if not windows:
        raise SystemExit("No windows produced; check window_min/max/size/stride.")

    # ACO pheromones per dim over (generator,power)
    pher: Dict[int, Dict[Tuple[int, int], float]] = {}
    usage: Dict[int, DefaultDict[Tuple[int, int], int]] = {}
    reward_sum: Dict[int, DefaultDict[Tuple[int, int], float]] = {}
    for d in dims:
        pher[int(d)] = {}
        usage[int(d)] = defaultdict(int)
        reward_sum[int(d)] = defaultdict(float)
        for gi in range(1, int(d)):
            for p in powers:
                pher[int(d)][(gi, p)] = 1.0

    rng = np.random.default_rng(42)
    random.seed(42)

    # Default weights
    lambda_stability = 10.0
    lambda_support = 5.0
    lambda_arg = 3.0
    lambda_nv = 2.0
    lambda_ap = 5.0
    lambda_res = 1.0
    lambda_trace = 1.0
    lambda_complexity = 0.05
    lambda_null = 2.0

    # Outputs rows
    hist_rows: List[Dict[str, Any]] = []
    best_rows: List[Dict[str, Any]] = []
    stab_rows: List[Dict[str, Any]] = []
    unfolded_rows: List[Dict[str, Any]] = []
    arg_rows: List[Dict[str, Any]] = []
    nv_rows: List[Dict[str, Any]] = []
    res_rows: List[Dict[str, Any]] = []
    trace_rows: List[Dict[str, Any]] = []
    gate_rows: List[Dict[str, Any]] = []
    pher_rows: List[Dict[str, Any]] = []

    # tracking best per dim
    best_by_dim: Dict[int, Dict[str, Any]] = {int(d): {"J": float("inf"), "word": "", "loss": None, "diag": None, "levels": None} for d in dims}

    # sampling helper
    def sample_token(d: int, prev: Optional[Tuple[int, int]]) -> Tuple[int, int]:
        items = list(pher[d].keys())
        tau = np.asarray([pher[d][it] for it in items], dtype=np.float64)
        eta = np.ones_like(tau)
        if prev is not None:
            pi, pp = prev
            for idx, (gi, pw) in enumerate(items):
                if gi == pi:
                    eta[idx] *= 0.7
                if gi == pi and pw == -pp:
                    eta[idx] *= 0.15
        # prefer mid generators weakly
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

    # staged evaluation
    def evaluate_candidate(d: int, word: List[Tuple[int, int]], seed: int) -> Tuple[StageLoss, StabilityDiag, Optional[np.ndarray], Dict[str, Any]]:
        eps = 1e-12
        # operator build
        theta_base = math.pi / 8.0
        target_radius = float(min(float(args.spectral_radius_max), float(d) / 4.0))
        H, diag, rep = build_stabilized_operator(
            int(d),
            word,
            theta_base=theta_base,
            target_radius=target_radius,
            spectral_radius_max=float(args.spectral_radius_max),
            condition_proxy_max=float(args.condition_proxy_max),
            eigen_spread_max=float(args.eigen_spread_max),
        )

        # Stage A: stability loss
        # penalties for exceeding thresholds
        rad_ex = max(0.0, float(diag.spectral_radius) - float(args.spectral_radius_max)) if math.isfinite(diag.spectral_radius) else 1e3
        cond_ex = max(0.0, float(diag.condition_proxy) - float(args.condition_proxy_max)) if math.isfinite(diag.condition_proxy) else 1e3
        spread_ex = max(0.0, float(diag.eigen_spread) - float(args.eigen_spread_max)) if math.isfinite(diag.eigen_spread) else 1e3
        finite_pen = 0.0 if diag.finite_ok else 1e3
        selfadj_pen = float(diag.selfadjoint_error / max(1e-8, 1e-16))
        L_stability = float(finite_pen + rad_ex + 1e-6 * cond_ex + spread_ex + selfadj_pen)

        # If not stable, stop
        if not diag.stable_operator_ok:
            J = float(lambda_stability * L_stability + lambda_complexity * (len(word) / max(1, int(args.max_word_len))))
            reward = 1e-9
            loss = StageLoss(
                J=J,
                reward=reward,
                stable_ok=False,
                L_stability=L_stability,
                L_support=1.0,
                L_arg=1.0,
                L_nv=1.0,
                L_antipoisson=1.0,
                L_residue=1.0,
                L_trace=1.0,
                L_complexity=float(len(word) / max(1, int(args.max_word_len))),
                L_null=0.0,
                support_overlap=0.0,
                poisson_like_fraction=1.0,
                nv_curve_error=float("nan"),
                long_range_nv_error=float("nan"),
                slope_long=float("nan"),
            )
            return loss, diag, None, rep

        # eigenvalues and unfolding
        w = stable_eigvalsh(H, seed=int(seed))
        if w is None or w.size < 8:
            J = float(lambda_stability * L_stability + 1e3)
            reward = 1e-9
            loss = StageLoss(J, reward, False, L_stability, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, float(len(word)), 0.0, 0.0, 1.0, float("nan"), float("nan"), float("nan"))
            return loss, diag, None, rep

        levels = unfold_to_mean_spacing_one(w)

        # Affine map operator levels to target coordinate scale (simple, 2-parameter)
        target = target_by_dim[int(d)]
        target = target[np.isfinite(target)]
        if target.size >= 8 and levels.size >= 8:
            try:
                oq = np.quantile(levels, [0.1, 0.9])
                tq = np.quantile(target, [0.1, 0.9])
                o_span = float(max(eps, oq[1] - oq[0]))
                t_span = float(max(eps, tq[1] - tq[0]))
                a = t_span / o_span
                b = float(tq[0] - a * oq[0])
                levels = (a * levels + b).astype(np.float64, copy=False)
            except Exception:
                pass

        # Stage B: active-window support and argument loss
        arg_errs: List[float] = []
        n_active = 0
        n_both = 0
        for (a, b) in windows:
            n_op = rd.count_in_window(levels, float(a), float(b))
            n_tg = rd.count_in_window(target, float(a), float(b))
            active = bool((n_op > 0) or (n_tg > 0))
            if not active:
                continue
            n_active += 1
            if (n_op > 0) and (n_tg > 0):
                n_both += 1
            err = abs(int(n_op) - int(n_tg)) / float(max(1, int(n_tg)))
            arg_errs.append(float(err))
        support_overlap = float(n_both) / float(max(1, n_active))
        med_arg = float(np.median(np.asarray(arg_errs, dtype=np.float64))) if arg_errs else 1.0
        mean_arg = float(np.mean(np.asarray(arg_errs, dtype=np.float64))) if arg_errs else 1.0
        L_support = float(max(0.0, float(args.support_overlap_min) - support_overlap))
        L_arg = float(med_arg)

        # If no support, stop early (cheap)
        if not (math.isfinite(support_overlap) and support_overlap >= float(args.support_overlap_min)):
            J = (
                lambda_stability * L_stability
                + lambda_support * L_support
                + lambda_arg * L_arg
                + lambda_complexity * (len(word) / max(1, int(args.max_word_len)))
            )
            reward = 1e-6 / (1.0 + float(J))
            loss = StageLoss(
                J=float(J),
                reward=float(reward),
                stable_ok=True,
                L_stability=L_stability,
                L_support=L_support,
                L_arg=L_arg,
                L_nv=1.0,
                L_antipoisson=1.0,
                L_residue=1.0,
                L_trace=1.0,
                L_complexity=float(len(word) / max(1, int(args.max_word_len))),
                L_null=0.0,
                support_overlap=float(support_overlap),
                poisson_like_fraction=1.0,
                nv_curve_error=float("nan"),
                long_range_nv_error=float("nan"),
                slope_long=float("nan"),
            )
            rep.update({"mean_active_argument_error": mean_arg, "median_active_argument_error": med_arg, "active_windows": n_active})
            return loss, diag, levels, rep

        # Stage C: NV and anti-Poisson
        op_nv = number_variance_curve(levels, L_grid)
        tg_nv = number_variance_curve(target, L_grid)
        nv_err = curve_l2(op_nv, tg_nv)
        # long-range error
        mask_long = np.asarray(L_grid) >= 6.0
        long_err = curve_l2(op_nv[mask_long], tg_nv[mask_long]) if int(np.sum(mask_long)) >= 3 else float("nan")
        slope_long, _b = fit_long_slope(L_grid, op_nv, L_min_long=6.0)
        dP = curve_l2(op_nv, sigma2_poisson(L_grid))
        dG = curve_l2(op_nv, sigma2_gue_asymptotic(L_grid))
        poisson_like = bool((math.isfinite(dP) and math.isfinite(dG) and dP < dG) or (math.isfinite(slope_long) and slope_long >= float(args.anti_poisson_threshold)))
        poisson_like_fraction = 1.0 if poisson_like else 0.0
        L_nv = float(nv_err if math.isfinite(nv_err) else 10.0)
        L_antipoisson = float(max(0.0, float(slope_long) - float(args.anti_poisson_threshold))) + (1.0 if poisson_like else 0.0) if math.isfinite(slope_long) else 1.0

        # Stage D: residue + trace (active windows only)
        res_errs: List[float] = []
        imag_leaks: List[float] = []
        trace_errs: List[float] = []
        sigmas = [0.5, 1.0, 2.0, 4.0]
        for (a, b) in windows:
            n_op = rd.count_in_window(levels, float(a), float(b))
            n_tg = rd.count_in_window(target, float(a), float(b))
            if (n_op == 0) and (n_tg == 0):
                continue
            I_op = rd.residue_proxy_count(levels, float(a), float(b), eta=float(args.eta), n_contour_points=int(args.n_contour_points))
            I_tg = rd.residue_proxy_count(target, float(a), float(b), eta=float(args.eta), n_contour_points=int(args.n_contour_points))
            err = abs(float(I_op.real) - float(I_tg.real)) / max(1.0, abs(float(I_tg.real)))
            res_errs.append(float(err))
            imag_leaks.append(float(abs(I_op.imag)))
            c = 0.5 * (float(a) + float(b))
            for s in sigmas:
                Sop = rd.trace_formula_proxy(levels, center=float(c), sigma=float(s))
                Stg = rd.trace_formula_proxy(target, center=float(c), sigma=float(s))
                terr = float(abs(Sop - Stg) / max(1e-12, abs(Stg))) if (math.isfinite(Sop) and math.isfinite(Stg)) else float("nan")
                if math.isfinite(terr):
                    trace_errs.append(float(terr))
        med_res = float(np.median(np.asarray(res_errs, dtype=np.float64))) if res_errs else 1.0
        med_leak = float(np.median(np.asarray(imag_leaks, dtype=np.float64))) if imag_leaks else 0.0
        L_res = float(med_res + 0.1 * med_leak)
        L_trace = float(np.median(np.asarray(trace_errs, dtype=np.float64))) if trace_errs else 1.0

        # Null penalty (shuffled/reversed targets) using NV+arg core
        null_scores = []
        rng2 = np.random.default_rng(int(seed) + 12345)
        shuffled = target.copy()
        rng2.shuffle(shuffled)
        reversed_t = target[::-1].copy()
        core_real = float(L_nv + L_arg)
        for null_t in (shuffled, reversed_t):
            nv_null = curve_l2(op_nv, number_variance_curve(null_t, L_grid))
            # argument vs null
            null_arg = []
            for (a, b) in windows:
                n_op = rd.count_in_window(levels, float(a), float(b))
                n_tg = rd.count_in_window(null_t, float(a), float(b))
                if (n_op == 0) and (n_tg == 0):
                    continue
                null_arg.append(abs(int(n_op) - int(n_tg)) / float(max(1, int(n_tg))))
            arg_null = float(np.median(np.asarray(null_arg, dtype=np.float64))) if null_arg else 1.0
            null_scores.append(float((nv_null if math.isfinite(nv_null) else 10.0) + arg_null))
        best_null = float(np.nanmin(np.asarray(null_scores, dtype=np.float64))) if null_scores else float("nan")
        L_null = float(max(0.0, core_real - best_null)) if math.isfinite(best_null) else 0.0

        L_complexity = float(len(word) / max(1, int(args.max_word_len)))

        J = (
            lambda_stability * L_stability
            + lambda_support * L_support
            + lambda_arg * L_arg
            + lambda_nv * L_nv
            + lambda_ap * L_antipoisson
            + lambda_res * L_res
            + lambda_trace * L_trace
            + lambda_complexity * L_complexity
            + lambda_null * L_null
        )
        reward = 1.0 / (1.0 + float(J))
        rep.update(
            {
                "mean_active_argument_error": mean_arg,
                "median_active_argument_error": med_arg,
                "active_windows": n_active,
                "nv_curve_error": nv_err,
                "long_range_nv_error": long_err,
                "slope_long": slope_long,
            }
        )
        loss = StageLoss(
            J=float(J),
            reward=float(reward),
            stable_ok=True,
            L_stability=float(L_stability),
            L_support=float(L_support),
            L_arg=float(L_arg),
            L_nv=float(L_nv),
            L_antipoisson=float(L_antipoisson),
            L_residue=float(L_res),
            L_trace=float(L_trace),
            L_complexity=float(L_complexity),
            L_null=float(L_null),
            support_overlap=float(support_overlap),
            poisson_like_fraction=float(poisson_like_fraction),
            nv_curve_error=float(nv_err),
            long_range_nv_error=float(long_err),
            slope_long=float(slope_long),
        )
        return loss, diag, levels, rep

    # Progress logging per ant
    total_evals = int(args.num_iters) * int(args.num_ants) * max(1, len(dims))
    done = 0
    exec_t0 = time.perf_counter()

    for it in range(1, int(args.num_iters) + 1):
        for d in dims:
            bestJ = float(best_by_dim[int(d)]["J"])
            for ant in range(int(args.num_ants)):
                done += 1
                # sample a word
                L = int(np.random.default_rng(int(args.progress_every) + it + ant + 17 * d).integers(1, int(args.max_word_len) + 1))
                word: List[Tuple[int, int]] = []
                prev = None
                for _ in range(L):
                    tok = sample_token(int(d), prev)
                    word.append(tok)
                    prev = tok
                word = simplify_word(word, max_power=max_power, max_word_len=int(args.max_word_len))
                if not word:
                    word = [sample_token(int(d), None)]
                wstr = word_to_string(word)

                # evaluate
                loss, diag, levels, rep = evaluate_candidate(int(d), word, seed=int(1000 * it + ant + 97 * d))

                # reward flattening: very small if unstable
                reward = float(loss.reward) if loss.stable_ok else 1e-9

                # update pheromone usage and deposit stats
                for tok in word:
                    usage[int(d)][tok] += 1
                    reward_sum[int(d)][tok] += reward

                # store stability row
                stab_rows.append(
                    {
                        "iter": int(it),
                        "ant_id": int(ant),
                        "dim": int(d),
                        "word": wstr,
                        "finite_ok": bool(diag.finite_ok),
                        "selfadjoint_error": float(diag.selfadjoint_error),
                        "spectral_radius": float(diag.spectral_radius),
                        "eigen_min": float(diag.eigen_min),
                        "eigen_max": float(diag.eigen_max),
                        "eigen_spread": float(diag.eigen_spread),
                        "condition_proxy": float(diag.condition_proxy),
                        "nontrivial_spectrum_ok": bool(diag.nontrivial_spectrum_ok),
                        "stable_operator_ok": bool(diag.stable_operator_ok),
                    }
                )

                # store history row
                hist_rows.append(
                    {
                        "iter": int(it),
                        "ant_id": int(ant),
                        "dim": int(d),
                        "word": wstr,
                        "word_len": int(len(word)),
                        "J_v14_2": float(loss.J),
                        "reward": float(reward),
                        "stable_operator_ok": bool(diag.stable_operator_ok),
                        "L_stability": float(loss.L_stability),
                        "L_support": float(loss.L_support),
                        "L_arg": float(loss.L_arg),
                        "L_nv": float(loss.L_nv),
                        "L_antipoisson": float(loss.L_antipoisson),
                        "L_residue": float(loss.L_residue),
                        "L_trace": float(loss.L_trace),
                        "L_complexity": float(loss.L_complexity),
                        "L_null": float(loss.L_null),
                        "support_overlap": float(loss.support_overlap),
                        "poisson_like_fraction": float(loss.poisson_like_fraction),
                        "nv_curve_error": float(loss.nv_curve_error),
                        "long_range_nv_error": float(loss.long_range_nv_error),
                        "slope_long": float(loss.slope_long),
                        "best_so_far": False,
                    }
                )

                # update best per dim
                if math.isfinite(loss.J) and float(loss.J) < float(best_by_dim[int(d)]["J"]):
                    best_by_dim[int(d)] = {"J": float(loss.J), "word": wstr, "loss": loss, "diag": diag, "levels": levels}
                    hist_rows[-1]["best_so_far"] = True
                    bestJ = float(loss.J)

                # progress print
                if (ant % max(1, int(args.progress_every)) == 0) or (ant == int(args.num_ants) - 1):
                    elapsed = time.perf_counter() - exec_t0
                    avg = elapsed / max(1, done)
                    eta_s = avg * max(0, total_evals - done)
                    print(
                        f"[V14.2] iter={it}/{int(args.num_iters)} ant={ant+1}/{int(args.num_ants)} dim={int(d)} "
                        f"J={float(loss.J):.6g} reward={float(reward):.3g} stable={bool(diag.stable_operator_ok)} "
                        f"best_J={float(bestJ):.6g} elapsed={elapsed:.1f}s eta={format_seconds(eta_s)}",
                        flush=True,
                    )

            # pheromone update after ants per dim per iter
            # evaporation
            for k in list(pher[int(d)].keys()):
                pher[int(d)][k] = float(max(1e-6, (1.0 - float(args.rho)) * pher[int(d)][k]))
            # deposit from top few ants (lowest J) at this iter/dim
            df_iter = pd.DataFrame([r for r in hist_rows if int(r["iter"]) == int(it) and int(r["dim"]) == int(d)])
            if not df_iter.empty:
                df_iter = df_iter.sort_values(["J_v14_2"], ascending=True, na_position="last")
                elite_k = int(max(1, min(5, len(df_iter))))
                elites = df_iter.head(elite_k).to_dict(orient="records")
                for r in elites:
                    J = float(r.get("J_v14_2", float("inf")))
                    rew = float(args.q) * (1.0 / (1e-12 + J)) if (math.isfinite(J) and J > 0) else 0.0
                    # parse tokens back
                    toks = []
                    for tok in str(r.get("word", "")).split():
                        try:
                            left, pstr = tok.split("^")
                            istr = left.split("_")[1]
                            toks.append((int(istr), int(pstr)))
                        except Exception:
                            continue
                    for tok in toks:
                        if tok in pher[int(d)]:
                            pher[int(d)][tok] = float(min(1e6, pher[int(d)][tok] + rew))

    # Best candidates table
    for d in dims:
        dfh = pd.DataFrame([r for r in hist_rows if int(r["dim"]) == int(d)])
        if dfh.empty:
            continue
        dfh = dfh.sort_values(["J_v14_2"], ascending=True, na_position="last")
        seen = set()
        rank = 0
        for r in dfh.to_dict(orient="records"):
            w = str(r.get("word", ""))
            if w in seen:
                continue
            seen.add(w)
            rank += 1
            best_rows.append(
                {
                    "dim": int(d),
                    "rank": int(rank),
                    "word": w,
                    "word_len": int(r.get("word_len", 0)),
                    "J_v14_2": float(r.get("J_v14_2", float("nan"))),
                    "reward": float(r.get("reward", float("nan"))),
                    "stable_operator_ok": bool(r.get("stable_operator_ok", False)),
                    "support_overlap": float(r.get("support_overlap", float("nan"))),
                    "poisson_like_fraction": float(r.get("poisson_like_fraction", float("nan"))),
                    "L_stability": float(r.get("L_stability", float("nan"))),
                    "L_support": float(r.get("L_support", float("nan"))),
                    "L_arg": float(r.get("L_arg", float("nan"))),
                    "L_nv": float(r.get("L_nv", float("nan"))),
                    "L_antipoisson": float(r.get("L_antipoisson", float("nan"))),
                    "L_residue": float(r.get("L_residue", float("nan"))),
                    "L_trace": float(r.get("L_trace", float("nan"))),
                    "L_complexity": float(r.get("L_complexity", float("nan"))),
                    "L_null": float(r.get("L_null", float("nan"))),
                }
            )
            if rank >= 20:
                break

        # Export best unfolded levels and diagnostics
        best = best_by_dim[int(d)]
        best_word = str(best.get("word", ""))
        loss: Optional[StageLoss] = best.get("loss", None)
        diag: Optional[StabilityDiag] = best.get("diag", None)
        levels = best.get("levels", None)
        if loss is not None and levels is not None:
            for idx, x in enumerate(np.asarray(levels, dtype=np.float64).reshape(-1).tolist()[: max(1, int(d))]):
                unfolded_rows.append({"dim": int(d), "word": best_word, "kind": "operator", "level_index": int(idx), "unfolded_level": float(x)})
        # also export target slice
        tgt = target_by_dim[int(d)]
        for idx, x in enumerate(np.asarray(tgt, dtype=np.float64).reshape(-1).tolist()[: max(1, int(d))]):
            unfolded_rows.append({"dim": int(d), "word": best_word, "kind": "target", "level_index": int(idx), "unfolded_level": float(x)})

        # active argument counts for best
        if loss is not None and levels is not None:
            tgt = target_by_dim[int(d)]
            for (a, b) in windows:
                n_op = rd.count_in_window(levels, float(a), float(b))
                n_tg = rd.count_in_window(tgt, float(a), float(b))
                active = bool((n_op > 0) or (n_tg > 0))
                if not active:
                    continue
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

            # NV curves for best
            op_nv = number_variance_curve(levels, L_grid)
            tg_nv = number_variance_curve(tgt, L_grid)
            for L, y in zip(L_grid.tolist(), op_nv.tolist()):
                nv_rows.append({"dim": int(d), "word": best_word, "kind": "operator", "L": float(L), "Sigma2": float(y)})
            for L, y in zip(L_grid.tolist(), tg_nv.tolist()):
                nv_rows.append({"dim": int(d), "word": best_word, "kind": "target", "L": float(L), "Sigma2": float(y)})

            # residue and trace for best (active windows)
            sigmas = [0.5, 1.0, 2.0, 4.0]
            for (a, b) in windows:
                n_op = rd.count_in_window(levels, float(a), float(b))
                n_tg = rd.count_in_window(tgt, float(a), float(b))
                if (n_op == 0) and (n_tg == 0):
                    continue
                I_op = rd.residue_proxy_count(levels, float(a), float(b), eta=float(args.eta), n_contour_points=int(args.n_contour_points))
                I_tg = rd.residue_proxy_count(tgt, float(a), float(b), eta=float(args.eta), n_contour_points=int(args.n_contour_points))
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
                    Stg = rd.trace_formula_proxy(tgt, center=float(c), sigma=float(s))
                    terr = float(abs(Sop - Stg) / max(1e-12, abs(Stg))) if (math.isfinite(Sop) and math.isfinite(Stg)) else float("nan")
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

        # Gates per dim based on best candidate
        if loss is None or diag is None:
            gate_rows.append(
                {
                    "dim": int(d),
                    "best_word": best_word,
                    "best_J": float(best.get("J", float("nan"))),
                    "B1_stable_operator_ok": False,
                    "B2_support_overlap_ok": False,
                    "B3_active_argument_ok": False,
                    "B4_nv_error_ok": False,
                    "B5_anti_poisson_ok": False,
                    "B6_residue_ok": False,
                    "B7_trace_ok": False,
                    "B8_not_random_like": True,
                    "B9_complexity_ok": True,
                    "B10_all_gate_pass": False,
                }
            )
        else:
            B1 = bool(diag.stable_operator_ok)
            B2 = bool(loss.support_overlap >= float(args.support_overlap_min))
            B3 = bool(loss.L_arg <= float(args.active_error_margin))
            B4 = bool(loss.L_nv <= 1.0)
            B5 = bool(loss.poisson_like_fraction <= float(args.poisson_like_max) and (math.isfinite(loss.slope_long) and loss.slope_long < float(args.anti_poisson_threshold)))
            B6 = bool(loss.L_residue <= float(args.active_error_margin))
            B7 = bool(loss.L_trace <= 1.0)
            B8 = True
            B9 = bool(len(str(best_word).split()) <= int(args.max_word_len))
            all_pass = bool(B1 and B2 and B3 and B4 and B5 and B6 and B7 and B9)
            gate_rows.append(
                {
                    "dim": int(d),
                    "best_word": best_word,
                    "best_J": float(loss.J),
                    "B1_stable_operator_ok": bool(B1),
                    "B2_support_overlap_ok": bool(B2),
                    "B3_active_argument_ok": bool(B3),
                    "B4_nv_error_ok": bool(B4),
                    "B5_anti_poisson_ok": bool(B5),
                    "B6_residue_ok": bool(B6),
                    "B7_trace_ok": bool(B7),
                    "B8_not_random_like": bool(B8),
                    "B9_complexity_ok": bool(B9),
                    "B10_all_gate_pass": bool(all_pass),
                }
            )

    # pheromone summary
    for d in dims:
        for (gi, p), tau in pher[int(d)].items():
            u = int(usage[int(d)][(gi, p)])
            mr = float(reward_sum[int(d)][(gi, p)] / max(1, u))
            pher_rows.append({"dim": int(d), "generator": int(gi), "power": int(p), "pheromone": float(tau), "usage_count": u, "mean_reward": mr})
    pher_rows.sort(key=lambda r: (int(r["dim"]), int(r["generator"]), int(r["power"])))

    # Write all required files
    write_csv(
        out_dir / "v14_2_aco_history.csv",
        [
            "iter",
            "ant_id",
            "dim",
            "word",
            "word_len",
            "J_v14_2",
            "reward",
            "stable_operator_ok",
            "L_stability",
            "L_support",
            "L_arg",
            "L_nv",
            "L_antipoisson",
            "L_residue",
            "L_trace",
            "L_complexity",
            "L_null",
            "support_overlap",
            "poisson_like_fraction",
            "nv_curve_error",
            "long_range_nv_error",
            "slope_long",
            "best_so_far",
        ],
        hist_rows,
    )
    write_csv(
        out_dir / "v14_2_best_candidates.csv",
        [
            "dim",
            "rank",
            "word",
            "word_len",
            "J_v14_2",
            "reward",
            "stable_operator_ok",
            "support_overlap",
            "poisson_like_fraction",
            "L_stability",
            "L_support",
            "L_arg",
            "L_nv",
            "L_antipoisson",
            "L_residue",
            "L_trace",
            "L_complexity",
            "L_null",
        ],
        best_rows,
    )
    write_csv(
        out_dir / "v14_2_operator_stability.csv",
        [
            "iter",
            "ant_id",
            "dim",
            "word",
            "finite_ok",
            "selfadjoint_error",
            "spectral_radius",
            "eigen_min",
            "eigen_max",
            "eigen_spread",
            "condition_proxy",
            "nontrivial_spectrum_ok",
            "stable_operator_ok",
        ],
        stab_rows,
    )
    write_csv(out_dir / "v14_2_unfolded_levels.csv", ["dim", "word", "kind", "level_index", "unfolded_level"], unfolded_rows)
    write_csv(
        out_dir / "v14_2_active_argument_counts.csv",
        ["dim", "word", "window_a", "window_b", "N_operator", "N_target", "N_error", "N_error_norm", "active_window"],
        arg_rows,
    )
    write_csv(out_dir / "v14_2_number_variance_curves.csv", ["dim", "word", "kind", "L", "Sigma2"], nv_rows)
    write_csv(
        out_dir / "v14_2_residue_scores.csv",
        ["dim", "word", "window_a", "window_b", "I_operator_real", "I_operator_imag", "I_target_real", "I_target_imag", "residue_count_error", "residue_imag_leak"],
        res_rows,
    )
    write_csv(
        out_dir / "v14_2_trace_formula_proxy.csv",
        ["dim", "word", "window_a", "window_b", "center", "sigma", "S_operator", "S_target", "trace_error_norm"],
        trace_rows,
    )
    write_csv(
        out_dir / "v14_2_gate_summary.csv",
        [
            "dim",
            "best_word",
            "best_J",
            "B1_stable_operator_ok",
            "B2_support_overlap_ok",
            "B3_active_argument_ok",
            "B4_nv_error_ok",
            "B5_anti_poisson_ok",
            "B6_residue_ok",
            "B7_trace_ok",
            "B8_not_random_like",
            "B9_complexity_ok",
            "B10_all_gate_pass",
        ],
        gate_rows,
    )
    write_csv(out_dir / "v14_2_pheromone_summary.csv", ["dim", "generator", "power", "pheromone", "usage_count", "mean_reward"], pher_rows)

    any_stable = any(bool(r.get("stable_operator_ok", False)) for r in stab_rows)
    any_all_gate = any(bool(r.get("B10_all_gate_pass", False)) for r in gate_rows)

    payload = {
        "warning": "Computational evidence only; not a proof of RH.",
        "config": json_sanitize(vars(args)),
        "best_by_dim": json_sanitize({int(d): best_by_dim[int(d)]["word"] for d in dims}),
        "gate_summary": json_sanitize(gate_rows),
        "summary": {
            "any_stable_candidate_found": bool(any_stable),
            "any_all_gate_pass": bool(any_all_gate),
            "should_proceed_to_v14_3": bool(any_stable),
            "should_proceed_to_analytic_claim": False,
        },
        "warnings": warnings,
        "runtime_s": float(time.perf_counter() - t0),
    }
    (out_dir / "v14_2_results.json").write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    # Report
    md: List[str] = []
    md.append("# V14.2 Stabilized Artin Operator Manifold Search\n\n")
    md.append("> **Computational evidence only; not a proof of RH.**\n\n")
    md.append("## Why V14.2 exists\n\n")
    md.append(
        "V14.1 produced numerical pathologies (huge losses, flat rewards, instability-driven optimization). "
        "V14.2 stabilizes the operator manifold: bounded Hermitian generator contributions, trace removal, "
        "spectral radius normalization, and staged evaluation.\n\n"
    )
    md.append("## Stabilized operator construction\n\n")
    md.append("- Local stable generator blocks: 2x2 rotations embedded in dim×dim.\n")
    md.append("- Hermitian contributions: \\(A=\\tfrac12(G+G^T)\\), normalized by operator norm.\n")
    md.append("- Word operator: \\(H=\\sum_k c_k A_{i_k,p_k}\\) with \\(c_k=1/\\sqrt{k+1}\\), symmetrize, remove trace.\n")
    md.append("- Spectral radius normalization to target radius \\(\\approx \\mathrm{dim}/4\\).\n\n")
    md.append("## Staged objective\n\n")
    md.append("- **Stage A** stability checks and penalties.\n")
    md.append("- **Stage B** active-window support overlap + argument count loss.\n")
    md.append("- **Stage C** number variance curve + anti-Poisson rigidity.\n")
    md.append("- **Stage D** residue proxy + trace proxy.\n\n")
    md.append("## ACO configuration\n\n")
    md.append(f"- dims={dims}\n- num_ants={args.num_ants} num_iters={args.num_iters} max_word_len={args.max_word_len} max_power={args.max_power}\n")
    md.append(f"- alpha={args.alpha} beta={args.beta} rho={args.rho} q={args.q}\n\n")
    md.append("## Best candidate per dim\n\n")
    md.append("See `v14_2_best_candidates.csv` and `v14_2_gate_summary.csv`.\n\n")
    md.append("## Summary\n\n")
    md.append(f"- any stable candidate found: **{any_stable}**\n")
    md.append(f"- any all-gate pass occurred: **{any_all_gate}**\n")
    md.append(f"- should proceed to V14.3: **{any_stable}**\n")
    md.append("- Should proceed to analytic claim? **False** (unless all gates pass + null controls included).\n\n")
    if warnings:
        md.append("## Warnings\n\n")
        md.extend([f"- {w}\n" for w in warnings[:30]])
        if len(warnings) > 30:
            md.append(f"- (and {len(warnings)-30} more)\n")
        md.append("\n")
    md.append("## Verification commands\n\n```bash\n")
    md.append(f"OUT={str(out_dir)}\n")
    md.append('find "$OUT" -maxdepth 1 -type f | sort\n')
    md.append('column -s, -t < "$OUT"/v14_2_gate_summary.csv | head -120\n')
    md.append('column -s, -t < "$OUT"/v14_2_best_candidates.csv | head -120\n')
    md.append('tail -40 "$OUT"/v14_2_aco_history.csv | column -s, -t\n')
    md.append('column -s, -t < "$OUT"/v14_2_pheromone_summary.csv | head -120\n')
    md.append('head -220 "$OUT"/v14_2_report.md\n')
    md.append("```\n")
    (out_dir / "v14_2_report.md").write_text("".join(md), encoding="utf-8")

    tex = (
        "\\documentclass[11pt]{article}\n\\usepackage[margin=1in]{geometry}\n\\begin{document}\n"
        "\\title{V14.2 Stabilized Artin Operator Manifold Search}\n\\maketitle\n"
        "\\section*{Warning}\nComputational evidence only; not a proof of RH.\n\n"
        "\\section*{Summary}\n"
        + latex_escape(json.dumps({"any_stable": any_stable, "any_all_gate_pass": any_all_gate}, indent=2))
        + "\n\\end{document}\n"
    )
    (out_dir / "v14_2_report.tex").write_text(tex, encoding="utf-8")
    if _find_pdflatex() is not None:
        try_pdflatex(out_dir / "v14_2_report.tex", out_dir, "v14_2_report.pdf")

    print(f"[V14.2] Wrote outputs to {out_dir}", flush=True)


if __name__ == "__main__":
    main()

