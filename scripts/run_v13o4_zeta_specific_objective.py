#!/usr/bin/env python3
"""
V13O.4 / V13P0: zeta-specific objective before analytic renormalization.

Adds zeta-specific long-range metrics beyond spectral_log_mse, spacing_mse_normalized, ks_wigner:
- pair_corr_error
- number_variance_error
- staircase_residual_error
- transfer_gap (train/test) + transfer_gap_penalty
- ensemble_margin_penalty

Computational evidence only; not a proof of the Riemann Hypothesis.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

# Avoid BLAS oversubscription in parallel runs (macOS/M2 friendly).
_THREAD_ENV_DEFAULTS = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
for _k, _v in _THREAD_ENV_DEFAULTS.items():
    os.environ.setdefault(_k, _v)

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

PRIMARY_CANDIDATE_ID = "seed_6"

_DTF = np.float64
ALPHA = 0.5
EPS_BUILDER = 0.6
GEO_WEIGHT = 10.0
CLIP_DEFAULT = (0.5, 99.5)
DIAG_SHIFT = 1e-6
ABS_CAP = 5.0

# Primary braid fallback (validated against candidate_operators.json seed_6 in V13O.3)
PRIMARY_WORD_FALLBACK = [-4, -2, -4, -2, -2, -1, -1]
REJECTED_WORD_FALLBACK = [6, 4, -1, 1, 1, 1, 4]

DIM_K_TRAIN = {64: 45, 128: 96, 256: 128}

DIM_PARAM: Dict[int, Dict[str, Any]] = {
    64: {
        "lambda_p": 2.121320343559643,
        "geo_sigma": 0.5045378491522287,
        "smooth_sigma": 0.7071067811865476,
        "beta0": 0.3,
        "tau": 200.0,
        "beta_floor": 0.03,
    },
    128: {
        "lambda_p": 3.0,
        "geo_sigma": 0.6,
        "smooth_sigma": 1.0,
        "beta0": 0.3,
        "tau": 500.0,
        "beta_floor": 0.03,
    },
    256: {
        "lambda_p": 2.121320343559643,
        "geo_sigma": 0.5045378491522287,
        "smooth_sigma": 1.25,
        "beta0": 0.3,
        "tau": 300.0,
        "beta_floor": 0.03,
    },
}

V_MODES_ALL = [
    "full_V",
    "frozen_V_after_5",
    "frozen_V_after_10",
    "weak_V",
    "very_weak_V",
    "target_blind_V",
    "density_only_V",
    "phase_only_V",
    "word_only_V",
]

TARGET_GROUPS_ALL = [
    "real_zeta",
    "shuffled_zeta",
    "reversed_zeta",
    "block_shuffled_zeta_block4",
    "block_shuffled_zeta_block8",
    "local_jitter_zeta_small",
    "local_jitter_zeta_medium",
    "density_matched_synthetic",
    "GUE_synthetic",
    "Poisson_synthetic",
]

CONTROL_TARGETS_FOR_MARGIN = [
    "GUE_synthetic",
    "Poisson_synthetic",
    "shuffled_zeta",
    "block_shuffled_zeta_block4",
    "block_shuffled_zeta_block8",
    "local_jitter_zeta_small",
    "local_jitter_zeta_medium",
    "density_matched_synthetic",
    "reversed_zeta",
]

DEFAULT_WEIGHTS = {
    "w_spectral": 0.25,
    "w_spacing": 0.25,
    "w_ks": 0.25,
    "w_pair": 1.0,
    "w_number_var": 1.0,
    "w_staircase": 0.5,
    "w_transfer": 0.5,
    "w_ensemble": 1.0,
}

# Default thresholds / acceptance checks
OPERATOR_DIFF_MAX = 1e-3
SELF_ADJOINTNESS_FRO_MAX = 1e-12
TRANSFER_GAP_SPECTRAL_MAX = 2.0
TRANSFER_GAP_PAIR_MAX = 1.0
TRANSFER_GAP_NUMBER_VAR_MAX = 1.0

PAIR_CORR_ERROR_MARGIN = 0.0
NUMBER_VAR_ERROR_MARGIN = 0.0
STAIRCASE_ERROR_MARGIN = 0.0


def _tqdm(iterable: Iterable[Any], total: Optional[int] = None, desc: str = "") -> Iterable[Any]:
    try:
        from tqdm import tqdm  # type: ignore

        return tqdm(iterable, total=total, desc=desc)
    except Exception:
        return iterable


def format_seconds(sec: float) -> str:
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _resolve(p: str) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = Path(ROOT) / path
    return path


def _require_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        print(f"[v13o4] ERROR: required input missing: {path}", flush=True)
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def json_sanitize(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, float):
        if math.isnan(obj) or not math.isfinite(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if math.isnan(v) or not math.isfinite(v):
            return None
        return v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return json_sanitize(obj.tolist())
    if isinstance(obj, dict):
        return {str(k): json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_sanitize(x) for x in obj]
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, str):
        return obj
    if isinstance(obj, (int,)):
        return int(obj)
    try:
        if isinstance(obj, Path):
            return str(obj)
    except Exception:
        pass
    return obj


def csv_cell(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, float) and (math.isnan(v) or not math.isfinite(v)):
        return ""
    if isinstance(v, (np.floating,)):
        x = float(v)
        if math.isnan(x) or not math.isfinite(x):
            return ""
        return x
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def load_v13_validate() -> Any:
    import importlib.util

    path = Path(ROOT) / "scripts" / "validate_candidates_v13.py"
    spec = importlib.util.spec_from_file_location("_v13_validate_o4", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_v13_validate_o4"] = mod
    spec.loader.exec_module(mod)
    return mod


def build_h(
    *,
    z_points: np.ndarray,
    word: List[int],
    v13l: Any,
    geo_sigma: float,
    laplacian_weight: float,
    geo_weight: float,
) -> np.ndarray:
    geodesics = [v13l.geodesic_entry_for_word([int(x) for x in word])]
    H_base, _ = v13l.build_h_base_no_potential(
        z_points=z_points,
        geodesics=geodesics,
        eps=float(EPS_BUILDER),
        geo_sigma=float(geo_sigma),
        geo_weight=float(geo_weight),
        laplacian_weight=float(laplacian_weight),
        distances=None,
        diag_shift=float(DIAG_SHIFT),
    )
    return np.asarray(H_base, dtype=_DTF, copy=True)


def gue_ord(n: int, rng: np.random.Generator) -> np.ndarray:
    a = rng.standard_normal((n, n))
    h = (a + a.T) / math.sqrt(2.0)
    w = np.sort(np.linalg.eigvalsh(h).astype(np.float64))
    w = w - float(w[0]) + 1e-6
    return w.astype(_DTF, copy=False)


def poisson_ord(n: int, rng: np.random.Generator) -> np.ndarray:
    s = rng.exponential(1.0, size=(n,)).astype(np.float64)
    c = np.cumsum(s)
    c = c - float(c[0]) + 1e-6
    return c.astype(_DTF, copy=False)


def full_shuffle_zeta(z_sorted: np.ndarray, k_train: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    need = 2 * int(k_train)
    z = np.sort(np.asarray(z_sorted[:need], dtype=np.float64).reshape(-1))
    gaps = np.maximum(np.diff(z), 1e-9)
    rng = np.random.default_rng(int(seed))
    order = np.arange(gaps.size)
    rng.shuffle(order)
    g2 = gaps[order]
    recon = np.zeros(need, dtype=np.float64)
    recon[0] = float(z[0])
    recon[1:] = z[0] + np.cumsum(g2)
    return recon[:k_train].astype(_DTF, copy=False), recon[k_train:need].astype(_DTF, copy=False)


def block_shuffle_zeta(z_sorted: np.ndarray, k_train: int, block: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    need = 2 * int(k_train)
    z = np.sort(np.asarray(z_sorted[:need], dtype=np.float64).reshape(-1))
    gaps = np.diff(z)
    B = int(block)
    n_blocks = max(1, len(gaps) // B)
    blocks = [gaps[i * B : (i + 1) * B].copy() for i in range(n_blocks)]
    rest = gaps[n_blocks * B :].copy() if n_blocks * B < len(gaps) else np.zeros((0,), dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    order = np.arange(len(blocks))
    rng.shuffle(order)
    new_gaps = np.concatenate([blocks[int(i)] for i in order] + ([rest] if rest.size else []))
    recon = np.zeros(need, dtype=np.float64)
    recon[0] = float(z[0])
    for i in range(1, need):
        gidx = i - 1
        recon[i] = recon[i - 1] + (float(new_gaps[gidx]) if gidx < new_gaps.size else 1e-3)
    return recon[:k_train].astype(_DTF, copy=False), recon[k_train:need].astype(_DTF, copy=False)


def local_jitter_zeta(z_sorted: np.ndarray, k_train: int, seed: int, amp_factor: float) -> Tuple[np.ndarray, np.ndarray]:
    need = 2 * int(k_train)
    z = np.sort(np.asarray(z_sorted[:need], dtype=np.float64).reshape(-1))
    gaps = np.diff(z)
    med = float(np.median(gaps)) if gaps.size else 1.0
    rng = np.random.default_rng(int(seed))
    noise = rng.normal(0.0, float(amp_factor) * med, size=gaps.shape)
    gaps2 = np.maximum(gaps + noise, 1e-9)
    gaps2 = gaps2 * (float(np.mean(gaps)) / max(float(np.mean(gaps2)), 1e-12))
    recon = np.zeros(need, dtype=np.float64)
    recon[0] = float(z[0])
    recon[1:] = z[0] + np.cumsum(gaps2)
    return recon[:k_train].astype(_DTF, copy=False), recon[k_train:need].astype(_DTF, copy=False)


def density_matched_synthetic(z_sorted: np.ndarray, k_train: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    need = 2 * int(k_train)
    z = np.sort(np.asarray(z_sorted[:need], dtype=np.float64).reshape(-1))
    gaps = np.maximum(np.diff(z), 1e-9)
    inv = 1.0 / gaps
    env = np.convolve(inv, np.ones(5) / 5.0, mode="same")
    rng = np.random.default_rng(int(seed))
    g2 = rng.exponential(1.0, size=gaps.shape).astype(np.float64)
    g2 = g2 * (env / (np.mean(g2) / max(np.mean(gaps), 1e-12) + 1e-12))
    g2 = g2 * (float(np.sum(gaps)) / max(float(np.sum(g2)), 1e-12))
    recon = np.zeros(need, dtype=np.float64)
    recon[0] = float(z[0])
    recon[1:] = z[0] + np.cumsum(g2)
    return recon[:k_train].astype(_DTF, copy=False), recon[k_train:need].astype(_DTF, copy=False)


def build_train_test_targets(
    name: str,
    z_sorted: np.ndarray,
    k_train: int,
    seed: int,
    target_rep: int,
) -> Tuple[np.ndarray, np.ndarray, bool, str]:
    """Return (train_ordinates, test_ordinates, train_ordered, window_note)."""
    need = 2 * int(k_train)
    base = np.sort(np.asarray(z_sorted[:need], dtype=_DTF).reshape(-1))
    sp = int(target_rep)
    if name == "real_zeta":
        return base[:k_train].copy(), base[k_train:need].copy(), False, "sorted_zeta"
    if name == "reversed_zeta":
        rev = base[::-1].copy()
        return rev[:k_train].copy(), rev[k_train:need].copy(), True, "reversed_prefix"
    if name == "block_shuffled_zeta_block4":
        tr, te = block_shuffle_zeta(z_sorted, k_train, 4, seed + 11 * sp)
        return tr, te, False, "block_shuffle_4"
    if name == "block_shuffled_zeta_block8":
        tr, te = block_shuffle_zeta(z_sorted, k_train, 8, seed + 13 * sp + 1)
        return tr, te, False, "block_shuffle_8"
    if name == "local_jitter_zeta_small":
        tr, te = local_jitter_zeta(z_sorted, k_train, seed + 17 * sp + 2, 0.05)
        return tr, te, False, "jitter_small"
    if name == "local_jitter_zeta_medium":
        tr, te = local_jitter_zeta(z_sorted, k_train, seed + 19 * sp + 3, 0.15)
        return tr, te, False, "jitter_medium"
    if name == "density_matched_synthetic":
        tr, te = density_matched_synthetic(z_sorted, k_train, seed + 23 * sp + 4)
        return tr, te, False, "density_matched"
    if name == "GUE_synthetic":
        g = gue_ord(need, np.random.default_rng(seed + 29 * sp + 5))
        return g[:k_train].copy(), g[k_train:need].copy(), False, "GUE"
    if name == "Poisson_synthetic":
        p = poisson_ord(need, np.random.default_rng(seed + 31 * sp + 6))
        return p[:k_train].copy(), p[k_train:need].copy(), False, "Poisson"
    if name == "shuffled_zeta":
        tr, te = full_shuffle_zeta(z_sorted, k_train, seed + 37 * sp + 7)
        return tr, te, False, "gap_shuffle_full"
    raise ValueError(name)


def target_has_replicas(name: str) -> bool:
    # real_zeta and reversed_zeta are deterministic given z_sorted slice
    return name not in ("real_zeta", "reversed_zeta")


def count_target_cells(n_controls: int) -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    for nm in TARGET_GROUPS_ALL:
        if target_has_replicas(nm):
            for c in range(int(n_controls)):
                out.append((nm, c))
        else:
            out.append((nm, 0))
    return out


def _safe_mean_spacing(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=np.float64).reshape(-1))
    if x.size < 3:
        return 1.0
    g = np.diff(x)
    g = g[np.isfinite(g) & (g > 0.0)]
    if g.size == 0:
        return 1.0
    m = float(np.mean(g))
    return m if (math.isfinite(m) and m > 0.0) else 1.0


def unfold_to_mean_spacing_one(x: np.ndarray) -> np.ndarray:
    """Unfold by global mean spacing: u = (x - x0)/mean_diff."""
    x = np.sort(np.asarray(x, dtype=np.float64).reshape(-1))
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.zeros((1,), dtype=np.float64)
    ms = _safe_mean_spacing(x)
    u = (x - float(x[0])) / max(ms, 1e-12)
    return u.astype(np.float64, copy=False)


def pair_corr_hist(
    unfolded: np.ndarray,
    *,
    r_max: float,
    bins: int,
) -> np.ndarray:
    """Histogram of |u_i - u_j| for local pairs, normalized to sum 1."""
    u = np.sort(np.asarray(unfolded, dtype=np.float64).reshape(-1))
    u = u[np.isfinite(u)]
    n = int(u.size)
    if n < 6:
        return np.zeros((bins,), dtype=np.float64)

    r_max_f = float(max(r_max, 1e-6))
    max_lag = int(max(2, min(n - 1, int(math.ceil(r_max_f)) + 6)))

    diffs: List[float] = []
    for i in range(n):
        ui = float(u[i])
        j_hi = min(n, i + 1 + max_lag)
        for j in range(i + 1, j_hi):
            d = abs(float(u[j]) - ui)
            if d <= r_max_f:
                diffs.append(d)
            else:
                break  # u sorted: further j only increases d

    if not diffs:
        return np.zeros((bins,), dtype=np.float64)
    d_arr = np.asarray(diffs, dtype=np.float64)
    h, _ = np.histogram(d_arr, bins=int(bins), range=(0.0, r_max_f))
    s = float(np.sum(h)) + 1e-12
    return (h.astype(np.float64) / s).astype(np.float64, copy=False)


def pair_corr_error(eig: np.ndarray, z: np.ndarray, k: int, *, r_max: float = 10.0, bins: int = 64) -> float:
    try:
        kk = int(k)
        e = np.sort(np.asarray(eig, dtype=np.float64).reshape(-1))[:kk]
        zz = np.sort(np.asarray(z, dtype=np.float64).reshape(-1))[:kk]
        if e.size < 6 or zz.size < 6:
            return 1.0e6
        ue = unfold_to_mean_spacing_one(e)
        uz = unfold_to_mean_spacing_one(zz)
        he = pair_corr_hist(ue, r_max=float(r_max), bins=int(bins))
        hz = pair_corr_hist(uz, r_max=float(r_max), bins=int(bins))
        return float(np.linalg.norm(he - hz, ord=1))
    except Exception:
        return 1.0e6


def number_variance_curve(unfolded: np.ndarray, Ls: Sequence[float]) -> Dict[float, float]:
    """Compute Σ²(L) over a grid of window start positions using searchsorted counts."""
    u = np.sort(np.asarray(unfolded, dtype=np.float64).reshape(-1))
    u = u[np.isfinite(u)]
    if u.size < 6:
        return {float(L): float("nan") for L in Ls}

    lo = float(u[0])
    hi = float(u[-1])
    out: Dict[float, float] = {}
    for L in Ls:
        Lf = float(L)
        if not (math.isfinite(Lf) and Lf > 0.0):
            out[Lf] = float("nan")
            continue
        span = hi - lo
        if span <= Lf + 1e-9:
            out[Lf] = float("nan")
            continue
        # Fixed grid of start positions (deterministic) for robustness across small n.
        n_starts = int(min(128, max(16, int(4 * span))))
        starts = np.linspace(lo, hi - Lf, num=n_starts)
        counts = np.empty((starts.size,), dtype=np.float64)
        for i, s in enumerate(starts):
            a = float(s)
            b = float(s) + Lf
            left = int(np.searchsorted(u, a, side="left"))
            right = int(np.searchsorted(u, b, side="left"))
            counts[i] = float(max(0, right - left))
        if counts.size < 2:
            out[Lf] = float("nan")
            continue
        out[Lf] = float(np.var(counts))
    return out


def number_variance_error(eig: np.ndarray, z: np.ndarray, k: int, *, Ls: Sequence[float] = (2, 4, 8, 16)) -> float:
    try:
        kk = int(k)
        e = np.sort(np.asarray(eig, dtype=np.float64).reshape(-1))[:kk]
        zz = np.sort(np.asarray(z, dtype=np.float64).reshape(-1))[:kk]
        if e.size < 6 or zz.size < 6:
            return 1.0e6
        ue = unfold_to_mean_spacing_one(e)
        uz = unfold_to_mean_spacing_one(zz)
        ce = number_variance_curve(ue, Ls)
        cz = number_variance_curve(uz, Ls)
        errs: List[float] = []
        for L in Ls:
            ve = float(ce.get(float(L), float("nan")))
            vz = float(cz.get(float(L), float("nan")))
            if not (math.isfinite(ve) and math.isfinite(vz)):
                continue
            denom = (vz * vz) + 1e-9
            errs.append(((ve - vz) ** 2) / denom)
        if not errs:
            return 1.0e6
        return float(np.mean(np.asarray(errs, dtype=np.float64)))
    except Exception:
        return 1.0e6


def _affine_align_to_window(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """Find a,b mapping x->a*x+b matching endpoints of y."""
    xs = np.sort(np.asarray(x, dtype=np.float64).reshape(-1))
    ys = np.sort(np.asarray(y, dtype=np.float64).reshape(-1))
    if xs.size < 2 or ys.size < 2:
        return 1.0, 0.0
    x0, x1 = float(xs[0]), float(xs[-1])
    y0, y1 = float(ys[0]), float(ys[-1])
    dx = x1 - x0
    if not (math.isfinite(dx) and abs(dx) > 1e-12):
        return 1.0, (y0 - x0)
    a = (y1 - y0) / dx
    b = y0 - a * x0
    if not (math.isfinite(a) and math.isfinite(b)):
        return 1.0, 0.0
    return float(a), float(b)


def staircase_residual_error(eig: np.ndarray, z: np.ndarray, k: int, *, grid_n: int = 256) -> float:
    """
    Compare staircase residuals after affine alignment and detrending.
    We compute d(t)=N_e(t)-N_z(t) on a grid, remove best linear fit in t, report MSE(resid).
    """
    try:
        kk = int(k)
        e0 = np.sort(np.asarray(eig, dtype=np.float64).reshape(-1))[:kk]
        z0 = np.sort(np.asarray(z, dtype=np.float64).reshape(-1))[:kk]
        if e0.size < 6 or z0.size < 6:
            return 1.0e6
        a, b = _affine_align_to_window(e0, z0)
        e = (a * e0 + b).astype(np.float64, copy=False)
        e = np.sort(e)
        z = np.sort(z0)
        lo = float(min(e[0], z[0]))
        hi = float(max(e[-1], z[-1]))
        if not (math.isfinite(lo) and math.isfinite(hi) and hi > lo):
            return 1.0e6
        tgrid = np.linspace(lo, hi, num=int(grid_n))
        Ne = np.searchsorted(e, tgrid, side="right").astype(np.float64)
        Nz = np.searchsorted(z, tgrid, side="right").astype(np.float64)
        d = (Ne - Nz).astype(np.float64)
        # remove linear trend
        coef = np.polyfit(tgrid, d, 1)
        pred = np.polyval(coef, tgrid)
        resid = d - pred
        mse = float(np.mean(resid * resid))
        return mse if (math.isfinite(mse)) else 1.0e6
    except Exception:
        return 1.0e6


def transfer_gap_penalty(
    *,
    gap_spectral: float,
    gap_pair: float,
    gap_number_var: float,
    max_gap_spectral: float = TRANSFER_GAP_SPECTRAL_MAX,
    max_gap_pair: float = TRANSFER_GAP_PAIR_MAX,
    max_gap_number_var: float = TRANSFER_GAP_NUMBER_VAR_MAX,
) -> float:
    def pos(x: float) -> float:
        return float(x) if (math.isfinite(float(x)) and float(x) > 0.0) else 0.0

    p = 0.0
    p += pos(float(gap_spectral) - float(max_gap_spectral)) / max(float(max_gap_spectral), 1e-9)
    p += pos(float(gap_pair) - float(max_gap_pair)) / max(float(max_gap_pair), 1e-9)
    p += pos(float(gap_number_var) - float(max_gap_number_var)) / max(float(max_gap_number_var), 1e-9)
    return float(p)


def meets_basic_accept(
    *,
    finite: bool,
    sa: float,
    od: float,
    eig_err: Optional[str],
) -> bool:
    if eig_err:
        return False
    return bool(
        finite
        and math.isfinite(sa)
        and sa <= SELF_ADJOINTNESS_FRO_MAX
        and math.isfinite(od)
        and od <= OPERATOR_DIFF_MAX
    )


def stable_control_id(word_group_id: str, target_group: str, target_rep: int, wj: Dict[str, Any]) -> int:
    stable_cid = int(target_rep if target_has_replicas(target_group) else 0)
    if word_group_id == "random_words_n30":
        stable_cid = int(wj.get("rw_index", 0)) * 1000 + stable_cid
    elif word_group_id == "random_symmetric_baseline":
        stable_cid = int(wj.get("sym_index", 0)) * 1000 + stable_cid
    return stable_cid


def _stable_hash32(s: str) -> int:
    """Deterministic small hash (avoid Python's salted hash())."""
    return int.from_bytes(hashlib.sha256(str(s).encode("utf-8")).digest()[:4], "little", signed=False)


def config_seed(base_seed: int, dim: int, config_index: int) -> int:
    """Required deterministic seed per config."""
    return int(base_seed) + 100000 * int(dim) + int(config_index)


def target_seed_from_config(cfg_seed: int, vm: str, tg: str, wji: int, target_rep: int) -> int:
    """Deterministic per-window seed derived from config_seed."""
    rep_for_seed = int(target_rep if target_has_replicas(tg) else 0)
    return int(cfg_seed) + (_stable_hash32(vm) % 100000) + (_stable_hash32(tg) % 100000) + 17 * int(wji) + 131 * rep_for_seed


def _serializable_job_config(wj: Dict[str, Any]) -> Dict[str, Any]:
    d = dict(wj)
    d.pop("matrix", None)
    return {k: (list(v) if isinstance(v, list) and k == "word" else v) for k, v in d.items()}


def build_word_jobs_for_dim(
    *,
    word_jobs_base: List[Dict[str, Any]],
    n_ctrl: int,
) -> List[Dict[str, Any]]:
    word_jobs: List[Dict[str, Any]] = list(word_jobs_base)
    for j in range(n_ctrl):
        word_jobs.append(
            {
                "id": "random_symmetric_baseline",
                "word": [],
                "kind": "sym",
                "sym_index": j,
            }
        )
    return word_jobs


def build_job_list(
    *,
    dims: List[int],
    v_modes: List[str],
    target_cells: List[Tuple[str, int]],
    word_jobs_base: List[Dict[str, Any]],
    n_ctrl: int,
    base_seed: int,
    fro_ref_by_dim: Dict[int, float],
    max_iter: int,
    tol: float,
) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    job_id = 0
    for dim in dims:
        fro_ref = float(fro_ref_by_dim[dim])
        word_jobs = build_word_jobs_for_dim(word_jobs_base=word_jobs_base, n_ctrl=n_ctrl)
        cfg = dict(DIM_PARAM[dim])
        for wji, wj in enumerate(word_jobs):
            wid = str(wj["id"])
            for vm in v_modes:
                smooth_use = float(cfg["smooth_sigma"])
                clip_lo, clip_hi = float(CLIP_DEFAULT[0]), float(CLIP_DEFAULT[1])
                for tg, target_rep in target_cells:
                    cfg_seed = config_seed(base_seed, dim, job_id)
                    seed_t = target_seed_from_config(cfg_seed, vm, tg, wji, target_rep)
                    scid = stable_control_id(wid, tg, target_rep, wj)
                    jobs.append(
                        {
                            "job_id": job_id,
                            "dim": int(dim),
                            "V_mode": vm,
                            "word_group": wid,
                            "target_group": tg,
                            "target_rep": int(target_rep),
                            "control_id": int(scid),
                            "wji": int(wji),
                            "wj": _serializable_job_config(wj),
                            "seed": int(base_seed),
                            "config_seed": int(cfg_seed),
                            "seed_t": int(seed_t),
                            "max_iter": int(max_iter),
                            "tol": float(tol),
                            "fro_ref": fro_ref,
                            "dim_config": cfg,
                            "smooth_sigma": smooth_use,
                            "clip_lo": clip_lo,
                            "clip_hi": clip_hi,
                        }
                    )
                    job_id += 1
    return jobs


# --- shared zeta ordinates in worker ---
_Z_SORTED_MAIN: Optional[np.ndarray] = None
_Z_SORTED_WORKER: Optional[np.ndarray] = None
_WORKER_V13_VALIDATE_MOD: Any = None


def _v13o4_worker_init(z_sorted: np.ndarray) -> None:
    global _Z_SORTED_WORKER
    _Z_SORTED_WORKER = np.asarray(z_sorted, dtype=_DTF).reshape(-1)


def _z_sorted_for_eval() -> np.ndarray:
    if _Z_SORTED_WORKER is not None:
        return _Z_SORTED_WORKER
    if _Z_SORTED_MAIN is not None:
        return _Z_SORTED_MAIN
    raise RuntimeError("z_sorted not configured (set _Z_SORTED_MAIN or worker init)")


def _get_v13_validate_cached() -> Any:
    global _WORKER_V13_VALIDATE_MOD
    if _WORKER_V13_VALIDATE_MOD is None:
        _WORKER_V13_VALIDATE_MOD = load_v13_validate()
    return _WORKER_V13_VALIDATE_MOD


def _jsonl_append(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(json_sanitize(obj), allow_nan=False) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)


def read_jobs_jsonl(path: Path) -> Dict[int, Dict[str, Any]]:
    """Latest line wins per job_id."""
    by_id: Dict[int, Dict[str, Any]] = {}
    if not path.is_file():
        return by_id
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                o = json.loads(ln)
            except json.JSONDecodeError:
                continue
            jid = o.get("job_id")
            if jid is None:
                continue
            by_id[int(jid)] = o
    return by_id


def _record_for_jsonl(r: Dict[str, Any]) -> Dict[str, Any]:
    """JSONL-safe record without large arrays."""
    rec = r.get("rec") or {}
    base = dict(rec)
    return {
        **base,
        "job_id": r.get("job_id"),
        "runtime_s": r.get("runtime_s"),
        "worker_pid": r.get("worker_pid"),
        "error": r.get("error"),
        "traceback": r.get("traceback"),
    }


def jsonl_row_to_result_record(o: Dict[str, Any]) -> Dict[str, Any]:
    META = frozenset({"job_id", "runtime_s", "worker_pid", "error", "traceback"})
    rec = {k: v for k, v in o.items() if k not in META}
    return {
        "job_id": int(o["job_id"]),
        "rec": rec,
        "error": o.get("error"),
        "traceback": o.get("traceback"),
        "runtime_s": o.get("runtime_s"),
        "worker_pid": o.get("worker_pid"),
    }


def evaluate_one_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """Top-level picklable job runner."""
    job_id = job.get("job_id")
    t0 = time.time()
    try:
        cfg_seed = int(job.get("config_seed") or (int(job["seed"]) + 1000003 * int(job_id)))
        _ = float(np.random.default_rng(cfg_seed).random())
        v = _get_v13_validate_cached()
        from core import v13l_self_consistent as v13l
        from core import v13o2_specificity as v2
        from core.artin_operator import sample_domain
        from core.v13m1_renormalized import spectral_metrics_window as smw

        dim = int(job["dim"])
        vm = str(job["V_mode"])
        tg = str(job["target_group"])
        wid = str(job["word_group"])
        target_rep = int(job["target_rep"])
        wji = int(job["wji"])
        wj = dict(job["wj"])
        k_train = int(DIM_K_TRAIN[dim])
        z_sorted = _z_sorted_for_eval()
        if z_sorted.size < 2 * k_train:
            raise RuntimeError(f"Need {2 * k_train} zeros; got {z_sorted.size}.")

        cfg = dict(job["dim_config"])
        base_seed = int(job["seed"])
        seed_t = int(job["seed_t"])
        Z = sample_domain(dim, seed=base_seed)
        fro_ref = float(job["fro_ref"])

        kind = str(wj.get("kind", "word"))
        if kind == "sym":
            sym_index = int(wj["sym_index"])
            r = np.random.default_rng(base_seed + dim * 104729 + sym_index).standard_normal((dim, dim))
            S = 0.5 * (r + r.T)
            fs = float(np.linalg.norm(S, ord="fro"))
            H_op = (S * (fro_ref / max(fs, 1e-12))).astype(_DTF, copy=False)
        elif kind == "ablate_V":
            H_op = build_h(
                z_points=Z,
                word=list(wj.get("word") or []),
                v13l=v13l,
                geo_sigma=float(cfg["geo_sigma"]),
                laplacian_weight=float(wj["lap"]),
                geo_weight=float(wj["geo_w"]),
            )
        else:
            H_op = build_h(
                z_points=Z,
                word=list(wj.get("word") or []),
                v13l=v13l,
                geo_sigma=float(cfg["geo_sigma"]),
                laplacian_weight=float(wj["lap"]),
                geo_weight=float(wj["geo_w"]),
            )

        smooth_use = float(job["smooth_sigma"])
        clip_lo, clip_hi = float(job["clip_lo"]), float(job["clip_hi"])
        tr, te, train_ordered, wnote = build_train_test_targets(tg, z_sorted, k_train, seed_t, target_rep)
        z_train_m = np.sort(np.asarray(tr, dtype=_DTF).reshape(-1))
        z_test_m = np.sort(np.asarray(te, dtype=_DTF).reshape(-1))
        k_al = int(min(dim, k_train, z_train_m.size, z_test_m.size))

        eig_err: Optional[str] = None
        try:
            if kind == "ablate_V":
                H_fin = np.asarray(H_op, dtype=_DTF, copy=True)
                train_out = {
                    "H_final": H_fin,
                    "H_base": H_fin,
                    "V_diag_last": np.zeros((dim,), _DTF),
                    "meta": {"converged_operator": True, "n_iter": 0, "eig_error": None},
                    "operator_diff_final": 0.0,
                }
            else:
                train_out = v2.train_v13o2_cell(
                    H_base=np.asarray(H_op, dtype=_DTF, copy=True),
                    z_pool_sorted=z_sorted,
                    train_ordinates=tr,
                    train_ordered=train_ordered,
                    dim=dim,
                    k_train=k_train,
                    alpha=float(ALPHA),
                    lambda_p_dim=float(cfg["lambda_p"]),
                    beta0=float(cfg["beta0"]),
                    tau_beta=float(cfg["tau"]),
                    beta_floor=float(cfg["beta_floor"]),
                    smooth_sigma_dim=smooth_use,
                    clip_lo=clip_lo,
                    clip_hi=clip_hi,
                    diag_shift=float(DIAG_SHIFT),
                    abs_cap_factor=float(ABS_CAP),
                    zeros_train_metric=z_train_m,
                    spacing_fn=v.spacing_mse_normalized,
                    ks_fn=v.ks_against_wigner_gue,
                    norm_gaps_fn=v.normalized_gaps,
                    max_iter=int(job["max_iter"]),
                    tol=float(job["tol"]),
                    v_mode=vm,
                    z_points=np.asarray(Z, dtype=np.complex128),
                    word=list(wj.get("word") or []),
                    on_train_iter=None,
                )
                ee = train_out.get("meta", {}).get("eig_error")
                eig_err = ee if isinstance(ee, str) else None
        except Exception as ex:
            eig_err = str(ex)
            train_out = {
                "H_final": np.asarray(H_op, dtype=_DTF, copy=True),
                "H_base": np.asarray(H_op, dtype=_DTF, copy=True),
                "V_diag_last": np.zeros((dim,), _DTF),
                "meta": {"converged_operator": False, "eig_error": eig_err},
                "operator_diff_final": float("nan"),
            }

        H_fin = np.asarray(train_out["H_final"], dtype=_DTF, copy=True)
        try:
            eig = np.sort(np.linalg.eigvalsh(0.5 * (H_fin + H_fin.T))).astype(_DTF)
        except Exception as ex:
            eig_err = eig_err or str(ex)
            eig = np.full((dim,), np.nan, dtype=_DTF)

        finite = bool(np.isfinite(H_fin).all() and np.isfinite(eig).all())
        sa = float(v2.self_adjointness_fro(H_fin))
        od = float(train_out.get("operator_diff_final", float("nan")))
        meta_c = train_out.get("meta") or {}
        converged_operator = bool(meta_c.get("converged_operator", False))

        # Base spectral metrics on train/test windows
        mt_tr = smw(
            np.sort(np.asarray(eig, dtype=_DTF).reshape(-1)),
            z_train_m,
            k_al,
            spacing_fn=v.spacing_mse_normalized,
            ks_fn=v.ks_against_wigner_gue,
            norm_gaps_fn=v.normalized_gaps,
        )
        mt_te = smw(
            np.sort(np.asarray(eig, dtype=_DTF).reshape(-1)),
            z_test_m,
            k_al,
            spacing_fn=v.spacing_mse_normalized,
            ks_fn=v.ks_against_wigner_gue,
            norm_gaps_fn=v.normalized_gaps,
        )

        # New metrics (train/test)
        pc_tr = pair_corr_error(eig, z_train_m, k_al)
        pc_te = pair_corr_error(eig, z_test_m, k_al)
        nv_tr = number_variance_error(eig, z_train_m, k_al)
        nv_te = number_variance_error(eig, z_test_m, k_al)
        st_tr = staircase_residual_error(eig, z_train_m, k_al)
        st_te = staircase_residual_error(eig, z_test_m, k_al)

        gap_sl = float(mt_te["spectral_log_mse"] - mt_tr["spectral_log_mse"])
        gap_pc = float(pc_te - pc_tr)
        gap_nv = float(nv_te - nv_tr)

        tpen = transfer_gap_penalty(gap_spectral=gap_sl, gap_pair=gap_pc, gap_number_var=gap_nv)

        rec: Dict[str, Any] = {
            "dim": dim,
            "V_mode": vm,
            "word_group": wid,
            "target_group": tg,
            "control_id": int(job["control_id"]),
            "target_rep": int(target_rep),
            "word_job_index": wji,
            "window_note": wnote,
            "train_k": k_train,
            "test_k": k_train,
            "spectral_log_mse_train": float(mt_tr["spectral_log_mse"]),
            "spacing_mse_normalized_train": float(mt_tr["spacing_mse_normalized"]),
            "ks_wigner_train": float(mt_tr["ks_wigner"]),
            "spectral_log_mse_test": float(mt_te["spectral_log_mse"]),
            "spacing_mse_normalized_test": float(mt_te["spacing_mse_normalized"]),
            "ks_wigner_test": float(mt_te["ks_wigner"]),
            "pair_corr_error_train": float(pc_tr),
            "pair_corr_error_test": float(pc_te),
            "number_variance_error_train": float(nv_tr),
            "number_variance_error_test": float(nv_te),
            "staircase_residual_error_train": float(st_tr),
            "staircase_residual_error_test": float(st_te),
            "gap_spectral": float(gap_sl),
            "gap_pair_corr": float(gap_pc),
            "gap_number_variance": float(gap_nv),
            "transfer_gap_penalty": float(tpen),
            "operator_diff_final": float(od),
            "converged_operator": bool(converged_operator),
            "finite": bool(finite),
            "self_adjointness_fro": float(sa),
            "eig_error": eig_err or "",
        }

        # per-row objective (without ensemble margin, filled later during aggregation)
        rec["ensemble_margin_penalty"] = 0.0

        def f(x: Any, default: float) -> float:
            try:
                v0 = float(x)
                return v0 if math.isfinite(v0) else default
            except Exception:
                return default

        w = DEFAULT_WEIGHTS
        j_te = (
            float(w["w_spectral"]) * f(mt_te.get("spectral_log_mse"), 1.0e6)
            + float(w["w_spacing"]) * f(mt_te.get("spacing_mse_normalized"), 1.0e6)
            + float(w["w_ks"]) * f(mt_te.get("ks_wigner"), 1.0)
            + float(w["w_pair"]) * f(pc_te, 1.0e6)
            + float(w["w_number_var"]) * f(nv_te, 1.0e6)
            + float(w["w_staircase"]) * f(st_te, 1.0e6)
            + float(w["w_transfer"]) * f(tpen, 1.0e6)
            + float(w["w_ensemble"]) * 0.0
        )
        rec["J_zeta_base"] = float(j_te)
        rec["J_zeta"] = float(j_te)  # updated later with ensemble_margin_penalty in aggregation

        accepted_basic = meets_basic_accept(finite=finite, sa=sa, od=od, eig_err=eig_err)
        rec["accepted_basic"] = bool(accepted_basic)

        out = {
            "job_id": int(job_id),
            "runtime_s": time.time() - t0,
            "worker_pid": os.getpid(),
            "error": None,
            "traceback": None,
            "rec": rec,
        }
        return out
    except Exception as e:
        # Minimal record for failures
        nan = float("nan")
        rec = {
            "dim": job.get("dim"),
            "V_mode": job.get("V_mode"),
            "word_group": job.get("word_group"),
            "target_group": job.get("target_group"),
            "control_id": job.get("control_id"),
            "target_rep": job.get("target_rep", 0),
            "word_job_index": job.get("wji"),
            "window_note": "",
            "train_k": int(DIM_K_TRAIN.get(int(job.get("dim") or 0), 0)),
            "test_k": int(DIM_K_TRAIN.get(int(job.get("dim") or 0), 0)),
            "spectral_log_mse_train": nan,
            "spacing_mse_normalized_train": nan,
            "ks_wigner_train": nan,
            "spectral_log_mse_test": nan,
            "spacing_mse_normalized_test": nan,
            "ks_wigner_test": nan,
            "pair_corr_error_train": nan,
            "pair_corr_error_test": nan,
            "number_variance_error_train": nan,
            "number_variance_error_test": nan,
            "staircase_residual_error_train": nan,
            "staircase_residual_error_test": nan,
            "gap_spectral": nan,
            "gap_pair_corr": nan,
            "gap_number_variance": nan,
            "transfer_gap_penalty": nan,
            "ensemble_margin_penalty": 0.0,
            "J_zeta_base": nan,
            "J_zeta": nan,
            "operator_diff_final": nan,
            "converged_operator": False,
            "finite": False,
            "self_adjointness_fro": nan,
            "accepted_basic": False,
            "eig_error": repr(e),
        }
        jid = job.get("job_id")
        return {
            "job_id": int(jid) if jid is not None else -1,
            "runtime_s": time.time() - t0,
            "worker_pid": os.getpid(),
            "error": repr(e),
            "traceback": traceback.format_exc(),
            "rec": rec,
        }


def _sort_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def key(r: Dict[str, Any]) -> Tuple:
        return (
            int(r.get("dim") or -1),
            str(r.get("V_mode") or ""),
            str(r.get("word_group") or ""),
            str(r.get("target_group") or ""),
            int(r.get("control_id") or 0),
        )

    return sorted(rows, key=key)


def dedupe_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Set[Tuple[Any, ...]] = set()
    out: List[Dict[str, Any]] = []
    for r in rows:
        key = (r["dim"], r["V_mode"], r["word_group"], r["target_group"], int(r["control_id"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _median_finite(xs: Sequence[float]) -> float:
    arr = np.asarray([float(x) for x in xs if math.isfinite(float(x))], dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.median(arr))


def _compute_ensemble_margin_penalties(
    rows: List[Dict[str, Any]],
    *,
    margin: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Compute ensemble_margin_penalty per (dim, V_mode, word_group) anchored on real_zeta.
    Updates rows in-place for target_group==real_zeta by adding penalty and updating J_zeta.
    Returns (updated_rows, margin_rows_csv).
    """
    # Index rows by cell
    by_cell: Dict[Tuple[int, str, str, str], List[Dict[str, Any]]] = {}
    for r in rows:
        key = (int(r["dim"]), str(r["V_mode"]), str(r["word_group"]), str(r["target_group"]))
        by_cell.setdefault(key, []).append(r)

    margin_rows: List[Dict[str, Any]] = []
    for (dim, vm, wg, tg), rs in list(by_cell.items()):
        if tg != "real_zeta":
            continue
        # pick representative J_real as median across any reps (should be 1)
        j_real = _median_finite([float(x.get("J_zeta_base", float("nan"))) for x in rs])
        if not math.isfinite(j_real):
            continue
        penalties: Dict[str, float] = {}
        for cn in CONTROL_TARGETS_FOR_MARGIN:
            ctl = by_cell.get((dim, vm, wg, cn), [])
            med_ctl = _median_finite([float(x.get("J_zeta_base", float("nan"))) for x in ctl])
            if not math.isfinite(med_ctl):
                penalties[cn] = float("nan")
                continue
            penalties[cn] = float(max(0.0, float(margin) + float(j_real) - float(med_ctl)))
        # aggregate penalty as sum over control groups with finite medians
        p_sum = float(sum(v for v in penalties.values() if math.isfinite(v)))

        for r in rs:
            r["ensemble_margin_penalty"] = p_sum
            r["J_zeta"] = float(r.get("J_zeta_base", float("nan"))) + float(DEFAULT_WEIGHTS["w_ensemble"]) * p_sum

        margin_rows.append(
            {
                "dim": dim,
                "V_mode": vm,
                "word_group": wg,
                "J_real_base": j_real,
                "margin": float(margin),
                "ensemble_margin_penalty_sum": p_sum,
                **{f"penalty_vs_{k}": penalties.get(k, float("nan")) for k in CONTROL_TARGETS_FOR_MARGIN},
            }
        )
    return rows, margin_rows


def _gate_summary_primary_full(
    rows: List[Dict[str, Any]],
    *,
    dims: List[int],
    margin: float,
) -> List[Dict[str, Any]]:
    """Gate summary for primary_word_seed6 + full_V per dim."""

    def filt(d: int, tg: str) -> List[Dict[str, Any]]:
        return [
            r
            for r in rows
            if int(r["dim"]) == int(d)
            and r["V_mode"] == "full_V"
            and r["word_group"] == "primary_word_seed6"
            and r["target_group"] == tg
        ]

    gates: List[Dict[str, Any]] = []
    for d in dims:
        real = filt(d, "real_zeta")
        j_real = _median_finite([float(r.get("J_zeta", float("nan"))) for r in real])
        pc_real = _median_finite([float(r.get("pair_corr_error_test", float("nan"))) for r in real])
        nv_real = _median_finite([float(r.get("number_variance_error_test", float("nan"))) for r in real])
        st_real = _median_finite([float(r.get("staircase_residual_error_test", float("nan"))) for r in real])
        gap_real = _median_finite([float(r.get("gap_spectral", float("nan"))) for r in real])
        finite_ok = bool(real) and all(bool(r.get("finite")) for r in real)
        sa_ok = bool(real) and all(float(r.get("self_adjointness_fro", 1.0)) <= SELF_ADJOINTNESS_FRO_MAX for r in real)
        od_ok = bool(real) and all(float(r.get("operator_diff_final", 1.0e9)) <= OPERATOR_DIFF_MAX for r in real)

        def medJ(tg: str) -> float:
            return _median_finite([float(r.get("J_zeta", float("nan"))) for r in filt(d, tg)])

        def med_metric(tg: str, key: str) -> float:
            return _median_finite([float(r.get(key, float("nan"))) for r in filt(d, tg)])

        med_shuf = medJ("shuffled_zeta")
        med_gue = medJ("GUE_synthetic")
        med_poi = medJ("Poisson_synthetic")

        # For metric-beats-median-controls, use the same controls pool
        pc_controls = _median_finite([med_metric(tg, "pair_corr_error_test") for tg in CONTROL_TARGETS_FOR_MARGIN])
        nv_controls = _median_finite([med_metric(tg, "number_variance_error_test") for tg in CONTROL_TARGETS_FOR_MARGIN])
        st_controls = _median_finite([med_metric(tg, "staircase_residual_error_test") for tg in CONTROL_TARGETS_FOR_MARGIN])

        def lt(a: float, b: float) -> bool:
            return math.isfinite(a) and math.isfinite(b) and a < b

        g1 = bool(finite_ok and sa_ok)
        g2 = bool(od_ok)
        g3 = bool(lt(j_real, med_shuf - float(margin)))
        g4 = bool(lt(j_real, med_gue - float(margin)))
        g5 = bool(lt(j_real, med_poi - float(margin)))
        g6 = bool(lt(pc_real, pc_controls - float(PAIR_CORR_ERROR_MARGIN)))
        g7 = bool(lt(nv_real, nv_controls - float(NUMBER_VAR_ERROR_MARGIN)))
        g8 = bool(lt(st_real, st_controls - float(STAIRCASE_ERROR_MARGIN)))
        g9 = bool(math.isfinite(gap_real) and gap_real <= TRANSFER_GAP_SPECTRAL_MAX)
        g10 = bool(real) and all(bool(r.get("accepted_basic")) for r in real)

        strict = bool(g1 and g2 and g3 and g4 and g5 and g6 and g7 and g8 and g9 and g10)
        gates.append(
            {
                "dim": int(d),
                "G1_finite_self_adjoint": g1,
                "G2_operator_diff_lt_1e-3": g2,
                "G3_J_real_lt_median_shuffled_minus_margin": g3,
                "G4_J_real_lt_median_GUE_minus_margin": g4,
                "G5_J_real_lt_median_Poisson_minus_margin": g5,
                "G6_pair_corr_beats_median_controls": g6,
                "G7_number_variance_beats_median_controls": g7,
                "G8_staircase_beats_median_controls": g8,
                "G9_transfer_gap_acceptable": g9,
                "G10_accepted_real_zeta": g10,
                "strict_zeta_specificity_pass": strict,
                "J_real": j_real,
                "median_J_shuffled": med_shuf,
                "median_J_GUE": med_gue,
                "median_J_Poisson": med_poi,
                "pair_corr_real": pc_real,
                "pair_corr_median_controls": pc_controls,
                "number_variance_real": nv_real,
                "number_variance_median_controls": nv_controls,
                "staircase_real": st_real,
                "staircase_median_controls": st_controls,
                "gap_spectral_real": gap_real,
            }
        )
    return gates


def main() -> None:
    ap = argparse.ArgumentParser(description="V13O.4 zeta-specific objective (computational evidence only).")
    ap.add_argument("--candidate_json", type=str, default="runs/v13_candidate_theorem/candidate_operators.json")
    ap.add_argument("--formula_json", type=str, default="runs/v13_operator_formula/formula_components_summary.json")
    ap.add_argument("--v13n_summary", type=str, default="runs/v13n_theorem_report/v13n_summary.json")
    ap.add_argument("--out_dir", type=str, default="runs/v13o4_zeta_specific_objective")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dims", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--max_iter", type=int, default=300)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--n_controls", type=int, default=30)
    ap.add_argument("--n_random_words", type=int, default=30)
    ap.add_argument("--margin", type=float, default=0.25)
    ap.add_argument("--progress_every", type=int, default=1)
    ap.add_argument("--workers", type=int, default=1, help="Parallel workers (default 1). If >1, run configs in parallel.")
    ap.add_argument(
        "--parallel_backend",
        type=str,
        default="multiprocessing",
        choices=["multiprocessing"],
        help="Parallel backend (currently multiprocessing-style processes).",
    )
    ap.add_argument("--chunksize", type=int, default=1, help="Submission batch sizing (default 1).")
    ap.add_argument("--fail_fast", action="store_true", help="Abort on first worker failure.")
    ap.add_argument("--resume", action="store_true", help="Skip finished job_ids from existing raw JSONL.")
    ap.add_argument("--raw_jsonl", type=str, default="v13o4_raw_jobs.jsonl")
    args = ap.parse_args()

    out_dir = _resolve(args.out_dir)
    ck_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ck_dir.mkdir(parents=True, exist_ok=True)

    cand_path = _resolve(args.candidate_json)
    form_path = _resolve(args.formula_json)
    v13n_path = _resolve(args.v13n_summary)
    cand = _require_json(cand_path)
    form = _require_json(form_path)
    v13n = _require_json(v13n_path)

    spect = cand.get("spectral_candidate") or {}
    if spect.get("id") != PRIMARY_CANDIDATE_ID:
        print(
            f"[v13o4] WARNING: expected spectral_candidate.id={PRIMARY_CANDIDATE_ID}, got {spect.get('id')}",
            flush=True,
        )
    primary_word = [int(x) for x in (spect.get("word") or PRIMARY_WORD_FALLBACK)]
    if primary_word != PRIMARY_WORD_FALLBACK:
        print(f"[v13o4] Using primary word from candidate_json (id={spect.get('id')}).", flush=True)

    rej = None
    for rc in cand.get("rejected_candidates") or []:
        if rc.get("id") == "seed_17":
            rej = [int(x) for x in rc.get("word") or REJECTED_WORD_FALLBACK]
            break
    rejected_word = rej or REJECTED_WORD_FALLBACK

    v = load_v13_validate()
    from core import v13l_self_consistent as v13l
    from core.artin_operator import sample_domain

    dims = [int(d) for d in args.dims if int(d) in DIM_K_TRAIN]
    if not dims:
        raise SystemExit("No valid dims in --dims (expected subset of 64,128,256).")

    n_ctrl = max(1, int(args.n_controls))
    n_rw = max(1, int(args.n_random_words))
    base_seed = int(args.seed)
    prog = max(1, int(args.progress_every))
    margin = float(args.margin)

    max_k = max(DIM_K_TRAIN[d] for d in dims)
    z_pool = v._load_zeros(max(512, 2 * max_k))
    z_sorted = np.sort(np.asarray(z_pool, dtype=_DTF).reshape(-1))
    z_sorted = z_sorted[np.isfinite(z_sorted) & (z_sorted > 0.0)]

    word_jobs_base: List[Dict[str, Any]] = [
        {"id": "primary_word_seed6", "word": list(primary_word), "lap": 1.0, "geo_w": GEO_WEIGHT, "kind": "word"},
        {"id": "rejected_word_seed17", "word": list(rejected_word), "lap": 1.0, "geo_w": GEO_WEIGHT, "kind": "word"},
        {"id": "ablate_K", "word": list(primary_word), "lap": 1.0, "geo_w": 0.0, "kind": "word"},
        {"id": "ablate_V", "word": list(primary_word), "lap": 1.0, "geo_w": GEO_WEIGHT, "kind": "ablate_V"},
        {"id": "ablate_L", "word": list(primary_word), "lap": 0.0, "geo_w": GEO_WEIGHT, "kind": "word"},
    ]
    for j in range(n_rw):
        alphabet = list(range(-6, 0)) + list(range(1, 7))
        rw = [int(np.random.default_rng(base_seed + 7919 + j).choice(alphabet)) for _ in range(len(primary_word))]
        word_jobs_base.append(
            {"id": "random_words_n30", "word": rw, "lap": 1.0, "geo_w": GEO_WEIGHT, "kind": "word", "rw_index": j}
        )

    target_cells = count_target_cells(n_ctrl)
    v_modes = list(V_MODES_ALL)

    raw_jsonl_arg = Path(args.raw_jsonl)
    raw_jobs_path = raw_jsonl_arg if raw_jsonl_arg.is_absolute() else out_dir / raw_jsonl_arg

    # fro ref cache
    fro_ref_cache: Dict[int, float] = {}
    for dim in dims:
        Zpre = sample_domain(dim, seed=base_seed)
        cfg_pre = DIM_PARAM[dim]
        fro_ref_cache[dim] = float(
            np.linalg.norm(
                build_h(
                    z_points=Zpre,
                    word=list(primary_word),
                    v13l=v13l,
                    geo_sigma=float(cfg_pre["geo_sigma"]),
                    laplacian_weight=1.0,
                    geo_weight=float(GEO_WEIGHT),
                ),
                ord="fro",
            )
        )

    for dim in dims:
        k_need = DIM_K_TRAIN[dim]
        if z_sorted.size < 2 * k_need:
            raise RuntimeError(f"Need {2 * k_need} zeros; got {z_sorted.size}.")

    jobs = build_job_list(
        dims=dims,
        v_modes=v_modes,
        target_cells=target_cells,
        word_jobs_base=word_jobs_base,
        n_ctrl=n_ctrl,
        base_seed=base_seed,
        fro_ref_by_dim=fro_ref_cache,
        max_iter=int(args.max_iter),
        tol=float(args.tol),
    )
    total_jobs = len(jobs)
    n_word_jobs = len(word_jobs_base) + n_ctrl
    print(
        f"[v13o4] workload grid: {len(dims)} dims × {n_word_jobs} word_jobs × {len(v_modes)} V_modes "
        f"× {len(target_cells)} target_cells → {total_jobs} independent jobs",
        flush=True,
    )

    done_ids_pre: Set[int] = set()
    if args.resume:
        done_ids_pre = set(read_jobs_jsonl(raw_jobs_path).keys())

    if not args.resume and raw_jobs_path.exists():
        raw_jobs_path.unlink()

    jobs_to_run = [j for j in jobs if int(j["job_id"]) not in done_ids_pre]

    global _Z_SORTED_MAIN
    _Z_SORTED_MAIN = z_sorted

    workers = max(1, int(args.workers))
    chunk_mul = max(1, int(args.chunksize))
    max_inflight = max(1, workers * chunk_mul)

    print(
        f"[v13o4] jobs total={total_jobs} to_run={len(jobs_to_run)} resume={bool(args.resume)} "
        f"workers={workers} parallel={bool(workers > 1)} backend={args.parallel_backend}",
        flush=True,
    )

    exec_t0 = time.perf_counter()
    done_total = len(done_ids_pre)

    def log_done(rec: Dict[str, Any], *, i_done: int, N: int) -> None:
        elapsed = time.perf_counter() - exec_t0
        session_finished = max(0, i_done - len(done_ids_pre))
        avg = elapsed / max(session_finished, 1)
        eta_sec = avg * max(N - i_done, 0)
        jz = rec.get("J_zeta")
        pair = rec.get("pair_corr_error_test")
        nv = rec.get("number_variance_error_test")
        st = rec.get("staircase_residual_error_test")
        acc = bool(rec.get("accepted_basic"))
        print(
            f"[v13o4] done {i_done}/{N} workers={workers} dim={rec.get('dim')} target={rec.get('target_group')} "
            f"word_group={rec.get('word_group')} V_mode={rec.get('V_mode')} "
            f"J_zeta={float(jz):.4g} pair={float(pair):.4g} numvar={float(nv):.4g} stair={float(st):.4g} "
            f"accepted={acc} elapsed={elapsed:.1f}s eta={format_seconds(eta_sec)}",
            flush=True,
        )

    def process_finished_row(row: Dict[str, Any]) -> None:
        nonlocal done_total
        done_total += 1
        _jsonl_append(raw_jobs_path, _record_for_jsonl(row))
        rec = row.get("rec") or {}
        if done_total == 1 or done_total == total_jobs or done_total % prog == 0:
            if all(k in rec for k in ("J_zeta", "pair_corr_error_test", "number_variance_error_test", "staircase_residual_error_test")):
                log_done(rec, i_done=done_total, N=total_jobs)

    if jobs_to_run:
        if workers > 1:
            # NOTE: ProcessPoolExecutor on some macOS/Python builds can raise PermissionError
            # from os.sysconf("SC_SEM_NSEMS_MAX"). Use multiprocessing Pool instead.
            import multiprocessing as mp

            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=workers, initializer=_v13o4_worker_init, initargs=(z_sorted,)) as pool:
                it = pool.imap_unordered(evaluate_one_job, jobs_to_run, chunksize=chunk_mul)
                for res in _tqdm(it, total=len(jobs_to_run), desc="v13o4"):
                    if args.fail_fast and res.get("error"):
                        pool.terminate()
                        pool.join()
                        raise RuntimeError(f"fail_fast: worker failure: {res.get('error')}")
                    process_finished_row(res)
        else:
            for job in _tqdm(jobs_to_run, total=len(jobs_to_run), desc="v13o4"):
                res = evaluate_one_job(job)
                if args.fail_fast and res.get("error"):
                    raise RuntimeError(f"fail_fast: worker failure: {res.get('error')}")
                process_finished_row(res)

    wall_s = time.perf_counter() - exec_t0

    final_by_id = read_jobs_jsonl(raw_jobs_path)
    if len(final_by_id) != total_jobs:
        print(
            f"[v13o4] WARNING: raw JSONL has {len(final_by_id)}/{total_jobs} jobs; aggregating available rows only.",
            flush=True,
        )

    combined = [jsonl_row_to_result_record(o) for o in final_by_id.values()]
    all_rows = [dict(r["rec"]) for r in combined if r.get("rec") is not None]
    all_rows = dedupe_rows(_sort_rows(all_rows))

    # Compute ensemble margin penalties and update J_zeta (real_zeta rows only)
    all_rows, margin_rows = _compute_ensemble_margin_penalties(all_rows, margin=margin)

    # Outputs: required CSVs
    def write_csv(path: Path, fields: List[str], rows: List[Dict[str, Any]]) -> None:
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for row in rows:
                w.writerow({k: csv_cell(row.get(k)) for k in fields})

    if all_rows:
        write_csv(out_dir / "v13o4_summary.csv", list(all_rows[0].keys()), all_rows)

    # group summary
    group_map: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for r in all_rows:
        key = (r["dim"], r["V_mode"], r["word_group"], r["target_group"])
        group_map.setdefault(key, []).append(r)

    group_summary: List[Dict[str, Any]] = []
    for key, rs in sorted(group_map.items(), key=lambda kv: kv[0]):
        d, vm, wg, tg = key
        Js = [float(x.get("J_zeta", float("nan"))) for x in rs if math.isfinite(float(x.get("J_zeta", float("nan"))))]
        acc = float(sum(1 for x in rs if x.get("accepted_basic"))) / max(1, len(rs))
        group_summary.append(
            {
                "dim": d,
                "V_mode": vm,
                "word_group": wg,
                "target_group": tg,
                "n": len(rs),
                "accepted_basic_rate": acc,
                "median_J_zeta": _median_finite(Js),
                "min_J_zeta": float(np.min(np.asarray(Js, dtype=np.float64))) if Js else float("nan"),
            }
        )
    if group_summary:
        write_csv(out_dir / "v13o4_group_summary.csv", list(group_summary[0].keys()), group_summary)

    # zeta-specific scores: focus on test-window metrics
    zeta_scores = [
        {
            "dim": r["dim"],
            "V_mode": r["V_mode"],
            "word_group": r["word_group"],
            "target_group": r["target_group"],
            "control_id": r["control_id"],
            "J_zeta": r.get("J_zeta"),
            "J_zeta_base": r.get("J_zeta_base"),
            "spectral_log_mse_test": r.get("spectral_log_mse_test"),
            "spacing_mse_normalized_test": r.get("spacing_mse_normalized_test"),
            "ks_wigner_test": r.get("ks_wigner_test"),
            "pair_corr_error_test": r.get("pair_corr_error_test"),
            "number_variance_error_test": r.get("number_variance_error_test"),
            "staircase_residual_error_test": r.get("staircase_residual_error_test"),
            "transfer_gap_penalty": r.get("transfer_gap_penalty"),
            "ensemble_margin_penalty": r.get("ensemble_margin_penalty"),
            "accepted_basic": r.get("accepted_basic"),
        }
        for r in all_rows
    ]
    if zeta_scores:
        write_csv(out_dir / "v13o4_zeta_specific_scores.csv", list(zeta_scores[0].keys()), zeta_scores)

    # per-metric summaries
    pair_rows = [
        {
            "dim": r["dim"],
            "V_mode": r["V_mode"],
            "word_group": r["word_group"],
            "target_group": r["target_group"],
            "control_id": r["control_id"],
            "pair_corr_error_train": r.get("pair_corr_error_train"),
            "pair_corr_error_test": r.get("pair_corr_error_test"),
            "gap_pair_corr": r.get("gap_pair_corr"),
        }
        for r in all_rows
    ]
    if pair_rows:
        write_csv(out_dir / "v13o4_pair_corr_summary.csv", list(pair_rows[0].keys()), pair_rows)

    nv_rows = [
        {
            "dim": r["dim"],
            "V_mode": r["V_mode"],
            "word_group": r["word_group"],
            "target_group": r["target_group"],
            "control_id": r["control_id"],
            "number_variance_error_train": r.get("number_variance_error_train"),
            "number_variance_error_test": r.get("number_variance_error_test"),
            "gap_number_variance": r.get("gap_number_variance"),
        }
        for r in all_rows
    ]
    if nv_rows:
        write_csv(out_dir / "v13o4_number_variance_summary.csv", list(nv_rows[0].keys()), nv_rows)

    st_rows = [
        {
            "dim": r["dim"],
            "V_mode": r["V_mode"],
            "word_group": r["word_group"],
            "target_group": r["target_group"],
            "control_id": r["control_id"],
            "staircase_residual_error_train": r.get("staircase_residual_error_train"),
            "staircase_residual_error_test": r.get("staircase_residual_error_test"),
        }
        for r in all_rows
    ]
    if st_rows:
        write_csv(out_dir / "v13o4_staircase_summary.csv", list(st_rows[0].keys()), st_rows)

    if margin_rows:
        write_csv(out_dir / "v13o4_ensemble_margins.csv", list(margin_rows[0].keys()), margin_rows)

    gate_rows = _gate_summary_primary_full(all_rows, dims=dims, margin=margin)
    if gate_rows:
        write_csv(out_dir / "v13o4_gate_summary.csv", list(gate_rows[0].keys()), gate_rows)

    strict_all_dims = bool(gate_rows) and all(bool(g.get("strict_zeta_specificity_pass")) for g in gate_rows)
    if strict_all_dims:
        interp = "STRICT_ZETA_SPECIFICITY"
    else:
        # partial heuristic: beats structural controls but not GUE/Poisson
        partial = False
        for g in gate_rows:
            structural_ok = bool(
                g.get("G3_J_real_lt_median_shuffled_minus_margin")
                and g.get("G6_pair_corr_beats_median_controls")
                and g.get("G7_number_variance_beats_median_controls")
            )
            fail_ensembles = not bool(
                g.get("G4_J_real_lt_median_GUE_minus_margin") and g.get("G5_J_real_lt_median_Poisson_minus_margin")
            )
            if structural_ok and fail_ensembles:
                partial = True
        interp = "PARTIAL_ZETA_SPECIFICITY" if partial else "NO_ZETA_SPECIFICITY"

    payload = {
        "warning": "Computational evidence only; not a proof of the Riemann Hypothesis.",
        "status": "V13O.4 zeta-specific objective",
        "interpretation": interp,
        "strict_zeta_specificity_all_dims": strict_all_dims,
        "J_zeta_definition": (
            "J_zeta = 0.25*spectral_log_mse + 0.25*spacing_mse_normalized + 0.25*ks_wigner "
            "+ 1.0*pair_corr_error + 1.0*number_variance_error + 0.5*staircase_residual_error "
            "+ 0.5*transfer_gap_penalty + 1.0*ensemble_margin_penalty"
        ),
        "weights": dict(DEFAULT_WEIGHTS),
        "acceptance_thresholds": {
            "operator_diff_max": OPERATOR_DIFF_MAX,
            "self_adjointness_fro_max": SELF_ADJOINTNESS_FRO_MAX,
            "ensemble_margin": margin,
            "transfer_gap_spectral_max": TRANSFER_GAP_SPECTRAL_MAX,
            "transfer_gap_pair_max": TRANSFER_GAP_PAIR_MAX,
            "transfer_gap_number_var_max": TRANSFER_GAP_NUMBER_VAR_MAX,
        },
        "inputs": {
            "candidate_json": str(cand_path),
            "formula_json": str(form_path),
            "v13n_summary": str(v13n_path),
            "candidate_id": spect.get("id"),
            "formula_meta_builder": (form.get("meta") or {}).get("builder_module"),
        },
        "v13n_operator_formula": v13n.get("operator_formula"),
        "dims": dims,
        "v_modes": v_modes,
        "target_groups": list(TARGET_GROUPS_ALL),
        "n_controls": n_ctrl,
        "n_random_words": n_rw,
        "total_jobs": total_jobs,
        "completed_jobs": len(final_by_id),
        "total_runtime_s": float(wall_s),
        "python_pid": os.getpid(),
        "raw_jobs_jsonl": str(raw_jobs_path.resolve()),
        "resume": bool(args.resume),
        "gate_summary": json_sanitize(gate_rows),
    }

    with open(out_dir / "v13o4_results.json", "w", encoding="utf-8") as f:
        json.dump(json_sanitize(payload), f, indent=2, allow_nan=False)

    md = [
        "# V13O.4 Zeta-specific objective (pre-renormalization)\n\n",
        "> **Computational evidence only; not a proof of the Riemann Hypothesis.**\n\n",
        "## Interpretation\n\n",
        f"**{interp}** (strict_zeta_specificity_all_dims={strict_all_dims}).\n\n",
        "## Objective\n\n",
        "`J_zeta` includes local spacing/KS terms plus **pair correlation**, **number variance**, and **staircase residual** errors, ",
        "plus penalties for **train/test transfer gap** and **ensemble margins**.\n\n",
        "## Gate summary (primary_word_seed6 + full_V)\n\n",
        "See `v13o4_gate_summary.csv`.\n\n",
        "## Outputs\n\n",
        "- `v13o4_results.json`\n",
        "- `v13o4_summary.csv`\n",
        "- `v13o4_group_summary.csv`\n",
        "- `v13o4_zeta_specific_scores.csv`\n",
        "- `v13o4_pair_corr_summary.csv`\n",
        "- `v13o4_number_variance_summary.csv`\n",
        "- `v13o4_staircase_summary.csv`\n",
        "- `v13o4_ensemble_margins.csv`\n",
        "- `v13o4_gate_summary.csv`\n\n",
    ]
    (out_dir / "v13o4_report.md").write_text("".join(md), encoding="utf-8")

    tex_body = (
        "\\documentclass[11pt]{article}\n\\usepackage[margin=1in]{geometry}\n\\begin{document}\n"
        "\\title{V13O.4 Zeta-specific objective}\n\\maketitle\n"
        "\\section*{Warning}\nComputational evidence only; not a proof of the Riemann Hypothesis.\n\n"
        "\\section*{Interpretation}\n"
        + latex_escape(interp)
        + "\n\n\\end{document}\n"
    )
    (out_dir / "v13o4_report.tex").write_text(tex_body, encoding="utf-8")

    if _find_pdflatex() is None:
        print("[v13o4] WARNING: pdflatex not found; skipping PDF.", flush=True)
    elif try_pdflatex(out_dir / "v13o4_report.tex", out_dir, "v13o4_report.pdf"):
        print(f"Wrote {out_dir / 'v13o4_report.pdf'}", flush=True)
    else:
        print("[v13o4] WARNING: pdflatex failed or did not produce v13o4_report.pdf.", flush=True)

    print(f"[v13o4] Wrote {out_dir / 'v13o4_results.json'}", flush=True)


if __name__ == "__main__":
    main()

