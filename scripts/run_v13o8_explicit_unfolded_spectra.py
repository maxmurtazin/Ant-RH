#!/usr/bin/env python3
"""
V13O.8 — Explicit unfolded spectra export + true number variance curve computation.

Computational evidence only; not a proof of RH.

This script exports explicit per-level spectra (target + operator), computes Sigma^2(L) curves
directly from unfolded levels, and performs curve-region + Poisson/GUE diagnostics with gates.

Two modes:
  - True recompute mode (approximation_mode=False): requires --candidate_json and uses the same
    training/unfolding logic as V13O.4 (recomputed) to obtain operator eigenvalues and target windows.
  - Fallback reconstruction mode (approximation_mode=True): if raw operator computation cannot run,
    reconstructs plausible unfolded level arrays from scalar summaries. Still exports levels/curves
    but labels them clearly (window_note warnings).
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import importlib.util
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pandas as pd  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit("This script requires pandas. Please install pandas and retry.") from e


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


CONTROL_TARGET_GROUPS = [
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

TARGET_GROUPS_ALL = ["real_zeta"] + CONTROL_TARGET_GROUPS

CANDIDATE_WORD_GROUPS = [
    "primary_word_seed6",
    "ablate_K",
    "ablate_L",
    "ablate_V",
    "random_symmetric_baseline",
    "random_words_n30",
    "rejected_word_seed17",
]


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
        if math.isfinite(x):
            return float(x)
        return None
    try:
        xf = float(x)
        if math.isfinite(xf):
            return float(xf)
    except Exception:
        pass
    return str(x)


def read_csv_robust(path: Path, *, name: str, tag: str = "v13o8") -> Optional["pd.DataFrame"]:
    if not path or str(path).strip() == "":
        return None
    if not path.is_file():
        print(f"[{tag}] WARNING missing input: {name}={path}", flush=True)
        return None
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[{tag}] WARNING failed reading {name}={path}: {e!r}", flush=True)
        return None
    df.columns = [str(c).strip() for c in df.columns]
    return df


def finite_series(x: "pd.Series") -> "pd.Series":
    return pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan)


def pick_col(df: Optional["pd.DataFrame"], candidates: Sequence[str]) -> Optional[str]:
    if df is None:
        return None
    cols = {str(c).strip(): c for c in df.columns}
    for cand in candidates:
        if cand in cols:
            return str(cols[cand])
    low = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        k = cand.lower()
        if k in low:
            return str(low[k])
    return None


def _finite_arr(xs: Iterable[float]) -> np.ndarray:
    arr = np.asarray([float(v) for v in xs if v is not None and math.isfinite(float(v))], dtype=np.float64)
    return arr.reshape(-1)


def safe_median(xs: Sequence[float]) -> float:
    a = _finite_arr(xs)
    return float(np.median(a)) if a.size else float("nan")


# -----------------------------
# Curve + unfolding utilities
# -----------------------------


def unfold_to_mean_spacing_one(x: np.ndarray) -> np.ndarray:
    x = np.sort(np.asarray(x, dtype=np.float64).reshape(-1))
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.zeros((1,), dtype=np.float64)
    g = np.diff(x)
    g = g[np.isfinite(g) & (g > 0.0)]
    ms = float(np.mean(g)) if g.size else 1.0
    ms = ms if (math.isfinite(ms) and ms > 0.0) else 1.0
    return ((x - float(x[0])) / max(ms, 1e-12)).astype(np.float64, copy=False)


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
    """
    Sigma^2(L) = Var(N(t,t+L)) for sliding windows.
    Uses deterministic start-grid between min(u) and max(u)-L, counting via searchsorted.
    """
    u = np.sort(np.asarray(levels_unfolded, dtype=np.float64).reshape(-1))
    u = u[np.isfinite(u)]
    if u.size < 5:
        return np.full_like(L_grid, np.nan, dtype=np.float64)
    lo = float(u[0])
    hi = float(u[-1])
    span = hi - lo
    out = np.full_like(L_grid, np.nan, dtype=np.float64)
    for i, L in enumerate(np.asarray(L_grid, dtype=np.float64).reshape(-1)):
        Lf = float(L)
        if not (math.isfinite(Lf) and Lf > 0.0):
            continue
        if span <= Lf + 1e-9:
            continue
        n_starts = int(min(256, max(24, int(6 * span))))
        starts = np.linspace(lo, hi - Lf, num=n_starts)
        left = np.searchsorted(u, starts, side="left")
        right = np.searchsorted(u, starts + Lf, side="left")
        counts = (right - left).astype(np.float64)
        if counts.size < 2:
            continue
        out[i] = float(np.var(counts))
    return out


def poisson_ref_curve(L_grid: np.ndarray) -> np.ndarray:
    L = np.asarray(L_grid, dtype=np.float64)
    return np.maximum(L, 0.0)


def gue_ref_curve(L_grid: np.ndarray) -> np.ndarray:
    # simple safe log reference: (1/pi^2)*log(2*pi*L + e)
    L = np.asarray(L_grid, dtype=np.float64)
    return (1.0 / (math.pi**2)) * np.log(2.0 * math.pi * np.maximum(L, 1e-6) + math.e)


def normalize_curve(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if y.size == 0 or not np.isfinite(y).any():
        return y
    med = float(np.nanmedian(y))
    denom = med if (math.isfinite(med) and abs(med) > 1e-12) else float(np.nanmean(np.abs(y)) + 1e-12)
    denom = max(float(denom), 1e-12)
    return (y / denom).astype(np.float64, copy=False)


def region_masks(L_grid: np.ndarray, tail_L_min: float) -> Dict[str, np.ndarray]:
    L = np.asarray(L_grid, dtype=np.float64)
    return {
        "short": (L <= 2.0),
        "mid": (L > 2.0) & (L <= float(tail_L_min)),
        "long": (L > float(tail_L_min)),
    }


def l2_error(a: np.ndarray, b: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    if aa.size == 0 or bb.size == 0 or aa.size != bb.size:
        return float("nan")
    m = np.isfinite(aa) & np.isfinite(bb)
    if mask is not None:
        m = m & np.asarray(mask, dtype=bool).reshape(-1)
    if m.sum() < 3:
        return float("nan")
    d = aa[m] - bb[m]
    return float(np.sqrt(np.mean(d**2)))


def linreg_slope(x: np.ndarray, y: np.ndarray, mask: np.ndarray) -> float:
    xx = np.asarray(x, dtype=np.float64).reshape(-1)
    yy = np.asarray(y, dtype=np.float64).reshape(-1)
    mm = np.asarray(mask, dtype=bool).reshape(-1) & np.isfinite(xx) & np.isfinite(yy)
    if mm.sum() < 3:
        return float("nan")
    xs = xx[mm]
    ys = yy[mm]
    try:
        return float(np.polyfit(xs, ys, 1)[0])
    except Exception:
        return float("nan")


def curve_metric_total(
    op_curve: np.ndarray,
    tgt_curve: np.ndarray,
    *,
    L_grid: np.ndarray,
    metric: str,
    tail_L_min: float,
    normalize: bool,
) -> Tuple[float, float]:
    """
    Returns (total_curve_error, slope_error).
    """
    oc = normalize_curve(op_curve) if normalize or metric == "shape_l2" else np.asarray(op_curve, dtype=np.float64)
    tc = normalize_curve(tgt_curve) if normalize or metric == "shape_l2" else np.asarray(tgt_curve, dtype=np.float64)
    if metric in ("shape_l2", "raw_l2"):
        total = l2_error(oc, tc)
        # slope_error as auxiliary
        masks = region_masks(L_grid, tail_L_min)
        slopes_o = [linreg_slope(L_grid, oc, masks[k]) for k in ("short", "mid", "long")]
        slopes_t = [linreg_slope(L_grid, tc, masks[k]) for k in ("short", "mid", "long")]
        slope_err = float(np.sqrt(np.nanmean((np.asarray(slopes_o) - np.asarray(slopes_t)) ** 2)))
        return float(total), float(slope_err)
    if metric == "slope_l2":
        masks = region_masks(L_grid, tail_L_min)
        slopes_o = np.asarray([linreg_slope(L_grid, op_curve, masks[k]) for k in ("short", "mid", "long")], dtype=np.float64)
        slopes_t = np.asarray([linreg_slope(L_grid, tgt_curve, masks[k]) for k in ("short", "mid", "long")], dtype=np.float64)
        if not np.isfinite(slopes_o).any() or not np.isfinite(slopes_t).any():
            return float("nan"), float("nan")
        total = float(np.sqrt(np.nanmean((slopes_o - slopes_t) ** 2)))
        return total, total
    return float("nan"), float("nan")


def classify_failure_region(short_e: float, mid_e: float, long_e: float) -> str:
    vals = {"short": short_e, "mid": mid_e, "long": long_e}
    best = None
    bestv = -1.0
    for k, v in vals.items():
        if math.isfinite(v) and v > bestv:
            bestv = float(v)
            best = k
    return str(best or "unknown")


# -----------------------------
# V13O.4 module loader (reuse)
# -----------------------------


def load_v13o4_module() -> Any:
    path = Path(ROOT) / "scripts" / "run_v13o4_zeta_specific_objective.py"
    spec = importlib.util.spec_from_file_location("_v13o4_for_v13o8", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_v13o4_for_v13o8"] = mod
    spec.loader.exec_module(mod)
    return mod


# -----------------------------
# Job definition + worker
# -----------------------------


@dataclass(frozen=True)
class Job:
    dim: int
    V_mode: str
    word_group: str
    target_group: str
    control_id: int
    seed: int


_O4: Any = None
_Z_SORTED: Optional[np.ndarray] = None
_FRO_REF: Dict[int, float] = {}
_WORD_JOBS: Dict[str, Dict[str, Any]] = {}
_DIM_PARAM: Dict[int, Dict[str, Any]] = {}
_DIM_K_TRAIN: Dict[int, int] = {}


def _init_worker(o4_path: str, z_sorted: np.ndarray, fro_ref: Dict[int, float], word_jobs: Dict[str, Dict[str, Any]], dim_param: Dict[int, Dict[str, Any]], dim_k: Dict[int, int]) -> None:
    global _O4, _Z_SORTED, _FRO_REF, _WORD_JOBS, _DIM_PARAM, _DIM_K_TRAIN
    # load module in worker
    spec = importlib.util.spec_from_file_location("_v13o4_for_v13o8_worker", Path(o4_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {o4_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_v13o4_for_v13o8_worker"] = mod
    spec.loader.exec_module(mod)
    _O4 = mod
    _Z_SORTED = np.asarray(z_sorted, dtype=np.float64).reshape(-1)
    _FRO_REF = dict({int(k): float(v) for k, v in fro_ref.items()})
    _WORD_JOBS = dict(word_jobs)
    _DIM_PARAM = dict(dim_param)
    _DIM_K_TRAIN = dict({int(k): int(v) for k, v in dim_k.items()})
    # also set internal zeta array for any reused helpers
    try:
        _O4._v13o4_worker_init(_Z_SORTED)  # type: ignore[attr-defined]
    except Exception:
        pass


def _approx_levels_for_target(dim: int, target_group: str, seed: int, k: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed) + 1009 * int(dim) + 17 * (hash(target_group) % 10000))
    if target_group == "GUE_synthetic":
        a = rng.standard_normal((k, k))
        h = (a + a.T) / math.sqrt(2.0)
        w = np.sort(np.linalg.eigvalsh(h).astype(np.float64))
        return (w - float(w[0]) + 1e-6).astype(np.float64, copy=False)
    if target_group == "Poisson_synthetic":
        gaps = rng.exponential(1.0, size=(k,)).astype(np.float64)
        c = np.cumsum(gaps)
        return (c - float(c[0]) + 1e-6).astype(np.float64, copy=False)
    # shuffled/jitter/density/reversed proxies: Poisson-ish with mild smoothing
    gaps = rng.exponential(1.0, size=(k,)).astype(np.float64)
    if "block_shuffled" in target_group:
        # introduce block structure
        B = 4 if "block4" in target_group else 8
        for i in range(0, gaps.size, B):
            gaps[i : i + B] = np.sort(gaps[i : i + B])
    if "local_jitter" in target_group:
        amp = 0.05 if "small" in target_group else 0.15
        gaps = np.maximum(gaps * (1.0 + amp * rng.standard_normal(gaps.shape)), 1e-9)
    c = np.cumsum(gaps)
    return (c - float(c[0]) + 1e-6).astype(np.float64, copy=False)


def _approx_operator_levels_from_target(target_levels: np.ndarray, *, seed: int, dim: int, word_group: str) -> np.ndarray:
    """
    Proxy operator spectrum: slightly perturbed target levels (keeps scale) to allow curve computation.
    """
    rng = np.random.default_rng(int(seed) + 7919 + 104729 * int(dim) + (hash(word_group) % 100000))
    x = np.sort(np.asarray(target_levels, dtype=np.float64).reshape(-1))
    if x.size < 3:
        return x
    gaps = np.diff(x)
    noise = 0.02 * rng.standard_normal(gaps.shape)
    g2 = np.maximum(gaps * (1.0 + noise), 1e-9)
    y = np.concatenate([[x[0]], x[0] + np.cumsum(g2)])
    return y.astype(np.float64, copy=False)


def _worker_compute(job: Job, *, L_grid: np.ndarray, tail_L_min: float, curve_metric: str, normalize_curves: bool, true_mode: bool) -> Dict[str, Any]:
    """
    Returns a dict with:
      - levels rows (target/operator)
      - spacings rows
      - curves rows
      - error/region/poisson diag rows (for this target_group)
      - meta flags
    """
    dim = int(job.dim)
    vm = str(job.V_mode)
    wg = str(job.word_group)
    tg = str(job.target_group)
    seed = int(job.seed)
    control_id = int(job.control_id)

    k = int(_DIM_K_TRAIN.get(dim, dim))
    window_note = ""
    approximation_used = not bool(true_mode)

    if true_mode and _O4 is not None and _Z_SORTED is not None and wg in _WORD_JOBS:
        try:
            from core import v13l_self_consistent as v13l
            from core import v13o2_specificity as v2
            from core.artin_operator import sample_domain
        except Exception as e:
            approximation_used = True
            window_note = f"approx_fallback_import_error={e!r}"
        else:
            try:
                z_sorted = np.asarray(_Z_SORTED, dtype=np.float64).reshape(-1)
                if z_sorted.size < 2 * k:
                    raise RuntimeError(f"Need {2*k} zeros; got {z_sorted.size}")
                cfg = dict(_DIM_PARAM.get(dim) or {})
                if not cfg:
                    cfg = dict(getattr(_O4, "DIM_PARAM", {}).get(dim) or {})
                fro_ref = float(_FRO_REF.get(dim, 1.0))
                Z = sample_domain(dim, seed=seed)
                wj = dict(_WORD_JOBS[wg])

                # build target train/test windows using same helper as V13O.4
                # choose deterministic target_rep from control_id for replica-capable targets
                target_rep = int(control_id)
                tr, te, train_ordered, wnote = _O4.build_train_test_targets(tg, z_sorted, k, seed, target_rep)  # type: ignore[attr-defined]
                window_note = str(wnote)
                z_train = np.sort(np.asarray(tr, dtype=np.float64).reshape(-1))
                z_test = np.sort(np.asarray(te, dtype=np.float64).reshape(-1))
                k_al = int(min(dim, k, z_train.size, z_test.size))
                z_test = z_test[:k_al]
                z_train = z_train[:k_al]

                kind = str(wj.get("kind", "word"))
                if kind == "sym":
                    sym_index = int(wj.get("sym_index", 0))
                    r = np.random.default_rng(seed + dim * 104729 + sym_index).standard_normal((dim, dim))
                    S = 0.5 * (r + r.T)
                    fs = float(np.linalg.norm(S, ord="fro"))
                    H_op = (S * (fro_ref / max(fs, 1e-12))).astype(np.float64, copy=False)
                elif kind == "ablate_V":
                    H_op = _O4.build_h(  # type: ignore[attr-defined]
                        z_points=Z,
                        word=list(wj.get("word") or []),
                        v13l=v13l,
                        geo_sigma=float(cfg.get("geo_sigma", 0.75)),
                        laplacian_weight=float(wj.get("lap", 1.0)),
                        geo_weight=float(wj.get("geo_w", 1.0)),
                    )
                else:
                    H_op = _O4.build_h(  # type: ignore[attr-defined]
                        z_points=Z,
                        word=list(wj.get("word") or []),
                        v13l=v13l,
                        geo_sigma=float(cfg.get("geo_sigma", 0.75)),
                        laplacian_weight=float(wj.get("lap", 1.0)),
                        geo_weight=float(wj.get("geo_w", 1.0)),
                    )

                smooth_use = float(cfg.get("smooth_sigma", 0.0))
                clip_lo = float(cfg.get("clip_lo", -3.0))
                clip_hi = float(cfg.get("clip_hi", 3.0))

                # train operator (skipping for ablate_V mirrors V13O.4)
                if kind == "ablate_V":
                    H_fin = np.asarray(H_op, dtype=np.float64, copy=True)
                else:
                    train_out = v2.train_v13o2_cell(
                        H_base=np.asarray(H_op, dtype=np.float64, copy=True),
                        z_pool_sorted=z_sorted.astype(np.float64, copy=False),
                        train_ordinates=z_train.astype(np.float64, copy=False),
                        train_ordered=bool(train_ordered),
                        dim=dim,
                        k_train=k,
                        alpha=float(getattr(_O4, "ALPHA", 0.10)),
                        lambda_p_dim=float(cfg.get("lambda_p", 0.02)),
                        beta0=float(cfg.get("beta0", 0.2)),
                        tau_beta=float(cfg.get("tau", 1.6)),
                        beta_floor=float(cfg.get("beta_floor", 0.02)),
                        smooth_sigma_dim=float(smooth_use),
                        clip_lo=float(clip_lo),
                        clip_hi=float(clip_hi),
                        diag_shift=float(getattr(_O4, "DIAG_SHIFT", 0.0)),
                        abs_cap_factor=float(getattr(_O4, "ABS_CAP", 5.0)),
                        zeros_train_metric=z_train.astype(np.float64, copy=False),
                        spacing_fn=_O4.v.spacing_mse_normalized,  # type: ignore[attr-defined]
                        ks_fn=_O4.v.ks_against_wigner_gue,  # type: ignore[attr-defined]
                        norm_gaps_fn=_O4.v.normalized_gaps,  # type: ignore[attr-defined]
                        max_iter=int(cfg.get("max_iter", 60)),
                        tol=float(cfg.get("tol", 1e-6)),
                        v_mode=vm,
                        z_points=np.asarray(Z, dtype=np.complex128),
                        word=list(wj.get("word") or []),
                        on_train_iter=None,
                    )
                    H_fin = np.asarray(train_out.get("H_final"), dtype=np.float64, copy=True)

                eig = np.sort(np.linalg.eigvalsh(0.5 * (H_fin + H_fin.T))).astype(np.float64)
                eig = eig[:k_al]
                approximation_used = False
                # now compute unfolded and curves for target/operator
                target_levels = z_test
                operator_levels = eig
            except Exception as e:
                approximation_used = True
                window_note = f"approx_fallback_compute_error={e!r}"
    if approximation_used:
        target_levels = _approx_levels_for_target(dim, tg, seed + 97 * control_id, k)
        operator_levels = _approx_operator_levels_from_target(target_levels, seed=seed + 97 * control_id, dim=dim, word_group=wg)
        window_note = window_note or "approximation_mode_reconstructed_levels"
        k_al = int(min(target_levels.size, operator_levels.size))
        target_levels = np.sort(target_levels)[:k_al]
        operator_levels = np.sort(operator_levels)[:k_al]

    # unfold
    u_target = unfold_to_mean_spacing_one(target_levels)
    u_op = unfold_to_mean_spacing_one(operator_levels)
    # spacing (unfolded spacing is also meaningful; but schema expects spacing only)
    sp_target = np.diff(np.sort(u_target)).astype(np.float64) if u_target.size >= 2 else np.asarray([], dtype=np.float64)
    sp_op = np.diff(np.sort(u_op)).astype(np.float64) if u_op.size >= 2 else np.asarray([], dtype=np.float64)

    # curves
    c_target = number_variance_curve(u_target, L_grid)
    c_op = number_variance_curve(u_op, L_grid)
    if normalize_curves:
        c_target_n = normalize_curve(c_target)
        c_op_n = normalize_curve(c_op)
    else:
        c_target_n = c_target
        c_op_n = c_op

    # errors by region
    masks = region_masks(L_grid, tail_L_min)
    short_e = l2_error(c_op_n, c_target_n, masks["short"])
    mid_e = l2_error(c_op_n, c_target_n, masks["mid"])
    long_e = l2_error(c_op_n, c_target_n, masks["long"])
    total_e, slope_e = curve_metric_total(c_op, c_target, L_grid=L_grid, metric=str(curve_metric), tail_L_min=tail_L_min, normalize=normalize_curves)

    # poisson/gue diagnostics (compare target curve only; operator is judged relative to target too)
    pois = poisson_ref_curve(L_grid)
    gue = gue_ref_curve(L_grid)
    slope_short = linreg_slope(L_grid, c_op_n, masks["short"])
    slope_mid = linreg_slope(L_grid, c_op_n, masks["mid"])
    slope_long = linreg_slope(L_grid, c_op_n, masks["long"])
    poisson_slope_short = 1.0
    poisson_slope_mid = 1.0
    poisson_slope_long = 1.0
    poisson_slope_err = float(np.sqrt(np.nanmean((np.asarray([slope_short, slope_mid, slope_long]) - 1.0) ** 2)))
    gue_err = l2_error(normalize_curve(c_op) if normalize_curves else c_op, normalize_curve(gue) if normalize_curves else gue, masks["long"])
    pois_err = l2_error(normalize_curve(c_op) if normalize_curves else c_op, normalize_curve(pois) if normalize_curves else pois, masks["long"])
    # scores in [0,1] (lower distance wins)
    gue_like_score = float(1.0 / (1.0 + max(0.0, float(gue_err)))) if math.isfinite(gue_err) else float("nan")
    poisson_like_score = float(1.0 / (1.0 + max(0.0, float(pois_err)))) if math.isfinite(pois_err) else float("nan")

    # pack rows
    levels_rows: List[Dict[str, Any]] = []
    for idx, (raw, un) in enumerate(zip(target_levels.tolist(), u_target.tolist())):
        levels_rows.append(
            {
                "dim": dim,
                "V_mode": vm,
                "word_group": wg,
                "target_group": tg,
                "seed": seed,
                "kind": "target",
                "level_index": int(idx),
                "raw_level": float(raw),
                "unfolded_level": float(un),
                "window_note": window_note,
            }
        )
    for idx, (raw, un) in enumerate(zip(operator_levels.tolist(), u_op.tolist())):
        levels_rows.append(
            {
                "dim": dim,
                "V_mode": vm,
                "word_group": wg,
                "target_group": tg,
                "seed": seed,
                "kind": "operator",
                "level_index": int(idx),
                "raw_level": float(raw),
                "unfolded_level": float(un),
                "window_note": window_note,
            }
        )

    spacing_rows: List[Dict[str, Any]] = []
    for idx, sp in enumerate(sp_target.tolist()):
        spacing_rows.append(
            {
                "dim": dim,
                "V_mode": vm,
                "word_group": wg,
                "target_group": tg,
                "seed": seed,
                "kind": "target",
                "spacing_index": int(idx),
                "spacing": float(sp),
            }
        )
    for idx, sp in enumerate(sp_op.tolist()):
        spacing_rows.append(
            {
                "dim": dim,
                "V_mode": vm,
                "word_group": wg,
                "target_group": tg,
                "seed": seed,
                "kind": "operator",
                "spacing_index": int(idx),
                "spacing": float(sp),
            }
        )

    curves_rows: List[Dict[str, Any]] = []
    for L, sig2 in zip(L_grid.tolist(), c_target.tolist()):
        curves_rows.append(
            {"dim": dim, "V_mode": vm, "word_group": wg, "target_group": tg, "seed": seed, "kind": "target", "L": float(L), "Sigma2": float(sig2)}
        )
    for L, sig2 in zip(L_grid.tolist(), c_op.tolist()):
        curves_rows.append(
            {"dim": dim, "V_mode": vm, "word_group": wg, "target_group": tg, "seed": seed, "kind": "operator", "L": float(L), "Sigma2": float(sig2)}
        )

    errors_row = {
        "dim": dim,
        "V_mode": vm,
        "word_group": wg,
        "target_group": tg,
        "seed": seed,
        "curve_metric": str(curve_metric),
        "total_curve_error": float(total_e),
        "short_error": float(short_e),
        "mid_error": float(mid_e),
        "long_error": float(long_e),
        "slope_error": float(slope_e),
        "poisson_slope_error": float(poisson_slope_err),
        "gue_rigidity_error": float(gue_err),
        "accepted_transfer": True,  # patched later if inputs provide transfer info
    }

    regions_row = {
        "dim": dim,
        "V_mode": vm,
        "word_group": wg,
        "target_group": tg,
        "seed": seed,
        "short_error": float(short_e),
        "mid_error": float(mid_e),
        "long_error": float(long_e),
        "short_percentile": float("nan"),
        "mid_percentile": float("nan"),
        "long_percentile": float("nan"),
        "failure_region": classify_failure_region(short_e, mid_e, long_e),
    }

    poisson_row = {
        "dim": dim,
        "V_mode": vm,
        "word_group": wg,
        "target_group": tg,
        "seed": seed,
        "slope_short": float(slope_short),
        "slope_mid": float(slope_mid),
        "slope_long": float(slope_long),
        "poisson_slope_short": float(poisson_slope_short),
        "poisson_slope_mid": float(poisson_slope_mid),
        "poisson_slope_long": float(poisson_slope_long),
        "gue_like_score": float(gue_like_score),
        "poisson_like_score": float(poisson_like_score),
        "classification": "POISSON_LIKE" if (math.isfinite(poisson_like_score) and math.isfinite(gue_like_score) and poisson_like_score > gue_like_score) else "GUE_LIKE",
    }

    return {
        "job": {"dim": dim, "V_mode": vm, "word_group": wg, "target_group": tg, "seed": seed, "control_id": control_id},
        "approximation_used": bool(approximation_used),
        "levels_rows": levels_rows,
        "spacing_rows": spacing_rows,
        "curves_rows": curves_rows,
        "errors_row": errors_row,
        "regions_row": regions_row,
        "poisson_row": poisson_row,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="V13O.8 explicit unfolded spectra + true NV curves (computational only).")

    ap.add_argument("--v13o4_summary", type=str, default="runs/v13o4_zeta_specific_objective/v13o4_summary.csv")
    ap.add_argument("--v13o4_group_summary", type=str, default="runs/v13o4_zeta_specific_objective/v13o4_group_summary.csv")
    ap.add_argument("--v13o4_zeta_scores", type=str, default="runs/v13o4_zeta_specific_objective/v13o4_zeta_specific_scores.csv")
    ap.add_argument("--v13o4_number_variance", type=str, default="runs/v13o4_zeta_specific_objective/v13o4_number_variance_summary.csv")
    ap.add_argument("--v13o4_pair_corr", type=str, default="runs/v13o4_zeta_specific_objective/v13o4_pair_corr_summary.csv")
    ap.add_argument("--v13o4_staircase", type=str, default="runs/v13o4_zeta_specific_objective/v13o4_staircase_summary.csv")
    ap.add_argument("--v13o4_ensemble_margins", type=str, default="runs/v13o4_zeta_specific_objective/v13o4_ensemble_margins.csv")
    ap.add_argument("--v13o6_nv_scores", type=str, default="runs/v13o6_number_variance_stabilization_smoke/v13o6_nv_stabilized_scores.csv")
    ap.add_argument("--v13o7_curve_summary", type=str, default="")

    ap.add_argument("--candidate_json", type=str, default="")
    ap.add_argument("--formula_json", type=str, default="")

    ap.add_argument("--out_dir", type=str, default="runs/v13o8_explicit_unfolded_spectra")
    ap.add_argument("--primary_word_group", type=str, default="primary_word_seed6")
    ap.add_argument("--primary_v_mode", type=str, default="full_V")
    ap.add_argument("--dims", type=int, nargs="+", default=[64, 128, 256])

    ap.add_argument("--L_min", type=float, default=0.5)
    ap.add_argument("--L_max", type=float, default=12.0)
    ap.add_argument("--n_L", type=int, default=48)
    ap.add_argument("--tail_L_min", type=float, default=6.0)
    ap.add_argument("--curve_metric", type=str, default="shape_l2", choices=["shape_l2", "raw_l2", "slope_l2"])
    ap.add_argument("--normalize_curves", action="store_true")

    ap.add_argument("--n_jobs", type=int, default=1)
    ap.add_argument("--progress_every", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    L_grid = make_L_grid(args.L_min, args.L_max, args.n_L)

    # Load summaries (optional)
    df_scores = read_csv_robust(_resolve(args.v13o4_zeta_scores), name="v13o4_zeta_scores", tag="v13o8")
    df_nv = read_csv_robust(_resolve(args.v13o4_number_variance), name="v13o4_number_variance", tag="v13o8")
    df_v13o6 = read_csv_robust(_resolve(args.v13o6_nv_scores), name="v13o6_nv_scores", tag="v13o8")
    _ = read_csv_robust(_resolve(args.v13o4_summary), name="v13o4_summary", tag="v13o8")
    _ = read_csv_robust(_resolve(args.v13o4_group_summary), name="v13o4_group_summary", tag="v13o8")
    _ = read_csv_robust(_resolve(args.v13o4_pair_corr), name="v13o4_pair_corr", tag="v13o8")
    _ = read_csv_robust(_resolve(args.v13o4_staircase), name="v13o4_staircase", tag="v13o8")
    _ = read_csv_robust(_resolve(args.v13o4_ensemble_margins), name="v13o4_ensemble_margins", tag="v13o8")
    df_v13o7 = read_csv_robust(_resolve(args.v13o7_curve_summary), name="v13o7_curve_summary", tag="v13o8") if str(args.v13o7_curve_summary).strip() else None

    # accepted_transfer map from v13o4_zeta_scores if available
    accepted_transfer_map: Dict[Tuple[int, str, str, str], bool] = {}
    accepted_real_map: Dict[Tuple[int, str, str], bool] = {}
    if df_scores is not None and all(c in df_scores.columns for c in ("dim", "V_mode", "word_group", "target_group")):
        dfx = df_scores.copy()
        dfx["dim"] = pd.to_numeric(dfx["dim"], errors="coerce").astype("Int64")
        for k in ("V_mode", "word_group", "target_group"):
            dfx[k] = dfx[k].astype(str).str.strip()
        pen_col = pick_col(dfx, ["transfer_gap_penalty"])
        acc_col = pick_col(dfx, ["accepted_basic", "accepted"])
        for (d, vm, wg, tg), g in dfx.groupby(["dim", "V_mode", "word_group", "target_group"]):
            ok = True
            if pen_col:
                pen = safe_median(finite_series(g[pen_col]).dropna().astype(float).tolist())
                ok = bool(math.isfinite(pen) and pen <= 1.0)
            if acc_col:
                acc_rate = float(np.mean(g[acc_col].astype(bool).to_numpy())) if acc_col in g.columns else 1.0
                ok = ok and bool(acc_rate >= 0.5)
            accepted_transfer_map[(int(d), str(vm), str(wg), str(tg))] = bool(ok)
            if str(tg) == "real_zeta":
                accepted_real_map[(int(d), str(vm), str(wg))] = bool(ok)

    # Decide true_mode availability
    candidate_path = _resolve(args.candidate_json) if str(args.candidate_json).strip() else None
    approximation_mode = True
    true_mode = False
    o4 = None
    z_sorted = None

    primary_word: List[int] = []
    rejected_word: List[int] = []

    if candidate_path is not None and candidate_path.is_file():
        try:
            cand = json.loads(candidate_path.read_text(encoding="utf-8"))
            # mirror V13O.4 naming
            primary_word = [int(x) for x in (cand.get("primary_candidate", {}) or {}).get("word") or []]
            for rc in cand.get("rejected_candidates") or []:
                if rc.get("id") == "seed_17":
                    rejected_word = [int(x) for x in rc.get("word") or []]
                    break
        except Exception as e:
            print(f"[v13o8] WARNING failed reading candidate_json={candidate_path}: {e!r}", flush=True)

    # attempt true mode if we can load v13o4 module + zeros + have a primary word
    if primary_word:
        try:
            o4 = load_v13o4_module()
            v13_validate = o4.load_v13_validate()
            dims = [int(d) for d in args.dims]
            # use DIM_K_TRAIN from v13o4 for consistent windows
            dim_k = {int(d): int(getattr(o4, "DIM_K_TRAIN", {}).get(int(d), int(d))) for d in dims}
            max_k = max(dim_k.values()) if dim_k else max(dims)
            z_pool = v13_validate._load_zeros(max(512, 2 * max_k))
            z_sorted = np.sort(np.asarray(z_pool, dtype=np.float64).reshape(-1))
            z_sorted = z_sorted[np.isfinite(z_sorted) & (z_sorted > 0.0)]
            approximation_mode = False
            true_mode = True
        except Exception as e:
            print(f"[v13o8] WARNING: cannot enable true_mode; falling back to approximation_mode. err={e!r}", flush=True)
            approximation_mode = True
            true_mode = False

    # build word jobs config (match V13O.4 shapes)
    base_seed = int(args.seed)
    rng = np.random.default_rng(base_seed + 7919)
    if not rejected_word:
        rejected_word = [1, -2, 3, -4, 5, -6]  # fallback; only used in approximation mode or missing candidate fields
    if not primary_word:
        primary_word = [1, 1, -1, 2, -2, 3]  # fallback

    # random word (single)
    alphabet = list(range(-6, 0)) + list(range(1, 7))
    rw = [int(rng.choice(alphabet)) for _ in range(len(primary_word))]

    # load GEO_WEIGHT from v13o4 if present
    GEO_WEIGHT = float(getattr(o4, "GEO_WEIGHT", 1.0)) if o4 is not None else 1.0

    word_jobs: Dict[str, Dict[str, Any]] = {
        "primary_word_seed6": {"id": "primary_word_seed6", "word": list(primary_word), "lap": 1.0, "geo_w": GEO_WEIGHT, "kind": "word"},
        "rejected_word_seed17": {"id": "rejected_word_seed17", "word": list(rejected_word), "lap": 1.0, "geo_w": GEO_WEIGHT, "kind": "word"},
        "ablate_K": {"id": "ablate_K", "word": list(primary_word), "lap": 1.0, "geo_w": 0.0, "kind": "word"},
        "ablate_V": {"id": "ablate_V", "word": list(primary_word), "lap": 1.0, "geo_w": GEO_WEIGHT, "kind": "ablate_V"},
        "ablate_L": {"id": "ablate_L", "word": list(primary_word), "lap": 0.0, "geo_w": GEO_WEIGHT, "kind": "word"},
        "random_words_n30": {"id": "random_words_n30", "word": list(rw), "lap": 1.0, "geo_w": GEO_WEIGHT, "kind": "word"},
        "random_symmetric_baseline": {"id": "random_symmetric_baseline", "sym_index": 0, "kind": "sym"},
    }

    # dim params + fro refs for true mode
    dim_param: Dict[int, Dict[str, Any]] = {}
    dim_k_train: Dict[int, int] = {}
    fro_ref: Dict[int, float] = {}
    if o4 is not None:
        dim_param = {int(k): dict(v) for k, v in (getattr(o4, "DIM_PARAM", {}) or {}).items()}
        dim_k_train = {int(k): int(v) for k, v in (getattr(o4, "DIM_K_TRAIN", {}) or {}).items()}
    for d in [int(x) for x in args.dims]:
        dim_k_train.setdefault(int(d), int(d))
        dim_param.setdefault(int(d), {"geo_sigma": 0.75, "lambda_p": 0.02, "beta0": 0.2, "tau": 1.6, "beta_floor": 0.02, "smooth_sigma": 0.0, "clip_lo": -3.0, "clip_hi": 3.0, "max_iter": 60, "tol": 1e-6})

    if true_mode and o4 is not None:
        try:
            from core import v13l_self_consistent as v13l
            from core.artin_operator import sample_domain

            for d in [int(x) for x in args.dims]:
                Zpre = sample_domain(int(d), seed=base_seed)
                cfg_pre = dim_param[int(d)]
                H0 = o4.build_h(  # type: ignore[attr-defined]
                    z_points=Zpre,
                    word=list(primary_word),
                    v13l=v13l,
                    geo_sigma=float(cfg_pre.get("geo_sigma", 0.75)),
                    laplacian_weight=1.0,
                    geo_weight=float(GEO_WEIGHT),
                )
                fro_ref[int(d)] = float(np.linalg.norm(np.asarray(H0, dtype=np.float64), ord="fro"))
        except Exception as e:
            print(f"[v13o8] WARNING fro_ref computation failed; true_mode may degrade. err={e!r}", flush=True)
            for d in [int(x) for x in args.dims]:
                fro_ref[int(d)] = 1.0

    # Build job list: (dim, V_mode, word_group, target_group, seed)
    jobs: List[Job] = []
    base = int(args.seed)
    for dim in [int(x) for x in args.dims]:
        for wg in CANDIDATE_WORD_GROUPS:
            if wg not in word_jobs:
                continue
            for tg in TARGET_GROUPS_ALL:
                # replica count: mimic V13O.4 behavior (controls have replicas, real/reversed deterministic)
                reps = 1 if tg in ("real_zeta", "reversed_zeta") else 4
                for rep in range(reps):
                    seed_job = int(base + 1000003 * dim + 1009 * rep + (abs(hash((wg, tg))) % 100000))
                    jobs.append(Job(dim=dim, V_mode=str(args.primary_v_mode), word_group=wg, target_group=tg, control_id=int(rep), seed=seed_job))
    jobs = sorted(jobs, key=lambda j: (j.dim, j.V_mode, j.word_group, j.target_group, j.control_id))
    total_jobs = len(jobs)

    print(f"[v13o8] jobs={total_jobs} true_mode={true_mode} approximation_mode={approximation_mode} n_jobs={int(args.n_jobs)}", flush=True)

    # Execute
    levels_rows_all: List[Dict[str, Any]] = []
    spacing_rows_all: List[Dict[str, Any]] = []
    curves_rows_all: List[Dict[str, Any]] = []
    errors_rows_all: List[Dict[str, Any]] = []
    regions_rows_all: List[Dict[str, Any]] = []
    poisson_rows_all: List[Dict[str, Any]] = []
    approx_used_any = False

    prog = max(1, int(args.progress_every))
    exec_t0 = time.perf_counter()

    def log_done(i_done: int) -> None:
        if i_done == 1 or i_done == total_jobs or i_done % prog == 0:
            elapsed = time.perf_counter() - exec_t0
            avg = elapsed / max(i_done, 1)
            eta = avg * max(total_jobs - i_done, 0)
            print(f"[V13O.8] completed {i_done}/{total_jobs} elapsed={elapsed:.1f}s eta={format_seconds(eta)}", flush=True)

    n_jobs = max(1, int(args.n_jobs))
    if n_jobs == 1:
        # init globals in-process if true_mode
        if true_mode and z_sorted is not None:
            _init_worker(str(Path(ROOT) / "scripts" / "run_v13o4_zeta_specific_objective.py"), z_sorted, fro_ref, word_jobs, dim_param, dim_k_train)
        for i, job in enumerate(jobs, start=1):
            out = _worker_compute(
                job,
                L_grid=L_grid,
                tail_L_min=float(args.tail_L_min),
                curve_metric=str(args.curve_metric),
                normalize_curves=bool(args.normalize_curves),
                true_mode=bool(true_mode),
            )
            approx_used_any = approx_used_any or bool(out.get("approximation_used"))
            levels_rows_all.extend(out["levels_rows"])
            spacing_rows_all.extend(out["spacing_rows"])
            curves_rows_all.extend(out["curves_rows"])
            errors_rows_all.append(out["errors_row"])
            regions_rows_all.append(out["regions_row"])
            poisson_rows_all.append(out["poisson_row"])
            log_done(i)
    else:
        # parallel: prefer threads in approximation_mode (avoids OS semaphore/sysconf restrictions)
        # and processes only when true_mode is enabled (heavy compute).
        use_processes = bool(true_mode)
        Executor = cf.ProcessPoolExecutor if use_processes else cf.ThreadPoolExecutor

        ex_kwargs: Dict[str, Any] = {"max_workers": n_jobs}
        if use_processes:
            o4_path = str(Path(ROOT) / "scripts" / "run_v13o4_zeta_specific_objective.py")
            z_arg = z_sorted if z_sorted is not None else np.asarray([], dtype=np.float64)
            ex_kwargs.update(
                {
                    "initializer": _init_worker,
                    "initargs": (o4_path, z_arg, fro_ref, word_jobs, dim_param, dim_k_train),
                }
            )

        try:
            with Executor(**ex_kwargs) as ex:  # type: ignore[arg-type]
                futs = []
                for job in jobs:
                    futs.append(
                        ex.submit(
                            _worker_compute,
                            job,
                            L_grid=L_grid,
                            tail_L_min=float(args.tail_L_min),
                            curve_metric=str(args.curve_metric),
                            normalize_curves=bool(args.normalize_curves),
                            true_mode=bool(true_mode),
                        )
                    )
                done = 0
                for fut in cf.as_completed(futs):
                    out = fut.result()
                    approx_used_any = approx_used_any or bool(out.get("approximation_used"))
                    levels_rows_all.extend(out["levels_rows"])
                    spacing_rows_all.extend(out["spacing_rows"])
                    curves_rows_all.extend(out["curves_rows"])
                    errors_rows_all.append(out["errors_row"])
                    regions_rows_all.append(out["regions_row"])
                    poisson_rows_all.append(out["poisson_row"])
                    done += 1
                    log_done(done)
        except Exception as e:
            print(f"[v13o8] WARNING parallel executor failed ({e!r}); falling back to n_jobs=1.", flush=True)
            for i, job in enumerate(jobs, start=1):
                out = _worker_compute(
                    job,
                    L_grid=L_grid,
                    tail_L_min=float(args.tail_L_min),
                    curve_metric=str(args.curve_metric),
                    normalize_curves=bool(args.normalize_curves),
                    true_mode=bool(true_mode),
                )
                approx_used_any = approx_used_any or bool(out.get("approximation_used"))
                levels_rows_all.extend(out["levels_rows"])
                spacing_rows_all.extend(out["spacing_rows"])
                curves_rows_all.extend(out["curves_rows"])
                errors_rows_all.append(out["errors_row"])
                regions_rows_all.append(out["regions_row"])
                poisson_rows_all.append(out["poisson_row"])
                log_done(i)

    # DataFrames and deterministic ordering
    df_levels = pd.DataFrame(levels_rows_all)
    df_spacing = pd.DataFrame(spacing_rows_all)
    df_curves = pd.DataFrame(curves_rows_all)
    df_err = pd.DataFrame(errors_rows_all)
    df_regions = pd.DataFrame(regions_rows_all)
    df_pois = pd.DataFrame(poisson_rows_all)

    for df in (df_levels, df_spacing, df_curves, df_err, df_regions, df_pois):
        if "dim" in df.columns:
            df["dim"] = pd.to_numeric(df["dim"], errors="coerce").astype("Int64")
        for k in ("V_mode", "word_group", "target_group", "kind"):
            if k in df.columns:
                df[k] = df[k].astype(str).str.strip()

    if not df_levels.empty:
        df_levels = df_levels.sort_values(["dim", "V_mode", "word_group", "target_group", "seed", "kind", "level_index"], ascending=True, na_position="last")
    if not df_spacing.empty:
        df_spacing = df_spacing.sort_values(["dim", "V_mode", "word_group", "target_group", "seed", "kind", "spacing_index"], ascending=True, na_position="last")
    if not df_curves.empty:
        df_curves = df_curves.sort_values(["dim", "V_mode", "word_group", "target_group", "seed", "kind", "L"], ascending=True, na_position="last")
    if not df_err.empty:
        df_err = df_err.sort_values(["dim", "V_mode", "word_group", "target_group", "seed"], ascending=True, na_position="last")
    if not df_regions.empty:
        df_regions = df_regions.sort_values(["dim", "V_mode", "word_group", "target_group", "seed"], ascending=True, na_position="last")
    if not df_pois.empty:
        df_pois = df_pois.sort_values(["dim", "V_mode", "word_group", "target_group", "seed"], ascending=True, na_position="last")

    # Patch accepted_transfer from map if present
    if not df_err.empty:
        acc = []
        for r in df_err.itertuples(index=False):
            key = (int(getattr(r, "dim")), str(getattr(r, "V_mode")), str(getattr(r, "word_group")), str(getattr(r, "target_group")))
            acc.append(bool(accepted_transfer_map.get(key, True)))
        df_err["accepted_transfer"] = acc

    # Percentiles vs controls (within each dim,vm,wg,seed): compare each region error against control distribution
    if not df_regions.empty:
        df_regions2 = df_regions.copy()
        df_regions2["short_percentile"] = np.nan
        df_regions2["mid_percentile"] = np.nan
        df_regions2["long_percentile"] = np.nan
        for (d, vm, wg, seed), g in df_regions2.groupby(["dim", "V_mode", "word_group", "seed"]):
            ctrl = g[g["target_group"].isin(CONTROL_TARGET_GROUPS)]
            for region in ("short", "mid", "long"):
                col = f"{region}_error"
                ctrl_vals = _finite_arr(ctrl[col].astype(float).tolist())
                if ctrl_vals.size == 0:
                    continue
                for idx, row in g.iterrows():
                    v = float(row[col])
                    if math.isfinite(v):
                        pct = float(np.mean(ctrl_vals <= v))
                        df_regions2.at[idx, f"{region}_percentile"] = pct
        df_regions = df_regions2

    # Gate summary per (dim, V_mode, word_group) based on real_zeta vs controls
    gate_rows: List[Dict[str, Any]] = []
    if not df_err.empty:
        for (d, vm, wg), g in df_err.groupby(["dim", "V_mode", "word_group"]):
            real = g[g["target_group"] == "real_zeta"]
            ctrl = g[g["target_group"].isin(CONTROL_TARGET_GROUPS)]
            real_e = safe_median(real["total_curve_error"].astype(float).tolist())
            ctrl_med = safe_median(ctrl["total_curve_error"].astype(float).tolist())
            # region pass thresholds: require region error <= median(control region error)
            reg_g = df_regions[(df_regions["dim"] == d) & (df_regions["V_mode"] == vm) & (df_regions["word_group"] == wg)]
            reg_real = reg_g[reg_g["target_group"] == "real_zeta"]
            reg_ctrl = reg_g[reg_g["target_group"].isin(CONTROL_TARGET_GROUPS)]
            s_short = safe_median(reg_real["short_error"].astype(float).tolist())
            s_mid = safe_median(reg_real["mid_error"].astype(float).tolist())
            s_long = safe_median(reg_real["long_error"].astype(float).tolist())
            c_short = safe_median(reg_ctrl["short_error"].astype(float).tolist())
            c_mid = safe_median(reg_ctrl["mid_error"].astype(float).tolist())
            c_long = safe_median(reg_ctrl["long_error"].astype(float).tolist())

            G1 = bool(math.isfinite(real_e))
            G2 = bool(True)  # unfolding-valid check is implicit if levels finite; patched below with levels count
            G3 = bool(math.isfinite(real_e) and math.isfinite(ctrl_med) and real_e < ctrl_med)
            G4 = bool(math.isfinite(s_short) and math.isfinite(c_short) and s_short <= c_short)
            G5 = bool(math.isfinite(s_mid) and math.isfinite(c_mid) and s_mid <= c_mid)
            G6 = bool(math.isfinite(s_long) and math.isfinite(c_long) and s_long <= c_long)

            # Poisson/GUE from poisson diagnostics table on real_zeta
            psub = df_pois[(df_pois["dim"] == d) & (df_pois["V_mode"] == vm) & (df_pois["word_group"] == wg) & (df_pois["target_group"] == "real_zeta")]
            poisson_like_score = safe_median(psub["poisson_like_score"].astype(float).tolist()) if not psub.empty else float("nan")
            gue_like_score = safe_median(psub["gue_like_score"].astype(float).tolist()) if not psub.empty else float("nan")
            G7 = bool(math.isfinite(poisson_like_score) and math.isfinite(gue_like_score) and poisson_like_score <= gue_like_score)
            G8 = bool(math.isfinite(gue_like_score) and gue_like_score >= 0.50)

            # transfer + acceptance
            G9 = bool(accepted_transfer_map.get((int(d), str(vm), str(wg), "real_zeta"), True))
            G10 = bool(accepted_real_map.get((int(d), str(vm), str(wg)), True))

            # level finiteness checks from exported levels
            lev = df_levels[(df_levels["dim"] == d) & (df_levels["V_mode"] == vm) & (df_levels["word_group"] == wg) & (df_levels["target_group"] == "real_zeta")]
            finite_levels = bool(not lev.empty and np.isfinite(lev["unfolded_level"].astype(float)).all())
            G1 = G1 and finite_levels
            G2 = finite_levels

            strict = bool(G1 and G2 and G3 and G4 and G5 and G6 and G7 and G8 and G9 and G10)
            relaxed = bool(G3 and (sum(1 for x in (G4, G5, G6) if x) >= 2) and (not (math.isfinite(poisson_like_score) and poisson_like_score > 0.75)))

            # classification
            if approximation_mode or approx_used_any:
                cls = "APPROXIMATION_MODE_ONLY"
            elif strict:
                cls = "TRUE_CURVE_SIGNAL"
            elif relaxed:
                cls = "PARTIAL_CURVE_SIGNAL"
            else:
                if not G7:
                    cls = "POISSONIZATION_FAILURE"
                else:
                    worst = classify_failure_region(s_short, s_mid, s_long)
                    cls = {"short": "SHORT_RANGE_FAILURE", "mid": "MID_RANGE_FAILURE", "long": "LONG_RANGE_FAILURE"}.get(worst, "NO_CURVE_SIGNAL")

            gate_rows.append(
                {
                    "dim": int(d),
                    "V_mode": str(vm),
                    "word_group": str(wg),
                    "G1_finite_levels": bool(G1),
                    "G2_unfolding_valid": bool(G2),
                    "G3_curve_error_beats_controls": bool(G3),
                    "G4_short_range_pass": bool(G4),
                    "G5_mid_range_pass": bool(G5),
                    "G6_long_range_pass": bool(G6),
                    "G7_not_poisson_like": bool(G7),
                    "G8_gue_rigidity_pass": bool(G8),
                    "G9_transfer_gap_ok": bool(G9),
                    "G10_real_accepted": bool(G10),
                    "strict_curve_specificity_pass": bool(strict),
                    "relaxed_curve_specificity_pass": bool(relaxed),
                    "total_curve_error": float(real_e),
                    "median_control_curve_error": float(ctrl_med),
                    "poisson_like_score": float(poisson_like_score),
                    "gue_like_score": float(gue_like_score),
                    "best_failure_region": classify_failure_region(s_short, s_mid, s_long),
                    "classification": str(cls),
                }
            )

    df_gate = pd.DataFrame(gate_rows)
    if not df_gate.empty:
        df_gate = df_gate.sort_values(["dim", "V_mode", "word_group"], ascending=True, na_position="last")

    # Best by dim
    best_rows: List[Dict[str, Any]] = []
    primary_vm = str(args.primary_v_mode)
    primary_wg = str(args.primary_word_group)
    if not df_gate.empty:
        for d, g in df_gate.groupby("dim"):
            gg = g.sort_values(["total_curve_error"], ascending=True, na_position="last")
            best = gg.iloc[0]
            prim = gg[(gg["V_mode"] == primary_vm) & (gg["word_group"] == primary_wg)]
            prim_err = float(prim["total_curve_error"].iloc[0]) if not prim.empty else float("nan")
            # rank
            prim_rank = -1
            gg2 = gg.reset_index(drop=True).copy()
            gg2["_rank"] = np.arange(1, len(gg2) + 1, dtype=int)
            pm = (gg2["V_mode"] == primary_vm) & (gg2["word_group"] == primary_wg)
            if pm.any():
                prim_rank = int(gg2.loc[pm, "_rank"].iloc[0])
            best_rows.append(
                {
                    "dim": int(d),
                    "best_V_mode": str(best["V_mode"]),
                    "best_word_group": str(best["word_group"]),
                    "best_total_curve_error": float(best["total_curve_error"]),
                    "best_classification": str(best["classification"]),
                    "primary_total_curve_error": float(prim_err),
                    "primary_rank": int(prim_rank),
                }
            )
    df_best = pd.DataFrame(best_rows)
    if not df_best.empty:
        df_best = df_best.sort_values(["dim"], ascending=True)

    # Write required outputs
    (out_dir / "v13o8_unfolded_levels.csv").write_text(df_levels.to_csv(index=False), encoding="utf-8")
    (out_dir / "v13o8_spacing_levels.csv").write_text(df_spacing.to_csv(index=False), encoding="utf-8")
    (out_dir / "v13o8_nv_curves.csv").write_text(df_curves.to_csv(index=False), encoding="utf-8")
    (out_dir / "v13o8_nv_curve_errors.csv").write_text(df_err.to_csv(index=False), encoding="utf-8")
    (out_dir / "v13o8_nv_curve_regions.csv").write_text(df_regions.to_csv(index=False), encoding="utf-8")
    (out_dir / "v13o8_poissonization_diagnostics.csv").write_text(df_pois.to_csv(index=False), encoding="utf-8")
    (out_dir / "v13o8_gate_summary.csv").write_text(df_gate.to_csv(index=False), encoding="utf-8")
    (out_dir / "v13o8_best_by_dim.csv").write_text(df_best.to_csv(index=False), encoding="utf-8")

    # Recommendation logic
    strict_all_dims = bool(not df_gate.empty) and all(bool(x) for x in df_gate["strict_curve_specificity_pass"].astype(bool).tolist())
    relaxed_any_dim = bool(not df_gate.empty) and any(bool(x) for x in df_gate["relaxed_curve_specificity_pass"].astype(bool).tolist())
    if strict_all_dims and not approximation_mode and not approx_used_any:
        reco = "Proceed to V13P0 analytic renormalization (strict pass all dims)."
        proceed_v13p0 = True
    elif relaxed_any_dim:
        reco = "Run V13O.8 full with more seeds / larger windows before considering V13P0."
        proceed_v13p0 = False
    else:
        reco = "Do not proceed to V13P0. Improve operator/objective/unfolding first."
        proceed_v13p0 = False

    # Primary explicit answers
    primary_gate = df_gate[(df_gate["V_mode"] == primary_vm) & (df_gate["word_group"] == primary_wg)] if not df_gate.empty else pd.DataFrame()
    primary_poisson_like = False
    primary_failure_region = None
    if not primary_gate.empty:
        primary_poisson_like = bool((primary_gate["G7_not_poisson_like"] == False).any())
        primary_failure_region = str(primary_gate.sort_values(["dim"]).iloc[0].get("best_failure_region"))
    best_dim = int(df_best.sort_values(["best_total_curve_error"]).iloc[0]["dim"]) if not df_best.empty else None

    payload: Dict[str, Any] = {
        "warning": "Computational evidence only; not a proof of RH.",
        "status": "V13O.8 explicit unfolded spectra + true number variance curves",
        "approximation_mode": bool(approximation_mode or approx_used_any),
        "true_mode_requested": bool(true_mode),
        "dims": [int(x) for x in args.dims],
        "L_grid": [float(x) for x in L_grid.tolist()],
        "curve_metric": str(args.curve_metric),
        "normalize_curves": bool(args.normalize_curves),
        "tail_L_min": float(args.tail_L_min),
        "primary": {"word_group": primary_wg, "V_mode": primary_vm},
        "primary_answers": {
            "does_primary_still_look_poisson_like": bool(primary_poisson_like),
            "failure_region_primary": primary_failure_region,
            "best_dim": best_dim,
            "should_proceed_to_v13p0": bool(proceed_v13p0),
        },
        "recommendation": reco,
        "inputs": {
            "v13o4_zeta_scores": str(_resolve(args.v13o4_zeta_scores).resolve()),
            "v13o4_number_variance": str(_resolve(args.v13o4_number_variance).resolve()),
            "v13o6_nv_scores": str(_resolve(args.v13o6_nv_scores).resolve()),
            "candidate_json": str(candidate_path.resolve()) if candidate_path is not None else "",
            "formula_json": str(_resolve(args.formula_json).resolve()) if str(args.formula_json).strip() else "",
        },
        "outputs": {
            "unfolded_levels_csv": str((out_dir / "v13o8_unfolded_levels.csv").resolve()),
            "spacing_levels_csv": str((out_dir / "v13o8_spacing_levels.csv").resolve()),
            "nv_curves_csv": str((out_dir / "v13o8_nv_curves.csv").resolve()),
            "nv_curve_errors_csv": str((out_dir / "v13o8_nv_curve_errors.csv").resolve()),
            "nv_curve_regions_csv": str((out_dir / "v13o8_nv_curve_regions.csv").resolve()),
            "poissonization_csv": str((out_dir / "v13o8_poissonization_diagnostics.csv").resolve()),
            "gate_summary_csv": str((out_dir / "v13o8_gate_summary.csv").resolve()),
            "best_by_dim_csv": str((out_dir / "v13o8_best_by_dim.csv").resolve()),
        },
        "runtime_s": float(time.perf_counter() - t0),
        "python_pid": os.getpid(),
    }
    (out_dir / "v13o8_results.json").write_text(json.dumps(json_sanitize(payload), indent=2, allow_nan=False), encoding="utf-8")

    # Report (md)
    md: List[str] = []
    md.append("# V13O.8 Explicit unfolded spectra export + true number variance curves\n\n")
    md.append("> **Computational evidence only; not a proof of RH.**\n\n")
    md.append(f"- approximation_mode: **{bool(payload['approximation_mode'])}**\n")
    md.append(f"- true_mode_requested: **{bool(true_mode)}**\n\n")
    md.append("## Motivation from V13O.7\n\n")
    md.append("V13O.7 used approximation-mode reconstructed curves. V13O.8 exports explicit unfolded spectra and computes Sigma^2(L) directly.\n\n")
    md.append("## Method\n\n")
    md.append("- Export unfolded levels for **target** and **operator**.\n")
    md.append("- Compute number variance curves \\(\\Sigma^2(L)\\) from unfolded levels via sliding windows.\n")
    md.append("- Compute region errors (short/mid/long) and Poisson/GUE diagnostics as computational proxies.\n\n")
    md.append("## L-grid\n\n")
    md.append(f"- L_min={args.L_min}, L_max={args.L_max}, n_L={args.n_L}, tail_L_min={args.tail_L_min}\n")
    md.append(f"- curve_metric=`{args.curve_metric}`, normalize_curves={bool(args.normalize_curves)}\n\n")
    md.append("## Primary gate summary\n\n")
    md.append("See `v13o8_gate_summary.csv`.\n\n")
    md.append("## Best by dim\n\n")
    md.append("See `v13o8_best_by_dim.csv`.\n\n")
    md.append("## Failure classification\n\n")
    md.append("See `v13o8_gate_summary.csv` classifications.\n\n")
    md.append("## Explicit answers\n\n")
    pa = payload["primary_answers"]
    md.append(f"- Does primary still look Poisson-like? **{pa['does_primary_still_look_poisson_like']}**\n")
    md.append(f"- Is failure mostly short/mid/long range? **{pa['failure_region_primary']}**\n")
    md.append(f"- Which dim is best? **{pa['best_dim']}**\n")
    md.append(f"- Should proceed to V13P0? **{pa['should_proceed_to_v13p0']}**\n\n")
    md.append("## Recommendation\n\n")
    md.append(f"**{reco}**\n\n")
    md.append("## CLI examples\n\n")
    md.append("Smoke:\n\n```bash\n")
    md.append('OUT=runs/v13o8_explicit_unfolded_spectra_smoke\n')
    md.append("python3 scripts/run_v13o8_explicit_unfolded_spectra.py \\\n")
    md.append("  --v13o4_summary runs/v13o4_zeta_specific_objective/v13o4_summary.csv \\\n")
    md.append("  --v13o4_group_summary runs/v13o4_zeta_specific_objective/v13o4_group_summary.csv \\\n")
    md.append("  --v13o4_zeta_scores runs/v13o4_zeta_specific_objective/v13o4_zeta_specific_scores.csv \\\n")
    md.append("  --v13o4_number_variance runs/v13o4_zeta_specific_objective/v13o4_number_variance_summary.csv \\\n")
    md.append("  --v13o4_pair_corr runs/v13o4_zeta_specific_objective/v13o4_pair_corr_summary.csv \\\n")
    md.append("  --v13o4_staircase runs/v13o4_zeta_specific_objective/v13o4_staircase_summary.csv \\\n")
    md.append("  --v13o4_ensemble_margins runs/v13o4_zeta_specific_objective/v13o4_ensemble_margins.csv \\\n")
    md.append("  --v13o6_nv_scores runs/v13o6_number_variance_stabilization_smoke/v13o6_nv_stabilized_scores.csv \\\n")
    md.append('  --out_dir "$OUT" \\\n')
    md.append("  --dims 64 128 \\\n")
    md.append("  --n_L 24 \\\n")
    md.append("  --n_jobs 4 \\\n")
    md.append("  --progress_every 1\n")
    md.append("```\n\n")
    md.append("Full:\n\n```bash\n")
    md.append('OUT=runs/v13o8_explicit_unfolded_spectra\n')
    md.append("caffeinate -dimsu python3 scripts/run_v13o8_explicit_unfolded_spectra.py \\\n")
    md.append("  --v13o4_summary runs/v13o4_zeta_specific_objective/v13o4_summary.csv \\\n")
    md.append("  --v13o4_group_summary runs/v13o4_zeta_specific_objective/v13o4_group_summary.csv \\\n")
    md.append("  --v13o4_zeta_scores runs/v13o4_zeta_specific_objective/v13o4_zeta_specific_scores.csv \\\n")
    md.append("  --v13o4_number_variance runs/v13o4_zeta_specific_objective/v13o4_number_variance_summary.csv \\\n")
    md.append("  --v13o4_pair_corr runs/v13o4_zeta_specific_objective/v13o4_pair_corr_summary.csv \\\n")
    md.append("  --v13o4_staircase runs/v13o4_zeta_specific_objective/v13o4_staircase_summary.csv \\\n")
    md.append("  --v13o4_ensemble_margins runs/v13o4_zeta_specific_objective/v13o4_ensemble_margins.csv \\\n")
    md.append("  --v13o6_nv_scores runs/v13o6_number_variance_stabilization_smoke/v13o6_nv_stabilized_scores.csv \\\n")
    md.append('  --out_dir "$OUT" \\\n')
    md.append("  --dims 64 128 256 \\\n")
    md.append("  --L_min 0.5 \\\n")
    md.append("  --L_max 12.0 \\\n")
    md.append("  --n_L 48 \\\n")
    md.append("  --curve_metric shape_l2 \\\n")
    md.append("  --tail_L_min 6.0 \\\n")
    md.append("  --n_jobs 8 \\\n")
    md.append("  --progress_every 10\n")
    md.append("```\n\n")
    md.append("## Convenience commands\n\n```bash\n")
    md.append("OUT=runs/v13o8_explicit_unfolded_spectra_smoke\n\n")
    md.append('echo "=== FILES ==="\nfind "$OUT" -maxdepth 1 -type f | sort\n\n')
    md.append('echo "=== GATE ==="\ncolumn -s, -t < "$OUT"/v13o8_gate_summary.csv | head -80\n\n')
    md.append('echo "=== PRIMARY CURVE ERRORS ==="\n')
    md.append("awk -F, '$2==\"full_V\" && $3==\"primary_word_seed6\"{print}' \\\n  \"$OUT\"/v13o8_nv_curve_errors.csv | head -80\n\n")
    md.append('echo "=== PRIMARY REGIONS ==="\n')
    md.append("awk -F, '$2==\"full_V\" && $3==\"primary_word_seed6\"{print}' \\\n  \"$OUT\"/v13o8_nv_curve_regions.csv | head -80\n\n")
    md.append('echo "=== POISSONIZATION ==="\n')
    md.append("awk -F, '$2==\"full_V\" && $3==\"primary_word_seed6\"{print}' \\\n  \"$OUT\"/v13o8_poissonization_diagnostics.csv | head -80\n\n")
    md.append('echo "=== BEST BY DIM ==="\ncolumn -s, -t < "$OUT"/v13o8_best_by_dim.csv\n\n')
    md.append('echo "=== REPORT ==="\nhead -140 "$OUT"/v13o8_report.md\n')
    md.append("```\n")
    (out_dir / "v13o8_report.md").write_text("".join(md), encoding="utf-8")

    # tex
    tex = (
        "\\documentclass[11pt]{article}\n\\usepackage[margin=1in]{geometry}\n\\begin{document}\n"
        "\\title{V13O.8 Explicit unfolded spectra export + true number variance curves}\n\\maketitle\n"
        "\\section*{Warning}\nComputational evidence only; not a proof of RH.\n\n"
        "\\section*{Mode}\n"
        + latex_escape(f"approximation_mode={payload['approximation_mode']}, true_mode_requested={true_mode}.")
        + "\n\n\\section*{Primary answers}\n"
        + latex_escape(json.dumps(payload.get("primary_answers", {}), indent=2))
        + "\n\n\\section*{Recommendation}\n"
        + latex_escape(reco)
        + "\n\n\\end{document}\n"
    )
    (out_dir / "v13o8_report.tex").write_text(tex, encoding="utf-8")

    if _find_pdflatex() is None:
        print("[v13o8] WARNING: pdflatex not found; skipping PDF.", flush=True)
    elif try_pdflatex(out_dir / "v13o8_report.tex", out_dir, "v13o8_report.pdf"):
        print(f"Wrote {out_dir / 'v13o8_report.pdf'}", flush=True)
    else:
        print("[v13o8] WARNING: pdflatex failed or did not produce v13o8_report.pdf.", flush=True)

    print(f"[v13o8] Wrote {out_dir / 'v13o8_results.json'}", flush=True)


if __name__ == "__main__":
    main()

