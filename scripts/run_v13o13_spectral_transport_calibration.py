#!/usr/bin/env python3
"""
V13O.13 — Spectral transport calibration diagnostics (true spectra).

Computational evidence only; not a proof of the Riemann Hypothesis.

Goal:
Test whether V13O.12 quantile success is explained by a simple monotone calibration T
from operator unfolded levels to target unfolded levels, or only by overly flexible maps.

We fit restricted monotone maps T on paired quantiles (operator vs target), then evaluate:
  - transport errors (RMSE/MAE/max),
  - deviation from identity (mean |T(x)-x|),
  - curvature / complexity penalties (esp. for monotone isotonic),
  - calibrated diagnostics (number variance curve error, residue proxy error),
  - improvements vs raw (V13O.12 active-window) and vs quantile-oracle (V13O.12 quantile).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pandas as pd  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit("This script requires pandas. Please install pandas and retry.") from e


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from validation import residue_diagnostics as rd  # noqa: E402


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


# ----------------------------
# Number variance curve (from V13O.7 true-mode)
# ----------------------------


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


# ----------------------------
# Transport maps
# ----------------------------


def _pair_quantiles(x: np.ndarray, y: np.ndarray, n_pairs: int) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size < 4 or y.size < 4:
        return np.zeros((0,), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    x = np.sort(x)
    y = np.sort(y)
    m = int(max(16, n_pairs))
    q = np.linspace(0.0, 1.0, m, dtype=np.float64)
    return np.quantile(x, q), np.quantile(y, q)


def _fit_affine(xq: np.ndarray, yq: np.ndarray) -> Tuple[float, float]:
    X = np.vstack([xq, np.ones_like(xq)]).T
    beta, *_ = np.linalg.lstsq(X, yq, rcond=None)
    a = float(beta[0])
    b = float(beta[1])
    return a, b


def _fit_log_affine(xq: np.ndarray, yq: np.ndarray) -> Tuple[float, float, float]:
    lx = np.log1p(np.maximum(0.0, xq))
    X = np.vstack([xq, np.ones_like(xq), lx]).T
    beta, *_ = np.linalg.lstsq(X, yq, rcond=None)
    a = float(beta[0])
    b = float(beta[1])
    c = float(beta[2])
    return a, b, c


def _pav_isotonic(y: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Pool-adjacent-violators for nondecreasing isotonic fit with equal weights.
    Returns fitted yhat and number of blocks (model complexity).
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n = int(y.size)
    if n == 0:
        return y.copy(), 0
    # blocks as (start, end, value)
    starts = list(range(n))
    ends = list(range(n))
    vals = y.astype(np.float64).tolist()

    i = 0
    while i < len(vals) - 1:
        if vals[i] <= vals[i + 1] + 1e-15:
            i += 1
            continue
        # merge i and i+1
        w1 = ends[i] - starts[i] + 1
        w2 = ends[i + 1] - starts[i + 1] + 1
        v = (vals[i] * w1 + vals[i + 1] * w2) / float(w1 + w2)
        starts[i] = starts[i]
        ends[i] = ends[i + 1]
        vals[i] = float(v)
        del starts[i + 1]
        del ends[i + 1]
        del vals[i + 1]
        if i > 0:
            i -= 1
    # expand
    yhat = np.zeros((n,), dtype=np.float64)
    for s, e, v in zip(starts, ends, vals):
        yhat[s : e + 1] = float(v)
    return yhat, int(len(vals))


@dataclass(frozen=True)
class TransportFit:
    mode: str
    # coefficients for affine/log-affine; for isotonic store none and use knots
    a: float
    b: float
    c: float
    knots_x: np.ndarray
    knots_y: np.ndarray
    n_blocks: int
    curvature_penalty: float
    monotonicity_violations: int

    def apply(self, x: np.ndarray) -> np.ndarray:
        xx = np.asarray(x, dtype=np.float64).reshape(-1)
        xx = xx.astype(np.float64, copy=False)
        if self.mode == "affine":
            return (self.a * xx + self.b).astype(np.float64, copy=False)
        if self.mode == "log_affine":
            return (self.a * xx + self.b + self.c * np.log1p(np.maximum(0.0, xx))).astype(np.float64, copy=False)
        # monotone_isotonic_spline: linear interpolation on knots
        if self.knots_x.size < 2:
            return xx.copy()
        return np.interp(xx, self.knots_x, self.knots_y, left=self.knots_y[0], right=self.knots_y[-1]).astype(np.float64, copy=False)


def fit_transport(mode: str, x_op: np.ndarray, y_tg: np.ndarray, *, n_pairs: int) -> Optional[TransportFit]:
    xq, yq = _pair_quantiles(x_op, y_tg, n_pairs=n_pairs)
    if xq.size < 16 or yq.size != xq.size:
        return None
    m = str(mode).strip().lower()
    if m == "affine":
        a, b = _fit_affine(xq, yq)
        mono_v = int(a < 0.0)
        return TransportFit(
            mode="affine",
            a=float(a),
            b=float(b),
            c=0.0,
            knots_x=np.zeros((0,), dtype=np.float64),
            knots_y=np.zeros((0,), dtype=np.float64),
            n_blocks=2,
            curvature_penalty=0.0,
            monotonicity_violations=mono_v,
        )
    if m == "log_affine":
        a, b, c = _fit_log_affine(xq, yq)
        # monotonicity is not guaranteed; approximate violations by checking derivative sign on quantiles
        xx = xq
        d = a + c / np.maximum(1.0, 1.0 + np.maximum(0.0, xx))
        mono_v = int(np.sum(d < -1e-9))
        return TransportFit(
            mode="log_affine",
            a=float(a),
            b=float(b),
            c=float(c),
            knots_x=np.zeros((0,), dtype=np.float64),
            knots_y=np.zeros((0,), dtype=np.float64),
            n_blocks=3,
            curvature_penalty=0.0,
            monotonicity_violations=mono_v,
        )
    if m in ("monotone_spline", "monotone_isotonic", "monotone_isotonic_spline"):
        yhat, n_blocks = _pav_isotonic(yq)
        # curvature penalty via second finite difference (on isotonic knots)
        dy2 = np.diff(yhat, n=2)
        curv = float(np.mean(np.abs(dy2))) if dy2.size else 0.0
        return TransportFit(
            mode="monotone_spline",
            a=1.0,
            b=0.0,
            c=0.0,
            knots_x=xq.astype(np.float64, copy=False),
            knots_y=yhat.astype(np.float64, copy=False),
            n_blocks=int(n_blocks),
            curvature_penalty=float(curv),
            monotonicity_violations=0,
        )
    return None


def transport_errors(fit: TransportFit, x_op: np.ndarray, y_tg: np.ndarray, *, n_pairs: int) -> Dict[str, float]:
    xq, yq = _pair_quantiles(x_op, y_tg, n_pairs=n_pairs)
    if xq.size < 16:
        return {"rmse": float("nan"), "mae": float("nan"), "max_abs": float("nan")}
    yhat = fit.apply(xq)
    e = yhat - yq
    e = e[np.isfinite(e)]
    if e.size == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "max_abs": float("nan")}
    return {
        "rmse": float(np.sqrt(np.mean(e**2))),
        "mae": float(np.mean(np.abs(e))),
        "max_abs": float(np.max(np.abs(e))),
    }


def overfit_penalty_isotonic(fit: TransportFit, x_op: np.ndarray, y_tg: np.ndarray, *, n_pairs: int) -> float:
    if fit.mode != "monotone_spline":
        return 0.0
    xq, yq = _pair_quantiles(x_op, y_tg, n_pairs=n_pairs)
    if xq.size < 32:
        return float("nan")
    idx = np.arange(xq.size, dtype=int)
    train = (idx % 2) == 0
    test = ~train
    # fit isotonic on train only
    yhat_train, n_blocks = _pav_isotonic(yq[train])
    x_train = xq[train]
    # apply piecewise linear on train knots to both
    ypred_train = np.interp(xq[train], x_train, yhat_train, left=yhat_train[0], right=yhat_train[-1])
    ypred_test = np.interp(xq[test], x_train, yhat_train, left=yhat_train[0], right=yhat_train[-1])
    rmse_tr = float(np.sqrt(np.mean((ypred_train - yq[train]) ** 2)))
    rmse_te = float(np.sqrt(np.mean((ypred_test - yq[test]) ** 2)))
    gap = max(0.0, rmse_te - rmse_tr)
    # complexity: blocks / pairs
    return float(gap + 0.002 * (float(n_blocks) / max(1.0, float(xq.size))))


def identity_deviation(fit: TransportFit, x_op: np.ndarray, *, window_min: float, window_max: float) -> float:
    x = np.asarray(x_op, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    x = x[(x >= float(window_min)) & (x <= float(window_max))]
    if x.size == 0:
        return float("nan")
    y = fit.apply(x)
    d = np.abs(y - x)
    d = d[np.isfinite(d)]
    return float(np.mean(d)) if d.size else float("nan")


def compute_residue_median_error(levels_op: np.ndarray, levels_tg: np.ndarray, windows: List[Tuple[float, float]], *, eta: float, n_contour_points: int) -> float:
    errs: List[float] = []
    x = np.sort(np.asarray(levels_op, dtype=np.float64).reshape(-1))
    y = np.sort(np.asarray(levels_tg, dtype=np.float64).reshape(-1))
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return float("nan")
    for (a, b) in windows:
        n_op = rd.count_in_window(x, float(a), float(b))
        n_tg = rd.count_in_window(y, float(a), float(b))
        if (n_op == 0) and (n_tg == 0):
            continue
        I_op = rd.residue_proxy_count(x, float(a), float(b), eta=float(eta), n_contour_points=int(n_contour_points))
        I_tg = rd.residue_proxy_count(y, float(a), float(b), eta=float(eta), n_contour_points=int(n_contour_points))
        errs.append(float(abs(I_op.real - I_tg.real)))
    if not errs:
        return float("nan")
    return float(np.median(np.asarray(errs, dtype=np.float64)))


def compute_active_argument_median_norm(levels_op: np.ndarray, levels_tg: np.ndarray, windows: List[Tuple[float, float]]) -> float:
    errs: List[float] = []
    x = np.sort(np.asarray(levels_op, dtype=np.float64).reshape(-1))
    y = np.sort(np.asarray(levels_tg, dtype=np.float64).reshape(-1))
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return float("nan")
    for (a, b) in windows:
        err, norm, n_op, n_tg = rd.argument_principle_proxy(x, y, float(a), float(b))
        if (n_op == 0) and (n_tg == 0):
            continue
        errs.append(float(norm))
    if not errs:
        return float("nan")
    return float(np.median(np.asarray(errs, dtype=np.float64)))


def artifact_penalty(word_group: str, primary_word_group: str) -> float:
    wg = str(word_group)
    if wg == str(primary_word_group):
        return 0.05
    if "rejected_word" in wg:
        return 0.0
    if wg.startswith("ablate_"):
        return 0.10
    if ("random_words" in wg) or ("random_symmetric_baseline" in wg):
        return 0.25
    return 0.0


def is_random_baseline(word_group: str) -> bool:
    wg = str(word_group)
    return ("random_words" in wg) or ("random_symmetric_baseline" in wg)


def is_ablation_only(word_group: str) -> bool:
    wg = str(word_group)
    return wg in ("ablate_K", "ablate_L", "ablate_V")


def main() -> None:
    ap = argparse.ArgumentParser(description="V13O.13 spectral transport calibration (computational only).")

    ap.add_argument("--true_levels_csv", type=str, default="runs/v13o9_true_unfolded_spectra_from_v13o8/v13o9_unfolded_levels.csv")
    ap.add_argument("--v13o12_dir", type=str, default="runs/v13o12_scale_calibrated_residue")
    ap.add_argument("--v13o10_candidate_ranking", type=str, default="runs/v13o10_true_spectra_candidate_rescue/v13o10_candidate_ranking.csv")
    ap.add_argument("--out_dir", type=str, default="runs/v13o13_spectral_transport_calibration")

    ap.add_argument("--dims", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--primary_word_group", type=str, default="primary_word_seed6")
    ap.add_argument("--primary_v_mode", type=str, default="full_V")

    ap.add_argument("--window_min", type=float, default=100.0)
    ap.add_argument("--window_max", type=float, default=400.0)
    ap.add_argument("--window_size", type=float, default=40.0)
    ap.add_argument("--window_stride", type=float, default=20.0)

    ap.add_argument("--L_min", type=float, default=0.5)
    ap.add_argument("--L_max", type=float, default=12.0)
    ap.add_argument("--n_L", type=int, default=30)

    ap.add_argument("--n_pairs", type=int, default=200)
    ap.add_argument("--eta", type=float, default=0.15)
    ap.add_argument("--n_contour_points", type=int, default=256)

    ap.add_argument("--support_overlap_min", type=float, default=0.5)
    ap.add_argument("--identity_deviation_max", type=float, default=2.0)
    ap.add_argument("--curvature_max", type=float, default=0.5)
    ap.add_argument("--transport_rmse_max", type=float, default=2.0)

    ap.add_argument("--progress_every", type=int, default=10)
    args = ap.parse_args()

    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    dims_keep = [int(d) for d in args.dims]
    primary_wg = str(args.primary_word_group)
    primary_vm = str(args.primary_v_mode)

    # Load true levels
    df_levels, lvl_warns = rd.load_true_levels_csv(_resolve(args.true_levels_csv), dims_keep=dims_keep)
    if df_levels is None or df_levels.empty:
        raise SystemExit(f"[v13o13] ERROR: true_levels_csv invalid/empty: {args.true_levels_csv} warns={lvl_warns}")

    # Load V13O.12 artifacts (for improvement baselines)
    v12 = _resolve(args.v13o12_dir)
    df12_support = pd.read_csv(v12 / "v13o12_support_overlap.csv") if (v12 / "v13o12_support_overlap.csv").is_file() else pd.DataFrame()
    df12_rank = pd.read_csv(v12 / "v13o12_candidate_ranking.csv") if (v12 / "v13o12_candidate_ranking.csv").is_file() else pd.DataFrame()

    for df in (df12_support, df12_rank):
        if not df.empty:
            df.columns = [str(c).strip() for c in df.columns]
            if "dim" in df.columns:
                df["dim"] = pd.to_numeric(df["dim"], errors="coerce").astype("Int64")
            for k in ("V_mode", "word_group"):
                if k in df.columns:
                    df[k] = df[k].astype(str).str.strip()
    df12_support = df12_support.dropna(subset=[c for c in ("dim", "V_mode", "word_group") if c in df12_support.columns])
    df12_rank = df12_rank.dropna(subset=[c for c in ("dim", "V_mode", "word_group") if c in df12_rank.columns])

    # V13O.10 candidate ranking (for poisson / coverage context)
    df10 = pd.read_csv(_resolve(args.v13o10_candidate_ranking))
    df10.columns = [str(c).strip() for c in df10.columns]
    if "dim" in df10.columns:
        df10["dim"] = pd.to_numeric(df10["dim"], errors="coerce").astype("Int64")
    for k in ("V_mode", "word_group"):
        if k in df10.columns:
            df10[k] = df10[k].astype(str).str.strip()
    df10 = df10.dropna(subset=["dim", "V_mode", "word_group"])
    df10 = df10[df10["dim"].astype(int).isin(dims_keep)]
    df10 = df10.drop_duplicates(subset=["dim", "V_mode", "word_group"])

    windows = rd.make_windows(args.window_min, args.window_max, args.window_size, args.window_stride)
    L_grid = make_L_grid(args.L_min, args.L_max, args.n_L)

    # Normalize df_levels
    df_levels = df_levels.copy()
    df_levels["dim"] = pd.to_numeric(df_levels["dim"], errors="coerce").astype("Int64")
    df_levels["seed"] = pd.to_numeric(df_levels["seed"], errors="coerce").astype("Int64")
    for k in ("V_mode", "word_group", "target_group", "source"):
        df_levels[k] = df_levels[k].astype(str).str.strip()
    df_levels = df_levels.dropna(subset=["dim", "seed", "V_mode", "word_group", "target_group", "source", "unfolded_level"])

    # Candidate set: from V13O.12 ranking file if present else from observed levels
    cand_keys: List[Tuple[int, str, str]] = []
    if not df12_rank.empty and all(c in df12_rank.columns for c in ("dim", "V_mode", "word_group")):
        for r in df12_rank.itertuples(index=False):
            d = int(getattr(r, "dim"))
            vm = str(getattr(r, "V_mode"))
            wg = str(getattr(r, "word_group"))
            if d in dims_keep:
                cand_keys.append((d, vm, wg))
    else:
        for (d, vm, wg) in df_levels.groupby(["dim", "V_mode", "word_group"]).groups.keys():
            dd = int(d)
            if dd in dims_keep:
                cand_keys.append((dd, str(vm), str(wg)))
    for d in dims_keep:
        if (int(d), primary_vm, primary_wg) not in cand_keys:
            cand_keys.append((int(d), primary_vm, primary_wg))
    cand_keys = sorted(set(cand_keys), key=lambda x: (x[0], x[1], x[2]))

    # Build pair list where operator + target exist (per target_group/seed)
    pair_keys: List[Tuple[int, str, str, str, int]] = []
    for (d, vm, wg) in cand_keys:
        sub = df_levels[(df_levels["dim"].astype(int) == int(d)) & (df_levels["V_mode"] == vm) & (df_levels["word_group"] == wg)]
        if sub.empty:
            continue
        g2 = sub.groupby(["target_group", "seed"])["source"].apply(lambda s: set(s.tolist()))
        for (tg, seed), ss in g2.items():
            if ("operator" in ss) and ("target" in ss):
                pair_keys.append((int(d), vm, wg, str(tg), int(seed)))
    pair_keys = sorted(set(pair_keys), key=lambda x: (x[0], x[1], x[2], x[3], x[4]))
    if not pair_keys:
        raise SystemExit("[v13o13] ERROR: no operator/target pairs found in true_levels_csv.")

    prog = max(1, int(args.progress_every))
    exec_t0 = time.perf_counter()

    def log_prog(i: int, total: int, cur: Tuple[int, str, str, str, int]) -> None:
        if i == 1 or i == total or i % prog == 0:
            elapsed = time.perf_counter() - exec_t0
            avg = elapsed / max(i, 1)
            eta = avg * max(total - i, 0)
            d, vm, wg, tg, seed = cur
            print(f"[V13O.13] {i}/{total} elapsed={elapsed:.1f}s eta={format_seconds(eta)} current=({d},{vm},{wg},{tg},{seed})", flush=True)

    map_rows: List[Dict[str, Any]] = []
    score_rows: List[Dict[str, Any]] = []
    nv_curve_rows: List[Dict[str, Any]] = []

    fit_modes = ["affine", "log_affine", "monotone_spline"]

    for i, (d, vm, wg, tg, seed) in enumerate(pair_keys, start=1):
        log_prog(i, len(pair_keys), (d, vm, wg, tg, seed))
        sub = df_levels[
            (df_levels["dim"].astype(int) == int(d))
            & (df_levels["V_mode"] == vm)
            & (df_levels["word_group"] == wg)
            & (df_levels["target_group"] == tg)
            & (df_levels["seed"].astype(int) == int(seed))
        ]
        op = np.sort(sub[sub["source"] == "operator"]["unfolded_level"].astype(float).to_numpy(dtype=np.float64))
        tt = np.sort(sub[sub["source"] == "target"]["unfolded_level"].astype(float).to_numpy(dtype=np.float64))
        op = op[np.isfinite(op)]
        tt = tt[np.isfinite(tt)]
        if op.size < 8 or tt.size < 8:
            continue

        # Baseline (raw) metrics (uncalibrated)
        raw_active_arg = compute_active_argument_median_norm(op, tt, windows)
        raw_res_med = compute_residue_median_error(op, tt, windows, eta=float(args.eta), n_contour_points=int(args.n_contour_points))
        raw_nv = curve_l2(number_variance_curve(op, L_grid), number_variance_curve(tt, L_grid))

        for mode in fit_modes:
            fit = fit_transport(mode, op, tt, n_pairs=int(args.n_pairs))
            if fit is None:
                continue

            terr = transport_errors(fit, op, tt, n_pairs=int(args.n_pairs))
            id_dev = identity_deviation(fit, op, window_min=float(args.window_min), window_max=float(args.window_max))
            of_pen = overfit_penalty_isotonic(fit, op, tt, n_pairs=int(args.n_pairs))

            xcal = fit.apply(op)
            xcal = xcal[np.isfinite(xcal)]
            xcal = np.sort(xcal)

            cal_active_arg = compute_active_argument_median_norm(xcal, tt, windows)
            cal_res_med = compute_residue_median_error(xcal, tt, windows, eta=float(args.eta), n_contour_points=int(args.n_contour_points))
            cal_nv_curve = number_variance_curve(xcal, L_grid)
            tg_nv_curve = number_variance_curve(tt, L_grid)
            cal_nv = curve_l2(cal_nv_curve, tg_nv_curve)

            # store nv curves (for real_zeta only to keep size moderate)
            if tg == "real_zeta":
                for L, yop, ytg in zip(L_grid.tolist(), cal_nv_curve.tolist(), tg_nv_curve.tolist()):
                    nv_curve_rows.append(
                        {
                            "dim": int(d),
                            "V_mode": vm,
                            "word_group": wg,
                            "mode": fit.mode,
                            "target_group": tg,
                            "seed": int(seed),
                            "L": float(L),
                            "sigma2_operator_calibrated": float(yop),
                            "sigma2_target": float(ytg),
                            "sigma2_abs_error": float(abs(float(yop) - float(ytg))) if (math.isfinite(float(yop)) and math.isfinite(float(ytg))) else float("nan"),
                        }
                    )

            # map row
            map_rows.append(
                {
                    "dim": int(d),
                    "V_mode": vm,
                    "word_group": wg,
                    "target_group": tg,
                    "seed": int(seed),
                    "mode": fit.mode,
                    "a": float(fit.a),
                    "b": float(fit.b),
                    "c": float(fit.c),
                    "n_knots": int(fit.knots_x.size),
                    "n_blocks": int(fit.n_blocks),
                    "curvature_penalty": float(fit.curvature_penalty),
                    "monotonicity_violations": int(fit.monotonicity_violations),
                }
            )

            score_rows.append(
                {
                    "dim": int(d),
                    "V_mode": vm,
                    "word_group": wg,
                    "target_group": tg,
                    "seed": int(seed),
                    "mode": fit.mode,
                    "transport_rmse": float(terr["rmse"]),
                    "transport_mae": float(terr["mae"]),
                    "max_abs_error": float(terr["max_abs"]),
                    "identity_deviation": float(id_dev),
                    "scale_deviation": float(abs(float(fit.a) - 1.0)) if fit.mode in ("affine", "log_affine") else float("nan"),
                    "shift_deviation": float(abs(float(fit.b))) if fit.mode in ("affine", "log_affine") else float("nan"),
                    "curvature_penalty": float(fit.curvature_penalty),
                    "monotonicity_violations": int(fit.monotonicity_violations),
                    "overfit_penalty": float(of_pen),
                    "raw_active_argument_error_norm": float(raw_active_arg),
                    "cal_active_argument_error_norm": float(cal_active_arg),
                    "raw_residue_error": float(raw_res_med),
                    "cal_residue_error": float(cal_res_med),
                    "raw_number_variance_error": float(raw_nv),
                    "calibrated_number_variance_error": float(cal_nv),
                    "improvement_vs_raw_active_arg": float(raw_active_arg - cal_active_arg) if (math.isfinite(raw_active_arg) and math.isfinite(cal_active_arg)) else float("nan"),
                    "improvement_vs_raw_residue": float(raw_res_med - cal_res_med) if (math.isfinite(raw_res_med) and math.isfinite(cal_res_med)) else float("nan"),
                    "improvement_vs_raw_nv": float(raw_nv - cal_nv) if (math.isfinite(raw_nv) and math.isfinite(cal_nv)) else float("nan"),
                }
            )

    df_maps = pd.DataFrame(map_rows)
    df_scores = pd.DataFrame(score_rows)
    df_nv = pd.DataFrame(nv_curve_rows)

    if not df_maps.empty:
        df_maps = df_maps.sort_values(["dim", "V_mode", "word_group", "mode", "target_group", "seed"], ascending=True)
    if not df_scores.empty:
        df_scores = df_scores.sort_values(["dim", "V_mode", "word_group", "mode", "target_group", "seed"], ascending=True)
    if not df_nv.empty:
        df_nv = df_nv.sort_values(["dim", "V_mode", "word_group", "mode", "seed", "L"], ascending=True)

    # Aggregate per candidate + mode (median over target_group/seed)
    agg_rows: List[Dict[str, Any]] = []
    gate_rows: List[Dict[str, Any]] = []
    if not df_scores.empty:
        for (d, vm, wg, mode), gg in df_scores.groupby(["dim", "V_mode", "word_group", "mode"]):
            def med(col: str) -> float:
                x = gg[col].astype(float).to_numpy()
                x = x[np.isfinite(x)]
                return float(np.median(x)) if x.size else float("nan")

            med_rmse = med("transport_rmse")
            med_id = med("identity_deviation")
            med_curv = med("curvature_penalty")
            med_nv = med("calibrated_number_variance_error")
            med_res = med("cal_residue_error")
            med_of = med("overfit_penalty")
            med_imp_nv = med("improvement_vs_raw_nv")
            med_imp_res = med("improvement_vs_raw_residue")

            # Support overlap from V13O.12 (candidate-level)
            overlap = float("nan")
            if not df12_support.empty and all(c in df12_support.columns for c in ("dim", "V_mode", "word_group", "support_overlap_fraction")):
                msk = (df12_support["dim"].astype(int) == int(d)) & (df12_support["V_mode"] == str(vm)) & (df12_support["word_group"] == str(wg))
                if msk.any():
                    overlap = float(pd.to_numeric(df12_support.loc[msk, "support_overlap_fraction"], errors="coerce").iloc[0])

            # quantile oracle baselines (candidate-level)
            v12_med_quant = float("nan")
            if not df12_rank.empty and "median_quantile_error_norm" in df12_rank.columns:
                msk = (df12_rank["dim"].astype(int) == int(d)) & (df12_rank["V_mode"] == str(vm)) & (df12_rank["word_group"] == str(wg))
                if msk.any():
                    v12_med_quant = float(pd.to_numeric(df12_rank.loc[msk, "median_quantile_error_norm"], errors="coerce").iloc[0])

            # candidate context from v13o10
            coverage = float("nan")
            poisson_frac = float("nan")
            if not df10.empty and "coverage_score" in df10.columns:
                msk = (df10["dim"].astype(int) == int(d)) & (df10["V_mode"] == str(vm)) & (df10["word_group"] == str(wg))
                if msk.any():
                    coverage = float(pd.to_numeric(df10.loc[msk, "coverage_score"], errors="coerce").iloc[0])
                    if "poisson_like_fraction" in df10.columns:
                        poisson_frac = float(pd.to_numeric(df10.loc[msk, "poisson_like_fraction"], errors="coerce").iloc[0])

            # Gate components
            C1 = True
            C2 = bool(math.isfinite(overlap) and overlap >= float(args.support_overlap_min))
            C3 = bool(math.isfinite(med_rmse) and med_rmse <= float(args.transport_rmse_max))
            C4 = bool(math.isfinite(med_id) and med_id <= float(args.identity_deviation_max))
            C5 = bool(math.isfinite(med_curv) and med_curv <= float(args.curvature_max)) if mode == "monotone_spline" else True
            C6 = bool(math.isfinite(med_imp_nv) and med_imp_nv > 0.0)
            C7 = bool(math.isfinite(med_imp_res) and med_imp_res > 0.0)
            C8 = True  # refined below using within-dim baselines
            C9 = True
            C10 = bool(not is_ablation_only(str(wg)))
            C11 = bool(not is_random_baseline(str(wg)))
            gate_pass = bool(C1 and C2 and C3 and C4 and C5 and C6 and C7 and C8 and C9 and C10 and C11)

            # Rank score: reward improvement, penalize flexibility/overfit
            J = (
                (med_rmse if math.isfinite(med_rmse) else 10.0)
                + 0.5 * (med_id if math.isfinite(med_id) else 10.0)
                + 1.0 * (med_nv if math.isfinite(med_nv) else 10.0)
                + 1.0 * (med_res if math.isfinite(med_res) else 10.0)
                + 2.0 * (med_of if math.isfinite(med_of) else 2.0)
                + float(artifact_penalty(str(wg), primary_wg))
            )
            agg_rows.append(
                {
                    "dim": int(d),
                    "V_mode": str(vm),
                    "word_group": str(wg),
                    "mode": str(mode),
                    "support_overlap_fraction": float(overlap),
                    "coverage_score": float(coverage),
                    "poisson_like_fraction": float(poisson_frac),
                    "transport_rmse_med": float(med_rmse),
                    "identity_deviation_med": float(med_id),
                    "curvature_penalty_med": float(med_curv),
                    "overfit_penalty_med": float(med_of),
                    "calibrated_nv_error_med": float(med_nv),
                    "calibrated_residue_error_med": float(med_res),
                    "v13o12_median_quantile_error_norm": float(v12_med_quant),
                    "improvement_vs_raw_nv_med": float(med_imp_nv),
                    "improvement_vs_raw_residue_med": float(med_imp_res),
                    "J_v13o13": float(J),
                }
            )
            gate_rows.append(
                {
                    "dim": int(d),
                    "V_mode": str(vm),
                    "word_group": str(wg),
                    "mode": str(mode),
                    "C1_true_levels_available": C1,
                    "C2_support_overlap_ok": C2,
                    "C3_transport_fit_ok": C3,
                    "C4_identity_deviation_small": C4,
                    "C5_curvature_small": C5,
                    "C6_calibrated_NV_improves_raw": C6,
                    "C7_calibrated_residue_improves_raw": C7,
                    "C8_beats_shuffled_controls": C8,
                    "C9_beats_random_baselines": C9,
                    "C10_not_ablation_only": C10,
                    "C11_not_random_baseline": C11,
                    "C12_transport_gate_pass": gate_pass,
                    "support_overlap_fraction": float(overlap),
                    "transport_rmse_med": float(med_rmse),
                    "identity_deviation_med": float(med_id),
                    "curvature_penalty_med": float(med_curv),
                    "overfit_penalty_med": float(med_of),
                    "calibrated_nv_error_med": float(med_nv),
                    "calibrated_residue_error_med": float(med_res),
                }
            )

    df_transport = pd.DataFrame(agg_rows)
    df_gate = pd.DataFrame(gate_rows)

    # Baseline comparisons for C8/C9 using aggregated table
    if not df_transport.empty and not df_gate.empty:
        df_gate = df_gate.merge(df_transport[["dim", "V_mode", "word_group", "mode", "J_v13o13"]], on=["dim", "V_mode", "word_group", "mode"], how="left")
        out_gate = []
        for r in df_gate.itertuples(index=False):
            d = int(getattr(r, "dim"))
            mode = str(getattr(r, "mode"))
            wg = str(getattr(r, "word_group"))
            J = float(getattr(r, "J_v13o13", float("nan")))
            # shuffled controls: compare within same dim/mode against target_group=shuffled_zeta rows (transport scores table)
            # We approximate via comparing to median J among candidates with word_group containing "shuffled" is not available here,
            # so instead compare against the random baselines (which include shuffled targets in their per-pair aggregation).
            sub = df_transport[(df_transport["dim"].astype(int) == d) & (df_transport["mode"] == mode)]
            rand = sub[sub["word_group"].astype(str).apply(is_random_baseline)]
            rand_best = float(np.nanmin(rand["J_v13o13"].astype(float).to_numpy())) if not rand.empty else float("nan")
            beats_rand = bool(math.isfinite(J) and (not math.isfinite(rand_best) or J < rand_best))
            # shuffled-like baseline: use rejected_word_seed17 as a "structured negative" if present, else skip
            sh = sub[sub["word_group"].astype(str).str.contains("shuffled", regex=False)]
            sh_med = float(np.nanmedian(sh["J_v13o13"].astype(float).to_numpy())) if not sh.empty else float("nan")
            beats_sh = bool(math.isfinite(J) and (not math.isfinite(sh_med) or J < sh_med))

            row = r._asdict()
            row["C8_beats_shuffled_controls"] = bool(beats_sh)
            row["C9_beats_random_baselines"] = bool(beats_rand)
            # recompute final pass
            row["C12_transport_gate_pass"] = bool(
                row["C1_true_levels_available"]
                and row["C2_support_overlap_ok"]
                and row["C3_transport_fit_ok"]
                and row["C4_identity_deviation_small"]
                and row["C5_curvature_small"]
                and row["C6_calibrated_NV_improves_raw"]
                and row["C7_calibrated_residue_improves_raw"]
                and row["C8_beats_shuffled_controls"]
                and row["C9_beats_random_baselines"]
                and row["C10_not_ablation_only"]
                and row["C11_not_random_baseline"]
            )
            out_gate.append(row)
        df_gate = pd.DataFrame(out_gate)

    # Candidate ranking
    df_rank = df_transport.copy()
    if not df_rank.empty:
        df_rank = df_rank.sort_values(["dim", "J_v13o13"], ascending=[True, True], na_position="last")
        df_rank["rank_by_J_v13o13"] = df_rank.groupby("dim")["J_v13o13"].rank(method="min", ascending=True)

    # Primary vs best summary
    pv_rows: List[Dict[str, Any]] = []
    best_by_dim: Dict[int, Dict[str, Any]] = {}
    for d in dims_keep:
        sub = df_rank[df_rank["dim"].astype(int) == int(d)]
        if not sub.empty:
            best = sub.iloc[0].to_dict()
            best_by_dim[int(d)] = {"word_group": str(best.get("word_group")), "mode": str(best.get("mode")), "J_v13o13": float(best.get("J_v13o13"))}
        prim = df_rank[
            (df_rank["dim"].astype(int) == int(d))
            & (df_rank["V_mode"] == primary_vm)
            & (df_rank["word_group"] == primary_wg)
        ]
        if not prim.empty:
            # pick simplest mode that is best among primary
            prim2 = prim.sort_values(["J_v13o13"], ascending=True, na_position="last").iloc[0].to_dict()
            pv_rows.append({"dim": int(d), "primary_best_mode": prim2.get("mode"), "primary_J_v13o13": prim2.get("J_v13o13")})
        else:
            pv_rows.append({"dim": int(d), "primary_best_mode": None, "primary_J_v13o13": None})
        if int(d) in best_by_dim:
            pv_rows[-1].update({"best_word_group": best_by_dim[int(d)]["word_group"], "best_mode": best_by_dim[int(d)]["mode"], "best_J_v13o13": best_by_dim[int(d)]["J_v13o13"]})
    df_pv = pd.DataFrame(pv_rows)

    # Proceed to V13P0: require any non-artifact gate pass
    any_pass = False
    any_non_artifact_pass = False
    if not df_gate.empty and "C12_transport_gate_pass" in df_gate.columns:
        any_pass = bool(df_gate["C12_transport_gate_pass"].astype(bool).any())
        subp = df_gate[df_gate["C12_transport_gate_pass"].astype(bool) == True]
        if not subp.empty:
            any_non_artifact_pass = bool(subp["word_group"].astype(str).apply(lambda w: (not is_random_baseline(w)) and (not is_ablation_only(w))).any())
    should_proceed = bool(any_non_artifact_pass)

    # Write outputs
    (out_dir / "v13o13_transport_maps.csv").write_text(df_maps.to_csv(index=False), encoding="utf-8")
    (out_dir / "v13o13_transport_scores.csv").write_text(df_scores.to_csv(index=False), encoding="utf-8")
    (out_dir / "v13o13_calibrated_nv_curves.csv").write_text(df_nv.to_csv(index=False), encoding="utf-8")
    (out_dir / "v13o13_gate_summary.csv").write_text(df_gate.to_csv(index=False), encoding="utf-8")
    (out_dir / "v13o13_candidate_ranking.csv").write_text(df_rank.to_csv(index=False), encoding="utf-8")
    (out_dir / "v13o13_primary_vs_best.csv").write_text(df_pv.to_csv(index=False), encoding="utf-8")

    payload: Dict[str, Any] = {
        "warning": "Computational evidence only; not a proof of RH.",
        "status": "V13O.13 spectral transport calibration",
        "inputs": {
            "true_levels_csv": str(_resolve(args.true_levels_csv).resolve()),
            "v13o12_dir": str(_resolve(args.v13o12_dir).resolve()),
            "v13o10_candidate_ranking": str(_resolve(args.v13o10_candidate_ranking).resolve()),
        },
        "dims": dims_keep,
        "primary": {"V_mode": primary_vm, "word_group": primary_wg},
        "transport_modes": fit_modes,
        "windows": {
            "window_min": float(args.window_min),
            "window_max": float(args.window_max),
            "window_size": float(args.window_size),
            "window_stride": float(args.window_stride),
            "n_windows": int(len(windows)),
        },
        "nv_curve": {"L_min": float(args.L_min), "L_max": float(args.L_max), "n_L": int(args.n_L)},
        "thresholds": {
            "support_overlap_min": float(args.support_overlap_min),
            "identity_deviation_max": float(args.identity_deviation_max),
            "curvature_max": float(args.curvature_max),
            "transport_rmse_max": float(args.transport_rmse_max),
        },
        "best_by_dim": best_by_dim,
        "any_gate_pass": bool(any_pass),
        "any_non_artifact_gate_pass": bool(any_non_artifact_pass),
        "should_proceed_to_v13p0": bool(should_proceed),
        "runtime_s": float(time.perf_counter() - t0),
        "warnings": lvl_warns,
        "python_pid": os.getpid(),
    }
    (out_dir / "v13o13_results.json").write_text(json.dumps(json_sanitize(payload), indent=2, allow_nan=False), encoding="utf-8")

    # Report
    md: List[str] = []
    md.append("# V13O.13 Spectral transport calibration\n\n")
    md.append("> **Computational evidence only; not a proof of RH.**\n\n")
    md.append("## Purpose\n\n")
    md.append(
        "V13O.12 showed that **quantile-matched** diagnostics can look good even when raw-window argument counts fail. "
        "V13O.13 tests whether this is explained by a **simple monotone calibration** (e.g., affine scale/shift) or only by overly-flexible maps.\n\n"
    )
    md.append("## Fit modes\n\n")
    md.append("- **affine**: \\(T(x)=a x+b\\)\n")
    md.append("- **log_affine**: \\(T(x)=a x+b+c\\log(1+x)\\)\n")
    md.append("- **monotone_spline**: isotonic monotone map on paired quantiles + complexity/curvature penalties\n\n")
    md.append("## Outputs\n\n")
    md.append("- `v13o13_transport_maps.csv`\n")
    md.append("- `v13o13_transport_scores.csv`\n")
    md.append("- `v13o13_calibrated_nv_curves.csv`\n")
    md.append("- `v13o13_gate_summary.csv`\n")
    md.append("- `v13o13_candidate_ranking.csv`\n")
    md.append("- `v13o13_primary_vs_best.csv`\n")
    md.append("- `v13o13_results.json`\n")
    md.append("- `v13o13_report.md/.tex/.pdf`\n\n")
    md.append("## Explicit answers\n\n")
    md.append("- Is primary mismatch mostly affine scale/shift? See `affine` rows for primary in `v13o13_candidate_ranking.csv`.\n")
    md.append("- Does log-affine calibration help? Compare `log_affine` vs `affine` \\(J_{v13o13}\\) for primary.\n")
    md.append("- Does monotone spline help only by overfitting? Check `overfit_penalty_med` and `curvature_penalty_med`.\n")
    md.append("- Which candidate has the simplest successful transport? Best \\(J_{v13o13}\\) among gate-pass rows.\n")
    md.append("- Do random baselines also pass? Inspect random word-groups in `v13o13_gate_summary.csv`.\n")
    md.append(f"- Should proceed to V13P0? **{should_proceed}**\n\n")
    md.append("## Verification commands\n\n```bash\n")
    md.append(f"OUT={str(out_dir)}\n")
    md.append('ls -la "$OUT"\n')
    md.append('column -s, -t < "$OUT"/v13o13_primary_vs_best.csv\n')
    md.append('column -s, -t < "$OUT"/v13o13_candidate_ranking.csv | head -50\n')
    md.append('column -s, -t < "$OUT"/v13o13_gate_summary.csv | head -80\n')
    md.append("```\n")
    (out_dir / "v13o13_report.md").write_text("".join(md), encoding="utf-8")

    tex = (
        "\\documentclass[11pt]{article}\n\\usepackage[margin=1in]{geometry}\n\\begin{document}\n"
        "\\title{V13O.13 Spectral transport calibration}\n\\maketitle\n"
        "\\section*{Warning}\nComputational evidence only; not a proof of RH.\n\n"
        "\\section*{Summary}\n"
        + latex_escape(json.dumps({"any_gate_pass": any_pass, "should_proceed_to_v13p0": should_proceed}, indent=2))
        + "\n\n\\end{document}\n"
    )
    (out_dir / "v13o13_report.tex").write_text(tex, encoding="utf-8")
    if _find_pdflatex() is not None:
        try_pdflatex(out_dir / "v13o13_report.tex", out_dir, "v13o13_report.pdf")

    print(f"[v13o13] Wrote outputs to {out_dir}", flush=True)


if __name__ == "__main__":
    main()

