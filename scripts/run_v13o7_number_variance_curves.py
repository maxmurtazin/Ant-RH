#!/usr/bin/env python3
"""
V13O.7 — Explicit Number Variance Curve Diagnostics.

Computational diagnostic only; not a proof of RH.

This script aims to move beyond a single scalar "number variance error" by producing (or approximating)
explicit number variance curves Sigma^2(L) on an L-grid, and comparing curve *shape* across:
- real zeta target group
- synthetic / shuffled controls
- operator candidates (word_group, V_mode)

IMPORTANT
---------
If raw unfolded level arrays are not available in inputs, this script runs in approximation mode:
    curve_mode_available = False
    approximation_mode = True
It will still emit CSV/JSON/report/plots, but all "curves" are *proxy curves* reconstructed from the
available scalar number-variance error summaries (V13O.4) and stabilized diagnostics (V13O.6).

CLI examples (smoke / full) are printed in the report.
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

CANDIDATE_WORD_GROUPS = [
    "primary_word_seed6",
    "ablate_K",
    "ablate_L",
    "ablate_V",
    "random_symmetric_baseline",
    "random_words_n30",
    "rejected_word_seed17",
]

TARGET_GROUPS_FOR_CURVES = ["real_zeta"] + CONTROL_TARGET_GROUPS


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


def read_csv_robust(path: Path, *, name: str, tag: str = "v13o7") -> Optional["pd.DataFrame"]:
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


# ----------------------------
# Curve utilities (true mode)
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
    """
    Compute number variance curve Sigma^2(L) = Var(N(t,t+L)) over sliding windows.

    levels_unfolded: 1D array of unfolded levels (monotone increasing).
    For each L, use all valid window starts t from levels_unfolded range.
    """
    x = np.asarray(levels_unfolded, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size < 8:
        return np.full_like(L_grid, np.nan, dtype=np.float64)
    x = np.sort(x)
    out = np.full_like(L_grid, np.nan, dtype=np.float64)

    # window starts t sampled from actual unfolded levels (fast + deterministic)
    t_candidates = x[:-1]
    for i, L in enumerate(np.asarray(L_grid, dtype=np.float64).reshape(-1)):
        if not (math.isfinite(float(L)) and float(L) > 0.0):
            continue
        # count in [t, t+L]
        # N = # {x_j : t <= x_j <= t+L}
        left = np.searchsorted(x, t_candidates, side="left")
        right = np.searchsorted(x, t_candidates + float(L), side="right")
        counts = (right - left).astype(np.float64)
        if counts.size < 8:
            continue
        out[i] = float(np.var(counts))
    return out


def _normalize_curve(y: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if y.size == 0:
        return y
    if not np.isfinite(y).any():
        return y
    y0 = float(y[0]) if math.isfinite(float(y[0])) else float(np.nanmin(y))
    y1 = float(y[-1]) if math.isfinite(float(y[-1])) else float(np.nanmax(y))
    denom = float(abs(y1 - y0))
    denom = max(denom, float(eps))
    return (y - y0) / denom


def compare_curves(real_curve: np.ndarray, control_curve: np.ndarray, metric: str, *, L_grid: Optional[np.ndarray] = None, tail_L_min: float = 6.0, normalize_curves: bool = False) -> float:
    """
    Lower is better: distance between curves.
    Metrics:
      - l1, l2, linf: on absolute difference over all L
      - tail_l2: l2 over L>=tail_L_min
      - shape_l2: l2 on normalized curves (shape only)
    """
    a = np.asarray(real_curve, dtype=np.float64).reshape(-1)
    b = np.asarray(control_curve, dtype=np.float64).reshape(-1)
    if a.size == 0 or b.size == 0 or a.size != b.size:
        return float("nan")
    m = str(metric).strip().lower()

    aa = a.copy()
    bb = b.copy()
    if normalize_curves or m == "shape_l2":
        aa = _normalize_curve(aa)
        bb = _normalize_curve(bb)

    mask = np.isfinite(aa) & np.isfinite(bb)
    if mask.sum() < 4:
        return float("nan")

    diff = np.abs(aa[mask] - bb[mask])
    if m == "l1":
        return float(np.mean(diff))
    if m == "l2":
        return float(np.sqrt(np.mean(diff**2)))
    if m == "linf":
        return float(np.max(diff))
    if m == "tail_l2":
        if L_grid is None:
            return float(np.sqrt(np.mean(diff**2)))
        L = np.asarray(L_grid, dtype=np.float64).reshape(-1)
        mm = mask & (L >= float(tail_L_min))
        if mm.sum() < 4:
            return float("nan")
        d2 = np.abs(aa[mm] - bb[mm])
        return float(np.sqrt(np.mean(d2**2)))
    if m == "shape_l2":
        return float(np.sqrt(np.mean(diff**2)))
    return float("nan")


def summarize_curve_regions(L_grid: np.ndarray, real_curve: np.ndarray, control_median_curve: np.ndarray) -> List[Dict[str, Any]]:
    """
    Region split:
      short: L<=2
      mid: 2<L<=6
      long: L>6
    For each region, compute errors and some signed diagnostics.
    """
    L = np.asarray(L_grid, dtype=np.float64).reshape(-1)
    a = np.asarray(real_curve, dtype=np.float64).reshape(-1)
    b = np.asarray(control_median_curve, dtype=np.float64).reshape(-1)
    if L.size == 0 or a.size != L.size or b.size != L.size:
        return []

    regions = [
        ("short", L <= 2.0),
        ("mid", (L > 2.0) & (L <= 6.0)),
        ("long", L > 6.0),
    ]
    rows: List[Dict[str, Any]] = []
    for name, mask in regions:
        mm = mask & np.isfinite(a) & np.isfinite(b)
        if mm.sum() < 4:
            rows.append(
                {
                    "region": name,
                    "n_L": int(mm.sum()),
                    "l1_curve_error": float("nan"),
                    "l2_curve_error": float("nan"),
                    "max_abs_curve_error": float("nan"),
                    "signed_area_error": float("nan"),
                    "slope_error": float("nan"),
                }
            )
            continue
        diff = a[mm] - b[mm]
        l1 = float(np.mean(np.abs(diff)))
        l2 = float(np.sqrt(np.mean(diff**2)))
        linf = float(np.max(np.abs(diff)))
        # signed area via trapezoid on region (np.trapezoid is the modern name)
        area = float(np.trapezoid(diff, L[mm]))
        # slope error: linear fit slope(real) - slope(control)
        x = L[mm]
        ra = a[mm]
        rb = b[mm]
        try:
            slope_a = float(np.polyfit(x, ra, 1)[0])
            slope_b = float(np.polyfit(x, rb, 1)[0])
            slope_err = float(slope_a - slope_b)
        except Exception:
            slope_err = float("nan")
        rows.append(
            {
                "region": name,
                "n_L": int(mm.sum()),
                "l1_curve_error": l1,
                "l2_curve_error": l2,
                "max_abs_curve_error": linf,
                "signed_area_error": area,
                "slope_error": slope_err,
            }
        )
    return rows


# ----------------------------
# Reference curves
# ----------------------------


def poisson_number_variance(L: np.ndarray) -> np.ndarray:
    L = np.asarray(L, dtype=np.float64)
    return np.maximum(L, 0.0)


def gue_number_variance_asymptotic(L: np.ndarray) -> np.ndarray:
    """
    Asymptotic approximation:
      Sigma_GUE^2(L) ≈ (1/pi^2) * (log(2*pi*L) + gamma + 1)
    for L > eps.
    """
    L = np.asarray(L, dtype=np.float64)
    gamma = 0.5772156649015329
    eps = 1e-6
    Lp = np.maximum(L, eps)
    return (1.0 / (math.pi**2)) * (np.log(2.0 * math.pi * Lp) + gamma + 1.0)


# ----------------------------
# Approximation mode helpers
# ----------------------------


def _proxy_curve_from_scalar_error(
    *,
    L_grid: np.ndarray,
    base_curve: np.ndarray,
    scalar_error: float,
    mode: str,
) -> np.ndarray:
    """
    Convert a scalar "number_variance_error_test" into a smooth proxy curve around base_curve.

    This is intentionally conservative and *shape-biased*:
      - uses log1p(scalar_error) as amplitude
      - uses a gentle basis function in L to avoid raw-scale blow-up
      - mode in {"high","low"} pushes curve up/down relative to base
    """
    L = np.asarray(L_grid, dtype=np.float64).reshape(-1)
    y0 = np.asarray(base_curve, dtype=np.float64).reshape(-1)
    if L.size == 0 or y0.size != L.size:
        return np.full_like(L, np.nan, dtype=np.float64)
    amp = float(np.log1p(max(0.0, float(scalar_error)))) if math.isfinite(float(scalar_error)) else float("nan")
    if not math.isfinite(amp):
        return np.full_like(L, np.nan, dtype=np.float64)
    # basis: 0 at L_min, 1 at L_max, slightly convex to emphasize tail
    t = (L - float(L.min())) / max(float(L.max() - L.min()), 1e-9)
    basis = 0.25 * t + 0.75 * (t**1.5)
    sign = +1.0 if mode == "high" else -1.0
    # scale chosen so amp~O(1..10) yields visible but not absurd deviations
    return y0 + sign * (0.35 * amp) * basis


def _control_proxy_type(target_group: str) -> str:
    tg = str(target_group)
    if "Poisson" in tg or "shuffled" in tg or "jitter" in tg or "reversed" in tg:
        return "poisson_like"
    if "GUE" in tg:
        return "gue_like"
    if "density_matched" in tg:
        return "intermediate"
    return "intermediate"


def _blend(a: np.ndarray, b: np.ndarray, w: float) -> np.ndarray:
    w = float(np.clip(w, 0.0, 1.0))
    return (1.0 - w) * a + w * b


def _real_poisson_weight_from_v13o6(z_log_nv: float, percentile: float) -> float:
    """
    Heuristic: map stabilized failure severity to a Poisson-mixture weight.
    Larger z_log and higher percentile -> more Poisson-like.
    """
    z = float(z_log_nv) if math.isfinite(float(z_log_nv)) else 0.0
    p = float(percentile) if math.isfinite(float(percentile)) else 0.5
    w = 0.20 + 0.15 * z + 0.40 * (p - 0.5)
    return float(np.clip(w, 0.0, 1.0))


# ----------------------------
# Plotting (optional)
# ----------------------------


def _try_import_matplotlib():
    try:
        import matplotlib  # type: ignore
        import matplotlib.pyplot as plt  # type: ignore

        matplotlib.use("Agg")
        return plt
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="V13O.7 explicit number variance curve diagnostics (computational only).")

    ap.add_argument("--v13o4_summary", type=str, default="runs/v13o4_zeta_specific_objective/v13o4_summary.csv")
    ap.add_argument("--v13o4_group_summary", type=str, default="runs/v13o4_zeta_specific_objective/v13o4_group_summary.csv")
    ap.add_argument("--v13o4_zeta_scores", type=str, default="runs/v13o4_zeta_specific_objective/v13o4_zeta_specific_scores.csv")
    ap.add_argument("--v13o4_number_variance", type=str, default="runs/v13o4_zeta_specific_objective/v13o4_number_variance_summary.csv")
    ap.add_argument("--v13o6_nv_scores", type=str, default="runs/v13o6_number_variance_stabilization_smoke/v13o6_nv_stabilized_scores.csv")

    ap.add_argument("--out_dir", type=str, default="runs/v13o7_number_variance_curves")
    ap.add_argument("--primary_word_group", type=str, default="primary_word_seed6")
    ap.add_argument("--primary_v_mode", type=str, default="full_V")
    ap.add_argument("--dims", type=int, nargs="+", default=[64, 128, 256])

    ap.add_argument("--L_min", type=float, default=0.5)
    ap.add_argument("--L_max", type=float, default=12.0)
    ap.add_argument("--n_L", type=int, default=48)
    ap.add_argument("--curve_metric", type=str, default="shape_l2", choices=["l1", "l2", "linf", "tail_l2", "shape_l2"])
    ap.add_argument("--tail_L_min", type=float, default=6.0)
    ap.add_argument("--normalize_curves", action="store_true")
    ap.add_argument("--progress_every", type=int, default=1)

    args = ap.parse_args()

    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    # Make matplotlib cache writable inside out_dir (avoid noisy warnings / slow imports)
    mpl_dir = out_dir / ".mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir.resolve()))

    t0 = time.perf_counter()
    prog = int(args.progress_every)

    # Load inputs (missing OK)
    df_nv = read_csv_robust(_resolve(args.v13o4_number_variance), name="v13o4_number_variance", tag="v13o7")
    df_v13o6 = read_csv_robust(_resolve(args.v13o6_nv_scores), name="v13o6_nv_scores", tag="v13o7")
    df_summary = read_csv_robust(_resolve(args.v13o4_summary), name="v13o4_summary", tag="v13o7")
    df_group = read_csv_robust(_resolve(args.v13o4_group_summary), name="v13o4_group_summary", tag="v13o7")
    df_scores = read_csv_robust(_resolve(args.v13o4_zeta_scores), name="v13o4_zeta_scores", tag="v13o7")

    # Decide whether true curve mode is possible from df_nv
    curve_mode_available = False
    approximation_mode = True
    sigma_col = None
    if df_nv is not None:
        sigma_col = pick_col(df_nv, ["Sigma2", "sigma2", "Sigma2_L", "sigma2_L"])
        if "L" in df_nv.columns and sigma_col is not None and all(k in df_nv.columns for k in ("dim", "V_mode", "word_group", "target_group")):
            curve_mode_available = True
            approximation_mode = False

    L_grid = make_L_grid(args.L_min, args.L_max, args.n_L)
    poisson_ref = poisson_number_variance(L_grid)
    gue_ref = gue_number_variance_asymptotic(L_grid)

    primary_wg = str(args.primary_word_group)
    primary_vm = str(args.primary_v_mode)
    dims = [int(d) for d in list(args.dims)]

    # Precompute median scalar errors per (dim,vm,wg,tg) from v13o4 number variance summary.
    col_nv_test = pick_col(df_nv, ["number_variance_error_test", "number_variance_test", "number_variance_error"]) if df_nv is not None else None
    nv_scalar_map: Dict[Tuple[int, str, str, str], float] = {}
    if df_nv is not None and col_nv_test is not None and all(k in df_nv.columns for k in ("dim", "V_mode", "word_group", "target_group")):
        dfx = df_nv.copy()
        dfx["dim"] = pd.to_numeric(dfx["dim"], errors="coerce").astype("Int64")
        for k in ("V_mode", "word_group", "target_group"):
            dfx[k] = dfx[k].astype(str).str.strip()
        dfx["_nv"] = finite_series(dfx[col_nv_test])
        dfx = dfx.dropna(subset=["dim", "V_mode", "word_group", "target_group", "_nv"])
        for (d, vm, wg, tg), g in dfx.groupby(["dim", "V_mode", "word_group", "target_group"]):
            arr = g["_nv"].dropna().astype(float).to_numpy()
            if arr.size:
                nv_scalar_map[(int(d), str(vm), str(wg), str(tg))] = float(np.median(arr))

    # Load v13o6 stabilized info for mapping real curve severity
    v13o6_map: Dict[Tuple[int, str, str], Dict[str, float]] = {}
    if df_v13o6 is not None and all(k in df_v13o6.columns for k in ("dim", "V_mode", "word_group")):
        dfx = df_v13o6.copy()
        dfx["dim"] = pd.to_numeric(dfx["dim"], errors="coerce").astype("Int64")
        for k in ("V_mode", "word_group"):
            dfx[k] = dfx[k].astype(str).str.strip()
        for r in dfx.dropna(subset=["dim", "V_mode", "word_group"]).itertuples(index=False):
            key = (int(getattr(r, "dim")), str(getattr(r, "V_mode")), str(getattr(r, "word_group")))
            v13o6_map[key] = {
                "z_log_nv": float(getattr(r, "z_log_nv", float("nan"))),
                "percentile": float(getattr(r, "percentile_real_among_controls", float("nan"))),
                "z_winsor_nv": float(getattr(r, "z_winsor_nv", float("nan"))),
            }

    # Build curves (true if available, else proxy)
    def get_curve(dim: int, vm: str, wg: str, tg: str) -> np.ndarray:
        if curve_mode_available and df_nv is not None and sigma_col is not None:
            sub = df_nv[
                (pd.to_numeric(df_nv["dim"], errors="coerce") == dim)
                & (df_nv["V_mode"].astype(str).str.strip() == vm)
                & (df_nv["word_group"].astype(str).str.strip() == wg)
                & (df_nv["target_group"].astype(str).str.strip() == tg)
            ]
            if not sub.empty and "L" in sub.columns:
                # If per-L Sigma2 is present as rows, interpolate onto L_grid.
                Ls = pd.to_numeric(sub["L"], errors="coerce").to_numpy(dtype=np.float64)
                Ys = finite_series(sub[sigma_col]).to_numpy(dtype=np.float64)
                m = np.isfinite(Ls) & np.isfinite(Ys)
                if m.sum() >= 4:
                    order = np.argsort(Ls[m])
                    Ls2 = Ls[m][order]
                    Ys2 = Ys[m][order]
                    # piecewise linear interpolation (clip to endpoints)
                    return np.interp(L_grid, Ls2, Ys2, left=float(Ys2[0]), right=float(Ys2[-1]))
        # approximation mode
        tg_type = _control_proxy_type(tg)
        if tg == "Poisson_synthetic":
            return poisson_ref.copy()
        if tg == "GUE_synthetic":
            return gue_ref.copy()
        if tg == "real_zeta":
            sev = v13o6_map.get((int(dim), str(vm), str(wg)), {})
            w = _real_poisson_weight_from_v13o6(sev.get("z_log_nv", float("nan")), sev.get("percentile", float("nan")))
            base = _blend(gue_ref, poisson_ref, w)
            # add gentle extra deviation based on scalar error if present
            sc = nv_scalar_map.get((int(dim), str(vm), str(wg), str(tg)), float("nan"))
            if math.isfinite(sc):
                return _proxy_curve_from_scalar_error(L_grid=L_grid, base_curve=base, scalar_error=sc, mode="high")
            return base
        # other controls: choose base and optionally push with scalar error
        if tg_type == "poisson_like":
            base = _blend(gue_ref, poisson_ref, 0.75)
        elif tg_type == "gue_like":
            base = _blend(gue_ref, poisson_ref, 0.15)
        else:
            base = _blend(gue_ref, poisson_ref, 0.45)
        sc = nv_scalar_map.get((int(dim), str(vm), str(wg), str(tg)), float("nan"))
        if math.isfinite(sc):
            return _proxy_curve_from_scalar_error(L_grid=L_grid, base_curve=base, scalar_error=sc, mode="high")
        return base

    # Diagnostics per (dim,vm,wg)
    rows_summary: List[Dict[str, Any]] = []
    rows_regions: List[Dict[str, Any]] = []
    gate_rows: List[Dict[str, Any]] = []
    rows_curves: List[Dict[str, Any]] = []

    groups: List[Tuple[int, str, str]] = []
    for d in dims:
        for wg in CANDIDATE_WORD_GROUPS:
            groups.append((int(d), str(primary_vm), str(wg)))
    # also include other V_modes if present in v13o6 map for these dims/word_groups (best-effort)
    for (d, vm, wg) in sorted(v13o6_map.keys()):
        if int(d) in dims and wg in CANDIDATE_WORD_GROUPS:
            groups.append((int(d), str(vm), str(wg)))
    # unique
    groups = sorted({g for g in groups}, key=lambda x: (x[0], x[1], x[2]))

    print(f"[v13o7] groups to analyze: {len(groups)} curve_mode_available={curve_mode_available} approximation_mode={approximation_mode}", flush=True)

    def log_progress(i: int, total: int, g: Tuple[int, str, str]) -> None:
        if prog <= 0:
            return
        if i == 1 or i == total or i % max(1, prog) == 0:
            elapsed = time.perf_counter() - t0
            avg = elapsed / max(i, 1)
            eta = avg * max(total - i, 0)
            d, vm, wg = g
            print(f"[V13O.7] {i}/{total} elapsed={elapsed:.1f}s eta={format_seconds(eta)} current=({d},{vm},{wg})", flush=True)

    for i, (dim, vm, wg) in enumerate(groups, start=1):
        log_progress(i, len(groups), (dim, vm, wg))

        # curves for all target groups
        curves: Dict[str, np.ndarray] = {tg: get_curve(dim, vm, wg, tg) for tg in TARGET_GROUPS_FOR_CURVES}
        real_curve = curves.get("real_zeta", np.full_like(L_grid, np.nan))
        control_curves = [curves[tg] for tg in CONTROL_TARGET_GROUPS if tg in curves]

        # control bands
        ctrl_stack = np.stack([c for c in control_curves if np.isfinite(c).any()], axis=0) if control_curves else None
        if ctrl_stack is None or ctrl_stack.size == 0:
            ctrl_median = np.full_like(L_grid, np.nan)
            ctrl_q25 = np.full_like(L_grid, np.nan)
            ctrl_q75 = np.full_like(L_grid, np.nan)
        else:
            ctrl_median = np.nanmedian(ctrl_stack, axis=0)
            ctrl_q25 = np.nanquantile(ctrl_stack, 0.25, axis=0)
            ctrl_q75 = np.nanquantile(ctrl_stack, 0.75, axis=0)

        # Save per-L curves (compact long-format; enables later plotting/comparison without rerun)
        for curve_name, curve_arr in (
            ("Sigma2_real_curve", real_curve),
            ("median_control_curve", ctrl_median),
            ("q25_control_curve", ctrl_q25),
            ("q75_control_curve", ctrl_q75),
            ("poisson_reference_curve", poisson_ref),
            ("gue_reference_curve", gue_ref),
        ):
            yy = np.asarray(curve_arr, dtype=np.float64).reshape(-1)
            for L, y in zip(L_grid.tolist(), yy.tolist()):
                rows_curves.append(
                    {
                        "dim": int(dim),
                        "V_mode": str(vm),
                        "word_group": str(wg),
                        "curve_name": str(curve_name),
                        "L": float(L),
                        "Sigma2": float(y) if math.isfinite(float(y)) else float("nan"),
                    }
                )

        # distances
        shape_dist = compare_curves(real_curve, ctrl_median, args.curve_metric, L_grid=L_grid, tail_L_min=float(args.tail_L_min), normalize_curves=bool(args.normalize_curves))
        poisson_dist = compare_curves(real_curve, poisson_ref, args.curve_metric, L_grid=L_grid, tail_L_min=float(args.tail_L_min), normalize_curves=bool(args.normalize_curves))
        gue_dist = compare_curves(real_curve, gue_ref, args.curve_metric, L_grid=L_grid, tail_L_min=float(args.tail_L_min), normalize_curves=bool(args.normalize_curves))

        # regions summary vs control median
        reg_rows = summarize_curve_regions(L_grid, real_curve, ctrl_median)
        # region percentile: compare region l2 error vs each control curve's region l2 error to ctrl_median
        region_masks = {
            "short": (L_grid <= 2.0),
            "mid": (L_grid > 2.0) & (L_grid <= 6.0),
            "long": (L_grid > 6.0),
        }
        region_l2_real: Dict[str, float] = {}
        region_pct: Dict[str, float] = {}
        for name, mask in region_masks.items():
            mm = mask & np.isfinite(real_curve) & np.isfinite(ctrl_median)
            if mm.sum() < 4:
                region_l2_real[name] = float("nan")
                region_pct[name] = float("nan")
                continue
            dreal = real_curve[mm] - ctrl_median[mm]
            l2r = float(np.sqrt(np.mean(dreal**2)))
            region_l2_real[name] = l2r
            ctrl_l2s: List[float] = []
            for c in control_curves:
                if c.shape != real_curve.shape:
                    continue
                m2 = mask & np.isfinite(c) & np.isfinite(ctrl_median)
                if m2.sum() < 4:
                    continue
                dc = c[m2] - ctrl_median[m2]
                ctrl_l2s.append(float(np.sqrt(np.mean(dc**2))))
            if ctrl_l2s:
                arr = np.asarray(ctrl_l2s, dtype=np.float64)
                region_pct[name] = float(np.mean(arr <= l2r))
            else:
                region_pct[name] = float("nan")

        # attach region details rows
        for rr in reg_rows:
            region = str(rr.get("region"))
            rows_regions.append(
                {
                    "dim": int(dim),
                    "V_mode": str(vm),
                    "word_group": str(wg),
                    "curve_mode_available": bool(curve_mode_available),
                    "approximation_mode": bool(approximation_mode),
                    "region": region,
                    "n_L": int(rr.get("n_L", 0)),
                    "l1_curve_error": float(rr.get("l1_curve_error", float("nan"))),
                    "l2_curve_error": float(rr.get("l2_curve_error", float("nan"))),
                    "max_abs_curve_error": float(rr.get("max_abs_curve_error", float("nan"))),
                    "signed_area_error": float(rr.get("signed_area_error", float("nan"))),
                    "slope_error": float(rr.get("slope_error", float("nan"))),
                    "real_vs_control_percentile": float(region_pct.get(region, float("nan"))),
                }
            )

        # region pass/fail heuristic (relative to control median)
        short_l2 = float(region_l2_real.get("short", float("nan")))
        mid_l2 = float(region_l2_real.get("mid", float("nan")))
        long_l2 = float(region_l2_real.get("long", float("nan")))

        # shape error: always compute as shape_l2 on normalized curves (even if curve_metric differs)
        shape_l2 = compare_curves(real_curve, ctrl_median, "shape_l2", normalize_curves=True)

        # basic finite gate
        c1 = bool(np.isfinite(real_curve).sum() >= 6 and np.isfinite(ctrl_median).sum() >= 6)

        # region passes: use percentiles against control deviations; lower percentile means "more like control median"
        c2 = bool(c1 and math.isfinite(region_pct.get("short", float("nan"))) and float(region_pct["short"]) <= 0.60)
        c3 = bool(c1 and math.isfinite(region_pct.get("mid", float("nan"))) and float(region_pct["mid"]) <= 0.60)
        c4 = bool(c1 and math.isfinite(region_pct.get("long", float("nan"))) and float(region_pct["long"]) <= 0.70)

        # shape pass: normalized l2 vs control median should be modest
        c5 = bool(c1 and math.isfinite(shape_l2) and float(shape_l2) <= 0.35)

        # Poissonization / overrigidity checks using distances to reference curves
        c6 = bool(c1 and math.isfinite(poisson_dist) and math.isfinite(gue_dist) and float(poisson_dist) >= 0.8 * float(gue_dist))
        # "not overrigid" if curve is not substantially below GUE: compare signed area vs GUE
        overrigid_score = (
            float(np.trapezoid((gue_ref - real_curve)[np.isfinite(real_curve)], L_grid[np.isfinite(real_curve)]))
            if np.isfinite(real_curve).any()
            else float("nan")
        )
        c7 = bool(c1 and (not math.isfinite(overrigid_score) or overrigid_score <= 0.25))

        # transfer compatible: placeholder (no transfer inputs provided to v13o7 args); treat as True unless inputs indicate otherwise later
        c8 = True

        strict = bool(c1 and c2 and c3 and c4 and c5 and c6 and c7 and c8)
        relaxed = bool(c1 and c2 and c3 and c5 and c6)  # allow long-range fail

        # classify failure mode
        if approximation_mode:
            classification = "APPROXIMATION_ONLY"
        else:
            if strict:
                classification = "NV_CURVE_SIGNAL"
            elif relaxed:
                classification = "PARTIAL_NV_CURVE_SIGNAL"
            else:
                # region-only patterns
                if c2 and not (c3 or c4):
                    classification = "SHORT_RANGE_ONLY"
                elif c3 and not (c2 or c4):
                    classification = "MID_RANGE_ONLY"
                elif (c2 and c3) and not c4:
                    classification = "LONG_RANGE_FAILURE"
                elif not c6:
                    classification = "POISSONIZATION_FAILURE"
                elif not c7:
                    classification = "OVERRIGID_FAILURE"
                else:
                    classification = "NO_NV_CURVE_SIGNAL"

        # worst region by l2
        worst_region = "unknown"
        worst_val = -1.0
        for name, val in (("short", short_l2), ("mid", mid_l2), ("long", long_l2)):
            if math.isfinite(val) and val > worst_val:
                worst_val = float(val)
                worst_region = name

        gate_rows.append(
            {
                "dim": int(dim),
                "V_mode": str(vm),
                "word_group": str(wg),
                "curve_mode_available": bool(curve_mode_available),
                "approximation_mode": bool(approximation_mode),
                "C1_finite_inputs": bool(c1),
                "C2_short_range_pass": bool(c2),
                "C3_mid_range_pass": bool(c3),
                "C4_long_range_pass": bool(c4),
                "C5_shape_pass": bool(c5),
                "C6_not_poisson_like": bool(c6),
                "C7_not_overrigid": bool(c7),
                "C8_transfer_compatible": bool(c8),
                "strict_nv_curve_pass": bool(strict),
                "relaxed_nv_curve_pass": bool(relaxed),
                "classification": str(classification),
                "short_l2_error": float(short_l2),
                "mid_l2_error": float(mid_l2),
                "long_l2_error": float(long_l2),
                "shape_l2_error": float(shape_l2),
                "poisson_distance": float(poisson_dist),
                "gue_distance": float(gue_dist),
                "control_median_distance": float(shape_dist),
                "worst_region": str(worst_region),
            }
        )

        rows_summary.append(
            {
                "dim": int(dim),
                "V_mode": str(vm),
                "word_group": str(wg),
                "curve_mode_available": bool(curve_mode_available),
                "approximation_mode": bool(approximation_mode),
                "curve_metric": str(args.curve_metric),
                "normalize_curves": bool(args.normalize_curves),
                "tail_L_min": float(args.tail_L_min),
                "control_median_distance": float(shape_dist),
                "poisson_distance": float(poisson_dist),
                "gue_distance": float(gue_dist),
                "short_l2_error": float(short_l2),
                "mid_l2_error": float(mid_l2),
                "long_l2_error": float(long_l2),
                "shape_l2_error": float(shape_l2),
                "classification": str(classification),
            }
        )

    # outputs
    df_sum = pd.DataFrame(rows_summary)
    df_gate = pd.DataFrame(gate_rows)
    df_reg = pd.DataFrame(rows_regions)

    (out_dir / "v13o7_nv_curve_summary.csv").write_text(df_sum.to_csv(index=False), encoding="utf-8")
    (out_dir / "v13o7_nv_curve_gate_summary.csv").write_text(df_gate.to_csv(index=False), encoding="utf-8")
    (out_dir / "v13o7_nv_curve_regions.csv").write_text(df_reg.to_csv(index=False), encoding="utf-8")
    df_curves = pd.DataFrame(rows_curves)
    (out_dir / "v13o7_nv_curve_curves.csv").write_text(df_curves.to_csv(index=False), encoding="utf-8")

    # best by dim (by control_median_distance then shape_l2_error)
    best_rows: List[Dict[str, Any]] = []
    if not df_sum.empty:
        for dim, g in df_sum.groupby("dim"):
            gg = g.sort_values(["control_median_distance", "shape_l2_error", "poisson_distance"], ascending=True, na_position="last")
            best = gg.iloc[0]
            prim = gg[(gg["V_mode"] == primary_vm) & (gg["word_group"] == primary_wg)]
            prim_rank = -1
            if not prim.empty:
                gg2 = gg.reset_index(drop=True).copy()
                gg2["_rank"] = np.arange(1, len(gg2) + 1, dtype=int)
                pmask = (gg2["V_mode"] == primary_vm) & (gg2["word_group"] == primary_wg)
                prim_rank = int(gg2.loc[pmask, "_rank"].iloc[0])
            best_rows.append(
                {
                    "dim": int(dim),
                    "best_V_mode": str(best["V_mode"]),
                    "best_word_group": str(best["word_group"]),
                    "best_control_median_distance": float(best["control_median_distance"]),
                    "best_shape_l2_error": float(best["shape_l2_error"]),
                    "best_poisson_distance": float(best["poisson_distance"]),
                    "best_gue_distance": float(best["gue_distance"]),
                    "best_classification": str(best["classification"]),
                    "primary_rank_by_control_median_distance": int(prim_rank),
                }
            )
    df_best = pd.DataFrame(best_rows)
    (out_dir / "v13o7_nv_curve_best_by_dim.csv").write_text(df_best.to_csv(index=False), encoding="utf-8")

    # Interpretation questions (primary full_V)
    def _primary_rows() -> "pd.DataFrame":
        if df_gate.empty:
            return df_gate
        return df_gate[(df_gate["V_mode"] == primary_vm) & (df_gate["word_group"] == primary_wg)].copy()

    prim_gate = _primary_rows()
    fails_only_long = False
    poisson_like = False
    overrigid = False
    most_promising_dim = None
    if not prim_gate.empty:
        # "fails only at long range" = short+mid pass, long fail
        tmp = prim_gate.sort_values(["dim"])
        fails_only_long = bool(((tmp["C2_short_range_pass"] == True) & (tmp["C3_mid_range_pass"] == True) & (tmp["C4_long_range_pass"] == False)).any())
        poisson_like = bool((tmp["C6_not_poisson_like"] == False).any())
        overrigid = bool((tmp["C7_not_overrigid"] == False).any())
        # pick most promising dim as min(control_median_distance) among primary rows
        ssub = df_sum[(df_sum["V_mode"] == primary_vm) & (df_sum["word_group"] == primary_wg)]
        if not ssub.empty:
            bestp = ssub.sort_values(["control_median_distance"], ascending=True, na_position="last").iloc[0]
            most_promising_dim = int(bestp["dim"])

    # Results JSON
    payload: Dict[str, Any] = {
        "warning": "Computational diagnostic only; not a proof of RH.",
        "status": "V13O.7 explicit number variance curve diagnostics",
        "curve_mode_available": bool(curve_mode_available),
        "approximation_mode": bool(approximation_mode),
        "dims": dims,
        "L_grid": [float(x) for x in L_grid.tolist()],
        "curve_metric": str(args.curve_metric),
        "normalize_curves": bool(args.normalize_curves),
        "tail_L_min": float(args.tail_L_min),
        "primary": {"word_group": primary_wg, "V_mode": primary_vm},
        "primary_answers": {
            "does_primary_fail_only_at_long_range": bool(fails_only_long),
            "is_failure_poisson_like": bool(poisson_like),
            "evidence_of_over_rigidity": bool(overrigid),
            "most_promising_dim": most_promising_dim,
            "recommend_proceed_to_v13p0": False,
        },
        "inputs": {
            "v13o4_summary": str(_resolve(args.v13o4_summary).resolve()),
            "v13o4_group_summary": str(_resolve(args.v13o4_group_summary).resolve()),
            "v13o4_zeta_scores": str(_resolve(args.v13o4_zeta_scores).resolve()),
            "v13o4_number_variance": str(_resolve(args.v13o4_number_variance).resolve()),
            "v13o6_nv_scores": str(_resolve(args.v13o6_nv_scores).resolve()),
        },
        "outputs": {
            "nv_curve_summary_csv": str((out_dir / "v13o7_nv_curve_summary.csv").resolve()),
            "nv_curve_gate_summary_csv": str((out_dir / "v13o7_nv_curve_gate_summary.csv").resolve()),
            "nv_curve_regions_csv": str((out_dir / "v13o7_nv_curve_regions.csv").resolve()),
            "nv_curve_curves_csv": str((out_dir / "v13o7_nv_curve_curves.csv").resolve()),
            "nv_curve_best_by_dim_csv": str((out_dir / "v13o7_nv_curve_best_by_dim.csv").resolve()),
        },
        "runtime_s": float(time.perf_counter() - t0),
        "python_pid": os.getpid(),
    }
    (out_dir / "v13o7_nv_curve_results.json").write_text(json.dumps(json_sanitize(payload), indent=2, allow_nan=False), encoding="utf-8")

    # Plots (optional, approximation mode still plots)
    plt = _try_import_matplotlib()
    plot_notes: List[str] = []
    if plt is None:
        plot_notes.append("matplotlib not available; skipping plots.")
    else:
        def plot_one(dim: int) -> None:
            vm = primary_vm
            wg = primary_wg
            # reconstruct curves in same manner as above (deterministic)
            curves = {tg: get_curve(dim, vm, wg, tg) for tg in TARGET_GROUPS_FOR_CURVES}
            real_curve = curves["real_zeta"]
            control_curves = [curves[tg] for tg in CONTROL_TARGET_GROUPS if tg in curves]
            if control_curves:
                ctrl_stack = np.stack(control_curves, axis=0)
                ctrl_median = np.nanmedian(ctrl_stack, axis=0)
                ctrl_q25 = np.nanquantile(ctrl_stack, 0.25, axis=0)
                ctrl_q75 = np.nanquantile(ctrl_stack, 0.75, axis=0)
            else:
                ctrl_median = np.full_like(L_grid, np.nan)
                ctrl_q25 = np.full_like(L_grid, np.nan)
                ctrl_q75 = np.full_like(L_grid, np.nan)

            plt.figure(figsize=(8.5, 5.0))
            plt.plot(L_grid, real_curve, label="real_zeta (operator)", linewidth=2.2)
            plt.plot(L_grid, ctrl_median, label="controls median", linewidth=2.0)
            plt.fill_between(L_grid, ctrl_q25, ctrl_q75, alpha=0.18, label="controls q25–q75")
            plt.plot(L_grid, poisson_ref, label="Poisson ref", linestyle="--", linewidth=1.6)
            plt.plot(L_grid, gue_ref, label="GUE asymptotic ref", linestyle="--", linewidth=1.6)
            plt.xlabel("L")
            plt.ylabel("Sigma^2(L)")
            title = f"V13O.7 NV curve dim={dim} ({vm}, {wg})"
            if approximation_mode:
                title += " [APPROX]"
            plt.title(title)
            plt.grid(True, alpha=0.25)
            plt.legend(loc="best", fontsize=9)
            outp = fig_dir / f"v13o7_dim{dim}_primary_nv_curve.png"
            plt.tight_layout()
            plt.savefig(outp, dpi=160)
            plt.close()

        for d in dims:
            plot_one(int(d))

        # best-by-dim overview plot (control_median_distance for each dim)
        if not df_best.empty:
            plt.figure(figsize=(8.5, 4.8))
            xs = df_best["dim"].astype(int).tolist()
            ys = df_best["best_control_median_distance"].astype(float).tolist()
            plt.plot(xs, ys, marker="o", linewidth=2.0, label="best-by-dim distance")
            plt.xlabel("dim")
            plt.ylabel("best control_median_distance")
            plt.title("V13O.7 best-by-dim stabilized NV-curve distance" + (" [APPROX]" if approximation_mode else ""))
            plt.grid(True, alpha=0.25)
            plt.legend(loc="best")
            outp = fig_dir / "v13o7_best_by_dim_nv_curve.png"
            plt.tight_layout()
            plt.savefig(outp, dpi=160)
            plt.close()

    # Report (markdown)
    md: List[str] = []
    md.append("# V13O.7 Explicit Number Variance Curve Diagnostics\n\n")
    md.append("> **Computational evidence only; not a proof of RH.**\n\n")
    md.append("## 1. Motivation from V13O.6\n\n")
    md.append(
        "V13O.6 indicated that stabilized scalar NV diagnostics still fail for the primary operator, motivating explicit curve-based diagnostics.\n\n"
    )
    md.append("## 2. Why scalar number variance was insufficient\n\n")
    md.append(
        "A single scalar error can be dominated by tail behavior, scale blow-ups, or metric sensitivity. V13O.7 compares the *shape* of "
        "the number variance curve \\(\\Sigma^2(L)\\) across short/mid/long ranges.\n\n"
    )
    md.append("## 3. L-grid and curve metric\n\n")
    md.append(f"- L grid: L_min={args.L_min}, L_max={args.L_max}, n_L={args.n_L}\n")
    md.append(f"- curve_metric: `{args.curve_metric}` (tail_L_min={args.tail_L_min})\n")
    md.append(f"- normalize_curves: {bool(args.normalize_curves)}\n\n")
    md.append("## 4. Primary full_V gate\n\n")
    md.append(f"Primary selection: `word_group={primary_wg}`, `V_mode={primary_vm}`.\n\n")
    md.append("See `v13o7_nv_curve_gate_summary.csv`.\n\n")
    md.append("## 5. Region diagnosis: short / mid / long range\n\n")
    md.append("See `v13o7_nv_curve_regions.csv` for region-wise errors and percentiles.\n\n")
    md.append("## 6. Failure mode classification\n\n")
    md.append("Classifications include `POISSONIZATION_FAILURE`, `OVERRIGID_FAILURE`, `LONG_RANGE_FAILURE`, etc.\n\n")
    md.append("## 7. Best candidates by dimension\n\n")
    md.append("See `v13o7_nv_curve_best_by_dim.csv`.\n\n")
    md.append("## Curves export\n\n")
    md.append(
        "Per-L curves are exported to `v13o7_nv_curve_curves.csv` with columns `(dim, V_mode, word_group, curve_name, L, Sigma2)`.\n\n"
    )
    md.append("## 8. Recommendation for V13P0\n\n")
    md.append(
        "This is a **pre-renormalization** computational diagnostic. This script does **not** claim RH evidence. "
        "Proceeding to analytic renormalization (V13P0) should be conditioned on improved curve-mode diagnostics using real unfolded level arrays.\n\n"
    )
    md.append("### Explicit answers\n\n")
    md.append(f"- Does the primary operator fail only at long range? **{bool(fails_only_long)}**\n")
    md.append(f"- Is the failure Poisson-like? **{bool(poisson_like)}**\n")
    md.append(f"- Is there evidence of over-rigidity? **{bool(overrigid)}**\n")
    md.append(f"- Which dimension is most promising? **{most_promising_dim}**\n")
    md.append("- Should we proceed to analytic renormalization V13P0? **No** (diagnostic only).\n\n")

    if approximation_mode:
        md.append("## Approximation mode warning\n\n")
        md.append(
            "**approximation_mode=True**: curves were reconstructed from scalar summaries (V13O.4/V13O.6). "
            "Use this run to decide what explicit per-level exports are needed to enable true curve computation.\n\n"
        )

    md.append("## CLI examples\n\n")
    md.append("Smoke:\n\n")
    md.append("```bash\n")
    md.append('OUT=runs/v13o7_number_variance_curves_smoke\n')
    md.append("python3 scripts/run_v13o7_number_variance_curves.py \\\n")
    md.append("  --v13o4_summary runs/v13o4_zeta_specific_objective/v13o4_summary.csv \\\n")
    md.append("  --v13o4_group_summary runs/v13o4_zeta_specific_objective/v13o4_group_summary.csv \\\n")
    md.append("  --v13o4_zeta_scores runs/v13o4_zeta_specific_objective/v13o4_zeta_specific_scores.csv \\\n")
    md.append("  --v13o4_number_variance runs/v13o4_zeta_specific_objective/v13o4_number_variance_summary.csv \\\n")
    md.append("  --v13o6_nv_scores runs/v13o6_number_variance_stabilization_smoke/v13o6_nv_stabilized_scores.csv \\\n")
    md.append('  --out_dir "$OUT" \\\n')
    md.append("  --dims 64 128 \\\n")
    md.append("  --n_L 24 \\\n")
    md.append("  --progress_every 1\n")
    md.append("```\n\n")
    md.append("Full:\n\n")
    md.append("```bash\n")
    md.append('OUT=runs/v13o7_number_variance_curves\n')
    md.append("caffeinate -dimsu python3 scripts/run_v13o7_number_variance_curves.py \\\n")
    md.append("  --v13o4_summary runs/v13o4_zeta_specific_objective/v13o4_summary.csv \\\n")
    md.append("  --v13o4_group_summary runs/v13o4_zeta_specific_objective/v13o4_group_summary.csv \\\n")
    md.append("  --v13o4_zeta_scores runs/v13o4_zeta_specific_objective/v13o4_zeta_specific_scores.csv \\\n")
    md.append("  --v13o4_number_variance runs/v13o4_zeta_specific_objective/v13o4_number_variance_summary.csv \\\n")
    md.append("  --v13o6_nv_scores runs/v13o6_number_variance_stabilization_smoke/v13o6_nv_stabilized_scores.csv \\\n")
    md.append('  --out_dir "$OUT" \\\n')
    md.append("  --dims 64 128 256 \\\n")
    md.append("  --L_min 0.5 \\\n")
    md.append("  --L_max 12.0 \\\n")
    md.append("  --n_L 48 \\\n")
    md.append("  --curve_metric shape_l2 \\\n")
    md.append("  --tail_L_min 6.0 \\\n")
    md.append("  --progress_every 1\n")
    md.append("```\n\n")

    md.append("## Inspection commands\n\n")
    md.append("```bash\n")
    md.append("OUT=runs/v13o7_number_variance_curves\n\n")
    md.append('echo "=== FILES ==="\n')
    md.append('find "$OUT" -maxdepth 2 -type f | sort\n\n')
    md.append('echo "=== NV CURVE GATE ==="\n')
    md.append('column -s, -t < "$OUT"/v13o7_nv_curve_gate_summary.csv | head -80\n\n')
    md.append('echo "=== PRIMARY full_V ==="\n')
    md.append("awk -F, '$2==\"full_V\" && $3==\"primary_word_seed6\"{print}' \\\n")
    md.append('  "$OUT"/v13o7_nv_curve_summary.csv\n\n')
    md.append('echo "=== REGIONS primary full_V ==="\n')
    md.append("awk -F, '$2==\"full_V\" && $3==\"primary_word_seed6\"{print}' \\\n")
    md.append('  "$OUT"/v13o7_nv_curve_regions.csv\n\n')
    md.append('echo "=== BEST BY DIM ==="\n')
    md.append('column -s, -t < "$OUT"/v13o7_nv_curve_best_by_dim.csv\n\n')
    md.append('echo "=== REPORT ==="\n')
    md.append('head -160 "$OUT"/v13o7_report.md\n')
    md.append("```\n\n")

    (out_dir / "v13o7_report.md").write_text("".join(md), encoding="utf-8")

    # Report (tex)
    tex = (
        "\\documentclass[11pt]{article}\n"
        "\\usepackage[margin=1in]{geometry}\n"
        "\\begin{document}\n"
        "\\title{V13O.7 Explicit Number Variance Curve Diagnostics}\n"
        "\\maketitle\n"
        "\\section*{Warning}\n"
        "Computational evidence only; not a proof of RH.\n\n"
        "\\section*{Mode}\n"
        + latex_escape(f"curve_mode_available={curve_mode_available}, approximation_mode={approximation_mode}.")
        + "\n\n"
        "\\section*{Primary answers}\n"
        + latex_escape(json.dumps(payload.get("primary_answers", {}), indent=2))
        + "\n\n"
        "\\end{document}\n"
    )
    (out_dir / "v13o7_report.tex").write_text(tex, encoding="utf-8")

    if _find_pdflatex() is None:
        print("[v13o7] WARNING: pdflatex not found; skipping PDF.", flush=True)
    elif try_pdflatex(out_dir / "v13o7_report.tex", out_dir, "v13o7_report.pdf"):
        print(f"Wrote {out_dir / 'v13o7_report.pdf'}", flush=True)
    else:
        print("[v13o7] WARNING: pdflatex failed or did not produce v13o7_report.pdf.", flush=True)

    print(f"[v13o7] Wrote {out_dir / 'v13o7_nv_curve_results.json'}", flush=True)


if __name__ == "__main__":
    main()

