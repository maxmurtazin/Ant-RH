#!/usr/bin/env python3
"""
V13O.6 — Number Variance Stabilization (diagnostic layer).

Post-processing layer over V13O.4/V13O.5 outputs (no operator recomputation).

Computational evidence only; not a proof of the Riemann Hypothesis.
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


def read_csv_robust(path: Path, *, name: str, tag: str = "v13o6") -> Optional["pd.DataFrame"]:
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
    arr = _finite_arr(xs)
    if arr.size == 0:
        return float("nan")
    return float(np.median(arr))


def safe_mean(xs: Sequence[float]) -> float:
    arr = _finite_arr(xs)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def mad_with_fallback(x: np.ndarray, *, eps: float) -> Tuple[float, float, Dict[str, float]]:
    """
    Returns (median, scale, diagnostics).

    Scale preference:
    - MAD (median absolute deviation)
    - fallback = max(IQR/1.349, std, abs(median)*1e-6, eps)
    """
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float(eps), {"mad": float("nan"), "iqr": float("nan"), "std": float("nan"), "fallback": float(eps)}

    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    q25 = float(np.percentile(x, 25))
    q75 = float(np.percentile(x, 75))
    iqr = float(q75 - q25)
    scale_iqr = float(iqr / 1.349) if math.isfinite(iqr) else float("nan")
    std = float(np.std(x))
    fallback = float(
        max(
            float(scale_iqr) if math.isfinite(scale_iqr) else 0.0,
            float(std) if math.isfinite(std) else 0.0,
            abs(float(med)) * 1e-6,
            float(eps),
        )
    )
    scale = float(mad) if (math.isfinite(mad) and mad > eps) else float(fallback)
    return med, max(scale, float(eps)), {"mad": mad, "iqr": iqr, "std": std, "fallback": fallback}


def robust_z_lower_is_better(x_real: float, controls: Sequence[float], *, eps: float) -> Tuple[float, float, float]:
    """
    Returns (z, median_control, scale_control).
    z = (x_real - median_control) / (scale_control + eps).
    """
    if not math.isfinite(float(x_real)):
        return float("nan"), float("nan"), float("nan")
    c = _finite_arr(controls)
    if c.size == 0:
        return float("nan"), float("nan"), float("nan")
    med, scale, _ = mad_with_fallback(c, eps=float(eps))
    return float((float(x_real) - float(med)) / max(float(scale), float(eps))), float(med), float(scale)


def winsorize(x: np.ndarray, *, q_low: float, q_high: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return x
    lo = float(np.quantile(x, q_low))
    hi = float(np.quantile(x, q_high))
    return np.clip(x, lo, hi)


def empirical_percentile_real_lower_is_better(x_real: float, controls: Sequence[float]) -> float:
    c = _finite_arr(controls)
    if c.size == 0 or not math.isfinite(float(x_real)):
        return float("nan")
    return float(np.mean(c <= float(x_real)))


def classify_nv(
    *,
    z_nv: float,
    z_log_nv: float,
    z_winsor_nv: float,
    percentile_real: float,
    n_controls: int,
) -> str:
    if n_controls < 3:
        return "NV_INDETERMINATE"

    strict = (
        math.isfinite(z_log_nv)
        and math.isfinite(z_winsor_nv)
        and math.isfinite(percentile_real)
        and (z_log_nv <= -0.25)
        and (z_winsor_nv <= -0.25)
        and (percentile_real <= 0.25)
    )
    if strict:
        return "NV_PASS_STRONG"

    weak_conditions = [
        math.isfinite(z_log_nv) and (z_log_nv <= 0.0),
        math.isfinite(z_winsor_nv) and (z_winsor_nv <= 0.0),
        math.isfinite(percentile_real) and (percentile_real <= 0.5),
    ]
    weak_pass = sum(1 for c in weak_conditions if c) >= 2
    if weak_pass:
        return "NV_PASS_WEAK"

    # scale-dominated heuristic: raw z looks huge positive but stabilized variants are neutral-ish
    if math.isfinite(z_nv) and z_nv >= 2.0:
        neutralish = (
            (not math.isfinite(z_log_nv) or abs(float(z_log_nv)) <= 0.25)
            and (not math.isfinite(z_winsor_nv) or abs(float(z_winsor_nv)) <= 0.25)
            and (not math.isfinite(percentile_real) or (0.25 <= float(percentile_real) <= 0.75))
        )
        if neutralish:
            return "NV_FAIL_SCALE_DOMINATED"

    # robust fail if stabilized views all fail
    robust_fail = (
        (not math.isfinite(z_log_nv) or z_log_nv > 0.0)
        and (not math.isfinite(z_winsor_nv) or z_winsor_nv > 0.0)
        and (not math.isfinite(percentile_real) or percentile_real > 0.5)
    )
    if robust_fail:
        return "NV_FAIL_ROBUST"

    return "NV_INDETERMINATE"


def maybe_curve_mode(
    df_nv: Optional["pd.DataFrame"],
    *,
    eps: float,
    progress_every: int,
) -> Dict[str, Any]:
    """
    Optional per-L curve diagnostics if per-L NV curve data exists.
    Expected style (best-effort, robust to missing columns):
    - columns: dim, V_mode, word_group, target_group, L, Sigma2 (or Sigma2_real / Sigma2_control)

    If unavailable, return curve_mode_available=False.
    """
    if df_nv is None:
        return {"curve_mode_available": False, "reason": "missing_df"}

    cols = set(map(str, df_nv.columns))
    if "L" not in cols:
        return {"curve_mode_available": False, "reason": "no_L_column"}

    sigma_col = pick_col(df_nv, ["Sigma2", "sigma2", "sigma2_L", "Sigma2_L"])
    if sigma_col is None:
        # allow alternate naming: Sigma2_real / Sigma2_control, but this requires row pairing
        return {"curve_mode_available": False, "reason": "no_sigma2_column"}

    required = {"dim", "V_mode", "word_group", "target_group", "L", sigma_col}
    if not required.issubset(cols):
        return {"curve_mode_available": False, "reason": "missing_required_cols"}

    dfx = df_nv.copy()
    dfx["dim"] = pd.to_numeric(dfx["dim"], errors="coerce").astype("Int64")
    for k in ("V_mode", "word_group", "target_group"):
        if k in dfx.columns:
            dfx[k] = dfx[k].astype(str).str.strip()
    dfx["L"] = pd.to_numeric(dfx["L"], errors="coerce")
    dfx["_sigma2"] = finite_series(dfx[sigma_col])
    dfx = dfx.dropna(subset=["dim", "V_mode", "word_group", "target_group", "L", "_sigma2"])

    if dfx.empty:
        return {"curve_mode_available": False, "reason": "no_finite_rows"}

    groups = sorted({(int(r.dim), str(r.V_mode), str(r.word_group)) for r in dfx[["dim", "V_mode", "word_group"]].itertuples(index=False)})
    prog = max(1, int(progress_every))
    t0 = time.perf_counter()

    rows: List[Dict[str, Any]] = []
    for i, (dim, vm, wg) in enumerate(groups, start=1):
        if i == 1 or i == len(groups) or i % prog == 0:
            elapsed = time.perf_counter() - t0
            avg = elapsed / max(i, 1)
            eta = avg * max(len(groups) - i, 0)
            print(f"[v13o6] curve-mode {i}/{len(groups)} elapsed={elapsed:.1f}s eta={format_seconds(eta)} current=({dim},{vm},{wg})", flush=True)

        sub = dfx[(dfx["dim"] == dim) & (dfx["V_mode"] == vm) & (dfx["word_group"] == wg)]
        if sub.empty:
            continue

        # for each L: compute robust z of real_zeta sigma2 vs controls sigma2 distribution
        for L, subL in sub.groupby("L"):
            real_vals = subL.loc[subL["target_group"] == "real_zeta", "_sigma2"].dropna().astype(float).tolist()
            ctrl_vals = subL.loc[subL["target_group"] != "real_zeta", "_sigma2"].dropna().astype(float).tolist()
            if not real_vals or not ctrl_vals:
                continue
            x_real = safe_median(real_vals)
            z, _, _ = robust_z_lower_is_better(x_real, ctrl_vals, eps=eps)
            rows.append(
                {
                    "dim": dim,
                    "V_mode": vm,
                    "word_group": wg,
                    "L": float(L),
                    "z_nv_L": float(z),
                    "sigma2_real_median": float(x_real),
                    "n_controls_L": int(len(_finite_arr(ctrl_vals))),
                }
            )

    if not rows:
        return {"curve_mode_available": False, "reason": "no_pairs_real_vs_controls"}

    dfz = pd.DataFrame(rows)
    agg_rows: List[Dict[str, Any]] = []
    for (dim, vm, wg), g in dfz.groupby(["dim", "V_mode", "word_group"]):
        zz = _finite_arr(g["z_nv_L"].astype(float).tolist())
        if zz.size == 0:
            continue
        z_sorted = np.sort(zz)
        trim = int(0.1 * z_sorted.size)
        trimmed = z_sorted[trim : (z_sorted.size - trim)] if z_sorted.size > 2 * trim else z_sorted
        median_z = float(np.median(z_sorted))
        trimmed_mean = float(np.mean(trimmed)) if trimmed.size else float("nan")
        max_bad = float(np.max(z_sorted))
        frac_pass = float(np.mean(z_sorted <= 0.0))
        curve_pass = bool((median_z <= 0.0) and (frac_pass >= 0.6))
        agg_rows.append(
            {
                "dim": int(dim),
                "V_mode": str(vm),
                "word_group": str(wg),
                "median_z_over_L": median_z,
                "trimmed_mean_z_over_L": trimmed_mean,
                "max_bad_L_z": max_bad,
                "fraction_L_pass": frac_pass,
                "curve_nv_pass": curve_pass,
                "n_L": int(z_sorted.size),
            }
        )

    return {
        "curve_mode_available": True,
        "curve_by_group": agg_rows,
        "curve_per_L": rows,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="V13O.6 number variance stabilization (computational evidence only).")

    ap.add_argument("--v13o4_number_variance", type=str, default="runs/v13o4_zeta_specific_objective/v13o4_number_variance_summary.csv")
    ap.add_argument("--v13o4_zeta_scores", type=str, default="runs/v13o4_zeta_specific_objective/v13o4_zeta_specific_scores.csv")
    ap.add_argument("--v13o5_robust_scores", type=str, default="runs/v13o5_robust_unfolded_normalization_smoke/v13o5_robust_scores.csv")
    ap.add_argument("--v13o5_gate_summary", type=str, default="runs/v13o5_robust_unfolded_normalization_smoke/v13o5_gate_summary.csv")

    ap.add_argument("--out_dir", type=str, default="runs/v13o6_number_variance_stabilization")
    ap.add_argument("--primary_word_group", type=str, default="primary_word_seed6")
    ap.add_argument("--primary_v_mode", type=str, default="full_V")
    ap.add_argument("--progress_every", type=int, default=1)

    ap.add_argument("--mad_eps", type=float, default=1e-9)
    ap.add_argument("--winsor_q_low", type=float, default=0.05)
    ap.add_argument("--winsor_q_high", type=float, default=0.95)

    args = ap.parse_args()

    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    prog = int(args.progress_every)
    mad_eps = float(args.mad_eps)

    # Load CSVs (missing files should not crash)
    df_nv = read_csv_robust(_resolve(args.v13o4_number_variance), name="v13o4_number_variance", tag="v13o6")
    df_zeta_scores = read_csv_robust(_resolve(args.v13o4_zeta_scores), name="v13o4_zeta_scores", tag="v13o6")
    df_v13o5_scores = read_csv_robust(_resolve(args.v13o5_robust_scores), name="v13o5_robust_scores", tag="v13o6")
    df_v13o5_gate = read_csv_robust(_resolve(args.v13o5_gate_summary), name="v13o5_gate_summary", tag="v13o6")

    # Normalize key columns for NV df
    if df_nv is not None:
        if "dim" in df_nv.columns:
            df_nv["dim"] = pd.to_numeric(df_nv["dim"], errors="coerce").astype("Int64")
        for k in ("V_mode", "word_group", "target_group"):
            if k in df_nv.columns:
                df_nv[k] = df_nv[k].astype(str).str.strip()
        if "control_id" in df_nv.columns:
            df_nv["control_id"] = df_nv["control_id"].astype(str).str.strip()

    col_nv_test = pick_col(df_nv, ["number_variance_error_test", "number_variance_test", "number_variance_error"])
    if df_nv is None or col_nv_test is None or "target_group" not in df_nv.columns:
        print("[v13o6] WARNING: NV input missing required columns; continuing with empty outputs.", flush=True)
        df_nv_use = None
    else:
        df_nv_use = df_nv.copy()
        df_nv_use["_nv"] = finite_series(df_nv_use[col_nv_test])

    # Determine groups: union across available dfs (prefer nv)
    groups_set: set[Tuple[int, str, str]] = set()
    for dfx in (df_nv_use, df_zeta_scores, df_v13o5_scores):
        if dfx is None:
            continue
        if not all(k in dfx.columns for k in ("dim", "V_mode", "word_group")):
            continue
        tmp = dfx[["dim", "V_mode", "word_group"]].dropna()
        if "dim" in tmp.columns:
            tmp["dim"] = pd.to_numeric(tmp["dim"], errors="coerce")
        for r in tmp.itertuples(index=False):
            if r[0] is None or (isinstance(r[0], float) and not math.isfinite(float(r[0]))):
                continue
            try:
                groups_set.add((int(r[0]), str(r[1]).strip(), str(r[2]).strip()))
            except Exception:
                continue
    groups = sorted(groups_set, key=lambda x: (int(x[0]), str(x[1]), str(x[2])))

    print(f"[v13o6] groups to analyze: {len(groups)}", flush=True)

    def log_progress(i: int, total: int, group: Tuple[int, str, str]) -> None:
        if prog <= 0:
            return
        if i == 1 or i == total or i % max(1, prog) == 0:
            elapsed = time.perf_counter() - t0
            avg = elapsed / max(i, 1)
            eta = avg * max(total - i, 0)
            d, vm, wg = group
            print(f"[V13O.6] {i}/{total} elapsed={elapsed:.1f}s eta={format_seconds(eta)} current=({d},{vm},{wg})", flush=True)

    rows_scores: List[Dict[str, Any]] = []
    rows_components: List[Dict[str, Any]] = []

    for i, (dim, vm, wg) in enumerate(groups, start=1):
        log_progress(i, len(groups), (dim, vm, wg))

        # slice
        if df_nv_use is None:
            sub = None
        else:
            sub = df_nv_use[(df_nv_use["dim"] == dim) & (df_nv_use["V_mode"] == vm) & (df_nv_use["word_group"] == wg)]

        nv_real = float("nan")
        nv_controls: List[float] = []
        nv_controls_all_nonreal: List[float] = []
        control_groups_present: List[str] = []

        if sub is not None and not sub.empty:
            real_vals = sub.loc[sub["target_group"] == "real_zeta", "_nv"].dropna().astype(float).tolist()
            nv_real = safe_median(real_vals)

            # controls: prefer the explicit list (subset if present)
            for tg in CONTROL_TARGET_GROUPS:
                vals = sub.loc[sub["target_group"] == tg, "_nv"].dropna().astype(float).tolist()
                if vals:
                    control_groups_present.append(tg)
                    nv_controls.extend([float(v) for v in vals if math.isfinite(float(v))])

            # fallback: any non-real rows if the explicit list is empty
            nv_controls_all_nonreal = sub.loc[sub["target_group"] != "real_zeta", "_nv"].dropna().astype(float).tolist()
            if not nv_controls and nv_controls_all_nonreal:
                nv_controls = [float(v) for v in nv_controls_all_nonreal if math.isfinite(float(v))]
                control_groups_present = sorted({str(t) for t in sub.loc[sub["target_group"] != "real_zeta", "target_group"].astype(str).tolist()})

        n_ctrl = int(len(_finite_arr(nv_controls)))

        z_nv, median_ctrl, scale_ctrl = robust_z_lower_is_better(nv_real, nv_controls, eps=mad_eps)

        # log1p stabilization
        log_nv_real = float(np.log1p(nv_real)) if math.isfinite(nv_real) and nv_real >= 0.0 else float("nan")
        log_nv_ctrl = [float(np.log1p(v)) for v in nv_controls if math.isfinite(float(v)) and float(v) >= 0.0]
        z_log_nv, median_log_ctrl, scale_log_ctrl = robust_z_lower_is_better(log_nv_real, log_nv_ctrl, eps=mad_eps)

        # winsorized controls (raw then log1p? spec says winsorize values at quantiles, then median/mad, then z)
        w_ctrl = winsorize(_finite_arr(nv_controls), q_low=float(args.winsor_q_low), q_high=float(args.winsor_q_high))
        z_winsor_nv, median_w_ctrl, scale_w_ctrl = robust_z_lower_is_better(nv_real, w_ctrl.tolist(), eps=mad_eps)

        percentile_real = empirical_percentile_real_lower_is_better(nv_real, nv_controls)
        rank_pass = bool(math.isfinite(percentile_real) and percentile_real <= 0.25)

        # raw diagnostic: diff where positive means real better (lower error)
        diff_control_minus_real = float(median_ctrl - nv_real) if (math.isfinite(median_ctrl) and math.isfinite(nv_real)) else float("nan")

        classification = classify_nv(
            z_nv=float(z_nv),
            z_log_nv=float(z_log_nv),
            z_winsor_nv=float(z_winsor_nv),
            percentile_real=float(percentile_real),
            n_controls=n_ctrl,
        )

        rows_scores.append(
            {
                "dim": int(dim),
                "V_mode": str(vm),
                "word_group": str(wg),
                "nv_real": float(nv_real),
                "median_control_nv": float(median_ctrl),
                "mad_or_fallback_control_nv": float(scale_ctrl),
                "z_nv_raw": float(z_nv),
                "log_nv_real": float(log_nv_real),
                "median_log_control_nv": float(median_log_ctrl),
                "mad_or_fallback_log_control_nv": float(scale_log_ctrl),
                "z_log_nv": float(z_log_nv),
                "median_winsor_control_nv": float(median_w_ctrl),
                "mad_or_fallback_winsor_control_nv": float(scale_w_ctrl),
                "z_winsor_nv": float(z_winsor_nv),
                "percentile_real_among_controls": float(percentile_real),
                "rank_pass_25pct": bool(rank_pass),
                "diff_control_minus_real": float(diff_control_minus_real),
                "n_controls_used": int(n_ctrl),
                "control_groups_used": "|".join([str(x) for x in control_groups_present]) if control_groups_present else "",
                "classification": str(classification),
            }
        )

        rows_components.append(
            {
                "dim": int(dim),
                "V_mode": str(vm),
                "word_group": str(wg),
                "nv_real": float(nv_real),
                "nv_controls_count": int(n_ctrl),
                "control_groups_used": "|".join([str(x) for x in control_groups_present]) if control_groups_present else "",
                "z_nv_raw": float(z_nv),
                "z_log_nv": float(z_log_nv),
                "z_winsor_nv": float(z_winsor_nv),
                "percentile_real_among_controls": float(percentile_real),
                "rank_pass_25pct": bool(rank_pass),
            }
        )

    df_scores_out = pd.DataFrame(rows_scores) if rows_scores else pd.DataFrame(
        columns=[
            "dim",
            "V_mode",
            "word_group",
            "nv_real",
            "median_control_nv",
            "mad_or_fallback_control_nv",
            "z_nv_raw",
            "log_nv_real",
            "median_log_control_nv",
            "mad_or_fallback_log_control_nv",
            "z_log_nv",
            "median_winsor_control_nv",
            "mad_or_fallback_winsor_control_nv",
            "z_winsor_nv",
            "percentile_real_among_controls",
            "rank_pass_25pct",
            "diff_control_minus_real",
            "n_controls_used",
            "control_groups_used",
            "classification",
        ]
    )

    df_comp_out = pd.DataFrame(rows_components) if rows_components else pd.DataFrame(
        columns=[
            "dim",
            "V_mode",
            "word_group",
            "nv_real",
            "nv_controls_count",
            "control_groups_used",
            "z_nv_raw",
            "z_log_nv",
            "z_winsor_nv",
            "percentile_real_among_controls",
            "rank_pass_25pct",
        ]
    )

    (out_dir / "v13o6_nv_stabilized_scores.csv").write_text(df_scores_out.to_csv(index=False), encoding="utf-8")
    (out_dir / "v13o6_nv_component_summary.csv").write_text(df_comp_out.to_csv(index=False), encoding="utf-8")

    # Primary full_V gate summary: per dim, only primary_word_group + primary_v_mode
    gate_rows: List[Dict[str, Any]] = []
    primary_wg = str(args.primary_word_group)
    primary_vm = str(args.primary_v_mode)

    if not df_scores_out.empty:
        prim = df_scores_out[(df_scores_out["V_mode"] == primary_vm) & (df_scores_out["word_group"] == primary_wg)].copy()
        if not prim.empty and "dim" in prim.columns:
            prim["dim"] = pd.to_numeric(prim["dim"], errors="coerce").astype("Int64")
            prim = prim.dropna(subset=["dim"])
            for r in prim.sort_values(["dim"]).itertuples(index=False):
                # tuple order matches columns; access by name via getattr
                z_nv_raw = float(getattr(r, "z_nv_raw"))
                z_log = float(getattr(r, "z_log_nv"))
                z_win = float(getattr(r, "z_winsor_nv"))
                pct = float(getattr(r, "percentile_real_among_controls"))
                n_ctrl = int(getattr(r, "n_controls_used"))
                cls = str(getattr(r, "classification"))

                s1 = bool(math.isfinite(z_log) and math.isfinite(z_win) and math.isfinite(pct) and n_ctrl >= 3)
                s2 = bool(s1 and z_log <= -0.25)
                s3 = bool(s1 and z_win <= -0.25)
                s4 = bool(s1 and pct <= 0.25)

                weak2 = sum(
                    1
                    for c in (
                        (s1 and z_log <= 0.0),
                        (s1 and z_win <= 0.0),
                        (s1 and pct <= 0.5),
                    )
                    if c
                )
                s5 = bool(s1 and weak2 >= 2)

                strict_pass = bool(s2 and s3 and s4)
                relaxed_pass = bool(s5)

                gate_rows.append(
                    {
                        "dim": int(getattr(r, "dim")),
                        "V_mode": primary_vm,
                        "word_group": primary_wg,
                        "n_controls_used": n_ctrl,
                        "S1_finite_inputs": s1,
                        "S2_log_nv_robust": s2,
                        "S3_winsor_nv_robust": s3,
                        "S4_rank_nv_pass": s4,
                        "S5_nv_weak_pass": s5,
                        "z_nv_raw": z_nv_raw,
                        "z_log_nv": z_log,
                        "z_winsor_nv": z_win,
                        "percentile_real_among_controls": pct,
                        "strict_nv_stabilized_pass": strict_pass,
                        "relaxed_nv_stabilized_pass": relaxed_pass,
                        "classification": cls,
                    }
                )

    df_gate_out = pd.DataFrame(gate_rows) if gate_rows else pd.DataFrame(
        columns=[
            "dim",
            "V_mode",
            "word_group",
            "n_controls_used",
            "S1_finite_inputs",
            "S2_log_nv_robust",
            "S3_winsor_nv_robust",
            "S4_rank_nv_pass",
            "S5_nv_weak_pass",
            "z_nv_raw",
            "z_log_nv",
            "z_winsor_nv",
            "percentile_real_among_controls",
            "strict_nv_stabilized_pass",
            "relaxed_nv_stabilized_pass",
            "classification",
        ]
    )
    (out_dir / "v13o6_nv_gate_summary.csv").write_text(df_gate_out.to_csv(index=False), encoding="utf-8")

    # Best-by-dim: rank candidates by a stabilized NV score (lower is better)
    # Heuristic score: avg(z_log, z_winsor) + (percentile - 0.5). Missing terms ignored.
    best_rows: List[Dict[str, Any]] = []
    if not df_scores_out.empty and "dim" in df_scores_out.columns:
        dfx = df_scores_out.copy()
        dfx["dim"] = pd.to_numeric(dfx["dim"], errors="coerce").astype("Int64")
        dfx = dfx.dropna(subset=["dim"])

        def stabilized_score(row: "pd.Series") -> float:
            terms: List[float] = []
            zlog = float(row.get("z_log_nv", float("nan")))
            zwang = float(row.get("z_winsor_nv", float("nan")))
            pct = float(row.get("percentile_real_among_controls", float("nan")))
            if math.isfinite(zlog):
                terms.append(zlog)
            if math.isfinite(zwang):
                terms.append(zwang)
            base = float(np.mean(np.asarray(terms, dtype=np.float64))) if terms else float("nan")
            if math.isfinite(pct):
                base = float(base + (pct - 0.5)) if math.isfinite(base) else float(pct - 0.5)
            return float(base)

        dfx["_stabilized_score"] = dfx.apply(stabilized_score, axis=1)

        for dim, g in dfx.groupby("dim"):
            gg = g.copy()
            gg = gg.sort_values(["_stabilized_score", "z_log_nv", "z_winsor_nv", "percentile_real_among_controls"], ascending=True, na_position="last")
            best = gg.iloc[0]
            # primary rank among all candidates for this dim
            primary_mask = (gg["V_mode"] == primary_vm) & (gg["word_group"] == primary_wg)
            if primary_mask.any():
                # rank by stabilized score
                gg2 = gg.reset_index(drop=True)
                gg2["_rank"] = np.arange(1, len(gg2) + 1, dtype=int)
                prim_rank = int(gg2.loc[primary_mask.values, "_rank"].iloc[0])
            else:
                prim_rank = -1

            best_rows.append(
                {
                    "dim": int(dim),
                    "best_V_mode": str(best["V_mode"]),
                    "best_word_group": str(best["word_group"]),
                    "best_stabilized_score": float(best["_stabilized_score"]),
                    "best_z_log_nv": float(best["z_log_nv"]),
                    "best_z_winsor_nv": float(best["z_winsor_nv"]),
                    "best_percentile_real_among_controls": float(best["percentile_real_among_controls"]),
                    "best_classification": str(best["classification"]),
                    "primary_present": bool(primary_mask.any()),
                    "primary_rank_by_stabilized_score": int(prim_rank),
                }
            )

    df_best = pd.DataFrame(best_rows) if best_rows else pd.DataFrame(
        columns=[
            "dim",
            "best_V_mode",
            "best_word_group",
            "best_stabilized_score",
            "best_z_log_nv",
            "best_z_winsor_nv",
            "best_percentile_real_among_controls",
            "best_classification",
            "primary_present",
            "primary_rank_by_stabilized_score",
        ]
    )
    (out_dir / "v13o6_nv_best_by_dim.csv").write_text(df_best.to_csv(index=False), encoding="utf-8")

    # Optional curve-mode diagnostics
    curve = maybe_curve_mode(df_nv, eps=mad_eps, progress_every=int(args.progress_every))
    curve_mode_available = bool(curve.get("curve_mode_available"))

    # Interpretation synthesis
    strict_all_dims = bool(gate_rows) and all(bool(r.get("strict_nv_stabilized_pass")) for r in gate_rows)
    relaxed_any = bool(gate_rows) and any(bool(r.get("relaxed_nv_stabilized_pass")) for r in gate_rows)

    scale_artifact_suspected = False
    if gate_rows:
        # if raw z looks bad but stabilized looks better (at least weak pass), flag
        for r in gate_rows:
            zraw = float(r.get("z_nv_raw", float("nan")))
            zlog = float(r.get("z_log_nv", float("nan")))
            zwang = float(r.get("z_winsor_nv", float("nan")))
            weak = bool(r.get("relaxed_nv_stabilized_pass"))
            if math.isfinite(zraw) and zraw >= 2.0 and weak and (math.isfinite(zlog) or math.isfinite(zwang)):
                scale_artifact_suspected = True

    if strict_all_dims:
        interpretation = "NV_STABILIZED_SIGNAL"
    elif relaxed_any and not strict_all_dims:
        interpretation = "PARTIAL_NV_SIGNAL"
    elif scale_artifact_suspected:
        interpretation = "SCALE_ARTIFACT_SUSPECTED"
    else:
        interpretation = "NO_NV_SIGNAL"

    payload: Dict[str, Any] = {
        "warning": "Computational evidence only; not a proof of the Riemann Hypothesis.",
        "status": "V13O.6 number variance stabilization",
        "interpretation": interpretation,
        "primary_word_group": primary_wg,
        "primary_v_mode": primary_vm,
        "strict_nv_stabilized_all_dims": strict_all_dims,
        "relaxed_nv_stabilized_any_dim": relaxed_any,
        "scale_artifact_suspected": scale_artifact_suspected,
        "control_target_groups_preferred": list(CONTROL_TARGET_GROUPS),
        "curve_mode_available": curve_mode_available,
        "curve_mode_reason": curve.get("reason") if not curve_mode_available else None,
        "inputs": {
            "v13o4_number_variance": str(_resolve(args.v13o4_number_variance).resolve()),
            "v13o4_zeta_scores": str(_resolve(args.v13o4_zeta_scores).resolve()),
            "v13o5_robust_scores": str(_resolve(args.v13o5_robust_scores).resolve()),
            "v13o5_gate_summary": str(_resolve(args.v13o5_gate_summary).resolve()),
        },
        "outputs": {
            "nv_stabilized_scores_csv": str((out_dir / "v13o6_nv_stabilized_scores.csv").resolve()),
            "nv_gate_summary_csv": str((out_dir / "v13o6_nv_gate_summary.csv").resolve()),
            "nv_component_summary_csv": str((out_dir / "v13o6_nv_component_summary.csv").resolve()),
            "nv_best_by_dim_csv": str((out_dir / "v13o6_nv_best_by_dim.csv").resolve()),
        },
        "gate_summary": json_sanitize(gate_rows),
        "curve_mode": json_sanitize(curve if curve_mode_available else {}),
        "runtime_s": float(time.perf_counter() - t0),
        "python_pid": os.getpid(),
    }
    with open(out_dir / "v13o6_results.json", "w", encoding="utf-8") as f:
        json.dump(json_sanitize(payload), f, indent=2, allow_nan=False)

    # Report: markdown
    md: List[str] = []
    md.append("# V13O.6 Number Variance Stabilization\n\n")
    md.append("> **Computational evidence only; not a proof of RH.**\n\n")
    md.append("## 1. Why V13O.6 exists\n\n")
    md.append(
        "V13O.5 indicated partial signal in **pair correlation** and **staircase residuals**, but overall gating failed because the "
        "**number variance** component behaved as a dominant, unstable bottleneck (heavy tails / scale sensitivity).\n\n"
    )
    md.append("## 2. Method\n\n")
    md.append(
        "For each `(dim, V_mode, word_group)`, we compare the real zeta number-variance error against a set of control target groups.\n\n"
    )
    md.append("- **Raw diagnostic**: robust z-score on raw NV error (lower is better).\n")
    md.append("- **Log-stabilized**: robust z-score on `log1p(NV)` (lower is better).\n")
    md.append("- **Winsorized controls**: winsorize control values at `[0.05, 0.95]`, then robust z-score (lower is better).\n")
    md.append("- **Rank-based**: empirical percentile of real among controls; lower indicates real is unusually small.\n\n")
    md.append("## 3. Primary full_V gate\n\n")
    md.append(f"Primary selection: `word_group={primary_wg}`, `V_mode={primary_vm}`.\n\n")
    md.append("See `v13o6_nv_gate_summary.csv`.\n\n")
    md.append("## 4. Best by dim\n\n")
    md.append("See `v13o6_nv_best_by_dim.csv` for lowest stabilized NV candidates per dimension.\n\n")
    md.append("## 5. Interpretation\n\n")
    md.append(f"**{interpretation}**\n\n")
    md.append("- `NV_STABILIZED_SIGNAL`: strict stabilized pass in all dims.\n")
    md.append("- `PARTIAL_NV_SIGNAL`: relaxed pass in at least one dim, but not all strict.\n")
    md.append("- `NO_NV_SIGNAL`: no relaxed pass.\n")
    md.append("- `SCALE_ARTIFACT_SUSPECTED`: raw fails badly but stabilized diagnostics improve strongly.\n\n")
    md.append("## 6. Recommendation\n\n")
    if not curve_mode_available:
        md.append(
            "Per-L curve mode was **not available** from the provided inputs (`curve_mode_available=False`).\n\n"
            "Recommend **V13O.7**: generate explicit number variance curves \\(\\Sigma^2(L)\\) over a range of L and compare **shape**, "
            "not only a single scalar magnitude.\n\n"
        )
    else:
        md.append("Per-L curve mode was available; see `v13o6_results.json` under `curve_mode`.\n\n")
    md.append("## Outputs\n\n")
    md.append("- `v13o6_nv_stabilized_scores.csv`\n")
    md.append("- `v13o6_nv_gate_summary.csv`\n")
    md.append("- `v13o6_nv_component_summary.csv`\n")
    md.append("- `v13o6_nv_best_by_dim.csv`\n")
    md.append("- `v13o6_results.json`\n")
    md.append("- `v13o6_report.md`\n")
    md.append("- `v13o6_report.tex`\n")
    md.append("- `v13o6_report.pdf` (if `pdflatex` exists)\n\n")
    (out_dir / "v13o6_report.md").write_text("".join(md), encoding="utf-8")

    # Report: latex (minimal; tables stay in CSV)
    tex = (
        "\\documentclass[11pt]{article}\n"
        "\\usepackage[margin=1in]{geometry}\n"
        "\\begin{document}\n"
        "\\title{V13O.6 Number Variance Stabilization}\n"
        "\\maketitle\n"
        "\\section*{Warning}\n"
        "Computational evidence only; not a proof of RH.\n\n"
        "\\section*{Interpretation}\n"
        + latex_escape(str(interpretation))
        + "\n\n"
        "\\section*{Primary gate}\n"
        + latex_escape(f"See v13o6_nv_gate_summary.csv (primary: word_group={primary_wg}, V_mode={primary_vm}).")
        + "\n\n"
        "\\section*{Best by dim}\n"
        + latex_escape("See v13o6_nv_best_by_dim.csv.")
        + "\n\n"
        "\\section*{Recommendation}\n"
        + latex_escape(
            "V13O.7 recommended to produce explicit Sigma^2(L) curves."
            if not curve_mode_available
            else "Curve mode available; see v13o6_results.json."
        )
        + "\n\n"
        "\\end{document}\n"
    )
    (out_dir / "v13o6_report.tex").write_text(tex, encoding="utf-8")

    if _find_pdflatex() is None:
        print("[v13o6] WARNING: pdflatex not found; skipping PDF.", flush=True)
    elif try_pdflatex(out_dir / "v13o6_report.tex", out_dir, "v13o6_report.pdf"):
        print(f"Wrote {out_dir / 'v13o6_report.pdf'}", flush=True)
    else:
        print("[v13o6] WARNING: pdflatex failed or did not produce v13o6_report.pdf.", flush=True)

    print(f"[v13o6] Wrote {out_dir / 'v13o6_results.json'}", flush=True)


if __name__ == "__main__":
    main()

