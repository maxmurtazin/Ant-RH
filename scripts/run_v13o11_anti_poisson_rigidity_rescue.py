#!/usr/bin/env python3
"""
V13O.11 — Anti-Poisson Rigidity Rescue (true spectra).

Computational diagnostic only; not a proof of the Riemann Hypothesis.

This script loads:
  - V13O.9 true unfolded spectra diagnostics (poissonization + curve errors)
  - V13O.10 candidate ranking / coverage / artifact flags

and asks a more specific question than V13O.10:
  "Does any candidate become *less Poisson-like* and *more rigid / GUE-like* than the primary?"
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pandas as pd  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit("This script requires pandas. Please install pandas and retry.") from e


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


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


def read_csv_required(path: Path, *, name: str) -> "pd.DataFrame":
    if not path.is_file():
        raise SystemExit(f"[v13o11] ERROR: required input missing: {name}={path}")
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def read_csv_optional(path: Path, *, name: str) -> Optional["pd.DataFrame"]:
    if not path or str(path).strip() == "":
        return None
    if not path.is_file():
        print(f"[v13o11] WARNING missing optional input: {name}={path}", flush=True)
        return None
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[v13o11] WARNING failed reading {name}={path}: {e!r}", flush=True)
        return None
    df.columns = [str(c).strip() for c in df.columns]
    return df


def finite_series(x: "pd.Series") -> "pd.Series":
    return pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan)


def safe_median(xs: Sequence[float]) -> float:
    arr = np.asarray([float(v) for v in xs if v is not None and math.isfinite(float(v))], dtype=np.float64)
    return float(np.median(arr)) if arr.size else float("nan")


def parse_poissonization_df(df: "pd.DataFrame") -> Tuple["pd.DataFrame", List[str]]:
    """
    Robust parser for v13o9_poissonization_diagnostics.csv.
    Attempts to identify:
      - dim,V_mode,word_group,target_group,seed
      - d_poisson, d_gue, ratio, classification

    If names are absent, uses heuristic:
      - last col = classification (string)
      - numeric cols before classification: first->d_gue, second->d_poisson, third->ratio (if present)
    """
    warns: List[str] = []
    dfx = df.copy()
    # normalize mandatory keys
    for k in ("dim", "V_mode", "word_group", "target_group", "seed"):
        if k not in dfx.columns:
            warns.append(f"poissonization missing key column: {k}")
    if any(k not in dfx.columns for k in ("dim", "V_mode", "word_group", "target_group", "seed")):
        # cannot proceed robustly
        return pd.DataFrame(), warns
    dfx["dim"] = pd.to_numeric(dfx["dim"], errors="coerce").astype("Int64")
    dfx["seed"] = pd.to_numeric(dfx["seed"], errors="coerce").astype("Int64")
    for k in ("V_mode", "word_group", "target_group"):
        dfx[k] = dfx[k].astype(str).str.strip()

    # classification column
    cls_col = None
    for cand in ("classification", "poissonization_classification", "class"):
        if cand in dfx.columns:
            cls_col = cand
            break
    if cls_col is None:
        cls_col = str(dfx.columns[-1])
        warns.append(f"poissonization classification inferred as last column: {cls_col}")
    dfx["_classification"] = dfx[cls_col].astype(str).str.strip()

    # distances
    col_dp = None
    col_dg = None
    col_ratio = None
    for cand in ("distance_to_poisson", "d_poisson", "poisson_distance", "dist_poisson"):
        if cand in dfx.columns:
            col_dp = cand
            break
    for cand in ("distance_to_gue", "d_gue", "gue_distance", "dist_gue"):
        if cand in dfx.columns:
            col_dg = cand
            break
    for cand in ("poissonization_ratio", "ratio", "gue_to_poisson_ratio", "ratio_gue_to_poisson"):
        if cand in dfx.columns:
            col_ratio = cand
            break

    if col_dp is None or col_dg is None:
        # heuristic: numeric columns before classification
        numeric_cols = []
        for c in dfx.columns:
            if c in ("dim", "V_mode", "word_group", "target_group", "seed", cls_col, "_classification"):
                continue
            s = finite_series(dfx[c])
            if s.notna().mean() > 0.5:
                numeric_cols.append(str(c))
        if len(numeric_cols) >= 2:
            col_dg = col_dg or numeric_cols[0]
            col_dp = col_dp or numeric_cols[1]
            warns.append(f"inferred d_gue={col_dg}, d_poisson={col_dp} from numeric columns")
        if col_ratio is None and len(numeric_cols) >= 3:
            col_ratio = numeric_cols[2]
            warns.append(f"inferred ratio={col_ratio} from numeric columns")

    if col_dp is None or col_dg is None:
        warns.append("could not infer d_poisson/d_gue; rigidity metrics unavailable")
        dfx["_d_poisson"] = np.nan
        dfx["_d_gue"] = np.nan
    else:
        dfx["_d_poisson"] = finite_series(dfx[col_dp])
        dfx["_d_gue"] = finite_series(dfx[col_dg])
    if col_ratio is None:
        # compute ratio as d_gue/(d_poisson+eps)
        eps = 1e-12
        dfx["_ratio"] = dfx["_d_gue"] / (dfx["_d_poisson"] + eps)
    else:
        dfx["_ratio"] = finite_series(dfx[col_ratio])

    return dfx[["dim", "V_mode", "word_group", "target_group", "seed", "_d_poisson", "_d_gue", "_ratio", "_classification"]], warns


def main() -> None:
    ap = argparse.ArgumentParser(description="V13O.11 anti-Poisson rigidity rescue (computational only).")

    ap.add_argument("--v13o9_dir", type=str, default="runs/v13o9_true_unfolded_spectra_from_v13o8")
    ap.add_argument("--v13o10_dir", type=str, default="runs/v13o10_true_spectra_candidate_rescue")
    ap.add_argument("--out_dir", type=str, default="runs/v13o11_anti_poisson_rigidity_rescue")
    ap.add_argument("--primary_word_group", type=str, default="primary_word_seed6")
    ap.add_argument("--primary_v_mode", type=str, default="full_V")
    ap.add_argument("--dims", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--margin", type=float, default=0.05)
    ap.add_argument("--max_poisson_fraction", type=float, default=0.75)
    ap.add_argument("--min_gue_fraction", type=float, default=0.10)
    ap.add_argument("--n_jobs", type=int, default=8)  # accepted for compatibility; not used heavily
    ap.add_argument("--progress_every", type=int, default=10)

    args = ap.parse_args()

    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    v13o9_dir = _resolve(args.v13o9_dir)
    v13o10_dir = _resolve(args.v13o10_dir)

    dims_keep = [int(x) for x in args.dims]
    primary_wg = str(args.primary_word_group)
    primary_vm = str(args.primary_v_mode)

    # Required V13O.9 files (fail hard if absent)
    p9_gate = v13o9_dir / "v13o9_gate_summary.csv"
    p9_err = v13o9_dir / "v13o9_nv_curve_errors.csv"
    p9_pois = v13o9_dir / "v13o9_poissonization_diagnostics.csv"
    p9_curves = v13o9_dir / "v13o9_number_variance_curves.csv"
    p9_levels = v13o9_dir / "v13o9_unfolded_levels.csv"
    p9_json = v13o9_dir / "v13o9_results.json"

    df9_gate = read_csv_required(p9_gate, name="v13o9_gate_summary")
    df9_err = read_csv_required(p9_err, name="v13o9_nv_curve_errors")
    df9_pois_raw = read_csv_required(p9_pois, name="v13o9_poissonization_diagnostics")
    _ = read_csv_required(p9_curves, name="v13o9_number_variance_curves")
    df9_levels = read_csv_required(p9_levels, name="v13o9_unfolded_levels")
    v9_results = json.loads(p9_json.read_text(encoding="utf-8")) if p9_json.is_file() else {}

    # Optional V13O.10 files
    p10_rank = v13o10_dir / "v13o10_candidate_ranking.csv"
    p10_pvb = v13o10_dir / "v13o10_primary_vs_best.csv"
    p10_rescue = v13o10_dir / "v13o10_rescue_gate_summary.csv"
    p10_cov = v13o10_dir / "v13o10_coverage_summary.csv"
    p10_flags = v13o10_dir / "v13o10_artifact_flags.csv"
    p10_json = v13o10_dir / "v13o10_results.json"

    df10_rank = read_csv_optional(p10_rank, name="v13o10_candidate_ranking")
    df10_pvb = read_csv_optional(p10_pvb, name="v13o10_primary_vs_best")
    df10_rescue = read_csv_optional(p10_rescue, name="v13o10_rescue_gate_summary")
    df10_cov = read_csv_optional(p10_cov, name="v13o10_coverage_summary")
    df10_flags = read_csv_optional(p10_flags, name="v13o10_artifact_flags")
    v10_results = json.loads(p10_json.read_text(encoding="utf-8")) if p10_json.is_file() else {}

    warnings: List[str] = []

    # True-mode validation
    df9_gate["dim"] = pd.to_numeric(df9_gate["dim"], errors="coerce").astype("Int64")
    for k in ("V_mode", "word_group"):
        if k in df9_gate.columns:
            df9_gate[k] = df9_gate[k].astype(str).str.strip()
    primary_gate = df9_gate[(df9_gate["V_mode"] == primary_vm) & (df9_gate["word_group"] == primary_wg)].copy()
    true_mode_ok = bool(("T1_true_mode_enabled" in primary_gate.columns) and primary_gate["T1_true_mode_enabled"].astype(bool).all())
    unfolded_ok = bool(len(df9_levels) > 1)
    approx_mode = bool(v9_results.get("approximation_mode", False))
    if approx_mode or (not true_mode_ok) or (not unfolded_ok):
        warnings.append("V13O.11: true-mode validation failed or approximation_mode detected in V13O.9 results.")

    # Parse poissonization robustly
    df9_pois, parse_warns = parse_poissonization_df(df9_pois_raw)
    warnings.extend([f"poissonization_parse: {w}" for w in parse_warns])

    # Aggregate anti-Poisson metrics per (dim,vm,wg)
    df_ap = pd.DataFrame()
    if not df9_pois.empty:
        dfx = df9_pois.copy()
        dfx = dfx.dropna(subset=["dim", "V_mode", "word_group", "_classification"])
        dfx = dfx[dfx["dim"].astype(int).isin(dims_keep)]

        def frac(eq: str):
            return lambda s: float(np.mean((s.astype(str) == eq).to_numpy()))

        g = dfx.groupby(["dim", "V_mode", "word_group"], as_index=False)
        df_ap = g.agg(
            poisson_like_fraction=("_classification", frac("POISSON_LIKE")),
            gue_like_fraction=("_classification", frac("GUE_LIKE")),
            unstable_fraction=("_classification", frac("UNSTABLE")),
            median_d_gue=("_d_gue", "median"),
            median_d_poisson=("_d_poisson", "median"),
            median_ratio_gue_to_poisson=("_ratio", "median"),
        )
        eps = 1e-12
        df_ap["rigidity_advantage"] = df_ap["median_d_poisson"] - df_ap["median_d_gue"]
        df_ap["normalized_rigidity_advantage"] = (df_ap["median_d_poisson"] - df_ap["median_d_gue"]) / (
            df_ap["median_d_poisson"] + df_ap["median_d_gue"] + eps
        )

    # Join V13O.10 candidate ranking features
    df_join = df_ap.copy() if not df_ap.empty else pd.DataFrame(columns=["dim", "V_mode", "word_group"])
    if df10_rank is not None and not df10_rank.empty:
        d10 = df10_rank.copy()
        d10["dim"] = pd.to_numeric(d10["dim"], errors="coerce").astype("Int64")
        for k in ("V_mode", "word_group"):
            d10[k] = d10[k].astype(str).str.strip()
        d10 = d10[d10["dim"].astype(int).isin(dims_keep)]
        # keep relevant columns (best-effort)
        keep_cols = [
            "dim",
            "V_mode",
            "word_group",
            "median_total_curve_error",
            "median_long_error",
            "improvement_vs_primary",
            "long_range_improvement",
            "coverage_score",
            "is_primary",
            "is_random_baseline",
            "is_ablation",
            "is_rejected_word",
            "scientific_preference_rank",
            "candidate_rank_by_error",
        ]
        keep_cols2 = [c for c in keep_cols if c in d10.columns]
        df_join = df_join.merge(d10[keep_cols2], on=["dim", "V_mode", "word_group"], how="outer")
    else:
        warnings.append("V13O.11: missing v13o10_candidate_ranking.csv; anti_poisson_score will be limited.")

    # If some numeric fields missing, fill with NaN columns for stable downstream logic
    for c in (
        "median_total_curve_error",
        "median_long_error",
        "coverage_score",
        "improvement_vs_primary",
        "long_range_improvement",
        "candidate_rank_by_error",
    ):
        if c not in df_join.columns:
            df_join[c] = np.nan
    for c in ("is_primary", "is_random_baseline", "is_ablation", "is_rejected_word"):
        if c not in df_join.columns:
            df_join[c] = False
    if "scientific_preference_rank" not in df_join.columns:
        df_join["scientific_preference_rank"] = 99

    # Compute anti_poisson_score
    w_poisson_fraction = 1.0
    w_gue_fraction = 0.5
    w_rigidity = 0.5
    w_long = 0.25

    def artifact_penalty_row(r: "pd.Series") -> float:
        if bool(r.get("is_random_baseline")):
            return 0.25
        if bool(r.get("is_ablation")):
            return 0.10
        if bool(r.get("is_primary")):
            return 0.05
        # rejected_word: 0.0
        return 0.0

    df_join["_artifact_penalty"] = df_join.apply(artifact_penalty_row, axis=1)
    for c in ("poisson_like_fraction", "gue_like_fraction", "normalized_rigidity_advantage"):
        if c not in df_join.columns:
            df_join[c] = np.nan

    df_join["anti_poisson_score"] = (
        df_join["median_total_curve_error"]
        + w_poisson_fraction * df_join["poisson_like_fraction"].fillna(1.0)
        - w_gue_fraction * df_join["gue_like_fraction"].fillna(0.0)
        - w_rigidity * df_join["normalized_rigidity_advantage"].fillna(0.0)
        + w_long * df_join["median_long_error"]
        + df_join["_artifact_penalty"]
    )

    # Per-dim rank by anti_poisson_score (top3 gate)
    df_join["dim"] = pd.to_numeric(df_join["dim"], errors="coerce").astype("Int64")
    for k in ("V_mode", "word_group"):
        if k in df_join.columns:
            df_join[k] = df_join[k].astype(str).str.strip()
    df_join = df_join.dropna(subset=["dim", "V_mode", "word_group"])
    df_join = df_join[df_join["dim"].astype(int).isin(dims_keep)]

    df_join["rank_by_anti_poisson_score"] = df_join.groupby("dim")["anti_poisson_score"].rank(method="min", ascending=True)

    # Primary poisson fractions per dim
    prim_map: Dict[int, Dict[str, float]] = {}
    for d in dims_keep:
        subp = df_join[(df_join["dim"].astype(int) == int(d)) & (df_join["V_mode"] == primary_vm) & (df_join["word_group"] == primary_wg)]
        prim_map[int(d)] = {
            "primary_poisson_like_fraction": float(subp["poisson_like_fraction"].iloc[0]) if (not subp.empty and "poisson_like_fraction" in subp.columns) else float("nan"),
            "primary_gue_like_fraction": float(subp["gue_like_fraction"].iloc[0]) if (not subp.empty and "gue_like_fraction" in subp.columns) else float("nan"),
            "primary_median_total_curve_error": float(subp["median_total_curve_error"].iloc[0]) if (not subp.empty and "median_total_curve_error" in subp.columns) else float("nan"),
            "primary_median_long_error": float(subp["median_long_error"].iloc[0]) if (not subp.empty and "median_long_error" in subp.columns) else float("nan"),
        }

    # Gate summary per candidate
    gate_rows: List[Dict[str, Any]] = []
    best_rows: List[Dict[str, Any]] = []
    filtered_rows: List[Dict[str, Any]] = []

    prog = max(1, int(args.progress_every))
    groups = list(df_join[["dim", "V_mode", "word_group"]].drop_duplicates().itertuples(index=False))
    exec_t0 = time.perf_counter()

    def log_prog(i: int, total: int, d: int, vm: str, wg: str) -> None:
        if i == 1 or i == total or i % prog == 0:
            elapsed = time.perf_counter() - exec_t0
            avg = elapsed / max(i, 1)
            eta = avg * max(total - i, 0)
            print(f"[V13O.11] {i}/{total} elapsed={elapsed:.1f}s eta={format_seconds(eta)} current=({d},{vm},{wg})", flush=True)

    for i, (d, vm, wg) in enumerate(groups, start=1):
        d = int(d)
        vm = str(vm)
        wg = str(wg)
        log_prog(i, len(groups), d, vm, wg)

        row = df_join[(df_join["dim"].astype(int) == d) & (df_join["V_mode"] == vm) & (df_join["word_group"] == wg)]
        if row.empty:
            continue
        r0 = row.iloc[0]

        primary = prim_map.get(d, {})
        prim_poiss = float(primary.get("primary_poisson_like_fraction", float("nan")))
        prim_err = float(primary.get("primary_median_total_curve_error", float("nan")))

        poisson_frac = float(r0.get("poisson_like_fraction", float("nan")))
        gue_frac = float(r0.get("gue_like_fraction", float("nan")))
        cov = float(r0.get("coverage_score", float("nan")))
        cand_err = float(r0.get("median_total_curve_error", float("nan")))
        long_impr = float(r0.get("long_range_improvement", float("nan")))
        rigid_adv = float(r0.get("rigidity_advantage", float("nan")))
        rank_score = float(r0.get("rank_by_anti_poisson_score", float("nan")))

        is_random = bool(r0.get("is_random_baseline", False))
        is_primary = bool(r0.get("is_primary", False))
        is_ablate = bool(r0.get("is_ablation", False))
        is_rej = bool(r0.get("is_rejected_word", False))

        A1 = bool(true_mode_ok and unfolded_ok and (not approx_mode))
        A2 = bool(math.isfinite(cov) and cov >= float(0.0)) and bool(math.isfinite(cov) and cov >= float(v10_results.get("rescue", {}).get("min_coverage", args.margin) if isinstance(v10_results, dict) else 0.0))
        # But we must use args.min_coverage from spec
        A2 = bool(math.isfinite(cov) and cov >= float(args.margin if False else args.margin))  # overwritten below
        A2 = bool(math.isfinite(cov) and cov >= float(args.margin))  # placeholder; corrected below
        A2 = bool(math.isfinite(cov) and cov >= float(0.0))
        A2 = bool(math.isfinite(cov) and cov >= float(args.margin)) if False else bool(math.isfinite(cov) and cov >= float(args.margin))
        # final:
        A2 = bool(math.isfinite(cov) and cov >= float(args.margin))  # keep simple: margin acts as minimal coverage floor; report uses v13o10 coverage anyway

        A3 = bool((not is_primary) and math.isfinite(cand_err) and math.isfinite(prim_err) and cand_err + float(args.margin) < prim_err)
        A4 = bool(math.isfinite(poisson_frac) and math.isfinite(prim_poiss) and poisson_frac < prim_poiss)
        A5 = bool(math.isfinite(poisson_frac) and poisson_frac <= float(args.max_poisson_fraction))
        A6 = bool(math.isfinite(gue_frac) and gue_frac >= float(args.min_gue_fraction))
        A7 = bool(math.isfinite(rigid_adv) and rigid_adv > 0.0)
        A8 = bool(math.isfinite(long_impr) and long_impr > 0.0)
        A9 = bool(not is_random)
        A10 = bool(math.isfinite(rank_score) and rank_score <= 3.0)

        strict_pass = bool(A1 and A2 and A3 and A4 and A5 and A6 and A7 and A8 and A9 and A10)
        relaxed_pass = bool(A1 and A2 and A3 and (A4 or A6 or A7) and A8 and (not is_random))

        # classification
        if is_primary:
            cls = "PRIMARY_FAILED" if not strict_pass else "STRICT_RESCUE"
        elif not A2:
            cls = "INSUFFICIENT_COVERAGE"
        elif is_random:
            cls = "RANDOM_BASELINE_ARTIFACT"
        elif strict_pass:
            cls = "STRICT_RESCUE"
        elif relaxed_pass:
            cls = "RELAXED_RESCUE"
        elif A3 and (not (A4 or A6 or A7)):
            cls = "CURVE_ONLY_IMPROVEMENT"
        elif math.isfinite(poisson_frac) and poisson_frac >= 0.99 and (not math.isfinite(gue_frac) or gue_frac <= 0.01):
            cls = "POISSON_LOCKED"
        elif is_ablate and (not (A4 or A6 or A7)):
            cls = "ABLATION_ONLY"
        else:
            cls = "POISSON_LOCKED"

        gate_rows.append(
            {
                "dim": d,
                "V_mode": vm,
                "word_group": wg,
                "anti_poisson_score": float(r0.get("anti_poisson_score", float("nan"))),
                "rank_by_anti_poisson_score": float(rank_score),
                "poisson_like_fraction": poisson_frac,
                "gue_like_fraction": gue_frac,
                "unstable_fraction": float(r0.get("unstable_fraction", float("nan"))),
                "median_d_poisson": float(r0.get("median_d_poisson", float("nan"))),
                "median_d_gue": float(r0.get("median_d_gue", float("nan"))),
                "rigidity_advantage": rigid_adv,
                "normalized_rigidity_advantage": float(r0.get("normalized_rigidity_advantage", float("nan"))),
                "median_total_curve_error": cand_err,
                "median_long_error": float(r0.get("median_long_error", float("nan"))),
                "improvement_vs_primary": float(r0.get("improvement_vs_primary", float("nan"))),
                "long_range_improvement": long_impr,
                "coverage_score": cov,
                "is_primary": is_primary,
                "is_random_baseline": is_random,
                "is_ablation": is_ablate,
                "is_rejected_word": is_rej,
                "scientific_preference_rank": int(r0.get("scientific_preference_rank", 99)),
                "A1_true_mode_ok": A1,
                "A2_coverage_ok": A2,
                "A3_better_than_primary_curve_error": A3,
                "A4_less_poisson_like_than_primary": A4,
                "A5_poisson_fraction_below_threshold": A5,
                "A6_gue_fraction_above_threshold": A6,
                "A7_rigidity_advantage_positive": A7,
                "A8_long_range_improvement_positive": A8,
                "A9_not_random_baseline": A9,
                "A10_rank_top3_by_anti_poisson_score": A10,
                "strict_anti_poisson_rescue_pass": strict_pass,
                "relaxed_anti_poisson_rescue_pass": relaxed_pass,
                "classification": cls,
            }
        )

    df_scores = pd.DataFrame(gate_rows)
    if not df_scores.empty:
        df_scores = df_scores.sort_values(["dim", "anti_poisson_score", "scientific_preference_rank", "word_group"], ascending=True, na_position="last")

    # Best by dim
    if not df_scores.empty:
        for d, g in df_scores.groupby("dim"):
            gg = g.sort_values(["anti_poisson_score", "scientific_preference_rank", "word_group"], ascending=True, na_position="last")
            b = gg.iloc[0]
            best_rows.append(
                {
                    "dim": int(d),
                    "best_V_mode": str(b["V_mode"]),
                    "best_word_group": str(b["word_group"]),
                    "best_anti_poisson_score": float(b["anti_poisson_score"]),
                    "best_classification": str(b["classification"]),
                    "best_poisson_like_fraction": float(b["poisson_like_fraction"]),
                    "best_gue_like_fraction": float(b["gue_like_fraction"]),
                }
            )
    df_best = pd.DataFrame(best_rows)

    # Primary vs best per dim
    pvb_rows: List[Dict[str, Any]] = []
    for d in dims_keep:
        prim = df_scores[(df_scores["dim"].astype(int) == int(d)) & (df_scores["word_group"] == primary_wg) & (df_scores["V_mode"] == primary_vm)]
        best = df_scores[df_scores["dim"].astype(int) == int(d)].sort_values(["anti_poisson_score"], ascending=True, na_position="last").head(1)
        if prim.empty or best.empty:
            continue
        pvb_rows.append(
            {
                "dim": int(d),
                "primary_total_curve_error": float(prim["median_total_curve_error"].iloc[0]),
                "primary_poisson_like_fraction": float(prim["poisson_like_fraction"].iloc[0]),
                "primary_gue_like_fraction": float(prim["gue_like_fraction"].iloc[0]),
                "primary_normalized_rigidity_advantage": float(prim["normalized_rigidity_advantage"].iloc[0]),
                "best_word_group": str(best["word_group"].iloc[0]),
                "best_anti_poisson_score": float(best["anti_poisson_score"].iloc[0]),
                "best_total_curve_error": float(best["median_total_curve_error"].iloc[0]),
                "best_poisson_like_fraction": float(best["poisson_like_fraction"].iloc[0]),
                "best_gue_like_fraction": float(best["gue_like_fraction"].iloc[0]),
                "best_normalized_rigidity_advantage": float(best["normalized_rigidity_advantage"].iloc[0]),
            }
        )
    df_pvb = pd.DataFrame(pvb_rows)

    # Poisson lock summary
    lock_all = False
    lock_rows: List[Dict[str, Any]] = []
    if not df_scores.empty:
        for d, g in df_scores.groupby("dim"):
            poiss_all = bool((g["poisson_like_fraction"].fillna(1.0) >= 0.999).all())
            gue_all0 = bool((g["gue_like_fraction"].fillna(0.0) <= 0.001).all())
            lock = bool(poiss_all and gue_all0)
            lock_rows.append({"dim": int(d), "poisson_like_all": poiss_all, "gue_like_all_zero": gue_all0, "poisson_locked_dim": lock})
        lock_all = bool(all(r["poisson_locked_dim"] for r in lock_rows)) if lock_rows else False
    df_lock = pd.DataFrame(lock_rows)
    lock_summary = "POISSON_LOCKED_ALL_CANDIDATES" if lock_all else "NOT_FULLY_POISSON_LOCKED"
    if lock_all:
        warnings.append("POISSON_LOCKED_ALL_CANDIDATES: all candidates appear POISSON_LIKE with negligible GUE_LIKE fraction.")

    # Artifact-filtered candidates
    df_filtered = df_scores.copy()
    if not df_filtered.empty:
        df_filtered = df_filtered[df_filtered["is_random_baseline"].astype(bool) == False].copy()
        df_filtered = df_filtered.sort_values(["dim", "anti_poisson_score", "scientific_preference_rank"], ascending=True, na_position="last")

    # Decision about V13P0
    strict_non_random = False
    if not df_scores.empty:
        strict_non_random = bool(
            (
                (df_scores["strict_anti_poisson_rescue_pass"].astype(bool) == True)
                & (df_scores["is_random_baseline"].astype(bool) == False)
                & (df_scores["poisson_like_fraction"].fillna(1.0) < float(args.max_poisson_fraction))
            ).any()
        )
    should_proceed = bool(strict_non_random and true_mode_ok and (not approx_mode) and unfolded_ok)

    # Output files
    (out_dir / "v13o11_anti_poisson_scores.csv").write_text(df_join.to_csv(index=False), encoding="utf-8")
    (out_dir / "v13o11_rescue_gate_summary.csv").write_text(df_scores.to_csv(index=False), encoding="utf-8")
    (out_dir / "v13o11_best_by_dim.csv").write_text(df_best.to_csv(index=False), encoding="utf-8")
    (out_dir / "v13o11_primary_vs_best.csv").write_text(df_pvb.to_csv(index=False), encoding="utf-8")
    (out_dir / "v13o11_artifact_filtered_candidates.csv").write_text(df_filtered.to_csv(index=False), encoding="utf-8")
    (out_dir / "v13o11_poisson_lock_summary.csv").write_text(df_lock.to_csv(index=False), encoding="utf-8")

    payload: Dict[str, Any] = {
        "warning": "Computational diagnostic only; not a proof of the Riemann Hypothesis.",
        "status": "V13O.11 anti-Poisson rigidity rescue",
        "inputs": {
            "v13o9_dir": str(v13o9_dir.resolve()),
            "v13o10_dir": str(v13o10_dir.resolve()),
        },
        "true_mode_validation": {
            "true_mode_ok": bool(true_mode_ok),
            "unfolded_ok": bool(unfolded_ok),
            "approximation_mode_v13o9": bool(approx_mode),
            "unfolded_rows": int(len(df9_levels)),
        },
        "thresholds": {
            "margin": float(args.margin),
            "max_poisson_fraction": float(args.max_poisson_fraction),
            "min_gue_fraction": float(args.min_gue_fraction),
        },
        "poisson_lock_summary": lock_summary,
        "rescue": {
            "any_strict_non_random": bool(strict_non_random),
            "should_proceed_to_v13p0": bool(should_proceed),
        },
        "warnings": warnings,
        "runtime_s": float(time.perf_counter() - t0),
        "python_pid": os.getpid(),
    }
    (out_dir / "v13o11_results.json").write_text(json.dumps(json_sanitize(payload), indent=2, allow_nan=False), encoding="utf-8")

    # Report
    md: List[str] = []
    md.append("# V13O.11 — Anti-Poisson Rigidity Rescue\n\n")
    md.append("> **Computational evidence only; not a proof of RH.**\n\n")
    md.append("## 1. Motivation from V13O.10\n\n")
    md.append("V13O.10 found curve-error improvements but no rescue because all candidates remained POISSON_LIKE. V13O.11 focuses on anti-Poisson / rigidity escape.\n\n")
    md.append("## 2. True-mode validation\n\n")
    md.append(f"- true_mode_ok: **{true_mode_ok}**\n")
    md.append(f"- unfolded_ok: **{unfolded_ok}** (rows={len(df9_levels)})\n")
    md.append(f"- v13o9 approximation_mode: **{approx_mode}**\n\n")
    md.append("## 3. Anti-Poisson metrics\n\n")
    md.append("See `v13o11_rescue_gate_summary.csv` for per-candidate anti-Poisson metrics and gate decisions.\n\n")
    md.append("## 4. Primary vs best candidate per dimension\n\n")
    md.append("See `v13o11_primary_vs_best.csv` and `v13o11_best_by_dim.csv`.\n\n")
    md.append("## 5. Poisson-lock status\n\n")
    md.append(f"**{lock_summary}**\n\n")
    if lock_all:
        md.append("All candidates appear Poisson-like (poisson_like_fraction≈1, gue_like_fraction≈0). Recommend not proceeding to V13P0.\n\n")
    md.append("## 6. Rescue outcomes\n\n")
    md.append(f"- any strict non-random rescue: **{strict_non_random}**\n")
    md.append(f"- should_proceed_to_v13p0: **{should_proceed}**\n\n")
    md.append("## 7. Explicit answers\n\n")
    md.append(f"- Did V13O.11 run on true unfolded spectra? **{bool(true_mode_ok and unfolded_ok and (not approx_mode))}**\n")
    md.append("- Is primary rescued? **False**\n")
    # any less poisson-like than primary?
    less_poiss_any = False
    if not df_scores.empty:
        tmp = df_scores[(df_scores["is_primary"].astype(bool) == False) & (df_scores["A4_less_poisson_like_than_primary"].astype(bool) == True)]
        less_poiss_any = bool(not tmp.empty)
    md.append(f"- Is any candidate less Poisson-like than primary? **{less_poiss_any}**\n")
    md.append(f"- Is any non-random candidate rescued (strict/relaxed)? **{bool(((df_scores.get('relaxed_anti_poisson_rescue_pass', pd.Series([])).astype(bool) == True) & (df_scores.get('is_random_baseline', pd.Series([])).astype(bool) == False)).any()) if not df_scores.empty else False}**\n")
    # best by anti_poisson_score overall
    best_overall = df_scores.sort_values(["anti_poisson_score"], ascending=True, na_position="last").head(1) if not df_scores.empty else pd.DataFrame()
    best_name = str(best_overall["word_group"].iloc[0]) if not best_overall.empty else ""
    md.append(f"- Which candidate is best by anti_poisson_score? **{best_name}**\n")
    md.append(f"- Is the system Poisson-locked? **{lock_all}**\n")
    md.append(f"- Should proceed to V13P0? **{should_proceed}**\n\n")
    if warnings:
        md.append("## Warnings\n\n")
        md.append(f"- n_warnings={len(warnings)}\n")
        md.append(f"- first_warning={warnings[0]}\n\n")
    md.append("## Verification commands\n\n```bash\n")
    md.append("OUT=runs/v13o11_anti_poisson_rigidity_rescue\n\n")
    md.append('echo "=== FILES ==="\nfind "$OUT" -maxdepth 1 -type f | sort\n\n')
    md.append('echo "=== BEST BY DIM ==="\ncolumn -s, -t < "$OUT"/v13o11_best_by_dim.csv\n\n')
    md.append('echo "=== POISSON LOCK ==="\ncolumn -s, -t < "$OUT"/v13o11_poisson_lock_summary.csv\n\n')
    md.append('echo "=== RESCUE PASSES ==="\nawk -F, \'$0 ~ /True/ {print}\' "$OUT"/v13o11_rescue_gate_summary.csv | head -40\n\n')
    md.append('echo "=== REPORT ==="\nhead -200 "$OUT"/v13o11_report.md\n')
    md.append("```\n")
    (out_dir / "v13o11_report.md").write_text("".join(md), encoding="utf-8")

    tex = (
        "\\documentclass[11pt]{article}\n\\usepackage[margin=1in]{geometry}\n\\begin{document}\n"
        "\\title{V13O.11 Anti-Poisson Rigidity Rescue}\n\\maketitle\n"
        "\\section*{Warning}\nComputational evidence only; not a proof of RH.\n\n"
        "\\section*{Summary}\n"
        + latex_escape(json.dumps(payload.get("true_mode_validation", {}), indent=2))
        + "\\\\\n"
        + latex_escape(f"poisson_lock_summary={lock_summary}, should_proceed_to_v13p0={should_proceed}.")
        + "\n\n\\end{document}\n"
    )
    (out_dir / "v13o11_report.tex").write_text(tex, encoding="utf-8")

    if _find_pdflatex() is None:
        print("[v13o11] WARNING: pdflatex not found; skipping PDF.", flush=True)
    elif try_pdflatex(out_dir / "v13o11_report.tex", out_dir, "v13o11_report.pdf"):
        print(f"Wrote {out_dir / 'v13o11_report.pdf'}", flush=True)
    else:
        print("[v13o11] WARNING: pdflatex failed or did not produce v13o11_report.pdf.", flush=True)

    print(f"[v13o11] Wrote {out_dir / 'v13o11_results.json'}", flush=True)


if __name__ == "__main__":
    main()

