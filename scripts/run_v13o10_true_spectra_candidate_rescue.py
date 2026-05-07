#!/usr/bin/env python3
"""
V13O.10 — True spectra candidate rescue (ranking layer over V13O.9 true-mode outputs).

Computational diagnostic only; not a proof of the Riemann Hypothesis.

This script loads V13O.9 outputs, validates true-mode, computes coverage and aggregated
curve/poissonization metrics per (dim,V_mode,word_group), ranks candidates against primary,
and applies a "rescue gate" intended to identify potentially interesting non-primary candidates.
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


def read_csv_robust(path: Path, *, name: str, tag: str = "v13o10") -> Optional["pd.DataFrame"]:
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


def finite_series(x: "pd.Series") -> "pd.Series":
    return pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan)


def safe_median(xs: Sequence[float]) -> float:
    arr = np.asarray([float(v) for v in xs if v is not None and math.isfinite(float(v))], dtype=np.float64)
    return float(np.median(arr)) if arr.size else float("nan")


def artifact_flags(word_group: str, primary_word_group: str) -> Dict[str, bool]:
    wg = str(word_group)
    return {
        "is_primary": wg == str(primary_word_group),
        "is_random_baseline": ("random_symmetric_baseline" in wg) or ("random_words" in wg),
        "is_ablation": wg.startswith("ablate_"),
        "is_rejected_word": "rejected_word" in wg,
    }


def scientific_preference_rank(flags: Dict[str, bool]) -> int:
    """
    Lower is more scientifically preferred (per spec):
      rejected_word > ablation > random_words > random_symmetric_baseline
    """
    if flags.get("is_rejected_word"):
        return 0
    if flags.get("is_ablation"):
        return 1
    if flags.get("is_random_baseline"):
        # random_words and random_symmetric_baseline are both baselines
        return 3 if flags.get("is_random_baseline") else 2
    return 2


def main() -> None:
    ap = argparse.ArgumentParser(description="V13O.10 candidate rescue over V13O.9 true spectra (computational only).")

    ap.add_argument("--v13o9_dir", type=str, default="")
    ap.add_argument("--v13o9_gate_summary", type=str, default="")
    ap.add_argument("--v13o9_nv_curve_errors", type=str, default="")
    ap.add_argument("--v13o9_poissonization", type=str, default="")
    ap.add_argument("--v13o9_best_by_dim", type=str, default="")
    ap.add_argument("--v13o9_unfolded_levels", type=str, default="")

    ap.add_argument("--out_dir", type=str, default="runs/v13o10_true_spectra_candidate_rescue")
    ap.add_argument("--primary_word_group", type=str, default="primary_word_seed6")
    ap.add_argument("--primary_v_mode", type=str, default="full_V")
    ap.add_argument("--dims", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--min_coverage", type=float, default=0.7)
    ap.add_argument("--max_rank_for_rescue", type=int, default=2)
    ap.add_argument("--progress_every", type=int, default=1)

    args = ap.parse_args()

    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    dims_keep = [int(x) for x in args.dims]
    primary_wg = str(args.primary_word_group)
    primary_vm = str(args.primary_v_mode)
    prog = max(1, int(args.progress_every))

    # Resolve V13O.9 paths
    v13o9_dir = _resolve(args.v13o9_dir) if str(args.v13o9_dir).strip() else None
    if v13o9_dir is not None:
        gate_p = v13o9_dir / "v13o9_gate_summary.csv"
        err_p = v13o9_dir / "v13o9_nv_curve_errors.csv"
        pois_p = v13o9_dir / "v13o9_poissonization_diagnostics.csv"
        best_p = v13o9_dir / "v13o9_best_by_dim.csv"
        lev_p = v13o9_dir / "v13o9_unfolded_levels.csv"
    else:
        gate_p = _resolve(args.v13o9_gate_summary) if str(args.v13o9_gate_summary).strip() else Path("")
        err_p = _resolve(args.v13o9_nv_curve_errors) if str(args.v13o9_nv_curve_errors).strip() else Path("")
        pois_p = _resolve(args.v13o9_poissonization) if str(args.v13o9_poissonization).strip() else Path("")
        best_p = _resolve(args.v13o9_best_by_dim) if str(args.v13o9_best_by_dim).strip() else Path("")
        lev_p = _resolve(args.v13o9_unfolded_levels) if str(args.v13o9_unfolded_levels).strip() else Path("")

    # Load
    df_gate = read_csv_robust(gate_p, name="v13o9_gate_summary")
    df_err = read_csv_robust(err_p, name="v13o9_nv_curve_errors")
    df_pois = read_csv_robust(pois_p, name="v13o9_poissonization_diagnostics")
    df_best = read_csv_robust(best_p, name="v13o9_best_by_dim")
    df_levels = read_csv_robust(lev_p, name="v13o9_unfolded_levels")

    warnings: List[str] = []
    missing_required = []
    for nm, df in (("gate_summary", df_gate), ("nv_curve_errors", df_err), ("poissonization", df_pois), ("unfolded_levels", df_levels)):
        if df is None:
            missing_required.append(nm)
    if missing_required:
        warnings.append(f"Missing required V13O.9 inputs: {missing_required}")

    # True-mode validation
    true_mode_ok = False
    unfolded_rows = 0
    if df_gate is not None and "T1_true_mode_enabled" in df_gate.columns:
        try:
            true_mode_ok = bool(df_gate["T1_true_mode_enabled"].astype(bool).any())
        except Exception:
            true_mode_ok = False
    if df_levels is not None:
        unfolded_rows = int(len(df_levels))
    unfolded_ok = bool(unfolded_rows > 1)
    approx_detected = (not true_mode_ok) or (not unfolded_ok)
    if approx_detected:
        warnings.append("V13O.10 detected non-true-mode or missing unfolded levels. Results will be limited.")

    # Coverage summary from nv_curve_errors (and optionally unfolded_levels)
    df_cov = pd.DataFrame(columns=["dim", "V_mode", "word_group", "n_target_groups", "n_seeds", "pair_coverage", "coverage_score"])
    if df_err is not None and all(c in df_err.columns for c in ("dim", "V_mode", "word_group", "target_group", "seed")):
        dfx = df_err.copy()
        dfx["dim"] = pd.to_numeric(dfx["dim"], errors="coerce").astype("Int64")
        dfx["seed"] = pd.to_numeric(dfx["seed"], errors="coerce").astype("Int64")
        for k in ("V_mode", "word_group", "target_group"):
            dfx[k] = dfx[k].astype(str).str.strip()
        dfx = dfx.dropna(subset=["dim", "V_mode", "word_group", "target_group", "seed"])
        dfx = dfx[dfx["dim"].astype(int).isin(dims_keep)]

        # infer operator/target coverage from unfolded_levels if possible
        pair_cov_map: Dict[Tuple[int, str, str], float] = {}
        if df_levels is not None and all(c in df_levels.columns for c in ("dim", "V_mode", "word_group", "target_group", "seed", "source")):
            dfl = df_levels.copy()
            dfl["dim"] = pd.to_numeric(dfl["dim"], errors="coerce").astype("Int64")
            dfl["seed"] = pd.to_numeric(dfl["seed"], errors="coerce").astype("Int64")
            for k in ("V_mode", "word_group", "target_group", "source"):
                dfl[k] = dfl[k].astype(str).str.strip()
            dfl = dfl.dropna(subset=["dim", "V_mode", "word_group", "target_group", "seed", "source"])
            dfl = dfl[dfl["dim"].astype(int).isin(dims_keep)]
            # for each (dim,vm,wg,tg,seed) check both operator and target exist
            g2 = dfl.groupby(["dim", "V_mode", "word_group", "target_group", "seed"])["source"].apply(lambda s: set(s.tolist()))
            ok_pairs = g2.apply(lambda ss: ("operator" in ss) and ("target" in ss))
            # aggregate per (dim,vm,wg)
            for (d, vm, wg), sub in ok_pairs.groupby(level=[0, 1, 2]):
                arr = sub.to_numpy(dtype=bool)
                pair_cov_map[(int(d), str(vm), str(wg))] = float(np.mean(arr)) if arr.size else float("nan")

        rows = []
        for (d, vm, wg), g in dfx.groupby(["dim", "V_mode", "word_group"]):
            n_tg = int(g["target_group"].nunique())
            n_seed = int(g["seed"].nunique())
            expected_tg = 1 + len(CONTROL_TARGET_GROUPS)  # real_zeta + controls
            # heuristic expected seeds: use max within this dim across groups (robust)
            coverage_tg = float(n_tg) / float(expected_tg) if expected_tg > 0 else 0.0
            # normalize seeds by max seen for this dim
            max_seed_dim = int(dfx[dfx["dim"] == d]["seed"].nunique()) if int(d) in dfx["dim"].astype(int).unique() else n_seed
            coverage_seed = float(n_seed) / float(max(1, max_seed_dim))
            pair_cov = float(pair_cov_map.get((int(d), str(vm), str(wg)), float("nan")))
            # blend; if pair_cov missing, ignore it
            parts = [coverage_tg, coverage_seed]
            if math.isfinite(pair_cov):
                parts.append(pair_cov)
            cov_score = float(np.mean(np.asarray(parts, dtype=np.float64))) if parts else 0.0
            rows.append(
                {
                    "dim": int(d),
                    "V_mode": str(vm),
                    "word_group": str(wg),
                    "n_target_groups": n_tg,
                    "n_seeds": n_seed,
                    "pair_coverage": pair_cov,
                    "coverage_score": cov_score,
                }
            )
        df_cov = pd.DataFrame(rows).sort_values(["dim", "V_mode", "word_group"], ascending=True)

    # Aggregate curve errors per (dim,vm,wg)
    df_err_agg = pd.DataFrame()
    if df_err is not None and all(c in df_err.columns for c in ("dim", "V_mode", "word_group", "total_curve_error")):
        dfx = df_err.copy()
        dfx["dim"] = pd.to_numeric(dfx["dim"], errors="coerce").astype("Int64")
        for k in ("V_mode", "word_group", "target_group"):
            if k in dfx.columns:
                dfx[k] = dfx[k].astype(str).str.strip()
        for c in ("total_curve_error", "short_error", "mid_error", "long_error"):
            if c in dfx.columns:
                dfx[c] = finite_series(dfx[c])
        dfx = dfx.dropna(subset=["dim", "V_mode", "word_group", "total_curve_error"])
        dfx = dfx[dfx["dim"].astype(int).isin(dims_keep)]

        def q(x: "pd.Series", qq: float) -> float:
            a = x.dropna().astype(float).to_numpy()
            return float(np.quantile(a, qq)) if a.size else float("nan")

        g = dfx.groupby(["dim", "V_mode", "word_group"], as_index=False)
        df_err_agg = g.agg(
            n_curve_rows=("total_curve_error", "count"),
            mean_total_curve_error=("total_curve_error", "mean"),
            median_total_curve_error=("total_curve_error", "median"),
            min_total_curve_error=("total_curve_error", "min"),
            std_total_curve_error=("total_curve_error", "std"),
            mean_short_error=("short_error", "mean"),
            mean_mid_error=("mid_error", "mean"),
            mean_long_error=("long_error", "mean"),
            median_long_error=("long_error", "median"),
        )
        # quantiles (separate groupby without as_index=False to keep stable join keys)
        gk = dfx.groupby(["dim", "V_mode", "word_group"])
        q25_df = gk["total_curve_error"].apply(lambda s: q(s, 0.25)).reset_index(name="q25_total_curve_error")
        q75_df = gk["total_curve_error"].apply(lambda s: q(s, 0.75)).reset_index(name="q75_total_curve_error")
        df_err_agg = df_err_agg.merge(q25_df, on=["dim", "V_mode", "word_group"], how="left")
        df_err_agg = df_err_agg.merge(q75_df, on=["dim", "V_mode", "word_group"], how="left")

    # Aggregate poissonization per (dim,vm,wg)
    df_pois_agg = pd.DataFrame()
    if df_pois is not None and all(c in df_pois.columns for c in ("dim", "V_mode", "word_group", "classification")):
        dfx = df_pois.copy()
        dfx["dim"] = pd.to_numeric(dfx["dim"], errors="coerce").astype("Int64")
        for k in ("V_mode", "word_group", "classification"):
            dfx[k] = dfx[k].astype(str).str.strip()
        # robust distance cols
        col_dp = pick_col(dfx, ["distance_to_poisson", "poisson_distance", "dist_poisson"])
        col_dg = pick_col(dfx, ["distance_to_gue", "gue_distance", "dist_gue"])
        col_ratio = pick_col(dfx, ["poissonization_ratio", "poisson_to_gue_ratio", "ratio"])
        for c in (col_dp, col_dg, col_ratio):
            if c and c in dfx.columns:
                dfx[c] = finite_series(dfx[c])
        dfx = dfx.dropna(subset=["dim", "V_mode", "word_group", "classification"])
        dfx = dfx[dfx["dim"].astype(int).isin(dims_keep)]

        def frac(eq: str):
            return lambda s: float(np.mean((s.astype(str) == eq).to_numpy()))

        g = dfx.groupby(["dim", "V_mode", "word_group"], as_index=False)
        out = g.agg(
            poisson_like_fraction=("classification", frac("POISSON_LIKE")),
            gue_like_fraction=("classification", frac("GUE_LIKE")),
            unstable_fraction=("classification", frac("UNSTABLE")),
        )
        # numeric aggregates must merge on keys (avoid multi-column assignment errors)
        if col_dp:
            dp_df = dfx.groupby(["dim", "V_mode", "word_group"])[col_dp].mean().reset_index(name="mean_poisson_distance")
            out = out.merge(dp_df, on=["dim", "V_mode", "word_group"], how="left")
        else:
            out["mean_poisson_distance"] = float("nan")
        if col_dg:
            dg_df = dfx.groupby(["dim", "V_mode", "word_group"])[col_dg].mean().reset_index(name="mean_gue_distance")
            out = out.merge(dg_df, on=["dim", "V_mode", "word_group"], how="left")
        else:
            out["mean_gue_distance"] = float("nan")
        if col_ratio:
            rr_df = dfx.groupby(["dim", "V_mode", "word_group"])[col_ratio].median().reset_index(name="poisson_to_gue_ratio_median")
            out = out.merge(rr_df, on=["dim", "V_mode", "word_group"], how="left")
        else:
            out["poisson_to_gue_ratio_median"] = float("nan")
        df_pois_agg = out

    # Merge candidate table
    df_cand = df_err_agg.copy() if not df_err_agg.empty else pd.DataFrame(columns=["dim", "V_mode", "word_group"])
    if not df_cand.empty:
        if not df_cov.empty:
            df_cand = df_cand.merge(df_cov, on=["dim", "V_mode", "word_group"], how="left")
        if not df_pois_agg.empty:
            df_cand = df_cand.merge(df_pois_agg, on=["dim", "V_mode", "word_group"], how="left")
    else:
        # minimal skeleton from gate/best if needed
        rows = []
        if df_gate is not None and all(c in df_gate.columns for c in ("dim", "V_mode", "word_group")):
            for r in df_gate.itertuples(index=False):
                rows.append({"dim": int(getattr(r, "dim")), "V_mode": str(getattr(r, "V_mode")), "word_group": str(getattr(r, "word_group"))})
        df_cand = pd.DataFrame(rows).drop_duplicates()

    # Artifact flags
    if not df_cand.empty:
        flags_rows = []
        for r in df_cand[["word_group"]].drop_duplicates().itertuples(index=False):
            wg = str(r[0])
            fl = artifact_flags(wg, primary_wg)
            flags_rows.append({"word_group": wg, **fl, "scientific_preference_rank": scientific_preference_rank(fl)})
        df_flags = pd.DataFrame(flags_rows).sort_values(["scientific_preference_rank", "word_group"], ascending=True)
    else:
        df_flags = pd.DataFrame(columns=["word_group", "is_primary", "is_random_baseline", "is_ablation", "is_rejected_word", "scientific_preference_rank"])

    # Primary recap from gate
    primary_gate = None
    primary_by_dim: Dict[int, Dict[str, Any]] = {}
    if df_gate is not None and not df_gate.empty:
        dg = df_gate.copy()
        dg["dim"] = pd.to_numeric(dg["dim"], errors="coerce").astype("Int64")
        for k in ("V_mode", "word_group"):
            if k in dg.columns:
                dg[k] = dg[k].astype(str).str.strip()
        primary_gate = dg[(dg["V_mode"] == primary_vm) & (dg["word_group"] == primary_wg)].copy()
        for r in primary_gate.itertuples(index=False):
            d = int(getattr(r, "dim"))
            primary_by_dim[d] = {
                "primary_total_curve_error": float(getattr(r, "primary_total_curve_error", float("nan"))),
                "primary_failure_region": str(getattr(r, "primary_failure_region", "")),
                "primary_poissonization_classification": str(getattr(r, "primary_poissonization_classification", "")),
                "primary_rank": int(getattr(r, "primary_rank", -1)),
                "T1_true_mode_enabled": bool(getattr(r, "T1_true_mode_enabled", False)),
            }

    # Candidate ranking per dim: rank by median_total_curve_error (fallback mean_total_curve_error)
    ranking_rows: List[Dict[str, Any]] = []
    rescue_rows: List[Dict[str, Any]] = []

    if not df_cand.empty:
        # attach flags
        df_cand2 = df_cand.merge(df_flags, on="word_group", how="left")
        # determine error col to rank on
        err_col = "median_total_curve_error" if "median_total_curve_error" in df_cand2.columns else "mean_total_curve_error"
        long_col = "median_long_error" if "median_long_error" in df_cand2.columns else "mean_long_error"

        for dim in dims_keep:
            sub = df_cand2[df_cand2["dim"].astype(int) == int(dim)].copy()
            if sub.empty:
                continue
            # rank (1=best). use a normal column name to survive itertuples access
            sub["candidate_rank_by_error"] = sub[err_col].rank(method="min", ascending=True)
            sub = sub.sort_values([err_col, "scientific_preference_rank", "word_group"], ascending=True, na_position="last")

            prim_info = primary_by_dim.get(int(dim), {})
            prim_err = float(prim_info.get("primary_total_curve_error", float("nan")))
            prim_long = float("nan")
            # estimate primary long error from aggregated table if present
            prim_row = sub[(sub["V_mode"] == primary_vm) & (sub["word_group"] == primary_wg)]
            if not prim_row.empty and long_col in prim_row.columns:
                prim_long = float(prim_row[long_col].iloc[0])

            prim_poisson_like_frac = float("nan")
            if not prim_row.empty and "poisson_like_fraction" in prim_row.columns:
                prim_poisson_like_frac = float(prim_row["poisson_like_fraction"].iloc[0])

            for r in sub.itertuples(index=False):
                wg = str(getattr(r, "word_group"))
                vm = str(getattr(r, "V_mode"))
                cand_err = float(getattr(r, err_col, float("nan")))
                cand_long = float(getattr(r, long_col, float("nan"))) if long_col in sub.columns else float("nan")

                improvement_vs_primary = float(prim_err - cand_err) if (math.isfinite(prim_err) and math.isfinite(cand_err)) else float("nan")
                improvement_ratio = float(cand_err / prim_err) if (math.isfinite(prim_err) and math.isfinite(cand_err) and prim_err != 0.0) else float("nan")
                long_impr = float(prim_long - cand_long) if (math.isfinite(prim_long) and math.isfinite(cand_long)) else float("nan")
                cand_rank_raw = float(getattr(r, "candidate_rank_by_error", float("nan")))
                cand_rank = int(cand_rank_raw) if math.isfinite(cand_rank_raw) else -1

                cand_poisson_like_frac = float(getattr(r, "poisson_like_fraction", float("nan")))

                ranking_rows.append(
                    {
                        "dim": int(dim),
                        "V_mode": vm,
                        "word_group": wg,
                        "candidate_rank_by_error": cand_rank,
                        "coverage_score": float(getattr(r, "coverage_score", float("nan"))),
                        "median_total_curve_error": float(getattr(r, "median_total_curve_error", float("nan"))),
                        "mean_total_curve_error": float(getattr(r, "mean_total_curve_error", float("nan"))),
                        "median_long_error": float(getattr(r, "median_long_error", float("nan"))),
                        "mean_long_error": float(getattr(r, "mean_long_error", float("nan"))),
                        "poisson_like_fraction": cand_poisson_like_frac,
                        "gue_like_fraction": float(getattr(r, "gue_like_fraction", float("nan"))),
                        "unstable_fraction": float(getattr(r, "unstable_fraction", float("nan"))),
                        "improvement_vs_primary": improvement_vs_primary,
                        "improvement_ratio": improvement_ratio,
                        "long_range_improvement": long_impr,
                        "is_primary": bool(getattr(r, "is_primary", False)),
                        "is_random_baseline": bool(getattr(r, "is_random_baseline", False)),
                        "is_ablation": bool(getattr(r, "is_ablation", False)),
                        "is_rejected_word": bool(getattr(r, "is_rejected_word", False)),
                        "scientific_preference_rank": int(getattr(r, "scientific_preference_rank", 99)),
                    }
                )

                # Rescue gate evaluation
                t_true = bool(true_mode_ok and unfolded_ok and (not approx_detected))
                cov_ok = bool(math.isfinite(float(getattr(r, "coverage_score", float("nan")))) and float(getattr(r, "coverage_score")) >= float(args.min_coverage))
                rank_ok = bool(cand_rank != -1 and cand_rank <= int(args.max_rank_for_rescue))
                better_than_primary = bool(math.isfinite(prim_err) and math.isfinite(cand_err) and cand_err < prim_err)
                not_primary = bool(wg != primary_wg)
                less_poisson = True
                if math.isfinite(cand_poisson_like_frac) and math.isfinite(prim_poisson_like_frac):
                    less_poisson = bool(cand_poisson_like_frac < prim_poisson_like_frac)
                long_improve_ok = True
                if math.isfinite(long_impr):
                    long_improve_ok = bool(long_impr > 0.0)

                rescue_pass = bool(t_true and cov_ok and rank_ok and better_than_primary and not_primary and less_poisson and long_improve_ok)

                rescue_rows.append(
                    {
                        "dim": int(dim),
                        "V_mode": vm,
                        "word_group": wg,
                        "rescue_pass": bool(rescue_pass),
                        "true_mode_ok": bool(t_true),
                        "coverage_ok": bool(cov_ok),
                        "rank_ok": bool(rank_ok),
                        "better_than_primary": bool(better_than_primary),
                        "less_poisson_like_than_primary": bool(less_poisson),
                        "long_range_improvement_ok": bool(long_improve_ok),
                        "candidate_rank_by_error": cand_rank,
                        "coverage_score": float(getattr(r, "coverage_score", float("nan"))),
                        "candidate_error": float(cand_err),
                        "primary_error": float(prim_err),
                        "candidate_poisson_like_fraction": cand_poisson_like_frac,
                        "primary_poisson_like_fraction": prim_poisson_like_frac,
                        "long_range_improvement": long_impr,
                        "is_random_baseline": bool(getattr(r, "is_random_baseline", False)),
                        "is_ablation": bool(getattr(r, "is_ablation", False)),
                        "is_rejected_word": bool(getattr(r, "is_rejected_word", False)),
                    }
                )

    df_rank = pd.DataFrame(ranking_rows)
    if not df_rank.empty:
        df_rank = df_rank.sort_values(["dim", "candidate_rank_by_error", "scientific_preference_rank", "word_group"], ascending=True)

    df_rescue = pd.DataFrame(rescue_rows)
    if not df_rescue.empty:
        df_rescue = df_rescue.sort_values(["dim", "rescue_pass", "candidate_rank_by_error", "word_group"], ascending=[True, False, True, True])

    # Primary vs best (per dim): use df_best if available, else infer from df_rank
    pvb_rows: List[Dict[str, Any]] = []
    for dim in dims_keep:
        best_wg = ""
        best_vm = ""
        best_err = float("nan")
        if df_best is not None and not df_best.empty and "dim" in df_best.columns:
            db = df_best.copy()
            db["dim"] = pd.to_numeric(db["dim"], errors="coerce").astype("Int64")
            b = db[db["dim"].astype(int) == int(dim)]
            if not b.empty:
                best_vm = str(b["best_V_mode"].iloc[0])
                best_wg = str(b["best_word_group"].iloc[0])
                best_err = float(pd.to_numeric(b["best_total_curve_error"], errors="coerce").iloc[0])
        elif not df_rank.empty:
            b = df_rank[df_rank["dim"].astype(int) == int(dim)].sort_values(["median_total_curve_error"], ascending=True, na_position="last")
            if not b.empty:
                best_vm = str(b["V_mode"].iloc[0])
                best_wg = str(b["word_group"].iloc[0])
                best_err = float(b["median_total_curve_error"].iloc[0])

        prim = primary_by_dim.get(int(dim), {})
        pvb_rows.append(
            {
                "dim": int(dim),
                "primary_V_mode": primary_vm,
                "primary_word_group": primary_wg,
                "primary_total_curve_error": float(prim.get("primary_total_curve_error", float("nan"))),
                "primary_failure_region": str(prim.get("primary_failure_region", "")),
                "primary_poissonization_classification": str(prim.get("primary_poissonization_classification", "")),
                "primary_rank": int(prim.get("primary_rank", -1)),
                "best_V_mode": best_vm,
                "best_word_group": best_wg,
                "best_total_curve_error": float(best_err),
                "best_improvement_vs_primary": float(prim.get("primary_total_curve_error", float("nan")) - best_err)
                if (math.isfinite(float(prim.get("primary_total_curve_error", float("nan")))) and math.isfinite(best_err))
                else float("nan"),
            }
        )
    df_pvb = pd.DataFrame(pvb_rows)

    # Overall classification per spec
    rescued = df_rescue[df_rescue["rescue_pass"] == True] if not df_rescue.empty else pd.DataFrame()
    any_rescue = bool(not rescued.empty)
    any_non_primary = any_rescue and bool((rescued["word_group"].astype(str) != primary_wg).any())
    any_non_artifact = any_rescue and bool((rescued["is_random_baseline"].astype(bool) == False).any())
    rescued_dims = sorted(set(rescued["dim"].astype(int).tolist())) if any_rescue else []

    if not any_rescue:
        overall_cls = "NO_RESCUE"
    else:
        # determine best rescue type
        non_art = rescued[rescued["is_random_baseline"].astype(bool) == False]
        if non_art.empty:
            overall_cls = "BASELINE_ONLY_RESCUE"
        elif (non_art["is_ablation"].astype(bool).any()) and not (non_art["is_rejected_word"].astype(bool).any()):
            overall_cls = "ABLATION_RESCUE"
        elif non_art["is_rejected_word"].astype(bool).any():
            overall_cls = "REJECTED_WORD_RESCUE"
        else:
            overall_cls = "STRONG_RESCUE" if len(set(non_art["dim"].astype(int).tolist())) >= 2 else "READY_FOR_V13P0_CANDIDATE"

    ready_for_v13p0 = bool(any_non_artifact and true_mode_ok and unfolded_ok)

    # Write outputs
    (out_dir / "v13o10_candidate_ranking.csv").write_text(df_rank.to_csv(index=False), encoding="utf-8")
    (out_dir / "v13o10_primary_vs_best.csv").write_text(df_pvb.to_csv(index=False), encoding="utf-8")
    (out_dir / "v13o10_rescue_gate_summary.csv").write_text(df_rescue.to_csv(index=False), encoding="utf-8")
    (out_dir / "v13o10_coverage_summary.csv").write_text(df_cov.to_csv(index=False), encoding="utf-8")
    (out_dir / "v13o10_artifact_flags.csv").write_text(df_flags.to_csv(index=False), encoding="utf-8")

    payload: Dict[str, Any] = {
        "warning": "Computational diagnostic only; not a proof of the Riemann Hypothesis.",
        "status": "V13O.10 true spectra candidate rescue",
        "inputs": {
            "v13o9_dir": str(v13o9_dir.resolve()) if v13o9_dir is not None else "",
            "v13o9_gate_summary": str(gate_p.resolve()) if str(gate_p) else "",
            "v13o9_nv_curve_errors": str(err_p.resolve()) if str(err_p) else "",
            "v13o9_poissonization": str(pois_p.resolve()) if str(pois_p) else "",
            "v13o9_best_by_dim": str(best_p.resolve()) if str(best_p) else "",
            "v13o9_unfolded_levels": str(lev_p.resolve()) if str(lev_p) else "",
        },
        "true_mode_validation": {
            "true_mode_ok": bool(true_mode_ok),
            "unfolded_ok": bool(unfolded_ok),
            "unfolded_rows": int(unfolded_rows),
            "approx_detected": bool(approx_detected),
        },
        "primary": {"word_group": primary_wg, "V_mode": primary_vm, "dims": dims_keep},
        "rescue": {
            "min_coverage": float(args.min_coverage),
            "max_rank_for_rescue": int(args.max_rank_for_rescue),
            "any_rescue": bool(any_rescue),
            "any_non_primary_rescue": bool(any_non_primary),
            "any_non_artifact_rescue": bool(any_non_artifact),
            "rescued_dims": rescued_dims,
            "classification": overall_cls,
            "ready_for_v13p0_candidate": bool(ready_for_v13p0),
        },
        "warnings": warnings,
        "runtime_s": float(time.perf_counter() - t0),
        "python_pid": os.getpid(),
    }
    (out_dir / "v13o10_results.json").write_text(json.dumps(json_sanitize(payload), indent=2, allow_nan=False), encoding="utf-8")

    # Report
    md: List[str] = []
    md.append("# V13O.10 True spectra candidate rescue\n\n")
    md.append("> **Computational evidence only; not a proof of RH.**\n\n")
    md.append("## 1. Motivation\n\n")
    md.append("V13O.9 ran on true unfolded spectra (from V13O.8 export) and the primary operator failed. V13O.10 ranks and attempts to rescue alternative candidates.\n\n")
    md.append("## 2. Inputs and true-mode validation\n\n")
    md.append(f"- true_mode_ok: **{bool(true_mode_ok)}**\n")
    md.append(f"- unfolded_ok: **{bool(unfolded_ok)}** (rows={unfolded_rows})\n")
    md.append(f"- approx_detected: **{bool(approx_detected)}**\n\n")
    if warnings:
        md.append("Warnings (summarized):\n\n")
        # keep short: do not dump thousands
        md.append(f"- n_warnings={len(warnings)}\n")
        md.append(f"- first_warning={warnings[0]}\n\n")
    md.append("## 3. Primary failure recap from V13O.9\n\n")
    md.append("See `v13o10_primary_vs_best.csv` and `v13o9_gate_summary.csv`.\n\n")
    md.append("## 4. Candidate ranking by true NV-curve error\n\n")
    md.append("See `v13o10_candidate_ranking.csv`.\n\n")
    md.append("## 5. Rescue gate\n\n")
    md.append("See `v13o10_rescue_gate_summary.csv`.\n\n")
    md.append("## 6. Artifact / baseline caution\n\n")
    md.append("See `v13o10_artifact_flags.csv`. Baselines are included but flagged.\n\n")
    md.append("## 7. Best candidate per dimension\n\n")
    md.append("See `v13o10_primary_vs_best.csv` and `v13o10_candidate_ranking.csv`.\n\n")
    md.append("## 8. Decision about V13P0\n\n")
    md.append("Proceed to V13P0 only if at least one **non-artifact** candidate passes rescue with stable coverage and improves long-range behavior.\n\n")
    md.append("### Explicit answers\n\n")
    md.append(f"- Did V13O.10 run on true unfolded spectra? **{bool(true_mode_ok and unfolded_ok)}**\n")
    md.append(f"- Is primary rescued? **{False}**\n")
    md.append(f"- Is any non-primary candidate rescued? **{bool(any_non_primary)}**\n")
    md.append("- Which candidate is best per dimension? See `v13o10_primary_vs_best.csv`.\n")
    md.append("- Which candidate is most scientifically interesting? Prefer `rejected_word` > `ablate_*` > `random_*`.\n")
    md.append("- Are best candidates potentially artifacts? See artifact flags + rescue gate.\n")
    md.append(f"- Should proceed to V13P0? **{bool(ready_for_v13p0 and any_non_artifact)}**\n\n")
    md.append("## Outputs\n\n")
    md.append("- `v13o10_candidate_ranking.csv`\n")
    md.append("- `v13o10_primary_vs_best.csv`\n")
    md.append("- `v13o10_rescue_gate_summary.csv`\n")
    md.append("- `v13o10_coverage_summary.csv`\n")
    md.append("- `v13o10_artifact_flags.csv`\n")
    md.append("- `v13o10_results.json`\n")
    md.append("- `v13o10_report.md`\n")
    md.append("- `v13o10_report.tex`\n")
    md.append("- `v13o10_report.pdf` (if `pdflatex` exists)\n\n")
    (out_dir / "v13o10_report.md").write_text("".join(md), encoding="utf-8")

    tex = (
        "\\documentclass[11pt]{article}\n\\usepackage[margin=1in]{geometry}\n\\begin{document}\n"
        "\\title{V13O.10 True spectra candidate rescue}\n\\maketitle\n"
        "\\section*{Warning}\nComputational evidence only; not a proof of RH.\n\n"
        "\\section*{Validation}\n"
        + latex_escape(json.dumps(payload.get("true_mode_validation", {}), indent=2))
        + "\n\n\\section*{Rescue classification}\n"
        + latex_escape(str(payload.get("rescue", {})))
        + "\n\n\\end{document}\n"
    )
    (out_dir / "v13o10_report.tex").write_text(tex, encoding="utf-8")

    if _find_pdflatex() is None:
        print("[v13o10] WARNING: pdflatex not found; skipping PDF.", flush=True)
    elif try_pdflatex(out_dir / "v13o10_report.tex", out_dir, "v13o10_report.pdf"):
        print(f"Wrote {out_dir / 'v13o10_report.pdf'}", flush=True)
    else:
        print("[v13o10] WARNING: pdflatex failed or did not produce v13o10_report.pdf.", flush=True)

    print(f"[v13o10] Wrote {out_dir / 'v13o10_results.json'}", flush=True)


if __name__ == "__main__":
    main()

