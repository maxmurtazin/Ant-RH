#!/usr/bin/env python3
"""
V13O.5 — Robust unfolded zeta objective with MAD-normalized long-range gates.

Post-processing layer over V13O.4 outputs (no operator recomputation).

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


def finite_series(x: "pd.Series") -> "pd.Series":
    return pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan)


def safe_median(xs: Sequence[float]) -> float:
    arr = np.asarray([float(v) for v in xs if v is not None and math.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.median(arr))


def safe_mean(xs: Sequence[float]) -> float:
    arr = np.asarray([float(v) for v in xs if v is not None and math.isfinite(float(v))], dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def mad_scale(x: np.ndarray, *, eps: float) -> float:
    """MAD with fallback to IQR/1.349 then std."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float(eps)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med)))
    if math.isfinite(mad) and mad > eps:
        return mad
    q25 = float(np.percentile(x, 25))
    q75 = float(np.percentile(x, 75))
    iqr = float(q75 - q25)
    scale_iqr = float(iqr / 1.349) if math.isfinite(iqr) else float("nan")
    std = float(np.std(x))
    cand = [scale_iqr, std, eps]
    cand2 = [c for c in cand if math.isfinite(float(c)) and float(c) > 0.0]
    return float(max(cand2)) if cand2 else float(eps)


def robust_z_lower_is_better(x_real: float, controls: Sequence[float], *, eps: float) -> float:
    """(x_real - median(controls)) / (MAD(controls)+eps)"""
    c = np.asarray([float(v) for v in controls if v is not None and math.isfinite(float(v))], dtype=np.float64)
    if c.size == 0 or not math.isfinite(float(x_real)):
        return float("nan")
    med = float(np.median(c))
    scale = mad_scale(c, eps=float(eps))
    return float((float(x_real) - med) / max(scale, float(eps)))


def read_csv_robust(path: Path, *, name: str) -> Optional["pd.DataFrame"]:
    if not path.is_file():
        print(f"[v13o5] WARNING missing input: {name}={path}", flush=True)
        return None
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[v13o5] WARNING failed reading {name}={path}: {e!r}", flush=True)
        return None
    # normalize columns
    df.columns = [str(c).strip() for c in df.columns]
    return df


def pick_col(df: "pd.DataFrame", candidates: Sequence[str]) -> Optional[str]:
    cols = {str(c).strip(): c for c in df.columns}
    for cand in candidates:
        if cand in cols:
            return str(cols[cand])
    # fallback: case-insensitive contains
    low = {str(c).strip().lower(): c for c in df.columns}
    for cand in candidates:
        k = cand.lower()
        if k in low:
            return str(low[k])
    return None


def group_keys(df: "pd.DataFrame") -> List[str]:
    ks = []
    for k in ("dim", "V_mode", "word_group"):
        if k in df.columns:
            ks.append(k)
    return ks


def compute_component_from_target_split(
    df: "pd.DataFrame",
    *,
    value_col: str,
    real_target_names: Sequence[str],
    group: Tuple[int, str, str],
    transform: Optional[str],
) -> Tuple[float, float, float, List[str]]:
    """
    For a given (dim,V_mode,word_group), compute:
    - real_median (from rows where target_group in real_target_names)
    - control_median (from rows where target_group not real)
    - robust z (lower is better)
    Return (z, real_med, control_med, missing_notes)
    """
    dim, vm, wg = group
    missing: List[str] = []
    if df is None or value_col not in df.columns:
        return float("nan"), float("nan"), float("nan"), ["missing_df_or_col"]
    if "target_group" not in df.columns:
        return float("nan"), float("nan"), float("nan"), ["missing_target_group"]

    sub = df[(df["dim"] == dim) & (df["V_mode"] == vm) & (df["word_group"] == wg)]
    if sub.empty:
        return float("nan"), float("nan"), float("nan"), ["missing_group_rows"]

    x = finite_series(sub[value_col])
    if transform == "log1p":
        x = np.log1p(x)
    # attach
    sub2 = sub.copy()
    sub2["_x"] = x

    real_mask = sub2["target_group"].astype(str).isin([str(t) for t in real_target_names])
    real_vals = sub2.loc[real_mask, "_x"].dropna().astype(float).tolist()
    ctrl_vals = sub2.loc[~real_mask, "_x"].dropna().astype(float).tolist()
    if not real_vals:
        missing.append("no_real_vals")
    if not ctrl_vals:
        missing.append("no_control_vals")
    real_med = safe_median(real_vals)
    ctrl_med = safe_median(ctrl_vals)
    return float("nan"), float(real_med), float(ctrl_med), missing  # z computed by caller with control list


def main() -> None:
    ap = argparse.ArgumentParser(description="V13O.5 robust re-scoring (computational evidence only).")
    ap.add_argument("--v13o4_summary", type=str, default="runs/v13o4_zeta_specific_objective/v13o4_summary.csv")
    ap.add_argument("--v13o4_group_summary", type=str, default="runs/v13o4_zeta_specific_objective/v13o4_group_summary.csv")
    ap.add_argument("--v13o4_zeta_scores", type=str, default="runs/v13o4_zeta_specific_objective/v13o4_zeta_specific_scores.csv")
    ap.add_argument("--v13o4_pair_corr", type=str, default="runs/v13o4_zeta_specific_objective/v13o4_pair_corr_summary.csv")
    ap.add_argument("--v13o4_number_variance", type=str, default="runs/v13o4_zeta_specific_objective/v13o4_number_variance_summary.csv")
    ap.add_argument("--v13o4_staircase", type=str, default="runs/v13o4_zeta_specific_objective/v13o4_staircase_summary.csv")
    ap.add_argument("--v13o4_ensemble_margins", type=str, default="runs/v13o4_zeta_specific_objective/v13o4_ensemble_margins.csv")
    ap.add_argument("--v13o4_gate_summary", type=str, default="runs/v13o4_zeta_specific_objective/v13o4_gate_summary.csv")
    ap.add_argument("--out_dir", type=str, default="runs/v13o5_robust_unfolded_normalization")
    ap.add_argument("--primary_word_group", type=str, default="primary_word_seed6")
    ap.add_argument("--primary_v_mode", type=str, default="full_V")
    ap.add_argument("--margin_z", type=float, default=0.25)
    ap.add_argument("--mad_eps", type=float, default=1e-9)
    ap.add_argument("--progress_every", type=int, default=1)

    ap.add_argument("--w_local", type=float, default=1.0)
    ap.add_argument("--w_pair", type=float, default=1.0)
    ap.add_argument("--w_number_variance", type=float, default=0.5)
    ap.add_argument("--w_staircase", type=float, default=1.0)
    ap.add_argument("--w_transfer", type=float, default=1.0)
    ap.add_argument("--w_ensemble", type=float, default=1.0)

    args = ap.parse_args()

    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    prog = max(1, int(args.progress_every))
    mad_eps = float(args.mad_eps)
    margin_z = float(args.margin_z)

    # Load CSVs (missing files should not crash)
    df_scores = read_csv_robust(_resolve(args.v13o4_zeta_scores), name="v13o4_zeta_scores")
    df_pair = read_csv_robust(_resolve(args.v13o4_pair_corr), name="v13o4_pair_corr")
    df_nv = read_csv_robust(_resolve(args.v13o4_number_variance), name="v13o4_number_variance")
    df_st = read_csv_robust(_resolve(args.v13o4_staircase), name="v13o4_staircase")
    df_em = read_csv_robust(_resolve(args.v13o4_ensemble_margins), name="v13o4_ensemble_margins")
    df_gate = read_csv_robust(_resolve(args.v13o4_gate_summary), name="v13o4_gate_summary")
    _ = read_csv_robust(_resolve(args.v13o4_summary), name="v13o4_summary")  # keep for debugging presence
    _ = read_csv_robust(_resolve(args.v13o4_group_summary), name="v13o4_group_summary")

    # Normalize key columns types
    for dfx in (df_scores, df_pair, df_nv, df_st, df_em, df_gate):
        if dfx is None:
            continue
        if "dim" in dfx.columns:
            dfx["dim"] = pd.to_numeric(dfx["dim"], errors="coerce").astype("Int64")
        for k in ("V_mode", "word_group", "target_group"):
            if k in dfx.columns:
                dfx[k] = dfx[k].astype(str).str.strip()

    # Determine groups to score: union of (dim,V_mode,word_group) seen anywhere
    groups_set = set()
    for dfx in (df_scores, df_pair, df_nv, df_st, df_em):
        if dfx is None:
            continue
        if not all(k in dfx.columns for k in ("dim", "V_mode", "word_group")):
            continue
        for row in dfx[["dim", "V_mode", "word_group"]].dropna().itertuples(index=False):
            groups_set.add((int(row[0]), str(row[1]), str(row[2])))
    groups = sorted(groups_set, key=lambda x: (int(x[0]), str(x[1]), str(x[2])))

    print(f"[v13o5] groups to score: {len(groups)}", flush=True)

    # Column picks (defensive)
    col_pair_test = pick_col(df_pair, ["pair_corr_error_test", "pair_corr_test", "pair_corr_error"]) if df_pair is not None else None
    if col_pair_test is None and df_pair is not None and "pair_corr_error_test" in df_pair.columns:
        col_pair_test = "pair_corr_error_test"
    col_nv_test = pick_col(df_nv, ["number_variance_error_test", "number_variance_test", "number_variance_error"]) if df_nv is not None else None
    col_st_test = pick_col(df_st, ["staircase_residual_error_test", "staircase_test", "staircase_error"]) if df_st is not None else None

    # local terms from scores
    col_sl = pick_col(df_scores, ["spectral_log_mse_test", "spectral_log_mse"]) if df_scores is not None else None
    col_sp = pick_col(df_scores, ["spacing_mse_normalized_test", "spacing_mse_normalized"]) if df_scores is not None else None
    col_ks = pick_col(df_scores, ["ks_wigner_test", "ks_wigner"]) if df_scores is not None else None
    col_j_base = pick_col(df_scores, ["J_zeta_base", "J_zeta_without_ensemble", "J_base"]) if df_scores is not None else None
    col_transfer_pen = pick_col(df_scores, ["transfer_gap_penalty"]) if df_scores is not None else None

    col_em_sum = pick_col(df_em, ["ensemble_margin_penalty_sum", "ensemble_margin_penalty", "ensemble_penalty_sum"]) if df_em is not None else None
    if df_em is not None and col_em_sum is None and "ensemble_margin_penalty_sum" in df_em.columns:
        col_em_sum = "ensemble_margin_penalty_sum"

    # Gate cols
    g1_col = pick_col(df_gate, ["G1_finite_self_adjoint"]) if df_gate is not None else None
    g2_col = pick_col(df_gate, ["G2_operator_diff_lt_1e-3"]) if df_gate is not None else None
    g9_col = pick_col(df_gate, ["G9_transfer_gap_acceptable"]) if df_gate is not None else None
    g10_col = pick_col(df_gate, ["G10_accepted_real_zeta"]) if df_gate is not None else None
    gate_dim_col = "dim" if (df_gate is not None and "dim" in df_gate.columns) else None

    # Build per-group robust z components
    rows_scores: List[Dict[str, Any]] = []

    w = {
        "local": float(args.w_local),
        "pair": float(args.w_pair),
        "nv": float(args.w_number_variance),
        "stair": float(args.w_staircase),
        "transfer": float(args.w_transfer),
        "ensemble": float(args.w_ensemble),
    }

    def log_progress(i: int, total: int, group: Tuple[int, str, str]) -> None:
        if i == 1 or i == total or i % prog == 0:
            elapsed = time.perf_counter() - t0
            avg = elapsed / max(i, 1)
            eta = avg * max(total - i, 0)
            d, vm, wg = group
            print(
                f"[V13O.5] {i}/{total} elapsed={elapsed:.1f}s eta={format_seconds(eta)} current=({d},{vm},{wg})",
                flush=True,
            )

    for i, grp in enumerate(groups, start=1):
        log_progress(i, len(groups), grp)
        dim, vm, wg = grp
        missing_components: List[str] = []

        # --- z_pair_corr ---
        z_pair = float("nan")
        raw_pair_real = float("nan")
        raw_pair_ctrl = float("nan")
        if df_pair is not None and col_pair_test is not None:
            sub = df_pair[(df_pair["dim"] == dim) & (df_pair["V_mode"] == vm) & (df_pair["word_group"] == wg)]
            if not sub.empty and "target_group" in sub.columns:
                x = finite_series(sub[col_pair_test])
                sub2 = sub.copy()
                sub2["_x"] = x
                real_vals = sub2.loc[sub2["target_group"] == "real_zeta", "_x"].dropna().astype(float).tolist()
                ctrl_vals = sub2.loc[sub2["target_group"] != "real_zeta", "_x"].dropna().astype(float).tolist()
                raw_pair_real = safe_median(real_vals)
                raw_pair_ctrl = safe_median(ctrl_vals)
                z_pair = robust_z_lower_is_better(raw_pair_real, ctrl_vals, eps=mad_eps)
            else:
                missing_components.append("pair_corr")
        else:
            missing_components.append("pair_corr")

        # --- z_number_variance (log1p) ---
        z_nv = float("nan")
        raw_nv_real = float("nan")
        raw_nv_ctrl = float("nan")
        if df_nv is not None and col_nv_test is not None:
            sub = df_nv[(df_nv["dim"] == dim) & (df_nv["V_mode"] == vm) & (df_nv["word_group"] == wg)]
            if not sub.empty and "target_group" in sub.columns:
                x_raw = finite_series(sub[col_nv_test])
                x = np.log1p(x_raw)
                sub2 = sub.copy()
                sub2["_x"] = x
                real_vals_raw = sub2.loc[sub2["target_group"] == "real_zeta", col_nv_test].pipe(finite_series).dropna().astype(float).tolist()
                ctrl_vals_raw = sub2.loc[sub2["target_group"] != "real_zeta", col_nv_test].pipe(finite_series).dropna().astype(float).tolist()
                real_vals = sub2.loc[sub2["target_group"] == "real_zeta", "_x"].dropna().astype(float).tolist()
                ctrl_vals = sub2.loc[sub2["target_group"] != "real_zeta", "_x"].dropna().astype(float).tolist()
                raw_nv_real = safe_median(real_vals_raw)
                raw_nv_ctrl = safe_median(ctrl_vals_raw)
                z_nv = robust_z_lower_is_better(safe_median(real_vals), ctrl_vals, eps=mad_eps)
            else:
                missing_components.append("number_variance")
        else:
            missing_components.append("number_variance")

        # --- z_staircase (log1p) ---
        z_stair = float("nan")
        raw_st_real = float("nan")
        raw_st_ctrl = float("nan")
        if df_st is not None and col_st_test is not None:
            sub = df_st[(df_st["dim"] == dim) & (df_st["V_mode"] == vm) & (df_st["word_group"] == wg)]
            if not sub.empty and "target_group" in sub.columns:
                x_raw = finite_series(sub[col_st_test])
                x = np.log1p(x_raw)
                sub2 = sub.copy()
                sub2["_x"] = x
                real_vals_raw = sub2.loc[sub2["target_group"] == "real_zeta", col_st_test].pipe(finite_series).dropna().astype(float).tolist()
                ctrl_vals_raw = sub2.loc[sub2["target_group"] != "real_zeta", col_st_test].pipe(finite_series).dropna().astype(float).tolist()
                real_vals = sub2.loc[sub2["target_group"] == "real_zeta", "_x"].dropna().astype(float).tolist()
                ctrl_vals = sub2.loc[sub2["target_group"] != "real_zeta", "_x"].dropna().astype(float).tolist()
                raw_st_real = safe_median(real_vals_raw)
                raw_st_ctrl = safe_median(ctrl_vals_raw)
                z_stair = robust_z_lower_is_better(safe_median(real_vals), ctrl_vals, eps=mad_eps)
            else:
                missing_components.append("staircase")
        else:
            missing_components.append("staircase")

        # --- z_ensemble from ensemble_margin_penalty_sum ---
        z_ens = float("nan")
        raw_ens = float("nan")
        if df_em is not None and col_em_sum is not None:
            sub = df_em[(df_em["dim"] == dim) & (df_em["V_mode"] == vm)]
            if not sub.empty and "word_group" in sub.columns:
                own = sub[sub["word_group"] == wg]
                raw_ens = safe_median(finite_series(own[col_em_sum]).dropna().astype(float).tolist())
                ctrl = sub[sub["word_group"] != wg]
                ctrl_vals = finite_series(ctrl[col_em_sum]).dropna().astype(float).tolist()
                z_ens = robust_z_lower_is_better(raw_ens, ctrl_vals, eps=mad_eps)
            else:
                missing_components.append("ensemble")
        else:
            missing_components.append("ensemble")

        # --- z_transfer: abs gap_spectral_real (primary/full_V only if available) ---
        z_transfer = float("nan")
        if df_gate is not None and gate_dim_col is not None:
            # Only primary/full_V available reliably; else NaN.
            if wg == str(args.primary_word_group) and vm == str(args.primary_v_mode):
                gsub = df_gate[df_gate[gate_dim_col] == dim]
                col_gap = pick_col(df_gate, ["gap_spectral_real", "gap_spectral"])
                if col_gap is not None and not gsub.empty:
                    x_real = float(finite_series(gsub[col_gap]).dropna().astype(float).median())
                    x_real = abs(x_real) if math.isfinite(x_real) else float("nan")
                    # control distribution: gaps across all dims (same dim) if present
                    ctrl_vals = finite_series(df_gate[df_gate[gate_dim_col] == dim][col_gap]).dropna().astype(float).abs().tolist()
                    z_transfer = robust_z_lower_is_better(x_real, ctrl_vals, eps=mad_eps)
            else:
                missing_components.append("transfer")
        else:
            missing_components.append("transfer")

        # --- z_local: robust score using local terms if present, else proxy from J_zeta_base ---
        z_local = float("nan")
        if df_scores is not None and "target_group" in df_scores.columns:
            sub = df_scores[(df_scores["dim"] == dim) & (df_scores["V_mode"] == vm) & (df_scores["word_group"] == wg)]
            if not sub.empty:
                # Prefer composite local proxy = spectral_log_mse + spacing + ks (all lower-better)
                if col_sl and col_sp and col_ks:
                    sl = finite_series(sub[col_sl])
                    sp = finite_series(sub[col_sp])
                    ks = finite_series(sub[col_ks])
                    loc = sl + sp + ks
                    sub2 = sub.copy()
                    sub2["_loc"] = loc
                    real_vals = sub2.loc[sub2["target_group"] == "real_zeta", "_loc"].dropna().astype(float).tolist()
                    ctrl_vals = sub2.loc[sub2["target_group"] != "real_zeta", "_loc"].dropna().astype(float).tolist()
                    z_local = robust_z_lower_is_better(safe_median(real_vals), ctrl_vals, eps=mad_eps)
                elif col_j_base:
                    jb = finite_series(sub[col_j_base])
                    sub2 = sub.copy()
                    sub2["_loc"] = jb
                    real_vals = sub2.loc[sub2["target_group"] == "real_zeta", "_loc"].dropna().astype(float).tolist()
                    ctrl_vals = sub2.loc[sub2["target_group"] != "real_zeta", "_loc"].dropna().astype(float).tolist()
                    z_local = robust_z_lower_is_better(safe_median(real_vals), ctrl_vals, eps=mad_eps)
                else:
                    missing_components.append("local")
            else:
                missing_components.append("local")
        else:
            missing_components.append("local")

        # --- Weighted robust objective (do not include NaNs) ---
        comps = [
            ("z_local", z_local, w["local"]),
            ("z_pair_corr", z_pair, w["pair"]),
            ("z_number_variance", z_nv, w["nv"]),
            ("z_staircase", z_stair, w["stair"]),
            ("z_transfer", z_transfer, w["transfer"]),
            ("z_ensemble", z_ens, w["ensemble"]),
        ]
        used = [(name, val, wt) for (name, val, wt) in comps if math.isfinite(float(val)) and math.isfinite(float(wt)) and wt != 0.0]
        if used:
            num = float(sum(float(wt) * float(val) for _, val, wt in used))
            den = float(sum(abs(float(wt)) for _, _, wt in used))
            J_robust = num / max(den, 1e-12)
        else:
            J_robust = float("nan")

        missing_set = sorted({m for m in missing_components})
        rows_scores.append(
            {
                "dim": dim,
                "V_mode": vm,
                "word_group": wg,
                "J_robust": J_robust,
                "n_components_used": len(used),
                "missing_components": ";".join(missing_set),
                "z_local": z_local,
                "z_pair_corr": z_pair,
                "z_number_variance": z_nv,
                "z_staircase": z_stair,
                "z_transfer": z_transfer,
                "z_ensemble": z_ens,
                "raw_pair_corr_real_median": raw_pair_real,
                "raw_pair_corr_control_median": raw_pair_ctrl,
                "raw_number_variance_real_median": raw_nv_real,
                "raw_number_variance_control_median": raw_nv_ctrl,
                "raw_staircase_real_median": raw_st_real,
                "raw_staircase_control_median": raw_st_ctrl,
                "raw_ensemble_penalty": raw_ens,
            }
        )

    df_out = pd.DataFrame(rows_scores)
    df_out = df_out.sort_values(["dim", "V_mode", "word_group"], kind="mergesort")
    (out_dir / "v13o5_robust_scores.csv").write_text(df_out.to_csv(index=False), encoding="utf-8")

    # Gate summary for primary
    gate_rows: List[Dict[str, Any]] = []
    dims = sorted({int(d) for d in df_out["dim"].dropna().astype(int).tolist()})
    for dim in dims:
        row = df_out[(df_out["dim"] == dim) & (df_out["V_mode"] == args.primary_v_mode) & (df_out["word_group"] == args.primary_word_group)]
        if row.empty:
            continue
        r0 = row.iloc[0].to_dict()
        z_pair = float(r0.get("z_pair_corr", float("nan")))
        z_nv = float(r0.get("z_number_variance", float("nan")))
        z_st = float(r0.get("z_staircase", float("nan")))
        z_tr = float(r0.get("z_transfer", float("nan")))
        z_en = float(r0.get("z_ensemble", float("nan")))
        J = float(r0.get("J_robust", float("nan")))

        # Pull V13O.4 basic gates if available
        r1 = False
        r6 = False
        r7 = False
        if df_gate is not None and gate_dim_col is not None:
            gsub = df_gate[df_gate[gate_dim_col] == dim]
            if not gsub.empty:
                def _bool_col(col: Optional[str]) -> bool:
                    if col is None or col not in gsub.columns:
                        return False
                    v = gsub[col].iloc[0]
                    if isinstance(v, (bool, np.bool_)):
                        return bool(v)
                    try:
                        s = str(v).strip().lower()
                        return s in ("true", "1", "yes")
                    except Exception:
                        return False
                g1 = _bool_col(g1_col)
                g2 = _bool_col(g2_col)
                r1 = bool(g1 and g2) if (g1_col and g2_col) else bool(g1 or g2)
                r6 = _bool_col(g9_col) if g9_col else (math.isfinite(z_tr) and z_tr <= 0.0)
                r7 = _bool_col(g10_col) if g10_col else True
            else:
                r1 = False
                r6 = math.isfinite(z_tr) and z_tr <= 0.0
                r7 = True
        else:
            r1 = False
            r6 = math.isfinite(z_tr) and z_tr <= 0.0
            r7 = True

        r2 = math.isfinite(z_pair) and z_pair <= -margin_z
        r3 = math.isfinite(z_nv) and z_nv <= -margin_z
        r3r = math.isfinite(z_nv) and z_nv <= 0.0
        r4 = math.isfinite(z_st) and z_st <= -margin_z
        r5 = math.isfinite(z_en) and z_en <= -margin_z

        strict = bool(r1 and r2 and r3 and r4 and r5 and r6 and r7)
        relaxed = bool(r1 and r2 and r3r and r4 and r5 and r6 and r7)

        gate_rows.append(
            {
                "dim": dim,
                "V_mode": args.primary_v_mode,
                "word_group": args.primary_word_group,
                "R1_finite_self_adjoint": r1,
                "R2_pair_corr_robust": r2,
                "R3_number_variance_robust": r3,
                "R3_relaxed_number_variance": r3r,
                "R4_staircase_robust": r4,
                "R5_ensemble_robust": r5,
                "R6_transfer_gap": r6,
                "R7_real_accepted": r7,
                "strict_robust_zeta_specificity_pass": strict,
                "relaxed_robust_zeta_specificity_pass": relaxed,
                "J_robust": J,
                "z_pair_corr": z_pair,
                "z_number_variance": z_nv,
                "z_staircase": z_st,
                "z_transfer": z_tr,
                "z_ensemble": z_en,
            }
        )

    df_gate_out = pd.DataFrame(gate_rows)
    if not df_gate_out.empty:
        df_gate_out = df_gate_out.sort_values(["dim"], kind="mergesort")
        (out_dir / "v13o5_gate_summary.csv").write_text(df_gate_out.to_csv(index=False), encoding="utf-8")
    else:
        (out_dir / "v13o5_gate_summary.csv").write_text("", encoding="utf-8")

    # Component summary: mean/median z per group
    comp_cols = ["z_local", "z_pair_corr", "z_number_variance", "z_staircase", "z_transfer", "z_ensemble", "J_robust"]
    df_comp = (
        df_out.groupby(["dim", "V_mode", "word_group"], as_index=False)[comp_cols]
        .agg(["median", "mean"])
    )
    df_comp.columns = ["_".join([c for c in col if c]) for col in df_comp.columns.to_flat_index()]  # type: ignore[attr-defined]
    (out_dir / "v13o5_component_summary.csv").write_text(df_comp.to_csv(index=False), encoding="utf-8")

    # Best by dim and primary rank
    best_rows: List[Dict[str, Any]] = []
    for dim in dims:
        sub = df_out[df_out["dim"] == dim].copy()
        sub = sub[np.isfinite(pd.to_numeric(sub["J_robust"], errors="coerce"))]
        sub = sub.sort_values(["J_robust"], kind="mergesort")
        best = sub.iloc[0].to_dict() if not sub.empty else {"dim": dim, "note": "no_finite_J_robust"}

        prim = df_out[
            (df_out["dim"] == dim) & (df_out["V_mode"] == args.primary_v_mode) & (df_out["word_group"] == args.primary_word_group)
        ]
        prim_row = prim.iloc[0].to_dict() if not prim.empty else {}

        rank = None
        if not sub.empty and not prim.empty and math.isfinite(float(prim_row.get("J_robust", float("nan")))):
            # 1-based rank among finite rows
            vals = sub["J_robust"].astype(float).tolist()
            jv = float(prim_row["J_robust"])
            rank = 1 + sum(1 for x in vals if math.isfinite(float(x)) and float(x) < jv)

        best_rows.append(
            {
                "dim": dim,
                "best_dim": best.get("dim"),
                "best_V_mode": best.get("V_mode"),
                "best_word_group": best.get("word_group"),
                "best_J_robust": best.get("J_robust"),
                "primary_V_mode": args.primary_v_mode,
                "primary_word_group": args.primary_word_group,
                "primary_J_robust": prim_row.get("J_robust"),
                "primary_rank_among_all": rank,
            }
        )
    pd.DataFrame(best_rows).to_csv(out_dir / "v13o5_best_by_dim.csv", index=False)

    # Classification across dims
    strict_all = bool(gate_rows) and all(bool(r.get("strict_robust_zeta_specificity_pass")) for r in gate_rows)
    relaxed_ct = sum(1 for r in gate_rows if bool(r.get("relaxed_robust_zeta_specificity_pass")))
    pair_ct = sum(1 for r in gate_rows if bool(r.get("R2_pair_corr_robust")))
    stair_ct = sum(1 for r in gate_rows if bool(r.get("R4_staircase_robust")))

    if strict_all and len(gate_rows) >= 1:
        classification = "ROBUST_ZETA_SPECIFICITY"
    elif relaxed_ct >= 2 and pair_ct >= 2 and stair_ct >= 2:
        classification = "PARTIAL_ROBUST_SPECIFICITY"
    elif stair_ct >= 2:
        classification = "STRUCTURAL_STAIRCASE_SIGNAL_ONLY"
    else:
        classification = "NO_ROBUST_ZETA_SPECIFICITY"

    # Report + results.json
    md = [
        "# V13O.5 Robust unfolded normalization\n\n",
        "> **Computational evidence only; not a proof of the Riemann Hypothesis.**\n\n",
        "## V13O.4 diagnosis\n\n",
        "V13O.4 introduced zeta-specific metrics but compared **raw scales**; some terms (notably number variance and ensemble penalties) ",
        "could dominate due to heavy tails and unstable long-range sensitivity.\n\n",
        "## Robust normalization\n\n",
        "For each component (lower is better), compute a robust z-score vs matched controls:\n\n",
        "\\[\n",
        "z = \\frac{x_{\\mathrm{real}} - \\mathrm{median}(x_{\\mathrm{controls}})}{\\mathrm{MAD}(x_{\\mathrm{controls}}) + \\varepsilon}\n",
        "\\]\n\n",
        "where MAD is the median absolute deviation, with fallback scale when MAD is too small.\n\n",
        "## Robust objective\n\n",
        "Weighted mean over available component z-scores (NaNs excluded):\n\n",
        f"- w_local={args.w_local}\n",
        f"- w_pair={args.w_pair}\n",
        f"- w_number_variance={args.w_number_variance}\n",
        f"- w_staircase={args.w_staircase}\n",
        f"- w_transfer={args.w_transfer}\n",
        f"- w_ensemble={args.w_ensemble}\n\n",
        "## Gate summary (primary)\n\n",
        f"Primary: `{args.primary_word_group}` with `{args.primary_v_mode}`.\n\n",
        f"margin_z={margin_z}\n\n",
        "See `v13o5_gate_summary.csv`.\n\n",
        "## Classification\n\n",
        f"**{classification}**\n\n",
        "## Interpretation\n\n",
        "V13O.5 is a normalized diagnostic layer before V13P0 analytic renormalization. ",
        "It does not prove RH and should be interpreted as computational evidence only.\n\n",
        "## Outputs\n\n",
        "- `v13o5_robust_scores.csv`\n",
        "- `v13o5_gate_summary.csv`\n",
        "- `v13o5_component_summary.csv`\n",
        "- `v13o5_best_by_dim.csv`\n",
        "- `v13o5_results.json`\n",
        "- `v13o5_report.md` / `.tex` / `.pdf` (if pdflatex exists)\n",
    ]
    (out_dir / "v13o5_report.md").write_text("".join(md), encoding="utf-8")

    tex = (
        "\\documentclass[11pt]{article}\n\\usepackage[margin=1in]{geometry}\n\\begin{document}\n"
        "\\title{V13O.5 Robust unfolded normalization}\n\\maketitle\n"
        "\\section*{Warning}\nComputational evidence only; not a proof of the Riemann Hypothesis.\n\n"
        "\\section*{Classification}\n"
        + latex_escape(classification)
        + "\n\\end{document}\n"
    )
    (out_dir / "v13o5_report.tex").write_text(tex, encoding="utf-8")
    if _find_pdflatex() is not None:
        try_pdflatex(out_dir / "v13o5_report.tex", out_dir, "v13o5_report.pdf")

    payload = {
        "warning": "Computational evidence only; not a proof of the Riemann Hypothesis.",
        "status": "V13O.5 robust unfolded normalization",
        "classification": classification,
        "primary_word_group": args.primary_word_group,
        "primary_v_mode": args.primary_v_mode,
        "margin_z": margin_z,
        "mad_eps": mad_eps,
        "weights": w,
        "n_groups_scored": int(len(groups)),
        "n_gate_rows": int(len(gate_rows)),
        "gate_rows": gate_rows,
        "outputs": {
            "robust_scores_csv": str((out_dir / "v13o5_robust_scores.csv").resolve()),
            "gate_summary_csv": str((out_dir / "v13o5_gate_summary.csv").resolve()),
            "component_summary_csv": str((out_dir / "v13o5_component_summary.csv").resolve()),
            "best_by_dim_csv": str((out_dir / "v13o5_best_by_dim.csv").resolve()),
            "report_md": str((out_dir / "v13o5_report.md").resolve()),
            "report_tex": str((out_dir / "v13o5_report.tex").resolve()),
        },
    }
    (out_dir / "v13o5_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    elapsed = time.perf_counter() - t0
    print(f"[v13o5] wrote outputs to {out_dir} in {elapsed:.2f}s", flush=True)
    print("\n--- Smoke command ---\n", flush=True)
    print(
        "python3 scripts/run_v13o5_robust_unfolded_normalization.py \\\n"
        "  --v13o4_summary runs/v13o4_zeta_specific_objective/v13o4_summary.csv \\\n"
        "  --v13o4_group_summary runs/v13o4_zeta_specific_objective/v13o4_group_summary.csv \\\n"
        "  --v13o4_zeta_scores runs/v13o4_zeta_specific_objective/v13o4_zeta_specific_scores.csv \\\n"
        "  --v13o4_pair_corr runs/v13o4_zeta_specific_objective/v13o4_pair_corr_summary.csv \\\n"
        "  --v13o4_number_variance runs/v13o4_zeta_specific_objective/v13o4_number_variance_summary.csv \\\n"
        "  --v13o4_staircase runs/v13o4_zeta_specific_objective/v13o4_staircase_summary.csv \\\n"
        "  --v13o4_ensemble_margins runs/v13o4_zeta_specific_objective/v13o4_ensemble_margins.csv \\\n"
        "  --v13o4_gate_summary runs/v13o4_zeta_specific_objective/v13o4_gate_summary.csv \\\n"
        "  --out_dir runs/v13o5_robust_unfolded_normalization_smoke \\\n"
        "  --primary_word_group primary_word_seed6 \\\n"
        "  --primary_v_mode full_V \\\n"
        "  --margin_z 0.25 \\\n"
        "  --progress_every 1\n",
        flush=True,
    )


if __name__ == "__main__":
    main()

