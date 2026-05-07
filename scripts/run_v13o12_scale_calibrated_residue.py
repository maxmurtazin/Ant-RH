#!/usr/bin/env python3
"""
V13O.12 — Scale-calibrated residue / argument-principle diagnostics (true spectra).

Computational diagnostic only; not a proof of the Riemann Hypothesis.

Motivation:
V13O.11 used medians over *all* raw windows, which can be artificially small when many windows are empty
(N_operator=0 and N_target=0 => N_error_norm=0). V13O.12 removes this artifact by:
  - active-window filtering,
  - support-overlap diagnostics,
  - quantile-matched argument-count diagnostics,
  - residue/trace proxies aggregated only over meaningful support.
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


def finite_series(x: "pd.Series") -> "pd.Series":
    return pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan)


def safe_median(xs: Sequence[float]) -> float:
    arr = np.asarray([float(v) for v in xs if v is not None and math.isfinite(float(v))], dtype=np.float64)
    return float(np.median(arr)) if arr.size else float("nan")


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


def count_in_range(x: np.ndarray, lo: float, hi: float) -> int:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0
    a = float(lo)
    b = float(hi)
    left = int(np.searchsorted(x, a, side="left"))
    right = int(np.searchsorted(x, b, side="right"))
    return int(max(0, right - left))


def main() -> None:
    ap = argparse.ArgumentParser(description="V13O.12 scale-calibrated residue diagnostics (computational only).")

    ap.add_argument("--true_levels_csv", type=str, default="runs/v13o9_true_unfolded_spectra_from_v13o8/v13o9_unfolded_levels.csv")
    ap.add_argument("--candidate_ranking", type=str, default="runs/v13o10_true_spectra_candidate_rescue/v13o10_candidate_ranking.csv")
    ap.add_argument("--zeros_csv", type=str, default="runs/zeros_100_400_precise.csv")
    ap.add_argument("--out_dir", type=str, default="runs/v13o12_scale_calibrated_residue")

    ap.add_argument("--dims", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--primary_word_group", type=str, default="primary_word_seed6")
    ap.add_argument("--primary_v_mode", type=str, default="full_V")

    ap.add_argument("--window_min", type=float, default=100.0)
    ap.add_argument("--window_max", type=float, default=400.0)
    ap.add_argument("--window_size", type=float, default=40.0)
    ap.add_argument("--window_stride", type=float, default=20.0)
    ap.add_argument("--n_quantile_bins", type=int, default=10)

    ap.add_argument("--eta", type=float, default=0.15)
    ap.add_argument("--n_contour_points", type=int, default=256)
    ap.add_argument("--top_k_candidates", type=int, default=7)

    ap.add_argument("--support_overlap_min", type=float, default=0.5)
    ap.add_argument("--active_error_margin", type=float, default=0.25)
    ap.add_argument("--quantile_error_margin", type=float, default=0.25)

    ap.add_argument("--n_jobs", type=int, default=8)  # accepted; implementation is mostly vectorized/serial
    ap.add_argument("--progress_every", type=int, default=10)

    args = ap.parse_args()

    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    dims_keep = [int(x) for x in args.dims]
    primary_wg = str(args.primary_word_group)
    primary_vm = str(args.primary_v_mode)

    # Load true levels (V13O.9 source-format or V13O.8 kind-format)
    df_levels, lvl_warns = rd.load_true_levels_csv(_resolve(args.true_levels_csv), dims_keep=dims_keep)
    if df_levels is None or df_levels.empty:
        raise SystemExit(f"[v13o12] ERROR: true_levels_csv invalid/empty: {args.true_levels_csv} warns={lvl_warns}")

    # Candidate ranking (for coverage and poisson-like penalty)
    df_rank = pd.read_csv(_resolve(args.candidate_ranking))
    df_rank.columns = [str(c).strip() for c in df_rank.columns]
    df_rank["dim"] = pd.to_numeric(df_rank.get("dim"), errors="coerce").astype("Int64")
    for k in ("V_mode", "word_group"):
        if k in df_rank.columns:
            df_rank[k] = df_rank[k].astype(str).str.strip()
    df_rank = df_rank.dropna(subset=["dim", "V_mode", "word_group"])
    df_rank = df_rank[df_rank["dim"].astype(int).isin(dims_keep)]

    # Zeros csv (loaded for provenance + optional cross-check; diagnostics use target arrays from true_levels_csv)
    zeros_raw, zeros_warns = rd.load_zeros_csv(_resolve(args.zeros_csv))
    zeros_unfolded = rd.unfold_to_mean_spacing_one(zeros_raw)  # for report only

    warnings: List[str] = []
    warnings.extend([f"true_levels_csv: {w}" for w in lvl_warns])
    warnings.extend([f"zeros_csv: {w}" for w in zeros_warns])

    windows = rd.make_windows(args.window_min, args.window_max, args.window_size, args.window_stride)
    if not windows:
        raise SystemExit("[v13o12] ERROR: no windows produced; check window_min/max/size/stride.")

    # Candidate selection: per dim take top_k by candidate_rank_by_error if present else by median_total_curve_error
    rank_col = "candidate_rank_by_error" if "candidate_rank_by_error" in df_rank.columns else None
    if rank_col is None and "median_total_curve_error" in df_rank.columns:
        df_rank["_rank_tmp"] = pd.to_numeric(df_rank["median_total_curve_error"], errors="coerce").rank(method="min", ascending=True)
        rank_col = "_rank_tmp"
    if rank_col is None:
        df_rank["_rank_tmp"] = np.arange(len(df_rank), dtype=int) + 1
        rank_col = "_rank_tmp"
        warnings.append("candidate ranking column not found; using input order for top_k")

    selected: List[Tuple[int, str, str]] = []
    for d, g in df_rank.groupby("dim"):
        gg = g.sort_values([rank_col], ascending=True, na_position="last")
        for r in gg.head(int(args.top_k_candidates)).itertuples(index=False):
            selected.append((int(getattr(r, "dim")), str(getattr(r, "V_mode")), str(getattr(r, "word_group"))))
    # ensure primary included
    for d in dims_keep:
        if (int(d), primary_vm, primary_wg) not in selected:
            selected.append((int(d), primary_vm, primary_wg))
    selected = sorted(set(selected), key=lambda x: (x[0], x[1], x[2]))

    # Build list of (dim,vm,wg,target_group,seed) pairs where both operator+target exist
    df_levels["dim"] = pd.to_numeric(df_levels["dim"], errors="coerce").astype("Int64")
    df_levels["seed"] = pd.to_numeric(df_levels["seed"], errors="coerce").astype("Int64")
    for k in ("V_mode", "word_group", "target_group", "source"):
        df_levels[k] = df_levels[k].astype(str).str.strip()
    df_levels = df_levels.dropna(subset=["dim", "seed", "V_mode", "word_group", "target_group", "source", "unfolded_level"])

    pair_rows: List[Tuple[int, str, str, str, int]] = []
    for (d, vm, wg) in selected:
        sub = df_levels[(df_levels["dim"].astype(int) == int(d)) & (df_levels["V_mode"] == vm) & (df_levels["word_group"] == wg)]
        if sub.empty:
            continue
        g2 = sub.groupby(["target_group", "seed"])["source"].apply(lambda s: set(s.tolist()))
        for (tg, seed), ss in g2.items():
            if ("operator" in ss) and ("target" in ss):
                pair_rows.append((int(d), vm, wg, str(tg), int(seed)))
    pair_rows = sorted(set(pair_rows), key=lambda x: (x[0], x[1], x[2], x[3], x[4]))

    if not pair_rows:
        raise SystemExit("[v13o12] ERROR: no operator/target level pairs found in true_levels_csv for selected candidates.")

    prog = max(1, int(args.progress_every))
    exec_t0 = time.perf_counter()

    def log_prog(i: int, total: int, cur: Tuple[int, str, str, str, int]) -> None:
        if i == 1 or i == total or i % prog == 0:
            elapsed = time.perf_counter() - exec_t0
            avg = elapsed / max(i, 1)
            eta = avg * max(total - i, 0)
            d, vm, wg, tg, seed = cur
            print(f"[V13O.12] {i}/{total} elapsed={elapsed:.1f}s eta={format_seconds(eta)} current=({d},{vm},{wg},{tg},{seed})", flush=True)

    # Storage
    active_rows: List[Dict[str, Any]] = []
    quant_rows: List[Dict[str, Any]] = []
    residue_rows: List[Dict[str, Any]] = []
    trace_rows: List[Dict[str, Any]] = []

    eps = 1e-12
    n_bins = int(max(2, args.n_quantile_bins))
    q_edges = np.linspace(0.0, 1.0, n_bins + 1)
    sigmas = [0.5, 1.0, 2.0, 4.0]

    # Compute per pair
    for i, (d, vm, wg, tg, seed) in enumerate(pair_rows, start=1):
        log_prog(i, len(pair_rows), (d, vm, wg, tg, seed))
        sub = df_levels[
            (df_levels["dim"].astype(int) == int(d))
            & (df_levels["V_mode"] == vm)
            & (df_levels["word_group"] == wg)
            & (df_levels["target_group"] == tg)
            & (df_levels["seed"].astype(int) == int(seed))
        ]
        op = np.sort(sub[sub["source"] == "operator"]["unfolded_level"].astype(float).to_numpy(dtype=np.float64))
        tg_levels = np.sort(sub[sub["source"] == "target"]["unfolded_level"].astype(float).to_numpy(dtype=np.float64))
        op = op[np.isfinite(op)]
        tg_levels = tg_levels[np.isfinite(tg_levels)]
        if op.size == 0 or tg_levels.size == 0:
            continue

        # A. Active-window argument counts
        for (a, b) in windows:
            N_op = count_in_range(op, float(a), float(b))
            N_tg = count_in_range(tg_levels, float(a), float(b))
            active = bool((N_op > 0) or (N_tg > 0))
            both_active = bool((N_op > 0) and (N_tg > 0))
            N_err = abs(int(N_op) - int(N_tg))
            N_norm = float(N_err) / float(max(1, int(N_tg)))
            active_rows.append(
                {
                    "dim": int(d),
                    "V_mode": vm,
                    "word_group": wg,
                    "target_group": tg,
                    "seed": int(seed),
                    "window_a": float(a),
                    "window_b": float(b),
                    "N_operator": int(N_op),
                    "N_target": int(N_tg),
                    "N_error": float(N_err),
                    "N_error_norm": float(N_norm),
                    "active": bool(active),
                    "both_active": bool(both_active),
                }
            )
            if not active:
                continue

            # C. Residue proxy on active windows (raw windows)
            I_op = rd.residue_proxy_count(op, float(a), float(b), eta=float(args.eta), n_contour_points=int(args.n_contour_points))
            I_tg = rd.residue_proxy_count(tg_levels, float(a), float(b), eta=float(args.eta), n_contour_points=int(args.n_contour_points))
            residue_rows.append(
                {
                    "dim": int(d),
                    "V_mode": vm,
                    "word_group": wg,
                    "target_group": tg,
                    "seed": int(seed),
                    "window_kind": "raw_active",
                    "q0": float("nan"),
                    "q1": float("nan"),
                    "operator_a": float(a),
                    "operator_b": float(b),
                    "target_a": float(a),
                    "target_b": float(b),
                    "I_operator_real": float(I_op.real),
                    "I_operator_imag": float(I_op.imag),
                    "I_target_real": float(I_tg.real),
                    "I_target_imag": float(I_tg.imag),
                    "residue_count_error": float(abs(I_op.real - I_tg.real)),
                    "residue_imag_leak": float(abs(I_op.imag)),
                }
            )

            # D. Trace proxy on active windows
            c = 0.5 * (float(a) + float(b))
            for s in sigmas:
                Sop = rd.trace_formula_proxy(op, center=c, sigma=float(s))
                Stg = rd.trace_formula_proxy(tg_levels, center=c, sigma=float(s))
                terr = float(abs(Sop - Stg) / max(eps, abs(Stg))) if math.isfinite(Sop) and math.isfinite(Stg) else float("nan")
                trace_rows.append(
                    {
                        "dim": int(d),
                        "V_mode": vm,
                        "word_group": wg,
                        "target_group": tg,
                        "seed": int(seed),
                        "window_a": float(a),
                        "window_b": float(b),
                        "center": float(c),
                        "sigma": float(s),
                        "trace_error_norm": float(terr),
                    }
                )

        # B. Quantile-matched argument counts (per pair) + quantile residue proxy
        if op.size >= 4 and tg_levels.size >= 4:
            for j in range(n_bins):
                q0 = float(q_edges[j])
                q1 = float(q_edges[j + 1])
                t_lo = float(np.quantile(tg_levels, q0))
                t_hi = float(np.quantile(tg_levels, q1))
                o_lo = float(np.quantile(op, q0))
                o_hi = float(np.quantile(op, q1))

                Nt = count_in_range(tg_levels, t_lo, t_hi)
                No = count_in_range(op, o_lo, o_hi)
                qerr = abs(int(No) - int(Nt))
                qnorm = float(qerr) / float(max(1, int(Nt)))
                w_ratio = float((o_hi - o_lo) / max(eps, (t_hi - t_lo)))
                dens_ratio = float(1.0 / max(eps, w_ratio))

                quant_rows.append(
                    {
                        "dim": int(d),
                        "V_mode": vm,
                        "word_group": wg,
                        "target_group": tg,
                        "seed": int(seed),
                        "q0": q0,
                        "q1": q1,
                        "operator_low": o_lo,
                        "operator_high": o_hi,
                        "target_low": t_lo,
                        "target_high": t_hi,
                        "N_operator": int(No),
                        "N_target": int(Nt),
                        "quantile_count_error": float(qerr),
                        "quantile_count_error_norm": float(qnorm),
                        "width_ratio": float(w_ratio),
                        "local_density_ratio": float(dens_ratio),
                    }
                )

                # C (quantile). Residue proxy on quantile-matched windows (separate operator/target windows)
                # Note: contours differ by construction; we compare the proxy counts, not a shared contour.
                Ioq = rd.residue_proxy_count(op, float(o_lo), float(o_hi), eta=float(args.eta), n_contour_points=int(args.n_contour_points))
                Itq = rd.residue_proxy_count(tg_levels, float(t_lo), float(t_hi), eta=float(args.eta), n_contour_points=int(args.n_contour_points))
                residue_rows.append(
                    {
                        "dim": int(d),
                        "V_mode": vm,
                        "word_group": wg,
                        "target_group": tg,
                        "seed": int(seed),
                        "window_kind": "quantile",
                        "q0": float(q0),
                        "q1": float(q1),
                        "operator_a": float(o_lo),
                        "operator_b": float(o_hi),
                        "target_a": float(t_lo),
                        "target_b": float(t_hi),
                        "I_operator_real": float(Ioq.real),
                        "I_operator_imag": float(Ioq.imag),
                        "I_target_real": float(Itq.real),
                        "I_target_imag": float(Itq.imag),
                        "residue_count_error": float(abs(Ioq.real - Itq.real)),
                        "residue_imag_leak": float(abs(Ioq.imag)),
                    }
                )

    # DataFrames
    df_active = pd.DataFrame(active_rows)
    df_support = pd.DataFrame(columns=[
        "dim","V_mode","word_group",
        "n_windows_total","n_windows_active","n_windows_both_active",
        "active_window_fraction","target_active_fraction","operator_active_fraction",
        "support_overlap_fraction",
        "median_active_error_norm","mean_active_error_norm","max_active_error_norm",
    ])
    df_quant = pd.DataFrame(quant_rows)
    df_res = pd.DataFrame(residue_rows)
    df_trace = pd.DataFrame(trace_rows)

    # Deterministic sorts
    if not df_active.empty:
        df_active = df_active.sort_values(["dim","V_mode","word_group","target_group","seed","window_a"], ascending=True)
    if not df_quant.empty:
        df_quant = df_quant.sort_values(["dim","V_mode","word_group","target_group","seed","q0"], ascending=True)
    if not df_res.empty:
        df_res = df_res.sort_values(["dim","V_mode","word_group","target_group","seed","window_kind","operator_a","q0"], ascending=True)
    if not df_trace.empty:
        df_trace = df_trace.sort_values(["dim","V_mode","word_group","target_group","seed","window_a","sigma"], ascending=True)

    # Support-overlap aggregation per candidate (dim,vm,wg)
    if not df_active.empty:
        dfx = df_active.copy()
        g = dfx.groupby(["dim","V_mode","word_group"], as_index=False)
        rows = []
        for (d, vm, wg), gg in dfx.groupby(["dim","V_mode","word_group"]):
            n_total = int(len(gg))
            n_active = int(gg["active"].astype(bool).sum())
            n_both = int(gg["both_active"].astype(bool).sum())
            op_active = int((gg["N_operator"].astype(int) > 0).sum())
            tg_active = int((gg["N_target"].astype(int) > 0).sum())
            active_frac = float(n_active) / float(max(1, n_total))
            op_frac = float(op_active) / float(max(1, n_total))
            tg_frac = float(tg_active) / float(max(1, n_total))
            overlap = float(n_both) / float(max(1, n_active))
            errs_active = gg.loc[gg["active"].astype(bool), "N_error_norm"].astype(float).to_numpy()
            errs_active = errs_active[np.isfinite(errs_active)]
            rows.append(
                {
                    "dim": int(d),
                    "V_mode": str(vm),
                    "word_group": str(wg),
                    "n_windows_total": n_total,
                    "n_windows_active": n_active,
                    "n_windows_both_active": n_both,
                    "active_window_fraction": active_frac,
                    "target_active_fraction": tg_frac,
                    "operator_active_fraction": op_frac,
                    "support_overlap_fraction": overlap,
                    "median_active_error_norm": float(np.median(errs_active)) if errs_active.size else float("nan"),
                    "mean_active_error_norm": float(np.mean(errs_active)) if errs_active.size else float("nan"),
                    "max_active_error_norm": float(np.max(errs_active)) if errs_active.size else float("nan"),
                }
            )
        df_support = pd.DataFrame(rows).sort_values(["dim","V_mode","word_group"], ascending=True)

    # Quantile aggregation per candidate
    quant_agg = pd.DataFrame(columns=[
        "dim","V_mode","word_group",
        "median_quantile_error_norm","mean_quantile_error_norm",
        "median_abs_log_width_ratio","max_abs_log_width_ratio",
    ])
    if not df_quant.empty:
        rows = []
        for (d, vm, wg), gg in df_quant.groupby(["dim","V_mode","word_group"]):
            qerr = gg["quantile_count_error_norm"].astype(float).to_numpy()
            qerr = qerr[np.isfinite(qerr)]
            wr = gg["width_ratio"].astype(float).to_numpy()
            wr = wr[np.isfinite(wr) & (wr > 0.0)]
            alog = np.abs(np.log(wr)) if wr.size else np.asarray([], dtype=np.float64)
            rows.append(
                {
                    "dim": int(d),
                    "V_mode": str(vm),
                    "word_group": str(wg),
                    "median_quantile_error_norm": float(np.median(qerr)) if qerr.size else float("nan"),
                    "mean_quantile_error_norm": float(np.mean(qerr)) if qerr.size else float("nan"),
                    "median_abs_log_width_ratio": float(np.median(alog)) if alog.size else float("nan"),
                    "max_abs_log_width_ratio": float(np.max(alog)) if alog.size else float("nan"),
                }
            )
        quant_agg = pd.DataFrame(rows).sort_values(["dim","V_mode","word_group"], ascending=True)

    # Residue aggregation per candidate (separate raw_active vs quantile)
    res_agg = pd.DataFrame(
        columns=[
            "dim",
            "V_mode",
            "word_group",
            "median_residue_count_error_raw_active",
            "median_residue_count_error_quantile",
            "median_residue_imag_leak_raw_active",
        ]
    )
    if not df_res.empty:
        rows = []
        for (d, vm, wg), gg in df_res.groupby(["dim", "V_mode", "word_group"]):
            gg_raw = gg[gg["window_kind"].astype(str) == "raw_active"]
            gg_q = gg[gg["window_kind"].astype(str) == "quantile"]

            ce_raw = gg_raw["residue_count_error"].astype(float).to_numpy()
            ce_raw = ce_raw[np.isfinite(ce_raw)]
            ce_q = gg_q["residue_count_error"].astype(float).to_numpy()
            ce_q = ce_q[np.isfinite(ce_q)]

            il_raw = gg_raw["residue_imag_leak"].astype(float).to_numpy()
            il_raw = il_raw[np.isfinite(il_raw)]
            rows.append(
                {
                    "dim": int(d),
                    "V_mode": str(vm),
                    "word_group": str(wg),
                    "median_residue_count_error_raw_active": float(np.median(ce_raw)) if ce_raw.size else float("nan"),
                    "median_residue_count_error_quantile": float(np.median(ce_q)) if ce_q.size else float("nan"),
                    "median_residue_imag_leak_raw_active": float(np.median(il_raw)) if il_raw.size else float("nan"),
                }
            )
        res_agg = pd.DataFrame(rows).sort_values(["dim","V_mode","word_group"], ascending=True)

    # Poisson-like penalty from candidate ranking if available
    pr = df_rank.copy()
    for c in ("poisson_like_fraction",):
        if c in pr.columns:
            pr[c] = finite_series(pr[c])
    if "coverage_score" in pr.columns:
        pr["coverage_score"] = finite_series(pr["coverage_score"])
    pr = pr[["dim","V_mode","word_group"] + [c for c in ("coverage_score","poisson_like_fraction") if c in pr.columns] + [c for c in pr.columns if c.startswith("is_")]].copy()
    pr = pr.drop_duplicates(subset=["dim","V_mode","word_group"])

    # Merge candidate table
    df_cand = df_support.merge(quant_agg, on=["dim","V_mode","word_group"], how="left")
    df_cand = df_cand.merge(res_agg, on=["dim","V_mode","word_group"], how="left")
    df_cand = df_cand.merge(pr, on=["dim","V_mode","word_group"], how="left")

    # Compute gate + ranking J
    gate_rows: List[Dict[str, Any]] = []
    rank_rows: List[Dict[str, Any]] = []

    for r in df_cand.itertuples(index=False):
        d = int(getattr(r, "dim"))
        vm = str(getattr(r, "V_mode"))
        wg = str(getattr(r, "word_group"))

        coverage = float(getattr(r, "coverage_score", float("nan")))
        poisson_frac = float(getattr(r, "poisson_like_fraction", float("nan")))
        poisson_pen = 0.0 if (math.isfinite(poisson_frac) and poisson_frac <= 0.75) else 1.0

        active_frac = float(getattr(r, "active_window_fraction", float("nan")))
        overlap = float(getattr(r, "support_overlap_fraction", float("nan")))
        med_active = float(getattr(r, "median_active_error_norm", float("nan")))
        med_q = float(getattr(r, "median_quantile_error_norm", float("nan")))
        med_w = float(getattr(r, "median_abs_log_width_ratio", float("nan")))
        med_res = float(getattr(r, "median_residue_count_error_raw_active", float("nan")))
        med_res_q = float(getattr(r, "median_residue_count_error_quantile", float("nan")))

        B1 = bool(math.isfinite(med_active) or math.isfinite(med_q))
        B2 = bool(math.isfinite(coverage) and coverage >= 0.5) if math.isfinite(coverage) else False
        B3 = bool(math.isfinite(active_frac) and active_frac >= 0.25)
        B4 = bool(math.isfinite(overlap) and overlap >= float(args.support_overlap_min))
        B5 = bool(math.isfinite(med_active) and med_active <= float(args.active_error_margin))
        B6 = bool(math.isfinite(med_q) and med_q <= float(args.quantile_error_margin))
        B7 = bool(math.isfinite(med_w) and med_w <= 1.0)
        B8 = bool(math.isfinite(med_res) and med_res <= float(args.active_error_margin))
        B9 = bool(poisson_pen == 0.0)
        B10 = bool(not is_random_baseline(wg))
        B11 = bool(not is_ablation_only(wg))
        gate_pass = bool(B1 and B2 and B3 and B4 and B5 and B6 and B7 and B8 and B9 and B10 and B11)

        support_pen = float(max(0.0, float(args.support_overlap_min) - overlap)) if math.isfinite(overlap) else 1.0
        J = (
            float(med_active if math.isfinite(med_active) else 1.0)
            + float(med_q if math.isfinite(med_q) else 1.0)
            + float(med_res if math.isfinite(med_res) else 1.0)
            + 0.25 * float(med_w if math.isfinite(med_w) else 1.0)
            + float(poisson_pen)
            + float(artifact_penalty(wg, primary_wg))
            + float(support_pen)
        )

        gate_rows.append(
            {
                "dim": d,
                "V_mode": vm,
                "word_group": wg,
                "B1_true_levels_available": B1,
                "B2_candidate_coverage_ok": B2,
                "B3_active_window_fraction_ok": B3,
                "B4_support_overlap_ok": B4,
                "B5_active_argument_error_small": B5,
                "B6_quantile_argument_error_small": B6,
                "B7_width_distortion_reasonable": B7,
                "B8_residue_error_small": B8,
                "B9_not_poisson_like": B9,
                "B10_not_random_baseline": B10,
                "B11_not_ablation_only": B11,
                "gate_pass": gate_pass,
                "active_window_fraction": active_frac,
                "support_overlap_fraction": overlap,
                "median_active_error_norm": med_active,
                "median_quantile_error_norm": med_q,
                "median_abs_log_width_ratio": med_w,
                "median_residue_count_error": med_res,
                "median_residue_count_error_quantile": med_res_q,
                "poisson_like_penalty": poisson_pen,
                "coverage_score": coverage,
            }
        )
        rank_rows.append(
            {
                "dim": d,
                "V_mode": vm,
                "word_group": wg,
                "J_v13o12": float(J),
                "gate_pass": gate_pass,
                "median_active_error_norm": med_active,
                "median_quantile_error_norm": med_q,
                "median_residue_count_error": med_res,
                "median_abs_log_width_ratio": med_w,
                "support_overlap_fraction": overlap,
                "support_penalty": support_pen,
                "poisson_like_penalty": poisson_pen,
                "artifact_penalty": float(artifact_penalty(wg, primary_wg)),
            }
        )

    df_gate = pd.DataFrame(gate_rows).sort_values(["dim","gate_pass","median_active_error_norm","word_group"], ascending=[True,False,True,True], na_position="last")
    df_rank12 = pd.DataFrame(rank_rows)
    if not df_rank12.empty:
        df_rank12 = df_rank12.sort_values(["dim","J_v13o12","gate_pass"], ascending=[True,True,False], na_position="last")
        df_rank12["rank_by_J_v13o12"] = df_rank12.groupby("dim")["J_v13o12"].rank(method="min", ascending=True)

    # Primary and best summaries for report
    best_by_dim: Dict[int, str] = {}
    primary_active_pass = {}
    primary_quant_pass = {}
    for d in dims_keep:
        sub = df_rank12[df_rank12["dim"].astype(int) == int(d)]
        if not sub.empty:
            best_by_dim[int(d)] = str(sub.iloc[0]["word_group"])
        prim = df_gate[(df_gate["dim"].astype(int) == int(d)) & (df_gate["V_mode"] == primary_vm) & (df_gate["word_group"] == primary_wg)]
        primary_active_pass[int(d)] = bool(not prim.empty and bool(prim["B5_active_argument_error_small"].iloc[0]))
        primary_quant_pass[int(d)] = bool(not prim.empty and bool(prim["B6_quantile_argument_error_small"].iloc[0]))

    any_pass = bool((not df_gate.empty) and df_gate["gate_pass"].astype(bool).any())
    any_non_artifact_pass = False
    if any_pass:
        subp = df_gate[(df_gate["gate_pass"].astype(bool) == True)]
        any_non_artifact_pass = bool(
            subp["word_group"].astype(str).apply(lambda w: (not is_random_baseline(w)) and (not is_ablation_only(w))).any()
        )

    should_proceed_v13p0 = bool(any_non_artifact_pass)

    # Write outputs
    (out_dir / "v13o12_active_argument_counts.csv").write_text(df_active.to_csv(index=False), encoding="utf-8")
    (out_dir / "v13o12_support_overlap.csv").write_text(df_support.to_csv(index=False), encoding="utf-8")
    (out_dir / "v13o12_quantile_argument_counts.csv").write_text(df_quant.to_csv(index=False), encoding="utf-8")
    (out_dir / "v13o12_residue_scores.csv").write_text(df_res.to_csv(index=False), encoding="utf-8")
    (out_dir / "v13o12_trace_formula_proxy.csv").write_text(df_trace.to_csv(index=False), encoding="utf-8")
    (out_dir / "v13o12_gate_summary.csv").write_text(df_gate.to_csv(index=False), encoding="utf-8")
    (out_dir / "v13o12_candidate_ranking.csv").write_text(df_rank12.to_csv(index=False), encoding="utf-8")

    payload: Dict[str, Any] = {
        "warning": "Computational diagnostic only; not a proof of the Riemann Hypothesis.",
        "status": "V13O.12 scale-calibrated residue diagnostics",
        "inputs": {
            "true_levels_csv": str(_resolve(args.true_levels_csv).resolve()),
            "candidate_ranking": str(_resolve(args.candidate_ranking).resolve()),
            "zeros_csv": str(_resolve(args.zeros_csv).resolve()),
        },
        "dims": dims_keep,
        "primary": {"word_group": primary_wg, "V_mode": primary_vm},
        "windows": {
            "window_min": float(args.window_min),
            "window_max": float(args.window_max),
            "window_size": float(args.window_size),
            "window_stride": float(args.window_stride),
            "n_windows": int(len(windows)),
        },
        "n_quantile_bins": int(n_bins),
        "eta": float(args.eta),
        "n_contour_points": int(args.n_contour_points),
        "thresholds": {
            "support_overlap_min": float(args.support_overlap_min),
            "active_error_margin": float(args.active_error_margin),
            "quantile_error_margin": float(args.quantile_error_margin),
        },
        "primary_pass": {
            "active_window_argument_pass_by_dim": primary_active_pass,
            "quantile_argument_pass_by_dim": primary_quant_pass,
        },
        "best_by_dim": best_by_dim,
        "any_gate_pass": bool(any_pass),
        "any_non_artifact_gate_pass": bool(any_non_artifact_pass),
        "should_proceed_to_v13p0": bool(should_proceed_v13p0),
        "warnings": warnings,
        "zeros_loaded_n": int(zeros_raw.size),
        "zeros_unfolded_span": float(zeros_unfolded[-1] - zeros_unfolded[0]) if zeros_unfolded.size >= 2 else float("nan"),
        "runtime_s": float(time.perf_counter() - t0),
        "python_pid": os.getpid(),
    }
    (out_dir / "v13o12_results.json").write_text(json.dumps(json_sanitize(payload), indent=2, allow_nan=False), encoding="utf-8")

    # Report
    md: List[str] = []
    md.append("# V13O.12 Scale-calibrated residue diagnostics\n\n")
    md.append("> **Computational evidence only; not a proof of RH.**\n\n")
    md.append("## Why V13O.12 exists\n\n")
    md.append(
        "V13O.11 could produce artificially small `median_argument_count_error_norm` because many windows were empty for both operator and target, "
        "so `N_error_norm=0`. V13O.12 fixes this by computing **active-window** statistics and support-overlap diagnostics.\n\n"
    )
    md.append("## Method\n\n")
    md.append("- **Active-window argument counts**: exclude windows where both counts are zero from medians.\n")
    md.append("- **Support overlap**: track fraction of windows where both operator and target are active.\n")
    md.append("- **Quantile-matched bins**: compare counts in matched quantile slices and measure width distortion.\n")
    md.append("- **Residue / trace proxies**: aggregate only over active windows.\n\n")
    md.append("## Configuration\n\n")
    md.append(f"- window=[{args.window_min},{args.window_max}] size={args.window_size} stride={args.window_stride} (n_windows={len(windows)})\n")
    md.append(f"- n_quantile_bins={n_bins}\n")
    md.append(f"- eta={args.eta}, n_contour_points={args.n_contour_points}\n")
    md.append(f"- support_overlap_min={args.support_overlap_min}\n")
    md.append(f"- active_error_margin={args.active_error_margin}, quantile_error_margin={args.quantile_error_margin}\n\n")
    md.append("## Primary result\n\n")
    md.append(f"Primary: `V_mode={primary_vm}`, `word_group={primary_wg}`.\n\n")
    md.append("See `v13o12_gate_summary.csv` and `v13o12_support_overlap.csv`.\n\n")
    md.append("## Best candidate result\n\n")
    md.append("See `v13o12_candidate_ranking.csv`.\n\n")
    md.append("## Explicit answers\n\n")
    md.append("- Did V13O.12 remove empty-window artifacts? **Yes (active-window filtering + support overlap)**\n")
    md.append(f"- Does primary pass active-window argument diagnostics? **{primary_active_pass}**\n")
    md.append(f"- Does primary pass quantile-matched diagnostics? **{primary_quant_pass}**\n")
    md.append("- Is the mismatch mainly scale/support or structural? See width distortion + support overlap columns.\n")
    md.append(f"- Which candidate is best? **{best_by_dim}**\n")
    md.append(f"- Is any non-artifact candidate rescued? **{any_non_artifact_pass}**\n")
    md.append(f"- Should proceed to V13P0? **{should_proceed_v13p0}**\n\n")
    md.append("## Outputs\n\n")
    md.append("- `v13o12_active_argument_counts.csv`\n")
    md.append("- `v13o12_support_overlap.csv`\n")
    md.append("- `v13o12_quantile_argument_counts.csv`\n")
    md.append("- `v13o12_residue_scores.csv`\n")
    md.append("- `v13o12_trace_formula_proxy.csv`\n")
    md.append("- `v13o12_gate_summary.csv`\n")
    md.append("- `v13o12_candidate_ranking.csv`\n")
    md.append("- `v13o12_results.json`\n")
    md.append("- `v13o12_report.md`\n")
    md.append("- `v13o12_report.tex`\n")
    md.append("- `v13o12_report.pdf` (if `pdflatex` exists)\n\n")
    if warnings:
        md.append("## Warnings (summary)\n\n")
        md.append(f"- n_warnings={len(warnings)}\n")
        md.append(f"- first_warning={warnings[0]}\n\n")
    md.append("## Verification commands\n\n```bash\n")
    md.append(f"OUT={str(out_dir)}\n\n")
    md.append('echo "=== FILES ==="\nfind "$OUT" -maxdepth 1 -type f | sort\n\n')
    md.append('echo "=== PRIMARY ROWS (gate) ==="\n')
    md.append(f"awk -F, '$2==\"{primary_vm}\" && $3==\"{primary_wg}\"{{print}}' \"$OUT\"/v13o12_gate_summary.csv\n\n")
    md.append('echo "=== TOP RANKING ==="\ncolumn -s, -t < "$OUT"/v13o12_candidate_ranking.csv | head -40\n\n')
    md.append('echo "=== SUPPORT OVERLAP ==="\ncolumn -s, -t < "$OUT"/v13o12_support_overlap.csv | head -40\n\n')
    md.append('echo "=== REPORT ==="\nhead -220 "$OUT"/v13o12_report.md\n')
    md.append("```\n")
    (out_dir / "v13o12_report.md").write_text("".join(md), encoding="utf-8")

    tex = (
        "\\documentclass[11pt]{article}\n\\usepackage[margin=1in]{geometry}\n\\begin{document}\n"
        "\\title{V13O.12 Scale-calibrated residue diagnostics}\n\\maketitle\n"
        "\\section*{Warning}\nComputational evidence only; not a proof of RH.\n\n"
        "\\section*{Summary}\n"
        + latex_escape(json.dumps({"any_gate_pass": any_pass, "should_proceed_to_v13p0": should_proceed_v13p0}, indent=2))
        + "\n\n\\end{document}\n"
    )
    (out_dir / "v13o12_report.tex").write_text(tex, encoding="utf-8")

    if _find_pdflatex() is None:
        print("[v13o12] WARNING: pdflatex not found; skipping PDF.", flush=True)
    elif try_pdflatex(out_dir / "v13o12_report.tex", out_dir, "v13o12_report.pdf"):
        print(f"Wrote {out_dir / 'v13o12_report.pdf'}", flush=True)
    else:
        print("[v13o12] WARNING: pdflatex failed or did not produce v13o12_report.pdf.", flush=True)

    print(f"[v13o12] Wrote {out_dir / 'v13o12_results.json'}", flush=True)


if __name__ == "__main__":
    main()

