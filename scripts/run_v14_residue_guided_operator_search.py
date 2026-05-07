#!/usr/bin/env python3
"""
V14 — Residue-Guided Hilbert–Pólya Operator Search.

Computational evidence only; not a proof of the Riemann Hypothesis.

This pipeline moves from post-hoc transport diagnostics (V13O.12–V13O.14) to a constructive
candidate ranking objective using residue / argument-principle / trace / number-variance /
anti-Poisson rigidity and null-control margins.

Design principles:
  - Robust to missing optional inputs; emits warnings and continues.
  - Active-window aggregation (avoids empty-window artifacts).
  - Deterministic sorting and outputs.
  - Always writes all CSVs (at least headers), JSON, and a report.
"""

from __future__ import annotations

import argparse
import csv
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
except Exception:  # pragma: no cover
    pd = None  # type: ignore


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    from validation import residue_diagnostics as rd  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit(f"V14 requires validation/residue_diagnostics.py importable: {e!r}")


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


def write_csv_rows(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for r in rows:
            out = {k: r.get(k) for k in fieldnames}
            w.writerow(out)


def df_to_csv(path: Path, df: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if pd is None:
        raise RuntimeError("pandas not available to write dataframe")
    path.write_text(df.to_csv(index=False), encoding="utf-8")


def safe_float(x: Any) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def is_random_baseline(word_group: str) -> bool:
    wg = str(word_group)
    return ("random_words_n30" in wg) or ("random_symmetric_baseline" in wg)


def is_ablation_only(word_group: str) -> bool:
    return str(word_group) in ("ablate_K", "ablate_L", "ablate_V")


def is_rejected(word_group: str) -> bool:
    return "rejected_word_seed17" in str(word_group)


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


def median_or_nan(xs: Sequence[float]) -> float:
    arr = np.asarray([float(x) for x in xs if x is not None and math.isfinite(float(x))], dtype=np.float64)
    return float(np.median(arr)) if arr.size else float("nan")


def mean_or_nan(xs: Sequence[float]) -> float:
    arr = np.asarray([float(x) for x in xs if x is not None and math.isfinite(float(x))], dtype=np.float64)
    return float(np.mean(arr)) if arr.size else float("nan")


def load_csv_optional(path: Path, *, name: str, warns: List[str]) -> Any:
    if not path or str(path).strip() == "":
        return None
    if not path.is_file():
        warns.append(f"missing optional input: {name}={path}")
        return None
    if pd is None:
        warns.append(f"pandas unavailable; cannot read optional: {name}={path}")
        return None
    try:
        df = pd.read_csv(path)
        df.columns = [str(c).strip() for c in df.columns]
        return df
    except Exception as e:
        warns.append(f"failed reading optional {name}={path}: {e!r}")
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="V14 residue-guided operator search (computational only).")

    ap.add_argument("--true_levels_csv", type=str, required=True)
    ap.add_argument("--v13o10_candidate_ranking", type=str, required=True)
    ap.add_argument("--v13o12_dir", type=str, default="")
    ap.add_argument("--v13o13_dir", type=str, default="")
    ap.add_argument("--v13o14_dir", type=str, default="")
    ap.add_argument("--zeros_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--dims", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--primary_word_group", type=str, default="primary_word_seed6")
    ap.add_argument("--primary_v_mode", type=str, default="full_V")

    ap.add_argument("--window_min", type=float, default=100.0)
    ap.add_argument("--window_max", type=float, default=400.0)
    ap.add_argument("--window_size", type=float, default=40.0)
    ap.add_argument("--window_stride", type=float, default=20.0)

    ap.add_argument("--L_min", type=float, default=0.5)
    ap.add_argument("--L_max", type=float, default=16.0)
    ap.add_argument("--n_L", type=int, default=64)

    ap.add_argument("--eta", type=float, default=0.15)
    ap.add_argument("--n_contour_points", type=int, default=256)
    ap.add_argument("--top_k_candidates", type=int, default=7)

    ap.add_argument("--lambda_spec", type=float, default=1.0)  # accepted (reserved)
    ap.add_argument("--lambda_nv", type=float, default=1.0)
    ap.add_argument("--lambda_residue", type=float, default=1.0)
    ap.add_argument("--lambda_argument", type=float, default=1.0)
    ap.add_argument("--lambda_trace", type=float, default=1.0)
    ap.add_argument("--lambda_antipoisson", type=float, default=1.0)
    ap.add_argument("--lambda_null_margin", type=float, default=1.0)
    ap.add_argument("--lambda_selfadjoint", type=float, default=1.0)

    ap.add_argument("--support_overlap_min", type=float, default=0.5)
    ap.add_argument("--active_error_margin", type=float, default=0.25)
    ap.add_argument("--residue_error_margin", type=float, default=0.25)
    ap.add_argument("--poisson_like_max", type=float, default=0.5)
    ap.add_argument("--null_margin_min", type=float, default=0.0)

    ap.add_argument("--n_jobs", type=int, default=8)  # accepted; current implementation is serial/deterministic
    ap.add_argument("--progress_every", type=int, default=10)

    args = ap.parse_args()

    if pd is None:
        raise SystemExit("V14 requires pandas in this repository (csv fallback is only for writing headers).")

    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    warns: List[str] = []

    # Optional dirs
    v12_dir = _resolve(args.v13o12_dir) if str(args.v13o12_dir).strip() else None
    v13_dir = _resolve(args.v13o13_dir) if str(args.v13o13_dir).strip() else None
    v14_dir = _resolve(args.v13o14_dir) if str(args.v13o14_dir).strip() else None

    # Load zeros (provenance + sanity)
    try:
        zeros_raw, zeros_warns = rd.load_zeros_csv(_resolve(args.zeros_csv))
        warns.extend([f"zeros_csv: {w}" for w in zeros_warns])
    except Exception as e:
        zeros_raw = np.asarray([], dtype=np.float64)
        warns.append(f"zeros_csv load failed: {e!r}")

    dims_keep = [int(d) for d in args.dims]
    primary_wg = str(args.primary_word_group)
    primary_vm = str(args.primary_v_mode)

    # Load true levels (supports source/kind formats via rd)
    df_levels, lvl_warns = rd.load_true_levels_csv(_resolve(args.true_levels_csv), dims_keep=dims_keep)
    if df_levels is None or df_levels.empty:
        raise SystemExit(f"[V14] ERROR: true_levels_csv invalid/empty: {args.true_levels_csv} warns={lvl_warns}")
    warns.extend([f"true_levels_csv: {w}" for w in lvl_warns])

    # Detect approximation flags if present in original CSV (best-effort)
    approx_detected_any = False
    try:
        df_raw = pd.read_csv(_resolve(args.true_levels_csv))
        df_raw.columns = [str(c).strip() for c in df_raw.columns]
        if "approximation_mode_reconstructed_levels" in df_raw.columns:
            approx_detected_any = bool(pd.to_numeric(df_raw["approximation_mode_reconstructed_levels"], errors="coerce").fillna(0).astype(int).sum() > 0)
            if approx_detected_any:
                warns.append("approximation_mode_reconstructed_levels detected in true_levels_csv (not true-mode)")
    except Exception:
        pass

    # Load candidate ranking V13O.10
    df10 = pd.read_csv(_resolve(args.v13o10_candidate_ranking))
    df10.columns = [str(c).strip() for c in df10.columns]
    if "dim" in df10.columns:
        df10["dim"] = pd.to_numeric(df10["dim"], errors="coerce").astype("Int64")
    for c in ("V_mode", "word_group"):
        if c in df10.columns:
            df10[c] = df10[c].astype(str).str.strip()
    df10 = df10.dropna(subset=["dim", "V_mode", "word_group"])
    df10 = df10[df10["dim"].astype(int).isin(dims_keep)]

    # Optional reuse: poisson_like_fraction from V13O.10 if present
    if "poisson_like_fraction" in df10.columns:
        df10["poisson_like_fraction"] = pd.to_numeric(df10["poisson_like_fraction"], errors="coerce").astype(float)
    if "coverage_score" in df10.columns:
        df10["coverage_score"] = pd.to_numeric(df10["coverage_score"], errors="coerce").astype(float)

    # Candidate selection: top-k per dim from df10 (if candidate_rank_by_error exists else by input order)
    rank_col = "candidate_rank_by_error" if "candidate_rank_by_error" in df10.columns else None
    if rank_col is None:
        df10["_rank_tmp"] = np.arange(len(df10), dtype=int) + 1
        rank_col = "_rank_tmp"
        warns.append("v13o10 ranking column candidate_rank_by_error not found; using input order for top_k selection")

    selected: List[Tuple[int, str, str]] = []
    for d, g in df10.groupby("dim"):
        gg = g.sort_values([rank_col], ascending=True, na_position="last")
        for r in gg.head(int(args.top_k_candidates)).itertuples(index=False):
            selected.append((int(getattr(r, "dim")), str(getattr(r, "V_mode")), str(getattr(r, "word_group"))))
    # Ensure primary + required null controls included
    null_word_groups = [
        "random_words_n30",
        "random_symmetric_baseline",
        "rejected_word_seed17",
        "ablate_K",
        "ablate_L",
        "ablate_V",
    ]
    for d in dims_keep:
        if (int(d), primary_vm, primary_wg) not in selected:
            selected.append((int(d), primary_vm, primary_wg))
        for wg in null_word_groups:
            selected.append((int(d), primary_vm, wg))
    selected = sorted(set(selected), key=lambda x: (x[0], x[1], x[2]))

    # Precompute windows and L-grid
    windows = rd.make_windows(args.window_min, args.window_max, args.window_size, args.window_stride)
    if not windows:
        raise SystemExit("[V14] ERROR: no windows produced; check window_min/max/size/stride.")
    L_grid = make_L_grid(args.L_min, args.L_max, args.n_L)

    # Normalize df_levels
    df_levels = df_levels.copy()
    df_levels["dim"] = pd.to_numeric(df_levels["dim"], errors="coerce").astype("Int64")
    df_levels["seed"] = pd.to_numeric(df_levels["seed"], errors="coerce").astype("Int64")
    for k in ("V_mode", "word_group", "target_group", "source"):
        df_levels[k] = df_levels[k].astype(str).str.strip()
    df_levels["unfolded_level"] = pd.to_numeric(df_levels["unfolded_level"], errors="coerce").astype(float)
    df_levels = df_levels.dropna(subset=["dim", "seed", "V_mode", "word_group", "target_group", "source", "unfolded_level"])

    # Utility: get operator/target levels for a candidate vs target_group
    def get_pair_levels(d: int, vm: str, wg: str, target_group: str) -> List[Tuple[str, int, np.ndarray, np.ndarray]]:
        out: List[Tuple[str, int, np.ndarray, np.ndarray]] = []
        sub = df_levels[
            (df_levels["dim"].astype(int) == int(d))
            & (df_levels["V_mode"] == vm)
            & (df_levels["word_group"] == wg)
            & (df_levels["target_group"] == target_group)
        ]
        if sub.empty:
            return out
        for seed, gg in sub.groupby("seed"):
            srcset = set(gg["source"].astype(str).tolist())
            if ("operator" not in srcset) or ("target" not in srcset):
                continue
            op = np.sort(gg[gg["source"] == "operator"]["unfolded_level"].astype(float).to_numpy(dtype=np.float64))
            tg = np.sort(gg[gg["source"] == "target"]["unfolded_level"].astype(float).to_numpy(dtype=np.float64))
            op = op[np.isfinite(op)]
            tg = tg[np.isfinite(tg)]
            if op.size < 8 or tg.size < 8:
                continue
            out.append((str(target_group), int(seed), op, tg))
        return out

    # Accumulators for required output CSVs
    arg_rows: List[Dict[str, Any]] = []
    res_rows: List[Dict[str, Any]] = []
    trace_rows: List[Dict[str, Any]] = []
    nv_rows: List[Dict[str, Any]] = []
    anti_rows: List[Dict[str, Any]] = []

    # Candidate-level aggregates vs real_zeta
    cand_metrics: Dict[Tuple[int, str, str], Dict[str, Any]] = {}

    sigmas = [0.5, 1.0, 2.0, 4.0]
    slope_threshold = 0.65
    eps = 1e-12

    prog = max(1, int(args.progress_every))
    exec_t0 = time.perf_counter()

    # Build job list: (candidate, target_group, seed) for real_zeta and also some controls for diagnostics
    control_target_groups = [
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

    job_list: List[Tuple[int, str, str, str, int, np.ndarray, np.ndarray]] = []
    for (d, vm, wg) in selected:
        # primary objective pairs are real_zeta
        for (tg, seed, op, zt) in get_pair_levels(d, vm, wg, "real_zeta"):
            job_list.append((d, vm, wg, tg, seed, op, zt))
        # controls: include if present (diagnostic only; can be empty)
        for ctrl in control_target_groups:
            for (tg, seed, op, zt) in get_pair_levels(d, vm, wg, ctrl):
                job_list.append((d, vm, wg, tg, seed, op, zt))

    if not job_list:
        warns.append("no operator/target level pairs found for selected candidates (including real_zeta); outputs will be empty")

    total = len(job_list)

    def log_prog(i: int, cur: Tuple[int, str, str, str, int]) -> None:
        if i == 1 or i == total or i % prog == 0:
            elapsed = time.perf_counter() - exec_t0
            avg = elapsed / max(i, 1)
            eta = avg * max(total - i, 0)
            d, vm, wg, tg, seed = cur
            print(f"[V14] {i}/{total} elapsed={elapsed:.1f}s eta={format_seconds(eta)} current=({d},{vm},{wg},{tg},{seed})", flush=True)

    for i, (d, vm, wg, tg, seed, op, zt) in enumerate(job_list, start=1):
        log_prog(i, (d, vm, wg, tg, seed))
        key = (int(d), str(vm), str(wg))
        cm = cand_metrics.get(key)
        if cm is None:
            cm = {
                "dim": int(d),
                "V_mode": str(vm),
                "word_group": str(wg),
                "pairs_real_zeta": 0,
                "pairs_controls": 0,
                "support_overlap_fracs": [],
                "argument_losses": [],
                "residue_losses": [],
                "imag_leaks": [],
                "trace_losses": [],
                "nv_losses": [],
                "anti_poisson_losses": [],
                "poisson_flags": [],
                "active_window_counts": [],
            }
            cand_metrics[key] = cm

        is_real = (tg == "real_zeta")
        if is_real:
            cm["pairs_real_zeta"] += 1
        else:
            cm["pairs_controls"] += 1

        # Active-window argument counts + residue + trace
        active_errs_direct: List[float] = []
        active_errs_res: List[float] = []
        active_res_count_losses: List[float] = []
        active_imag_leaks: List[float] = []
        trace_errs: List[float] = []

        n_active = 0
        n_both = 0

        for (a, b) in windows:
            n_op = rd.count_in_window(op, float(a), float(b))
            n_tg = rd.count_in_window(zt, float(a), float(b))
            active = bool((n_op > 0) or (n_tg > 0))
            both_active = bool((n_op > 0) and (n_tg > 0))
            if active:
                n_active += 1
            if both_active:
                n_both += 1

            # Direct count error (argument principle proxy #1)
            err_dir = abs(int(n_op) - int(n_tg))
            err_dir_norm = float(err_dir) / float(max(1, int(n_tg)))

            # Residue proxy count (argument principle proxy #2)
            I_op = rd.residue_proxy_count(op, float(a), float(b), eta=float(args.eta), n_contour_points=int(args.n_contour_points))
            I_tg = rd.residue_proxy_count(zt, float(a), float(b), eta=float(args.eta), n_contour_points=int(args.n_contour_points))
            n_op_res = float(I_op.real)
            n_tg_res = float(I_tg.real)
            err_res = abs(n_op_res - n_tg_res)
            err_res_norm = float(err_res) / float(max(1.0, abs(n_tg_res)))
            imag_leak = float(abs(I_op.imag))

            arg_rows.append(
                {
                    "dim": int(d),
                    "V_mode": str(vm),
                    "word_group": str(wg),
                    "target_group": str(tg),
                    "seed": int(seed),
                    "window_a": float(a),
                    "window_b": float(b),
                    "N_operator_direct": int(n_op),
                    "N_target_direct": int(n_tg),
                    "N_operator_residue": float(n_op_res),
                    "N_target_residue": float(n_tg_res),
                    "N_error_direct": float(err_dir),
                    "N_error_residue": float(err_res),
                    "N_error_direct_norm": float(err_dir_norm),
                    "N_error_residue_norm": float(err_res_norm),
                    "active_window": bool(active),
                }
            )
            res_rows.append(
                {
                    "dim": int(d),
                    "V_mode": str(vm),
                    "word_group": str(wg),
                    "target_group": str(tg),
                    "seed": int(seed),
                    "window_a": float(a),
                    "window_b": float(b),
                    "I_operator_real": float(I_op.real),
                    "I_operator_imag": float(I_op.imag),
                    "I_target_real": float(I_tg.real),
                    "I_target_imag": float(I_tg.imag),
                    "residue_count_error": float(abs(I_op.real - I_tg.real)),
                    "residue_imag_leak": float(imag_leak),
                }
            )

            if not active:
                continue

            active_errs_direct.append(float(err_dir_norm))
            active_errs_res.append(float(err_res_norm))
            active_res_count_losses.append(float(err_res_norm))
            active_imag_leaks.append(float(imag_leak))

            # Trace proxy per active window
            c = 0.5 * (float(a) + float(b))
            for s in sigmas:
                Sop = rd.trace_formula_proxy(op, center=float(c), sigma=float(s))
                Stg = rd.trace_formula_proxy(zt, center=float(c), sigma=float(s))
                terr = float(abs(Sop - Stg) / max(eps, abs(Stg))) if (math.isfinite(Sop) and math.isfinite(Stg)) else float("nan")
                trace_rows.append(
                    {
                        "dim": int(d),
                        "V_mode": str(vm),
                        "word_group": str(wg),
                        "target_group": str(tg),
                        "seed": int(seed),
                        "window_a": float(a),
                        "window_b": float(b),
                        "center": float(c),
                        "sigma": float(s),
                        "S_operator": float(Sop),
                        "S_target": float(Stg),
                        "trace_error_norm": float(terr),
                    }
                )
                if math.isfinite(terr):
                    trace_errs.append(float(terr))

        support_overlap = float(n_both) / float(max(1, n_active))

        # Number variance curves + anti-Poisson slope (pair-level)
        op_nv = number_variance_curve(op, L_grid)
        tg_nv = number_variance_curve(zt, L_grid)
        nv_loss = curve_l2(op_nv, tg_nv)
        slope_long, intercept_long = fit_long_slope(L_grid, op_nv, L_min_long=6.0)
        poisson_like_flag = bool(math.isfinite(slope_long) and slope_long >= slope_threshold)
        anti_poisson_loss = float(max(0.0, (slope_long - slope_threshold))) if math.isfinite(slope_long) else float("nan")

        for L, y in zip(L_grid.tolist(), op_nv.tolist()):
            nv_rows.append(
                {
                    "dim": int(d),
                    "V_mode": str(vm),
                    "word_group": str(wg),
                    "target_group": str(tg),
                    "seed": int(seed),
                    "source": "operator",
                    "L": float(L),
                    "Sigma2": float(y),
                }
            )
        for L, y in zip(L_grid.tolist(), tg_nv.tolist()):
            nv_rows.append(
                {
                    "dim": int(d),
                    "V_mode": str(vm),
                    "word_group": str(wg),
                    "target_group": str(tg),
                    "seed": int(seed),
                    "source": "target",
                    "L": float(L),
                    "Sigma2": float(y),
                }
            )
        anti_rows.append(
            {
                "dim": int(d),
                "V_mode": str(vm),
                "word_group": str(wg),
                "target_group": str(tg),
                "seed": int(seed),
                "slope_long": float(slope_long),
                "intercept_long": float(intercept_long),
                "poisson_like_flag": bool(poisson_like_flag),
                "anti_poisson_rigidity_loss": float(anti_poisson_loss),
            }
        )

        # Candidate-level accumulate only for real_zeta objective components
        if is_real:
            cm["support_overlap_fracs"].append(float(support_overlap))
            cm["argument_losses"].append(median_or_nan(active_errs_direct))
            cm["residue_losses"].append(median_or_nan(active_res_count_losses))
            cm["imag_leaks"].append(median_or_nan(active_imag_leaks))
            cm["trace_losses"].append(median_or_nan(trace_errs))
            cm["nv_losses"].append(float(nv_loss))
            cm["anti_poisson_losses"].append(float(anti_poisson_loss))
            cm["poisson_flags"].append(float(1.0 if poisson_like_flag else 0.0))
            cm["active_window_counts"].append(int(n_active))

    # Candidate scores + null margins + gate summary
    cand_rows: List[Dict[str, Any]] = []
    null_rows: List[Dict[str, Any]] = []
    gate_rows: List[Dict[str, Any]] = []

    # Precompute J_v14 for each candidate (real_zeta only)
    J_by_candidate: Dict[Tuple[int, str, str], float] = {}
    poisson_by_candidate: Dict[Tuple[int, str, str], float] = {}

    for key, cm in cand_metrics.items():
        d, vm, wg = key
        support = median_or_nan(cm["support_overlap_fracs"])
        arg_loss = median_or_nan(cm["argument_losses"])
        res_loss = median_or_nan(cm["residue_losses"])
        imag_leak = median_or_nan(cm["imag_leaks"])
        trace_loss = median_or_nan(cm["trace_losses"])
        nv_loss = median_or_nan(cm["nv_losses"])
        anti_loss = median_or_nan(cm["anti_poisson_losses"])
        pois_frac_from_slope = mean_or_nan(cm["poisson_flags"])

        # If v13o10 poisson_like_fraction available, use it (more stable), else use slope-based.
        pois_prev = float("nan")
        if "poisson_like_fraction" in df10.columns:
            m = (df10["dim"].astype(int) == int(d)) & (df10["V_mode"] == str(vm)) & (df10["word_group"] == str(wg))
            if m.any():
                pois_prev = safe_float(df10.loc[m, "poisson_like_fraction"].iloc[0])
        pois_frac = float(pois_prev) if math.isfinite(pois_prev) else float(pois_frac_from_slope if math.isfinite(pois_frac_from_slope) else 1.0)

        # Null margin computed later; provisional for now
        null_margin_loss = 0.0
        selfadjoint_loss = 0.0

        J = (
            float(args.lambda_nv) * (nv_loss if math.isfinite(nv_loss) else 1.0)
            + float(args.lambda_residue) * (res_loss if math.isfinite(res_loss) else 1.0)
            + float(args.lambda_argument) * (arg_loss if math.isfinite(arg_loss) else 1.0)
            + float(args.lambda_trace) * (trace_loss if math.isfinite(trace_loss) else 1.0)
            + float(args.lambda_antipoisson) * (anti_loss if math.isfinite(anti_loss) else 1.0)
            + float(args.lambda_null_margin) * float(null_margin_loss)
            + float(args.lambda_selfadjoint) * float(selfadjoint_loss)
        )

        J_by_candidate[key] = float(J)
        poisson_by_candidate[key] = float(pois_frac)

    # Null margins: compare against null operator word_groups (same dim/V_mode)
    for key, Jcand in J_by_candidate.items():
        d, vm, wg = key
        # Gather null Js
        nulls_random = []
        nulls_ablation = []
        null_rejected = []
        for wg0 in ("random_words_n30", "random_symmetric_baseline"):
            k0 = (int(d), str(vm), wg0)
            if k0 in J_by_candidate and math.isfinite(J_by_candidate[k0]):
                nulls_random.append(float(J_by_candidate[k0]))
        for wg0 in ("ablate_K", "ablate_L", "ablate_V"):
            k0 = (int(d), str(vm), wg0)
            if k0 in J_by_candidate and math.isfinite(J_by_candidate[k0]):
                nulls_ablation.append(float(J_by_candidate[k0]))
        k_rej = (int(d), str(vm), "rejected_word_seed17")
        if k_rej in J_by_candidate and math.isfinite(J_by_candidate[k_rej]):
            null_rejected.append(float(J_by_candidate[k_rej]))

        best_random_J = float(np.nanmin(np.asarray(nulls_random, dtype=np.float64))) if nulls_random else float("nan")
        best_ablation_J = float(np.nanmin(np.asarray(nulls_ablation, dtype=np.float64))) if nulls_ablation else float("nan")
        rejected_word_J = float(np.nanmin(np.asarray(null_rejected, dtype=np.float64))) if null_rejected else float("nan")

        margins = []
        for v in (best_random_J, best_ablation_J, rejected_word_J):
            if math.isfinite(v) and math.isfinite(Jcand):
                margins.append(float(v - Jcand))
        null_margin = float(np.nanmin(np.asarray(margins, dtype=np.float64))) if margins else float("nan")
        beats_random = bool(math.isfinite(best_random_J) and math.isfinite(Jcand) and (Jcand < best_random_J))
        beats_ablations = bool(math.isfinite(best_ablation_J) and math.isfinite(Jcand) and (Jcand < best_ablation_J))
        beats_rejected = bool(math.isfinite(rejected_word_J) and math.isfinite(Jcand) and (Jcand < rejected_word_J))

        null_margin_loss = float(max(0.0, float(args.null_margin_min) - null_margin)) if math.isfinite(null_margin) else 1.0

        null_rows.append(
            {
                "dim": int(d),
                "V_mode": str(vm),
                "word_group": str(wg),
                "J_candidate": float(Jcand),
                "best_random_J": float(best_random_J),
                "rejected_word_J": float(rejected_word_J),
                "best_ablation_J": float(best_ablation_J),
                "null_margin": float(null_margin),
                "beats_random": bool(beats_random),
                "beats_rejected": bool(beats_rejected),
                "beats_ablations": bool(beats_ablations),
                "null_margin_loss": float(null_margin_loss),
            }
        )

    df_null = pd.DataFrame(null_rows)
    if not df_null.empty:
        df_null = df_null.sort_values(["dim", "V_mode", "word_group"], ascending=True)

    # Update J_v14 with null margin losses
    null_loss_map = {(int(r.dim), str(r.V_mode), str(r.word_group)): float(r.null_margin_loss) for r in df_null.itertuples(index=False)} if not df_null.empty else {}
    margin_map = {(int(r.dim), str(r.V_mode), str(r.word_group)): float(r.null_margin) for r in df_null.itertuples(index=False)} if not df_null.empty else {}
    beats_map = {(int(r.dim), str(r.V_mode), str(r.word_group)): (bool(r.beats_random), bool(r.beats_rejected), bool(r.beats_ablations)) for r in df_null.itertuples(index=False)} if not df_null.empty else {}

    for key, cm in cand_metrics.items():
        d, vm, wg = key
        support = median_or_nan(cm["support_overlap_fracs"])
        arg_loss = median_or_nan(cm["argument_losses"])
        res_loss = median_or_nan(cm["residue_losses"])
        imag_leak = median_or_nan(cm["imag_leaks"])
        trace_loss = median_or_nan(cm["trace_losses"])
        nv_loss = median_or_nan(cm["nv_losses"])
        anti_loss = median_or_nan(cm["anti_poisson_losses"])
        active_ct = int(np.nanmax(np.asarray(cm["active_window_counts"], dtype=np.float64))) if cm["active_window_counts"] else 0
        pois_frac = poisson_by_candidate.get(key, 1.0)

        null_margin_loss = null_loss_map.get(key, 1.0)
        null_margin = margin_map.get(key, float("nan"))
        beats_random, beats_rejected, beats_ablations = beats_map.get(key, (False, False, False))

        self_adjoint_available = False
        selfadjoint_loss = 0.0
        if not self_adjoint_available:
            # Warn once per run
            pass

        J = (
            float(args.lambda_nv) * (nv_loss if math.isfinite(nv_loss) else 1.0)
            + float(args.lambda_residue) * (res_loss if math.isfinite(res_loss) else 1.0)
            + float(args.lambda_argument) * (arg_loss if math.isfinite(arg_loss) else 1.0)
            + float(args.lambda_trace) * (trace_loss if math.isfinite(trace_loss) else 1.0)
            + float(args.lambda_antipoisson) * (anti_loss if math.isfinite(anti_loss) else 1.0)
            + float(args.lambda_null_margin) * float(null_margin_loss)
            + float(args.lambda_selfadjoint) * float(selfadjoint_loss)
        )
        J_by_candidate[key] = float(J)

    warns.append("self-adjointness verification unavailable (no raw operator matrices provided); gate can only be PROVISIONAL at best.")

    # Rank by J_v14 per dim
    cand_keys = sorted(J_by_candidate.keys(), key=lambda x: (int(x[0]), str(x[1]), str(x[2])))
    # build per-dim ordering
    rank_map: Dict[Tuple[int, str, str], int] = {}
    for d in dims_keep:
        sub = [(k, J_by_candidate[k]) for k in cand_keys if int(k[0]) == int(d) and math.isfinite(J_by_candidate[k])]
        sub.sort(key=lambda kv: float(kv[1]))
        for rnk, (k, _) in enumerate(sub, start=1):
            rank_map[k] = int(rnk)

    # Candidate scores rows + gate summary rows
    for key in cand_keys:
        d, vm, wg = key
        cm = cand_metrics.get(key, None)
        if cm is None:
            continue
        support = median_or_nan(cm["support_overlap_fracs"])
        arg_loss = median_or_nan(cm["argument_losses"])
        res_loss = median_or_nan(cm["residue_losses"])
        imag_leak = median_or_nan(cm["imag_leaks"])
        trace_loss = median_or_nan(cm["trace_losses"])
        nv_loss = median_or_nan(cm["nv_losses"])
        anti_loss = median_or_nan(cm["anti_poisson_losses"])
        active_ct = int(np.nanmax(np.asarray(cm["active_window_counts"], dtype=np.float64))) if cm["active_window_counts"] else 0
        pois_frac = poisson_by_candidate.get(key, 1.0)
        J = float(J_by_candidate.get(key, float("nan")))
        rank = int(rank_map.get(key, 999999))

        null_margin_loss = null_loss_map.get(key, 1.0)
        null_margin = margin_map.get(key, float("nan"))
        beats_random, beats_rejected, beats_ablations = beats_map.get(key, (False, False, False))

        is_primary = bool((str(vm) == primary_vm) and (str(wg) == primary_wg))
        is_rand = bool(is_random_baseline(wg))
        is_ab = bool(is_ablation_only(wg))
        is_rej = bool(is_rejected(wg))

        true_mode_ok = True
        approximation_detected = bool(approx_detected_any)

        # Gate logic
        G1 = bool(true_mode_ok and (not approximation_detected))
        G2 = bool(math.isfinite(support) and support >= float(args.support_overlap_min))
        G3 = bool(math.isfinite(arg_loss) and arg_loss <= float(args.active_error_margin))
        G4 = bool(math.isfinite(res_loss) and res_loss <= float(args.residue_error_margin))
        G5 = bool(math.isfinite(pois_frac) and pois_frac <= float(args.poisson_like_max))
        G6 = bool(math.isfinite(null_margin) and null_margin >= float(args.null_margin_min))
        G7 = bool(beats_random)
        G8 = bool(beats_rejected)
        G9 = bool(beats_ablations)
        G10 = bool((not is_rand) and (not is_ab))  # not artifact
        selfadj_status = "UNKNOWN"
        gate_pass_core = bool(G1 and G2 and G3 and G4 and G5 and G6 and G7 and G8 and G9 and G10)

        classification = "FAIL_GENERAL"
        gate_pass = False
        if not true_mode_ok:
            classification = "FAIL_MISSING_LEVELS"
        elif approximation_detected:
            classification = "FAIL_APPROXIMATION_MODE"
        elif not G2:
            classification = "FAIL_SUPPORT"
        elif not G3:
            classification = "FAIL_ARGUMENT"
        elif not G4:
            classification = "FAIL_RESIDUE"
        elif not G5:
            classification = "FAIL_POISSONIZATION"
        elif not G6:
            classification = "FAIL_NULL_MARGIN"
        elif not G7:
            classification = "FAIL_RANDOM_BASELINE"
        elif not G8:
            classification = "FAIL_REJECTED_WORD"
        elif not G9:
            classification = "FAIL_ABLATION"
        elif gate_pass_core:
            # self-adjointness unknown => provisional
            classification = "PROVISIONAL_SELFADJOINT_UNKNOWN"
            gate_pass = False
        else:
            classification = "FAIL_GENERAL"

        cand_rows.append(
            {
                "dim": int(d),
                "V_mode": str(vm),
                "word_group": str(wg),
                "J_v14": float(J),
                "rank_by_J_v14": int(rank),
                "number_variance_curve_loss": float(nv_loss),
                "argument_count_loss": float(arg_loss),
                "residue_count_loss": float(res_loss),
                "residue_imag_leak": float(imag_leak),
                "trace_formula_proxy_loss": float(trace_loss),
                "anti_poisson_rigidity_loss": float(anti_loss),
                "null_margin_loss": float(null_margin_loss),
                "selfadjoint_loss": float(0.0),
                "support_overlap_fraction": float(support),
                "active_window_count": int(active_ct),
                "poisson_like_fraction": float(pois_frac),
                "null_margin": float(null_margin),
                "beats_random": bool(beats_random),
                "beats_rejected": bool(beats_rejected),
                "beats_ablations": bool(beats_ablations),
                "is_primary": bool(is_primary),
                "is_random_baseline": bool(is_rand),
                "is_ablation": bool(is_ab),
                "is_rejected_word": bool(is_rej),
                "true_mode_ok": bool(true_mode_ok),
                "approximation_detected": bool(approximation_detected),
                "classification": str(classification),
                "gate_pass": bool(gate_pass),
            }
        )
        gate_rows.append(
            {
                "dim": int(d),
                "V_mode": str(vm),
                "word_group": str(wg),
                "G1_true_mode": bool(G1),
                "G2_support_overlap": bool(G2),
                "G3_argument_pass": bool(G3),
                "G4_residue_pass": bool(G4),
                "G5_not_poisson_like": bool(G5),
                "G6_null_margin_pass": bool(G6),
                "G7_beats_random": bool(G7),
                "G8_beats_rejected": bool(G8),
                "G9_beats_ablations": bool(G9),
                "G10_not_artifact": bool(G10),
                "G11_selfadjoint_status": str(selfadj_status),
                "gate_pass": bool(gate_pass),
                "classification": str(classification),
                "J_v14": float(J),
                "rank_by_J_v14": int(rank),
            }
        )

    df_scores = pd.DataFrame(cand_rows)
    if not df_scores.empty:
        df_scores = df_scores.sort_values(["dim", "J_v14", "word_group"], ascending=[True, True, True], na_position="last")

    df_gate = pd.DataFrame(gate_rows)
    if not df_gate.empty:
        df_gate = df_gate.sort_values(["dim", "rank_by_J_v14", "word_group"], ascending=[True, True, True], na_position="last")

    df_arg = pd.DataFrame(arg_rows)
    if not df_arg.empty:
        df_arg = df_arg.sort_values(["dim", "V_mode", "word_group", "target_group", "seed", "window_a"], ascending=True)

    df_res = pd.DataFrame(res_rows)
    if not df_res.empty:
        df_res = df_res.sort_values(["dim", "V_mode", "word_group", "target_group", "seed", "window_a"], ascending=True)

    df_trace = pd.DataFrame(trace_rows)
    if not df_trace.empty:
        df_trace = df_trace.sort_values(["dim", "V_mode", "word_group", "target_group", "seed", "window_a", "sigma"], ascending=True)

    df_nv = pd.DataFrame(nv_rows)
    if not df_nv.empty:
        df_nv = df_nv.sort_values(["dim", "V_mode", "word_group", "target_group", "seed", "source", "L"], ascending=True)

    df_anti = pd.DataFrame(anti_rows)
    if not df_anti.empty:
        df_anti = df_anti.sort_values(["dim", "V_mode", "word_group", "target_group", "seed"], ascending=True)

    # Best-by-dim
    best_rows: List[Dict[str, Any]] = []
    for d in dims_keep:
        sub = df_scores[df_scores["dim"].astype(int) == int(d)]
        prim = sub[(sub["V_mode"] == primary_vm) & (sub["word_group"] == primary_wg)]
        prim_J = float(prim["J_v14"].iloc[0]) if not prim.empty else float("nan")
        prim_rank = int(prim["rank_by_J_v14"].iloc[0]) if not prim.empty else 999999
        prim_cls = str(prim["classification"].iloc[0]) if not prim.empty else "FAIL_MISSING_LEVELS"

        best = sub.sort_values(["J_v14"], ascending=True, na_position="last").iloc[0] if not sub.empty else None
        best_vm = str(best["V_mode"]) if best is not None else ""
        best_wg = str(best["word_group"]) if best is not None else ""
        best_J = float(best["J_v14"]) if best is not None else float("nan")
        best_cls = str(best["classification"]) if best is not None else "FAIL_MISSING_LEVELS"

        should_enter_v14_1 = bool(best is not None and best_cls in ("PASS", "PROVISIONAL_SELFADJOINT_UNKNOWN"))
        best_rows.append(
            {
                "dim": int(d),
                "best_V_mode": best_vm,
                "best_word_group": best_wg,
                "best_J_v14": float(best_J),
                "best_classification": str(best_cls),
                "primary_J_v14": float(prim_J),
                "primary_rank": int(prim_rank),
                "primary_classification": str(prim_cls),
                "should_enter_v14_1": bool(should_enter_v14_1),
            }
        )
    df_best = pd.DataFrame(best_rows).sort_values(["dim"], ascending=True)

    # Warnings file
    (out_dir / "v14_warnings.txt").write_text("\n".join(warns) + ("\n" if warns else ""), encoding="utf-8")

    # Ensure all required CSV outputs exist (even empty)
    def ensure_csv(df: "pd.DataFrame", path: Path, cols: Sequence[str]) -> None:
        if df is None or df.empty:
            df0 = pd.DataFrame(columns=list(cols))
            df_to_csv(path, df0)
        else:
            # ensure cols exist
            for c in cols:
                if c not in df.columns:
                    df[c] = np.nan
            df_to_csv(path, df[list(cols)])

    ensure_csv(
        df_scores,
        out_dir / "v14_candidate_scores.csv",
        [
            "dim",
            "V_mode",
            "word_group",
            "J_v14",
            "rank_by_J_v14",
            "number_variance_curve_loss",
            "argument_count_loss",
            "residue_count_loss",
            "residue_imag_leak",
            "trace_formula_proxy_loss",
            "anti_poisson_rigidity_loss",
            "null_margin_loss",
            "selfadjoint_loss",
            "support_overlap_fraction",
            "active_window_count",
            "poisson_like_fraction",
            "null_margin",
            "beats_random",
            "beats_rejected",
            "beats_ablations",
            "is_primary",
            "is_random_baseline",
            "is_ablation",
            "is_rejected_word",
            "true_mode_ok",
            "approximation_detected",
            "classification",
            "gate_pass",
        ],
    )
    ensure_csv(
        df_arg,
        out_dir / "v14_argument_counts.csv",
        [
            "dim",
            "V_mode",
            "word_group",
            "target_group",
            "seed",
            "window_a",
            "window_b",
            "N_operator_direct",
            "N_target_direct",
            "N_operator_residue",
            "N_target_residue",
            "N_error_direct",
            "N_error_residue",
            "N_error_direct_norm",
            "N_error_residue_norm",
            "active_window",
        ],
    )
    ensure_csv(
        df_res,
        out_dir / "v14_residue_scores.csv",
        [
            "dim",
            "V_mode",
            "word_group",
            "target_group",
            "seed",
            "window_a",
            "window_b",
            "I_operator_real",
            "I_operator_imag",
            "I_target_real",
            "I_target_imag",
            "residue_count_error",
            "residue_imag_leak",
        ],
    )
    ensure_csv(
        df_trace,
        out_dir / "v14_trace_proxy.csv",
        [
            "dim",
            "V_mode",
            "word_group",
            "target_group",
            "seed",
            "window_a",
            "window_b",
            "center",
            "sigma",
            "S_operator",
            "S_target",
            "trace_error_norm",
        ],
    )
    ensure_csv(
        df_nv,
        out_dir / "v14_number_variance_curves.csv",
        ["dim", "V_mode", "word_group", "target_group", "seed", "source", "L", "Sigma2"],
    )
    ensure_csv(
        df_anti,
        out_dir / "v14_antipoisson_summary.csv",
        ["dim", "V_mode", "word_group", "target_group", "seed", "slope_long", "intercept_long", "poisson_like_flag", "anti_poisson_rigidity_loss"],
    )
    ensure_csv(
        df_null,
        out_dir / "v14_null_margins.csv",
        [
            "dim",
            "V_mode",
            "word_group",
            "J_candidate",
            "best_random_J",
            "rejected_word_J",
            "best_ablation_J",
            "null_margin",
            "beats_random",
            "beats_rejected",
            "beats_ablations",
        ],
    )
    ensure_csv(
        df_gate,
        out_dir / "v14_gate_summary.csv",
        [
            "dim",
            "V_mode",
            "word_group",
            "G1_true_mode",
            "G2_support_overlap",
            "G3_argument_pass",
            "G4_residue_pass",
            "G5_not_poisson_like",
            "G6_null_margin_pass",
            "G7_beats_random",
            "G8_beats_rejected",
            "G9_beats_ablations",
            "G10_not_artifact",
            "G11_selfadjoint_status",
            "gate_pass",
            "classification",
            "J_v14",
            "rank_by_J_v14",
        ],
    )
    ensure_csv(
        df_best,
        out_dir / "v14_best_by_dim.csv",
        [
            "dim",
            "best_V_mode",
            "best_word_group",
            "best_J_v14",
            "best_classification",
            "primary_J_v14",
            "primary_rank",
            "primary_classification",
            "should_enter_v14_1",
        ],
    )

    # Results JSON
    any_pass = bool((not df_scores.empty) and df_scores["classification"].astype(str).isin(["PASS"]).any())
    any_provisional = bool((not df_scores.empty) and df_scores["classification"].astype(str).isin(["PROVISIONAL_SELFADJOINT_UNKNOWN"]).any())
    payload: Dict[str, Any] = {
        "warning": "Computational evidence only; not a proof of RH.",
        "status": "V14 residue-guided operator search",
        "config": {
            "dims": dims_keep,
            "primary_word_group": primary_wg,
            "primary_v_mode": primary_vm,
            "windows": {
                "window_min": float(args.window_min),
                "window_max": float(args.window_max),
                "window_size": float(args.window_size),
                "window_stride": float(args.window_stride),
                "n_windows": int(len(windows)),
            },
            "nv_curve": {"L_min": float(args.L_min), "L_max": float(args.L_max), "n_L": int(args.n_L)},
            "eta": float(args.eta),
            "n_contour_points": int(args.n_contour_points),
            "top_k_candidates": int(args.top_k_candidates),
            "lambdas": {
                "lambda_nv": float(args.lambda_nv),
                "lambda_residue": float(args.lambda_residue),
                "lambda_argument": float(args.lambda_argument),
                "lambda_trace": float(args.lambda_trace),
                "lambda_antipoisson": float(args.lambda_antipoisson),
                "lambda_null_margin": float(args.lambda_null_margin),
                "lambda_selfadjoint": float(args.lambda_selfadjoint),
            },
            "thresholds": {
                "support_overlap_min": float(args.support_overlap_min),
                "active_error_margin": float(args.active_error_margin),
                "residue_error_margin": float(args.residue_error_margin),
                "poisson_like_max": float(args.poisson_like_max),
                "null_margin_min": float(args.null_margin_min),
            },
        },
        "inputs": {
            "true_levels_csv": str(_resolve(args.true_levels_csv).resolve()),
            "v13o10_candidate_ranking": str(_resolve(args.v13o10_candidate_ranking).resolve()),
            "v13o12_dir": str(v12_dir.resolve()) if v12_dir and v12_dir.is_dir() else None,
            "v13o13_dir": str(v13_dir.resolve()) if v13_dir and v13_dir.is_dir() else None,
            "v13o14_dir": str(v14_dir.resolve()) if v14_dir and v14_dir.is_dir() else None,
            "zeros_csv": str(_resolve(args.zeros_csv).resolve()),
        },
        "outputs": {
            "out_dir": str(out_dir.resolve()),
            "v14_candidate_scores.csv": "v14_candidate_scores.csv",
            "v14_gate_summary.csv": "v14_gate_summary.csv",
            "v14_best_by_dim.csv": "v14_best_by_dim.csv",
        },
        "summary": {
            "approximation_detected_any": bool(approx_detected_any),
            "any_pass": bool(any_pass),
            "any_provisional": bool(any_provisional),
        },
        "best_by_dim": df_best.to_dict(orient="records") if not df_best.empty else [],
        "warnings": warns,
        "runtime_s": float(time.perf_counter() - t0),
        "zeros_loaded_n": int(zeros_raw.size),
        "python_pid": os.getpid(),
    }
    (out_dir / "v14_results.json").write_text(json.dumps(json_sanitize(payload), indent=2, allow_nan=False), encoding="utf-8")

    # Report
    should_enter_v14_1 = bool(any_pass or any_provisional)
    md: List[str] = []
    md.append("# V14 Residue-Guided Hilbert–Pólya Operator Search\n\n")
    md.append("> **Computational evidence only; not a proof of RH.**\n\n")
    md.append("## Why V14 follows V13O.14\n\n")
    md.append(
        "V13O.14 indicated transport calibration is not zeta-specific evidence (poissonization not fixed, support mismatch, null controls competitive). "
        "V14 therefore ranks candidates by residue/argument-principle/trace/NV/anti-Poisson objectives with explicit null margins.\n\n"
    )
    md.append("## Mathematical idea (proxy)\n\n")
    md.append("- Resolvent: \\(R_{\\lambda}(z)=\\sum_j 1/(z-\\lambda_j)\\)\n")
    md.append("- Proxy count: \\(N([a,b])\\approx (1/2\\pi i)\\oint R(z)\\,dz\\) via rectangle contour + trapezoid.\n\n")
    md.append("## Objective\n\n")
    md.append("Total score (lower is better):\n\n")
    md.append(
        "\\[\n"
        "J_{v14} = \\lambda_{nv} L_{nv} + \\lambda_{res} L_{res} + \\lambda_{arg} L_{arg} + \\lambda_{trace} L_{trace} + "
        "\\lambda_{ap} L_{antiP} + \\lambda_{null} L_{null} + \\lambda_{sa} L_{sa}\n"
        "\\]\n\n"
    )
    md.append("All medians are over **active windows only**.\n\n")
    md.append("## Gate definition\n\n")
    md.append("- True mode required (no approximation detected)\n")
    md.append("- Support overlap >= threshold\n")
    md.append("- Argument loss <= margin; residue loss <= margin\n")
    md.append("- Not Poisson-like\n")
    md.append("- Null margin >= threshold AND beats random/rejected/ablations\n")
    md.append("- Not artifact (not random baseline, not ablation-only)\n")
    md.append("- Self-adjointness: if unavailable => **PROVISIONAL** only\n\n")
    md.append("## Primary result\n\n")
    md.append(f"Primary: `V_mode={primary_vm}`, `word_group={primary_wg}`.\n\n")
    md.append("See `v14_candidate_scores.csv` and `v14_gate_summary.csv`.\n\n")
    md.append("## Best candidates by dimension\n\n")
    md.append("See `v14_best_by_dim.csv`.\n\n")
    md.append("## Null-control conclusion\n\n")
    md.append("See `v14_null_margins.csv`.\n\n")
    md.append("## Explicit answers\n\n")
    md.append(f"- Did V14 run in true mode? **{not approx_detected_any}**\n")
    md.append("- Did primary pass? Check `primary_classification` in `v14_best_by_dim.csv`.\n")
    md.append("- Did any non-artifact candidate pass? Check `classification` in `v14_candidate_scores.csv`.\n")
    md.append("- Did residue constraints improve over V13O.12? Compare `v14_residue_scores.csv` vs V13O.12 outputs.\n")
    md.append("- Did anti-Poisson rigidity improve? See `v14_antipoisson_summary.csv` and `poisson_like_fraction`.\n")
    md.append("- Did primary beat rejected_word_seed17? See `beats_rejected` in `v14_null_margins.csv`.\n")
    md.append(f"- Should proceed to V14.1? **{should_enter_v14_1}**\n\n")
    if not should_enter_v14_1:
        md.append("**Recommendation**: No candidate passes. Run V14.1 differentiable residue-guided re-training.\n\n")
    else:
        md.append("**Recommendation**: If only PROVISIONAL passes, verify self-adjointness before V14.1.\n\n")
    md.append("## Warnings\n\n")
    md.append("See `v14_warnings.txt`.\n\n")
    md.append("## Verification block\n\n```bash\n")
    md.append(f"OUT={str(out_dir)}\n")
    md.append('find "$OUT" -maxdepth 1 -type f | sort\n')
    md.append('column -s, -t < "$OUT"/v14_gate_summary.csv | head -120\n')
    md.append('column -s, -t < "$OUT"/v14_candidate_scores.csv | head -120\n')
    md.append('column -s, -t < "$OUT"/v14_best_by_dim.csv\n')
    md.append('column -s, -t < "$OUT"/v14_null_margins.csv | head -120\n')
    md.append('head -220 "$OUT"/v14_report.md\n')
    md.append("```\n")
    (out_dir / "v14_report.md").write_text("".join(md), encoding="utf-8")

    tex = (
        "\\documentclass[11pt]{article}\n\\usepackage[margin=1in]{geometry}\n\\begin{document}\n"
        "\\title{V14 Residue-Guided Hilbert--P\\'olya Operator Search}\n\\maketitle\n"
        "\\section*{Warning}\nComputational evidence only; not a proof of RH.\n\n"
        "\\section*{Summary}\n"
        + latex_escape(json.dumps({"any_pass": any_pass, "any_provisional": any_provisional, "should_enter_v14_1": should_enter_v14_1}, indent=2))
        + "\n\n\\end{document}\n"
    )
    (out_dir / "v14_report.tex").write_text(tex, encoding="utf-8")
    if _find_pdflatex() is None:
        warns.append("pdflatex not found; skipping v14_report.pdf")
    else:
        try_pdflatex(out_dir / "v14_report.tex", out_dir, "v14_report.pdf")

    # update warnings file (in case pdflatex missing appended)
    (out_dir / "v14_warnings.txt").write_text("\n".join(warns) + ("\n" if warns else ""), encoding="utf-8")

    print(f"[V14] Wrote outputs to {out_dir}", flush=True)


if __name__ == "__main__":
    main()

