#!/usr/bin/env python3
"""
V13O.9 — True unfolded spectra diagnostics (direct number variance curves).

Computational evidence only; not a proof of the Riemann Hypothesis.

CRITICAL RULE:
-------------
This script must NEVER silently use reconstructed/proxy/scalar-based spectra.
If explicit unfolded level arrays are unavailable from the provided inputs, it must:
  - approximation_mode=True
  - true_zeta_specificity_pass=False
  - classification=INVALID_APPROXIMATION_MODE
and still write all required output files + report.

True-mode data source:
----------------------
V13O.9 expects that explicit unfolded level arrays are present in at least one of the provided input CSVs
as rows with columns containing:
  dim, V_mode, word_group, target_group, seed, unfolded_level
Optionally level_index and source (operator/target/control) may exist; if missing, defaults are applied.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
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


def read_csv_robust(path: Path, *, name: str, tag: str = "v13o9") -> Optional["pd.DataFrame"]:
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


def safe_median(xs: Sequence[float]) -> float:
    arr = np.asarray([float(v) for v in xs if v is not None and math.isfinite(float(v))], dtype=np.float64)
    return float(np.median(arr)) if arr.size else float("nan")


def make_L_grid(L_min: float, L_max: float, n_L: int) -> np.ndarray:
    L_min = float(L_min)
    L_max = float(L_max)
    n_L = int(n_L)
    if not (math.isfinite(L_min) and math.isfinite(L_max)) or n_L < 2:
        return np.asarray([0.5, 1.0], dtype=np.float64)
    if L_max <= L_min:
        L_max = L_min + 1.0
    return np.linspace(L_min, L_max, n_L, dtype=np.float64)


def poisson_sigma2(L: np.ndarray) -> np.ndarray:
    L = np.asarray(L, dtype=np.float64)
    return np.maximum(L, 0.0)


def gue_sigma2_proxy(L: np.ndarray) -> np.ndarray:
    gamma = 0.5772156649015329
    eps = 1e-9
    Lp = np.maximum(np.asarray(L, dtype=np.float64), eps)
    return (1.0 / (math.pi**2)) * (np.log(2.0 * math.pi * Lp) + gamma + 1.0)


def number_variance_curve(unfolded_levels: np.ndarray, L_grid: np.ndarray) -> np.ndarray:
    """
    For unfolded levels x sorted:
      N_i(L) = #{ x_j in [x_i, x_i+L] }
      Sigma^2(L) = Var_i N_i(L)

    Efficient:
      right = searchsorted(x, x+L, side='right')
      counts = right - arange(n)
    """
    x = np.sort(np.asarray(unfolded_levels, dtype=np.float64).reshape(-1))
    x = x[np.isfinite(x)]
    n = int(x.size)
    out = np.full_like(np.asarray(L_grid, dtype=np.float64).reshape(-1), np.nan, dtype=np.float64)
    if n < 20:
        return out
    for i, L in enumerate(np.asarray(L_grid, dtype=np.float64).reshape(-1)):
        Lf = float(L)
        if not (math.isfinite(Lf) and Lf > 0.0):
            continue
        right = np.searchsorted(x, x + Lf, side="right")
        counts = (right - np.arange(n, dtype=np.int64)).astype(np.float64)
        if counts.size < 2:
            continue
        out[i] = float(np.var(counts))
    return out


def region_masks(L_grid: np.ndarray, tail_L_min: float) -> Dict[str, np.ndarray]:
    L = np.asarray(L_grid, dtype=np.float64).reshape(-1)
    return {
        "short": (L < 2.0),
        "mid": (L >= 2.0) & (L < float(tail_L_min)),
        "long": (L >= float(tail_L_min)),
    }


def mse(a: np.ndarray, b: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    if aa.size == 0 or bb.size == 0 or aa.size != bb.size:
        return float("nan")
    m = np.isfinite(aa) & np.isfinite(bb)
    if mask is not None:
        m = m & np.asarray(mask, dtype=bool).reshape(-1)
    if int(m.sum()) < 3:
        return float("nan")
    d = aa[m] - bb[m]
    return float(np.mean(d**2))


def failure_region(short_e: float, mid_e: float, long_e: float) -> str:
    vals = {"short": short_e, "mid": mid_e, "long": long_e}
    worst = "unknown"
    worst_v = -1.0
    for k, v in vals.items():
        if math.isfinite(v) and float(v) > worst_v:
            worst_v = float(v)
            worst = str(k)
    return worst


def poissonization_class(distance_to_poisson: float, distance_to_gue: float) -> str:
    if not (math.isfinite(distance_to_poisson) and math.isfinite(distance_to_gue)):
        return "UNSTABLE"
    dp = float(distance_to_poisson)
    dg = float(distance_to_gue)
    if dp <= 0.0 and dg <= 0.0:
        return "UNSTABLE"
    # close within 10%
    if abs(dp - dg) <= 0.1 * max(dp, dg, 1e-12):
        return "MIXED"
    return "POISSON_LIKE" if dp < dg else "GUE_LIKE"


def _extract_true_unfolded_levels_from_df(df: "pd.DataFrame") -> Optional["pd.DataFrame"]:
    """
    Try to interpret df as a true unfolded-level table.
    Requires at least: dim, V_mode, word_group, target_group, seed, unfolded_level.
    If 'source' is missing, infer:
      - target_group == real_zeta => operator+target ambiguous; default to operator for now
      - otherwise source='control'
    """
    need = {"dim", "V_mode", "word_group", "target_group", "seed", "unfolded_level"}
    cols = set(map(str, df.columns))
    if not need.issubset(cols):
        return None
    out = df.copy()
    out["dim"] = pd.to_numeric(out["dim"], errors="coerce").astype("Int64")
    for k in ("V_mode", "word_group", "target_group"):
        out[k] = out[k].astype(str).str.strip()
    out["seed"] = pd.to_numeric(out["seed"], errors="coerce").astype("Int64")
    out["unfolded_level"] = finite_series(out["unfolded_level"])
    if "level_index" not in out.columns:
        out["level_index"] = np.arange(len(out), dtype=int)
    else:
        out["level_index"] = pd.to_numeric(out["level_index"], errors="coerce").fillna(-1).astype(int)
    if "source" not in out.columns:
        # cannot reliably distinguish operator vs target without explicit marker
        out["source"] = "unknown"
    else:
        out["source"] = out["source"].astype(str).str.strip()
    out = out.dropna(subset=["dim", "V_mode", "word_group", "target_group", "seed", "unfolded_level"])
    return out


def load_true_levels_csv(path: Path, *, dims_keep: Sequence[int]) -> Tuple[Optional["pd.DataFrame"], List[str]]:
    """
    Load explicit unfolded levels from a user-provided CSV.

    Supports:
      - V13O.9 source-format:
        dim,V_mode,word_group,target_group,seed,source,level_index,unfolded_level
      - V13O.8 kind-format:
        dim,V_mode,word_group,target_group,seed,kind,level_index,raw_level,unfolded_level,window_note
        where kind in {operator,target} (and possibly others).
    """
    warns: List[str] = []
    if not path.is_file():
        return None, [f"true_levels_csv missing: {path}"]
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return None, [f"true_levels_csv read failed: {e!r}"]
    df.columns = [str(c).strip() for c in df.columns]
    if df.empty:
        return None, ["true_levels_csv is empty"]

    # kind -> source compatibility
    if "source" not in df.columns and "kind" in df.columns:
        df = df.copy()
        df["source"] = df["kind"].astype(str).str.strip()

    required = ["dim", "V_mode", "word_group", "target_group", "seed", "source", "level_index", "unfolded_level"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return None, [f"true_levels_csv missing required columns: {missing}"]

    out = df[required].copy()
    out["dim"] = pd.to_numeric(out["dim"], errors="coerce").astype("Int64")
    out["seed"] = pd.to_numeric(out["seed"], errors="coerce").astype("Int64")
    out["level_index"] = pd.to_numeric(out["level_index"], errors="coerce").astype("Int64")
    for k in ("V_mode", "word_group", "target_group", "source"):
        out[k] = out[k].astype(str).str.strip()
    out["unfolded_level"] = finite_series(out["unfolded_level"])
    out = out.dropna(subset=["dim", "seed", "level_index", "unfolded_level"])

    dims_keep_set = {int(d) for d in dims_keep}
    out = out[out["dim"].astype(int).isin(dims_keep_set)]

    if out.empty:
        return None, ["true_levels_csv has no rows after filtering dims / NaNs"]

    # normalize source labels (accept V13O.8 kind labels)
    out["source"] = out["source"].replace(
        {
            "kind": "kind",
            "target": "target",
            "operator": "operator",
            "control": "control",
        }
    )
    # if someone used V13O.8 kind with extra labels, keep them but warn
    bad = sorted({s for s in out["source"].astype(str).unique().tolist() if s not in ("operator", "target", "control")})
    if bad:
        warns.append(f"true_levels_csv contains nonstandard source labels: {bad} (allowed: operator/target/control)")

    out = out.sort_values(["dim", "V_mode", "word_group", "target_group", "seed", "source", "level_index"], ascending=True)
    return out, warns


def _choose_levels(
    df_levels: "pd.DataFrame",
    *,
    dim: int,
    V_mode: str,
    word_group: str,
    target_group: str,
    seed: int,
    source: str,
) -> np.ndarray:
    sub = df_levels[
        (df_levels["dim"] == int(dim))
        & (df_levels["V_mode"] == str(V_mode))
        & (df_levels["word_group"] == str(word_group))
        & (df_levels["target_group"] == str(target_group))
        & (df_levels["seed"] == int(seed))
    ]
    if "source" in sub.columns and str(source) != "any":
        sub2 = sub[sub["source"] == str(source)]
        if not sub2.empty:
            sub = sub2
    xs = sub["unfolded_level"].astype(float).to_numpy(dtype=np.float64)
    xs = xs[np.isfinite(xs)]
    return np.sort(xs)


def main() -> None:
    ap = argparse.ArgumentParser(description="V13O.9 true unfolded spectra diagnostics (computational only).")

    ap.add_argument("--true_levels_csv", type=str, default="")

    ap.add_argument("--v13o4_summary", type=str, default="runs/v13o4_zeta_specific_objective/v13o4_summary.csv")
    ap.add_argument("--v13o4_group_summary", type=str, default="runs/v13o4_zeta_specific_objective/v13o4_group_summary.csv")
    ap.add_argument("--v13o4_zeta_scores", type=str, default="runs/v13o4_zeta_specific_objective/v13o4_zeta_specific_scores.csv")
    ap.add_argument("--v13o4_pair_corr", type=str, default="runs/v13o4_zeta_specific_objective/v13o4_pair_corr_summary.csv")
    ap.add_argument("--v13o4_number_variance", type=str, default="runs/v13o4_zeta_specific_objective/v13o4_number_variance_summary.csv")
    ap.add_argument("--v13o4_staircase", type=str, default="runs/v13o4_zeta_specific_objective/v13o4_staircase_summary.csv")
    ap.add_argument("--v13o6_nv_scores", type=str, default="runs/v13o6_number_variance_stabilization_smoke/v13o6_nv_stabilized_scores.csv")

    ap.add_argument("--out_dir", type=str, default="runs/v13o9_true_unfolded_spectra")
    ap.add_argument("--dims", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--primary_word_group", type=str, default="primary_word_seed6")
    ap.add_argument("--primary_v_mode", type=str, default="full_V")

    ap.add_argument("--L_min", type=float, default=0.5)
    ap.add_argument("--L_max", type=float, default=12.0)
    ap.add_argument("--n_L", type=int, default=48)
    ap.add_argument("--tail_L_min", type=float, default=6.0)

    ap.add_argument("--n_jobs", type=int, default=1)
    ap.add_argument("--progress_every", type=int, default=1)

    args = ap.parse_args()

    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    L_grid = make_L_grid(args.L_min, args.L_max, args.n_L)
    masks = region_masks(L_grid, float(args.tail_L_min))
    ref_p = poisson_sigma2(L_grid)
    ref_g = gue_sigma2_proxy(L_grid)
    eps = 1e-12

    # Load the provided CSVs. None of these are guaranteed to contain unfolded levels.
    dfs_in: List[Tuple[str, Optional["pd.DataFrame"]]] = [
        ("v13o4_summary", read_csv_robust(_resolve(args.v13o4_summary), name="v13o4_summary")),
        ("v13o4_group_summary", read_csv_robust(_resolve(args.v13o4_group_summary), name="v13o4_group_summary")),
        ("v13o4_zeta_scores", read_csv_robust(_resolve(args.v13o4_zeta_scores), name="v13o4_zeta_scores")),
        ("v13o4_pair_corr", read_csv_robust(_resolve(args.v13o4_pair_corr), name="v13o4_pair_corr")),
        ("v13o4_number_variance", read_csv_robust(_resolve(args.v13o4_number_variance), name="v13o4_number_variance")),
        ("v13o4_staircase", read_csv_robust(_resolve(args.v13o4_staircase), name="v13o4_staircase")),
        ("v13o6_nv_scores", read_csv_robust(_resolve(args.v13o6_nv_scores), name="v13o6_nv_scores")),
    ]

    # Prefer explicit true_levels_csv if provided.
    df_levels: Optional["pd.DataFrame"] = None
    true_source_name: Optional[str] = None
    warnings: List[str] = []

    if str(args.true_levels_csv).strip():
        tl_path = _resolve(str(args.true_levels_csv).strip())
        df_tl, wtl = load_true_levels_csv(tl_path, dims_keep=[int(x) for x in args.dims])
        warnings.extend(wtl)
        if df_tl is not None and not df_tl.empty:
            df_levels = df_tl
            true_source_name = str(args.true_levels_csv).strip()
        else:
            # explicit request but invalid/empty -> force approximation_mode
            df_levels = None
            true_source_name = str(args.true_levels_csv).strip()

    # Otherwise, try to find explicit unfolded levels in any provided DataFrame.
    for name, df in dfs_in:
        if df_levels is not None:
            break
        if df is None:
            continue
        if "unfolded_level" in df.columns:
            cand = _extract_true_unfolded_levels_from_df(df)
            if cand is not None and not cand.empty:
                df_levels = cand
                true_source_name = name
                break

    approximation_mode = df_levels is None
    true_mode_requested = bool(df_levels is not None)

    # Prepare empty outputs (always written)
    df_unfolded_out = pd.DataFrame(
        columns=["dim", "V_mode", "word_group", "target_group", "seed", "source", "level_index", "unfolded_level"]
    )
    df_curves_out = pd.DataFrame(
        columns=["dim", "V_mode", "word_group", "target_group", "seed", "source", "L", "Sigma2"]
    )
    df_errors_out = pd.DataFrame(
        columns=["dim", "V_mode", "word_group", "target_group", "seed", "total_curve_error", "short_error", "mid_error", "long_error", "failure_region"]
    )
    df_pois_out = pd.DataFrame(
        columns=[
            "dim",
            "V_mode",
            "word_group",
            "target_group",
            "seed",
            "distance_to_poisson",
            "distance_to_gue",
            "poissonization_ratio",
            "classification",
        ]
    )
    df_gate_out = pd.DataFrame(
        columns=[
            "dim",
            "V_mode",
            "word_group",
            "T1_true_mode_enabled",
            "T2_unfolded_levels_available",
            "T3_primary_beats_shuffled",
            "T4_primary_beats_block_shuffle",
            "T5_primary_beats_reversed",
            "T6_primary_beats_GUE",
            "T7_primary_beats_Poisson",
            "T8_not_poisson_like",
            "T9_long_range_pass",
            "T10_primary_rank_top2",
            "true_zeta_specificity_pass",
            "classification",
            "primary_total_curve_error",
            "primary_failure_region",
            "primary_poissonization_classification",
            "primary_rank",
        ]
    )
    df_best_out = pd.DataFrame(
        columns=["dim", "best_V_mode", "best_word_group", "best_total_curve_error", "best_classification", "primary_total_curve_error", "primary_rank"]
    )

    if approximation_mode:
        warnings.append(
            "Explicit unfolded level arrays were NOT found (or were invalid/empty). "
            "V13O.9 refuses to compute 'true' diagnostics from reconstructed/proxy data."
        )

    # If true unfolded levels are available, compute diagnostics.
    if not approximation_mode and df_levels is not None:
        # Normalize key types
        df_unfolded_out = df_levels[
            ["dim", "V_mode", "word_group", "target_group", "seed", "source", "level_index", "unfolded_level"]
        ].copy()
        df_unfolded_out["dim"] = pd.to_numeric(df_unfolded_out["dim"], errors="coerce").astype("Int64")
        df_unfolded_out["seed"] = pd.to_numeric(df_unfolded_out["seed"], errors="coerce").astype("Int64")
        for k in ("V_mode", "word_group", "target_group", "source"):
            df_unfolded_out[k] = df_unfolded_out[k].astype(str).str.strip()
        df_unfolded_out["unfolded_level"] = finite_series(df_unfolded_out["unfolded_level"])
        df_unfolded_out = df_unfolded_out.dropna(subset=["dim", "V_mode", "word_group", "target_group", "seed", "unfolded_level"])
        df_unfolded_out = df_unfolded_out.sort_values(
            ["dim", "V_mode", "word_group", "target_group", "seed", "source", "level_index"], ascending=True
        )

        # Determine seeds present per group
        dims = [int(d) for d in args.dims]
        primary_vm = str(args.primary_v_mode)
        primary_wg = str(args.primary_word_group)

        groups = sorted(
            {
                (int(r.dim), str(r.V_mode), str(r.word_group), int(r.seed))
                for r in df_unfolded_out[["dim", "V_mode", "word_group", "seed"]].dropna().itertuples(index=False)
                if int(r.dim) in dims
            },
            key=lambda x: (x[0], x[1], x[2], x[3]),
        )

        # For diagnostics, require valid source labels.
        if (df_unfolded_out["source"].astype(str) == "unknown").any():
            approximation_mode = True
            true_mode_requested = False
            warnings.append("Found unfolded_level rows but missing valid 'source' (operator/target/control). Marking INVALID_APPROXIMATION_MODE.")
        else:
            # Build per-job computations for primary groups only (dim,vm,wg,seed,target_group)
            jobs: List[Tuple[int, str, str, int, str]] = []
            for (d, vm, wg, seed) in groups:
                for tg in ["real_zeta"] + CONTROL_TARGET_GROUPS:
                    jobs.append((d, vm, wg, seed, tg))
            jobs = sorted(jobs, key=lambda x: (x[0], x[1], x[2], x[3], x[4]))

            prog = max(1, int(args.progress_every))
            exec_t0 = time.perf_counter()

            curves_rows: List[Dict[str, Any]] = []
            errors_rows: List[Dict[str, Any]] = []
            pois_rows: List[Dict[str, Any]] = []

            def compute_one(job: Tuple[int, str, str, int, str]) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[str]]:
                d, vm, wg, seed, tg = job
                x_op = _choose_levels(df_unfolded_out, dim=d, V_mode=vm, word_group=wg, target_group=tg, seed=seed, source="operator")
                x_tg = _choose_levels(df_unfolded_out, dim=d, V_mode=vm, word_group=wg, target_group=tg, seed=seed, source="target")
                x_ctrl = _choose_levels(df_unfolded_out, dim=d, V_mode=vm, word_group=wg, target_group=tg, seed=seed, source="control")

                # Require both operator+target arrays to exist; otherwise skip with warning (no crash)
                if x_op.size == 0 or x_tg.size == 0:
                    return [], None, None, f"missing_levels dim={d} V_mode={vm} word_group={wg} target_group={tg} seed={seed} (need operator+target)"

                # We require operator+target for curve errors; for controls, operator/target same semantics.
                if x_op.size < 20 or x_tg.size < 20 or (not np.isfinite(x_op).all()) or (not np.isfinite(x_tg).all()):
                    c_op = np.full_like(L_grid, np.nan, dtype=np.float64)
                    c_tg = np.full_like(L_grid, np.nan, dtype=np.float64)
                    unstable = True
                else:
                    c_op = number_variance_curve(x_op, L_grid)
                    c_tg = number_variance_curve(x_tg, L_grid)
                    unstable = bool((not np.isfinite(c_op).any()) or (not np.isfinite(c_tg).any()))

                # Write curves for operator/target/control (if control levels exist)
                rows: List[Dict[str, Any]] = []
                for L, s2 in zip(L_grid.tolist(), c_op.tolist()):
                    rows.append({"dim": d, "V_mode": vm, "word_group": wg, "target_group": tg, "seed": seed, "source": "operator", "L": float(L), "Sigma2": float(s2)})
                for L, s2 in zip(L_grid.tolist(), c_tg.tolist()):
                    rows.append({"dim": d, "V_mode": vm, "word_group": wg, "target_group": tg, "seed": seed, "source": "target", "L": float(L), "Sigma2": float(s2)})
                if x_ctrl.size >= 20 and np.isfinite(x_ctrl).all():
                    c_ctrl = number_variance_curve(x_ctrl, L_grid)
                    for L, s2 in zip(L_grid.tolist(), c_ctrl.tolist()):
                        rows.append({"dim": d, "V_mode": vm, "word_group": wg, "target_group": tg, "seed": seed, "source": "control", "L": float(L), "Sigma2": float(s2)})

                # Errors vs target
                total = mse(c_op, c_tg)
                sh = mse(c_op, c_tg, masks["short"])
                mi = mse(c_op, c_tg, masks["mid"])
                lo = mse(c_op, c_tg, masks["long"])
                fr = failure_region(sh, mi, lo)

                err_row = {
                    "dim": d,
                    "V_mode": vm,
                    "word_group": wg,
                    "target_group": tg,
                    "seed": seed,
                    "total_curve_error": float(total),
                    "short_error": float(sh),
                    "mid_error": float(mi),
                    "long_error": float(lo),
                    "failure_region": str(fr),
                }

                # Poissonization vs references for operator curve
                dp = mse(c_op, ref_p) if not unstable else float("nan")
                dg = mse(c_op, ref_g) if not unstable else float("nan")
                ratio = float(dg / (dp + eps)) if (math.isfinite(dp) and math.isfinite(dg)) else float("nan")
                cls = poissonization_class(float(dp), float(dg))
                pois_row = {
                    "dim": d,
                    "V_mode": vm,
                    "word_group": wg,
                    "target_group": tg,
                    "seed": seed,
                    "distance_to_poisson": float(dp),
                    "distance_to_gue": float(dg),
                    "poissonization_ratio": float(ratio),
                    "classification": str(cls),
                }
                return rows, err_row, pois_row, None

            def log_done(i_done: int, total: int) -> None:
                if i_done == 1 or i_done == total or i_done % prog == 0:
                    elapsed = time.perf_counter() - exec_t0
                    avg = elapsed / max(i_done, 1)
                    eta = avg * max(total - i_done, 0)
                    print(f"[V13O.9] completed {i_done}/{total} elapsed={elapsed:.1f}s eta={format_seconds(eta)}", flush=True)

            n_jobs = max(1, int(args.n_jobs))
            if n_jobs == 1:
                for i, jb in enumerate(jobs, start=1):
                    rows, er, pr, warn = compute_one(jb)
                    if warn:
                        warnings.append(warn)
                    curves_rows.extend(rows)
                    if er is not None:
                        errors_rows.append(er)
                    if pr is not None:
                        pois_rows.append(pr)
                    log_done(i, len(jobs))
            else:
                # ThreadPool (safe) — computation is mostly numpy/searchsorted; avoids OS semaphore sysconf restrictions.
                with cf.ThreadPoolExecutor(max_workers=n_jobs) as ex:
                    futs = [ex.submit(compute_one, jb) for jb in jobs]
                    done = 0
                    for fut in cf.as_completed(futs):
                        rows, er, pr, warn = fut.result()
                        if warn:
                            warnings.append(warn)
                        curves_rows.extend(rows)
                        if er is not None:
                            errors_rows.append(er)
                        if pr is not None:
                            pois_rows.append(pr)
                        done += 1
                        log_done(done, len(jobs))

            df_curves_out = pd.DataFrame(curves_rows)
            df_errors_out = pd.DataFrame(errors_rows)
            df_pois_out = pd.DataFrame(pois_rows)

            # deterministic sort
            for df in (df_curves_out, df_errors_out, df_pois_out):
                if df.empty:
                    continue
                df["dim"] = pd.to_numeric(df["dim"], errors="coerce").astype("Int64")
                df["seed"] = pd.to_numeric(df["seed"], errors="coerce").astype("Int64")
                for k in ("V_mode", "word_group", "target_group", "source", "classification"):
                    if k in df.columns:
                        df[k] = df[k].astype(str).str.strip()
            if not df_curves_out.empty:
                df_curves_out = df_curves_out.sort_values(["dim", "V_mode", "word_group", "target_group", "seed", "source", "L"])
            if not df_errors_out.empty:
                df_errors_out = df_errors_out.sort_values(["dim", "V_mode", "word_group", "target_group", "seed"])
            if not df_pois_out.empty:
                df_pois_out = df_pois_out.sort_values(["dim", "V_mode", "word_group", "target_group", "seed"])

            # Gate summary for primary: compare primary total error vs median controls by control group families
            gate_rows: List[Dict[str, Any]] = []
            best_rows: List[Dict[str, Any]] = []
            for dim in [int(d) for d in args.dims]:
                # primary aggregate over seeds (median)
                prim = df_errors_out[
                    (df_errors_out["dim"] == dim)
                    & (df_errors_out["V_mode"] == primary_vm)
                    & (df_errors_out["word_group"] == primary_wg)
                    & (df_errors_out["target_group"] == "real_zeta")
                ]
                prim_total = safe_median(prim["total_curve_error"].astype(float).tolist()) if not prim.empty else float("nan")
                prim_long = safe_median(prim["long_error"].astype(float).tolist()) if not prim.empty else float("nan")
                prim_fail = str(prim["failure_region"].astype(str).iloc[0]) if not prim.empty else "unknown"

                ppois = df_pois_out[
                    (df_pois_out["dim"] == dim)
                    & (df_pois_out["V_mode"] == primary_vm)
                    & (df_pois_out["word_group"] == primary_wg)
                    & (df_pois_out["target_group"] == "real_zeta")
                ]
                prim_pcls = str(ppois["classification"].astype(str).iloc[0]) if not ppois.empty else "UNSTABLE"

                # candidates (rank by total curve error) for this dim, real_zeta only
                cand = df_errors_out[(df_errors_out["dim"] == dim) & (df_errors_out["target_group"] == "real_zeta")].copy()
                cand = cand.dropna(subset=["total_curve_error"])
                cand = cand.sort_values(["total_curve_error"], ascending=True, na_position="last")
                primary_rank = -1
                if not cand.empty:
                    cand2 = cand.reset_index(drop=True).copy()
                    cand2["_rank"] = np.arange(1, len(cand2) + 1, dtype=int)
                    pm = (cand2["V_mode"] == primary_vm) & (cand2["word_group"] == primary_wg)
                    if pm.any():
                        primary_rank = int(cand2.loc[pm, "_rank"].iloc[0])
                    best = cand2.iloc[0]
                    best_rows.append(
                        {
                            "dim": int(dim),
                            "best_V_mode": str(best["V_mode"]),
                            "best_word_group": str(best["word_group"]),
                            "best_total_curve_error": float(best["total_curve_error"]),
                            "best_classification": "AVAILABLE",
                            "primary_total_curve_error": float(prim_total),
                            "primary_rank": int(primary_rank),
                        }
                    )

                # control medians for comparisons (use total_curve_error from primary operator but against control target_groups)
                def ctrl_med(tg_name: str) -> float:
                    sub = df_errors_out[
                        (df_errors_out["dim"] == dim)
                        & (df_errors_out["V_mode"] == primary_vm)
                        & (df_errors_out["word_group"] == primary_wg)
                        & (df_errors_out["target_group"] == tg_name)
                    ]
                    return safe_median(sub["total_curve_error"].astype(float).tolist()) if not sub.empty else float("nan")

                med_shuf = ctrl_med("shuffled_zeta")
                med_block = safe_median([ctrl_med("block_shuffled_zeta_block4"), ctrl_med("block_shuffled_zeta_block8")])
                med_rev = ctrl_med("reversed_zeta")
                med_gue = ctrl_med("GUE_synthetic")
                med_poi = ctrl_med("Poisson_synthetic")

                # long-range control median for T9
                long_ctrl = df_errors_out[
                    (df_errors_out["dim"] == dim)
                    & (df_errors_out["V_mode"] == primary_vm)
                    & (df_errors_out["word_group"] == primary_wg)
                    & (df_errors_out["target_group"].isin(CONTROL_TARGET_GROUPS))
                ]
                med_long_ctrl = safe_median(long_ctrl["long_error"].astype(float).tolist()) if not long_ctrl.empty else float("nan")

                T1 = True
                T2 = bool(math.isfinite(prim_total))
                T3 = bool(math.isfinite(prim_total) and math.isfinite(med_shuf) and prim_total < med_shuf)
                T4 = bool(math.isfinite(prim_total) and math.isfinite(med_block) and prim_total < med_block)
                T5 = bool(math.isfinite(prim_total) and math.isfinite(med_rev) and prim_total < med_rev)
                T6 = bool(math.isfinite(prim_total) and math.isfinite(med_gue) and prim_total < med_gue)
                T7 = bool(math.isfinite(prim_total) and math.isfinite(med_poi) and prim_total < med_poi)
                T8 = bool(str(prim_pcls) != "POISSON_LIKE")
                T9 = bool(math.isfinite(prim_long) and math.isfinite(med_long_ctrl) and prim_long <= med_long_ctrl)
                T10 = bool(primary_rank != -1 and primary_rank <= 2)

                true_pass = bool(T1 and T2 and T3 and T4 and T5 and T6 and T7 and T8 and T9 and T10)
                gate_cls = "PASS" if true_pass else "FAIL"

                gate_rows.append(
                    {
                        "dim": int(dim),
                        "V_mode": primary_vm,
                        "word_group": primary_wg,
                        "T1_true_mode_enabled": bool(T1),
                        "T2_unfolded_levels_available": bool(T2),
                        "T3_primary_beats_shuffled": bool(T3),
                        "T4_primary_beats_block_shuffle": bool(T4),
                        "T5_primary_beats_reversed": bool(T5),
                        "T6_primary_beats_GUE": bool(T6),
                        "T7_primary_beats_Poisson": bool(T7),
                        "T8_not_poisson_like": bool(T8),
                        "T9_long_range_pass": bool(T9),
                        "T10_primary_rank_top2": bool(T10),
                        "true_zeta_specificity_pass": bool(true_pass),
                        "classification": str(gate_cls),
                        "primary_total_curve_error": float(prim_total),
                        "primary_failure_region": str(prim_fail),
                        "primary_poissonization_classification": str(prim_pcls),
                        "primary_rank": int(primary_rank),
                    }
                )

            df_gate_out = pd.DataFrame(gate_rows)
            df_best_out = pd.DataFrame(best_rows)

    # Enforce critical invalid behavior if approximation_mode is on.
    if approximation_mode:
        df_gate_out = pd.DataFrame(
            [
                {
                    "dim": int(d),
                    "V_mode": str(args.primary_v_mode),
                    "word_group": str(args.primary_word_group),
                    "T1_true_mode_enabled": False,
                    "T2_unfolded_levels_available": False,
                    "T3_primary_beats_shuffled": False,
                    "T4_primary_beats_block_shuffle": False,
                    "T5_primary_beats_reversed": False,
                    "T6_primary_beats_GUE": False,
                    "T7_primary_beats_Poisson": False,
                    "T8_not_poisson_like": False,
                    "T9_long_range_pass": False,
                    "T10_primary_rank_top2": False,
                    "true_zeta_specificity_pass": False,
                    "classification": "INVALID_APPROXIMATION_MODE",
                    "primary_total_curve_error": float("nan"),
                    "primary_failure_region": "unknown",
                    "primary_poissonization_classification": "UNSTABLE",
                    "primary_rank": -1,
                }
                for d in [int(x) for x in args.dims]
            ]
        )
        df_best_out = pd.DataFrame(
            [
                {
                    "dim": int(d),
                    "best_V_mode": "",
                    "best_word_group": "",
                    "best_total_curve_error": float("nan"),
                    "best_classification": "INVALID_APPROXIMATION_MODE",
                    "primary_total_curve_error": float("nan"),
                    "primary_rank": -1,
                }
                for d in [int(x) for x in args.dims]
            ]
        )

    # Write outputs
    (out_dir / "v13o9_unfolded_levels.csv").write_text(df_unfolded_out.to_csv(index=False), encoding="utf-8")
    (out_dir / "v13o9_number_variance_curves.csv").write_text(df_curves_out.to_csv(index=False), encoding="utf-8")
    (out_dir / "v13o9_nv_curve_errors.csv").write_text(df_errors_out.to_csv(index=False), encoding="utf-8")
    (out_dir / "v13o9_poissonization_diagnostics.csv").write_text(df_pois_out.to_csv(index=False), encoding="utf-8")
    (out_dir / "v13o9_gate_summary.csv").write_text(df_gate_out.to_csv(index=False), encoding="utf-8")
    (out_dir / "v13o9_best_by_dim.csv").write_text(df_best_out.to_csv(index=False), encoding="utf-8")

    # Decision rule for V13P0
    proceed_v13p0 = False
    if not approximation_mode and not df_gate_out.empty and "true_zeta_specificity_pass" in df_gate_out.columns:
        proceed_v13p0 = bool(df_gate_out["true_zeta_specificity_pass"].astype(bool).any())

    # Compose results JSON
    payload = {
        "warning": "Computational evidence only; not a proof of the Riemann Hypothesis.",
        "status": "V13O.9 true unfolded spectra diagnostics",
        "approximation_mode": bool(approximation_mode),
        "true_mode_requested": bool(true_mode_requested),
        "true_levels_source": true_source_name or "",
        "n_unfolded_rows_exported": int(len(df_unfolded_out)),
        "n_curve_rows_exported": int(len(df_curves_out)),
        "dims": [int(x) for x in args.dims],
        "L_grid": [float(x) for x in L_grid.tolist()],
        "tail_L_min": float(args.tail_L_min),
        "primary": {"word_group": str(args.primary_word_group), "V_mode": str(args.primary_v_mode)},
        "should_proceed_to_v13p0": bool((not approximation_mode) and proceed_v13p0),
        "warnings": warnings,
        "inputs": {k: str(_resolve(getattr(args, k)).resolve()) for k in vars(args) if k.startswith("v13o")},
        "outputs": {
            "results_json": str((out_dir / "v13o9_results.json").resolve()),
            "unfolded_levels_csv": str((out_dir / "v13o9_unfolded_levels.csv").resolve()),
            "number_variance_curves_csv": str((out_dir / "v13o9_number_variance_curves.csv").resolve()),
            "nv_curve_errors_csv": str((out_dir / "v13o9_nv_curve_errors.csv").resolve()),
            "poissonization_diagnostics_csv": str((out_dir / "v13o9_poissonization_diagnostics.csv").resolve()),
            "gate_summary_csv": str((out_dir / "v13o9_gate_summary.csv").resolve()),
            "best_by_dim_csv": str((out_dir / "v13o9_best_by_dim.csv").resolve()),
            "report_md": str((out_dir / "v13o9_report.md").resolve()),
            "report_tex": str((out_dir / "v13o9_report.tex").resolve()),
            "report_pdf": str((out_dir / "v13o9_report.pdf").resolve()),
        },
        "runtime_s": float(time.perf_counter() - t0),
        "python_pid": os.getpid(),
    }
    (out_dir / "v13o9_results.json").write_text(json.dumps(json_sanitize(payload), indent=2, allow_nan=False), encoding="utf-8")

    # Report (md)
    md: List[str] = []
    md.append("# V13O.9 True unfolded spectra diagnostics\n\n")
    md.append("> **Computational evidence only; not a proof of the Riemann Hypothesis.**\n\n")
    md.append(f"- approximation_mode: **{payload['approximation_mode']}**\n")
    md.append(f"- true_mode_requested: **{payload['true_mode_requested']}**\n")
    md.append(f"- true_levels_source: `{payload['true_levels_source']}`\n")
    md.append(f"- unfolded rows exported: **{payload['n_unfolded_rows_exported']}**\n\n")
    md.append("## L-grid\n\n")
    md.append(f"- L_min={args.L_min}, L_max={args.L_max}, n_L={args.n_L}, tail_L_min={args.tail_L_min}\n\n")
    md.append("## Primary gate summary\n\n")
    md.append("See `v13o9_gate_summary.csv`.\n\n")
    md.append("## Best by dim\n\n")
    md.append("See `v13o9_best_by_dim.csv`.\n\n")
    md.append("## Failure mode summary\n\n")
    if warnings:
        md.append("Warnings:\n\n")
        for w in warnings:
            md.append(f"- {w}\n")
        md.append("\n")
    md.append("## Explicit answers\n\n")
    md.append(f"- Did V13O.9 run in true mode? **{not approximation_mode}**\n")
    md.append(f"- Are unfolded spectra available? **{payload['n_unfolded_rows_exported'] > 0 and (not approximation_mode)}**\n")
    # Primary poisson-like + failure region placeholders (from gate file if available)
    if not df_gate_out.empty:
        md.append(f"- Does primary still look Poisson-like? **{(df_gate_out['primary_poissonization_classification'].astype(str) == 'POISSON_LIKE').any()}**\n")
        md.append(f"- Is failure mostly short/mid/long range? **{df_gate_out['primary_failure_region'].astype(str).iloc[0]}**\n")
    else:
        md.append("- Does primary still look Poisson-like? **Unknown**\n")
        md.append("- Is failure mostly short/mid/long range? **Unknown**\n")
    # best dim
    best_dim = None
    if not df_best_out.empty and "best_total_curve_error" in df_best_out.columns:
        dfb = df_best_out.copy()
        dfb["best_total_curve_error"] = pd.to_numeric(dfb["best_total_curve_error"], errors="coerce")
        dfb = dfb.sort_values(["best_total_curve_error"], ascending=True, na_position="last")
        if not dfb.empty and math.isfinite(float(dfb["best_total_curve_error"].iloc[0])):
            best_dim = int(dfb["dim"].iloc[0])
    md.append(f"- Which dim is best? **{best_dim}**\n")
    md.append(f"- Should proceed to V13P0? **{payload['should_proceed_to_v13p0']}**\n\n")
    md.append("## Decision rule\n\n")
    md.append("Proceed to V13P0 only if `true_zeta_specificity_pass=True` for at least one dimension and `approximation_mode=False`.\n\n")
    md.append("## CLI examples\n\n")
    md.append("Smoke:\n\n```bash\n")
    md.append('OUT=runs/v13o9_true_unfolded_spectra_smoke\n\n')
    md.append("python3 scripts/run_v13o9_true_unfolded_spectra.py \\\n")
    md.append("  --v13o4_summary runs/v13o4_zeta_specific_objective/v13o4_summary.csv \\\n")
    md.append("  --v13o4_group_summary runs/v13o4_zeta_specific_objective/v13o4_group_summary.csv \\\n")
    md.append("  --v13o4_zeta_scores runs/v13o4_zeta_specific_objective/v13o4_zeta_specific_scores.csv \\\n")
    md.append("  --v13o4_pair_corr runs/v13o4_zeta_specific_objective/v13o4_pair_corr_summary.csv \\\n")
    md.append("  --v13o4_number_variance runs/v13o4_zeta_specific_objective/v13o4_number_variance_summary.csv \\\n")
    md.append("  --v13o4_staircase runs/v13o4_zeta_specific_objective/v13o4_staircase_summary.csv \\\n")
    md.append("  --v13o6_nv_scores runs/v13o6_number_variance_stabilization_smoke/v13o6_nv_stabilized_scores.csv \\\n")
    md.append('  --out_dir "$OUT" \\\n')
    md.append("  --dims 64 128 \\\n")
    md.append("  --primary_word_group primary_word_seed6 \\\n")
    md.append("  --primary_v_mode full_V \\\n")
    md.append("  --L_min 0.5 \\\n")
    md.append("  --L_max 12.0 \\\n")
    md.append("  --n_L 24 \\\n")
    md.append("  --tail_L_min 6.0 \\\n")
    md.append("  --n_jobs 4 \\\n")
    md.append("  --progress_every 1\n")
    md.append("```\n\n")
    md.append("Full:\n\n```bash\n")
    md.append('OUT=runs/v13o9_true_unfolded_spectra\n\n')
    md.append("caffeinate -dimsu python3 scripts/run_v13o9_true_unfolded_spectra.py \\\n")
    md.append("  --v13o4_summary runs/v13o4_zeta_specific_objective/v13o4_summary.csv \\\n")
    md.append("  --v13o4_group_summary runs/v13o4_zeta_specific_objective/v13o4_group_summary.csv \\\n")
    md.append("  --v13o4_zeta_scores runs/v13o4_zeta_specific_objective/v13o4_zeta_specific_scores.csv \\\n")
    md.append("  --v13o4_pair_corr runs/v13o4_zeta_specific_objective/v13o4_pair_corr_summary.csv \\\n")
    md.append("  --v13o4_number_variance runs/v13o4_zeta_specific_objective/v13o4_number_variance_summary.csv \\\n")
    md.append("  --v13o4_staircase runs/v13o4_zeta_specific_objective/v13o4_staircase_summary.csv \\\n")
    md.append("  --v13o6_nv_scores runs/v13o6_number_variance_stabilization_smoke/v13o6_nv_stabilized_scores.csv \\\n")
    md.append('  --out_dir "$OUT" \\\n')
    md.append("  --dims 64 128 256 \\\n")
    md.append("  --primary_word_group primary_word_seed6 \\\n")
    md.append("  --primary_v_mode full_V \\\n")
    md.append("  --L_min 0.5 \\\n")
    md.append("  --L_max 16.0 \\\n")
    md.append("  --n_L 64 \\\n")
    md.append("  --tail_L_min 6.0 \\\n")
    md.append("  --n_jobs 8 \\\n")
    md.append("  --progress_every 10\n")
    md.append("```\n\n")
    md.append("## Verification commands\n\n```bash\n")
    md.append("OUT=runs/v13o9_true_unfolded_spectra\n\n")
    md.append('echo "=== FILES ==="\nfind "$OUT" -maxdepth 1 -type f | sort\n\n')
    md.append('echo "=== GATE ==="\ncolumn -s, -t < "$OUT"/v13o9_gate_summary.csv | head -80\n\n')
    md.append('echo "=== PRIMARY NV CURVE ERRORS ==="\n')
    md.append("awk -F, '$2==\"full_V\" && $3==\"primary_word_seed6\"{print}' \\\n  \"$OUT\"/v13o9_nv_curve_errors.csv | head -80\n\n")
    md.append('echo "=== PRIMARY POISSONIZATION ==="\n')
    md.append("awk -F, '$2==\"full_V\" && $3==\"primary_word_seed6\"{print}' \\\n  \"$OUT\"/v13o9_poissonization_diagnostics.csv | head -80\n\n")
    md.append('echo "=== BEST BY DIM ==="\ncolumn -s, -t < "$OUT"/v13o9_best_by_dim.csv\n\n')
    md.append('echo "=== REPORT ==="\nhead -160 "$OUT"/v13o9_report.md\n')
    md.append("```\n")

    (out_dir / "v13o9_report.md").write_text("".join(md), encoding="utf-8")

    tex = (
        "\\documentclass[11pt]{article}\n\\usepackage[margin=1in]{geometry}\n\\begin{document}\n"
        "\\title{V13O.9 True unfolded spectra diagnostics}\n\\maketitle\n"
        "\\section*{Warning}\nComputational evidence only; not a proof of RH.\n\n"
        "\\section*{Mode}\n"
        + latex_escape(f"approximation_mode={approximation_mode}, true_levels_source={true_source_name or ''}.")
        + "\n\n\\section*{Decision}\n"
        + latex_escape(f"should_proceed_to_v13p0={payload['should_proceed_to_v13p0']}.")
        + "\n\n\\end{document}\n"
    )
    (out_dir / "v13o9_report.tex").write_text(tex, encoding="utf-8")

    if _find_pdflatex() is None:
        print("[v13o9] WARNING: pdflatex not found; skipping PDF.", flush=True)
    elif try_pdflatex(out_dir / "v13o9_report.tex", out_dir, "v13o9_report.pdf"):
        print(f"Wrote {out_dir / 'v13o9_report.pdf'}", flush=True)
    else:
        print("[v13o9] WARNING: pdflatex failed or did not produce v13o9_report.pdf.", flush=True)

    print(f"[v13o9] Wrote {out_dir / 'v13o9_results.json'}", flush=True)


if __name__ == "__main__":
    main()

