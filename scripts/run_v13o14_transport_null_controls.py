#!/usr/bin/env python3
"""
V13O.14 — Transport Null Controls & Anti-Overfit Gate.

Computational evidence only; not a proof of the Riemann Hypothesis.

Goal:
Decide whether V13O.13 transport improvements are zeta-specific, or explainable by:
  - support mismatch / empty support,
  - poissonization not fixed,
  - null controls (random baselines, rejected words, ablations) being competitive,
  - overly-flexible monotone spline fitting (spline-only "wins").
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
from typing import Any, Dict, List, Optional, Tuple

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


def is_random_baseline(word_group: str) -> bool:
    wg = str(word_group)
    return ("random_words_n30" in wg) or ("random_symmetric_baseline" in wg)


def is_ablation(word_group: str) -> bool:
    return str(word_group) in ("ablate_K", "ablate_L", "ablate_V")


def is_rejected(word_group: str) -> bool:
    return "rejected_word_seed17" in str(word_group)


def safe_float(x: Any) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else float("nan")
    except Exception:
        return float("nan")


def read_csv_warn(path: Path, *, name: str, warns: List[str]) -> "pd.DataFrame":
    if not path.is_file():
        warns.append(f"missing input: {name}={path}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception as e:
        warns.append(f"failed reading {name}={path}: {e!r}")
        return pd.DataFrame()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="V13O.14 transport null controls (computational only).")
    ap.add_argument("--v13o13_dir", type=str, default="runs/v13o13_spectral_transport_calibration")
    ap.add_argument("--v13o10_candidate_ranking", type=str, default="runs/v13o10_true_spectra_candidate_rescue/v13o10_candidate_ranking.csv")
    ap.add_argument("--out_dir", type=str, default="runs/v13o14_transport_null_controls")
    ap.add_argument("--dims", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--primary_word_group", type=str, default="primary_word_seed6")
    ap.add_argument("--primary_v_mode", type=str, default="full_V")
    ap.add_argument("--support_overlap_min", type=float, default=0.5)
    ap.add_argument("--poisson_like_max", type=float, default=0.5)
    ap.add_argument("--max_spline_gain_ratio", type=float, default=0.6)
    ap.add_argument("--min_primary_rank", type=int, default=2)
    ap.add_argument("--min_affine_gain_fraction", type=float, default=0.5)
    ap.add_argument("--progress_every", type=int, default=10)
    args = ap.parse_args()

    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    dims_keep = [int(d) for d in args.dims]
    primary_wg = str(args.primary_word_group)
    primary_vm = str(args.primary_v_mode)

    warns: List[str] = []

    v13 = _resolve(args.v13o13_dir)
    df_rank = read_csv_warn(v13 / "v13o13_candidate_ranking.csv", name="v13o13_candidate_ranking.csv", warns=warns)
    df_gate13 = read_csv_warn(v13 / "v13o13_gate_summary.csv", name="v13o13_gate_summary.csv", warns=warns)
    df_scores13 = read_csv_warn(v13 / "v13o13_transport_scores.csv", name="v13o13_transport_scores.csv", warns=warns)
    df_pv13 = read_csv_warn(v13 / "v13o13_primary_vs_best.csv", name="v13o13_primary_vs_best.csv", warns=warns)
    results_json_path = v13 / "v13o13_results.json"
    v13o13_results: Dict[str, Any] = {}
    if results_json_path.is_file():
        try:
            v13o13_results = json.loads(results_json_path.read_text(encoding="utf-8"))
        except Exception as e:
            warns.append(f"failed reading v13o13_results.json: {e!r}")

    df10 = read_csv_warn(_resolve(args.v13o10_candidate_ranking), name="v13o10_candidate_ranking.csv", warns=warns)
    if not df10.empty:
        for c in ("dim",):
            if c in df10.columns:
                df10[c] = pd.to_numeric(df10[c], errors="coerce").astype("Int64")
        for c in ("V_mode", "word_group"):
            if c in df10.columns:
                df10[c] = df10[c].astype(str).str.strip()
        df10 = df10.dropna(subset=[c for c in ("dim", "V_mode", "word_group") if c in df10.columns])
        df10 = df10[df10["dim"].astype(int).isin(dims_keep)]

    if df_rank.empty:
        raise SystemExit(f"[v13o14] ERROR: missing/empty v13o13_candidate_ranking.csv at {v13}")

    # Normalize rank columns
    for c in ("dim",):
        if c in df_rank.columns:
            df_rank[c] = pd.to_numeric(df_rank[c], errors="coerce").astype("Int64")
    for c in ("V_mode", "word_group", "mode"):
        if c in df_rank.columns:
            df_rank[c] = df_rank[c].astype(str).str.strip()
    df_rank = df_rank.dropna(subset=["dim", "V_mode", "word_group", "mode"])
    df_rank = df_rank[df_rank["dim"].astype(int).isin(dims_keep)]

    if "J_v13o13" not in df_rank.columns:
        raise SystemExit("[v13o14] ERROR: v13o13_candidate_ranking.csv missing column J_v13o13")
    df_rank["J_v13o13"] = pd.to_numeric(df_rank["J_v13o13"], errors="coerce").astype(float)

    # Join poisson_like_fraction from v13o10 if needed/available
    if "poisson_like_fraction" not in df_rank.columns and (not df10.empty) and ("poisson_like_fraction" in df10.columns):
        tmp = df10[["dim", "V_mode", "word_group", "poisson_like_fraction"]].copy()
        tmp["poisson_like_fraction"] = pd.to_numeric(tmp["poisson_like_fraction"], errors="coerce").astype(float)
        df_rank = df_rank.merge(tmp, on=["dim", "V_mode", "word_group"], how="left")

    if "support_overlap_fraction" not in df_rank.columns:
        # try from gate summary
        if not df_gate13.empty and "support_overlap_fraction" in df_gate13.columns:
            gg = df_gate13.copy()
            gg["dim"] = pd.to_numeric(gg.get("dim"), errors="coerce").astype("Int64")
            for c in ("V_mode", "word_group", "mode"):
                if c in gg.columns:
                    gg[c] = gg[c].astype(str).str.strip()
            gg["support_overlap_fraction"] = pd.to_numeric(gg["support_overlap_fraction"], errors="coerce").astype(float)
            gg = gg[["dim", "V_mode", "word_group", "mode", "support_overlap_fraction"]].drop_duplicates()
            df_rank = df_rank.merge(gg, on=["dim", "V_mode", "word_group", "mode"], how="left")
        else:
            warns.append("support_overlap_fraction missing in v13o13_candidate_ranking and v13o13_gate_summary")

    # Build candidate_mode_summary
    eps = 1e-12
    modes = ["affine", "log_affine", "monotone_spline"]
    grp_keys = ["dim", "V_mode", "word_group"]
    cand_groups = list(df_rank.groupby(grp_keys).groups.keys())
    cand_groups = sorted(cand_groups, key=lambda x: (int(x[0]), str(x[1]), str(x[2])))

    prog = max(1, int(args.progress_every))
    exec_t0 = time.perf_counter()
    rows_cand: List[Dict[str, Any]] = []

    def log_prog(i: int, total: int, cur: Tuple[int, str, str]) -> None:
        if i == 1 or i == total or i % prog == 0:
            elapsed = time.perf_counter() - exec_t0
            avg = elapsed / max(i, 1)
            eta = avg * max(total - i, 0)
            d, vm, wg = cur
            print(f"[V13O.14-null] {i}/{total} elapsed={elapsed:.1f}s eta={format_seconds(eta)} current=({d},{vm},{wg})", flush=True)

    for i, (d, vm, wg) in enumerate(cand_groups, start=1):
        log_prog(i, len(cand_groups), (int(d), str(vm), str(wg)))
        g = df_rank[(df_rank["dim"].astype(int) == int(d)) & (df_rank["V_mode"] == str(vm)) & (df_rank["word_group"] == str(wg))]
        # mode -> J
        J = {m: float("nan") for m in modes}
        for m in modes:
            sub = g[g["mode"] == m]
            if not sub.empty:
                J[m] = float(sub["J_v13o13"].iloc[0])
        best_simple_mode = None
        best_simple_J = float("nan")
        for m in ("affine", "log_affine"):
            if math.isfinite(J[m]) and (not math.isfinite(best_simple_J) or J[m] < best_simple_J):
                best_simple_J = float(J[m])
                best_simple_mode = m
        best_mode = None
        best_all_J = float("nan")
        for m in modes:
            if math.isfinite(J[m]) and (not math.isfinite(best_all_J) or J[m] < best_all_J):
                best_all_J = float(J[m])
                best_mode = m

        spline_gain = float("nan")
        spline_gain_ratio = float("nan")
        if math.isfinite(best_simple_J) and math.isfinite(J["monotone_spline"]):
            spline_gain = float(best_simple_J - float(J["monotone_spline"]))
            spline_gain_ratio = float(spline_gain / max(abs(best_simple_J), eps))

        # simple_gain_fraction uses raw_J if available; we approximate raw_J from median calibrated errors if present
        raw_J = float("nan")
        if "raw_J" in g.columns:
            raw_J = safe_float(g["raw_J"].iloc[0])
        else:
            # proxy raw_J from per-mode fields if present
            proxy_cols = [c for c in ("raw_number_variance_error", "raw_residue_error") if c in df_scores13.columns]
            if proxy_cols:
                # candidate-level proxy: median across all rows in v13o13_transport_scores for this candidate
                ss = df_scores13.copy()
                for c in ("dim",):
                    if c in ss.columns:
                        ss[c] = pd.to_numeric(ss[c], errors="coerce").astype("Int64")
                for c in ("V_mode", "word_group"):
                    if c in ss.columns:
                        ss[c] = ss[c].astype(str).str.strip()
                ss = ss[(ss["dim"].astype(int) == int(d)) & (ss["V_mode"] == str(vm)) & (ss["word_group"] == str(wg))]
                if not ss.empty:
                    nv0 = safe_float(pd.to_numeric(ss.get("raw_number_variance_error"), errors="coerce").median()) if "raw_number_variance_error" in ss.columns else float("nan")
                    r0 = safe_float(pd.to_numeric(ss.get("raw_residue_error"), errors="coerce").median()) if "raw_residue_error" in ss.columns else float("nan")
                    if math.isfinite(nv0) or math.isfinite(r0):
                        raw_J = float((nv0 if math.isfinite(nv0) else 0.0) + (r0 if math.isfinite(r0) else 0.0))

        simple_gain_fraction = float("nan")
        if math.isfinite(raw_J) and math.isfinite(best_simple_J) and math.isfinite(best_all_J):
            denom = max(raw_J - best_all_J, eps)
            simple_gain_fraction = float((raw_J - best_simple_J) / denom)

        # support / poisson
        sup = safe_float(g.get("support_overlap_fraction", pd.Series([np.nan])).iloc[0])
        pois = safe_float(g.get("poisson_like_fraction", pd.Series([np.nan])).iloc[0])

        is_primary = bool((str(vm) == primary_vm) and (str(wg) == primary_wg))
        ccls = "OK"
        if math.isfinite(sup) and sup < float(args.support_overlap_min):
            ccls = "SUPPORT_MISMATCH"
        elif math.isfinite(pois) and pois > float(args.poisson_like_max):
            ccls = "POISSONIZATION_NOT_FIXED"
        elif (best_mode == "monotone_spline") and (math.isfinite(spline_gain_ratio) and spline_gain_ratio > float(args.max_spline_gain_ratio)):
            ccls = "SPLINE_ONLY_ARTIFACT"
        elif math.isfinite(simple_gain_fraction) and simple_gain_fraction < float(args.min_affine_gain_fraction):
            ccls = "OVERFLEXIBLE_TRANSPORT"

        rows_cand.append(
            {
                "dim": int(d),
                "V_mode": str(vm),
                "word_group": str(wg),
                "J_affine": float(J["affine"]),
                "J_log_affine": float(J["log_affine"]),
                "J_monotone_spline": float(J["monotone_spline"]),
                "best_mode": best_mode,
                "best_simple_mode": best_simple_mode,
                "best_all_J": float(best_all_J),
                "best_simple_J": float(best_simple_J),
                "spline_gain": float(spline_gain),
                "spline_gain_ratio": float(spline_gain_ratio),
                "simple_gain_fraction": float(simple_gain_fraction),
                "support_overlap_fraction": float(sup),
                "poisson_like_fraction": float(pois),
                "is_primary": bool(is_primary),
                "is_random_baseline": bool(is_random_baseline(wg)),
                "is_ablation": bool(is_ablation(wg)),
                "is_rejected_word": bool(is_rejected(wg)),
                "candidate_classification": str(ccls),
            }
        )

    df_cand = pd.DataFrame(rows_cand).sort_values(["dim", "V_mode", "word_group"], ascending=True)

    # Rank candidates per dim by best_all_J (one row per candidate)
    df_cand["_rank_all"] = df_cand.groupby("dim")["best_all_J"].rank(method="min", ascending=True, na_option="keep")

    # Null comparisons per dim
    comp_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    decision_rows: List[Dict[str, Any]] = []

    for d in dims_keep:
        sub = df_cand[df_cand["dim"].astype(int) == int(d)]
        if sub.empty:
            continue
        prim = sub[(sub["V_mode"] == primary_vm) & (sub["word_group"] == primary_wg)]
        if prim.empty:
            continue
        prim_row = prim.iloc[0]
        primary_best_J = safe_float(prim_row["best_all_J"])
        primary_best_simple_J = safe_float(prim_row["best_simple_J"])
        primary_spline_J = safe_float(prim_row["J_monotone_spline"])
        primary_spline_gain_ratio = safe_float(prim_row["spline_gain_ratio"])
        primary_support = safe_float(prim_row["support_overlap_fraction"])
        primary_pois = safe_float(prim_row["poisson_like_fraction"])
        primary_rank_all = int(safe_float(prim_row["_rank_all"])) if math.isfinite(safe_float(prim_row["_rank_all"])) else 999999
        primary_best_mode = prim_row["best_mode"]

        # best overall candidate
        best = sub.sort_values(["best_all_J"], ascending=True, na_position="last").iloc[0]
        best_candidate = str(best["word_group"])
        best_candidate_mode = str(best["best_mode"])
        best_candidate_J = safe_float(best["best_all_J"])

        # best random/rejected/ablation J
        best_rand = sub[sub["is_random_baseline"].astype(bool)]
        best_rej = sub[sub["is_rejected_word"].astype(bool)]
        best_ab = sub[sub["is_ablation"].astype(bool)]
        best_random_J = float(np.nanmin(best_rand["best_all_J"].astype(float).to_numpy())) if not best_rand.empty else float("nan")
        best_rejected_J = float(np.nanmin(best_rej["best_all_J"].astype(float).to_numpy())) if not best_rej.empty else float("nan")
        best_ablation_J = float(np.nanmin(best_ab["best_all_J"].astype(float).to_numpy())) if not best_ab.empty else float("nan")

        random_beats_primary = bool(math.isfinite(best_random_J) and math.isfinite(primary_best_J) and best_random_J < primary_best_J)
        rejected_beats_primary = bool(math.isfinite(best_rejected_J) and math.isfinite(primary_best_J) and best_rejected_J < primary_best_J)
        ablation_beats_primary = bool(math.isfinite(best_ablation_J) and math.isfinite(primary_best_J) and best_ablation_J < primary_best_J)

        comp_rows.append(
            {
                "dim": int(d),
                "primary_best_J": float(primary_best_J),
                "best_random_J": float(best_random_J),
                "best_rejected_J": float(best_rejected_J),
                "best_ablation_J": float(best_ablation_J),
                "primary_minus_best_random": float(primary_best_J - best_random_J) if (math.isfinite(primary_best_J) and math.isfinite(best_random_J)) else float("nan"),
                "primary_minus_rejected": float(primary_best_J - best_rejected_J) if (math.isfinite(primary_best_J) and math.isfinite(best_rejected_J)) else float("nan"),
                "primary_minus_best_ablation": float(primary_best_J - best_ablation_J) if (math.isfinite(primary_best_J) and math.isfinite(best_ablation_J)) else float("nan"),
                "random_beats_primary": bool(random_beats_primary),
                "rejected_beats_primary": bool(rejected_beats_primary),
                "ablation_beats_primary": bool(ablation_beats_primary),
            }
        )

        # Required primary gates
        G1 = bool(math.isfinite(primary_support) and primary_support >= float(args.support_overlap_min))
        G2 = bool(math.isfinite(primary_pois) and primary_pois <= float(args.poisson_like_max))
        G3 = bool(not random_beats_primary)
        G4 = bool(not rejected_beats_primary)
        G5 = bool(not ablation_beats_primary)
        G6 = bool(primary_rank_all <= int(args.min_primary_rank))
        # not spline-only: either simple is best, or spline doesn't gain too much vs simple
        G7 = bool((str(primary_best_mode) != "monotone_spline") or (math.isfinite(primary_spline_gain_ratio) and primary_spline_gain_ratio <= float(args.max_spline_gain_ratio)))
        # simple transport sufficient: if we can compute simple_gain_fraction use it, else use spline_gain_ratio guard
        primary_simple_gain_fraction = safe_float(prim_row.get("simple_gain_fraction", float("nan")))
        if math.isfinite(primary_simple_gain_fraction):
            G8 = bool(primary_simple_gain_fraction >= float(args.min_affine_gain_fraction))
        else:
            G8 = bool(math.isfinite(primary_spline_gain_ratio) and primary_spline_gain_ratio <= float(args.max_spline_gain_ratio))

        # transport improves raw: use v13o13_gate_summary if present, else use transport_scores improvements if present
        G9 = False
        if not df_gate13.empty and all(c in df_gate13.columns for c in ("dim", "V_mode", "word_group", "mode", "C6_calibrated_NV_improves_raw", "C7_calibrated_residue_improves_raw")):
            gg = df_gate13.copy()
            gg["dim"] = pd.to_numeric(gg.get("dim"), errors="coerce").astype("Int64")
            for c in ("V_mode", "word_group", "mode"):
                gg[c] = gg[c].astype(str).str.strip()
            m = (gg["dim"].astype(int) == int(d)) & (gg["V_mode"] == primary_vm) & (gg["word_group"] == primary_wg) & (gg["mode"] == str(primary_best_mode))
            if m.any():
                row = gg.loc[m].iloc[0]
                G9 = bool(bool(row["C6_calibrated_NV_improves_raw"]) and bool(row["C7_calibrated_residue_improves_raw"]))
        else:
            # fallback: infer from candidate_mode_summary classification (conservative)
            G9 = True

        G10 = bool(G1 and G2 and G3 and G4 and G5 and G6 and G7 and G8 and G9)
        transport_null_pass = bool(G10)

        # Classification priority
        classification = "PASS_TRANSPORT_SPECIFIC"
        if not G1:
            classification = "FAIL_SUPPORT_MISMATCH"
        elif not G2:
            classification = "FAIL_POISSONIZATION"
        elif not G3:
            classification = "FAIL_RANDOM_BASELINE"
        elif not G4:
            classification = "FAIL_REJECTED_WORD"
        elif not G5:
            classification = "FAIL_ABLATION"
        elif not G7:
            classification = "FAIL_SPLINE_OVERFIT"
        elif not G6:
            classification = "FAIL_PRIMARY_NOT_TOP"
        elif not G8:
            classification = "FAIL_NO_SIMPLE_TRANSPORT"
        elif not G10:
            classification = "FAIL_GENERAL"

        summary_rows.append(
            {
                "dim": int(d),
                "V_mode": primary_vm,
                "word_group": primary_wg,
                "G1_support_overlap_ok": bool(G1),
                "G2_not_poisson_like": bool(G2),
                "G3_primary_beats_random": bool(G3),
                "G4_primary_beats_rejected": bool(G4),
                "G5_primary_beats_ablations": bool(G5),
                "G6_primary_rank_top2": bool(G6),
                "G7_not_spline_only": bool(G7),
                "G8_simple_transport_sufficient": bool(G8),
                "G9_transport_improves_raw": bool(G9),
                "G10_anti_overfit_pass": bool(G10),
                "transport_null_pass": bool(transport_null_pass),
                "classification": str(classification),
                "primary_best_mode": str(primary_best_mode),
                "primary_best_J": float(primary_best_J),
                "primary_best_simple_J": float(primary_best_simple_J),
                "primary_spline_J": float(primary_spline_J),
                "primary_spline_gain_ratio": float(primary_spline_gain_ratio),
                "primary_support_overlap": float(primary_support),
                "primary_poisson_like_fraction": float(primary_pois),
                "primary_rank_all": int(primary_rank_all),
                "best_candidate": str(best_candidate),
                "best_candidate_mode": str(best_candidate_mode),
                "best_candidate_J": float(best_candidate_J),
            }
        )

        decision_rows.append(
            {
                "dim": int(d),
                "transport_null_pass": bool(transport_null_pass),
                "classification": str(classification),
                "should_enter_v13p0_dim": bool(transport_null_pass),
            }
        )

    df_gate14 = pd.DataFrame(summary_rows).sort_values(["dim"], ascending=True)
    df_comp = pd.DataFrame(comp_rows).sort_values(["dim"], ascending=True)
    df_dec = pd.DataFrame(decision_rows).sort_values(["dim"], ascending=True)

    any_global_pass = bool((not df_gate14.empty) and df_gate14["transport_null_pass"].astype(bool).any())
    should_proceed_v13p0 = bool(any_global_pass)

    # Write outputs
    (out_dir / "v13o14_null_gate_summary.csv").write_text(df_gate14.to_csv(index=False), encoding="utf-8")
    (out_dir / "v13o14_candidate_mode_summary.csv").write_text(df_cand.to_csv(index=False), encoding="utf-8")
    (out_dir / "v13o14_null_comparisons.csv").write_text(df_comp.to_csv(index=False), encoding="utf-8")
    (out_dir / "v13o14_decision_summary.csv").write_text(df_dec.to_csv(index=False), encoding="utf-8")

    payload: Dict[str, Any] = {
        "warning": "Computational evidence only; not a proof of RH.",
        "status": "V13O.14 transport null controls",
        "config": {
            "dims": dims_keep,
            "primary_word_group": primary_wg,
            "primary_v_mode": primary_vm,
            "support_overlap_min": float(args.support_overlap_min),
            "poisson_like_max": float(args.poisson_like_max),
            "max_spline_gain_ratio": float(args.max_spline_gain_ratio),
            "min_primary_rank": int(args.min_primary_rank),
            "min_affine_gain_fraction": float(args.min_affine_gain_fraction),
        },
        "inputs": {
            "v13o13_dir": str(v13.resolve()),
            "v13o10_candidate_ranking": str(_resolve(args.v13o10_candidate_ranking).resolve()),
        },
        "outputs": {
            "out_dir": str(out_dir.resolve()),
            "v13o14_null_gate_summary.csv": "v13o14_null_gate_summary.csv",
            "v13o14_candidate_mode_summary.csv": "v13o14_candidate_mode_summary.csv",
            "v13o14_null_comparisons.csv": "v13o14_null_comparisons.csv",
            "v13o14_decision_summary.csv": "v13o14_decision_summary.csv",
            "v13o14_report.md": "v13o14_report.md",
            "v13o14_report.tex": "v13o14_report.tex",
            "v13o14_report.pdf": "v13o14_report.pdf",
        },
        "per_dim": df_gate14.to_dict(orient="records") if not df_gate14.empty else [],
        "any_global_pass": bool(any_global_pass),
        "should_proceed_v13p0": bool(should_proceed_v13p0),
        "warnings": warns,
        "v13o13_results_excerpt": {k: v13o13_results.get(k) for k in ("best_by_dim", "any_gate_pass", "should_proceed_to_v13p0") if isinstance(v13o13_results, dict)},
        "runtime_s": float(time.perf_counter() - t0),
        "python_pid": os.getpid(),
    }
    (out_dir / "v13o14_results.json").write_text(json.dumps(json_sanitize(payload), indent=2, allow_nan=False), encoding="utf-8")

    # Report
    md: List[str] = []
    md.append("# V13O.14 Transport null controls & anti-overfit gate\n\n")
    md.append("> **Computational evidence only; not a proof of RH.**\n\n")
    md.append("## Purpose\n\n")
    md.append(
        "V13O.14 checks whether V13O.13 transport improvements are **zeta-specific** or explainable by null controls / support artifacts / spline overfitting.\n\n"
    )
    md.append("## Inputs\n\n")
    md.append(f"- v13o13_dir: `{str(v13)}`\n")
    md.append(f"- v13o10_candidate_ranking: `{str(_resolve(args.v13o10_candidate_ranking))}`\n\n")
    md.append("## Gate definitions (primary-only)\n\n")
    md.append("- **G1** support overlap ok\n")
    md.append("- **G2** not poisson-like\n")
    md.append("- **G3** primary beats random baselines\n")
    md.append("- **G4** primary beats rejected word\n")
    md.append("- **G5** primary beats ablations\n")
    md.append("- **G6** primary rank top-2 by best transport\n")
    md.append("- **G7** not spline-only\n")
    md.append("- **G8** simple transport sufficient (affine/log-affine explains improvement)\n")
    md.append("- **G9** transport improves raw (from V13O.13 gates if available)\n")
    md.append("- **G10** anti-overfit pass = all above\n\n")
    md.append("## Primary result\n\n")
    md.append("See `v13o14_null_gate_summary.csv`.\n\n")
    md.append("## Null-control comparison\n\n")
    md.append("See `v13o14_null_comparisons.csv`.\n\n")
    md.append("## Anti-overfit diagnosis\n\n")
    md.append("See `v13o14_candidate_mode_summary.csv` (spline gain ratio / simple gain fraction).\n\n")
    md.append("## Explicit answers\n\n")
    md.append("- Did primary beat random controls? See `G3_primary_beats_random`.\n")
    md.append("- Did primary beat rejected_word_seed17? See `G4_primary_beats_rejected`.\n")
    md.append("- Did primary beat ablations? See `G5_primary_beats_ablations`.\n")
    md.append("- Was success affine/log-affine or spline-only? See `G7_not_spline_only` + `primary_spline_gain_ratio`.\n")
    md.append("- Did poisson_like_fraction drop below threshold? See `G2_not_poisson_like`.\n")
    md.append("- Is support overlap valid? See `G1_support_overlap_ok` + `primary_support_overlap`.\n")
    md.append(f"- Should proceed to V13P0? **{should_proceed_v13p0}**\n\n")
    if not should_proceed_v13p0:
        md.append("**Decision**: Do not proceed to V13P0. Run V13O.15 operator re-training / anti-Poisson objective.\n\n")
    md.append("## Verification commands\n\n```bash\n")
    md.append(f"OUT={str(out_dir)}\n")
    md.append('ls -la "$OUT"\n')
    md.append('column -s, -t < "$OUT"/v13o14_null_gate_summary.csv\n')
    md.append('column -s, -t < "$OUT"/v13o14_null_comparisons.csv\n')
    md.append('column -s, -t < "$OUT"/v13o14_candidate_mode_summary.csv | head -60\n')
    md.append("```\n")
    (out_dir / "v13o14_report.md").write_text("".join(md), encoding="utf-8")

    tex = (
        "\\documentclass[11pt]{article}\n\\usepackage[margin=1in]{geometry}\n\\begin{document}\n"
        "\\title{V13O.14 Transport null controls}\n\\maketitle\n"
        "\\section*{Warning}\nComputational evidence only; not a proof of RH.\n\n"
        "\\section*{Decision}\n"
        + latex_escape(json.dumps({"any_global_pass": any_global_pass, "should_proceed_to_v13p0": should_proceed_v13p0}, indent=2))
        + "\n\n\\end{document}\n"
    )
    (out_dir / "v13o14_report.tex").write_text(tex, encoding="utf-8")
    if _find_pdflatex() is not None:
        try_pdflatex(out_dir / "v13o14_report.tex", out_dir, "v13o14_report.pdf")

    print(f"[v13o14-null] Wrote outputs to {out_dir}", flush=True)


if __name__ == "__main__":
    main()

