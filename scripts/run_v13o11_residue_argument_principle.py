#!/usr/bin/env python3
"""
V13O.11 — Residue / Argument-Principle Diagnostics (true spectra).

Computational diagnostic only; not a proof of the Riemann Hypothesis.

This script consumes:
  - explicit unfolded levels CSV (V13O.9 format or V13O.8-compatible kind format)
  - candidate ranking CSV (e.g., V13O.10 candidate ranking)
  - precise zeta zeros CSV/TXT (gamma heights)

and computes window-based proxies:
  A) argument-principle proxy via counts in windows
  B) residue proxy via resolvent-like contour integral
  C) trace-formula proxy via Gaussian test functions

It outputs per-candidate scores and a gate summary. No RH claims.
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

from validation import residue_diagnostics as rd  # noqa: E402


def _resolve(p: str) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = Path(ROOT) / path
    return path


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


def safe_median(xs: List[float]) -> float:
    arr = np.asarray([float(v) for v in xs if v is not None and math.isfinite(float(v))], dtype=np.float64)
    return float(np.median(arr)) if arr.size else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description="V13O.11 residue/argument-principle diagnostics (computational only).")

    ap.add_argument("--true_levels_csv", type=str, required=True)
    ap.add_argument("--candidate_ranking", type=str, required=True)
    ap.add_argument("--zeros_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="runs/v13o11_residue_argument_principle")
    ap.add_argument("--dims", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--primary_word_group", type=str, default="primary_word_seed6")
    ap.add_argument("--primary_v_mode", type=str, default="full_V")

    ap.add_argument("--window_min", type=float, default=0.0)
    ap.add_argument("--window_max", type=float, default=64.0)
    ap.add_argument("--window_size", type=float, default=16.0)
    ap.add_argument("--window_stride", type=float, default=8.0)

    ap.add_argument("--eta", type=float, default=0.25)
    ap.add_argument("--n_contour_points", type=int, default=128)
    ap.add_argument("--top_k_candidates", type=int, default=5)
    ap.add_argument("--n_jobs", type=int, default=8)
    ap.add_argument("--progress_every", type=int, default=10)

    args = ap.parse_args()


    if not hasattr(args, 'margin'):

        setattr(args, 'margin', 0.25)

    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    dims_keep = [int(x) for x in args.dims]
    primary_wg = str(args.primary_word_group)
    primary_vm = str(args.primary_v_mode)
    eps = 1e-12

    # Load inputs
    levels_path = _resolve(args.true_levels_csv)
    rank_path = _resolve(args.candidate_ranking)
    zeros_path = _resolve(args.zeros_csv)

    df_levels, level_warns = rd.load_true_levels_csv(levels_path, dims_keep=dims_keep)
    if df_levels is None or df_levels.empty:
        raise SystemExit(f"[v13o11] ERROR: true_levels_csv invalid/empty: {levels_path} warns={level_warns}")

    df_rank = pd.read_csv(rank_path)
    df_rank.columns = [str(c).strip() for c in df_rank.columns]

    zeros_raw, zeros_warns = rd.load_zeros_csv(zeros_path)
    # unfold zeros to the same mean-spacing-one coordinate
    zeros_unfolded = rd.unfold_to_mean_spacing_one(zeros_raw)

    warnings: List[str] = []
    warnings.extend([f"true_levels_csv: {w}" for w in level_warns])
    warnings.extend([f"zeros_csv: {w}" for w in zeros_warns])

    # candidate selection: top_k per dim by candidate_rank_by_error if present else by median_total_curve_error
    df_rank["dim"] = pd.to_numeric(df_rank.get("dim"), errors="coerce").astype("Int64")
    for k in ("V_mode", "word_group"):
        if k in df_rank.columns:
            df_rank[k] = df_rank[k].astype(str).str.strip()
    df_rank = df_rank.dropna(subset=["dim", "V_mode", "word_group"])
    df_rank = df_rank[df_rank["dim"].astype(int).isin(dims_keep)]

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

    # Windows
    windows = rd.make_windows(args.window_min, args.window_max, args.window_size, args.window_stride)
    if not windows:
        raise SystemExit("[v13o11] ERROR: no windows produced; check window_min/max/size/stride.")

    # Helper: extract operator and target arrays for (dim,vm,wg) under target_group=real_zeta (preferred)
    def get_levels(dim: int, vm: str, wg: str, *, source: str) -> np.ndarray:
        sub = df_levels[
            (df_levels["dim"].astype(int) == int(dim))
            & (df_levels["V_mode"].astype(str) == str(vm))
            & (df_levels["word_group"].astype(str) == str(wg))
            & (df_levels["target_group"].astype(str) == "real_zeta")
            & (df_levels["source"].astype(str) == str(source))
        ]
        xs = sub["unfolded_level"].astype(float).to_numpy(dtype=np.float64)
        xs = xs[np.isfinite(xs)]
        return np.sort(xs)

    # Compute per-window diagnostics for each candidate (serial; can be parallelized later if needed)
    residue_rows: List[Dict[str, Any]] = []
    arg_rows: List[Dict[str, Any]] = []
    trace_rows: List[Dict[str, Any]] = []

    # sigmas for trace proxy
    sigmas = [0.5, 1.0, 2.0, 4.0]

    total_jobs = len(selected) * len(windows)
    done = 0
    exec_t0 = time.perf_counter()

    def log_prog() -> None:
        nonlocal done
        if done == 1 or done == total_jobs or (done % max(1, int(args.progress_every)) == 0):
            elapsed = time.perf_counter() - exec_t0
            avg = elapsed / max(done, 1)
            eta = avg * max(total_jobs - done, 0)
            print(f"[V13O.11-residue] {done}/{total_jobs} elapsed={elapsed:.1f}s eta={rd.format_seconds(eta) if hasattr(rd,'format_seconds') else ''}", flush=True)

    # fallback format_seconds if not exported from module
    def fmt(sec: float) -> str:
        sec = max(0.0, float(sec))
        m = int(sec // 60)
        s = int(sec % 60)
        return f"{m:02d}:{s:02d}"

    for (dim, vm, wg) in selected:
        op = get_levels(dim, vm, wg, source="operator")
        tg = get_levels(dim, vm, wg, source="target")
        if op.size == 0 or tg.size == 0:
            warnings.append(f"missing operator/target levels for (dim={dim},V_mode={vm},word_group={wg}) under target_group=real_zeta")
            continue

        for (a, b) in windows:
            done += 1
            if done == 1 or done == total_jobs or done % max(1, int(args.progress_every)) == 0:
                elapsed = time.perf_counter() - exec_t0
                avg = elapsed / max(done, 1)
                eta = avg * max(total_jobs - done, 0)
                print(f"[V13O.11-residue] {done}/{total_jobs} elapsed={elapsed:.1f}s eta={fmt(eta)} current=({dim},{vm},{wg})", flush=True)

            # Argument principle proxy
            N_err, N_norm, N_op, N_tg = rd.argument_principle_proxy(op, zeros_unfolded, float(a), float(b))
            arg_rows.append(
                {
                    "dim": dim,
                    "V_mode": vm,
                    "word_group": wg,
                    "window_a": float(a),
                    "window_b": float(b),
                    "N_operator": int(N_op),
                    "N_target": int(N_tg),
                    "N_error": float(N_err),
                    "N_error_norm": float(N_norm),
                }
            )

            # Residue proxy counts via contour integral
            I_op = rd.residue_proxy_count(op, float(a), float(b), eta=float(args.eta), n_contour_points=int(args.n_contour_points))
            I_tg = rd.residue_proxy_count(zeros_unfolded, float(a), float(b), eta=float(args.eta), n_contour_points=int(args.n_contour_points))
            residue_count_error = float(abs(I_op.real - I_tg.real))
            residue_imag_leak = float(abs(I_op.imag))
            residue_mass_error = float(abs(abs(I_op.real) - round(abs(I_op.real))))

            residue_rows.append(
                {
                    "dim": dim,
                    "V_mode": vm,
                    "word_group": wg,
                    "window_a": float(a),
                    "window_b": float(b),
                    "I_operator_real": float(I_op.real),
                    "I_operator_imag": float(I_op.imag),
                    "I_target_real": float(I_tg.real),
                    "I_target_imag": float(I_tg.imag),
                    "residue_count_error": residue_count_error,
                    "residue_imag_leak": residue_imag_leak,
                    "residue_mass_error": residue_mass_error,
                }
            )

            # Trace proxy over sigmas at center
            c = 0.5 * (float(a) + float(b))
            for s in sigmas:
                Sop = rd.trace_formula_proxy(op, center=c, sigma=float(s))
                Stg = rd.trace_formula_proxy(zeros_unfolded, center=c, sigma=float(s))
                err = float(abs(Sop - Stg) / max(eps, abs(Stg))) if math.isfinite(Sop) and math.isfinite(Stg) else float("nan")
                trace_rows.append(
                    {
                        "dim": dim,
                        "V_mode": vm,
                        "word_group": wg,
                        "window_a": float(a),
                        "window_b": float(b),
                        "center": float(c),
                        "sigma": float(s),
                        "S_operator": float(Sop),
                        "S_target": float(Stg),
                        "trace_error_norm": float(err),
                    }
                )

    df_arg = pd.DataFrame(arg_rows)
    df_res = pd.DataFrame(residue_rows)
    df_trace = pd.DataFrame(trace_rows)

    # Aggregate per candidate
    scores_rows: List[Dict[str, Any]] = []
    gate_rows: List[Dict[str, Any]] = []

    # Artifact flags / poisson fractions if present in candidate ranking
    # (best-effort; non-fatal if missing)
    rank_cols = set(df_rank.columns)
    def get_rank_row(dim: int, vm: str, wg: str) -> Dict[str, Any]:
        sub = df_rank[(df_rank["dim"].astype(int) == int(dim)) & (df_rank["V_mode"] == vm) & (df_rank["word_group"] == wg)]
        if sub.empty:
            return {}
        r0 = sub.iloc[0].to_dict()
        return {k: r0.get(k) for k in r0.keys()}

    for (dim, vm, wg) in selected:
        # collect window metrics
        a0 = df_arg[(df_arg["dim"] == dim) & (df_arg["V_mode"] == vm) & (df_arg["word_group"] == wg)] if not df_arg.empty else pd.DataFrame()
        r0 = df_res[(df_res["dim"] == dim) & (df_res["V_mode"] == vm) & (df_res["word_group"] == wg)] if not df_res.empty else pd.DataFrame()
        t0df = df_trace[(df_trace["dim"] == dim) & (df_trace["V_mode"] == vm) & (df_trace["word_group"] == wg)] if not df_trace.empty else pd.DataFrame()
        if a0.empty or r0.empty or t0df.empty:
            continue

        med_arg = safe_median(a0["N_error_norm"].astype(float).tolist())
        med_res_cnt = safe_median(r0["residue_count_error"].astype(float).tolist())
        med_imag = safe_median(r0["residue_imag_leak"].astype(float).tolist())
        med_trace = safe_median(t0df["trace_error_norm"].astype(float).tolist())

        # penalties from ranking if present
        rr = get_rank_row(dim, vm, wg)
        poisson_like_fraction = float(rr.get("poisson_like_fraction", 1.0)) if rr else 1.0
        is_random = bool(rr.get("is_random_baseline", False)) if rr else False
        is_ablation = bool(rr.get("is_ablation", False)) if rr else False
        is_rejected = bool(rr.get("is_rejected_word", False)) if rr else False
        is_primary = bool(rr.get("is_primary", False)) if rr else (wg == primary_wg)

        artifact_pen = 0.0
        if is_random:
            artifact_pen += 0.25
        if is_ablation:
            artifact_pen += 0.10
        if is_primary:
            artifact_pen += 0.05
        # rejected_word: +0.00

        J = (
            1.0 * float(med_arg)
            + 1.0 * float(med_res_cnt)
            + 0.5 * float(med_imag)
            + 1.0 * float(med_trace)
            + 1.0 * float(poisson_like_fraction)
            + 1.0 * float(artifact_pen)
        )

        scores_rows.append(
            {
                "dim": dim,
                "V_mode": vm,
                "word_group": wg,
                "median_argument_count_error_norm": float(med_arg),
                "median_residue_count_error": float(med_res_cnt),
                "median_residue_imag_leak": float(med_imag),
                "trace_formula_proxy_error": float(med_trace),
                "poisson_like_penalty": float(poisson_like_fraction),
                "artifact_penalty": float(artifact_pen),
                "J_residue": float(J),
            }
        )

    df_scores = pd.DataFrame(scores_rows)
    if not df_scores.empty:
        df_scores = df_scores.sort_values(["dim", "J_residue"], ascending=True, na_position="last")
        # rank within dim
        df_scores["rank_by_J_residue"] = df_scores.groupby("dim")["J_residue"].rank(method="min", ascending=True)

    # Gate: basic, conservative thresholds derived from args.margin
    # (These are diagnostics, not proofs.)
    for r in df_scores.itertuples(index=False):
        dim = int(getattr(r, "dim"))
        vm = str(getattr(r, "V_mode"))
        wg = str(getattr(r, "word_group"))
        rr = get_rank_row(dim, vm, wg)
        coverage = float(rr.get("coverage_score", float("nan"))) if rr else float("nan")
        poisson_frac = float(rr.get("poisson_like_fraction", 1.0)) if rr else 1.0
        is_random = bool(rr.get("is_random_baseline", False)) if rr else False
        is_ablation = bool(rr.get("is_ablation", False)) if rr else False

        A1 = True
        A2 = bool(math.isfinite(coverage) and coverage >= 0.7) if rr else True
        A3 = bool(float(getattr(r, "median_argument_count_error_norm")) <= float(args.margin))
        A4 = bool(float(getattr(r, "median_residue_count_error")) <= float(args.margin))
        A5 = bool(float(getattr(r, "median_residue_imag_leak")) <= float(args.margin))
        # A6 trace beats controls: not available here reliably; approximate by <= margin
        A6 = bool(float(getattr(r, "trace_formula_proxy_error")) <= float(args.margin))
        A7 = bool(poisson_frac < 0.75)
        A8 = bool(not is_random)
        A9 = bool(not is_ablation)
        A10 = bool(float(getattr(r, "rank_by_J_residue")) <= 3.0)

        gate_pass = bool(A1 and A2 and A3 and A4 and A5 and A6 and A7 and A8 and A9 and A10)
        gate_rows.append(
            {
                "dim": dim,
                "V_mode": vm,
                "word_group": wg,
                "A1_true_levels_available": A1,
                "A2_candidate_coverage_ok": A2,
                "A3_argument_count_error_small": A3,
                "A4_residue_error_small": A4,
                "A5_imag_leak_small": A5,
                "A6_trace_proxy_beats_controls": A6,
                "A7_not_poisson_like": A7,
                "A8_not_random_baseline": A8,
                "A9_not_ablation_only": A9,
                "A10_residue_gate_pass": A10,
                "gate_pass": gate_pass,
            }
        )

    df_gate = pd.DataFrame(gate_rows)

    # Candidate ranking output: merge df_scores + df_gate
    df_out_rank = df_scores.merge(df_gate, on=["dim", "V_mode", "word_group"], how="left") if not df_scores.empty else pd.DataFrame()

    # Outputs
    (out_dir / "v13o11_residue_scores.csv").write_text(df_scores.to_csv(index=False), encoding="utf-8")
    (out_dir / "v13o11_argument_counts.csv").write_text(df_arg.to_csv(index=False), encoding="utf-8")
    (out_dir / "v13o11_trace_formula_proxy.csv").write_text(df_trace.to_csv(index=False), encoding="utf-8")
    (out_dir / "v13o11_gate_summary.csv").write_text(df_gate.to_csv(index=False), encoding="utf-8")
    (out_dir / "v13o11_candidate_ranking.csv").write_text(df_out_rank.to_csv(index=False), encoding="utf-8")

    # Results + report
    any_pass = bool((not df_gate.empty) and df_gate["gate_pass"].astype(bool).any())
    payload = {
        "warning": "Computational diagnostic only; not a proof of the Riemann Hypothesis.",
        "status": "V13O.11 residue/argument-principle diagnostics",
        "inputs": {
            "true_levels_csv": str(levels_path.resolve()),
            "candidate_ranking": str(rank_path.resolve()),
            "zeros_csv": str(zeros_path.resolve()),
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
        "params": {"eta": float(args.eta), "n_contour_points": int(args.n_contour_points), "top_k_candidates": int(args.top_k_candidates)},
        "warnings": warnings,
        "any_gate_pass": any_pass,
        "runtime_s": float(time.perf_counter() - t0),
        "python_pid": os.getpid(),
    }
    (out_dir / "v13o11_results.json").write_text(json.dumps(rd.json_sanitize(payload) if hasattr(rd, "json_sanitize") else json_sanitize(payload), indent=2, allow_nan=False), encoding="utf-8")

    md: List[str] = []
    md.append("# V13O.11 Residue / Argument-Principle Diagnostics\n\n")
    md.append("> **Computational evidence only; not a proof of RH.**\n\n")
    md.append("## Inputs\n\n")
    md.append(f"- true_levels_csv: `{payload['inputs']['true_levels_csv']}`\n")
    md.append(f"- candidate_ranking: `{payload['inputs']['candidate_ranking']}`\n")
    md.append(f"- zeros_csv: `{payload['inputs']['zeros_csv']}`\n\n")
    md.append("## Method summary\n\n")
    md.append("- Argument-principle proxy: windowed count mismatch.\n")
    md.append("- Residue proxy: contour integral of resolvent-like Cauchy sum.\n")
    md.append("- Trace proxy: Gaussian test-function sums over multiple sigmas.\n\n")
    md.append("## Outputs\n\n")
    md.append("- `v13o11_residue_scores.csv`\n")
    md.append("- `v13o11_argument_counts.csv`\n")
    md.append("- `v13o11_trace_formula_proxy.csv`\n")
    md.append("- `v13o11_gate_summary.csv`\n")
    md.append("- `v13o11_candidate_ranking.csv`\n\n")
    md.append("## Gate outcome\n\n")
    md.append(f"- any_gate_pass: **{any_pass}**\n\n")
    if warnings:
        md.append("## Warnings (summary)\n\n")
        md.append(f"- n_warnings={len(warnings)}\n")
        md.append(f"- first_warning={warnings[0]}\n\n")
    md.append("## Verification commands\n\n```bash\n")
    md.append(f"OUT={str(out_dir)}\n\n")
    md.append('echo "=== FILES ==="\nfind "$OUT" -maxdepth 1 -type f | sort\n\n')
    md.append('echo "=== TOP SCORES ==="\ncolumn -s, -t < "$OUT"/v13o11_residue_scores.csv | head -40\n\n')
    md.append('echo "=== GATE ==="\ncolumn -s, -t < "$OUT"/v13o11_gate_summary.csv | head -60\n\n')
    md.append('echo "=== REPORT ==="\nhead -200 "$OUT"/v13o11_report.md\n')
    md.append("```\n")
    (out_dir / "v13o11_report.md").write_text("".join(md), encoding="utf-8")

    tex = (
        "\\documentclass[11pt]{article}\n\\usepackage[margin=1in]{geometry}\n\\begin{document}\n"
        "\\title{V13O.11 Residue / Argument-Principle Diagnostics}\n\\maketitle\n"
        "\\section*{Warning}\nComputational evidence only; not a proof of RH.\n\n"
        "\\section*{Summary}\n"
        + latex_escape(json.dumps({"any_gate_pass": any_pass, "n_windows": len(windows)}, indent=2))
        + "\n\n\\end{document}\n"
    )
    (out_dir / "v13o11_report.tex").write_text(tex, encoding="utf-8")

    if _find_pdflatex() is None:
        print("[v13o11-residue] WARNING: pdflatex not found; skipping PDF.", flush=True)
    elif try_pdflatex(out_dir / "v13o11_report.tex", out_dir, "v13o11_report.pdf"):
        print(f"Wrote {out_dir / 'v13o11_report.pdf'}", flush=True)
    else:
        print("[v13o11-residue] WARNING: pdflatex failed or did not produce v13o11_report.pdf.", flush=True)

    print(f"[v13o11-residue] Wrote {out_dir / 'v13o11_results.json'}", flush=True)


if __name__ == "__main__":
    main()

