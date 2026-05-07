#!/usr/bin/env python3
"""
V14.6b — Real Null-Control Stage E (standalone script)

Computational evidence only; not a proof of RH.

This script post-processes an existing V14.6 output directory and replaces
placeholder null-control gates with real comparisons against prior/null pools:
  - V13O.14 null controls (random/rejected/ablation groups)
  - V14.2 best/ACO history (prior stabilized Artin search)
  - V14.5 ACO history + ablation summary (rank-based semantic DTES search)

Robustness:
  - Missing files/columns never crash the run.
  - Missing sources are recorded (missing_source=True) and corresponding gates are False.
  - No silent passing: if no null distribution (>=3 values), zscore gate is False.

Outputs:
  - v14_6b_prior_baseline_pool.csv
  - v14_6b_real_null_comparisons.csv
  - v14_6b_real_null_gate_summary.csv
  - v14_6b_decision_summary.csv
  - v14_6b_results.json
  - v14_6b_report.md/.tex/.pdf (pdf only if pdflatex exists)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pandas as pd  # type: ignore

    _HAVE_PANDAS = True
except Exception:
    pd = None  # type: ignore
    _HAVE_PANDAS = False


def _resolve(p: str) -> Path:
    return Path(p).expanduser().resolve()


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


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
        v = float(x)
        return float(v) if math.isfinite(v) else None
    except Exception:
        return str(x)


def write_text(path: Path, s: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(s, encoding="utf-8")


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def _read_csv_best_effort(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    if _HAVE_PANDAS:
        try:
            df = pd.read_csv(path)  # type: ignore
            return df.to_dict(orient="records")  # type: ignore
        except Exception:
            pass
    # fallback to stdlib csv
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def _find_pdflatex() -> Optional[str]:
    w = shutil.which("pdflatex")
    if w:
        return w
    for p in ("/Library/TeX/texbin/pdflatex",):
        if Path(p).is_file():
            return p
    return None


def try_pdflatex(tex_path: Path, out_dir: Path, pdf_name: str) -> bool:
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
        return r.returncode == 0 and (out_dir / pdf_name).is_file()
    except Exception:
        return False


def pick_first_present(row: Dict[str, Any], keys: Sequence[str]) -> float:
    for k in keys:
        if k in row:
            v = safe_float(row.get(k, float("nan")))
            if math.isfinite(v):
                return float(v)
    return float("nan")


@dataclass
class BaselineRow:
    dim: int
    source: str
    baseline_kind: str  # random / rejected / ablation / prior_artin / primary / other
    label: str
    mode: str
    J: float
    source_file: str
    missing_source: bool
    notes: str


def normalize_baseline_rows(rows: List[BaselineRow]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "dim": int(r.dim),
                "source": str(r.source),
                "baseline_kind": str(r.baseline_kind),
                "label": str(r.label),
                "mode": str(r.mode),
                "J": float(r.J) if math.isfinite(float(r.J)) else float("nan"),
                "source_file": str(r.source_file),
                "missing_source": bool(r.missing_source),
                "notes": str(r.notes),
            }
        )
    return out


def compute_null_stats(best_gan_J: float, pool_Js: List[float]) -> Tuple[float, float]:
    xs = [float(x) for x in pool_Js if math.isfinite(float(x))]
    if len(xs) < 3 or not math.isfinite(float(best_gan_J)):
        return float("nan"), float("nan")
    mu = float(np.mean(xs))
    sd = float(np.std(xs))
    z = float((mu - float(best_gan_J)) / sd) if sd > 1e-12 else float("nan")
    pct = float(np.mean([1.0 if x <= float(best_gan_J) else 0.0 for x in xs]))
    return z, pct


def main() -> None:
    ap = argparse.ArgumentParser(description="V14.6b Real Null-Control Stage E (post-process V14.6 run).")
    ap.add_argument("--v14_6_dir", type=str, required=True)
    ap.add_argument("--v13o14_dir", type=str, required=True)
    ap.add_argument("--v14_5_dir", type=str, required=True)
    ap.add_argument("--v14_2_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="")
    ap.add_argument("--dims", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--primary_word_group", type=str, default="primary_word_seed6")
    ap.add_argument("--rejected_word_group", type=str, default="rejected_word_seed17")
    ap.add_argument("--random_word_groups", type=str, nargs="+", default=["random_words_n30", "random_symmetric_baseline"])
    ap.add_argument("--ablation_word_groups", type=str, nargs="+", default=["ablate_K", "ablate_L", "ablate_V"])
    ap.add_argument("--null_separation_margin", type=float, default=0.0)
    ap.add_argument("--null_zscore_margin", type=float, default=1.0)
    ap.add_argument("--require_beats_random", action="store_true")
    ap.add_argument("--require_beats_rejected", action="store_true")
    ap.add_argument("--require_beats_ablation", action="store_true")
    ap.add_argument("--require_beats_prior_artin", action="store_true")
    ap.add_argument("--progress_every", type=int, default=10)
    args = ap.parse_args()

    v14_6_dir = _resolve(args.v14_6_dir)
    v13o14_dir = _resolve(args.v13o14_dir)
    v14_5_dir = _resolve(args.v14_5_dir)
    v14_2_dir = _resolve(args.v14_2_dir)

    out_dir = _resolve(args.out_dir) if args.out_dir else (v14_6_dir / "v14_6b_real_null_stage_e")
    out_dir.mkdir(parents=True, exist_ok=True)

    dims = [int(d) for d in args.dims]
    progress_every = max(1, int(args.progress_every))

    # Load V14.6 candidates
    v14_6_gate = _read_csv_best_effort(v14_6_dir / "v14_6_gate_summary.csv")
    v14_6_rank = _read_csv_best_effort(v14_6_dir / "v14_6_candidate_ranking.csv")
    v14_6_abl = _read_csv_best_effort(v14_6_dir / "v14_6_mode_ablation_summary.csv")

    # Map best GAN candidate J per dim/mode
    best_gan: Dict[Tuple[int, str], Dict[str, Any]] = {}
    # prefer ranking (rank==1)
    for r in v14_6_rank:
        d = safe_int(r.get("dim", 0))
        if d not in dims:
            continue
        mode = str(r.get("mode", "")).strip()
        rank = safe_int(r.get("rank", 999999))
        if mode == "" or rank != 1:
            continue
        J = safe_float(r.get("J_total", float("nan")))
        if not math.isfinite(J):
            continue
        best_gan[(d, mode)] = {
            "dim": d,
            "mode": mode,
            "word": str(r.get("word", "")),
            "best_gan_J": float(J),
            "critic_score": safe_float(r.get("critic_score", float("nan"))),
            "final_reward": safe_float(r.get("final_reward", float("nan"))),
            "prior_v14_6_all_gate_pass": str(r.get("all_gate_pass", "")).lower() == "true",
        }
    # fallback to gate summary
    if not best_gan:
        for r in v14_6_gate:
            d = safe_int(r.get("dim", 0))
            if d not in dims:
                continue
            mode = str(r.get("mode", "")).strip()
            if mode == "":
                continue
            J = safe_float(r.get("J_total", float("nan")))
            if not math.isfinite(J):
                continue
            best_gan[(d, mode)] = {
                "dim": d,
                "mode": mode,
                "word": str(r.get("word", "")),
                "best_gan_J": float(J),
                "critic_score": safe_float(r.get("critic_score", float("nan"))),
                "final_reward": safe_float(r.get("final_reward", float("nan"))),
                "prior_v14_6_all_gate_pass": str(r.get("all_gate_pass", "")).lower() == "true",
            }

    # Baseline pool construction
    baseline_rows: List[BaselineRow] = []
    missing_sources: List[str] = []

    # --- V13O.14 ---
    v13_mode = _read_csv_best_effort(v13o14_dir / "v13o14_candidate_mode_summary.csv")
    v13_null = _read_csv_best_effort(v13o14_dir / "v13o14_null_comparisons.csv")
    v13_gate = _read_csv_best_effort(v13o14_dir / "v13o14_null_gate_summary.csv")

    if not v13_mode:
        missing_sources.append("v13o14_candidate_mode_summary.csv")
    if not v13_null:
        missing_sources.append("v13o14_null_comparisons.csv")

    # candidate_mode_summary has word_group and best_all_J/best_simple_J
    for r in v13_mode:
        d = safe_int(r.get("dim", 0))
        if d not in dims:
            continue
        group = str(r.get("word_group", "")).strip()
        if not group:
            continue
        J = pick_first_present(r, ["best_all_J", "best_simple_J", "J_monotone_spline", "J_log_affine", "J_affine"])
        if not math.isfinite(J):
            continue
        if group == str(args.primary_word_group):
            kind = "primary"
        elif group == str(args.rejected_word_group):
            kind = "rejected"
        elif group in set(args.random_word_groups):
            kind = "random"
        elif group in set(args.ablation_word_groups):
            kind = "ablation"
        else:
            kind = "other"
        baseline_rows.append(
            BaselineRow(
                dim=d,
                source="v13o14",
                baseline_kind=kind,
                label=group,
                mode=str(r.get("best_mode", "")) or "unknown",
                J=float(J),
                source_file="v13o14_candidate_mode_summary.csv",
                missing_source=False,
                notes=str(r.get("candidate_classification", "")),
            )
        )

    # null_comparisons gives best_random/best_rejected/best_ablation/primary_best (best-of)
    for r in v13_null:
        d = safe_int(r.get("dim", 0))
        if d not in dims:
            continue
        pr = safe_float(r.get("primary_best_J", float("nan")))
        br = safe_float(r.get("best_random_J", float("nan")))
        bj = safe_float(r.get("best_rejected_J", float("nan")))
        ba = safe_float(r.get("best_ablation_J", float("nan")))
        if math.isfinite(pr):
            baseline_rows.append(BaselineRow(d, "v13o14", "primary", str(args.primary_word_group), "best_of", float(pr), "v13o14_null_comparisons.csv", False, "best_of"))
        if math.isfinite(br):
            baseline_rows.append(BaselineRow(d, "v13o14", "random", "best_random_controls", "best_of", float(br), "v13o14_null_comparisons.csv", False, "best_of"))
        if math.isfinite(bj):
            baseline_rows.append(BaselineRow(d, "v13o14", "rejected", str(args.rejected_word_group), "best_of", float(bj), "v13o14_null_comparisons.csv", False, "best_of"))
        if math.isfinite(ba):
            baseline_rows.append(BaselineRow(d, "v13o14", "ablation", "best_ablations", "best_of", float(ba), "v13o14_null_comparisons.csv", False, "best_of"))

    # --- V14.5 ---
    v14_5_gate = _read_csv_best_effort(v14_5_dir / "v14_5_gate_summary.csv")
    v14_5_abl = _read_csv_best_effort(v14_5_dir / "v14_5_ablation_summary.csv")
    v14_5_hist = _read_csv_best_effort(v14_5_dir / "v14_5_aco_history.csv")
    if not v14_5_gate:
        missing_sources.append("v14_5_gate_summary.csv")
    if not v14_5_abl:
        missing_sources.append("v14_5_ablation_summary.csv")
    if not v14_5_hist:
        missing_sources.append("v14_5_aco_history.csv")

    # ablation_summary gives per-dim best_J per mode
    for r in v14_5_abl:
        d = safe_int(r.get("dim", 0))
        if d not in dims:
            continue
        for k in [
            "numeric_only_best_J",
            "semantic_only_best_J",
            "hybrid_numeric_semantic_best_J",
            "hybrid_ranked_anticollapse_best_J",
        ]:
            if k in r:
                J = safe_float(r.get(k, float("nan")))
                if math.isfinite(J):
                    baseline_rows.append(BaselineRow(d, "v14_5", "prior_artin", k.replace("_best_J", ""), "best_of", float(J), "v14_5_ablation_summary.csv", False, "ablation_summary"))

    # aco_history provides a distribution (raw_J)
    for r in v14_5_hist:
        d = safe_int(r.get("dim", 0))
        if d not in dims:
            continue
        J = safe_float(r.get("raw_J", float("nan")))
        if math.isfinite(J):
            baseline_rows.append(BaselineRow(d, "v14_5", "prior_artin", "v14_5_history", str(r.get("mode", "")) or "unknown", float(J), "v14_5_aco_history.csv", False, "aco_history"))

    # --- V14.2 ---
    v14_2_gate = _read_csv_best_effort(v14_2_dir / "v14_2_gate_summary.csv")
    v14_2_best = _read_csv_best_effort(v14_2_dir / "v14_2_best_candidates.csv")
    v14_2_hist = _read_csv_best_effort(v14_2_dir / "v14_2_aco_history.csv")
    if not v14_2_gate:
        missing_sources.append("v14_2_gate_summary.csv")
    if not v14_2_best:
        missing_sources.append("v14_2_best_candidates.csv")
    if not v14_2_hist:
        missing_sources.append("v14_2_aco_history.csv")

    # best_candidates: J_v14_2 or similar
    for r in v14_2_best:
        d = safe_int(r.get("dim", 0))
        if d not in dims:
            continue
        J = pick_first_present(r, ["J_v14_2", "J_total", "best_J", "J"])
        if math.isfinite(J):
            baseline_rows.append(BaselineRow(d, "v14_2", "prior_artin", "best_v14_2", "best_of", float(J), "v14_2_best_candidates.csv", False, "best_candidates"))

    # history: staged J might be J_total or J_v14_2
    for r in v14_2_hist:
        d = safe_int(r.get("dim", 0))
        if d not in dims:
            continue
        J = pick_first_present(r, ["raw_J", "J_v14_2", "J_total", "J"])
        if math.isfinite(J):
            baseline_rows.append(BaselineRow(d, "v14_2", "prior_artin", "v14_2_history", "aco_history", float(J), "v14_2_aco_history.csv", False, "aco_history"))

    # If any required file missing, add placeholder baseline rows marking missing_source=True for that dim
    def add_missing_placeholders(source: str, dim: int, labels: List[str]) -> None:
        for lab in labels:
            baseline_rows.append(
                BaselineRow(
                    dim=dim,
                    source=source,
                    baseline_kind="missing",
                    label=lab,
                    mode="",
                    J=float("nan"),
                    source_file="",
                    missing_source=True,
                    notes="missing source file(s)",
                )
            )

    for d in dims:
        if not v13_mode and not v13_null:
            add_missing_placeholders("v13o14", d, ["random", "rejected", "ablation"])
        if not v14_5_abl and not v14_5_hist:
            add_missing_placeholders("v14_5", d, ["prior_artin"])
        if not v14_2_best and not v14_2_hist:
            add_missing_placeholders("v14_2", d, ["prior_artin"])

    # Baseline pool rows normalized
    baseline_pool_rows = normalize_baseline_rows(baseline_rows)
    write_csv(
        out_dir / "v14_6b_prior_baseline_pool.csv",
        fieldnames=["dim", "source", "baseline_kind", "label", "mode", "J", "source_file", "missing_source", "notes"],
        rows=baseline_pool_rows,
    )

    # Comparisons per dim/mode
    comparisons: List[Dict[str, Any]] = []
    gate_rows: List[Dict[str, Any]] = []

    # prepare V14.6 prior gate map (optional)
    v14_6_gate_map: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for r in v14_6_gate:
        d = safe_int(r.get("dim", 0))
        if d not in dims:
            continue
        mode = str(r.get("mode", "")).strip()
        if mode:
            v14_6_gate_map[(d, mode)] = r

    def best_baseline(dim: int, kinds: Sequence[str]) -> Optional[float]:
        Js = []
        for r in baseline_rows:
            if int(r.dim) != int(dim):
                continue
            if r.missing_source:
                continue
            if r.baseline_kind in kinds:
                if math.isfinite(float(r.J)):
                    Js.append(float(r.J))
        return float(min(Js)) if Js else None

    def pool_Js(dim: int) -> List[float]:
        xs = []
        for r in baseline_rows:
            if int(r.dim) != int(dim):
                continue
            if r.missing_source:
                continue
            if r.baseline_kind == "primary":
                continue
            if math.isfinite(float(r.J)):
                xs.append(float(r.J))
        return xs

    keys = sorted(best_gan.keys())
    for idx, (d, mode) in enumerate(keys):
        if (idx + 1) % progress_every == 0:
            print(f"[V14.6b] progress {idx+1}/{len(keys)} dim={d} mode={mode}", flush=True)

        gan = best_gan[(d, mode)]
        best_gan_J = float(gan["best_gan_J"])
        best_random_J = best_baseline(d, ["random"])
        best_rejected_J = best_baseline(d, ["rejected"])
        best_ablation_J = best_baseline(d, ["ablation"])
        best_prior_artin_J = best_baseline(d, ["prior_artin"])
        # best_null_J over all non-primary null/prior controls
        best_null_J = best_baseline(d, ["random", "rejected", "ablation", "prior_artin", "other"])
        null_sep = float(best_null_J - best_gan_J) if (best_null_J is not None and math.isfinite(best_gan_J)) else float("nan")
        z, pct = compute_null_stats(best_gan_J, pool_Js(d))

        margin = float(args.null_separation_margin)
        beats_random = bool(best_random_J is not None and best_gan_J < float(best_random_J) - margin)
        beats_rejected = bool(best_rejected_J is not None and best_gan_J < float(best_rejected_J) - margin)
        beats_ablation = bool(best_ablation_J is not None and best_gan_J < float(best_ablation_J) - margin)
        beats_prior_artin = bool(best_prior_artin_J is not None and best_gan_J < float(best_prior_artin_J) - margin)

        null_sep_ok = bool(math.isfinite(null_sep) and null_sep > float(args.null_separation_margin))
        null_z_ok = bool(math.isfinite(z) and z >= float(args.null_zscore_margin))

        missing_reason = []
        if best_random_J is None:
            missing_reason.append("missing_random_controls")
        if best_rejected_J is None:
            missing_reason.append("missing_rejected_control")
        if best_ablation_J is None:
            missing_reason.append("missing_ablation_controls")
        if best_prior_artin_J is None:
            missing_reason.append("missing_prior_artin")
        if len(pool_Js(d)) < 3:
            missing_reason.append("null_distribution_too_small(<3)")

        comparisons.append(
            {
                "dim": int(d),
                "mode": str(mode),
                "best_gan_J": float(best_gan_J),
                "best_random_J": float(best_random_J) if best_random_J is not None else float("nan"),
                "best_rejected_J": float(best_rejected_J) if best_rejected_J is not None else float("nan"),
                "best_ablation_J": float(best_ablation_J) if best_ablation_J is not None else float("nan"),
                "best_prior_artin_J": float(best_prior_artin_J) if best_prior_artin_J is not None else float("nan"),
                "best_null_J": float(best_null_J) if best_null_J is not None else float("nan"),
                "gan_minus_best_random": float(best_gan_J - best_random_J) if best_random_J is not None else float("nan"),
                "gan_minus_rejected": float(best_gan_J - best_rejected_J) if best_rejected_J is not None else float("nan"),
                "gan_minus_best_ablation": float(best_gan_J - best_ablation_J) if best_ablation_J is not None else float("nan"),
                "gan_minus_prior_artin": float(best_gan_J - best_prior_artin_J) if best_prior_artin_J is not None else float("nan"),
                "null_separation": float(null_sep),
                "null_zscore": float(z),
                "null_percentile": float(pct),
                "beats_random": bool(beats_random),
                "beats_rejected": bool(beats_rejected),
                "beats_ablation": bool(beats_ablation),
                "beats_prior_artin": bool(beats_prior_artin),
                "null_separation_ok": bool(null_sep_ok),
                "null_zscore_ok": bool(null_z_ok),
                "missing_baseline_reason": "|".join(missing_reason),
            }
        )

        # Preserve prior V14.6 gates if present
        g0 = v14_6_gate_map.get((d, mode), {})
        prior_all = bool(gan.get("prior_v14_6_all_gate_pass", False))
        # user asked to preserve these gate names where available
        G1 = str(g0.get("G1_stable_operator", g0.get("G1_stable", ""))).lower() == "true"
        G2 = str(g0.get("G2_support_overlap_ok", "")).lower() == "true"
        G3 = str(g0.get("G3_active_argument_ok", "")).lower() == "true"
        G4 = str(g0.get("G4_residue_error_ok", "")).lower() == "true"
        G5 = str(g0.get("G6_trace_proxy_ok", g0.get("G5_trace_proxy_ok", ""))).lower() == "true"
        G6 = str(g0.get("G4_number_variance_ok", g0.get("G6_transport_or_nv_ok", ""))).lower() == "true"
        G7 = str(g0.get("G7_not_poisson_like", "")).lower() == "true"

        # Stage E gates
        E8 = bool(beats_random) if best_random_J is not None else False
        E9 = bool(beats_rejected) if best_rejected_J is not None else False
        E10 = bool(beats_ablation) if best_ablation_J is not None else False
        E11 = bool(beats_prior_artin) if best_prior_artin_J is not None else False
        E12 = bool(null_sep_ok)
        E13 = bool(null_z_ok)

        # requirements
        req_ok = True
        if args.require_beats_random:
            req_ok = req_ok and E8
        if args.require_beats_rejected:
            req_ok = req_ok and E9
        if args.require_beats_ablation:
            req_ok = req_ok and E10
        if args.require_beats_prior_artin:
            req_ok = req_ok and E11
        # always require G12 (null separation) as per user
        req_ok = req_ok and E12

        all_gate_pass = bool(prior_all and req_ok and E13)

        # classification
        if not prior_all:
            cls = "FAIL_PRIOR_V14_6_GATE"
        elif any("missing_" in x for x in missing_reason):
            cls = "MISSING_BASELINES"
        elif not E12:
            cls = "FAIL_NULL_SEPARATION"
        elif not E13:
            cls = "FAIL_NULL_ZSCORE"
        elif args.require_beats_random and not E8:
            cls = "FAIL_RANDOM_CONTROL"
        elif args.require_beats_rejected and not E9:
            cls = "FAIL_REJECTED_CONTROL"
        elif args.require_beats_ablation and not E10:
            cls = "FAIL_ABLATION_CONTROL"
        elif args.require_beats_prior_artin and not E11:
            cls = "FAIL_PRIOR_ARTIN"
        elif all_gate_pass:
            cls = "PASS_REAL_NULL_STAGE_E"
        else:
            cls = "FAIL_REAL_NULL_STAGE_E"

        gate_rows.append(
            {
                "dim": int(d),
                "mode": str(mode),
                "word": str(gan.get("word", "")),
                "best_gan_J": float(best_gan_J),
                "prior_v14_6_all_gate_pass": bool(prior_all),
                "G1_stable": bool(G1),
                "G2_support_overlap_ok": bool(G2),
                "G3_active_argument_ok": bool(G3),
                "G4_residue_error_ok": bool(G4),
                "G5_trace_proxy_ok": bool(G5),
                "G6_transport_or_nv_ok": bool(G6),
                "G7_not_poisson_like": bool(G7),
                "G8_beats_random_controls": bool(E8),
                "G9_beats_rejected_control": bool(E9),
                "G10_beats_ablation_controls": bool(E10),
                "G11_beats_prior_artin_search": bool(E11),
                "G12_null_separation_ok": bool(E12),
                "G13_null_zscore_ok": bool(E13),
                "null_separation": float(null_sep),
                "null_zscore": float(z),
                "null_percentile": float(pct),
                "missing_baseline_reason": "|".join(missing_reason),
                "all_gate_pass": bool(all_gate_pass),
                "classification": str(cls),
            }
        )

    write_csv(
        out_dir / "v14_6b_real_null_comparisons.csv",
        fieldnames=list(comparisons[0].keys()) if comparisons else ["dim", "mode", "best_gan_J"],
        rows=comparisons,
    )
    write_csv(
        out_dir / "v14_6b_real_null_gate_summary.csv",
        fieldnames=list(gate_rows[0].keys()) if gate_rows else ["dim", "mode", "all_gate_pass"],
        rows=gate_rows,
    )

    # Decision summary
    any_pass = any(bool(r.get("all_gate_pass", False)) for r in gate_rows)
    decision_rows = [
        {
            "proceed_to_v14_7": bool(any_pass),
            "analytic_claim": False,
            "n_gate_rows": int(len(gate_rows)),
            "n_all_gate_pass": int(sum(1 for r in gate_rows if bool(r.get("all_gate_pass", False)))),
            "missing_sources": "|".join(sorted(set(missing_sources))),
        }
    ]
    write_csv(
        out_dir / "v14_6b_decision_summary.csv",
        fieldnames=list(decision_rows[0].keys()),
        rows=decision_rows,
    )

    results = {
        "version": "v14_6b",
        "computational_only": True,
        "v14_6_dir": str(v14_6_dir),
        "v13o14_dir": str(v13o14_dir),
        "v14_5_dir": str(v14_5_dir),
        "v14_2_dir": str(v14_2_dir),
        "out_dir": str(out_dir),
        "dims": dims,
        "require_flags": {
            "require_beats_random": bool(args.require_beats_random),
            "require_beats_rejected": bool(args.require_beats_rejected),
            "require_beats_ablation": bool(args.require_beats_ablation),
            "require_beats_prior_artin": bool(args.require_beats_prior_artin),
        },
        "margins": {"null_separation_margin": float(args.null_separation_margin), "null_zscore_margin": float(args.null_zscore_margin)},
        "missing_sources": sorted(set(missing_sources)),
        "decision_summary": decision_rows,
    }
    write_text(out_dir / "v14_6b_results.json", json.dumps(json_sanitize(results), indent=2) + "\n")

    # Report
    OUT_ABS = str(out_dir.resolve())
    md = []
    md.append("# V14.6b — Real Null-Control Stage E\n\n")
    md.append("> Computational evidence only; not a proof of RH.\n\n")
    md.append("## Purpose\n")
    md.append("This script replaces placeholder null controls in V14.6 by loading real null/prior baseline pools (V13O.14, V14.2, V14.5) and recomputing Stage-E gates.\n\n")
    md.append("## Missing data policy\n")
    md.append("- Missing sources do **not** crash the run.\n")
    md.append("- Missing baseline group => corresponding gate is **False** and `missing_baseline_reason` records why.\n")
    md.append("- If null distribution has <3 values => `null_zscore`/`null_percentile` are NaN and `G13_null_zscore_ok=False`.\n\n")
    md.append("## Decision\n")
    md.append(f"- proceed_to_v14_7: **{decision_rows[0]['proceed_to_v14_7']}**\n")
    md.append("- analytic_claim: **False**\n\n")
    md.append("## Outputs\n")
    md.append("- `v14_6b_prior_baseline_pool.csv`\n")
    md.append("- `v14_6b_real_null_comparisons.csv`\n")
    md.append("- `v14_6b_real_null_gate_summary.csv`\n")
    md.append("- `v14_6b_decision_summary.csv`\n\n")
    md.append("## Verification commands\n")
    md.append("```bash\n")
    md.append(f'OUT="{OUT_ABS}"\n\n')
    md.append('echo "=== FILES ==="\nfind "$OUT" -maxdepth 1 -type f | sort\n\n')
    md.append('echo "=== NULL GATE ==="\ncolumn -s, -t < "$OUT"/v14_6b_real_null_gate_summary.csv | head -120\n\n')
    md.append('echo "=== NULL COMPARISONS ==="\ncolumn -s, -t < "$OUT"/v14_6b_real_null_comparisons.csv | head -120\n\n')
    md.append('echo "=== BASELINE POOL ==="\ncolumn -s, -t < "$OUT"/v14_6b_prior_baseline_pool.csv | head -120\n\n')
    md.append('echo "=== DECISION ==="\ncolumn -s, -t < "$OUT"/v14_6b_decision_summary.csv\n\n')
    md.append('echo "=== REPORT ==="\nhead -240 "$OUT"/v14_6b_report.md\n')
    md.append("```\n\n")
    md.append("> Computational evidence only; not a proof of RH.\n")
    write_text(out_dir / "v14_6b_report.md", "".join(md))

    tex = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\begin{document}
\section*{V14.6b --- Real Null-Control Stage E}
\textbf{Computational evidence only; not a proof of RH.}

\subsection*{Summary}
This report post-processes V14.6 by loading real null/prior baselines (V13O.14, V14.2, V14.5)
and recomputing Stage-E gates. Missing baselines never silently pass.
\end{document}
"""
    tex_path = out_dir / "v14_6b_report.tex"
    write_text(tex_path, tex)
    try_pdflatex(tex_path, out_dir, "v14_6b_report.pdf")


if __name__ == "__main__":
    main()

