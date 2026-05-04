#!/usr/bin/env python3
"""
V13L.2: Pareto-stabilized self-consistent RH iteration (grid over schedule / smooth / clip / λ_p).

  python3 scripts/run_v13l2_pareto_fixed.py \\
    --candidate_json runs/v13_candidate_theorem/candidate_operators.json \\
    --out_dir runs/v13l2_pareto_fixed \\
    --dim 128 --zeros 128 --seed 42 --max_iter 200 --tol 1e-3
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

if not os.environ.get("MPLCONFIGDIR"):
    _mpl_cfg = Path(ROOT) / ".mpl_cache"
    try:
        _mpl_cfg.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(_mpl_cfg)
    except OSError:
        pass

try:
    import matplotlib

    matplotlib.use("Agg")
except Exception:
    pass

_DTF = np.float64

PRIMARY_ID = "seed_6"
PRIMARY_WORD = [-4, -2, -4, -2, -2, -1, -1]
ALPHA = 0.5

BETA0_GRID = [0.15, 0.2, 0.25, 0.3]
TAU_GRID = [25, 50, 100, 200]
SIGMA_GRID = [1.0, 2.0, 4.0]
CLIP_GRID: List[Tuple[float, float]] = [(0.5, 99.5), (1.0, 99.0), (2.0, 98.0)]
LAMBDA_P_EFF_GRID = [2.0, 2.5, 3.0]

SUMMARY_FIELDS = [
    "beta0",
    "tau",
    "smooth_sigma",
    "clip_lo",
    "clip_hi",
    "lambda_p_eff",
    "best_iter_by_J",
    "best_J",
    "final_operator_diff",
    "converged_operator",
    "stopped_stagnation",
    "best_spectral_log_mse",
    "best_spacing_mse_normalized",
    "best_ks_wigner",
    "best_operator_diff",
    "meets_all_at_best_checkpoint",
    "checkpoint_uid",
    "checkpoint_path",
    "eig_error",
]


def _load_v13_validate() -> Any:
    path = Path(ROOT) / "scripts" / "validate_candidates_v13.py"
    spec = importlib.util.spec_from_file_location("_v13_validate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_v13_validate"] = mod
    spec.loader.exec_module(mod)
    return mod


def latex_escape(s: str) -> str:
    t = str(s)
    t = t.replace("\\", "\\textbackslash{}")
    t = t.replace("{", "\\{").replace("}", "\\}")
    t = t.replace("_", "\\_")
    t = t.replace("%", "\\%")
    t = t.replace("#", "\\#")
    t = t.replace("&", "\\&")
    t = t.replace("$", "\\$")
    t = t.replace("^", "\\textasciicircum{}")
    t = t.replace("~", "\\textasciitilde{}")
    return t


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


def operator_matrix_sha256(H: np.ndarray) -> str:
    hc = np.ascontiguousarray(np.asarray(H, dtype=np.float64))
    return hashlib.sha256(hc.tobytes()).hexdigest()


def metrics_from_H(v: Any, v13l: Any, H: np.ndarray, zeros: np.ndarray) -> Dict[str, Any]:
    from core import v13l1_stabilized as v13l1mod

    H = np.asarray(H, dtype=_DTF, copy=False)
    if not np.isfinite(H).all():
        return {"finite": False, "eig_error": "nonfinite_H"}
    try:
        eig = np.sort(np.linalg.eigvalsh(0.5 * (H + H.T)).astype(_DTF))
    except Exception as ex:
        return {"finite": False, "eig_error": repr(ex)}
    if not np.isfinite(eig).all():
        return {"finite": False, "eig_error": "nonfinite_eig"}
    m = v13l.spectral_metrics(
        eig,
        zeros,
        spacing_fn=v.spacing_mse_normalized,
        ks_fn=v.ks_against_wigner_gue,
        norm_gaps_fn=v.normalized_gaps,
    )
    out = {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in m.items()}
    out["eig_min"] = float(np.min(eig))
    out["eig_max"] = float(np.max(eig))
    out["self_adjointness_fro"] = float(v13l1mod.self_adjointness_fro(H))
    out["finite"] = True
    out["eig_error"] = None
    return out


def checkpoint_uid(
    beta0: float,
    tau: float,
    sig: float,
    clo: float,
    chi: float,
    lp: float,
) -> str:
    key = json.dumps(
        {"b0": beta0, "tau": tau, "sig": sig, "clo": clo, "chi": chi, "lp": lp},
        sort_keys=True,
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]


def acceptance_failures(
    od: float,
    sl: float,
    sp: float,
    ks: float,
) -> Dict[str, bool]:
    return {
        "operator_diff_gt_1e-3": not (math.isfinite(od) and od <= 1e-3),
        "spectral_log_mse_gt_7_48": not (math.isfinite(sl) and sl <= 7.48),
        "spacing_mse_normalized_gt_16_84": not (math.isfinite(sp) and sp <= 16.84),
        "ks_wigner_gt_0_451": not (math.isfinite(ks) and ks <= 0.451),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="V13L.2 Pareto-stabilized self-consistent grid.")
    ap.add_argument("--candidate_json", type=str, default="runs/v13_candidate_theorem/candidate_operators.json")
    ap.add_argument("--out_dir", type=str, default="runs/v13l2_pareto_fixed")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--zeros", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_iter", type=int, default=200)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--eps", type=float, default=0.6)
    ap.add_argument("--geo_sigma", type=float, default=0.6)
    ap.add_argument("--geo_weight", type=float, default=10.0)
    ap.add_argument("--potential_weight_v13j", type=float, default=0.25)
    args = ap.parse_args()

    cand_path = Path(args.candidate_json)
    if not cand_path.is_absolute():
        cand_path = Path(ROOT) / cand_path
    if not cand_path.is_file():
        raise SystemExit(f"missing --candidate_json: {cand_path}")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = Path(ROOT) / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    ck_dir = out_dir / "checkpoints"
    ck_dir.mkdir(parents=True, exist_ok=True)

    v = _load_v13_validate()
    from core import v13l2_pareto as v13l2
    from core import v13l_self_consistent as v13l
    from core import v13l1_stabilized as v13l1
    from core.artin_operator import sample_domain

    try:
        from core.artin_operator_word_sensitive import build_word_sensitive_operator
    except ImportError as e:
        raise SystemExit(f"word_sensitive builder required: {e}") from e

    zeros = v._load_zeros(int(args.zeros))
    if zeros.size < int(args.dim):
        raise ValueError(f"need zeros >= dim, got {zeros.size} < {args.dim}")

    Z = sample_domain(int(args.dim), seed=int(args.seed))
    word = list(PRIMARY_WORD)
    geodesics = [v13l.geodesic_entry_for_word(word)]

    with open(cand_path, "r", encoding="utf-8") as f:
        cj = json.load(f)
    sc = cj.get("spectral_candidate") if isinstance(cj.get("spectral_candidate"), dict) else {}
    if str(sc.get("id", "")) == PRIMARY_ID and isinstance(sc.get("word"), list):
        wj = [int(x) for x in sc["word"]]
        if wj != word:
            print(f"[v13l2] note: JSON seed_6 word {wj} differs from primary {word}; using primary word.", flush=True)

    H_base, _ = v13l.build_h_base_no_potential(
        z_points=Z,
        geodesics=geodesics,
        eps=float(args.eps),
        geo_sigma=float(args.geo_sigma),
        geo_weight=float(args.geo_weight),
        distances=None,
        diag_shift=1e-6,
    )

    H_v13j, _ = build_word_sensitive_operator(
        z_points=Z,
        distances=None,
        geodesics=geodesics,
        eps=float(args.eps),
        geo_sigma=float(args.geo_sigma),
        geo_weight=float(args.geo_weight),
        potential_weight=float(args.potential_weight_v13j),
        diag_shift=1e-6,
    )
    v13j_metrics = metrics_from_H(v, v13l, np.asarray(H_v13j, dtype=_DTF), zeros)

    H_l, _, rows_l, meta_l = v13l.run_self_consistent_loop(
        H_base=H_base,
        gamma=zeros,
        alpha=float(ALPHA),
        lambda_p=3.0,
        beta=0.3,
        max_iter=int(args.max_iter),
        tol=float(args.tol),
        diag_shift=1e-6,
        smooth_sigma=2.0,
        use_smooth=True,
        zeros_eval=zeros,
        spacing_fn=v.spacing_mse_normalized,
        ks_fn=v.ks_against_wigner_gue,
        norm_gaps_fn=v.normalized_gaps,
    )
    fin_l = rows_l[-1] if rows_l else {}
    v13l_metrics = {
        "label": "V13L fixed-beta (β=0.3, λ_p=3)",
        "spectral_log_mse_final": float(fin_l.get("spectral_log_mse", float("nan"))),
        "spacing_mse_normalized_final": float(fin_l.get("spacing_mse_normalized", float("nan"))),
        "ks_wigner_final": float(fin_l.get("ks_wigner", float("nan"))),
        "n_iter": int(meta_l.get("n_iter", 0)),
        "converged": bool(meta_l.get("converged", False)),
    }

    H_l1, _, rows_l1, meta_l1 = v13l1.run_stabilized_self_consistent(
        H_base=H_base,
        gamma=zeros,
        alpha=float(ALPHA),
        lambda_p=3.0,
        max_iter=int(args.max_iter),
        tol=float(args.tol),
        diag_shift=1e-6,
        beta0=0.3,
        tau_beta=50.0,
        beta_floor=0.05,
        delta_smooth_sigmas=(2.0, 4.0, 8.0),
        clip_percentiles=(1.0, 99.0),
        zeros_eval=zeros,
        spacing_fn=v.spacing_mse_normalized,
        ks_fn=v.ks_against_wigner_gue,
        norm_gaps_fn=v.normalized_gaps,
    )
    fin_l1 = rows_l1[-1] if rows_l1 else {}
    v13l1_metrics = {
        "label": "V13L.1 stabilized (defaults)",
        "spectral_log_mse_final": float(fin_l1.get("spectral_log_mse", float("nan"))),
        "spacing_mse_normalized_final": float(fin_l1.get("spacing_mse_normalized", float("nan"))),
        "ks_wigner_final": float(fin_l1.get("ks_wigner", float("nan"))),
        "operator_diff_final": float(fin_l1.get("operator_diff", float("nan"))),
        "n_iter": int(meta_l1.get("n_iter", 0)),
        "converged_operator": bool(meta_l1.get("converged_operator", False)),
    }

    summary_rows: List[Dict[str, Any]] = []
    all_traces: Dict[str, List[Dict[str, Any]]] = {}
    total = len(BETA0_GRID) * len(TAU_GRID) * len(SIGMA_GRID) * len(CLIP_GRID) * len(LAMBDA_P_EFF_GRID)
    done = 0

    global_meets: List[Dict[str, Any]] = []
    global_by_j: List[Dict[str, Any]] = []

    for b0 in BETA0_GRID:
        for tau in TAU_GRID:
            for sig in SIGMA_GRID:
                for clo, chi in CLIP_GRID:
                    for lp_eff in LAMBDA_P_EFF_GRID:
                        uid = checkpoint_uid(b0, tau, sig, clo, chi, lp_eff)
                        ck_path = ck_dir / f"{uid}_best_J.npy"

                        out = v13l2.run_pareto_cell(
                            H_base=H_base,
                            gamma=zeros,
                            alpha=float(ALPHA),
                            lambda_p_eff=float(lp_eff),
                            beta0=float(b0),
                            tau_beta=float(tau),
                            beta_floor=0.03,
                            smooth_sigma=float(sig),
                            clip_percentiles=(float(clo), float(chi)),
                            max_iter=int(args.max_iter),
                            tol=float(args.tol),
                            diag_shift=1e-6,
                            abs_cap_factor=5.0,
                            zeros_eval=zeros,
                            spacing_fn=v.spacing_mse_normalized,
                            ks_fn=v.ks_against_wigner_gue,
                            norm_gaps_fn=v.normalized_gaps,
                        )
                        np.save(str(ck_path), np.asarray(out["H_best_J"], dtype=_DTF))

                        meta = out.get("meta") or {}
                        row = {
                            "beta0": float(b0),
                            "tau": float(tau),
                            "smooth_sigma": float(sig),
                            "clip_lo": float(clo),
                            "clip_hi": float(chi),
                            "lambda_p_eff": float(lp_eff),
                            "best_iter_by_J": int(out["best_iter_by_J"]),
                            "best_J": float(out["best_J"]),
                            "final_operator_diff": float(out["final_operator_diff"]),
                            "converged_operator": bool(meta.get("converged_operator", False)),
                            "stopped_stagnation": bool(meta.get("stopped_stagnation", False)),
                            "best_spectral_log_mse": float(out["best_spectral_log_mse"]),
                            "best_spacing_mse_normalized": float(out["best_spacing_mse_normalized"]),
                            "best_ks_wigner": float(out["best_ks_wigner"]),
                            "best_operator_diff": float(out["best_operator_diff"]),
                            "meets_all_at_best_checkpoint": bool(out["meets_all_at_best_checkpoint"]),
                            "checkpoint_uid": uid,
                            "checkpoint_path": str(ck_path.relative_to(out_dir)),
                            "eig_error": meta.get("eig_error"),
                        }
                        summary_rows.append(row)
                        entry = dict(row)
                        global_by_j.append(entry)
                        if out["meets_all_at_best_checkpoint"]:
                            global_meets.append(entry)

                        all_traces[uid] = list(out["rows"])

                        done += 1
                        if done % 50 == 0 or done == total:
                            print(f"[v13l2] progress {done}/{total}", flush=True)

    def j_key(r: Dict[str, Any]) -> float:
        x = r.get("best_J")
        return float(x) if isinstance(x, (int, float)) and math.isfinite(float(x)) else float("inf")

    feasible = [r for r in global_meets if j_key(r) < float("inf")]
    if feasible:
        winner = min(feasible, key=j_key)
        selection = "feasible_min_J"
    else:
        winner = min(global_by_j, key=j_key) if global_by_j else {}
        selection = "pareto_surrogate_min_best_J"

    win_uid = str(winner.get("checkpoint_uid", ""))
    win_rows = all_traces.get(win_uid, [])
    if winner.get("checkpoint_path"):
        H_pareto_best = np.load(str(out_dir / str(winner["checkpoint_path"])))
        np.save(str(out_dir / "h_star_pareto_best.npy"), np.asarray(H_pareto_best, dtype=_DTF))
        pareto_best_metrics = metrics_from_H(v, v13l, H_pareto_best, zeros)
        pareto_best_hash = operator_matrix_sha256(H_pareto_best)
    else:
        pareto_best_metrics = {}
        pareto_best_hash = ""

    ff = acceptance_failures(
        float(winner.get("best_operator_diff", float("nan"))),
        float(winner.get("best_spectral_log_mse", float("nan"))),
        float(winner.get("best_spacing_mse_normalized", float("nan"))),
        float(winner.get("best_ks_wigner", float("nan"))),
    )

    v13l2_best_summary = {
        "selection_rule": selection,
        "winner_hyperparameters": {k: winner.get(k) for k in SUMMARY_FIELDS},
        "acceptance_failures_at_winner_best_checkpoint": ff,
        "meets_all_at_best_checkpoint": bool(winner.get("meets_all_at_best_checkpoint", False)),
        "operator_hash": pareto_best_hash,
        "metrics_on_loaded_H": pareto_best_metrics,
    }

    conv_fields = [
        "iter",
        "spectral_log_mse",
        "spacing_mse_normalized",
        "ks_wigner",
        "operator_diff",
        "delta_norm",
        "eig_min",
        "eig_max",
        "pareto_objective",
    ]
    with open(out_dir / "v13l2_convergence_best.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=conv_fields, extrasaction="ignore")
        w.writeheader()
        for r in win_rows:
            w.writerow({k: r.get(k) for k in conv_fields})

    with open(out_dir / "v13l2_summary.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in summary_rows:
            w.writerow({k: r.get(k) for k in SUMMARY_FIELDS})

    payload = {
        "meta": {
            "candidate_id": PRIMARY_ID,
            "word": word,
            "alpha": ALPHA,
            "grid": {
                "beta0": BETA0_GRID,
                "tau": TAU_GRID,
                "smooth_sigma": SIGMA_GRID,
                "clip_percentiles": [list(x) for x in CLIP_GRID],
                "lambda_p_eff": LAMBDA_P_EFF_GRID,
                "n_cells": int(total),
            },
            "dim": int(args.dim),
            "seed": int(args.seed),
            "max_iter": int(args.max_iter),
            "tol": float(args.tol),
        },
        "comparison": {
            "v13j_baseline": v13j_metrics,
            "v13l_fixed_beta": v13l_metrics,
            "v13l1_stabilized_defaults": v13l1_metrics,
            "v13l2_pareto_best": v13l2_best_summary,
        },
        "pareto_objective_definition": "spectral_log_mse + 0.05*spacing_mse_normalized + ks_wigner + 0.1*min(operator_diff,1)",
        "acceptance_thresholds": {
            "operator_diff": 1e-3,
            "spectral_log_mse": 7.48,
            "spacing_mse_normalized": 16.84,
            "ks_wigner": 0.451,
        },
        "n_feasible_cells": len(feasible),
        "winner": v13l2_best_summary,
        "summary": summary_rows,
    }
    with open(out_dir / "v13l2_results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=True)

    md = [
        "# V13L.2: Pareto-stabilized self-consistent operator\n\n",
        "## Comparison (same Z, primary word)\n\n",
        "| Model | spectral (repr.) | spacing (norm) | KS | notes |\n",
        "|---|---:|---:|---:|---|\n",
        f"| V13J baseline | {v13j_metrics.get('spectral_log_mse', float('nan'))} | "
        f"{v13j_metrics.get('spacing_mse_normalized', float('nan'))} | {v13j_metrics.get('ks_wigner', float('nan'))} | sin V |\n",
        f"| V13L fixed-β | {v13l_metrics.get('spectral_log_mse_final', float('nan'))} | "
        f"{v13l_metrics.get('spacing_mse_normalized_final', float('nan'))} | {v13l_metrics.get('ks_wigner_final', float('nan'))} | final iter |\n",
        f"| V13L.1 default | {v13l1_metrics.get('spectral_log_mse_final', float('nan'))} | "
        f"{v13l1_metrics.get('spacing_mse_normalized_final', float('nan'))} | {v13l1_metrics.get('ks_wigner_final', float('nan'))} | stabilized |\n",
        f"| **V13L.2 Pareto best** | **{winner.get('best_spectral_log_mse', float('nan'))}** | "
        f"**{winner.get('best_spacing_mse_normalized', float('nan'))}** | **{winner.get('best_ks_wigner', float('nan'))}** | best-J checkpoint |\n\n",
        f"- Selection: **{selection}** (among cells meeting all thresholds, minimize `best_J`; else global min `best_J`).\n",
        f"- Winner meets all at best checkpoint: **{winner.get('meets_all_at_best_checkpoint')}**.\n",
        f"- Failed gates at winner (if any): {json.dumps(ff)}.\n\n",
        "## Outputs\n\n",
        "`v13l2_summary.csv`, `v13l2_convergence_best.csv` (winning hyperparameters), `checkpoints/*.npy`, `h_star_pareto_best.npy`.\n\n",
    ]
    (out_dir / "v13l2_report.md").write_text("".join(md), encoding="utf-8")

    tex = [
        "\\documentclass[11pt]{article}\n",
        "\\usepackage{amsmath,amssymb}\n",
        "\\usepackage[T1]{fontenc}\n",
        "\\usepackage[margin=1in]{geometry}\n",
        "\\title{V13L.2 Pareto-stabilized}\n\\date{}\n\\begin{document}\n\\maketitle\n",
        "\\section{Selection}\n",
        latex_escape(selection) + "\n\n",
        "\\section{Acceptance failures at winner checkpoint}\n",
        latex_escape(json.dumps(ff)) + "\n\n",
        "\\end{document}\n",
    ]
    (out_dir / "v13l2_report.tex").write_text("".join(tex), encoding="utf-8")

    if try_pdflatex(out_dir / "v13l2_report.tex", out_dir, "v13l2_report.pdf"):
        print(f"Wrote {out_dir / 'v13l2_report.pdf'}")
    else:
        print("PDF skipped.")

    print(f"Wrote {out_dir / 'v13l2_results.json'} ({len(summary_rows)} grid cells)")
    print(f"Wrote {out_dir / 'v13l2_summary.csv'}")
    print(f"Wrote {out_dir / 'v13l2_convergence_best.csv'} ({len(win_rows)} rows)")
    print(f"Wrote {out_dir / 'h_star_pareto_best.npy'}")


if __name__ == "__main__":
    main()