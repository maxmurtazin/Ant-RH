#!/usr/bin/env python3
"""
V13M.2: convergence polish on fixed V13M.1 best renormalized configs (dim 128 and 256).

  python3 scripts/run_v13m2_convergence_polish.py \\
    --out_dir runs/v13m2_convergence_polish \\
    --seed 42 --max_iter 500 --tol 1e-3
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
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

_DTF = np.float64

PRIMARY_WORD = [-4, -2, -4, -2, -2, -1, -1]

ALPHA = 0.5
EPS = 0.6
GEO_WEIGHT_BASE = 10.0

# Fixed V13M.1 best configs (explicit numerics; do not re-derive from q/r/s grids).
POLISH_TARGETS: List[Dict[str, Any]] = [
    {
        "dim": 128,
        "zeros_eff": 96,
        "renorm_family": "manual",
        "lambda_p_dim": 3.0,
        "geo_sigma_dim": 0.6,
        "smooth_sigma_base": 1.0,
    },
    {
        "dim": 256,
        "zeros_eff": 128,
        "renorm_family": "fixed128",
        "lambda_p_dim": 2.121320343559643,
        "geo_sigma_dim": 0.5045378491522287,
        "smooth_sigma_base": 1.0,
    },
]

TAU_GRID = [200.0, 300.0, 500.0]
BETA_FLOOR_GRID = [0.03, 0.02, 0.01]
BETA0_GRID = [0.2, 0.25, 0.3]
CLIP_GRID: List[Tuple[float, float]] = [(1.0, 99.0), (0.5, 99.5)]
SMOOTH_MULT_GRID = [1.0, 1.25, 1.5]

SA_TOL = 1e-12


def _load_v13_validate() -> Any:
    path = Path(ROOT) / "scripts" / "validate_candidates_v13.py"
    spec = importlib.util.spec_from_file_location("_v13_validate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_v13_validate"] = mod
    spec.loader.exec_module(mod)
    return mod


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
    t = t.replace("#", "\\#")
    t = t.replace("&", "\\&")
    t = t.replace("$", "\\$")
    t = t.replace("^", "\\textasciicircum{}")
    t = t.replace("~", "\\textasciitilde{}")
    return t


def numpy_json(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return float(x) if isinstance(x, np.floating) else int(x)
    if isinstance(x, dict):
        return {k: numpy_json(v) for k, v in x.items()}
    if isinstance(x, list):
        return [numpy_json(v) for v in x]
    if isinstance(x, tuple):
        return [numpy_json(v) for v in x]
    return x


def meets_acceptance_v13m2(
    *,
    operator_diff: float,
    spacing_mse_normalized: float,
    ks_wigner: float,
    finite: bool,
    self_adjointness_fro: float,
) -> bool:
    return bool(
        math.isfinite(operator_diff)
        and operator_diff <= 1e-3
        and math.isfinite(spacing_mse_normalized)
        and spacing_mse_normalized <= 1.2
        and math.isfinite(ks_wigner)
        and ks_wigner <= 0.25
        and finite
        and math.isfinite(self_adjointness_fro)
        and self_adjointness_fro <= SA_TOL
    )


def rank_key_for_best(summary: Dict[str, Any]) -> Tuple[int, float, float]:
    """Lexicographic key: minimize (not_accepted, final_od, final_J)."""
    accepted = bool(summary.get("accepted"))
    fod = float(summary.get("final_operator_diff", float("inf")))
    if not math.isfinite(fod):
        fod = float("inf")
    fj = float(summary.get("pareto_objective_final", float("inf")))
    if not math.isfinite(fj):
        fj = float("inf")
    return (0 if accepted else 1, fod, fj)


def run_polish_single(
    *,
    H_base: np.ndarray,
    z_sorted: np.ndarray,
    base: Dict[str, Any],
    tau: float,
    beta_floor: float,
    beta0: float,
    clip_lo: float,
    clip_hi: float,
    smooth_mult: float,
    v: Any,
    v13l1: Any,
    m1: Any,
    max_iter: int,
    tol: float,
) -> Dict[str, Any]:
    dim = int(base["dim"])
    zeros_eff = int(base["zeros_eff"])
    ze = int(min(zeros_eff, int(z_sorted.size)))
    k_align = int(min(dim, ze))
    zeros_metric = z_sorted[: max(k_align, 1)].astype(_DTF, copy=False)
    smooth_sigma = float(base["smooth_sigma_base"]) * float(smooth_mult)

    out = m1.run_renormalized_cell(
        H_base=H_base,
        z_pool_positive=z_sorted,
        dim=dim,
        zeros_eff=zeros_eff,
        alpha=float(ALPHA),
        lambda_p_dim=float(base["lambda_p_dim"]),
        beta0=float(beta0),
        tau_beta=float(tau),
        beta_floor=float(beta_floor),
        smooth_sigma_dim=float(smooth_sigma),
        clip_percentiles=(float(clip_lo), float(clip_hi)),
        diag_shift=1e-6,
        abs_cap_factor=5.0,
        zeros_true_for_metrics=zeros_metric,
        k_align=k_align,
        spacing_fn=v.spacing_mse_normalized,
        ks_fn=v.ks_against_wigner_gue,
        norm_gaps_fn=v.normalized_gaps,
        stagnation_stop_only_if_operator_diff_below=1e-3,
        max_iter=int(max_iter),
        tol=float(tol),
    )

    H_final = np.asarray(out["H_final"], dtype=_DTF, copy=True)
    finite_H = bool(np.isfinite(H_final).all())
    sa = float(v13l1.self_adjointness_fro(H_final))

    rows = list(out.get("rows") or [])
    rN = rows[-1] if rows else {}

    def gf(dct: Dict[str, Any], k: str) -> float:
        x = dct.get(k)
        return float(x) if isinstance(x, (int, float)) and math.isfinite(float(x)) else float("nan")

    fod = gf(rN, "operator_diff")
    fsp = gf(rN, "spacing_mse_normalized")
    fks = gf(rN, "ks_wigner")
    fJ = gf(rN, "pareto_objective")

    accepted = meets_acceptance_v13m2(
        operator_diff=fod,
        spacing_mse_normalized=fsp,
        ks_wigner=fks,
        finite=finite_H,
        self_adjointness_fro=sa,
    )

    meta = out.get("meta") or {}

    enriched_rows: List[Dict[str, Any]] = []
    for rw in rows:
        enriched_rows.append(
            {
                "iter": int(rw.get("iter", 0)),
                "dim": dim,
                "zeros_eff": zeros_eff,
                "spectral_log_mse": rw.get("spectral_log_mse"),
                "spacing_mse_normalized": rw.get("spacing_mse_normalized"),
                "ks_wigner": rw.get("ks_wigner"),
                "operator_diff": rw.get("operator_diff"),
                "delta_norm": rw.get("delta_norm"),
                "pareto_objective": rw.get("pareto_objective"),
            }
        )

    summary = {
        "dim": dim,
        "zeros_eff": zeros_eff,
        "renorm_family": base["renorm_family"],
        "tau": float(tau),
        "beta_floor": float(beta_floor),
        "beta0": float(beta0),
        "clip_lo": float(clip_lo),
        "clip_hi": float(clip_hi),
        "smooth_sigma_multiplier": float(smooth_mult),
        "smooth_sigma_effective": float(smooth_sigma),
        "lambda_p_dim": float(base["lambda_p_dim"]),
        "geo_sigma_dim": float(base["geo_sigma_dim"]),
        "converged_operator": bool(meta.get("converged_operator", False)),
        "stopped_stagnation": bool(meta.get("stopped_stagnation", False)),
        "n_iter": int(meta.get("n_iter", 0)),
        "final_operator_diff": fod,
        "final_spacing_mse_normalized": fsp,
        "final_ks_wigner": fks,
        "final_spectral_log_mse": gf(rN, "spectral_log_mse"),
        "pareto_objective_final": fJ,
        "finite": finite_H,
        "self_adjointness_fro": sa,
        "accepted": accepted,
        "eig_error": meta.get("eig_error"),
    }

    return {
        "summary": summary,
        "rows": enriched_rows,
        "H_final": H_final,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="V13M.2 convergence polish on V13M.1 best configs.")
    ap.add_argument("--out_dir", type=str, default="runs/v13m2_convergence_polish")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_iter", type=int, default=500)
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = Path(ROOT) / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    v = _load_v13_validate()
    from core import v13l_self_consistent as v13l
    from core import v13l1_stabilized as v13l1
    from core import v13m1_renormalized as m1
    from core.artin_operator import sample_domain

    z_pool = v._load_zeros(512)
    z_sorted = np.asarray(z_pool, dtype=_DTF).reshape(-1)
    z_sorted = z_sorted[np.isfinite(z_sorted) & (z_sorted > 0.0)]
    z_sorted = np.sort(z_sorted)
    if z_sorted.size < 256:
        raise ValueError("need at least 256 positive zeta zeros")

    geodesics = [v13l.geodesic_entry_for_word(list(PRIMARY_WORD))]

    all_summaries: List[Dict[str, Any]] = []
    best_by_dim: Dict[int, Dict[str, Any]] = {}
    best_rows_by_dim: Dict[int, List[Dict[str, Any]]] = {}
    best_H_by_dim: Dict[int, np.ndarray] = {}

    n_grid = len(TAU_GRID) * len(BETA_FLOOR_GRID) * len(BETA0_GRID) * len(CLIP_GRID) * len(SMOOTH_MULT_GRID)
    total = len(POLISH_TARGETS) * n_grid
    done = 0

    for base in POLISH_TARGETS:
        dim = int(base["dim"])
        Z = sample_domain(dim, seed=int(args.seed))
        H_base, _ = v13l.build_h_base_no_potential(
            z_points=Z,
            geodesics=geodesics,
            eps=float(EPS),
            geo_sigma=float(base["geo_sigma_dim"]),
            geo_weight=float(GEO_WEIGHT_BASE),
            distances=None,
            diag_shift=1e-6,
        )

        best_key: Optional[Tuple[int, float, float]] = None
        best_summary: Optional[Dict[str, Any]] = None
        best_rows: Optional[List[Dict[str, Any]]] = None
        best_H: Optional[np.ndarray] = None

        for tau, bf, b0, (clo, chi), smult in itertools.product(
            TAU_GRID, BETA_FLOOR_GRID, BETA0_GRID, CLIP_GRID, SMOOTH_MULT_GRID
        ):
            done += 1
            print(
                f"[v13m2] ({done}/{total}) dim={dim} tau={tau} beta_floor={bf} beta0={b0} "
                f"clip=({clo},{chi}) smooth_mult={smult}",
                flush=True,
            )
            pack = run_polish_single(
                H_base=H_base,
                z_sorted=z_sorted,
                base=base,
                tau=tau,
                beta_floor=bf,
                beta0=b0,
                clip_lo=clo,
                clip_hi=chi,
                smooth_mult=smult,
                v=v,
                v13l1=v13l1,
                m1=m1,
                max_iter=int(args.max_iter),
                tol=float(args.tol),
            )
            summ = pack["summary"]
            all_summaries.append(summ)

            k = rank_key_for_best(summ)
            if best_key is None or k < best_key:
                best_key = k
                best_summary = dict(summ)
                best_rows = list(pack["rows"])
                best_H = np.asarray(pack["H_final"], dtype=_DTF, copy=True)

        if best_summary is not None and best_H is not None and best_rows is not None:
            best_by_dim[dim] = best_summary
            best_rows_by_dim[dim] = best_rows
            best_H_by_dim[dim] = best_H

    summary_fields = list(all_summaries[0].keys()) if all_summaries else []
    with open(out_dir / "v13m2_summary.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields, extrasaction="ignore")
        w.writeheader()
        for s in all_summaries:
            w.writerow({k: s.get(k) for k in summary_fields})

    bd_rows = [best_by_dim[d] for d in sorted(best_by_dim)]
    bd_fields = list(bd_rows[0].keys()) if bd_rows else summary_fields
    with open(out_dir / "v13m2_best_by_dim.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=bd_fields, extrasaction="ignore")
        w.writeheader()
        for row in bd_rows:
            w.writerow({k: row.get(k) for k in bd_fields})

    for d in (128, 256):
        if d in best_H_by_dim:
            np.save(str(out_dir / f"h_star_dim{d}_polished.npy"), np.asarray(best_H_by_dim[d], dtype=_DTF))

    payload = {
        "warning": "Computational evidence only; not a proof of the Riemann Hypothesis.",
        "primary_candidate": {"id": "seed_6", "word": list(PRIMARY_WORD)},
        "polish_baselines": POLISH_TARGETS,
        "search_grid": {
            "tau": TAU_GRID,
            "beta_floor": BETA_FLOOR_GRID,
            "beta0": BETA0_GRID,
            "clip_percentiles": [list(x) for x in CLIP_GRID],
            "smooth_sigma_multiplier": SMOOTH_MULT_GRID,
        },
        "acceptance_v13m2": {
            "operator_diff_max": 1e-3,
            "spacing_mse_normalized_max": 1.2,
            "ks_wigner_max": 0.25,
            "self_adjointness_fro_max": SA_TOL,
            "finite_required": True,
        },
        "stagnation_rule": "Stagnation early exit only when operator_diff < 1e-3; otherwise counter resets (V13M.2 polish).",
        "runs": all_summaries,
        "best_by_dim": {str(k): v for k, v in best_by_dim.items()},
        "best_convergence_by_dim": {str(k): v for k, v in best_rows_by_dim.items()},
    }
    with open(out_dir / "v13m2_results.json", "w", encoding="utf-8") as f:
        json.dump(numpy_json(payload), f, indent=2, allow_nan=True)

    n_acc = sum(1 for s in all_summaries if s.get("accepted"))
    md = [
        "# V13M.2: Convergence polish (renormalized scaling)\n\n",
        "## 1. Warning\n\n",
        "> **Computational evidence only; not a proof of the Riemann Hypothesis.**\n\n",
        "## 2. Acceptance (all must hold on **final** iterate)\n\n",
        "- `operator_diff <= 1e-3`\n",
        "- `spacing_mse_normalized <= 1.2`\n",
        "- `ks_wigner <= 0.25`\n",
        "- `finite` matrix entries\n",
        f"- `self_adjointness_fro <= {SA_TOL:g}` (numerical zero skew part)\n\n",
        f"**Accepted runs (full grid):** {n_acc} / {len(all_summaries)}\n\n",
        "## 3. Best run per dimension (rank: accepted, then lowest final `operator_diff`, then lowest final `J`)\n\n",
        "| dim | accepted | final od | final spacing | final KS | tau | beta_floor | beta0 | clip | smooth_mult | converged | stagnation_stop | n_iter |\n",
        "|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|---:|\n",
    ]
    for d in sorted(best_by_dim):
        s = best_by_dim[d]
        md.append(
            f"| {d} | {s.get('accepted')} | {s.get('final_operator_diff')} | {s.get('final_spacing_mse_normalized')} | "
            f"{s.get('final_ks_wigner')} | {s.get('tau')} | {s.get('beta_floor')} | {s.get('beta0')} | "
            f"({s.get('clip_lo')},{s.get('clip_hi')}) | {s.get('smooth_sigma_multiplier')} | "
            f"{s.get('converged_operator')} | {s.get('stopped_stagnation')} | {s.get('n_iter')} |\n"
        )
    md.append("\n## 4. Files\n\n")
    md.append("- `v13m2_results.json`, `v13m2_summary.csv`, `v13m2_best_by_dim.csv`\n")
    md.append("- `h_star_dim128_polished.npy`, `h_star_dim256_polished.npy`\n")
    (out_dir / "v13m2_report.md").write_text("".join(md), encoding="utf-8")

    tex = [
        "\\documentclass[11pt]{article}\n",
        "\\usepackage{amsmath,amssymb}\n",
        "\\usepackage[T1]{fontenc}\n",
        "\\usepackage[margin=1in]{geometry}\n",
        "\\title{V13M.2 Convergence polish}\n\\date{}\n\\begin{document}\n\\maketitle\n",
        "\\section*{Warning}\n",
        "Computational evidence only; not a proof of RH.\n\n",
        "\\section{Accepted count}\n",
        f"{latex_escape(str(n_acc))} / {latex_escape(str(len(all_summaries)))}\n\n",
        "\\end{document}\n",
    ]
    (out_dir / "v13m2_report.tex").write_text("".join(tex), encoding="utf-8")

    if try_pdflatex(out_dir / "v13m2_report.tex", out_dir, "v13m2_report.pdf"):
        print(f"Wrote {out_dir / 'v13m2_report.pdf'}")
    else:
        print("PDF skipped (pdflatex missing or failed).", flush=True)

    print(f"Wrote {out_dir / 'v13m2_results.json'}")
    print(f"Wrote {out_dir / 'v13m2_summary.csv'} ({len(all_summaries)} runs)")
    print(f"Accepted: {n_acc}/{len(all_summaries)}")


if __name__ == "__main__":
    main()
