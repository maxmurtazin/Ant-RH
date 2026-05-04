#!/usr/bin/env python3
"""
V13M.1: renormalized scaling law for the V13L.2-style self-consistent RH operator
(effective zeros_eff, scaled lambda_p / geo_sigma / smooth_sigma).

  python3 scripts/run_v13m1_renormalized_scaling.py \\
    --candidate_json runs/v13_candidate_theorem/candidate_operators.json \\
    --out_dir runs/v13m1_renormalized_scaling \\
    --seed 42 --max_iter 200 --tol 1e-3
"""

from __future__ import annotations

import argparse
import csv
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

_DTF = np.float64

PRIMARY_WORD = [-4, -2, -4, -2, -2, -1, -1]
PRIMARY_ID = "seed_6"

ALPHA = 0.5
LAMBDA_P_BASE = 3.0
BETA0 = 0.3
TAU = 200.0
SMOOTH_SIGMA_BASE = 1.0
CLIP_LO, CLIP_HI = 1.0, 99.0
GEO_WEIGHT_BASE = 10.0
GEO_SIGMA_BASE = 0.6
EPS = 0.6

DIMS = [64, 128, 256]
ZEROS_FAMILIES = ["linear", "fixed128", "sqrt", "power075", "manual"]
Q_GRID = [-0.5, 0.0, 0.5]
R_GRID = [-0.25, 0.0, 0.25]
S_GRID = [0.0, 0.25, 0.5]


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


def classify_v13m1(
    *,
    by_dim: Dict[int, Dict[str, Any]],
    last5_min_od: Dict[int, float],
) -> Tuple[str, Dict[str, bool]]:
    """stable_renormalized if A–D pass; partially if A and some of B–D; else unstable."""
    flags: Dict[str, bool] = {}
    for d in DIMS:
        flags[f"A_finite_sa_dim{d}"] = False
    flags["B_od_128_256"] = False
    flags["C_spacing_no_explode"] = False
    flags["D_ks_128_256"] = False

    A = True
    for d in DIMS:
        s = by_dim.get(d)
        if not s:
            A = False
            flags[f"A_finite_sa_dim{d}"] = False
            continue
        fin = bool(s.get("finite", False))
        sa = float(s.get("self_adjointness_fro", float("nan")))
        ok = fin and math.isfinite(sa) and sa <= 1e-5
        flags[f"A_finite_sa_dim{d}"] = ok
        A = A and ok

    od128 = float(by_dim.get(128, {}).get("operator_diff_best", float("nan")))
    od256 = float(by_dim.get(256, {}).get("operator_diff_best", float("nan")))
    p128 = last5_min_od.get(128, float("nan"))
    p256 = last5_min_od.get(256, float("nan"))
    B = (
        (math.isfinite(od128) and od128 <= 1e-3)
        or (math.isfinite(p128) and p128 <= 1e-2)
    ) and (
        (math.isfinite(od256) and od256 <= 1e-3)
        or (math.isfinite(p256) and p256 <= 1e-2)
    )
    flags["B_od_128_256"] = B

    sp128 = float(by_dim.get(128, {}).get("spacing_mse_normalized_best", float("nan")))
    sp256 = float(by_dim.get(256, {}).get("spacing_mse_normalized_best", float("nan")))
    C = math.isfinite(sp128) and math.isfinite(sp256) and sp128 > 0 and (sp256 / sp128) < 5.0
    flags["C_spacing_no_explode"] = C

    ks128 = float(by_dim.get(128, {}).get("ks_wigner_best", float("nan")))
    ks256 = float(by_dim.get(256, {}).get("ks_wigner_best", float("nan")))
    D = math.isfinite(ks128) and math.isfinite(ks256) and ks128 <= 0.5 and ks256 <= 0.5
    flags["D_ks_128_256"] = D

    sl128 = float(by_dim.get(128, {}).get("spectral_log_mse_best", float("nan")))
    sl256 = float(by_dim.get(256, {}).get("spectral_log_mse_best", float("nan")))
    E = (not math.isfinite(sl128)) or (not math.isfinite(sl256)) or sl128 <= 0 or (sl256 / sl128) < 3.0
    flags["E_spectral_same_order"] = bool(E)

    if A and B and C and D:
        return "stable_renormalized", flags
    if A and (B or C or D):
        return "partially_stable", flags
    return "unstable", flags


def run_one_config(
    *,
    dim: int,
    renorm_family: str,
    q: float,
    r: float,
    s: float,
    z_pool: np.ndarray,
    seed: int,
    max_iter: int,
    tol: float,
    v: Any,
    v13l: Any,
    v13l1: Any,
    m1: Any,
    geodesics: List[Dict[str, Any]],
) -> Dict[str, Any]:
    from core.artin_operator import sample_domain

    zeros_eff_nominal = int(m1.zeros_eff_for_family(renorm_family, int(dim)))
    z_sorted = np.asarray(z_pool, dtype=_DTF).reshape(-1)
    z_sorted = z_sorted[np.isfinite(z_sorted) & (z_sorted > 0.0)]
    z_sorted = np.sort(z_sorted)
    ze = int(min(zeros_eff_nominal, int(z_sorted.size)))
    k_align = int(min(int(dim), ze))

    lambda_p_dim = float(m1.lambda_p_scaled(LAMBDA_P_BASE, int(dim), q))
    geo_sigma_dim = float(m1.geo_sigma_scaled(GEO_SIGMA_BASE, int(dim), r))
    smooth_sigma_dim = float(m1.smooth_sigma_scaled(SMOOTH_SIGMA_BASE, int(dim), s))

    Z = sample_domain(int(dim), seed=int(seed))
    H_base, _ = v13l.build_h_base_no_potential(
        z_points=Z,
        geodesics=geodesics,
        eps=float(EPS),
        geo_sigma=float(geo_sigma_dim),
        geo_weight=float(GEO_WEIGHT_BASE),
        distances=None,
        diag_shift=1e-6,
    )

    zeros_metric = z_sorted[: max(k_align, 1)].astype(_DTF, copy=False)

    out = m1.run_renormalized_cell(
        H_base=H_base,
        z_pool_positive=z_sorted,
        dim=int(dim),
        zeros_eff=int(zeros_eff_nominal),
        alpha=float(ALPHA),
        lambda_p_dim=float(lambda_p_dim),
        beta0=float(BETA0),
        tau_beta=float(TAU),
        beta_floor=0.03,
        smooth_sigma_dim=float(smooth_sigma_dim),
        clip_percentiles=(float(CLIP_LO), float(CLIP_HI)),
        diag_shift=1e-6,
        abs_cap_factor=5.0,
        zeros_true_for_metrics=zeros_metric,
        k_align=int(k_align),
        spacing_fn=v.spacing_mse_normalized,
        ks_fn=v.ks_against_wigner_gue,
        norm_gaps_fn=v.normalized_gaps,
        max_iter=int(max_iter),
        tol=float(tol),
    )

    H_best = np.asarray(out["H_best_J"], dtype=_DTF, copy=True)
    finite_H = bool(np.isfinite(H_best).all())
    sa = float(v13l1.self_adjointness_fro(H_best))

    eig_best = np.sort(np.linalg.eigvalsh(0.5 * (H_best + H_best.T)).astype(_DTF))
    eig_best = eig_best[np.isfinite(eig_best)]

    rows = list(out.get("rows") or [])
    r0 = rows[0] if rows else {}
    rN = rows[-1] if rows else {}

    def gf(dct: Dict[str, Any], k: str) -> float:
        x = dct.get(k)
        return float(x) if isinstance(x, (int, float)) and math.isfinite(float(x)) else float("nan")

    meta = out.get("meta") or {}
    last5 = rows[-5:] if len(rows) >= 5 else rows
    last5_min_od = min((float(x["operator_diff"]) for x in last5), default=float("nan"))

    bcp = float(
        m1.checkpoint_pick_score(
            float(out.get("best_spectral_log_mse", float("nan"))),
            float(out.get("best_spacing_mse_normalized", float("nan"))),
            float(out.get("best_ks_wigner", float("nan"))),
            float(out.get("best_operator_diff", float("nan"))),
        )
    )

    summary = {
        "dim": int(dim),
        "zeros_eff_nominal": int(zeros_eff_nominal),
        "zeros_eff_used": int(ze),
        "k_align": int(k_align),
        "renorm_family": str(renorm_family),
        "q": float(q),
        "r": float(r),
        "s": float(s),
        "lambda_p_dim": float(lambda_p_dim),
        "geo_sigma_dim": float(geo_sigma_dim),
        "smooth_sigma_dim": float(smooth_sigma_dim),
        "converged_operator": bool(meta.get("converged_operator", False)),
        "stopped_stagnation": bool(meta.get("stopped_stagnation", False)),
        "n_iter": int(meta.get("n_iter", 0)),
        "best_iter": int(out.get("best_iter_by_J", 0)),
        "best_J": float(out.get("best_J", float("nan"))),
        "best_checkpoint_pick": bcp,
        "spectral_log_mse_initial": gf(r0, "spectral_log_mse"),
        "spectral_log_mse_final": gf(rN, "spectral_log_mse"),
        "spectral_log_mse_best": float(out.get("best_spectral_log_mse", float("nan"))),
        "spacing_mse_normalized_initial": gf(r0, "spacing_mse_normalized"),
        "spacing_mse_normalized_final": gf(rN, "spacing_mse_normalized"),
        "spacing_mse_normalized_best": float(out.get("best_spacing_mse_normalized", float("nan"))),
        "ks_wigner_initial": gf(r0, "ks_wigner"),
        "ks_wigner_final": gf(rN, "ks_wigner"),
        "ks_wigner_best": float(out.get("best_ks_wigner", float("nan"))),
        "operator_diff_best": float(out.get("best_operator_diff", float("nan"))),
        "delta_norm_best": float(out.get("best_delta_norm", float("nan"))),
        "final_operator_diff": float(out.get("final_operator_diff", float("nan"))),
        "last5_min_operator_diff": float(last5_min_od),
        "eig_min": float(np.min(eig_best)) if eig_best.size else float("nan"),
        "eig_max": float(np.max(eig_best)) if eig_best.size else float("nan"),
        "self_adjointness_fro": sa,
        "finite": bool(finite_H and math.isfinite(sa)),
        "eig_error": meta.get("eig_error"),
        "config_key": f"d{dim}_z{renorm_family}_q{q}_r{r}_s{s}",
    }

    enriched_rows: List[Dict[str, Any]] = []
    for rw in rows:
        t = int(rw.get("iter", 0))
        enriched_rows.append(
            {
                "iter": t,
                "dim": int(dim),
                "zeros_eff": int(zeros_eff_nominal),
                "renorm_family": str(renorm_family),
                "q": float(q),
                "r": float(r),
                "s": float(s),
                "lambda_p_dim": float(lambda_p_dim),
                "geo_sigma_dim": float(geo_sigma_dim),
                "smooth_sigma_dim": float(smooth_sigma_dim),
                "beta_t": float(v13l1.beta_schedule(t, beta0=BETA0, tau=TAU, beta_floor=0.03)),
                "spectral_log_mse": rw.get("spectral_log_mse"),
                "spacing_mse_normalized": rw.get("spacing_mse_normalized"),
                "ks_wigner": rw.get("ks_wigner"),
                "operator_diff": rw.get("operator_diff"),
                "delta_norm": rw.get("delta_norm"),
                "eig_min": rw.get("eig_min"),
                "eig_max": rw.get("eig_max"),
                "pareto_objective": rw.get("pareto_objective"),
            }
        )

    return {
        "summary": summary,
        "rows": enriched_rows,
        "H_best": H_best,
        "eig_best": eig_best,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="V13M.1 renormalized scaling for V13L.2-style operator.")
    ap.add_argument("--candidate_json", type=str, default="runs/v13_candidate_theorem/candidate_operators.json")
    ap.add_argument("--out_dir", type=str, default="runs/v13m1_renormalized_scaling")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_iter", type=int, default=200)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--smoke", action="store_true", help="Single tiny config (dim=64, linear, q=r=s=0).")
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

    v = _load_v13_validate()
    from core import v13l_self_consistent as v13l
    from core import v13l1_stabilized as v13l1
    from core import v13m1_renormalized as m1

    z_pool = v._load_zeros(512)
    if z_pool.size < 256:
        raise ValueError("need at least 256 zeta zeros")

    geodesics = [v13l.geodesic_entry_for_word(list(PRIMARY_WORD))]

    with open(cand_path, "r", encoding="utf-8") as f:
        cj = json.load(f)
    sc = cj.get("spectral_candidate") if isinstance(cj.get("spectral_candidate"), dict) else {}
    if str(sc.get("id", "")) == PRIMARY_ID and isinstance(sc.get("word"), list):
        wj = [int(x) for x in sc["word"]]
        if wj != list(PRIMARY_WORD):
            print(f"[v13m1] note: JSON word differs from primary; using {PRIMARY_WORD}.", flush=True)

    dims = list(DIMS)
    fams = list(ZEROS_FAMILIES)
    qs = list(Q_GRID)
    rs = list(R_GRID)
    ss = list(S_GRID)
    if args.smoke:
        dims = [64]
        fams = ["linear"]
        qs = [0.0]
        rs = [0.0]
        ss = [0.0]

    all_summaries: List[Dict[str, Any]] = []
    best_pack_by_dim: Dict[int, Dict[str, Any]] = {}
    best_overall: Optional[Dict[str, Any]] = None
    best_pick_global = float("inf")

    total = len(dims) * len(fams) * len(qs) * len(rs) * len(ss)
    done = 0
    for dim in dims:
        for fam in fams:
            for q in qs:
                for r in rs:
                    for s in ss:
                        done += 1
                        print(f"[v13m1] ({done}/{total}) dim={dim} fam={fam} q={q} r={r} s={s}", flush=True)
                        pack = run_one_config(
                            dim=int(dim),
                            renorm_family=str(fam),
                            q=float(q),
                            r=float(r),
                            s=float(s),
                            z_pool=z_pool,
                            seed=int(args.seed),
                            max_iter=int(args.max_iter),
                            tol=float(args.tol),
                            v=v,
                            v13l=v13l,
                            v13l1=v13l1,
                            m1=m1,
                            geodesics=geodesics,
                        )
                        summ = pack["summary"]
                        all_summaries.append(summ)

                        pick = float(summ.get("best_checkpoint_pick", float("nan")))
                        if math.isfinite(pick):
                            bd = int(dim)
                            cur = best_pack_by_dim.get(bd)
                            if cur is None or float(cur["summary"]["best_checkpoint_pick"]) > pick:
                                best_pack_by_dim[bd] = {"summary": dict(summ), "rows": pack["rows"], "H_best": pack["H_best"]}
                            if pick < best_pick_global:
                                best_pick_global = pick
                                best_overall = {"summary": dict(summ), "rows": pack["rows"], "H_best": pack["H_best"]}

    summary_fields = list(all_summaries[0].keys()) if all_summaries else []
    if not summary_fields:
        summary_fields = [
            "dim",
            "zeros_eff_nominal",
            "renorm_family",
            "q",
            "r",
            "s",
            "best_J",
            "spectral_log_mse_best",
            "spacing_mse_normalized_best",
            "ks_wigner_best",
            "operator_diff_best",
            "finite",
            "self_adjointness_fro",
        ]

    with open(out_dir / "v13m1_summary.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields, extrasaction="ignore")
        w.writeheader()
        for s in all_summaries:
            w.writerow({k: s.get(k) for k in summary_fields})

    by_dim_summary: Dict[int, Dict[str, Any]] = {}
    last5_min_by_dim: Dict[int, float] = {}
    for d in DIMS:
        p = best_pack_by_dim.get(d)
        if p:
            by_dim_summary[d] = dict(p["summary"])
            rows = p.get("rows") or []
            tail = rows[-5:] if len(rows) >= 5 else rows
            last5_min_by_dim[d] = min((float(x["operator_diff"]) for x in tail), default=float("nan"))
        else:
            last5_min_by_dim[d] = float("nan")

    best_dim_rows = [by_dim_summary[d] for d in DIMS if d in by_dim_summary]
    bd_fields = list(best_dim_rows[0].keys()) if best_dim_rows else summary_fields
    with open(out_dir / "v13m1_best_by_dim.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=bd_fields, extrasaction="ignore")
        w.writeheader()
        for row in best_dim_rows:
            w.writerow({k: row.get(k) for k in bd_fields})

    ov_fields = list(best_overall["summary"].keys()) if best_overall else summary_fields
    with open(out_dir / "v13m1_best_overall.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ov_fields, extrasaction="ignore")
        w.writeheader()
        if best_overall:
            w.writerow({k: best_overall["summary"].get(k) for k in ov_fields})

    conv_fields = [
        "iter",
        "dim",
        "zeros_eff",
        "renorm_family",
        "q",
        "r",
        "s",
        "lambda_p_dim",
        "geo_sigma_dim",
        "smooth_sigma_dim",
        "beta_t",
        "spectral_log_mse",
        "spacing_mse_normalized",
        "ks_wigner",
        "operator_diff",
        "delta_norm",
        "eig_min",
        "eig_max",
        "pareto_objective",
    ]
    for d in DIMS:
        p = best_pack_by_dim.get(d)
        if not p:
            continue
        fn = out_dir / f"v13m1_convergence_dim{d}_best.csv"
        with open(fn, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=conv_fields, extrasaction="ignore")
            w.writeheader()
            for row in p.get("rows") or []:
                w.writerow({k: row.get(k) for k in conv_fields})
        np.save(str(out_dir / f"h_star_dim{d}_best.npy"), np.asarray(p["H_best"], dtype=_DTF))

    label, crit_flags = classify_v13m1(by_dim=by_dim_summary, last5_min_od=last5_min_by_dim)

    best_law = None
    if best_overall:
        s0 = best_overall["summary"]
        best_law = {
            "renorm_family": s0.get("renorm_family"),
            "zeros_eff_nominal": s0.get("zeros_eff_nominal"),
            "q": s0.get("q"),
            "r": s0.get("r"),
            "s": s0.get("s"),
            "best_checkpoint_pick": s0.get("best_checkpoint_pick"),
            "best_J_at_checkpoint": s0.get("best_J"),
        }

    if label == "stable_renormalized":
        rec = "V13N spectral triple theorem report"
    else:
        rec = "V13M.2 adaptive renormalization / learned scaling"

    payload = {
        "warning": "Computational evidence only; not a proof of the Riemann Hypothesis.",
        "classification": label,
        "criteria_flags": crit_flags,
        "best_scaling_law_overall": best_law,
        "best_by_dim": {str(k): v for k, v in by_dim_summary.items()},
        "hyperparameters": {
            "alpha": ALPHA,
            "lambda_p_base": LAMBDA_P_BASE,
            "beta0": BETA0,
            "tau": TAU,
            "smooth_sigma_base": SMOOTH_SIGMA_BASE,
            "clip_lo": CLIP_LO,
            "clip_hi": CLIP_HI,
            "geo_weight_base": GEO_WEIGHT_BASE,
            "geo_sigma_base": GEO_SIGMA_BASE,
            "eps": EPS,
            "beta_t": "max(0.03, beta0 * exp(-t / tau))",
            "lambda_p_dim": "lambda_p_base * (dim/128)^q",
            "geo_sigma_dim": "geo_sigma_base * (dim/128)^r",
            "smooth_sigma_dim": "smooth_sigma_base * (dim/128)^s",
        },
        "grid": {
            "dims": dims,
            "zeros_families": fams,
            "q": qs,
            "r": rs,
            "s": ss,
        },
        "runs": all_summaries,
        "recommendation": rec,
    }
    with open(out_dir / "v13m1_results.json", "w", encoding="utf-8") as f:
        json.dump(numpy_json(payload), f, indent=2, allow_nan=True)

    def md_table_row(s: Dict[str, Any]) -> str:
        return (
            f"| {s.get('dim')} | {s.get('spectral_log_mse_best')} | {s.get('spacing_mse_normalized_best')} | "
            f"{s.get('ks_wigner_best')} | {s.get('operator_diff_best')} | {s.get('converged_operator')} | "
            f"{s.get('zeros_eff_nominal')} | {s.get('renorm_family')} | {s.get('q')} | {s.get('r')} | {s.get('s')} | "
            f"{s.get('best_iter')} | {s.get('self_adjointness_fro')} |\n"
        )

    md = [
        "# V13M.1: Renormalized scaling (zeros_eff and hyperparameter scaling)\n\n",
        "## 1. Warning\n\n",
        "> **Computational evidence only; not a proof of the Riemann Hypothesis.**\n\n",
        "## 2. Classification\n\n",
        f"**{label}**\n\n",
        f"- Criteria flags: `{json.dumps(crit_flags)}`\n\n",
        "## 3. Best scaling law (minimum uncapped checkpoint score overall)\n\n",
        "Per-iteration rows still record clipped Pareto ``J``; overall winner uses ``spectral_log_mse + 0.05*spacing + ks + 0.1*operator_diff`` without clipping ``operator_diff`` so huge updates cannot hide in ``min(operator_diff,1)``.\n\n",
        f"```json\n{json.dumps(best_law, indent=2)}\n```\n\n",
        "## 4. Best configuration by dimension (minimum J per dim)\n\n",
        "| dim | best spectral_log_mse | best spacing (norm) | best KS | op_diff best | converged | zeros_eff | family | q | r | s | best_iter | ‖skew‖_F |\n",
        "|---:|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---:|\n",
    ]
    for d in DIMS:
        if d in by_dim_summary:
            md.append(md_table_row(by_dim_summary[d]))
    md.append("\n## 4b. Spectral scaling (criterion E, informational)\n\n")
    md.append(
        f"- **E (spectral_log_mse same order or improve vs dim128):** `{crit_flags.get('E_spectral_same_order')}`\n\n"
    )
    md.append("\n## 5. Zeros_eff interpretation\n\n")
    md.append(
        "Renormalization shrinks the aligned spectral window relative to `dim` when `zeros_eff < dim`, "
        "so the potential is informed by a shorter prefix of rescaled zeta targets; mismatch is interpolated "
        "across all indices before smoothing and clipping. Sublinear `zeros_eff` (e.g. `sqrt`, `power075`, `manual`) "
        "acts like a fixed effective density of constraints as dimension grows.\n\n"
    )
    md.append("## 6. Recommendation\n\n")
    md.append(f"- **{rec}**\n\n")
    md.append("## 7. Files\n\n")
    md.append("- `v13m1_results.json`, `v13m1_summary.csv`, `v13m1_best_by_dim.csv`, `v13m1_best_overall.csv`\n")
    md.append("- `v13m1_convergence_dim{64,128,256}_best.csv`, `h_star_dim{64,128,256}_best.npy`\n")
    (out_dir / "v13m1_report.md").write_text("".join(md), encoding="utf-8")

    tex = [
        "\\documentclass[11pt]{article}\n",
        "\\usepackage{amsmath,amssymb}\n",
        "\\usepackage[T1]{fontenc}\n",
        "\\usepackage[margin=1in]{geometry}\n",
        "\\title{V13M.1 Renormalized scaling}\n\\date{}\n\\begin{document}\n\\maketitle\n",
        "\\section*{Warning}\n",
        "Computational evidence only; not a proof of RH.\n\n",
        "\\section{Classification}\n",
        latex_escape(label) + "\n\n",
        "\\section{Best law (overall)}\n",
        latex_escape(json.dumps(best_law or {})) + "\n\n",
        "\\section{Recommendation}\n",
        latex_escape(rec) + "\n\n",
        "\\end{document}\n",
    ]
    (out_dir / "v13m1_report.tex").write_text("".join(tex), encoding="utf-8")

    if try_pdflatex(out_dir / "v13m1_report.tex", out_dir, "v13m1_report.pdf"):
        print(f"Wrote {out_dir / 'v13m1_report.pdf'}")
    else:
        print("PDF skipped (pdflatex missing or failed).", flush=True)

    print(f"Wrote {out_dir / 'v13m1_results.json'}")
    print(f"Wrote {out_dir / 'v13m1_summary.csv'} ({len(all_summaries)} configs)")
    print(f"Classification: {label}")


if __name__ == "__main__":
    main()
