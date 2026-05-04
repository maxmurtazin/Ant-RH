#!/usr/bin/env python3
"""
V13M: continuum / scaling validation for the V13L.2 Pareto-fixed self-consistent operator.

  python3 scripts/run_v13m_scaling_validation.py \\
    --candidate_json runs/v13_candidate_theorem/candidate_operators.json \\
    --out_dir runs/v13m_scaling_validation \\
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

PRIMARY_WORD = [-4, -2, -4, -2, -2, -1, -1]
PRIMARY_ID = "seed_6"

# V13L.2 best (fixed)
ALPHA = 0.5
LAMBDA_P_EFF = 3.0
BETA0 = 0.3
TAU = 200.0
SMOOTH_SIGMA = 1.0
CLIP_LO, CLIP_HI = 1.0, 99.0
GEO_WEIGHT = 10.0
GEO_SIGMA = 0.6
EPS = 0.6

MAIN_GRID: List[Tuple[int, int]] = [(64, 64), (128, 128), (256, 256)]
CROSS_GRID: List[Tuple[int, int]] = [(64, 128), (128, 64), (128, 256), (256, 128)]


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


def condition_proxy(eig: np.ndarray) -> float:
    e = np.sort(np.asarray(eig, dtype=_DTF).reshape(-1))
    e = e[np.isfinite(e)]
    if e.size < 2:
        return float("nan")
    sp = np.diff(e)
    sp = sp[np.isfinite(sp)]
    if sp.size == 0:
        return float("nan")
    med = float(np.median(sp))
    return float(abs(float(e[-1]) - float(e[0])) / max(1e-12, med))


def quantile_l1_distance(eig_a: np.ndarray, eig_b: np.ndarray, nq: int = 64) -> float:
    """Empirical-quantile L1 on sorted eigenvalues (Wasserstein-like when distributions are close)."""
    a = np.sort(np.asarray(eig_a, dtype=_DTF).reshape(-1))
    b = np.sort(np.asarray(eig_b, dtype=_DTF).reshape(-1))
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    qs = np.linspace(0.0, 1.0, int(nq), dtype=_DTF)
    qa = np.quantile(a, qs)
    qb = np.quantile(b, qs)
    return float(np.mean(np.abs(qa - qb)))


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


def run_one_scale(
    *,
    dim: int,
    zeros_n: int,
    z_pool: np.ndarray,
    seed: int,
    max_iter: int,
    tol: float,
    v: Any,
    v13l: Any,
    v13l1mod: Any,
    v13l2: Any,
    geodesics: List[Dict[str, Any]],
) -> Dict[str, Any]:
    from core.artin_operator import sample_domain

    Z = sample_domain(int(dim), seed=int(seed))
    H_base, _ = v13l.build_h_base_no_potential(
        z_points=Z,
        geodesics=geodesics,
        eps=float(EPS),
        geo_sigma=float(GEO_SIGMA),
        geo_weight=float(GEO_WEIGHT),
        distances=None,
        diag_shift=1e-6,
    )

    z = np.asarray(z_pool, dtype=_DTF).reshape(-1)
    z = z[np.isfinite(z) & (z > 0.0)]
    z = np.sort(z)
    need = max(int(dim), int(zeros_n))
    if z.size < need:
        raise ValueError(f"need at least {need} positive zeros, got {z.size}")
    gamma = z.astype(_DTF, copy=False)
    zeros_eval = z[: int(zeros_n)].astype(_DTF, copy=False)

    out = v13l2.run_pareto_cell(
        H_base=H_base,
        gamma=gamma,
        alpha=float(ALPHA),
        lambda_p_eff=float(LAMBDA_P_EFF),
        beta0=float(BETA0),
        tau_beta=float(TAU),
        beta_floor=0.03,
        smooth_sigma=float(SMOOTH_SIGMA),
        clip_percentiles=(float(CLIP_LO), float(CLIP_HI)),
        max_iter=int(max_iter),
        tol=float(tol),
        diag_shift=1e-6,
        abs_cap_factor=5.0,
        zeros_eval=zeros_eval,
        spacing_fn=v.spacing_mse_normalized,
        ks_fn=v.ks_against_wigner_gue,
        norm_gaps_fn=v.normalized_gaps,
    )

    rows = list(out.get("rows") or [])
    for r in rows:
        r["dim"] = int(dim)
        r["zeros"] = int(zeros_n)

    meta = out.get("meta") or {}
    H_best = np.asarray(out["H_best_J"], dtype=_DTF, copy=True)
    finite_H = bool(np.isfinite(H_best).all())
    sa = float(v13l1mod.self_adjointness_fro(H_best))

    eig_best = np.sort(np.linalg.eigvalsh(0.5 * (H_best + H_best.T)).astype(_DTF))
    eig_best = eig_best[np.isfinite(eig_best)]
    sp = np.diff(eig_best) if eig_best.size >= 2 else np.array([], dtype=_DTF)
    sp_m = float(np.mean(sp)) if sp.size else float("nan")
    sp_s = float(np.std(sp)) if sp.size else float("nan")
    spec_range = float(eig_best[-1] - eig_best[0]) if eig_best.size else float("nan")
    cp = condition_proxy(eig_best)

    r0 = rows[0] if rows else {}
    rN = rows[-1] if rows else {}

    def gf(d: Dict[str, Any], k: str) -> float:
        x = d.get(k)
        return float(x) if isinstance(x, (int, float)) and math.isfinite(float(x)) else float("nan")

    summary = {
        "dim": int(dim),
        "zeros": int(zeros_n),
        "converged_operator": bool(meta.get("converged_operator", False)),
        "stopped_stagnation": bool(meta.get("stopped_stagnation", False)),
        "n_iter": int(meta.get("n_iter", 0)),
        "best_iter": int(out.get("best_iter_by_J", 0)),
        "best_J": float(out.get("best_J", float("nan"))),
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
        "final_operator_diff": float(out.get("final_operator_diff", float("nan"))),
        "eig_min": float(np.min(eig_best)) if eig_best.size else float("nan"),
        "eig_max": float(np.max(eig_best)) if eig_best.size else float("nan"),
        "self_adjointness_fro": sa,
        "finite": bool(finite_H and math.isfinite(sa)),
        "condition_proxy": float(cp),
        "spectral_range": float(spec_range),
        "eig_spacing_mean": sp_m,
        "eig_spacing_std": sp_s,
        "eig_error": meta.get("eig_error"),
    }

    return {
        "summary": summary,
        "rows": rows,
        "H_best": H_best,
        "eig_best": eig_best,
    }


def classify_stability(square_summaries: Dict[int, Dict[str, Any]]) -> str:
    """square_summaries keys dim 64,128,256 for (d,d) runs."""
    if 64 not in square_summaries or 128 not in square_summaries:
        return "incomplete_grid"
    c64 = bool(square_summaries.get(64, {}).get("converged_operator", False))
    c128 = bool(square_summaries.get(128, {}).get("converged_operator", False))
    c256 = bool(square_summaries.get(256, {}).get("converged_operator", False))
    if not (c64 and c128):
        return "unstable"
    if not c256:
        return "partially_stable"
    s64 = square_summaries[64].get("spectral_log_mse_best", float("nan"))
    s128 = square_summaries[128].get("spectral_log_mse_best", float("nan"))
    s256 = square_summaries[256].get("spectral_log_mse_best", float("nan"))
    if not all(map(math.isfinite, (s64, s128, s256))):
        return "partially_stable"
    r = max(s64, s128, 1e-12)
    if s256 / r > 10.0:
        return "partially_stable"
    ks256 = square_summaries[256].get("ks_wigner_best", float("nan"))
    ks128 = square_summaries[128].get("ks_wigner_best", float("nan"))
    if math.isfinite(ks256) and math.isfinite(ks128) and ks256 > ks128 * 3.0 + 0.2:
        return "partially_stable"
    return "stable"


def main() -> None:
    ap = argparse.ArgumentParser(description="V13M scaling validation for V13L.2-fixed operator.")
    ap.add_argument("--candidate_json", type=str, default="runs/v13_candidate_theorem/candidate_operators.json")
    ap.add_argument("--out_dir", type=str, default="runs/v13m_scaling_validation")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_iter", type=int, default=200)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--no_cross", action="store_true", help="Skip optional (dim,zeros) cross pairs.")
    ap.add_argument("--smoke", action="store_true", help="Tiny grid (24,24) only for CI/dev.")
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
    from core import v13l2_pareto as v13l2

    z_pool = v._load_zeros(512)
    if z_pool.size < 256:
        raise ValueError("need at least 256 zeta zeros for scaling runs")

    geodesics = [v13l.geodesic_entry_for_word(list(PRIMARY_WORD))]

    with open(cand_path, "r", encoding="utf-8") as f:
        cj = json.load(f)
    sc = cj.get("spectral_candidate") if isinstance(cj.get("spectral_candidate"), dict) else {}
    if str(sc.get("id", "")) == PRIMARY_ID and isinstance(sc.get("word"), list):
        wj = [int(x) for x in sc["word"]]
        if wj != list(PRIMARY_WORD):
            print(f"[v13m] note: JSON seed_6 word differs from primary; using primary word {PRIMARY_WORD}.", flush=True)

    main_pairs: List[Tuple[int, int]] = list(MAIN_GRID)
    cross_pairs: List[Tuple[int, int]] = list(CROSS_GRID)
    if args.smoke:
        main_pairs = [(24, 24)]
        cross_pairs = []
    pairs: List[Tuple[int, int]] = list(main_pairs)
    if not args.no_cross:
        pairs.extend(cross_pairs)

    all_results: List[Dict[str, Any]] = []
    square_eigs: Dict[int, np.ndarray] = {}
    square_summaries: Dict[int, Dict[str, Any]] = {}

    for dim, zn in pairs:
        key = f"{dim}_{zn}"
        print(f"[v13m] run dim={dim} zeros={zn}", flush=True)
        pack = run_one_scale(
            dim=dim,
            zeros_n=zn,
            z_pool=z_pool,
            seed=int(args.seed),
            max_iter=int(args.max_iter),
            tol=float(args.tol),
            v=v,
            v13l=v13l,
            v13l1mod=v13l1,
            v13l2=v13l2,
            geodesics=geodesics,
        )
        summ = pack["summary"]
        all_results.append({"run_key": key, **summ})

        if dim == zn and dim in (64, 128, 256):
            square_summaries[dim] = summ
            square_eigs[dim] = np.asarray(pack["eig_best"], dtype=_DTF, copy=True)
            np.save(str(out_dir / f"h_star_dim{dim}.npy"), np.asarray(pack["H_best"], dtype=_DTF))
            conv_fields = [
                "iter",
                "dim",
                "zeros",
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
            rows_out: List[Dict[str, Any]] = []
            for r in pack["rows"]:
                t = int(r.get("iter", 0))
                rows_out.append(
                    {
                        "iter": t,
                        "dim": int(dim),
                        "zeros": int(zn),
                        "beta_t": float(v13l1.beta_schedule(t, beta0=BETA0, tau=TAU, beta_floor=0.03)),
                        "spectral_log_mse": r.get("spectral_log_mse"),
                        "spacing_mse_normalized": r.get("spacing_mse_normalized"),
                        "ks_wigner": r.get("ks_wigner"),
                        "operator_diff": r.get("operator_diff"),
                        "delta_norm": r.get("delta_norm"),
                        "eig_min": r.get("eig_min"),
                        "eig_max": r.get("eig_max"),
                        "pareto_objective": r.get("pareto_objective"),
                    }
                )
            with open(out_dir / f"v13m_convergence_dim{dim}.csv", "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=conv_fields, extrasaction="ignore")
                w.writeheader()
                for row in rows_out:
                    w.writerow({k: row.get(k) for k in conv_fields})

    # Scaling analysis (main diagonal)
    ref_dim = 128
    ref_sl = square_summaries.get(ref_dim, {}).get("spectral_log_mse_best", float("nan"))
    trends: Dict[str, Any] = {}
    for d in (64, 128, 256):
        if d not in square_summaries:
            continue
        s = square_summaries[d]
        sl = s.get("spectral_log_mse_best", float("nan"))
        trends[f"dim_{d}"] = {
            "normalized_spectral_log_mse_vs_dim128": float(sl / ref_sl) if math.isfinite(sl) and math.isfinite(ref_sl) and ref_sl > 0 else float("nan"),
            "best_spacing_mse_normalized": s.get("spacing_mse_normalized_best"),
            "best_ks_wigner": s.get("ks_wigner_best"),
            "operator_diff_best": s.get("operator_diff_best"),
        }

    wd: Dict[str, float] = {}
    if 64 in square_eigs and 128 in square_eigs:
        wd["quantile_l1_dim64_vs_dim128"] = quantile_l1_distance(square_eigs[64], square_eigs[128])
    if 128 in square_eigs and 256 in square_eigs:
        wd["quantile_l1_dim128_vs_dim256"] = quantile_l1_distance(square_eigs[128], square_eigs[256])
    if 64 in square_eigs and 256 in square_eigs:
        wd["quantile_l1_dim64_vs_dim256"] = quantile_l1_distance(square_eigs[64], square_eigs[256])

    stability = classify_stability(square_summaries)

    continuum_note = (
        "If stable or partially stable, a formal continuum candidate could be expressed as "
        r"$H^* = \overline{\lim_{d\to\infty} H^*_d}$ in an appropriate operator norm topology; "
        "a rigorous theorem still requires analytic compactness / tightness estimates and is not claimed here."
    )
    if stability == "stable":
        next_step = "V13N theorem report / spectral triple formalization"
    elif stability == "unstable":
        next_step = "V13M.1 renormalization of geo_sigma, lambda_p, and spectral scaling"
    elif stability == "incomplete_grid":
        next_step = "Re-run without --smoke so square (64,64), (128,128), (256,256) completes."
    else:
        next_step = "V13N with explicit finite-dim error budgets, or V13M.1 if dim=256 tail degrades in production"

    summary_fields = [
        "dim",
        "zeros",
        "best_spectral_log_mse",
        "best_spacing_mse_normalized",
        "best_ks_wigner",
        "operator_diff_best",
        "converged_operator",
        "best_iter",
        "self_adjointness_fro",
        "spectral_log_mse_initial",
        "spectral_log_mse_final",
        "spacing_mse_normalized_initial",
        "spacing_mse_normalized_final",
        "ks_wigner_initial",
        "ks_wigner_final",
        "final_operator_diff",
        "finite",
        "condition_proxy",
        "spectral_range",
        "eig_spacing_mean",
        "eig_spacing_std",
        "stopped_stagnation",
        "n_iter",
        "eig_error",
    ]

    def row_for_csv(s: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "dim": s.get("dim"),
            "zeros": s.get("zeros"),
            "best_spectral_log_mse": s.get("spectral_log_mse_best"),
            "best_spacing_mse_normalized": s.get("spacing_mse_normalized_best"),
            "best_ks_wigner": s.get("ks_wigner_best"),
            "operator_diff_best": s.get("operator_diff_best"),
            "converged_operator": s.get("converged_operator"),
            "best_iter": s.get("best_iter"),
            "self_adjointness_fro": s.get("self_adjointness_fro"),
            "spectral_log_mse_initial": s.get("spectral_log_mse_initial"),
            "spectral_log_mse_final": s.get("spectral_log_mse_final"),
            "spacing_mse_normalized_initial": s.get("spacing_mse_normalized_initial"),
            "spacing_mse_normalized_final": s.get("spacing_mse_normalized_final"),
            "ks_wigner_initial": s.get("ks_wigner_initial"),
            "ks_wigner_final": s.get("ks_wigner_final"),
            "final_operator_diff": s.get("final_operator_diff"),
            "finite": s.get("finite"),
            "condition_proxy": s.get("condition_proxy"),
            "spectral_range": s.get("spectral_range"),
            "eig_spacing_mean": s.get("eig_spacing_mean"),
            "eig_spacing_std": s.get("eig_spacing_std"),
            "stopped_stagnation": s.get("stopped_stagnation"),
            "n_iter": s.get("n_iter"),
            "eig_error": s.get("eig_error"),
        }

    with open(out_dir / "v13m_summary.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields, extrasaction="ignore")
        w.writeheader()
        for block in all_results:
            s = {k: v for k, v in block.items() if k != "run_key"}
            w.writerow(row_for_csv(s))

    payload = {
        "warning": "Computational scaling evidence only; not a proof of the Riemann Hypothesis.",
        "operator_formula_tex": (
            r"H^*_d = \mathrm{Sym}\bigl(L_d + 10\,K_{w,d} + \lambda_p V(H^*_d)\bigr) + \varepsilon I,\quad "
            r"\lambda_p=3,\ \varepsilon=10^{-6}."
        ),
        "hyperparameters": {
            "alpha": ALPHA,
            "lambda_p_eff": LAMBDA_P_EFF,
            "beta0": BETA0,
            "tau": TAU,
            "smooth_sigma": SMOOTH_SIGMA,
            "clip_lo": CLIP_LO,
            "clip_hi": CLIP_HI,
            "geo_weight": GEO_WEIGHT,
            "geo_sigma": GEO_SIGMA,
            "eps": EPS,
            "beta_t_formula": "max(0.03, beta0 * exp(-t / tau))",
        },
        "runs": all_results,
        "square_diagonal_summaries": {str(k): v for k, v in square_summaries.items()},
        "scaling_trends_vs_dim128": trends,
        "eigenvalue_distribution_distances": wd,
        "stability_classification": stability,
        "continuum_hypothesis_note": continuum_note,
        "next_step_recommendation": next_step,
        "acceptance": {
            "dim64_and_dim128_converged": bool(
                square_summaries.get(64, {}).get("converged_operator")
                and square_summaries.get(128, {}).get("converged_operator")
            ),
            "dim256_attempted": 256 in square_summaries,
            "all_square_runs_finite": all(
                bool(square_summaries.get(d, {}).get("finite", False)) for d in (64, 128, 256) if d in square_summaries
            ),
        },
    }
    with open(out_dir / "v13m_results.json", "w", encoding="utf-8") as f:
        json.dump(numpy_json(payload), f, indent=2, allow_nan=True)

    # Markdown report
    md = [
        "# V13M: Continuum / scaling validation\n\n",
        "## 1. Warning\n\n",
        "> **Computational scaling evidence only; not a proof of the Riemann Hypothesis.**\n\n",
        "## 2. Operator formula (finite $d$)\n\n",
        r"$$H^*_d = \mathrm{Sym}\bigl(L_d + 10\,K_{w,d} + 3\,V(H^*_d)\bigr) + \varepsilon I, \quad \varepsilon = 10^{-6},$$" "\n\n",
        "with the same V13L.2 Pareto-fixed map on the diagonal potential (scheduled $\\beta_t$, smoothed $\\delta$, percentile clip on $V_{\\mathrm{diag}}$).\n\n",
        "## 3. Scaling table (best-checkpoint metrics, square grid)\n\n",
        "| dim | zeros | best spectral_log_mse | best spacing (norm) | best KS | op_diff best | converged | best_iter | ‖skew‖_F |\n",
        "|---:|---:|---:|---:|---:|---:|---|---:|---:|\n",
    ]
    for d in (64, 128, 256):
        if d not in square_summaries:
            continue
        s = square_summaries[d]
        md.append(
            f"| {d} | {d} | {s.get('spectral_log_mse_best')} | {s.get('spacing_mse_normalized_best')} | "
            f"{s.get('ks_wigner_best')} | {s.get('operator_diff_best')} | {s.get('converged_operator')} | "
            f"{s.get('best_iter')} | {s.get('self_adjointness_fro')} |\n"
        )
    md.append("\n## 4. Stability interpretation\n\n")
    md.append(f"**Classification:** `{stability}`.\n\n")
    md.append("## 5. Continuum hypothesis (informal)\n\n")
    md.append(continuum_note + "\n\n")
    md.append("## 6. Next step\n\n")
    md.append(f"- **{next_step}**\n\n")
    md.append("## Extra: quantile L1 between eigenvalue distributions (square runs)\n\n")
    md.append(f"- {json.dumps(wd, indent=2)}\n\n")
    md.append("## Files\n\n")
    md.append("- `v13m_results.json`, `v13m_summary.csv`\n")
    md.append("- `v13m_convergence_dim{64,128,256}.csv`, `h_star_dim{64,128,256}.npy` (square diagonal only)\n")
    (out_dir / "v13m_report.md").write_text("".join(md), encoding="utf-8")

    tex = [
        "\\documentclass[11pt]{article}\n",
        "\\usepackage{amsmath,amssymb}\n",
        "\\usepackage[T1]{fontenc}\n",
        "\\usepackage[margin=1in]{geometry}\n",
        "\\title{V13M Scaling validation}\n\\date{}\n\\begin{document}\n\\maketitle\n",
        "\\section*{Warning}\n",
        "Computational scaling evidence only; not a proof of RH.\n\n",
        "\\section{Stability}\n",
        latex_escape(stability) + "\n\n",
        "\\section{Next step}\n",
        latex_escape(next_step) + "\n\n",
        "\\end{document}\n",
    ]
    (out_dir / "v13m_report.tex").write_text("".join(tex), encoding="utf-8")

    if try_pdflatex(out_dir / "v13m_report.tex", out_dir, "v13m_report.pdf"):
        print(f"Wrote {out_dir / 'v13m_report.pdf'}")
    else:
        print("PDF skipped.")

    print(f"Wrote {out_dir / 'v13m_results.json'}")
    print(f"Wrote {out_dir / 'v13m_summary.csv'} ({len(all_results)} runs)")
    print(f"Stability: {stability}")


if __name__ == "__main__":
    main()
