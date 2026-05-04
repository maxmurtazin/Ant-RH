#!/usr/bin/env python3
"""
V13K: zeta-informed diagonal potentials + scale-invariant operator grid.

  python3 scripts/run_v13k_zeta_informed.py \\
    --candidate_json runs/v13_candidate_theorem/candidate_operators.json \\
    --out_dir runs/v13k_zeta_informed \\
    --dim 128 --zeros 128 --seed 42
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

_DTF = np.float64

GEO_SIGMAS = [0.3, 0.6, 1.2]
GEO_WEIGHTS = [3.0, 10.0, 30.0]
POTENTIAL_WEIGHTS = [0.25, 1.0, 3.0, 10.0]
MODES = ["baseline", "zero_phase", "spacing_phase", "log_zero_phase", "self_consistent_phase"]
WANTED_IDS = ["seed_6", "seed_12", "seed_14", "seed_20"]

CSV_FIELDS = [
    "candidate_id",
    "word",
    "potential_mode",
    "geo_sigma",
    "geo_weight",
    "potential_weight",
    "eps",
    "dim",
    "seed",
    "spectral_raw_mse",
    "spectral_log_mse",
    "spacing_mse_normalized",
    "ks_wigner",
    "eig_min",
    "eig_max",
    "operator_fro_norm",
    "potential_norm",
    "geodesic_norm",
    "base_norm",
    "condition_proxy",
    "rank_by_spectral_log_mse",
    "eig_error",
    "operator_hash",
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
            timeout=180,
        )
        return r.returncode == 0 and (out_dir / pdf_basename).is_file()
    except (OSError, subprocess.TimeoutExpired):
        return False


def load_candidate_bundle(
    candidate_path: Path,
    pareto_path: Optional[Path],
) -> List[Dict[str, Any]]:
    with open(candidate_path, "r", encoding="utf-8") as f:
        cj = json.load(f)
    out: List[Dict[str, Any]] = []
    seen: set[str] = set()

    def add(cid: str, word: List[int]) -> None:
        if cid in seen:
            return
        seen.add(cid)
        out.append({"id": cid, "word": list(word)})

    sc = cj.get("spectral_candidate") if isinstance(cj.get("spectral_candidate"), dict) else {}
    gc = cj.get("geometric_candidate") if isinstance(cj.get("geometric_candidate"), dict) else {}
    if sc.get("id") and isinstance(sc.get("word"), list):
        add(str(sc["id"]), [int(x) for x in sc["word"]])
    if gc.get("id") and isinstance(gc.get("word"), list):
        add(str(gc["id"]), [int(x) for x in gc["word"]])

    want_extra = {"seed_14", "seed_20"}
    by_pareto: Dict[str, List[int]] = {}
    if pareto_path and pareto_path.is_file():
        with open(pareto_path, "r", encoding="utf-8") as f:
            pj = json.load(f)
        for row in pj.get("candidates") or []:
            if not isinstance(row, dict):
                continue
            rid = str(row.get("id", ""))
            w = row.get("word_used") or row.get("word")
            if rid in want_extra and isinstance(w, list):
                by_pareto[rid] = [int(x) for x in w]

    for rid in sorted(want_extra):
        if rid in by_pareto and rid not in seen:
            add(rid, by_pareto[rid])

    return out


def order_candidates(cands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_id = {str(c["id"]): c for c in cands}
    ordered: List[Dict[str, Any]] = []
    for wid in WANTED_IDS:
        if wid in by_id:
            ordered.append(by_id[wid])
    return ordered


def operator_matrix_sha256(H: np.ndarray) -> str:
    hc = np.ascontiguousarray(np.asarray(H, dtype=np.float64))
    return hashlib.sha256(hc.tobytes()).hexdigest()


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


def run_one(
    v: Any,
    vk: Any,
    *,
    cand: Dict[str, Any],
    Z: np.ndarray,
    zeros: np.ndarray,
    mode: str,
    eps: float,
    geo_sigma: float,
    geo_weight: float,
    potential_weight: float,
    seed: int,
    tau: Optional[float],
) -> Dict[str, Any]:
    """Single grid cell; deterministic Z passed in."""
    cid = str(cand.get("id", "unknown"))
    word = [int(x) for x in cand["word"]]
    geodesics = [vk.geodesic_entry_for_word(word)]

    row: Dict[str, Any] = {k: None for k in CSV_FIELDS}
    row["candidate_id"] = cid
    row["word"] = json.dumps(word)
    row["potential_mode"] = str(mode)
    row["geo_sigma"] = float(geo_sigma)
    row["geo_weight"] = float(geo_weight)
    row["potential_weight"] = float(potential_weight)
    row["eps"] = float(eps)
    row["dim"] = int(Z.shape[0])
    row["seed"] = int(seed)
    row["eig_error"] = None

    for k in (
        "spectral_raw_mse",
        "spectral_log_mse",
        "spacing_mse_normalized",
        "ks_wigner",
        "eig_min",
        "eig_max",
        "operator_fro_norm",
        "potential_norm",
        "geodesic_norm",
        "base_norm",
        "condition_proxy",
    ):
        row[k] = float("nan")

    try:
        H, meta = vk.build_v13k_operator(
            z_points=Z,
            geodesics=geodesics,
            zeros=zeros,
            mode=str(mode),
            eps=float(eps),
            geo_sigma=float(geo_sigma),
            geo_weight=float(geo_weight),
            potential_weight=float(potential_weight),
            distances=None,
            tau=tau,
            diag_shift=1e-6,
            baseline_eig_for_self_consistent=None,
        )
        H = np.asarray(H, dtype=_DTF, copy=False)
        row["operator_hash"] = operator_matrix_sha256(H)
        row["operator_fro_norm"] = float(meta.get("operator_fro_norm", np.linalg.norm(H, ord="fro")))
        row["potential_norm"] = float(meta.get("potential_norm", float("nan")))
        row["geodesic_norm"] = float(meta.get("geodesic_norm", float("nan")))
        row["base_norm"] = float(meta.get("base_norm", float("nan")))

        Ht = v._symmetrize_torch(H)
        eig, err = v._eigvalsh_safe(Ht)
        if err or eig.size == 0:
            row["eig_error"] = err or "empty_eigenvalues"
            return row
        if not np.isfinite(eig).all():
            row["eig_error"] = "nonfinite_eigenvalues"
            return row

        eig = np.sort(eig.astype(_DTF, copy=False))
        row["eig_min"] = float(np.min(eig))
        row["eig_max"] = float(np.max(eig))
        row["condition_proxy"] = condition_proxy(eig)

        z = np.asarray(zeros, dtype=_DTF).reshape(-1)
        z = z[np.isfinite(z)]
        k = int(min(int(eig.size), int(z.size)))
        if k >= 1:
            e_k = np.sort(eig[:k])
            z_k = np.sort(z[:k])
            raw_mse = float(np.mean((e_k - z_k) ** 2))
            row["spectral_raw_mse"] = raw_mse
            row["spectral_log_mse"] = float(np.log1p(max(0.0, raw_mse)))
        if k >= 2:
            row["spacing_mse_normalized"] = float(v.spacing_mse_normalized(eig[:k], z[:k]))
            nu = v.normalized_gaps(np.sort(eig[:k]))
            if nu.size >= 2:
                row["ks_wigner"] = float(v.ks_against_wigner_gue(nu))
    except Exception as ex:
        row["eig_error"] = repr(ex)
    return row


def assign_ranks(rows: List[Dict[str, Any]]) -> None:
    def keyf(r: Dict[str, Any]) -> Tuple[int, float]:
        x = r.get("spectral_log_mse")
        if isinstance(x, (int, float)) and math.isfinite(float(x)):
            return (0, float(x))
        return (1, float("inf"))

    rows_sorted = sorted(rows, key=keyf)
    for i, r in enumerate(rows_sorted):
        r["rank_by_spectral_log_mse"] = int(i + 1)


def best_finite(rows: List[Dict[str, Any]], key: str) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    best_v = float("inf")
    for r in rows:
        x = r.get(key)
        if not isinstance(x, (int, float)) or not math.isfinite(float(x)):
            continue
        xv = float(x)
        if xv < best_v:
            best_v = xv
            best = r
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description="V13K zeta-informed potential grid (V13J backbone).")
    ap.add_argument("--candidate_json", type=str, default="runs/v13_candidate_theorem/candidate_operators.json")
    ap.add_argument("--pareto_json", type=str, default="runs/v13_top5_pareto/pareto_results.json")
    ap.add_argument("--out_dir", type=str, default="runs/v13k_zeta_informed")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--zeros", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eps", type=float, default=0.6)
    ap.add_argument("--tau", type=float, default=None, help="Override tau (shared for index modes; zero_phase uses same if set).")
    args = ap.parse_args()

    v = _load_v13_validate()
    from core import v13k_zeta_operator as vk  # noqa: WPS433 (runtime import after path)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = Path(ROOT) / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cand_path = Path(args.candidate_json)
    if not cand_path.is_absolute():
        cand_path = Path(ROOT) / cand_path
    if not cand_path.is_file():
        raise SystemExit(f"missing candidate_json: {cand_path}")

    pareto_path = Path(args.pareto_json)
    if not pareto_path.is_absolute():
        pareto_path = Path(ROOT) / pareto_path

    zeros = v._load_zeros(int(args.zeros))
    if zeros.size < int(args.dim):
        raise ValueError(f"need zeros >= dim, got {zeros.size} < {args.dim}")

    bundle = load_candidate_bundle(cand_path, pareto_path if pareto_path.is_file() else None)
    cands = order_candidates(bundle)
    missing = [w for w in WANTED_IDS if w not in {str(c["id"]) for c in cands}]
    if missing:
        raise SystemExit(
            f"missing required candidate id(s) {missing}; need {WANTED_IDS}. "
            f"Ensure candidate_json has spectral/geometric ids and pareto_json supplies seed_14/seed_20 words."
        )

    from core.artin_operator import sample_domain

    Z = sample_domain(int(args.dim), seed=int(args.seed))
    tau_opt: Optional[float] = float(args.tau) if args.tau is not None else None

    rows: List[Dict[str, Any]] = []
    for cand in cands:
        for mode in MODES:
            for gs in GEO_SIGMAS:
                for gw in GEO_WEIGHTS:
                    for pw in POTENTIAL_WEIGHTS:
                        r = run_one(
                            v,
                            vk,
                            cand=cand,
                            Z=Z,
                            zeros=zeros,
                            mode=mode,
                            eps=float(args.eps),
                            geo_sigma=float(gs),
                            geo_weight=float(gw),
                            potential_weight=float(pw),
                            seed=int(args.seed),
                            tau=tau_opt,
                        )
                        rows.append(r)

    assign_ranks(rows)

    baseline_rows = [r for r in rows if r.get("potential_mode") == "baseline"]
    best_baseline_grid = best_finite(baseline_rows, "spectral_log_mse")

    v13j_ref: List[Dict[str, Any]] = []
    for cand in cands:
        cid = str(cand["id"])
        ref = next(
            (
                r
                for r in rows
                if str(r.get("candidate_id")) == cid
                and r.get("potential_mode") == "baseline"
                and float(r.get("geo_sigma", -1)) == 0.6
                and float(r.get("geo_weight", -1)) == 10.0
                and float(r.get("potential_weight", -1)) == 0.25
            ),
            None,
        )
        if ref is not None:
            v13j_ref.append(dict(ref))

    best_overall = best_finite(rows, "spectral_log_mse")
    if best_overall is None:
        raise SystemExit("no finite spectral_log_mse in grid; check operators / zeros.")

    best_by_mode: Dict[str, Any] = {}
    for m in MODES:
        sub = [r for r in rows if r.get("potential_mode") == m]
        bm = best_finite(sub, "spectral_log_mse")
        if bm is not None:
            best_by_mode[m] = dict(bm)

    ref_best = best_finite(v13j_ref, "spectral_log_mse")
    bo_sl = float(best_overall["spectral_log_mse"])
    ref_sl = float(ref_best["spectral_log_mse"]) if ref_best is not None else float("nan")
    zeta_beats_v13j_defaults = bool(math.isfinite(ref_sl) and bo_sl < ref_sl)

    best_mode_name = min(best_by_mode.keys(), key=lambda m: float(best_by_mode[m]["spectral_log_mse"])) if best_by_mode else ""
    best_cand_id = str(best_overall.get("candidate_id", ""))

    # Scale survival: coefficient of variation of spectral_log_mse over geo grid for best mode
    survival: Dict[str, Any] = {}
    for m in MODES:
        sub = [r for r in rows if r.get("potential_mode") == m and str(r.get("candidate_id")) == best_cand_id]
        xs = [float(r["spectral_log_mse"]) for r in sub if math.isfinite(float(r.get("spectral_log_mse", float("nan"))))]
        if len(xs) > 1:
            a = np.asarray(xs, dtype=np.float64)
            survival[m] = {
                "n": int(a.size),
                "std": float(np.std(a)),
                "mean": float(np.mean(a)),
                "cv": float(np.std(a) / (abs(np.mean(a)) + 1e-12)),
            }
        else:
            survival[m] = {"n": len(xs), "std": float("nan"), "mean": float("nan"), "cv": float("nan")}

    dom = {
        "best_overall_base_norm": float(best_overall.get("base_norm", float("nan"))),
        "best_overall_geodesic_norm": float(best_overall.get("geodesic_norm", float("nan"))),
        "best_overall_potential_norm": float(best_overall.get("potential_norm", float("nan"))),
        "potential_share_of_lk_plus_v": float("nan"),
    }
    b0 = dom["best_overall_base_norm"]
    g0 = dom["best_overall_geodesic_norm"]
    p0 = dom["best_overall_potential_norm"]
    if math.isfinite(b0) and math.isfinite(g0) and math.isfinite(p0):
        den = b0 + g0 + p0 + 1e-12
        dom["potential_share_of_lk_plus_v"] = float(p0 / den)

    payload: Dict[str, Any] = {
        "meta": {
            "dim": int(args.dim),
            "zeros_n": int(zeros.size),
            "seed": int(args.seed),
            "eps": float(args.eps),
            "tau_cli": tau_opt,
            "geo_sigmas": GEO_SIGMAS,
            "geo_weights": GEO_WEIGHTS,
            "potential_weights": POTENTIAL_WEIGHTS,
            "modes": MODES,
            "candidates": [{"id": c["id"], "word": c["word"]} for c in cands],
        },
        "best_overall": dict(best_overall),
        "best_by_mode": best_by_mode,
        "best_baseline_on_grid": dict(best_baseline_grid) if best_baseline_grid else None,
        "v13j_reference_aco_defaults": v13j_ref,
        "comparison": {
            "spectral_log_mse_best_overall": bo_sl,
            "spectral_log_mse_best_v13j_defaults": ref_sl,
            "zeta_informed_beats_v13j_aco_default_baseline": zeta_beats_v13j_defaults,
            "best_potential_mode": best_mode_name,
            "best_candidate_id": best_cand_id,
            "dominance_norms": dom,
            "scale_survival_cv_by_mode_best_candidate": survival,
        },
        "rows": rows,
    }

    with open(out_dir / "v13k_results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=True)

    with open(out_dir / "v13k_summary.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in CSV_FIELDS})

    with open(out_dir / "v13k_best_by_mode.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for m in MODES:
            if m in best_by_mode:
                w.writerow({k: best_by_mode[m].get(k) for k in CSV_FIELDS})

    bl_sl = float(best_baseline_grid["spectral_log_mse"]) if best_baseline_grid else float("nan")
    nonbase_best = best_finite([r for r in rows if r.get("potential_mode") != "baseline"], "spectral_log_mse")
    nb_sl = float(nonbase_best["spectral_log_mse"]) if nonbase_best else float("nan")
    nb_mode = str(nonbase_best.get("potential_mode", "")) if nonbase_best else ""

    ref_lines = ["| candidate | V13J baseline `spectral_log_mse` (σ=0.6, λ_g=10, λ_p=0.25) |\n|---|---|\n"]
    for rr in v13j_ref:
        ref_lines.append(
            f"| `{rr.get('candidate_id')}` | {float(rr.get('spectral_log_mse', float('nan'))):.8g} |\n"
        )
    ref_lines.append("\n")

    md = [
        "# V13K: Zeta-informed potential and scale grid\n\n",
        "Operator form matches V13J: "
        "$H_w = \\mathrm{Sym}(\\lambda_L L + \\lambda_g \\widehat{K}_w + \\lambda_p \\widehat{V}_w) + \\varepsilon I$, "
        "with **baseline** $\\widehat{V}_w$ the original diagonal $\\sin$ potential; other modes replace only the diagonal vector "
        "before the same max-normalization and $\\lambda_p$ scaling. Laplacian and geodesic kernel factors are unchanged.\n\n",
        "## Comparison: `best_overall` vs V13J baseline\n\n",
        f"- **best_overall** `spectral_log_mse` = **{bo_sl:.8g}** (candidate `{best_cand_id}`, mode `{best_overall.get('potential_mode')}`, "
        f"σ={best_overall.get('geo_sigma')}, λ_g={best_overall.get('geo_weight')}, λ_p={best_overall.get('potential_weight')}).\n",
        f"- **Best baseline row on the full scale grid** (same backbone, sin potential): **{bl_sl:.8g}**.\n",
        f"- **Best non-baseline (zeta-informed) row** on the grid: **{nb_sl:.8g}** (`{nb_mode}`).\n",
        f"- **V13J ACO-style reference** (baseline only, fixed hyperparameters 0.6 / 10 / 0.25): best `spectral_log_mse` among those four rows = **{ref_sl:.8g}**.\n\n",
        "".join(ref_lines),
        "## Answers (required)\n\n",
        "**1. Does zeta-informed V improve spectral_log_mse over baseline?** "
        f"Across all zeta-informed modes, the best `spectral_log_mse` is {nb_sl:.8g} vs {bl_sl:.8g} for the best pure-baseline configuration on the same grid. "
        f"Against the canonical V13J default-hyperparameter baseline rows, best overall is {bo_sl:.8g} vs {ref_sl:.8g}; "
        f"the flag `zeta_informed_beats_v13j_aco_default_baseline` in `v13k_results.json` is **{zeta_beats_v13j_defaults}** "
        "(true means `best_overall` strictly improves on the best of those four reference rows).\n\n",
        f"**2. Which potential mode is best?** Per-mode minima are in `v13k_best_by_mode.csv`; the mode with the lowest of those minima is **`{best_mode_name}`**.\n\n",
        f"**3. Which candidate remains best?** **`{best_cand_id}`** by `spectral_log_mse` over the full grid (`best_overall`).\n\n",
        "**4. Does performance survive geo_sigma / geo_weight scaling?** "
        "Inspect `comparison.scale_survival_cv_by_mode_best_candidate` in `v13k_results.json`: for the overall-best candidate, "
        "each mode lists the coefficient of variation of `spectral_log_mse` over all 36 hyperparameter triples. "
        "Smaller CV indicates less sensitivity to the geo and potential weights in the grid.\n\n",
        "**5. Is the operator still dominated by L + K, or does V become meaningful?** "
        f"At `best_overall`, Frobenius norms: $\\|H_0\\|_F$ = {b0:.6g}, $\\| \\lambda_g K \\|_F$ = {g0:.6g}, "
        f"$\\| \\lambda_p V \\|_F$ = {p0:.6g}. The potential share "
        f"$\\|V\\|_F / (\\|H_0\\|_F + \\|G\\|_F + \\|V\\|_F)$ ≈ **{dom.get('potential_share_of_lk_plus_v', float('nan')):.4f}**.\n\n",
        "## Files\n\n",
        "- `v13k_summary.csv` — one row per candidate × mode × (geo_sigma, geo_weight, potential_weight).\n",
        "- `v13k_best_by_mode.csv` — best row per potential mode.\n",
        "- `v13k_results.json` — all rows, `best_overall`, `v13j_reference_aco_defaults`, and `comparison`.\n\n",
    ]
    (out_dir / "v13k_report.md").write_text("".join(md), encoding="utf-8")

    tex = [
        "\\documentclass[11pt]{article}\n",
        "\\usepackage{amsmath,amssymb}\n",
        "\\usepackage[T1]{fontenc}\n",
        "\\usepackage[margin=1in]{geometry}\n",
        "\\title{V13K Zeta-informed potential}\n\\date{}\n\\begin{document}\n\\maketitle\n",
        "\\section{Summary}\n",
        f"Best overall $\\log(1+\\mathrm{{MSE}})$ spectral loss: {latex_escape(f'{bo_sl:.8g}')}.\n\n",
        f"Best mode: \\texttt{{{latex_escape(best_mode_name)}}}; best candidate: \\texttt{{{latex_escape(best_cand_id)}}}.\n\n",
        f"Beats V13J ACO-default baseline row: \\textbf{{{latex_escape(str(zeta_beats_v13j_defaults))}}}.\n\n",
        "\\section{Dominance (Frobenius norms at best overall)}\n",
        f"Base (L): {latex_escape(str(b0))}, Geodesic ($K$): {latex_escape(str(g0))}, "
        f"Potential ($V$): {latex_escape(str(p0))}.\n\n",
        "\\end{document}\n",
    ]
    (out_dir / "v13k_report.tex").write_text("".join(tex), encoding="utf-8")

    if try_pdflatex(out_dir / "v13k_report.tex", out_dir, "v13k_report.pdf"):
        print(f"Wrote {out_dir / 'v13k_report.pdf'}")
    else:
        print("PDF skipped (pdflatex missing or failed).")

    print(f"Wrote {out_dir / 'v13k_results.json'} ({len(rows)} rows)")
    print(f"Wrote {out_dir / 'v13k_summary.csv'}")
    print(f"Wrote {out_dir / 'v13k_best_by_mode.csv'}")
    print(f"Wrote {out_dir / 'v13k_report.md'} / v13k_report.tex")


if __name__ == "__main__":
    main()
