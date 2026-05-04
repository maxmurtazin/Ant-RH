#!/usr/bin/env python3
"""
V13L: self-consistent operator via fixed-point / damped spectral feedback.

  python3 scripts/run_v13l_self_consistent.py \\
    --candidate_json runs/v13_candidate_theorem/candidate_operators.json \\
    --out_dir runs/v13l_self_consistent \\
    --dim 128 --zeros 128 --seed 42 --max_iter 30 --tol 1e-4
"""

from __future__ import annotations

import argparse
import csv
import hashlib
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

WANTED_IDS = ["seed_6", "seed_12", "seed_14", "seed_20"]
ALPHAS = [0.1, 0.3, 0.5, 1.0]
LAMBDA_PS = [0.5, 1.0, 3.0]


def _load_v13_validate() -> Any:
    import importlib.util

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


def load_candidate_bundle(candidate_path: Path, pareto_path: Optional[Path]) -> List[Dict[str, Any]]:
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


def operator_matrix_sha256(H: np.ndarray) -> str:
    hc = np.ascontiguousarray(np.asarray(H, dtype=np.float64))
    return hashlib.sha256(hc.tobytes()).hexdigest()


def order_candidates(cands: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_id = {str(c["id"]): c for c in cands}
    ordered: List[Dict[str, Any]] = []
    for wid in WANTED_IDS:
        if wid in by_id:
            ordered.append(by_id[wid])
    return ordered


def numpy_to_jsonable(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return float(x) if isinstance(x, np.floating) else int(x)
    if isinstance(x, dict):
        return {k: numpy_to_jsonable(v) for k, v in x.items()}
    if isinstance(x, list):
        return [numpy_to_jsonable(v) for v in x]
    return x


def plot_convergence(
    out_path: Path,
    series_by_label: List[Tuple[str, List[float]]],
    *,
    title: str,
    ylabel: str,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for label, ys in series_by_label:
        xs = list(range(len(ys)))
        ax.plot(xs, ys, marker="o", ms=3, lw=1.2, label=label[:48])
    ax.set_xlabel("iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path.is_file()


def plot_eig_alignment(
    out_path: Path,
    idx: np.ndarray,
    eig_before: np.ndarray,
    eig_after: np.ndarray,
    gamma_true: np.ndarray,
    *,
    title: str,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(idx, gamma_true, "k--", lw=1.5, label="zeta zeros (sorted)")
    ax.plot(idx, eig_before, "s", ms=4, alpha=0.7, label="eig before")
    ax.plot(idx, eig_after, "o", ms=4, alpha=0.7, label="eig after")
    ax.set_xlabel("index")
    ax.set_ylabel("value")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path.is_file()


def main() -> None:
    ap = argparse.ArgumentParser(description="V13L self-consistent RH-style operator iteration.")
    ap.add_argument("--candidate_json", type=str, default="runs/v13_candidate_theorem/candidate_operators.json")
    ap.add_argument("--pareto_json", type=str, default="runs/v13_top5_pareto/pareto_results.json")
    ap.add_argument("--out_dir", type=str, default="runs/v13l_self_consistent")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--zeros", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_iter", type=int, default=30)
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--eps", type=float, default=0.6)
    ap.add_argument("--geo_sigma", type=float, default=0.6)
    ap.add_argument("--geo_weight", type=float, default=10.0)
    ap.add_argument("--beta", type=float, default=0.3, help="Damping: H <- (1-beta) H + beta H_new.")
    ap.add_argument("--smooth_sigma", type=float, default=2.0, help="Gaussian smooth on V_diag (0 to disable).")
    ap.add_argument("--no_smooth", action="store_true", help="Disable Gaussian smoothing on V_diag.")
    args = ap.parse_args()

    v = _load_v13_validate()
    from core import v13l_self_consistent as v13l
    from core.artin_operator import sample_domain

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = Path(ROOT) / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    h_dir = out_dir / "h_star"
    h_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

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
            f"missing candidate id(s) {missing}; need {WANTED_IDS} and pareto for seed_14/seed_20."
        )

    Z = sample_domain(int(args.dim), seed=int(args.seed))
    use_smooth = not bool(args.no_smooth)
    sm_sig = 0.0 if not use_smooth else float(args.smooth_sigma)

    spacing_fn = v.spacing_mse_normalized
    ks_fn = v.ks_against_wigner_gue
    norm_gaps_fn = v.normalized_gaps

    summary_rows: List[Dict[str, Any]] = []
    conv_rows: List[Dict[str, Any]] = []
    experiments: List[Dict[str, Any]] = []
    candidate_monotonic: Dict[str, bool] = {str(c["id"]): False for c in cands}

    for cand in cands:
        cid = str(cand["id"])
        word = [int(x) for x in cand["word"]]
        geodesics = [v13l.geodesic_entry_for_word(word)]

        H_base, _hb_meta = v13l.build_h_base_no_potential(
            z_points=Z,
            geodesics=geodesics,
            eps=float(args.eps),
            geo_sigma=float(args.geo_sigma),
            geo_weight=float(args.geo_weight),
            distances=None,
            diag_shift=1e-6,
        )
        eig_before = np.sort(np.linalg.eigvalsh(0.5 * (H_base + H_base.T)).astype(_DTF))
        k = int(min(int(eig_before.size), int(zeros.size)))
        z_true = np.sort(np.asarray(zeros[:k], dtype=_DTF))
        idx = np.arange(k, dtype=_DTF)

        for alpha in ALPHAS:
            for lp in LAMBDA_PS:
                H_star, eig_star, metrics_list, run_meta = v13l.run_self_consistent_loop(
                    H_base=H_base,
                    gamma=zeros,
                    alpha=float(alpha),
                    lambda_p=float(lp),
                    beta=float(args.beta),
                    max_iter=int(args.max_iter),
                    tol=float(args.tol),
                    diag_shift=1e-6,
                    smooth_sigma=sm_sig,
                    use_smooth=use_smooth,
                    zeros_eval=zeros,
                    spacing_fn=spacing_fn,
                    ks_fn=ks_fn,
                    norm_gaps_fn=norm_gaps_fn,
                )

                if run_meta.get("monotonic_spectral_prefix"):
                    candidate_monotonic[cid] = True

                h_path = h_dir / f"{cid}_alpha{alpha}_lp{lp}.npy"
                H_arr = np.asarray(H_star, dtype=_DTF, copy=False)
                np.save(str(h_path), H_arr)

                init0 = metrics_list[0] if metrics_list else {}
                fin = metrics_list[-1] if metrics_list else {}
                init_sl = float(init0.get("spectral_log_mse", float("nan")))
                fin_sl = float(fin.get("spectral_log_mse", float("nan")))
                impr = float(init_sl - fin_sl) if math.isfinite(init_sl) and math.isfinite(fin_sl) else float("nan")

                summary_rows.append(
                    {
                        "candidate_id": cid,
                        "word": json.dumps(word),
                        "alpha": float(alpha),
                        "lambda_p": float(lp),
                        "beta": float(args.beta),
                        "geo_sigma": float(args.geo_sigma),
                        "geo_weight": float(args.geo_weight),
                        "eps": float(args.eps),
                        "dim": int(args.dim),
                        "seed": int(args.seed),
                        "max_iter": int(args.max_iter),
                        "n_iter": int(run_meta.get("n_iter", 0)),
                        "converged": bool(run_meta.get("converged", False)),
                        "eig_error": run_meta.get("eig_error"),
                        "spectral_log_mse_initial": init_sl,
                        "spectral_log_mse_final": fin_sl,
                        "improvement_spectral_log": impr,
                        "spacing_mse_normalized_initial": init0.get("spacing_mse_normalized", float("nan")),
                        "spacing_mse_initial": init0.get("spacing_mse_normalized", float("nan")),
                        "spacing_mse_normalized_final": fin.get("spacing_mse_normalized", float("nan")),
                        "spacing_mse_final": fin.get("spacing_mse_normalized", float("nan")),
                        "ks_wigner_initial": init0.get("ks_wigner", float("nan")),
                        "ks_wigner_final": fin.get("ks_wigner", float("nan")),
                        "delta_norm_final": fin.get("delta_norm", float("nan")),
                        "operator_diff_final": fin.get("operator_diff", float("nan")),
                        "monotonic_spectral_prefix": bool(run_meta.get("monotonic_spectral_prefix", False)),
                        "operator_hash": operator_matrix_sha256(H_arr),
                        "h_star_path": str(h_path.relative_to(out_dir)),
                    }
                )

                for m in metrics_list:
                    conv_rows.append(
                        {
                            "candidate_id": cid,
                            "alpha": float(alpha),
                            "lambda_p": float(lp),
                            "iter": int(m["iter"]),
                            "spectral_log_mse": m["spectral_log_mse"],
                            "spacing_mse_normalized": m["spacing_mse_normalized"],
                            "spacing_mse": m["spacing_mse_normalized"],
                            "ks_wigner": m["ks_wigner"],
                            "delta_norm": m["delta_norm"],
                            "operator_diff": m["operator_diff"],
                        }
                    )

                experiments.append(
                    {
                        "candidate_id": cid,
                        "word": word,
                        "alpha": float(alpha),
                        "lambda_p": float(lp),
                        "metrics_list": metrics_list,
                        "run_meta": run_meta,
                        "eig_before_first_k": eig_before[:k].tolist(),
                        "eig_star_first_k": (np.asarray(eig_star[:k], dtype=_DTF).tolist() if eig_star.size else []),
                        "gamma_true_first_k": z_true.tolist(),
                        "h_star_path": str(h_path.relative_to(out_dir)),
                    }
                )

    # Best experiment by final spectral_log_mse
    def fin_sl_row(r: Dict[str, Any]) -> float:
        x = r.get("spectral_log_mse_final")
        return float(x) if isinstance(x, (int, float)) and math.isfinite(float(x)) else float("inf")

    summary_rows_sorted = sorted(summary_rows, key=fin_sl_row)
    best_self_consistent: Dict[str, Any] = dict(summary_rows_sorted[0]) if summary_rows_sorted else {}
    if best_self_consistent:
        bid = str(best_self_consistent.get("candidate_id"))
        ba, blp = float(best_self_consistent["alpha"]), float(best_self_consistent["lambda_p"])
        bexp = next(
            (e for e in experiments if e["candidate_id"] == bid and e["alpha"] == ba and e["lambda_p"] == blp),
            None,
        )
        if bexp:
            best_self_consistent["eig_before_first_k"] = bexp.get("eig_before_first_k")
            best_self_consistent["eig_star_first_k"] = bexp.get("eig_star_first_k")
            best_self_consistent["gamma_true_first_k"] = bexp.get("gamma_true_first_k")
            best_self_consistent["metrics_list"] = bexp.get("metrics_list")

    any_monotonic = any(candidate_monotonic.values())
    acceptance = {
        "any_candidate_monotonic_spectral_prefix_len3": bool(any_monotonic),
        "per_candidate": dict(candidate_monotonic),
        "note": "Monotonic means first three spectral_log_mse values are non-increasing with net improvement (m0>m2).",
    }

    criteria_best: Dict[str, Any] = {}
    ml = best_self_consistent.get("metrics_list") if isinstance(best_self_consistent, dict) else None
    if isinstance(ml, list) and len(ml) >= 2:
        m0, m1 = ml[0], ml[-1]
        s0 = float(m0.get("spacing_mse_normalized", float("nan")))
        s1 = float(m1.get("spacing_mse_normalized", float("nan")))
        k0 = float(m0.get("ks_wigner", float("nan")))
        k1 = float(m1.get("ks_wigner", float("nan")))
        criteria_best = {
            "spectral_log_mse_decreased_overall": float(m0.get("spectral_log_mse", float("nan")))
            > float(m1.get("spectral_log_mse", float("nan"))),
            "spacing_mse_stable_or_improved": bool(
                (not math.isfinite(s0) or not math.isfinite(s1)) or (s1 <= s0 * (1.0 + 1e-9))
            ),
            "ks_wigner_improved_lower_better": bool(
                (not math.isfinite(k0) or not math.isfinite(k1)) or (k1 < k0)
            ),
            "operator_diff_final": float(m1.get("operator_diff", float("nan"))),
            "converged": bool(best_self_consistent.get("converged", False)),
        }
    acceptance["criteria_best_run_vs_initial"] = criteria_best

    # Figures: per candidate best run for that candidate
    for cand in cands:
        cid = str(cand["id"])
        sub = [r for r in summary_rows if str(r["candidate_id"]) == cid]
        if not sub:
            continue
        best_c = min(sub, key=fin_sl_row)
        a0, lp0 = float(best_c["alpha"]), float(best_c["lambda_p"])
        exp = next(
            (e for e in experiments if e["candidate_id"] == cid and e["alpha"] == a0 and e["lambda_p"] == lp0),
            None,
        )
        if exp and exp.get("metrics_list"):
            sl = [float(m["spectral_log_mse"]) for m in exp["metrics_list"]]
            dn = [float(m["delta_norm"]) for m in exp["metrics_list"]]
            plot_convergence(
                fig_dir / f"{cid}_spectral_log_mse.png",
                [(f"α={a0}, λ_p={lp0}", sl)],
                title=f"{cid}: spectral_log_mse vs iteration",
                ylabel="spectral_log_mse",
            )
            plot_convergence(
                fig_dir / f"{cid}_delta_norm.png",
                [(f"α={a0}, λ_p={lp0}", dn)],
                title=f"{cid}: delta_norm vs iteration",
                ylabel="mean |λ_i - γ_i|",
            )
            eb = np.asarray(exp["eig_before_first_k"], dtype=_DTF)
            ea = np.asarray(exp["eig_star_first_k"], dtype=_DTF)
            gt = np.asarray(exp["gamma_true_first_k"], dtype=_DTF)
            if eb.size == ea.size == gt.size and eb.size > 0:
                plot_eig_alignment(
                    fig_dir / f"{cid}_eig_alignment.png",
                    np.arange(eb.size, dtype=_DTF),
                    eb,
                    ea,
                    gt,
                    title=f"{cid}: eigenvalues vs zeros (best α={a0}, λ_p={lp0})",
                )

    payload: Dict[str, Any] = {
        "meta": {
            "dim": int(args.dim),
            "zeros_n": int(zeros.size),
            "seed": int(args.seed),
            "max_iter": int(args.max_iter),
            "tol": float(args.tol),
            "beta": float(args.beta),
            "alphas": ALPHAS,
            "lambda_ps": LAMBDA_PS,
            "smooth_sigma": sm_sig,
            "use_smooth": use_smooth,
        },
        "best_self_consistent": numpy_to_jsonable(best_self_consistent),
        "acceptance": numpy_to_jsonable(acceptance),
        "summary": summary_rows,
        "experiments": numpy_to_jsonable(experiments),
    }

    with open(out_dir / "v13l_results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=True)

    sum_fields = list(summary_rows[0].keys()) if summary_rows else []
    with open(out_dir / "v13l_summary.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sum_fields, extrasaction="ignore")
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    conv_fields = list(conv_rows[0].keys()) if conv_rows else []
    with open(out_dir / "v13l_convergence.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=conv_fields, extrasaction="ignore")
        w.writeheader()
        for r in conv_rows:
            w.writerow(r)

    best_id = str(best_self_consistent.get("candidate_id", ""))
    md = [
        "# V13L: Self-consistent operator (fixed-point iteration)\n\n",
        "## Method\n\n",
        "- **H_base** = Sym(λ_L L + λ_g K̂_w) + ε I with **no** sin potential (`potential_weight=0`), same V13J kernel normalization.\n",
        "- **Targets**: first n sorted zeta ordinates, **linearly rescaled** to the spectral interval "
        "[min λ(H_base), max λ(H_base)] to form γ_scaled (sorted).\n",
        "- **Iteration**: sorted λ_t vs γ_scaled → δ_i = λ_i − γ_i, diagonal V with V_i = −α δ_i (optional Gaussian smooth), "
        "**H_new** = Sym(H_base + λ_p V) + ε I, then **H ← (1−β)H + β H_new** with β=" + f"{float(args.beta)}.\n",
        "- **Convergence**: stop when ‖H_new − H_t‖_F < tol before the damped update, or after max_iter.\n\n",
        "## Acceptance checks\n\n",
        f"- Any candidate with **non-increasing** `spectral_log_mse` over the **first three** iterations "
        f"(lower is better), with a net drop from iter 0 to iter 2: **{any_monotonic}**.\n",
        "- Per-candidate flags: " + json.dumps(candidate_monotonic) + ".\n\n",
        "## Best configuration\n\n",
        f"- **`best_self_consistent`** (lowest final `spectral_log_mse`): candidate `{best_id}`, "
        f"α={best_self_consistent.get('alpha')}, λ_p={best_self_consistent.get('lambda_p')}, "
        f"final spectral_log_mse={best_self_consistent.get('spectral_log_mse_final')}, "
        f"converged={best_self_consistent.get('converged')}, n_iter={best_self_consistent.get('n_iter')}.\n\n",
        "## Evaluation criteria (inspect CSV/JSON)\n\n",
        "1. **spectral_log_mse** along iterations: `v13l_convergence.csv` (and per-experiment `metrics_list` in `v13l_results.json`).\n",
        "2. **spacing_mse_normalized** (`spacing_mse` column duplicates this): same files.\n",
        "3. **ks_wigner**: same.\n",
        "4. **Convergence**: `operator_diff` column and `converged` in summary.\n",
        "5. **Alignment**: compare `eig_before_first_k` vs `eig_star_first_k` vs `gamma_true_first_k` in JSON; figures in `figures/`.\n\n",
        "## Criteria snapshot (`best_self_consistent` run)\n\n",
        f"- From `acceptance.criteria_best_run_vs_initial`: {json.dumps(criteria_best, indent=2)}\n\n",
        "## Files\n\n",
        "- `v13l_summary.csv` — final metrics per (candidate, α, λ_p).\n",
        "- `v13l_convergence.csv` — per-iteration metrics.\n",
        "- `h_star/*.npy` — final H* matrices.\n",
        "- `figures/*.png` — convergence and alignment plots (best α, λ_p per candidate).\n\n",
    ]
    (out_dir / "v13l_report.md").write_text("".join(md), encoding="utf-8")

    fig_note = ""
    _sp = fig_dir / f"{best_id}_spectral_log_mse.png"
    if best_id and _sp.is_file():
        fig_note = (
            "\\begin{center}\n\\includegraphics[width=0.92\\textwidth]{"
            + f"figures/{best_id}_spectral_log_mse.png"
            + "}\n\\end{center}\n\n"
        )

    tex = [
        "\\documentclass[11pt]{article}\n",
        "\\usepackage{amsmath,amssymb,graphicx}\n",
        "\\usepackage[T1]{fontenc}\n",
        "\\usepackage[margin=1in]{geometry}\n",
        "\\title{V13L Self-consistent operator}\n\\date{}\n\\begin{document}\n\\maketitle\n",
        "\\section{Summary}\n",
        f"Best candidate (lowest final spectral log MSE): \\texttt{{{latex_escape(best_id)}}} with "
        f"$\\alpha={best_self_consistent.get('alpha')}$, $\\lambda_p={best_self_consistent.get('lambda_p')}$.\n\n",
        f"Acceptance (monotonic first three spectral\\_log\\_mse): \\textbf{{{latex_escape(str(any_monotonic))}}}.\n\n",
        fig_note,
        "\\end{document}\n",
    ]
    (out_dir / "v13l_report.tex").write_text("".join(tex), encoding="utf-8")

    if try_pdflatex(out_dir / "v13l_report.tex", out_dir, "v13l_report.pdf"):
        print(f"Wrote {out_dir / 'v13l_report.pdf'}")
    else:
        print("PDF skipped (pdflatex missing or failed).")

    print(f"Wrote {out_dir / 'v13l_results.json'}")
    print(f"Wrote {out_dir / 'v13l_summary.csv'} ({len(summary_rows)} rows)")
    print(f"Wrote {out_dir / 'v13l_convergence.csv'} ({len(conv_rows)} rows)")
    print(f"Wrote {out_dir / 'v13l_report.md'} / v13l_report.tex")


if __name__ == "__main__":
    main()
