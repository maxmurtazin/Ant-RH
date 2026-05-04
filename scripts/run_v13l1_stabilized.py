#!/usr/bin/env python3
"""
V13L.1: stabilized fixed-point self-consistent iteration for the primary seed_6 configuration.

  python3 scripts/run_v13l1_stabilized.py \\
    --candidate_json runs/v13_candidate_theorem/candidate_operators.json \\
    --out_dir runs/v13l1_stabilized \\
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
from typing import Any, Dict, List, Optional

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
LAMBDA_P = 3.0


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


def operator_matrix_sha256(H: np.ndarray) -> str:
    hc = np.ascontiguousarray(np.asarray(H, dtype=np.float64))
    return hashlib.sha256(hc.tobytes()).hexdigest()


def metrics_from_H(
    v: Any,
    v13l: Any,
    H: np.ndarray,
    zeros: np.ndarray,
) -> Dict[str, Any]:
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


def main() -> None:
    ap = argparse.ArgumentParser(description="V13L.1 stabilized self-consistent iteration (seed_6 primary).")
    ap.add_argument("--candidate_json", type=str, default="runs/v13_candidate_theorem/candidate_operators.json")
    ap.add_argument("--out_dir", type=str, default="runs/v13l1_stabilized")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--zeros", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_iter", type=int, default=200)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--eps", type=float, default=0.6)
    ap.add_argument("--geo_sigma", type=float, default=0.6)
    ap.add_argument("--geo_weight", type=float, default=10.0)
    ap.add_argument("--potential_weight_v13j", type=float, default=0.25, help="V13J baseline sin-V weight.")
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

    # Optional: confirm JSON mentions seed_6 (informational only)
    with open(cand_path, "r", encoding="utf-8") as f:
        cj = json.load(f)
    sc = cj.get("spectral_candidate") if isinstance(cj.get("spectral_candidate"), dict) else {}
    if str(sc.get("id", "")) == PRIMARY_ID and isinstance(sc.get("word"), list):
        wj = [int(x) for x in sc["word"]]
        if wj != word:
            print(f"[v13l1] note: candidate_json seed_6 word {wj} differs from primary {word}; using primary word.", flush=True)

    H_base, _ = v13l.build_h_base_no_potential(
        z_points=Z,
        geodesics=geodesics,
        eps=float(args.eps),
        geo_sigma=float(args.geo_sigma),
        geo_weight=float(args.geo_weight),
        distances=None,
        diag_shift=1e-6,
    )

    # --- V13J baseline (sin V, ACO-style weights) ---
    H_v13j, _rep_j = build_word_sensitive_operator(
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

    # --- V13L same configuration (fixed beta=0.3, original smoothing on V_diag) ---
    H_l, eig_l, rows_l, meta_l = v13l.run_self_consistent_loop(
        H_base=H_base,
        gamma=zeros,
        alpha=float(ALPHA),
        lambda_p=float(LAMBDA_P),
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
        "spectral_log_mse": float(fin_l.get("spectral_log_mse", float("nan"))),
        "spacing_mse_normalized": float(fin_l.get("spacing_mse_normalized", float("nan"))),
        "ks_wigner": float(fin_l.get("ks_wigner", float("nan"))),
        "delta_norm": float(fin_l.get("delta_norm", float("nan"))),
        "operator_diff_final": float(fin_l.get("operator_diff", float("nan"))),
        "n_iter": int(meta_l.get("n_iter", 0)),
        "converged": bool(meta_l.get("converged", False)),
        "eig_error": meta_l.get("eig_error"),
        "self_adjointness_fro": float(v13l1.self_adjointness_fro(H_l)),
    }

    # --- V13L.1 stabilized ---
    H_star, eig_star, conv_rows, meta_1 = v13l1.run_stabilized_self_consistent(
        H_base=H_base,
        gamma=zeros,
        alpha=float(ALPHA),
        lambda_p=float(LAMBDA_P),
        max_iter=int(args.max_iter),
        tol=float(args.tol),
        diag_shift=1e-6,
        zeros_eval=zeros,
        spacing_fn=v.spacing_mse_normalized,
        ks_fn=v.ks_against_wigner_gue,
        norm_gaps_fn=v.normalized_gaps,
    )
    np.save(str(out_dir / "h_star_stabilized.npy"), np.asarray(H_star, dtype=_DTF))

    fin = conv_rows[-1] if conv_rows else {}
    final_metrics = metrics_from_H(v, v13l, H_star, zeros)
    final_metrics.update(
        {
            "operator_diff_final": float(fin.get("operator_diff", float("nan"))),
            "delta_norm_final": float(fin.get("delta_norm", float("nan"))),
            "n_iter": int(meta_1.get("n_iter", 0)),
            "converged_operator": bool(meta_1.get("converged_operator", False)),
            "stopped_stagnation": bool(meta_1.get("stopped_stagnation", False)),
            "self_adjointness_fro_final": float(meta_1.get("self_adjointness_fro_final", float("nan"))),
        }
    )

    thr = {
        "operator_diff_final_lt": 1e-3,
        "spectral_log_mse_lte": 7.48,
        "spacing_mse_normalized_lte": 16.84,
        "ks_wigner_lte": 0.451,
    }
    od = float(final_metrics.get("operator_diff_final", float("nan")))
    sl = float(final_metrics.get("spectral_log_mse", float("nan")))
    sp = float(final_metrics.get("spacing_mse_normalized", float("nan")))
    ks = float(final_metrics.get("ks_wigner", float("nan")))
    sa = float(final_metrics.get("self_adjointness_fro_final", float("nan")))

    success = {
        "all_finite": bool(final_metrics.get("finite", False)),
        "operator_diff_final_lt_1e-3": bool(math.isfinite(od) and od < thr["operator_diff_final_lt"]),
        "spectral_log_mse_lte_7_48": bool(math.isfinite(sl) and sl <= thr["spectral_log_mse_lte"]),
        "spacing_mse_normalized_lte_16_84": bool(math.isfinite(sp) and sp <= thr["spacing_mse_normalized_lte"]),
        "ks_wigner_lte_0_451": bool(math.isfinite(ks) and ks <= thr["ks_wigner_lte"]),
        "self_adjointness_fro_near_zero": bool(math.isfinite(sa) and sa < 1e-8),
    }
    success["all_key_criteria"] = all(
        [
            success["all_finite"],
            success["operator_diff_final_lt_1e-3"],
            success["spectral_log_mse_lte_7_48"],
            success["spacing_mse_normalized_lte_16_84"],
            success["ks_wigner_lte_0_451"],
        ]
    )

    payload: Dict[str, Any] = {
        "meta": {
            "candidate_id": PRIMARY_ID,
            "word": word,
            "alpha": ALPHA,
            "lambda_p": LAMBDA_P,
            "dim": int(args.dim),
            "zeros_n": int(zeros.size),
            "seed": int(args.seed),
            "max_iter": int(args.max_iter),
            "tol": float(args.tol),
            "stabilization": {
                "beta_schedule": "max(0.05, 0.3*exp(-t/50))",
                "v_diag_clip": "percentile 1%–99% then abs cap",
                "delta_smoothing_sigmas": [2.0, 4.0, 8.0],
                "mixing": "H <- (1-beta_t) H + beta_t H_new with H_new = Sym(H_base + lambda_p diag(V))",
            },
        },
        "comparison": {
            "v13j_baseline": v13j_metrics,
            "v13l_same_config_fixed_beta": v13l_metrics,
            "v13l1_stabilized": dict(final_metrics),
        },
        "success_criteria_thresholds": thr,
        "success_flags": success,
        "run_meta": {k: v for k, v in meta_1.items()},
        "operator_hash_stabilized": operator_matrix_sha256(H_star),
        "convergence": conv_rows,
    }

    with open(out_dir / "v13l1_results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=True)

    csv_fields = [
        "iter",
        "beta_t",
        "spectral_log_mse",
        "spacing_mse_normalized",
        "ks_wigner",
        "delta_norm",
        "operator_diff",
        "eig_min",
        "eig_max",
    ]
    with open(out_dir / "v13l1_convergence.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        w.writeheader()
        for r in conv_rows:
            w.writerow({k: r.get(k) for k in csv_fields})

    # Convergence plot
    try:
        import matplotlib.pyplot as plt

        if conv_rows:
            xs = [int(r["iter"]) for r in conv_rows]
            ys = [float(r["spectral_log_mse"]) for r in conv_rows]
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(xs, ys, "b-", lw=1.2)
            ax.set_xlabel("iteration")
            ax.set_ylabel("spectral_log_mse")
            ax.set_title("V13L.1 stabilized: spectral log MSE")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(out_dir / "v13l1_spectral_log_mse.png", dpi=120)
            plt.close(fig)
    except Exception:
        pass

    md = [
        "# V13L.1: Stabilized self-consistent iteration\n\n",
        "## Configuration\n\n",
        f"- Candidate **{PRIMARY_ID}**, word `{word}`, α={ALPHA}, λ_p={LAMBDA_P}.\n",
        f"- dim={args.dim}, zeros={args.zeros}, seed={args.seed}, max_iter={args.max_iter}, tol={args.tol}.\n\n",
        "## Stabilization\n\n",
        "- **β schedule:** β_t = max(0.05, 0.3·exp(−t/50)).\n",
        "- **V_diag:** Gaussian smooth on **δ** with σ ∈ {2, 4, 8} in sequence; then clip to **1%–99%** percentiles; then absolute cap from spectrum width.\n",
        "- **Mixing:** H ← (1−β_t)H + β_t H_new with H_new = Sym(H_base + λ_p diag(V)) + εI (Anderson-style map).\n",
        "- **Stop:** ‖H_new−H‖_F < tol **or** fewer than 1e−5 improvement in spectral_log_mse for 20 consecutive iterations.\n\n",
        "## Comparison (same Z, same word)\n\n",
        "| Model | spectral_log_mse | spacing_mse_norm | ks_wigner | notes |\n",
        "|---|---:|---:|---:|---|\n",
        f"| V13J baseline | {v13j_metrics.get('spectral_log_mse', float('nan'))} | {v13j_metrics.get('spacing_mse_normalized', float('nan'))} | {v13j_metrics.get('ks_wigner', float('nan'))} | sin V, λ_p={args.potential_weight_v13j} |\n",
        f"| V13L (β=0.3 fixed) | {v13l_metrics.get('spectral_log_mse', float('nan'))} | {v13l_metrics.get('spacing_mse_normalized', float('nan'))} | {v13l_metrics.get('ks_wigner', float('nan'))} | n_iter={v13l_metrics.get('n_iter')} |\n",
        f"| **V13L.1 stabilized** | **{sl}** | **{sp}** | **{ks}** | n_iter={final_metrics.get('n_iter')}, ‖H_new−H‖_final={od} |\n\n",
        "## Key success criteria (automated flags)\n\n",
        f"- Thresholds: {json.dumps(thr)}\n",
        f"- Flags: {json.dumps(success, indent=2)}\n\n",
        f"- Self-adjoint skew Fro norm (final H*): {sa}\n\n",
        "## Outputs\n\n",
        "- `v13l1_results.json`, `v13l1_convergence.csv`, `h_star_stabilized.npy`\n",
        "- `v13l1_spectral_log_mse.png` (if matplotlib available)\n\n",
    ]
    (out_dir / "v13l1_report.md").write_text("".join(md), encoding="utf-8")

    fig_tex = ""
    if (out_dir / "v13l1_spectral_log_mse.png").is_file():
        fig_tex = (
            "\\begin{center}\n\\includegraphics[width=0.85\\textwidth]{v13l1_spectral_log_mse.png}\n"
            "\\end{center}\n\n"
        )

    tex = [
        "\\documentclass[11pt]{article}\n",
        "\\usepackage{amsmath,amssymb,graphicx}\n",
        "\\usepackage[T1]{fontenc}\n",
        "\\usepackage[margin=1in]{geometry}\n",
        "\\title{V13L.1 Stabilized self-consistent operator}\n\\date{}\n\\begin{document}\n\\maketitle\n",
        "\\section{Comparison}\n",
        "V13J baseline vs V13L fixed $\\beta$ vs V13L.1 stabilized (see \\texttt{v13l1\\_report.md} for table).\n\n",
        f"Final stabilized spectral\\_log\\_mse: {latex_escape(str(sl))}.\n\n",
        fig_tex,
        "\\section{Success flags}\n",
        latex_escape(json.dumps(success)) + "\n\n",
        "\\end{document}\n",
    ]
    (out_dir / "v13l1_report.tex").write_text("".join(tex), encoding="utf-8")

    if try_pdflatex(out_dir / "v13l1_report.tex", out_dir, "v13l1_report.pdf"):
        print(f"Wrote {out_dir / 'v13l1_report.pdf'}")
    else:
        print("PDF skipped (pdflatex missing or failed).")

    print(f"Wrote {out_dir / 'v13l1_results.json'}")
    print(f"Wrote {out_dir / 'v13l1_convergence.csv'} ({len(conv_rows)} rows)")
    print(f"Wrote {out_dir / 'h_star_stabilized.npy'}")
    print(f"Success all_key_criteria: {success.get('all_key_criteria')}")


if __name__ == "__main__":
    main()
