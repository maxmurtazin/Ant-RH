#!/usr/bin/env python3
"""
V13J: explicit code-equivalent formula for word-sensitive H_w and numerical verification.

  python3 scripts/make_v13_operator_formula.py \\
    --candidate_json runs/v13_candidate_theorem/candidate_operators.json \\
    --out_dir runs/v13_operator_formula \\
    --dim 128 --zeros 128 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
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

from core.artin_operator import sample_domain  # noqa: E402
from core.artin_operator_word_sensitive import (  # noqa: E402
    build_word_sensitive_components,
    build_word_sensitive_operator,
)


def latex_escape(s: str) -> str:
    t = str(s)
    t = t.replace("\\", "\\textbackslash{}")
    t = t.replace("{", "\\{").replace("}", "\\}")
    t = t.replace("_", "\\_")
    t = t.replace("%", "\\%")
    t = t.replace("&", "\\&")
    t = t.replace("$", "\\$")
    return t


def _find_pdflatex() -> Optional[str]:
    w = shutil.which("pdflatex")
    if w:
        return w
    for p in ("/Library/TeX/texbin/pdflatex", "/usr/local/texlive/2026/bin/universal-darwin/pdflatex"):
        if Path(p).is_file():
            return p
    return None


def try_pdflatex(tex_path: Path, out_dir: Path) -> bool:
    exe = _find_pdflatex()  # PATH or common TeX Live macOS paths
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
        return r.returncode == 0 and (out_dir / "v13_operator_formula.pdf").is_file()
    except (OSError, subprocess.TimeoutExpired):
        return False


def geodesic_entry(word: List[int]) -> Dict[str, Any]:
    w = [int(x) for x in word]
    return {"a_list": w, "length": float(max(1.0, len(w))), "is_hyperbolic": True, "primitive": True}


def load_candidates(
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

    want = {"seed_14", "seed_20"}
    if pareto_path and pareto_path.is_file():
        with open(pareto_path, "r", encoding="utf-8") as f:
            pj = json.load(f)
        for row in pj.get("candidates") or []:
            if not isinstance(row, dict):
                continue
            rid = str(row.get("id", ""))
            w = row.get("word_used") or row.get("word")
            if rid in want and isinstance(w, list):
                add(rid, [int(x) for x in w])
    return out


def verify_one(
    cand: Dict[str, Any],
    *,
    dim: int,
    eps: float,
    geo_weight: float,
    geo_sigma: float,
    potential_weight: float,
    seed: int,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    Z = sample_domain(int(dim), seed=int(seed))
    g = [geodesic_entry(cand["word"])]
    H_a, rep = build_word_sensitive_operator(
        z_points=Z,
        distances=None,
        geodesics=g,
        eps=float(eps),
        geo_sigma=float(geo_sigma),
        geo_weight=float(geo_weight),
        potential_weight=float(potential_weight),
    )
    comp = build_word_sensitive_components(
        z_points=Z,
        distances=None,
        geodesics=g,
        eps=float(eps),
        geo_sigma=float(geo_sigma),
        geo_weight=float(geo_weight),
        potential_weight=float(potential_weight),
    )
    H_b = comp["H_final"]
    diff = np.asarray(H_a, dtype=np.float64) - np.asarray(H_b, dtype=np.float64)
    fa = float(np.linalg.norm(H_a, ord="fro"))
    fd = float(np.linalg.norm(diff, ord="fro"))
    mad = float(np.max(np.abs(diff)))
    rel = fd / max(1e-12, fa)
    ver = {
        "candidate_id": str(cand["id"]),
        "word": list(cand["word"]),
        "operator_fro_norm": fa,
        "reconstructed_fro_norm": float(np.linalg.norm(H_b, ord="fro")),
        "fro_norm_diff": fd,
        "relative_fro_diff": rel,
        "max_abs_diff": mad,
        "verification_passed": bool(rel < 1e-10 and mad < 1e-9),
    }
    norms_row = {
        "candidate_id": str(cand["id"]),
        "fro_H0": comp["component_norms"]["fro_H0"],
        "fro_B": comp["component_norms"]["fro_B"],
        "fro_G": comp["component_norms"]["fro_G_weighted"],
        "fro_V": comp["component_norms"]["fro_V_weighted"],
        "fro_H_tilde": comp["component_norms"]["fro_H_tilde"],
        "fro_H_sym": comp["component_norms"]["fro_H_sym"],
    }
    summary_comp = {
        "candidate_id": str(cand["id"]),
        "component_norms": comp["component_norms"],
        "formula_text": comp["formula_text"],
        "latex_formula": comp["latex_formula"],
        "rep": rep,
    }
    return ver, norms_row, summary_comp


def main() -> None:
    ap = argparse.ArgumentParser(description="V13J explicit word-sensitive operator formula + verification.")
    ap.add_argument("--candidate_json", type=str, default="runs/v13_candidate_theorem/candidate_operators.json")
    ap.add_argument("--pareto_json", type=str, default="runs/v13_top5_pareto/pareto_results.json")
    ap.add_argument("--out_dir", type=str, default="runs/v13_operator_formula")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--zeros", type=int, default=128, help="Documented zeta window size (operator uses --dim sample points).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eps", type=float, default=0.6)
    ap.add_argument("--geo_weight", type=float, default=10.0)
    ap.add_argument("--geo_sigma", type=float, default=0.6)
    ap.add_argument("--potential_weight", type=float, default=0.25)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = Path(ROOT) / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cand_path = Path(args.candidate_json)
    if not cand_path.is_absolute():
        cand_path = Path(ROOT) / cand_path
    pareto_path = Path(args.pareto_json)
    if not pareto_path.is_absolute():
        pareto_path = Path(ROOT) / pareto_path

    if not cand_path.is_file():
        raise FileNotFoundError(cand_path)

    cands = load_candidates(cand_path, pareto_path if pareto_path.is_file() else None)
    if not cands:
        raise SystemExit("no candidates loaded")

    ver_rows: List[Dict[str, Any]] = []
    norm_rows: List[Dict[str, Any]] = []
    summaries: List[Dict[str, Any]] = []
    formula_once: Dict[str, str] = {}

    for c in cands:
        try:
            ver, nr, sc = verify_one(
                c,
                dim=int(args.dim),
                eps=float(args.eps),
                geo_weight=float(args.geo_weight),
                geo_sigma=float(args.geo_sigma),
                potential_weight=float(args.potential_weight),
                seed=int(args.seed),
            )
            ver_rows.append(ver)
            norm_rows.append(nr)
            summaries.append(
                {
                    "candidate_id": sc["candidate_id"],
                    "component_norms": sc["component_norms"],
                    "rep_operator_fro": sc["rep"].get("operator_fro_norm"),
                }
            )
            if not formula_once:
                formula_once["formula_text"] = sc["formula_text"]
                formula_once["latex_formula"] = sc["latex_formula"]
        except Exception as ex:
            ver_rows.append(
                {
                    "candidate_id": str(c.get("id")),
                    "word": c.get("word"),
                    "error": repr(ex),
                    "verification_passed": False,
                }
            )

    all_pass = all(bool(v.get("verification_passed")) for v in ver_rows if "verification_passed" in v)

    with open(out_dir / "formula_components_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": {
                    "dim": int(args.dim),
                    "zeros_n_doc": int(args.zeros),
                    "seed": int(args.seed),
                    "eps": float(args.eps),
                    "geo_weight": float(args.geo_weight),
                    "geo_sigma": float(args.geo_sigma),
                    "potential_weight": float(args.potential_weight),
                    "builder_module": "core.artin_operator_word_sensitive",
                    "aco_path_note": "ArtinACO uses build_word_sensitive_operator from this module when operator_builder=word_sensitive (see core/artin_aco.py).",
                    "ramsey_nijenhuis_note": "Ramsey and Nijenhuis terms guide selection through the ACO loss but are not additive matrix terms in H_w.",
                    "ncg_note": "The NCG commutator / DTES-triple regularizer affects loss when enabled; it does not add a separate summand to H_w unless future code wires that in.",
                    "all_verifications_passed": all_pass,
                },
                "formula_text": formula_once.get("formula_text", ""),
                "latex_formula": formula_once.get("latex_formula", ""),
                "per_candidate": summaries,
            },
            f,
            indent=2,
            allow_nan=True,
        )

    with open(out_dir / "formula_verification.csv", "w", encoding="utf-8", newline="") as f:
        cols = [
            "candidate_id",
            "word",
            "operator_fro_norm",
            "reconstructed_fro_norm",
            "fro_norm_diff",
            "relative_fro_diff",
            "max_abs_diff",
            "verification_passed",
        ]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in ver_rows:
            row = dict(r)
            if isinstance(row.get("word"), list):
                row["word"] = json.dumps(row["word"])
            w.writerow({k: row.get(k) for k in cols})

    with open(out_dir / "candidate_component_norms.csv", "w", encoding="utf-8", newline="") as f:
        cols = ["candidate_id", "fro_H0", "fro_B", "fro_G", "fro_V", "fro_H_tilde", "fro_H_sym"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for nr in norm_rows:
            w.writerow({k: nr.get(k) for k in cols})

    # --- Markdown ---
    w6 = [-4, -2, -4, -2, -2, -1, -1]
    w12 = [-6, -1, -4, 1, 4, 1, 5]
    md: List[str] = [
        "# V13J Explicit Operator Formula\n\n",
        "## A. Warning\n\n",
        "> This is an experimental computational formula extraction, not a proof of RH.\n\n",
        "## B. Word-sensitive operator\n\n",
        r"Fix a word $w=(a_1,\ldots,a_m)$ with $a_i\in\mathbb{Z}$. Sample $n=\texttt{dim}$ points $z_p$ in the upper half-plane (`sample_domain`). "
        "Pairwise hyperbolic distances $d_{pq}$ come from `build_laplacian` in `core/artin_operator.py`.\n\n",
        "## C. Code-equivalent formula\n\n",
        "`core/artin_aco.py` selects `operator_builder=\"word_sensitive\"` and calls `build_word_sensitive_operator` from **`core/artin_operator_word_sensitive.py`** (same implementation as this extraction).\n\n",
        "```\n",
        formula_once.get("formula_text", "(run verification to populate)"),
        "\n```\n\n",
        "**Matrix assembly in code:**\n\n",
        r"$H_w = \mathrm{Sym}(\tilde H_w) + \varepsilon I$ with $\varepsilon=10^{-6}$, "
        r"$\mathrm{Sym}(M)=\frac12(M+M^{\mathsf T})$, and "
        r"$\tilde H_w = \lambda_L L + \lambda_g \widehat{K}_w + \lambda_p \widehat{V}_w$ "
        f"using defaults $\\lambda_L=1$, $\\lambda_g={args.geo_weight}$, $\\lambda_p={args.potential_weight}$ for this report run.\n\n",
        "## D. Component definitions\n\n",
        "- **H0** = `laplacian_weight * L`: graph Laplacian from the heat-kernel adjacency `exp(-d^2/eps)` on hyperbolic distances.\n",
        "- **B (braid/action)**: **no separate B matrix** appears in `build_word_sensitive_operator`; braid/word data enter only through amplitudes and phases inside **G** and **V**.\n",
        "- **G** = `geo_weight * K_norm`: sum of per-word kernels `K_w[p,q]=A_w exp(-d_{pq}^2/\\sigma_w^2) cos(...)` then **max-abs** normalized over the full kernel sum before scaling.\n",
        "- **V** = `potential_weight * V_norm`: diagonal from accumulated `A_w sin(hash_phase + Im(z_q))`, max-abs normalized like **K**.\n",
        "- **Ramsey / Nijenhuis**: **not** added to `H_w`; they are **loss regularizers** in ACO, not summands of the operator matrix.\n\n",
        "## E. Candidate formulas\n\n",
        f"- **H_spectral** (default id `seed_6`): word {json.dumps(w6)}.\n",
        f"- **H_geometric** (default id `seed_12`): word {json.dumps(w12)}.\n",
        "- Additional Pareto-front ids **seed_14**, **seed_20** are included when their words appear in `pareto_results.json`.\n\n",
        "## F. Verification table\n\n",
        "| candidate_id | word | operator_fro_norm | reconstructed_fro_norm | fro_norm_diff | relative_fro_diff | max_abs_diff | verification_passed |\n",
        "|---|---:|---:|---:|---:|---:|---:|:---:|\n",
    ]
    for r in ver_rows:
        wid = str(r.get("candidate_id", ""))
        w = r.get("word", [])
        ws = json.dumps(w) if isinstance(w, list) else str(w)
        md.append(
            f"| `{wid}` | `{ws}` | {r.get('operator_fro_norm', '')} | {r.get('reconstructed_fro_norm', '')} | "
            f"{r.get('fro_norm_diff', '')} | {r.get('relative_fro_diff', '')} | {r.get('max_abs_diff', '')} | {r.get('verification_passed', '')} |\n"
        )
    md.append("\n## G. Component norm table\n\n")
    md.append("| candidate_id | ||H0||_F | ||B||_F | ||G||_F | ||V||_F | ||H_tilde||_F | ||H_sym||_F |\n|---|---:|---:|---:|---:|---:|---:|\n")
    for nr in norm_rows:
        md.append(
            f"| `{nr.get('candidate_id')}` | {nr.get('fro_H0')} | {nr.get('fro_B')} | {nr.get('fro_G')} | {nr.get('fro_V')} | "
            f"{nr.get('fro_H_tilde')} | {nr.get('fro_H_sym')} |\n"
        )
    md.append("\n## H. Interpretation\n\n")
    md.append(
        "- **Exact match:** `relative_fro_diff` should be ~0 (floating noise only); the decomposition is **instrumentation of the same code path** (`_build_word_sensitive_core`).\n"
    )
    md.append(
        "- **Dominant term:** compare `||H0||_F`, `||G||_F`, `||V||_F` in the CSV; typically the Laplacian plus scaled geodesic kernel dominates for default weights.\n"
    )
    md.append(
        "- **seed_6 vs seed_12:** different words yield different $(A_w,\\sigma_w,\\text{phase})$ tuples, hence different **K_raw** and **V** patterns even on the same point set $Z$.\n"
    )
    md.append(f"- **Global verification:** all checks passed: **{all_pass}**.\n")

    (out_dir / "v13_operator_formula.md").write_text("".join(md), encoding="utf-8")

    tex = [
        "\\documentclass[11pt]{article}\n\\usepackage{amsmath,amssymb}\n\\usepackage[T1]{fontenc}\n\\usepackage[margin=1in]{geometry}\n",
        "\\title{V13J Explicit Word-Sensitive Operator Formula}\n\\date{}\n\\begin{document}\n\\maketitle\n",
        "\\section*{Warning}\n",
        "This is an experimental computational formula extraction, not a proof of RH.\n\n",
        "\\section{Assembly}\n",
        latex_escape(formula_once.get("latex_formula", "")) + "\n\n",
        "\\section{Verification}\nAll runs passed: \\texttt{"
        + ("true" if all_pass else "false")
        + "}. See \\texttt{formula\\_verification.csv}.\n\n\\end{document}\n",
    ]
    (out_dir / "v13_operator_formula.tex").write_text("".join(tex), encoding="utf-8")

    if try_pdflatex(out_dir / "v13_operator_formula.tex", out_dir):
        print(f"Wrote {out_dir / 'v13_operator_formula.pdf'}")
    else:
        print("PDF skipped (pdflatex missing or failed).")

    print(f"Wrote {out_dir / 'v13_operator_formula.md'}")
    print(f"Wrote {out_dir / 'v13_operator_formula.tex'}")
    print(f"Wrote {out_dir / 'formula_components_summary.json'}")
    print(f"Wrote {out_dir / 'formula_verification.csv'}")
    print(f"Wrote {out_dir / 'candidate_component_norms.csv'}")
    print(f"all_verifications_passed={all_pass}")


if __name__ == "__main__":
    main()
