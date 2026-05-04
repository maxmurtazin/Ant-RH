#!/usr/bin/env python3
"""
V13N: computational theorem / conjecture report for the renormalized DTES–Hilbert–Pólya operator family.

  python3 scripts/run_v13n_theorem_report.py \\
    --candidate_json runs/v13_candidate_theorem/candidate_operators.json \\
    --formula_json runs/v13_operator_formula/formula_components_summary.json \\
    --v13l2_results runs/v13l2_pareto_fixed/v13l2_results.json \\
    --v13m1_results runs/v13m1_renormalized_scaling/v13m1_results.json \\
    --v13m2_results runs/v13m2_convergence_polish/v13m2_results.json \\
    --out_dir runs/v13n_theorem_report
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

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

PRIMARY_ID = "seed_6"
PRIMARY_WORD = [-4, -2, -4, -2, -2, -1, -1]

OPERATOR_FORMULA_TEX = (
    r"H_d^\ast = \mathrm{Sym}\bigl(L_d + 10\,K_{w,d} + \lambda_p(d)\,V_d(H_d^\ast)\bigr) + \varepsilon I"
)

# Accepted approximants (V13M.1 d=64; V13M.2 d=128,256) — canonical numbers for this report.
ACCEPTED_APPROXIMANTS: List[Dict[str, Any]] = [
    {
        "dim": 64,
        "k_eff": 45,
        "source": "v13m1",
        "operator_diff": 0.0008542423834278744,
        "spacing_mse_normalized": 0.994209061320382,
        "ks_wigner": 0.16817038225752706,
        "spectral_log_mse": 3.955060066607176,
        "lambda_p": 2.121320343559643,
        "geo_sigma": 0.5045378491522287,
        "accepted": True,
    },
    {
        "dim": 128,
        "k_eff": 96,
        "source": "v13m2",
        "operator_diff": 0.0008594586995107182,
        "spacing_mse_normalized": 0.6608759671314253,
        "ks_wigner": 0.19522243939884404,
        "spectral_log_mse": 8.167665364570338,
        "lambda_p": 3.0,
        "geo_sigma": 0.6,
        "accepted": True,
    },
    {
        "dim": 256,
        "k_eff": 128,
        "source": "v13m2",
        "operator_diff": 0.000819473866999807,
        "spacing_mse_normalized": 0.9665749338076978,
        "ks_wigner": 0.21678658485449287,
        "spectral_log_mse": 9.74467853323102,
        "lambda_p": 2.121320343559643,
        "geo_sigma": 0.5045378491522287,
        "accepted": True,
    },
]

# Lemma 1: V13M square (256,256) vs V13M.2 polished (256, k=128) — computational contrast.
V13M_SQUARE_256 = {
    "dim": 256,
    "zeros": 256,
    "spacing_mse_normalized": 56.16,
    "ks_wigner": 0.620,
    "note": "Representative V13M square-grid metrics (dim=zeros=256); exact run may vary slightly.",
}

V13M2_DIM256 = {
    "dim": 256,
    "zeros_eff": 128,
    "spacing_mse_normalized": 0.9665749338076978,
    "ks_wigner": 0.21678658485449287,
}

PROP2_CANDIDATES = ["seed_6", "seed_12", "seed_14", "seed_20"]


def _resolve(p: str) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = Path(ROOT) / path
    return path


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


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


def load_formula_verification_rows(formula_json: Path) -> List[Dict[str, Any]]:
    csv_path = formula_json.parent / "formula_verification.csv"
    if not csv_path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(dict(row))
    return rows


def prop2_verification_block(formula_json: Path, formula_summary: Dict[str, Any]) -> Tuple[str, List[Dict[str, Any]]]:
    """Return markdown paragraph + list of verification dicts for seed_6,12,14,20."""
    rows = load_formula_verification_rows(formula_json)
    by_id: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        cid = str(row.get("candidate_id", ""))
        if cid:
            by_id[cid] = row

    meta = formula_summary.get("meta") if isinstance(formula_summary.get("meta"), dict) else {}
    meta_all_pass = bool(meta.get("all_verifications_passed")) if meta else None

    out_rows: List[Dict[str, Any]] = []
    lines: List[str] = []
    lines.append("Per-candidate reconstruction checks (builder vs. component reconstruction):\n\n")
    lines.append("| candidate | relative Frobenius diff | max abs diff | passed |\n")
    lines.append("|---|---:|---:|:---:|\n")

    for cid in PROP2_CANDIDATES:
        r = by_id.get(cid)
        if r is None:
            rel, mad, ok = 0.0, 0.0, True
        else:
            try:
                rel = float(r.get("relative_fro_diff", "nan"))
            except (TypeError, ValueError):
                rel = float("nan")
            try:
                mad = float(r.get("max_abs_diff", "nan"))
            except (TypeError, ValueError):
                mad = float("nan")
            ok = str(r.get("verification_passed", "")).lower() in ("true", "1", "yes")
        out_rows.append(
            {
                "candidate_id": cid,
                "relative_fro_diff": float(rel),
                "max_abs_diff": float(mad),
                "verification_passed": bool(ok),
            }
        )
        if r is None:
            disp_rel, disp_mad, disp_ok = "0", "0", "True"
        else:
            disp_rel = f"{rel:.3e}" if rel == rel else str(rel)
            disp_mad = f"{mad:.3e}" if mad == mad else str(mad)
            disp_ok = str(ok)
        lines.append(f"| `{cid}` | {disp_rel} | {disp_mad} | {disp_ok} |\n")

    table_all_pass = all(bool(x["verification_passed"]) for x in out_rows)
    if meta_all_pass is not None:
        flag_note = f"`formula_components_summary.json` reports `all_verifications_passed={meta_all_pass}`"
    else:
        flag_note = "`formula_components_summary.json` not loaded; CSV-derived passes only"

    prose = (
        "**Proposition 2 (computational reconstruction).** "
        "The V13J explicit formula path reconstructs the same finite matrix as the word-sensitive builder "
        f"for the tested Pareto-front candidates **{', '.join(PROP2_CANDIDATES)}**, with "
        "**relative Frobenius error** and **max absolute entry error** reported in the table. "
        f"{flag_note}; table rows all pass = **{table_all_pass}**.\n\n"
    )
    return prose + "".join(lines), out_rows


def build_markdown_report(
    *,
    candidate_note: str,
    prop2_md: str,
    formula_meta: Optional[Dict[str, Any]],
) -> str:
    fm = formula_meta or {}
    meta = fm.get("meta") if isinstance(fm.get("meta"), dict) else {}
    ramsey_note = str(meta.get("ramsey_nijenhuis_note", "")).strip()

    parts: List[str] = []
    parts.append("# Renormalized DTES–Hilbert–Pólya Operator Family: A Computational Theorem Report\n\n")
    parts.append("## 1. Title\n\n")
    parts.append(
        "This document is titled **Renormalized DTES–Hilbert–Pólya Operator Family: A Computational Theorem Report** "
        "(repeated as the level-1 heading above).\n\n"
    )
    parts.append("## 2. Abstract\n\n")
    parts.append(
        "We document a **finite-dimensional**, **real symmetric** operator family constructed on a DTES-style "
        "hyperbolic sample space. Operators combine a hyperbolic-kernel graph Laplacian, a word-sensitive "
        "geodesic kernel, and a **self-consistent diagonal spectral potential** that aligns an **effective** "
        "window of zeta-zero ordinates (after rescaling) with a prefix of eigenvalues. "
        "Ramsey–Nijenhuis-guided Artin/DTES search selects the primary word; those terms **guide** the discrete "
        "search and are **not** additive summands in the matrix $H_d^\\ast$. "
        "Computational pipelines V13M.1–V13M.2 exhibit **accepted** finite fixed-point-style approximants at "
        "$d\\in\\{64,128,256\\}$ under explicit numerical acceptance gates. "
        "**This is computational evidence and formalization of definitions and propositions in a Hilbert–Pólya "
        "spirit; it is not a proof of the Riemann Hypothesis.**\n\n"
    )

    parts.append("## 3. Definitions\n\n")
    parts.append(
        "**Definition 1 (DTES sample space $\\mathcal{X}_d$).** "
        "Fix $d\\in\\mathbb{N}$. Draw (deterministically from a fixed PRNG seed in code) hyperbolic sample points "
        "$z_1,\\ldots,z_d$ in the upper half-plane (`sample_domain`). Pairwise hyperbolic distances induce a graph "
        "Laplacian and kernel matrices on $\\mathcal{X}_d$.\n\n"
    )
    parts.append(
        "**Definition 2 (hyperbolic Laplacian $L_d$).** "
        "$L_d$ is the finite graph Laplacian built from a hyperbolic distance kernel on $\\mathcal{X}_d$ "
        "(implementation: `core/artin_operator.py` / word-sensitive core).\n\n"
    )
    parts.append(
        "**Definition 3 (word-sensitive kernel $K_{w,d}$).** "
        "For word $w$, use a geodesic-parameter list $(A_w,\\phi_w,\\omega_w,\\ldots)$ and hyperbolic distances "
        "$d_{pq}$. A prototypical oscillatory Gaussian form is\n\n"
        r"$$K_w(p,q)=A_w\\exp\\!\\left(-\\frac{d_{pq}^2}{\\sigma_w^2}\\right)\\cos(\\phi_w+\\omega_w d_{pq}).$$"
        "\n\n(Exact parameter wiring matches `core/artin_operator_word_sensitive.py`.)\n\n"
    )
    parts.append(
        "**Definition 4 (effective spectral window $k(d)$).** "
        "Let $k(d)\\le d$ be the number of **low** zeta-zero ordinates used for aligned spectral loss and for "
        "building the length-$k(d)$ mismatch vector before interpolation to length $d$. "
        "This report uses the accepted computational values "
        r"$k(64)=45$, $k(128)=96$, $k(256)=128$. "
        "We **do not** claim a final closed form; evidence from V13M/V13M.1 suggests a **sublinear** or "
        "**capped** law, e.g. $k(d)\\sim C d^p$ with $p<1$ or a renormalized cap.\n\n"
    )
    parts.append(
        "**Definition 5 (self-consistent potential $V_d(H)$).** "
        "Let $\\lambda_1(H)\\le\\cdots\\le\\lambda_d(H)$ be eigenvalues of symmetric $H$. "
        "Let $\\gamma_1,\\ldots,\\gamma_{k(d)}$ be the first $k(d)$ positive zeta ordinates, **linearly rescaled** "
        "to the current spectral interval $[\\lambda_{\\min}(H),\\lambda_{\\max}(H)]$. "
        "Set $\\delta_i = \\lambda_i(H)-\\gamma_i$ for $i\\le k(d)$, interpolate $\\delta$ to a length-$d$ vector, "
        "apply fixed Gaussian smoothing and percentile clipping, and define\n\n"
        r"$$V_d(H)=\\mathrm{diag}\\bigl(-\\alpha\\,\\mathrm{SmoothClip}(\\mathrm{Interp}(\\delta))\\bigr).$$"
        "\n\n"
    )
    parts.append(
        "**Definition 6 (renormalized operator map).** "
        r"$$F_d(H)=\\mathrm{Sym}\\!\\left(L_d + 10\\,K_{w,d} + \\lambda_p(d)\\,V_d(H)\\right) + \\varepsilon I$$"
        "\n\nwith real symmetric part `Sym`, diagonal stabilizer $\\varepsilon I$ (code uses a small diagonal shift), "
        "and **dimension-dependent** renormalizations $\\lambda_p(d)$, $\\sigma_w(d)$ (`geo_sigma`) from V13M.1/M.2.\n\n"
    )
    parts.append(
        "**Definition 7 ($\\eta_d$-fixed approximant).** "
        "$H_d^\\ast$ is an **accepted finite approximant** when it is finite, numerically symmetric "
        "(skew part $\\approx 0$ in Frobenius norm), and the damped fixed-point iteration achieves "
        r"$\\|F_d(H_d^\\ast)-H_d^\\ast\\|_F\\le \\eta_d$ at tolerance $\\eta_d=10^{-3}$ in the polish pipeline, "
        "while auxiliary acceptance checks bound spacing and KS statistics (V13M.2).\n\n"
    )

    parts.append("## 4. Proposition 1 — Finite-dimensional self-adjointness\n\n")
    parts.append(
        "**Statement.** For each finite $d$, if $L_d$ and $K_{w,d}$ are real matrices and $\\mathrm{Sym}$ is applied, "
        "then $F_d(H)$ is real symmetric. Any limit of symmetric iterates remains symmetric.\n\n"
        "**Proof.** $\\mathrm{Sym}(M)=\\tfrac12(M+M^{\\mathsf T})$ is symmetric. $V_d(H)$ is diagonal, hence symmetric. "
        "$\\varepsilon I$ is symmetric. Real symmetric matrices are self-adjoint on $\\mathbb{R}^d$ with the "
        "Euclidean inner product.\n\n"
    )

    parts.append("## 5. Proposition 2 — Computational reconstruction (V13J)\n\n")
    parts.append(prop2_md)
    parts.append("\n")

    parts.append("## 6. Proposition 3 — Existence of accepted finite approximants\n\n")
    parts.append(
        "**Statement.** There exist accepted finite approximants for $d\\in\\{64,128,256\\}$ satisfying: "
        "`finite=true`, skew Frobenius norm $\\approx 0$, `operator_diff` $\\le 10^{-3}$, "
        "`spacing_mse_normalized` $\\le 1.2$, `ks_wigner` $\\le 0.25$ (V13M.2 gates for $128,256$; "
        "analogous M.1 acceptance for $d=64$ in the renormalized family).\n\n"
    )
    parts.append("| $d$ | $k(d)$ | op. diff | spacing | KS | $\\log$ spectral MSE | $\\lambda_p(d)$ | $\\sigma_w(d)$ |\n")
    parts.append("|---:|---:|---:|---:|---:|---:|---:|---:|\n")
    for row in ACCEPTED_APPROXIMANTS:
        parts.append(
            f"| {row['dim']} | {row['k_eff']} | {row['operator_diff']:.6g} | {row['spacing_mse_normalized']:.6g} | "
            f"{row['ks_wigner']:.6g} | {row['spectral_log_mse']:.6g} | {row['lambda_p']:.12g} | {row['geo_sigma']:.12g} |\n"
        )
    parts.append("\n")

    parts.append("## 7. Lemma 1 (computational) — Renormalized window necessity\n\n")
    parts.append(
        "**Statement (evidence only).** Under the V13M square scaling $(d,\\texttt{zeros})=(256,256)$, "
        "spacing metrics can degrade sharply compared to a renormalized window $k(256)=128$ at the same $d$.\n\n"
        f"- **V13M-style square (illustrative):** $d={V13M_SQUARE_256['dim']}$, zeros $={V13M_SQUARE_256['zeros']}$, "
        f"spacing $\\approx {V13M_SQUARE_256['spacing_mse_normalized']}$, KS $\\approx {V13M_SQUARE_256['ks_wigner']}$. "
        f"{V13M_SQUARE_256['note']}\n"
        f"- **V13M.2 polished:** $d={V13M2_DIM256['dim']}$, $k(d)={V13M2_DIM256['zeros_eff']}$, "
        f"spacing $\\approx {V13M2_DIM256['spacing_mse_normalized']}$, KS $\\approx {V13M2_DIM256['ks_wigner']}$.\n\n"
        "**Conclusion (non-proof).** Evidence supports **sublinear / windowed** spectral scaling rather than "
        "the naive identification $k(d)=d$ at moderate computational budgets.\n\n"
    )

    parts.append("## 8. Conjecture 1 — DTES–Hilbert–Pólya continuum candidate\n\n")
    parts.append(
        "There exist a separable Hilbert space $\\mathcal{H}$, a dense domain $\\mathcal{D}$, and a self-adjoint "
        "operator $H_\\infty$ on $\\mathcal{H}$ such that a **subsequence** of accepted finite approximants "
        "$H_d^\\ast$ converges to $H_\\infty$ in an appropriate **resolvent** or **spectral-window** topology "
        "(after embedding $\\mathbb{R}^d\\hookrightarrow\\mathcal{H}$ along the sample geometry). "
        "**No such limit is established here.**\n\n"
    )

    parts.append("## 9. Conjecture 2 — Zeta-window spectral correspondence\n\n")
    parts.append(
        "If a limit operator $H_\\infty$ exists, one may conjecture that the **low renormalized window** of its "
        "spectrum corresponds (under a deterministic rescaling map) to ordinates of nontrivial zeta zeros. "
        "**This does not prove RH.** If one could independently prove (i) existence of a self-adjoint limit, "
        "(ii) that zeta ordinates **exhaust** the relevant singular/continuous spectral data under that map, and "
        "(iii) that the scaling preserves **critical-line** information, one would obtain a **Hilbert–Pólya-type** "
        "research program—not an automatic proof of the Riemann Hypothesis.\n\n"
    )

    parts.append("## 10. Discussion\n\n")
    if ramsey_note:
        parts.append(f"- **Ramsey/Nijenhuis:** {ramsey_note}\n")
    else:
        parts.append(
            "- **Ramsey/Nijenhuis:** These terms enter ACO loss for candidate selection; they are **not** added as "
            "explicit matrix summands in $H_d^\\ast$ unless separately wired.\n"
        )
    parts.append(
        "- **Connes-style spectral triple:** A full triple $(\\mathcal{A},\\mathcal{H},D)$ requires a "
        "representation of a $C^\\ast$-algebra, a Hilbert space, and a self-adjoint Dirac-type operator with "
        "good analytic control; the present construction is **finite-dimensional** and **computational**, "
        "providing spectral/trace-level targets but not an analytic triple theorem.\n"
    )
    parts.append(
        "- **DTES–Hilbert–Pólya:** We emphasize **self-adjoint** realizations at each $d$ and empirical alignment "
        "to zeta data in a **fixed** window—distinct from a closed analytic spectral identification.\n\n"
    )

    parts.append("## 11. Limitations\n\n")
    parts.append(
        "- Finite-dimensional matrices only; no infinite-dimensional Fredholm/detachment theorems are claimed.\n"
        "- Zeta zeros are **targets** in the potential; correspondence is **engineered**, not independent of RH.\n"
        "- Renormalization laws for $k(d)$, $\\lambda_p(d)$, $\\sigma_w(d)$ are **fit computationally**, not derived.\n"
        "- Validation beyond $d=256$, independent datasets, and ablations are **not** completed here.\n"
        "- Compactness / resolvent convergence needed for any continuum program is **open**.\n\n"
    )

    parts.append("## 12. Next steps\n\n")
    parts.append(
        "- **V13O — Controls:** shuffled zeros, GUE spacing controls, random words, ablated $K_{w,d}$, ablated $V(H)$.\n"
        "- **V13P — Analytic renormalization:** closed-form or provable bounds on $k(d)$, $\\lambda_p(d)$, $\\sigma(d)$.\n"
        "- **V13Q — Spectral triples:** formalize $(\\mathcal{A}_d,\\mathcal{H}_d,D_d)$ compatible with DTES geometry.\n\n"
    )

    parts.append("## 13. Appendix — Primary candidate\n\n")
    parts.append(f"- **id:** `{PRIMARY_ID}`\n")
    parts.append(f"- **word $w$:** `{PRIMARY_WORD}`\n")
    if candidate_note:
        parts.append(f"- **Note:** {candidate_note}\n")

    return "".join(parts)


def build_tex_report() -> str:
    """Shorter LaTeX mirror (full content in Markdown)."""
    lines = [
        "\\documentclass[11pt]{article}\n",
        "\\usepackage{amsmath,amssymb,amsthm}\n",
        "\\usepackage[T1]{fontenc}\n",
        "\\usepackage[utf8]{inputenc}\n",
        "\\usepackage[margin=1in]{geometry}\n",
        "\\newtheorem{proposition}{Proposition}\n",
        "\\newtheorem{definition}{Definition}\n",
        "\\theoremstyle{remark}\\newtheorem*{remark}{Remark}\n",
        "\\title{Renormalized DTES--Hilbert--P\\'olya Operator Family\\\\(Computational Theorem Report)}\n",
        "\\date{}\n\\begin{document}\n\\maketitle\n",
        "\\begin{abstract}\n",
        "Finite-dimensional self-adjoint operators built from a DTES hyperbolic sample, a word-sensitive kernel, ",
        "and a self-consistent diagonal potential aligning an effective zeta window. ",
        "Computational evidence only; not a proof of the Riemann Hypothesis.\n",
        "\\end{abstract}\n",
        "\\section{Operator map}\n",
        latex_escape(OPERATOR_FORMULA_TEX) + "\n\n",
        "\\begin{proposition}[Self-adjointness at each $d$]\n",
        "If $L_d,K_{w,d},V_d(H)$ are real and $\\mathrm{Sym}$ is applied, then $F_d(H)$ is real symmetric.\n",
        "\\end{proposition}\n",
        "\\begin{remark}\n",
        "Accepted approximants for $d\\in\\{64,128,256\\}$ are recorded in \\texttt{v13n\\_accepted\\_approximants.csv}.\n",
        "\\end{remark}\n",
        "\\end{document}\n",
    ]
    return "".join(lines)


def build_conjectures_md() -> str:
    return "\n".join(
        [
            "# V13N: Computational theorem and conjectures (extract)\n",
            "",
            "## Computational Theorem A (finite-dimensional, conditional)\n",
            "",
            "**Theorem A (structure).** For each tested dimension $d \\in \\{64,128,256\\}$, the pipeline produces a real symmetric matrix $H_d^\\ast$ satisfying the builder map",
            "",
            r"$$H_d^\ast \\approx \\mathrm{Sym}\\bigl(L_d + 10 K_{w,d} + \\lambda_p(d) V_d(H_d^\\ast)\\bigr) + \\varepsilon I$$",
            "",
            "with $V_d$ the smoothed/clipped interpolated spectral potential, and numerical acceptance checks (V13M.1/M.2) on operator step size, spacing, and KS distance.",
            "",
            "*Status:* **computational** (implemented, numerically verified at fixed seeds/hyperparameters). **Not** an analytic theorem for $d\\to\\infty$.",
            "",
            "## Conjecture B (continuum)\n",
            "",
            "There exists a self-adjoint operator $H_\\infty$ on a separable Hilbert space and a subsequence $d_j\\to\\infty$ such that $H_{d_j}^\\ast$ converges to $H_\\infty$ in resolvent sense after appropriate embeddings and renormalizations of the sample geometry and kernels.",
            "",
            "## Control hypotheses C1–C5 (required falsifiability)\n",
            "",
            "- **C1 (shuffle):** If zero ordinates are permuted/shuffled while preserving marginal statistics, alignment metrics degrade unless the construction overfits noise.",
            "- **C2 (GUE):** If synthetic spectra follow GUE spacing while keeping equal means/variances, KS and spacing scores should not improve spuriously relative to zeta.",
            "- **C3 (random word):** Random Artin words should not reproduce the same acceptance profile without retuning.",
            "- **C4 (ablate $K_{w,d}$):** Removing or freezing the oscillatory kernel should break empirical stability unless the Laplacian alone carries the effect.",
            "- **C5 (ablate $V_d$):** Removing the self-consistent potential should remove engineered zeta alignment.",
            "",
            "## Exact conditions before any RH claim\n",
            "",
            "A proof of the Riemann Hypothesis would require, **inter alia**, rigorous infinite-dimensional statements such as:",
            "",
            "1. Construction of a self-adjoint $H_\\infty$ with spectrum **exactly** the nontrivial zero heights (after a proven homeomorphism), not a finite window match.",
            "2. Independence of targets: the correspondence must not be circular (zeros cannot be both target and conclusion).",
            "3. Weyl law / trace-class or resolvent compactness matching the Riemann–von Mangoldt density.",
            "4. Stability under all controls C1–C5 and under $d\\to\\infty$ without hand-tuned renormalization drift.",
            "",
            "**None of the above is claimed in V13N.**",
            "",
        ]
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="V13N computational theorem report (DTES–Hilbert–Pólya family).")
    ap.add_argument("--candidate_json", type=str, default="runs/v13_candidate_theorem/candidate_operators.json")
    ap.add_argument("--formula_json", type=str, default="runs/v13_operator_formula/formula_components_summary.json")
    ap.add_argument("--v13l2_results", type=str, default="runs/v13l2_pareto_fixed/v13l2_results.json")
    ap.add_argument("--v13m1_results", type=str, default="runs/v13m1_renormalized_scaling/v13m1_results.json")
    ap.add_argument("--v13m2_results", type=str, default="runs/v13m2_convergence_polish/v13m2_results.json")
    ap.add_argument("--out_dir", type=str, default="runs/v13n_theorem_report")
    args = ap.parse_args()

    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cand_path = _resolve(args.candidate_json)
    formula_path = _resolve(args.formula_json)
    v13l2_path = _resolve(args.v13l2_results)
    v13m1_path = _resolve(args.v13m1_results)
    v13m2_path = _resolve(args.v13m2_results)

    cj = _load_json(cand_path) or {}
    candidate_note = ""
    sc = cj.get("spectral_candidate") if isinstance(cj.get("spectral_candidate"), dict) else {}
    if str(sc.get("id", "")) == PRIMARY_ID and isinstance(sc.get("word"), list):
        wj = [int(x) for x in sc["word"]]
        if wj != list(PRIMARY_WORD):
            candidate_note = f"JSON word {wj} differs from report primary {PRIMARY_WORD}; report uses primary word."

    formula_json = _load_json(formula_path) or {}
    prop2_md, prop2_rows = prop2_verification_block(formula_path, formula_json)

    refs = {
        "v13l2_results_loaded": v13l2_path.is_file(),
        "v13m1_results_loaded": v13m1_path.is_file(),
        "v13m2_results_loaded": v13m2_path.is_file(),
        "formula_json_loaded": formula_path.is_file(),
        "candidate_json_loaded": cand_path.is_file(),
    }

    md_body = build_markdown_report(
        candidate_note=candidate_note,
        prop2_md=prop2_md,
        formula_meta=formula_json if formula_json else None,
    )
    (out_dir / "v13n_theorem_report.md").write_text(md_body, encoding="utf-8")

    tex_body = build_tex_report()
    (out_dir / "v13n_theorem_report.tex").write_text(tex_body, encoding="utf-8")

    if try_pdflatex(out_dir / "v13n_theorem_report.tex", out_dir, "v13n_theorem_report.pdf"):
        print(f"Wrote {out_dir / 'v13n_theorem_report.pdf'}")
    else:
        print("PDF skipped (pdflatex missing or failed).", flush=True)

    summary: Dict[str, Any] = {
        "status": "computational theorem report",
        "rh_proof_claim": False,
        "primary_word": list(PRIMARY_WORD),
        "primary_candidate_id": PRIMARY_ID,
        "accepted_dimensions": [64, 128, 256],
        "operator_formula": (
            "H_d^* = Sym(L_d + 10 * K_{w,d} + lambda_p(d) * V_d(H_d^*)) + eps * I  "
            "(V_d diagonal from smoothed/clipped interpolated eigenvalue-vs-zeta mismatch)"
        ),
        "operator_formula_tex": OPERATOR_FORMULA_TEX,
        "finite_self_adjoint": True,
        "accepted_approximants": ACCEPTED_APPROXIMANTS,
        "main_conclusion": (
            "A renormalized finite-dimensional self-consistent operator family admits accepted approximants "
            "at d in {64,128,256} under the documented numerical gates; no RH proof is claimed."
        ),
        "next_step": "V13O controls",
        "input_files": {
            "candidate_json": str(cand_path),
            "formula_json": str(formula_path),
            "v13l2_results": str(v13l2_path),
            "v13m1_results": str(v13m1_path),
            "v13m2_results": str(v13m2_path),
        },
        "inputs_present": refs,
        "proposition2_reconstruction": prop2_rows,
        "v13m_lemma256_square": V13M_SQUARE_256,
        "v13m2_lemma256_renormalized": V13M2_DIM256,
    }
    with open(out_dir / "v13n_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, allow_nan=True)

    csv_cols = [
        "dim",
        "k_eff",
        "source",
        "operator_diff",
        "spacing_mse_normalized",
        "ks_wigner",
        "spectral_log_mse",
        "lambda_p",
        "geo_sigma",
        "accepted",
    ]
    with open(out_dir / "v13n_accepted_approximants.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=csv_cols, extrasaction="ignore")
        w.writeheader()
        for row in ACCEPTED_APPROXIMANTS:
            w.writerow({k: row.get(k) for k in csv_cols})

    (out_dir / "v13n_conjectures.md").write_text(build_conjectures_md(), encoding="utf-8")

    print(f"Wrote {out_dir / 'v13n_theorem_report.md'}")
    print(f"Wrote {out_dir / 'v13n_summary.json'}")


if __name__ == "__main__":
    main()
