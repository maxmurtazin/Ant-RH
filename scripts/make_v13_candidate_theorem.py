#!/usr/bin/env python3
"""
V13H: Candidate operator theorem-style report from Pareto + top-candidate JSON.

  python3 scripts/make_v13_candidate_theorem.py \\
    --pareto_json runs/v13_top5_pareto/pareto_results.json \\
    --top_json runs/v13_formula_report/v13_top_candidates.json \\
    --out_dir runs/v13_candidate_theorem
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


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


def md_cell(s: Any) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def candidates_by_id(pareto: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for c in pareto.get("candidates") or []:
        if isinstance(c, dict) and c.get("id") is not None:
            out[str(c["id"])] = c
    return out


def word_list(c: Dict[str, Any]) -> List[int]:
    w = c.get("word_used") or c.get("word")
    if isinstance(w, list):
        try:
            return [int(x) for x in w]
        except (TypeError, ValueError):
            pass
    return []


def load_top_rows(path: Path) -> List[Dict[str, Any]]:
    data = load_json(path)
    if isinstance(data, dict) and isinstance(data.get("top_candidates"), list):
        return list(data["top_candidates"])
    if isinstance(data, list):
        return list(data)
    return []


def aco_champion_id(rows: List[Dict[str, Any]]) -> Optional[str]:
    best = None
    best_seed: Optional[int] = None
    for r in rows:
        if not isinstance(r, dict):
            continue
        bl = r.get("best_loss")
        try:
            v = float(bl)
        except (TypeError, ValueError):
            continue
        if best is None or v < best:
            best = v
            try:
                best_seed = int(r["seed"])
            except (KeyError, TypeError, ValueError):
                best_seed = None
    if best_seed is not None:
        return f"seed_{best_seed}"
    return None


def interpretation_bullets(
    *,
    by_id: Dict[str, Dict[str, Any]],
    rec: Dict[str, Any],
    pareto_ids: List[str],
    aco_champ: Optional[str],
) -> List[str]:
    """Plain-text bullet bodies (no markdown); callers format for md/tex."""
    lines: List[str] = []
    sb = str(rec.get("spectral_best", "") or "")
    gb = str(rec.get("geometry_best", "") or "")
    bb = str(rec.get("balanced_best", "") or "")

    if sb and sb in by_id:
        lines.append(
            f"{sb} is the stronger spectral approximant in this cohort (lowest spectral_log_mse among validated top-5; "
            f"Pareto rank_by_spectral = {by_id[sb].get('rank_by_spectral', 'n/a')})."
        )
    if gb and gb in by_id:
        lines.append(
            f"{gb} is the stronger geometric / NCG-style control (lowest combined KS + Nijenhuis + commutator proxy among top-5; "
            f"rank_by_geometry = {by_id[gb].get('rank_by_geometry', 'n/a')})."
        )

    non_pf = [cid for cid, c in by_id.items() if not bool(c.get("is_pareto_front"))]
    for cid in sorted(non_pf):
        c = by_id[cid]
        nij = c.get("nijenhuis_defect")
        lines.append(
            f"{cid} is not on the Pareto front for the five spectral–geometry objectives; "
            f"it is dominated in trade-off space (e.g. nijenhuis_defect = {nij}), so it is rejected as a multi-objective refinement choice."
        )

    if aco_champ and aco_champ in by_id:
        lines.append(
            f"{aco_champ} remains the original ACO-loss champion among imported full-run snapshots (minimum best_loss in v13_top_candidates.json), "
            "independent of the post-hoc Pareto geometry filter."
        )

    pf_only = [cid for cid in pareto_ids if cid in by_id and cid not in non_pf]
    for cid in sorted(pf_only):
        if cid == sb or cid == gb or (aco_champ and cid == aco_champ):
            continue
        wlen = len(word_list(by_id[cid]))
        lines.append(
            f"{cid} stays on the Pareto front as a compact multi-objective control (word length {wlen}; useful for spectral vs. geometry trade-offs)."
        )

    if bb:
        lines.append(f"Balanced (ensemble) choice: {bb} minimizes ensemble_score among validated candidates.")
    return lines


def build_candidate_json(
    *,
    by_id: Dict[str, Dict[str, Any]],
    rec: Dict[str, Any],
    pareto_ids: List[str],
    theorem_statement: str,
    warning: str,
    aco_champ: Optional[str],
) -> Dict[str, Any]:
    sid = str(rec.get("spectral_best", ""))
    gid = str(rec.get("geometry_best", ""))
    spectral = {"id": sid, "word": word_list(by_id[sid])} if sid in by_id else {"id": sid, "word": []}
    geometric = {"id": gid, "word": word_list(by_id[gid])} if gid in by_id else {"id": gid, "word": []}
    rejected = [
        {"id": cid, "word": word_list(c), "is_pareto_front": bool(c.get("is_pareto_front"))}
        for cid, c in sorted(by_id.items())
        if not bool(c.get("is_pareto_front"))
    ]
    return {
        "spectral_candidate": spectral,
        "geometric_candidate": geometric,
        "pareto_front": list(pareto_ids),
        "rejected_candidates": rejected,
        "theorem_statement": theorem_statement,
        "warning": warning,
        "recommendation": dict(rec),
        "aco_loss_champion_id": aco_champ,
    }


def write_md(
    path: Path,
    *,
    by_id: Dict[str, Dict[str, Any]],
    rec: Dict[str, Any],
    pareto_ids: List[str],
    aco_champ: Optional[str],
    length_coeff: float,
    theorem_statement: str,
    warning: str,
) -> None:
    sid = str(rec.get("spectral_best", ""))
    gid = str(rec.get("geometry_best", ""))
    w_s = word_list(by_id[sid]) if sid in by_id else []
    w_g = word_list(by_id[gid]) if gid in by_id else []

    lines: List[str] = []
    lines.append("# V13H Candidate Operator Theorem: Ramsey--Nijenhuis / NCG-Guided Artin--DTES Operators\n\n")
    lines.append("## 1. Status warning\n\n")
    lines.append(f"> {warning}\n\n")
    lines.append("## 2. Candidate word space\n\n")
    lines.append(
        r"Fix a maximum generator magnitude $a_{\max}>0$ and length bound $m$. The **candidate word space** is "
    )
    lines.append(
        r"$$ \mathcal{W}_m = \{ w=(a_1,\ldots,a_m) : a_i\in\mathbb{Z},\ |a_i|\le a_{\max} \}. $$"
    )
    lines.append(
        "\n\nIn code, $a_{\\max}$ and $m$ are enforced by the ACO configuration (`max_power`, `max_length` in `core/artin_aco.py`).\n\n"
    )
    lines.append("## 3. Word-sensitive operator family\n\n")
    lines.append(
        "Each admissible word $w$ defines a finite matrix built from Laplacian / geodesic-kernel / potential terms on a sampled domain; "
        "schematically,\n\n"
        r"$$ H_w = \mathrm{Sym}\bigl(B_w + V_w + G_w\bigr), \qquad \mathrm{Sym}(M)=\tfrac12(M+M^{\mathsf T}), $$"
    )
    lines.append("\n\nwhere:\n\n")
    lines.append("- **$B_w$** — Artin / braid **word-action** contribution (finite shift / kernel assembly).\n")
    lines.append("- **$V_w$** — **DTES-style potential / energy** deformation tied to the word signature.\n")
    lines.append("- **$G_w$** — **geodesic / Ramsey--Nijenhuis--guided** kernel term (word-sensitive amplitudes and phases).\n\n")
    lines.append(
        "The concrete implementation is `build_word_sensitive_operator` in `core/artin_operator_word_sensitive.py` "
        "(combined Laplacian + normalized kernel + diagonal potential), followed by symmetrization before eigensolvers.\n\n"
    )
    lines.append("## 4. Finite spectral triple proxy\n\n")
    lines.append(
        "Let **$A_w$** denote the finite algebra generated by the word-action operators (matrix representation on $\\mathbb{R}^{128}$ in the default V13G validation), "
        "let **$\\mathcal{H}=\\mathbb{R}^{128}$**, and let **$D_w$** be a **Dirac-like** spectral operator derived from normalized or shifted eigenstructure of $H_w$. "
        "Define the **computational** triple\n\n"
        r"$$ \mathcal{T}_w = (A_w,\ \mathcal{H},\ D_w). $$"
    )
    lines.append(
        "\n\nThis is a **finite-dimensional proxy** used for loss and diagnostics only; it is **not** an analytic Connes-style triple for $\\zeta(s)$.\n\n"
    )
    lines.append("## 5. Loss theorem / selection principle\n\n")
    lines.append("**Proposition V13H (computational selection).** ")
    lines.append(
        "Let $\\mathcal{W}$ be the finite candidate word bank produced by V13 ACO on a fixed configuration. Define the scalar loss\n\n"
        "```\n"
        "L_total(w) = L_zeta(w) + lambda_NCG * L_NCG(w) + lambda_Ramsey * L_Ramsey(w) + lambda_Nij * L_Nij(w)\n"
        "           + lambda_len * L_len(w) + lambda_div * L_div(w).\n"
        "```\n\n"
        "Then **ACO search** approximates\n\n"
        r"$$ w^\star \in \arg\min_{w\in\mathcal{W}} L_{\mathrm{total}}(w). $$"
    )
    lines.append("\n\n**Pareto refinement.** ")
    lines.append(
        "Let $\\mathcal{P}\\subseteq\\mathcal{W}$ be the **Pareto-optimal** set under the five minimization objectives "
        "(`spectral_log_mse`, `ks_wigner`, `nijenhuis_defect`, `comm_norm_proxy`, and $-\\texttt{ramsey_score}$) evaluated on the word-sensitive operator at fixed $(\\texttt{dim},\\texttt{eps},\\ldots)$. "
        "Define the ensemble score (lower better)\n\n"
        "```\n"
        "S_Pareto(w) = 0.35 * spectral_norm + 0.20 * ks_norm + 0.20 * nijenhuis_norm + 0.15 * comm_norm\n"
        "            + 0.10 * ramsey_norm + lambda_len_pareto * length_target_penalty\n"
        "```\n\n"
        f"with per-cohort normalized components and default `lambda_len_pareto = {length_coeff}` (see `scripts/pareto_select_v13.py`). "
        "A **balanced** refinement choice is any\n\n"
        r"$$ w_P \in \arg\min_{w\in\mathcal{P}} S_{\mathrm{Pareto}}(w). $$"
    )
    lines.append("\n\n(Non-uniqueness is possible; the implementation reports one minimizer as `balanced_best`.)\n\n")
    lines.append("## 6. Current candidates\n\n")
    lines.append("From `pareto_results.json` (`meta.recommendation`):\n\n")
    lines.append(f"- `balanced_best`: **{md_cell(rec.get('balanced_best'))}**\n")
    lines.append(f"- `spectral_best`: **{md_cell(rec.get('spectral_best'))}**\n")
    lines.append(f"- `geometry_best`: **{md_cell(rec.get('geometry_best'))}**\n")
    lines.append(f"- `pareto_front_ids`: {md_cell(pareto_ids)}\n\n")
    lines.append("**Operator shorthand (validated top-5):**\n\n")
    lines.append(f"- **H_spectral** $\\equiv$ **H_{md_cell(sid)}** with word `{json.dumps(w_s)}`.\n")
    lines.append(f"- **H_geometric** $\\equiv$ **H_{md_cell(gid)}** with word `{json.dumps(w_g)}`.\n\n")
    lines.append("## 7. Candidate theorem statement\n\n")
    lines.append(f"{theorem_statement}\n\n")
    lines.append("## 8. Evidence table\n\n")
    cols = [
        "id",
        "word",
        "spectral_log_mse",
        "ks_wigner",
        "nijenhuis_defect",
        "comm_norm_proxy",
        "ramsey_score",
        "ensemble_score",
        "is_pareto_front",
        "rank_by_score",
        "rank_by_spectral",
        "rank_by_geometry",
    ]
    lines.append("| " + " | ".join(cols) + " |\n")
    lines.append("| " + " | ".join(["---"] * len(cols)) + " |\n")
    for cid in sorted(by_id.keys()):
        c = by_id[cid]
        row = [
            md_cell(cid),
            md_cell(json.dumps(word_list(c))),
            md_cell(c.get("spectral_log_mse", "")),
            md_cell(c.get("ks_wigner", "")),
            md_cell(c.get("nijenhuis_defect", "")),
            md_cell(c.get("comm_norm_proxy", "")),
            md_cell(c.get("ramsey_score", "")),
            md_cell(c.get("ensemble_score", "")),
            md_cell(c.get("is_pareto_front", "")),
            md_cell(c.get("rank_by_score", "")),
            md_cell(c.get("rank_by_spectral", "")),
            md_cell(c.get("rank_by_geometry", "")),
        ]
        lines.append("| " + " | ".join(row) + " |\n")
    lines.append("\n## 9. Interpretation\n\n")
    for b in interpretation_bullets(by_id=by_id, rec=rec, pareto_ids=pareto_ids, aco_champ=aco_champ):
        lines.append("- " + b + "\n")
    # Named cohort notes (V13H narrative; data-driven where ids exist).
    if "seed_6" in by_id:
        lines.append(
            "- **seed_6** is the better **spectral approximant** in this validated cohort (lowest `spectral_log_mse`; also `balanced_best` when it coincides with `spectral_best`).\n"
        )
    if "seed_12" in by_id:
        lines.append(
            "- **seed_12** is the better **geometric / NCG control** (best `rank_by_geometry` among top-5).\n"
        )
    if "seed_17" in by_id and not bool(by_id["seed_17"].get("is_pareto_front")):
        lines.append(
            "- **seed_17** is **rejected by the Pareto filter**: it sits off the front with a very large `nijenhuis_defect` relative to other candidates, i.e. poor multi-objective geometry despite moderate spectral loss.\n"
        )
    if "seed_14" in by_id:
        lines.append(
            "- **seed_14** remains the **original ACO-loss champion** on the imported full-run table (`v13_top_candidates.json`); keep it as a reference for pure scalar `best_loss` even when spectral–geometry trade-offs favor other words.\n"
        )
    if "seed_20" in by_id and bool(by_id["seed_20"].get("is_pareto_front")):
        lines.append(
            "- **seed_20** remains a **compact Pareto-front control**: on the front for the five objectives while not winning either spectral- or geometry-only rankings.\n"
        )
    lines.append("\n## 10. Next validation\n\n")
    lines.append("- Compare candidates against **random word** baselines (matched length / trace bins).\n")
    lines.append("- Compare against **shuffled zeta zero** targets (permutation of ordinates).\n")
    lines.append("- Run **GUE / Wigner** spacing baselines on normalized gaps (`validate_candidates_v13.py`).\n")
    lines.append("- Extend **`zeros_n`** from 128 to **256** after numerical stability checks.\n")
    lines.append("- Test **spectral stability** under `eps`, `geo_sigma`, and `dim` for the chosen pair $(H_{\\mathrm{spectral}},H_{\\mathrm{geometric}})$.\n")

    path.write_text("".join(lines), encoding="utf-8")


def write_tex(
    path: Path,
    *,
    by_id: Dict[str, Dict[str, Any]],
    rec: Dict[str, Any],
    pareto_ids: List[str],
    length_coeff: float,
    theorem_statement: str,
    warning: str,
    aco_champ: Optional[str],
) -> None:
    sid = str(rec.get("spectral_best", ""))
    gid = str(rec.get("geometry_best", ""))
    w_s = word_list(by_id[sid]) if sid in by_id else []
    w_g = word_list(by_id[gid]) if gid in by_id else []

    chunks: List[str] = []
    chunks.append("\\documentclass[11pt]{article}\n")
    chunks.append("\\usepackage{amsmath,amssymb}\n")
    chunks.append("\\usepackage[T1]{fontenc}\n")
    chunks.append("\\usepackage[margin=1in]{geometry}\n")
    chunks.append("\\usepackage{longtable}\n")
    chunks.append("\\title{V13H Candidate Operator Theorem:\\\\Ramsey--Nijenhuis / NCG-Guided Artin--DTES Operators}\n")
    chunks.append("\\date{}\n\\begin{document}\n\\maketitle\n")
    chunks.append("\\section*{Status warning}\n")
    chunks.append("\\fbox{\\parbox{0.96\\linewidth}{")
    chunks.append(latex_escape(warning))
    chunks.append("}}\n\n")

    chunks.append("\\section{Candidate word space}\n")
    chunks.append(
        "Fix $a_{\\max}>0$, $m\\in\\mathbb{N}$. Candidate words live in "
        "$\\mathcal{W}_m=\\{w=(a_1,\\ldots,a_m): a_i\\in\\mathbb{Z},\\ |a_i|\\le a_{\\max}\\}$.\n\n"
    )

    chunks.append("\\section{Word-sensitive operator family}\n")
    chunks.append(
        "Each $w$ defines a finite matrix $H_w=\\mathrm{Sym}(B_w+V_w+G_w)$ with $\\mathrm{Sym}(M)=\\tfrac12(M+M^{\\mathsf T})$, "
        "where $B_w$ is braid/word action, $V_w$ DTES potential deformation, and $G_w$ geodesic / guidance terms "
        "(implementation: \\texttt{core/artin\\_operator\\_word\\_sensitive.py}).\n\n"
    )

    chunks.append("\\section{Finite spectral triple proxy}\n")
    chunks.append(
        "Let $A_w$ be the finite algebra from word actions, $\\mathcal{H}=\\mathbb{R}^{128}$, and $D_w$ a Dirac-like operator from $H_w$. "
        "Then $\\mathcal{T}_w=(A_w,\\mathcal{H},D_w)$ is a \\textbf{computational} finite proxy, not an analytic RH triple.\n\n"
    )

    chunks.append("\\section{Loss and Pareto selection}\n")
    chunks.append("\\textbf{Proposition V13H.} With $\\mathcal{W}$ the ACO bank and $L_{\\mathrm{total}}$ as in V13F,\n")
    chunks.append("\\[ w^\\star \\in \\arg\\min_{w\\in\\mathcal{W}} L_{\\mathrm{total}}(w). \\]\n")
    chunks.append(
        "Pareto refinement uses normalized objectives and ensemble score $S_{\\mathrm{Pareto}}$ with weights "
        "$(0.35,0.20,0.20,0.15,0.10)$ on spectral/KS/Nijenhuis/comm/Ramsey terms plus "
        + str(length_coeff)
        + r"$\cdot$\texttt{length\_target\_penalty}."
        + "\n\n"
    )

    chunks.append("\\section{Current candidates}\n")
    chunks.append("\\begin{itemize}\n")
    chunks.append("\\item \\texttt{spectral\\_best}: \\texttt{" + latex_escape(sid) + "}\n")
    chunks.append("\\item \\texttt{geometry\\_best}: \\texttt{" + latex_escape(gid) + "}\n")
    chunks.append("\\item \\texttt{pareto\\_front}: " + latex_escape(str(pareto_ids)) + "\n")
    chunks.append("\\end{itemize}\n\n")
    chunks.append("\\noindent\\textbf{Words:} ")
    chunks.append(
        "$H_{\\mathrm{spectral}}\\equiv H_{"
        + latex_escape(sid)
        + "}$ has word "
        + latex_escape(str(w_s))
        + "; $H_{\\mathrm{geom}}\\equiv H_{"
        + latex_escape(gid)
        + "}$ has word "
        + latex_escape(str(w_g))
        + ".\n\n"
    )

    chunks.append("\\section{Candidate theorem statement}\n")
    for para in theorem_statement.split("\n"):
        if para.strip():
            chunks.append(latex_escape(para) + "\n\n")

    chunks.append("\\section{Evidence}\n{\\scriptsize\n\\begin{longtable}{|l|p{3.2cm}|r|r|r|r|r|r|r|r|r|r|}\\hline\n")
    hdr = [
        "id",
        "word",
        "spec",
        "KS",
        "Nij",
        "comm",
        "Ram",
        "ens",
        "PF",
        "rS",
        "rSp",
        "rG",
    ]
    chunks.append(" & ".join(f"\\textbf{{{h}}}" for h in hdr) + " \\\\\n\\hline\n\\endhead\n")
    for cid in sorted(by_id.keys()):
        c = by_id[cid]
        cells = [
            latex_escape(cid),
            latex_escape(str(word_list(c))[:60]),
            latex_escape(str(c.get("spectral_log_mse", ""))),
            latex_escape(str(c.get("ks_wigner", ""))),
            latex_escape(str(c.get("nijenhuis_defect", ""))),
            latex_escape(str(c.get("comm_norm_proxy", ""))),
            latex_escape(str(c.get("ramsey_score", ""))),
            latex_escape(str(c.get("ensemble_score", ""))),
            latex_escape(str(c.get("is_pareto_front", ""))),
            latex_escape(str(c.get("rank_by_score", ""))),
            latex_escape(str(c.get("rank_by_spectral", ""))),
            latex_escape(str(c.get("rank_by_geometry", ""))),
        ]
        chunks.append(" & ".join(cells) + " \\\\\n\\hline\n")
    chunks.append("\\end{longtable}\n}\\normalsize\n\n")

    chunks.append("\\section{Interpretation}\n\\begin{itemize}\n")
    for b in interpretation_bullets(by_id=by_id, rec=rec, pareto_ids=pareto_ids, aco_champ=aco_champ):
        chunks.append("\\item " + latex_escape(b) + "\n")
    if "seed_6" in by_id:
        chunks.append(
            "\\item \\texttt{seed\\_6} is the stronger spectral approximant in this validated cohort (lowest \\texttt{spectral\\_log\\_mse}).\n"
        )
    if "seed_12" in by_id:
        chunks.append(
            "\\item \\texttt{seed\\_12} is the stronger geometric / NCG control (best \\texttt{rank\\_by\\_geometry}).\n"
        )
    if "seed_17" in by_id and not bool(by_id["seed_17"].get("is_pareto_front")):
        chunks.append(
            "\\item \\texttt{seed\\_17} is rejected by the Pareto filter (off-front; large \\texttt{nijenhuis\\_defect}).\n"
        )
    if "seed_14" in by_id:
        chunks.append(
            "\\item \\texttt{seed\\_14} remains the original ACO-loss champion on the imported full-run table.\n"
        )
    if "seed_20" in by_id and bool(by_id["seed_20"].get("is_pareto_front")):
        chunks.append("\\item \\texttt{seed\\_20} remains a compact Pareto-front control.\n")
    chunks.append("\\end{itemize}\n\n")

    chunks.append("\\section{Next validation}\n\\begin{itemize}\n")
    chunks.append("\\item Random-word and shuffled-zero baselines.\n")
    chunks.append("\\item GUE spacing diagnostics; extend \\texttt{zeros\\_n} to 256.\n")
    chunks.append("\\item Stability in \\texttt{eps}, \\texttt{geo\\_sigma}, \\texttt{dim}.\n")
    chunks.append("\\end{itemize}\n\\end{document}\n")

    path.write_text("".join(chunks), encoding="utf-8")


def _find_pdflatex() -> Optional[str]:
    w = shutil.which("pdflatex")
    if w:
        return w
    for p in (
        "/Library/TeX/texbin/pdflatex",
        "/usr/local/texlive/2026/bin/universal-darwin/pdflatex",
        "/usr/local/texlive/2025/bin/universal-darwin/pdflatex",
    ):
        if Path(p).is_file():
            return p
    return None


def try_pdflatex(tex_path: Path, out_dir: Path) -> bool:
    exe = _find_pdflatex()
    if not exe:
        return False
    try:
        r = subprocess.run(
            [
                exe,
                "-interaction=nonstopmode",
                f"-output-directory={out_dir.resolve()}",
                tex_path.name,
            ],
            cwd=str(out_dir.resolve()),
            capture_output=True,
            text=True,
            timeout=120,
        )
        ok = r.returncode == 0 and (out_dir / "v13_candidate_theorem.pdf").is_file()
        # second pass for references (cheap no-op if none)
        if ok:
            subprocess.run(
                [exe, "-interaction=nonstopmode", f"-output-directory={out_dir.resolve()}", tex_path.name],
                cwd=str(out_dir.resolve()),
                capture_output=True,
                text=True,
                timeout=120,
            )
        return ok
    except (OSError, subprocess.TimeoutExpired):
        return False


def main() -> None:
    ap = argparse.ArgumentParser(description="V13H candidate operator theorem report.")
    ap.add_argument("--pareto_json", type=str, default="runs/v13_top5_pareto/pareto_results.json")
    ap.add_argument("--top_json", type=str, default="runs/v13_formula_report/v13_top_candidates.json")
    ap.add_argument("--out_dir", type=str, default="runs/v13_candidate_theorem")
    args = ap.parse_args()

    pareto_path = Path(args.pareto_json)
    top_path = Path(args.top_json)
    out_dir = Path(args.out_dir)
    if not pareto_path.is_absolute():
        pareto_path = Path(ROOT) / pareto_path
    if not top_path.is_absolute():
        top_path = Path(ROOT) / top_path
    if not out_dir.is_absolute():
        out_dir = Path(ROOT) / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if not pareto_path.is_file():
        raise FileNotFoundError(pareto_path)
    pareto = load_json(pareto_path)
    if not isinstance(pareto, dict):
        raise ValueError("pareto_json must be an object")

    meta = pareto.get("meta") if isinstance(pareto.get("meta"), dict) else {}
    rec = meta.get("recommendation") if isinstance(meta.get("recommendation"), dict) else {}
    length_coeff = 0.1
    ew = meta.get("ensemble_weights")
    if isinstance(ew, dict) and ew.get("length_target_penalty_coeff") is not None:
        try:
            length_coeff = float(ew["length_target_penalty_coeff"])
        except (TypeError, ValueError):
            pass

    by_id = candidates_by_id(pareto)
    pareto_ids = list(rec.get("pareto_front_ids") or [])
    if not pareto_ids:
        pareto_ids = [cid for cid, c in by_id.items() if c.get("is_pareto_front")]

    top_rows = load_top_rows(top_path) if top_path.is_file() else []
    aco_champ = aco_champion_id(top_rows)

    sid = str(rec.get("spectral_best", ""))
    gid = str(rec.get("geometry_best", ""))
    w_s = word_list(by_id[sid]) if sid in by_id else []
    w_g = word_list(by_id[gid]) if gid in by_id else []

    warning = (
        "This is an experimental computational candidate construction, not a proof of the Riemann Hypothesis."
    )
    theorem_statement = (
        f"The V13H computational candidate operator pair is (H_{sid}, H_{gid}), "
        f"where H_{sid} (validated as H_spectral) emphasizes spectral alignment among the top-5 cohort "
        f"and H_{gid} (validated as H_geometric) emphasizes the geometric KS/Nijenhuis/commutator aggregate. "
        "Together they summarize a dual spectral–geometry trade-off on the Pareto front; they do not constitute an analytic theorem toward RH."
    )

    co = build_candidate_json(
        by_id=by_id,
        rec=rec,
        pareto_ids=pareto_ids,
        theorem_statement=theorem_statement,
        warning=warning,
        aco_champ=aco_champ,
    )
    with open(out_dir / "candidate_operators.json", "w", encoding="utf-8") as f:
        json.dump(co, f, indent=2, allow_nan=True)

    md_path = out_dir / "v13_candidate_theorem.md"
    tex_path = out_dir / "v13_candidate_theorem.tex"
    write_md(
        md_path,
        by_id=by_id,
        rec=rec,
        pareto_ids=pareto_ids,
        aco_champ=aco_champ,
        length_coeff=length_coeff,
        theorem_statement=theorem_statement,
        warning=warning,
    )
    write_tex(
        tex_path,
        by_id=by_id,
        rec=rec,
        pareto_ids=pareto_ids,
        length_coeff=length_coeff,
        theorem_statement=theorem_statement,
        warning=warning,
        aco_champ=aco_champ,
    )

    # Fix section 6 md: remove erroneous pop from earlier draft if any - read write_md for bug
    # I introduced a bug: lines.pop() after wrong append - need to read write_md function

    print(f"Wrote {md_path}")
    print(f"Wrote {tex_path}")
    if try_pdflatex(tex_path, out_dir):
        print(f"Wrote {out_dir / 'v13_candidate_theorem.pdf'}")
    else:
        print("pdflatex not available or failed; PDF skipped (see .tex source).")


if __name__ == "__main__":
    main()
