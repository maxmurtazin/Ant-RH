#!/usr/bin/env python3
"""
V13F: Markdown + LaTeX formula report and top-candidate export from full-run ``artin_aco_best.json`` files.

Run from repo root:

  python3 scripts/make_v13_formula_report.py --out_dir runs/v13_formula_report
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DEFAULT_TOP_JSONS: List[str] = [
    os.path.join(ROOT, "runs/v13_full_seed_14/artin_aco_best.json"),
    os.path.join(ROOT, "runs/v13_full_seed_17/artin_aco_best.json"),
    os.path.join(ROOT, "runs/v13_full_seed_20/artin_aco_best.json"),
    os.path.join(ROOT, "runs/v13_full_seed_6/artin_aco_best.json"),
    os.path.join(ROOT, "runs/v13_full_seed_12/artin_aco_best.json"),
]

TABLE_COLUMNS: List[str] = [
    "seed",
    "source_json",
    "best_loss",
    "best_words",
    "best_lengths",
    "global_best_iter",
    "trace",
    "lambda_ncg",
    "lambda_ramsey",
    "lambda_nijenhuis",
    "lambda_length",
    "spec_nonfinite_count_total",
    "spec_clip_count_total",
    "n_spec_used_final",
    "L_spec_log_final",
    "comm_norm_mean_final",
    "nijenhuis_defect_min_final",
    "ramsey_score_max_final",
]


def _f(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _i(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def parse_seed_from_path(path: Path) -> Optional[int]:
    m = re.search(r"seed[_-]?(\d+)", path.as_posix(), re.I)
    if m:
        return int(m.group(1))
    return None


def coalesce_best_loss(d: Dict[str, Any]) -> float:
    for key in ("best_loss", "global_best_loss"):
        v = _f(d.get(key))
        if v is not None:
            return float(v)
    return float("inf")


def extract_row(path: Path, data: Dict[str, Any]) -> Dict[str, Any]:
    seed = _i(data.get("seed"))
    if seed is None:
        seed = parse_seed_from_path(path)

    gbc = data.get("global_best_candidate")
    trace_val: Any = None
    if isinstance(gbc, dict):
        trace_val = gbc.get("trace")

    bw = data.get("best_words")
    if not isinstance(bw, list):
        bw = []
    first_word: Any = None
    if bw and isinstance(bw[0], list):
        first_word = bw[0]
    elif isinstance(gbc, dict) and isinstance(gbc.get("a_list"), list):
        first_word = gbc["a_list"]

    blen = data.get("best_lengths")
    if not isinstance(blen, list) and first_word is not None:
        blen = [float(len(first_word))]
    if not isinstance(blen, list):
        blen = []

    loss = _f(data.get("best_loss"))
    if loss is None:
        loss = _f(data.get("global_best_loss"))

    return {
        "seed": seed,
        "source_json": str(path.as_posix()),
        "best_loss": loss,
        "best_words": bw if bw else ([first_word] if first_word is not None else []),
        "best_lengths": blen,
        "global_best_iter": _i(data.get("global_best_iter")),
        "trace": _f(trace_val),
        "lambda_ncg": _f(data.get("lambda_ncg")),
        "lambda_ramsey": _f(data.get("lambda_ramsey")),
        "lambda_nijenhuis": _f(data.get("lambda_nijenhuis")),
        "lambda_length": _f(data.get("lambda_length")),
        "spec_nonfinite_count_total": _i(data.get("spec_nonfinite_count_total")),
        "spec_clip_count_total": _i(data.get("spec_clip_count_total")),
        "n_spec_used_final": _f(data.get("n_spec_used_final")),
        "L_spec_log_final": _f(data.get("L_spec_log_final")),
        "comm_norm_mean_final": _f(data.get("comm_norm_mean_final")),
        "nijenhuis_defect_min_final": _f(data.get("nijenhuis_defect_min_final")),
        "ramsey_score_max_final": _f(data.get("ramsey_score_max_final")),
    }


def load_runs(paths: List[str]) -> Tuple[List[Dict[str, Any]], List[str]]:
    rows: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for raw in paths:
        p = Path(raw)
        if not p.is_file():
            warnings.append(f"missing_json:{p.as_posix()}")
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                warnings.append(f"invalid_json_object:{p.as_posix()}")
                continue
            rows.append(extract_row(p, data))
        except Exception as ex:
            warnings.append(f"read_error:{p.as_posix()}:{ex!r}")
    return rows, warnings


def fmt_words_cell(words: Any) -> str:
    if not isinstance(words, list) or not words:
        return ""
    if words and isinstance(words[0], list):
        return json.dumps(words[0])
    return json.dumps(words)


def fmt_lengths_cell(lengths: Any) -> str:
    if isinstance(lengths, list) and lengths:
        return json.dumps(lengths)
    return ""


def md_escape_cell(s: str) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")


def latex_truncate(s: str, n: int = 48) -> str:
    t = str(s)
    return t if len(t) <= n else t[: n - 3] + "..."


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


def top5_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def key(r: Dict[str, Any]) -> Tuple[float, str]:
        return (coalesce_best_loss(r), str(r.get("source_json", "")))

    s = sorted(rows, key=key)
    return s[:5]


def _as_int(x: Any) -> int:
    if x is None:
        return 0
    try:
        return int(x)
    except (TypeError, ValueError):
        return 0


def aggregate_spec_counts(rows: List[Dict[str, Any]]) -> Tuple[int, int]:
    t_nf = sum(_as_int(r.get("spec_nonfinite_count_total")) for r in rows)
    t_clip = sum(_as_int(r.get("spec_clip_count_total")) for r in rows)
    return t_nf, t_clip


def primary_candidate(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not rows:
        return None
    return min(rows, key=lambda r: (coalesce_best_loss(r), str(r.get("source_json", ""))))


def interpretation_notes(top: List[Dict[str, Any]]) -> Dict[str, Any]:
    notes: Dict[str, Any] = {}
    if not top:
        return {"notes": "No loaded runs; add JSON paths under runs/v13_full_seed_*/artin_aco_best.json."}

    def sid(r: Dict[str, Any]) -> str:
        s = r.get("seed")
        return f"seed_{s}" if s is not None else Path(str(r.get("source_json", ""))).parent.name or "?"

    # spectral champion: min L_spec_log_final
    spec_rows = [(r, r.get("L_spec_log_final")) for r in top]
    spec_rows = [(r, v) for r, v in spec_rows if isinstance(v, (int, float)) and math.isfinite(float(v))]
    if spec_rows:
        rmin = min(spec_rows, key=lambda x: float(x[1]))[0]
        notes["spectral_champion"] = f"{sid(rmin)} (L_spec_log_final={rmin.get('L_spec_log_final')})"
    else:
        notes["spectral_champion"] = "n/a (no finite L_spec_log_final in top set)"

    cand_lens: List[Tuple[Dict[str, Any], int]] = []
    for r in top:
        bw = r.get("best_words")
        if isinstance(bw, list) and bw and isinstance(bw[0], list):
            cand_lens.append((r, len(bw[0])))
    if cand_lens:
        rshort, ln = min(cand_lens, key=lambda x: x[1])
        notes["shortest_candidate"] = f"{sid(rshort)} (|w|={ln})"
    else:
        notes["shortest_candidate"] = "n/a"

    nij_rows = [(r, r.get("nijenhuis_defect_min_final")) for r in top]
    nij_rows = [(r, float(v)) for r, v in nij_rows if isinstance(v, (int, float)) and math.isfinite(float(v))]
    if nij_rows:
        r_nij = min(nij_rows, key=lambda x: x[1])[0]
        notes["best_nijenhuis_min"] = f"{sid(r_nij)} (nijenhuis_defect_min_final={r_nij.get('nijenhuis_defect_min_final')})"
    else:
        notes["best_nijenhuis_min"] = "n/a"

    comm_rows = [(r, r.get("comm_norm_mean_final")) for r in top]
    comm_rows = [(r, float(v)) for r, v in comm_rows if isinstance(v, (int, float)) and math.isfinite(float(v))]
    if comm_rows:
        r_c = min(comm_rows, key=lambda x: x[1])[0]
        notes["best_comm_norm_mean"] = f"{sid(r_c)} (comm_norm_mean_final={r_c.get('comm_norm_mean_final')})"
    else:
        notes["best_comm_norm_mean"] = "n/a"

    nf_ok = bool(top) and all(_as_int(r.get("spec_nonfinite_count_total")) == 0 for r in top)
    clip_ok = bool(top) and all(_as_int(r.get("spec_clip_count_total")) == 0 for r in top)
    notes["clean_spectral_status"] = (
        "All listed runs report spec_nonfinite_count_total=0 and spec_clip_count_total=0."
        if nf_ok and clip_ok and top
        else "Some runs report non-zero non-finite or clip counts, or counts are missing."
    )
    return notes


def build_markdown(
    *,
    top: List[Dict[str, Any]],
    all_rows: List[Dict[str, Any]],
    warnings: List[str],
    totals_nf: int,
    totals_clip: int,
    primary: Optional[Dict[str, Any]],
    interp: Dict[str, Any],
) -> str:
    lines: List[str] = []
    lines.append("# V13F Formula Report: Ramsey--Nijenhuis Guided ACO for Ant-RH\n")
    lines.append("## 1. Executive summary\n")
    lines.append(
        "Full-run ant-colony search evaluates **finite-dimensional** word-sensitive operators and produces "
        "**finite** eigenvalue spectra for the spectral loss term when configuration rejects non-finite spectra.\n"
    )
    lines.append(
        f"Across **{len(all_rows)}** loaded `artin_aco_best.json` file(s): "
        f"`spec_nonfinite_count_total` summed to **{totals_nf}**; "
        f"`spec_clip_count_total` summed to **{totals_clip}** "
        "(per-file totals from each run's final snapshot). "
        "In a clean full-run ensemble these totals are **0 / 0** (no non-finite spectra; no clip events).\n"
    )
    if primary is not None:
        pl = primary.get("best_loss")
        ps = primary.get("seed")
        lines.append(
            f"**Primary candidate (minimum `best_loss` among loaded JSONs):** seed={ps!s}, "
            f"best_loss={pl!s}, source `{md_escape_cell(str(primary.get('source_json', '')))}`.\n"
        )
    else:
        lines.append("**Primary candidate:** none (no JSON files loaded).\n")
    if warnings:
        lines.append("\n**Load warnings:**\n")
        for w in warnings:
            lines.append(f"- `{md_escape_cell(w)}`\n")
    lines.append("\n---\n")
    lines.append("## 2. Word space\n")
    lines.append(r"Words are finite integer sequences \(w = (a_1,\ldots,a_m)\) with \(a_i \in \mathbb{Z}\), ")
    lines.append("encoding Artin-generator exponents / braid alphabet tokens used by the ACO search.\n")
    lines.append("\n---\n")
    lines.append("## 3. Operator family\n")
    lines.append(
        r"For each word \(w\), the **word-sensitive** Artin / DTES-style kernel builds a symmetric matrix "
        r"`H_w = H_DTES(w; dim, eps, geo_sigma, geo_weight, potential_weight)` on a sampled domain of "
        r"`dim` points (see `core/artin_operator_word_sensitive.py`). **Different words induce different** "
        r"`H_w`. Before spectral loss, `H_w` is **symmetrized** (and stabilized) for eigen-solvers used in training.\n"
    )
    lines.append("\n---\n")
    lines.append("## 4. Total loss formula\n")
    lines.append("```\n")
    lines.append(
        "L_total(w) = L_zeta(w) + lambda_NCG * L_NCG(w) + lambda_Ramsey * L_Ramsey(w) + lambda_Nij * L_Nij(w)\n"
        "           + lambda_len * L_len(w) + lambda_div * L_div(w)\n"
    )
    lines.append("```\n")
    lines.append("(Coefficients `lambda_*` are CLI-hyperparameters in `core/artin_aco.py`; DTES triple terms may be folded into the NCG channel depending on configuration.)\n")
    lines.append("\n---\n")
    lines.append("## 5. Component definitions\n")
    lines.append("- **L_zeta(w)** $\\approx \\log(1 + \\mathrm{MSE}(\\mathrm{normspec}(H_w), \\mathrm{normzeros\\_zeta}))$ — spectral alignment to normalized zeta ordinates.\n")
    lines.append("- **L_NCG(w)** $\\approx \\|[D_w, A_w]\\|_F / \\max(\\varepsilon, \\|D_w\\|_F\\|A_w\\|_F)$ — commutator / triple proxy (finite-dimensional).\n")
    lines.append("- **L_Ramsey(w)** $= 1 - \\mathrm{RamseyScore}(w)$ — run-length / block anti-monochrome style signal.\n")
    lines.append("- **L_Nij(w)** $= \\mathrm{NijenhuisDefect}(w)$ — torsion proxy from a finite shift operator built from the word.\n")
    lines.append("- **L_len(w)** $= |\\mathrm{length}(w) - \\mathrm{target\\_length}| / \\mathrm{target\\_length}$.\n")
    lines.append("- **L_div(w)** $= \\mathrm{diversity\\_penalty}(w, \\mathrm{bank})$ — discourages collapse to a single motif.\n")
    lines.append("\n---\n")
    lines.append("## 6. Spectral triple interpretation\n")
    lines.append(
        "Let **A_w** be the algebra / shift action generated from the word (finite matrix representation), "
        "**H** the finite-dimensional Hilbert space $\\mathbb{R}^{dim}$ (or $\\mathbb{C}^{dim}$ with the same real "
        "symmetric reduction where applicable), and **D_w** a **Dirac-like** diagonal or spectral operator derived from "
        "eigenstructure of `H_w`. The tuple **(A_w, H, D_w)** is a **computational finite-dimensional spectral-triple proxy** "
        "used for regularization and diagnostics; **it is not a proof of the Riemann Hypothesis**.\n"
    )
    lines.append("\n> **Warning:** This is an experimental computational framework, **not a proof of the Riemann Hypothesis**.\n")
    lines.append("\n---\n")
    lines.append("## 7. Top-5 full-run table\n")
    headers = TABLE_COLUMNS
    lines.append("| " + " | ".join(headers) + " |\n")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |\n")
    for r in top:
        cells = [
            str(r.get("seed", "")),
            md_escape_cell(str(r.get("source_json", ""))),
            str(r.get("best_loss", "")),
            md_escape_cell(fmt_words_cell(r.get("best_words"))),
            md_escape_cell(fmt_lengths_cell(r.get("best_lengths"))),
            str(r.get("global_best_iter", "")),
            str(r.get("trace", "")),
            str(r.get("lambda_ncg", "")),
            str(r.get("lambda_ramsey", "")),
            str(r.get("lambda_nijenhuis", "")),
            str(r.get("lambda_length", "")),
            str(r.get("spec_nonfinite_count_total", "")),
            str(r.get("spec_clip_count_total", "")),
            str(r.get("n_spec_used_final", "")),
            str(r.get("L_spec_log_final", "")),
            str(r.get("comm_norm_mean_final", "")),
            str(r.get("nijenhuis_defect_min_final", "")),
            str(r.get("ramsey_score_max_final", "")),
        ]
        lines.append("| " + " | ".join(cells) + " |\n")
    lines.append("\n---\n")
    lines.append("## 8. Interpretation (automatic notes)\n")
    for k, v in interp.items():
        if k == "notes":
            lines.append(f"- **notes:** {v}\n")
        else:
            lines.append(f"- **{k}:** {v}\n")
    lines.append("\n---\n")
    lines.append("## 9. Next experiments\n")
    lines.append("- Validate top candidates with `python3 scripts/validate_candidates_v13.py` (word-sensitive operator path).\n")
    lines.append("- Run Pareto selection on merged validation outputs: `python3 scripts/pareto_select_v13.py`.\n")
    lines.append("- Compare against **random words** and **shuffled** word controls (same length / trace bins).\n")
    lines.append("- Add **GUE / Wigner** spacing diagnostics (already in V13C validation pipeline).\n")
    lines.append("- After stability checks, increase zeta ordinate count (e.g. **128 → 256** zeros) for spectral MSE.\n")
    return "".join(lines)


def build_latex(
    *,
    top: List[Dict[str, Any]],
    all_rows: List[Dict[str, Any]],
    warnings: List[str],
    totals_nf: int,
    totals_clip: int,
    primary: Optional[Dict[str, Any]],
    interp: Dict[str, Any],
) -> str:
    chunks: List[str] = []
    chunks.append("\\documentclass[11pt]{article}\n")
    chunks.append("\\usepackage{amsmath,amssymb}\n")
    chunks.append("\\usepackage[T1]{fontenc}\n")
    chunks.append("\\usepackage[margin=1in]{geometry}\n")
    chunks.append("\\usepackage{longtable}\n")
    chunks.append("\\title{V13F Formula Report: Ramsey--Nijenhuis Guided ACO for Ant-RH}\n")
    chunks.append("\\date{}\n")
    chunks.append("\\begin{document}\n\\maketitle\n")

    chunks.append("\\section*{Executive summary}\n")
    chunks.append(
        "Full-run search uses finite-dimensional word-sensitive operators with finite spectra under the configured "
        "reject/clipping policy. "
    )
    chunks.append(
        "Across "
        + str(len(all_rows))
        + r" loaded JSON file(s), \texttt{spec\_nonfinite\_count\_total} summed to "
        + str(totals_nf)
        + r" and \texttt{spec\_clip\_count\_total} summed to "
        + str(totals_clip)
        + ".\n\n"
    )
    if primary:
        chunks.append(
            "\\textbf{Primary candidate:} minimum \\texttt{best\\_loss} among loaded files: "
            "\\texttt{best\\_loss}="
            + latex_escape(str(primary.get("best_loss")))
            + ", \\texttt{seed}="
            + latex_escape(str(primary.get("seed")))
            + ".\n\n"
        )
    else:
        chunks.append("\\textbf{Primary candidate:} none (no files loaded).\n\n")
    if warnings:
        chunks.append("\\paragraph{Load warnings.}\n\\begin{itemize}\n")
        for w in warnings:
            chunks.append(f"\\item \\texttt{{{latex_escape(w)}}}\n")
        chunks.append("\\end{itemize}\n\n")

    chunks.append("\\section{Word space}\n")
    chunks.append(
        "A word is $w=(a_1,\\ldots,a_m)$ with $a_i\\in\\mathbb{Z}$, representing generator exponents / tokens.\n\n"
    )

    chunks.append("\\section{Operator family}\n")
    chunks.append(
        "The operator $H_w = H_{\\mathrm{DTES}}(w;\\mathrm{dim},\\varepsilon,\\sigma_g,w_g,w_p)$ is built by the "
        "word-sensitive Artin/DTES operator family; distinct words yield distinct matrices. "
        "$H_w$ is symmetrized before spectral evaluation.\n\n"
    )

    chunks.append("\\section{Total loss}\n")
    chunks.append("\\begin{equation*}\n")
    chunks.append("\\begin{aligned}\n")
    chunks.append(
        "L_{\\mathrm{total}}(w) &= L_{\\zeta}(w)\n"
        "+ \\lambda_{\\mathrm{NCG}} L_{\\mathrm{NCG}}(w)\n"
        "+ \\lambda_{\\mathrm{Ramsey}} L_{\\mathrm{Ramsey}}(w)\n"
        "+ \\lambda_{\\mathrm{Nij}} L_{\\mathrm{Nij}}(w) \\\\\n"
        "&\\quad + \\lambda_{\\mathrm{len}} L_{\\mathrm{len}}(w)\n"
        "+ \\lambda_{\\mathrm{div}} L_{\\mathrm{div}}(w).\n"
    )
    chunks.append("\\end{aligned}\n\\end{equation*}\n\n")

    chunks.append("\\section{Component definitions}\n")
    chunks.append("\\begin{itemize}\n")
    chunks.append(
        "\\item $L_{\\zeta}(w) = \\log(1 + \\mathrm{MSE}(\\mathrm{normspec}(H_w),\\mathrm{normzeros\\_zeta}))$.\n"
    )
    chunks.append(
        "\\item $L_{\\mathrm{NCG}}(w) = \\|[D_w,A_w]\\|_F / \\max(\\varepsilon,\\|D_w\\|_F\\|A_w\\|_F)$.\n"
    )
    chunks.append("\\item $L_{\\mathrm{Ramsey}}(w) = 1 - \\mathrm{RamseyScore}(w)$.\n")
    chunks.append("\\item $L_{\\mathrm{Nij}}(w) = \\mathrm{NijenhuisDefect}(w)$.\n")
    chunks.append(
        "\\item $L_{\\mathrm{len}}(w) = |\\mathrm{length}(w)-\\mathrm{target\\_length}|/\\mathrm{target\\_length}$.\n"
    )
    chunks.append("\\item $L_{\\mathrm{div}}(w) = \\mathrm{diversity\\_penalty}(w,\\mathrm{bank})$.\n")
    chunks.append("\\end{itemize}\n\n")

    chunks.append("\\section{Spectral triple interpretation}\n")
    chunks.append(
        "Let $A_w$ be the finite shift/algebra action, $\\mathcal{H}\\cong\\mathbb{R}^{\\mathrm{dim}}$ the working Hilbert space, "
        "and $D_w$ a Dirac-like spectral operator derived from $H_w$. The triple $(A_w,\\mathcal{H},D_w)$ is a "
        "\\textbf{finite computational spectral-triple proxy}, not an analytic proof of RH.\n\n"
    )

    chunks.append(
        "\\fbox{\\parbox{0.96\\linewidth}{\\textbf{Warning:} "
        "This is an experimental computational framework, \\textbf{not a proof of the Riemann Hypothesis}.}}"
    )
    chunks.append("\n\n")

    chunks.append("\\section{Top-5 full-run snapshot}\n")
    chunks.append("{\\scriptsize\n")
    chunks.append("\\begin{longtable}{|p{0.6cm}|p{2.8cm}|p{1.1cm}|p{2.2cm}|p{1.4cm}|p{0.7cm}|p{0.9cm}|p{0.55cm}|p{0.55cm}|p{0.55cm}|p{0.55cm}|p{0.45cm}|p{0.45cm}|p{0.55cm}|p{0.9cm}|p{0.9cm}|p{0.9cm}|p{0.9cm}|}\n")
    chunks.append("\\hline\n")
    hdr = [
        "seed",
        "source",
        "best\\_loss",
        "best\\_words",
        "best\\_len",
        "gb\\_iter",
        "trace",
        "$\\lambda$ncg",
        "$\\lambda$R",
        "$\\lambda$Nij",
        "$\\lambda$len",
        "nf",
        "clip",
        "n\\_sp",
        "Lsp\\_log",
        "comm",
        "nij\\_min",
        "ram\\_max",
    ]
    chunks.append(" & ".join(f"\\textbf{{{h}}}" for h in hdr) + " \\\\\n\\hline\n\\endfirsthead\n")
    chunks.append(" & ".join(f"\\textbf{{{h}}}" for h in hdr) + " \\\\\n\\hline\n\\endhead\n")
    for r in top:
        bw = fmt_words_cell(r.get("best_words"))
        bl = fmt_lengths_cell(r.get("best_lengths"))
        src = latex_truncate(str(r.get("source_json", "")), 40)
        cells = [
            latex_escape(str(r.get("seed", ""))),
            latex_escape(src),
            latex_escape(str(r.get("best_loss", ""))),
            latex_escape(latex_truncate(bw, 36)),
            latex_escape(bl),
            latex_escape(str(r.get("global_best_iter", ""))),
            latex_escape(str(r.get("trace", ""))),
            latex_escape(str(r.get("lambda_ncg", ""))),
            latex_escape(str(r.get("lambda_ramsey", ""))),
            latex_escape(str(r.get("lambda_nijenhuis", ""))),
            latex_escape(str(r.get("lambda_length", ""))),
            latex_escape(str(r.get("spec_nonfinite_count_total", ""))),
            latex_escape(str(r.get("spec_clip_count_total", ""))),
            latex_escape(str(r.get("n_spec_used_final", ""))),
            latex_escape(str(r.get("L_spec_log_final", ""))),
            latex_escape(str(r.get("comm_norm_mean_final", ""))),
            latex_escape(str(r.get("nijenhuis_defect_min_final", ""))),
            latex_escape(str(r.get("ramsey_score_max_final", ""))),
        ]
        chunks.append(" & ".join(cells) + " \\\\\n\\hline\n")
    chunks.append("\\end{longtable}\n}\\normalsize\n\n")

    chunks.append("\\section{Automatic interpretation}\n\\begin{itemize}\n")
    for k, v in interp.items():
        chunks.append(f"\\item \\textbf{{{latex_escape(k)}}}: {latex_escape(str(v))}\n")
    chunks.append("\\end{itemize}\n\n")

    chunks.append("\\section{Next experiments}\n\\begin{itemize}\n")
    chunks.append("\\item Validate with \\texttt{scripts/validate\\_candidates\\_v13.py}.\n")
    chunks.append("\\item Pareto selection: \\texttt{scripts/pareto\\_select\\_v13.py}.\n")
    chunks.append("\\item Random/shuffled control baselines.\n")
    chunks.append("\\item GUE spacing diagnostics (V13C).\n")
    chunks.append("\\item Increase zeros 128 $\\to$ 256 after stability.\n")
    chunks.append("\\end{itemize}\n\n")

    chunks.append("\\end{document}\n")
    return "".join(chunks)


def json_safe_row(r: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(r)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="V13F formula report from artin_aco_best.json full runs.")
    ap.add_argument("--out_dir", type=str, default="runs/v13_formula_report")
    ap.add_argument(
        "--top_jsons",
        type=str,
        nargs="*",
        default=[],
        help="Optional explicit list of artin_aco_best.json paths (overrides built-in defaults if non-empty).",
    )
    args = ap.parse_args()

    paths = [str(Path(p).resolve()) for p in args.top_jsons] if args.top_jsons else list(DEFAULT_TOP_JSONS)
    rows, warnings = load_runs(paths)
    top = top5_rows(rows)
    totals_nf, totals_clip = aggregate_spec_counts(rows)
    primary = primary_candidate(rows)
    interp = interpretation_notes(top)

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = Path(ROOT) / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    md = build_markdown(
        top=top,
        all_rows=rows,
        warnings=warnings,
        totals_nf=totals_nf,
        totals_clip=totals_clip,
        primary=primary,
        interp=interp,
    )
    tex = build_latex(
        top=top,
        all_rows=rows,
        warnings=warnings,
        totals_nf=totals_nf,
        totals_clip=totals_clip,
        primary=primary,
        interp=interp,
    )

    with open(out_dir / "v13_formula_report.md", "w", encoding="utf-8") as f:
        f.write(md)
    with open(out_dir / "v13_formula_report.tex", "w", encoding="utf-8") as f:
        f.write(tex)

    with open(out_dir / "v13_top_candidates.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": {
                    "spec_nonfinite_count_total_sum": totals_nf,
                    "spec_clip_count_total_sum": totals_clip,
                    "n_loaded": len(rows),
                    "primary_source": primary.get("source_json") if primary else None,
                    "primary_best_loss": primary.get("best_loss") if primary else None,
                    "warnings": warnings,
                },
                "top_candidates": [json_safe_row(r) for r in top],
            },
            f,
            indent=2,
            allow_nan=True,
        )

    with open(out_dir / "v13_top_candidates.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TABLE_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in top:
            row = {k: r.get(k) for k in TABLE_COLUMNS}
            bw = row.get("best_words")
            if isinstance(bw, list):
                row["best_words"] = json.dumps(bw)
            bl = row.get("best_lengths")
            if isinstance(bl, list):
                row["best_lengths"] = json.dumps(bl)
            w.writerow(row)

    print(f"Wrote {out_dir / 'v13_formula_report.md'}")
    print(f"Wrote {out_dir / 'v13_formula_report.tex'}")
    print(f"Wrote {out_dir / 'v13_top_candidates.csv'}")
    print(f"Wrote {out_dir / 'v13_top_candidates.json'}")
    if warnings:
        for x in warnings:
            print(f"[warning] {x}")


if __name__ == "__main__":
    main()
