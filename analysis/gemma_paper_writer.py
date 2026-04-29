#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.llm_runner import LLMRunner


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _read_text(path: Path) -> Optional[str]:
    try:
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def _load_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def _to_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _trend(values: List[float]) -> str:
    n = len(values)
    if n < 2:
        return "flat"
    xs = list(range(n))
    x_mean = sum(xs) / n
    y_mean = sum(values) / n
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, values))
    den = sum((x - x_mean) ** 2 for x in xs)
    slope = num / den if den else 0.0
    if slope > 0:
        return "increasing"
    if slope < 0:
        return "decreasing"
    return "flat"


def _git_hash(repo_root: Path) -> Optional[str]:
    try:
        p = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            check=False,
        )
        if p.returncode != 0:
            return None
        out = (p.stdout or "").strip()
        return out or None
    except Exception:
        return None


def _metrics_summary(root: Path) -> Dict[str, Any]:
    lab_md = _read_text(root / "runs/lab_journal.md")
    lab_jsonl = _read_text(root / "runs/lab_journal.jsonl")
    gemma_analysis = _read_json(root / "runs/gemma_analysis.json") or {}
    operator_symbolic = _read_json(root / "runs/operator_symbolic.json") or {}
    operator_stability = _read_json(root / "runs/operator_stability_report.json") or {}
    selberg = _read_json(root / "runs/selberg_trace_report.json") or {}
    config_used = _read_text(root / "runs/v12_config_used.yaml")

    aco_rows = _load_csv(root / "runs/artin_aco_history.csv")
    rl_rows = _load_csv(root / "runs/artin_rl/train_history.csv")

    aco_best = [v for v in (_to_float(r.get("best_loss")) for r in aco_rows) if v is not None]
    rl_reward = [v for v in (_to_float(r.get("mean_reward")) for r in rl_rows) if v is not None]

    aco_best_last = aco_best[-1] if aco_best else None
    rl_reward_last = rl_reward[-1] if rl_reward else None
    aco_trend = _trend(aco_best)
    rl_trend = _trend(rl_reward)

    spectral_loss = _to_float(operator_stability.get("spectral_loss"))
    spacing_loss = _to_float(operator_stability.get("spacing_loss"))
    eig_success = bool(
        operator_stability.get("eigh_success", operator_stability.get("eigh_after", {}).get("eigh_success", False))
    )
    has_instability = bool(
        int(operator_stability.get("nan_count", 0) or 0) > 0
        or int(operator_stability.get("inf_count", 0) or 0) > 0
        or not eig_success
    )

    no_improvement = aco_trend != "decreasing"
    preliminary = (aco_best_last is not None and aco_best_last > 1e4) or (spectral_loss is not None and spectral_loss > 1e4)

    return {
        "journal_summary": {
            "lab_journal_md_excerpt": (lab_md[-2000:] if lab_md else None),
            "lab_journal_jsonl_last": (lab_jsonl.strip().splitlines()[-1] if lab_jsonl and lab_jsonl.strip() else None),
        },
        "metrics_summary": {
            "aco_best_loss_last": aco_best_last,
            "aco_best_loss_trend": aco_trend,
            "rl_mean_reward_last": rl_reward_last,
            "rl_mean_reward_trend": rl_trend,
            "spectral_loss": spectral_loss,
            "spacing_loss": spacing_loss,
            "eigensolver_succeeded": eig_success,
            "instability_flag": has_instability,
            "selberg_loss": _to_float(selberg.get("loss")),
            "selberg_relative_error": _to_float(selberg.get("relative_error")),
            "gemma_analysis_main_issue": gemma_analysis.get("main_issue"),
            "operator_symbolic_available": bool(operator_symbolic),
            "quality_flags": {
                "results_preliminary": bool(preliminary),
                "method_not_converged": bool(no_improvement),
            },
        },
        "config_excerpt": (config_used[:1500] if config_used else None),
        "strict_flags": {
            "aco_not_learning": aco_trend == "increasing",
            "alignment_poor": (spectral_loss is not None and spectral_loss > 1e4),
            "instability": has_instability,
            "results_preliminary": bool(preliminary),
            "method_not_converged": bool(no_improvement),
        },
    }


def _section_prompt(section_name: str, payload: Dict[str, Any]) -> str:
    return (
        "You are a researcher writing a NeurIPS-style paper.\n\n"
        "Project:\n"
        "Ant-RH: operator discovery for zeta zeros.\n\n"
        "Use these experiment logs:\n"
        f"{json.dumps(payload.get('journal_summary', {}), indent=2)}\n\n"
        "Metrics:\n"
        f"{json.dumps(payload.get('metrics_summary', {}), indent=2)}\n\n"
        "Write:\n"
        "- clear scientific text\n"
        "- no hype\n"
        "- precise claims\n"
        "- identify limitations\n\n"
        "IMPORTANT:\n"
        "If results are weak, explicitly say so.\n"
        "If ACO is not learning, write it explicitly.\n"
        "If spectral_loss is large, write: alignment is poor.\n"
        "If instability exists, highlight it.\n"
        "No hallucinated success.\n\n"
        f"Generate only the section body for: {section_name}\n"
        "Keep it concise (1-4 short paragraphs).\n"
    )


def _llm_write_section(section_name: str, payload: Dict[str, Any], llama_cli: str, model_path: str) -> Optional[str]:
    try:
        runner = LLMRunner(
            model_path=str(model_path),
            llama_cli=str(llama_cli),
            n_ctx=4096,
            n_threads=6,
            n_gpu_layers=0,
            timeout_s=60.0,
        )
        text = runner.generate(_section_prompt(section_name, payload), max_tokens=600, temperature=0.2)
        out = (text or "").strip()
        return out if out else None
    except Exception:
        return None


def _rule_based_section(section_name: str, payload: Dict[str, Any]) -> str:
    m = payload["metrics_summary"]
    flags = payload["strict_flags"]
    aco_line = (
        "ACO is not learning in this run."
        if flags["aco_not_learning"]
        else "ACO shows a non-increasing loss trend in this run."
    )
    align_line = "Operator/spectrum alignment is poor." if flags["alignment_poor"] else "Observed operator/spectrum mismatch is moderate."
    instability_line = (
        "Instability is present in the eigensolver/stability checks."
        if flags["instability"]
        else "Stability checks pass under the current eigensolver configuration."
    )
    prelim_line = "Results are preliminary." if flags["results_preliminary"] else "Results remain exploratory."
    conv_line = "Method not converged." if flags["method_not_converged"] else "Method shows partial convergence signals."

    if section_name == "Title":
        return "Ant-RH: Exploratory Operator Discovery for Zeta Zeros with ACO and RL"
    if section_name == "Abstract":
        return (
            "We study operator discovery for zeta zeros using Artin billiard-derived candidates, "
            "ACO/RL search, and spectral matching objectives. "
            f"{aco_line} {align_line} {instability_line} "
            f"Current measurements include spectral_loss={m.get('spectral_loss')} and "
            f"ACO best_loss={m.get('aco_best_loss_last')}. {prelim_line} {conv_line}"
        )
    if section_name == "Introduction":
        return (
            "The objective is to discover operators whose spectra align with zeta-zero statistics. "
            "This setting combines symbolic geodesic structure with numerical optimization and reinforcement learning. "
            "The present draft emphasizes empirical honesty over performance claims."
        )
    if section_name == "Method":
        return (
            "Candidates are generated from an Artin billiard-inspired symbolic process. "
            "ACO and RL explore word-level structures under losses tied to Selberg trace and spectral criteria. "
            "A structured operator and eigensolver stability checks are used before evaluating spectral mismatch."
        )
    if section_name == "Experiments":
        return (
            "Experiments use the V12 Hydra configuration with staged execution (artin, selberg, operator, aco, rl, stability, analysis). "
            "We report ACO loss trajectories, RL reward trends, stability diagnostics, and Selberg relative error."
        )
    if section_name == "Results":
        return (
            f"{aco_line} RL mean reward trend is {m.get('rl_mean_reward_trend')}. "
            f"Spectral loss is {m.get('spectral_loss')} and spacing loss is {m.get('spacing_loss')}. "
            f"{instability_line} {prelim_line} {conv_line}"
        )
    if section_name == "Discussion":
        return (
            "Current bottlenecks are optimization quality and objective alignment. "
            f"{align_line} The mismatch between improving RL reward and weak ACO convergence suggests objective misalignment. "
            "Additional ablations are needed before scaling."
        )
    if section_name == "Conclusion":
        return (
            "This run provides a reproducible but weak baseline for Ant-RH operator discovery. "
            f"{prelim_line} {conv_line} Future work should prioritize objective redesign, stronger convergence diagnostics, and controlled scale-up."
        )
    return "Section unavailable."


def _enforce_rules(section_name: str, text: str, payload: Dict[str, Any]) -> str:
    out = (text or "").strip()
    flags = payload["strict_flags"]
    if section_name in {"Abstract", "Results", "Discussion", "Conclusion"}:
        if flags["aco_not_learning"] and "ACO is not learning" not in out:
            out += "\n\nACO is not learning in this run."
        if flags["alignment_poor"] and "alignment is poor" not in out.lower():
            out += "\n\nOperator/spectrum alignment is poor."
        if flags["instability"] and "instability" not in out.lower():
            out += "\n\nInstability is present and must be addressed."
        if flags["results_preliminary"] and "preliminary" not in out.lower():
            out += "\n\nResults are preliminary."
        if flags["method_not_converged"] and "not converged" not in out.lower():
            out += "\n\nMethod not converged."
    return out.strip()


def _append_or_init_section(path: Path, section_title: str, section_text: str, run_tag: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    update_block = f"### Update {run_tag}\n\n{section_text.strip()}\n"
    if path.exists():
        with path.open("a", encoding="utf-8") as f:
            f.write("\n" + update_block)
    else:
        with path.open("w", encoding="utf-8") as f:
            f.write(f"# {section_title}\n\n{update_block}")


def _compose_markdown(sections: Dict[str, str]) -> str:
    order = [
        ("Title", "title"),
        ("Abstract", "abstract"),
        ("Introduction", "introduction"),
        ("Method", "method"),
        ("Experiments", "experiments"),
        ("Results", "results"),
        ("Discussion", "discussion"),
        ("Conclusion", "conclusion"),
    ]
    title_block = (sections.get("title", "") or "").strip()
    title_lines = [ln.strip() for ln in title_block.splitlines() if ln.strip()]
    title_content = "Ant-RH Paper Draft"
    for ln in reversed(title_lines):
        if not ln.startswith("#"):
            title_content = ln
            break
    lines = [f"# {title_content}", ""]
    for heading, key in order[1:]:
        lines.append(f"## {heading}")
        lines.append("")
        body = (sections.get(key, "") or "").strip()
        body_lines = []
        for ln in body.splitlines():
            if ln.strip().startswith("# "):
                continue
            body_lines.append(ln)
        lines.append("\n".join(body_lines).strip())
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _escape_tex(s: str) -> str:
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    out = s
    for k, v in repl.items():
        out = out.replace(k, v)
    return out


def _md_section_to_tex(md: str) -> str:
    lines = []
    for raw in (md or "").splitlines():
        line = raw.strip()
        if not line:
            lines.append("")
            continue
        if line.startswith("### "):
            lines.append(r"\paragraph{" + _escape_tex(line[4:].strip()) + "}")
            continue
        if line.startswith("## "):
            lines.append(r"\subsection{" + _escape_tex(line[3:].strip()) + "}")
            continue
        if line.startswith("# "):
            lines.append(r"\section{" + _escape_tex(line[2:].strip()) + "}")
            continue
        lines.append(_escape_tex(line))
    return "\n".join(lines)


def _compose_tex(sections: Dict[str, str]) -> str:
    title_block = (sections.get("title", "") or "").strip()
    title_lines = [ln.strip() for ln in title_block.splitlines() if ln.strip()]
    title = "Ant-RH Draft"
    for ln in reversed(title_lines):
        if not ln.startswith("#"):
            title = ln
            break
    return (
        r"\documentclass{article}" "\n"
        r"\usepackage[margin=1in]{geometry}" "\n"
        r"\usepackage[T1]{fontenc}" "\n"
        r"\begin{document}" "\n"
        + r"\title{" + _escape_tex(title) + "}\n"
        + r"\author{Ant-RH}" "\n"
        + r"\date{\today}" "\n"
        + r"\maketitle" "\n\n"
        + r"\section*{Abstract}" "\n"
        + _md_section_to_tex(sections.get("abstract", ""))
        + "\n\n"
        + r"\section{Introduction}" "\n"
        + _md_section_to_tex(sections.get("introduction", ""))
        + "\n\n"
        + r"\section{Method}" "\n"
        + _md_section_to_tex(sections.get("method", ""))
        + "\n\n"
        + r"\section{Experiments}" "\n"
        + _md_section_to_tex(sections.get("experiments", ""))
        + "\n\n"
        + r"\section{Results}" "\n"
        + _md_section_to_tex(sections.get("results", ""))
        + "\n\n"
        + r"\section{Discussion}" "\n"
        + _md_section_to_tex(sections.get("discussion", ""))
        + "\n\n"
        + r"\section{Conclusion}" "\n"
        + _md_section_to_tex(sections.get("conclusion", ""))
        + "\n\n"
        + r"\end{document}" "\n"
    )


def _read_section_file(path: Path) -> str:
    txt = _read_text(path)
    return txt if txt is not None else ""


def _strip_section_title_line(text: str) -> str:
    lines = (text or "").splitlines()
    out: List[str] = []
    skipped = False
    for ln in lines:
        if not skipped and ln.strip().startswith("# "):
            skipped = True
            continue
        out.append(ln)
    return "\n".join(out).strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="Gemma paper writer for Ant-RH")
    ap.add_argument("--backend", type=str, default="llama_cpp", choices=["llama_cpp", "rule_based"])
    ap.add_argument("--llama_cli", type=str, default="llama-cli")
    ap.add_argument(
        "--model_path",
        type=str,
        default="/Users/machome/models/gemma/gemma-3-1b-it-Q5_K_M.gguf",
    )
    ap.add_argument("--out_md", type=str, default="runs/paper_draft.md")
    ap.add_argument("--out_tex", type=str, default="runs/paper_draft.tex")
    ap.add_argument("--sections_dir", type=str, default="runs/paper_sections")
    args = ap.parse_args()

    root = Path(ROOT)
    payload = _metrics_summary(root)
    commit = _git_hash(root)
    now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    run_tag = f"{now} | {commit or 'no-commit'}"

    section_defs = [
        ("Title", "title.md", "title"),
        ("Abstract", "abstract.md", "abstract"),
        ("Introduction", "introduction.md", "introduction"),
        ("Method", "method.md", "method"),
        ("Experiments", "experiments.md", "experiments"),
        ("Results", "results.md", "results"),
        ("Discussion", "discussion.md", "discussion"),
        ("Conclusion", "conclusion.md", "conclusion"),
    ]

    sections_path = root / args.sections_dir
    sections_path.mkdir(parents=True, exist_ok=True)

    for section_name, filename, _ in section_defs:
        txt = None
        if args.backend == "llama_cpp":
            txt = _llm_write_section(section_name, payload, args.llama_cli, args.model_path)
        if not txt:
            txt = _rule_based_section(section_name, payload)
        txt = _enforce_rules(section_name, txt, payload)
        _append_or_init_section(sections_path / filename, section_name, txt, run_tag)

    merged: Dict[str, str] = {}
    for _, filename, key in section_defs:
        merged[key] = _strip_section_title_line(_read_section_file(sections_path / filename))

    md = _compose_markdown(merged)
    tex = _compose_tex(merged)

    out_md = root / args.out_md
    out_tex = root / args.out_tex
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(md, encoding="utf-8")
    out_tex.write_text(tex, encoding="utf-8")


if __name__ == "__main__":
    main()

