#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.llm_runner import LLMRunner


GLOSSARY = (
    "Glossary:\n"
    "DTES = Deformable Tropical Energy Space.\n"
    "Ant-RH = ACO/RL/operator-search pipeline for zeta-zero / Hilbert-Polya style experiments.\n"
    "Artin billiard = symbolic/geometric layer based on PSL(2,Z).\n"
    "Selberg trace = spectral-geometric consistency constraint.\n"
    "Topological LLM = next-token model over braid/DTES DSL.\n"
    "Gemma = local assistant/planner/analyzer, not proof engine.\n"
    "Do not reinterpret DTES as Dynamic Time Warping.\n"
)


def _fit_text(text: str, max_chars: int) -> str:
    text = str(text or "")
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n\n...[TRUNCATED]...\n\n" + text[-half:]


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _extract_make_targets(makefile_text: str) -> List[str]:
    targets: List[str] = []
    for line in makefile_text.splitlines():
        if not line or line.startswith(("\t", "#", ".")):
            continue
        if ":" not in line:
            continue
        name = line.split(":", 1)[0].strip()
        if name and " " not in name and name not in targets:
            targets.append(name)
    return targets


def _extract_commands(makefile_text: str, wanted: List[str]) -> Dict[str, str]:
    lines = makefile_text.splitlines()
    commands: Dict[str, str] = {}
    for idx, line in enumerate(lines):
        if ":" not in line or line.startswith(("\t", "#", ".")):
            continue
        name = line.split(":", 1)[0].strip()
        if name not in wanted:
            continue
        parts: List[str] = []
        j = idx + 1
        while j < len(lines) and lines[j].startswith("\t"):
            parts.append(lines[j].lstrip())
            j += 1
        commands[name] = "\n".join(parts).strip()
    return commands


def _module_summary(path: Path, max_lines: int = 120) -> Dict[str, Any]:
    text = _read_text(path)
    lines = text.splitlines()
    defs = [ln.strip() for ln in lines if ln.strip().startswith(("def ", "class "))]
    imports = [ln.strip() for ln in lines if ln.strip().startswith(("import ", "from "))]
    first_doc = ""
    m = re.search(r'"""(.*?)"""', text, re.S)
    if m:
        first_doc = " ".join(m.group(1).strip().split())
    return {
        "path": path.as_posix(),
        "symbols": defs[:12],
        "imports": imports[:10],
        "doc": first_doc,
        "excerpt": _fit_text("\n".join(lines[:max_lines]), 3500),
    }


def _llm_polish(runner: LLMRunner | None, prompt: str, max_prompt_chars: int, max_tokens: int = 700) -> str:
    if runner is None:
        return ""
    try:
        return runner.generate(_fit_text(prompt, max_prompt_chars), max_tokens=max_tokens, temperature=0.2).strip()
    except Exception:
        return ""


def _build_inputs(root: Path) -> Dict[str, Any]:
    makefile_text = _read_text(root / "Makefile")
    project_memory = _read_text(root / "runs/gemma_project_memory.md")
    project_summary = _read_text(root / "runs/project_summary.md")
    lab_journal = _read_text(root / "runs/lab_journal.md")
    topo_report = _read_text(root / "runs/topological_lm/report.md")
    pde_report = _read_text(root / "runs/operator_pde_report.md")
    gemma_analysis = _read_json(root / "runs/gemma_analysis.json")
    core_files = sorted((root / "core").glob("*.py"))
    analysis_files = sorted((root / "analysis").glob("*.py"))
    interesting_targets = [
        "run-v12",
        "smoke-v12",
        "artin",
        "selberg",
        "operator",
        "aco",
        "aco-gemma",
        "rl",
        "stability",
        "analyze-gemma",
        "lab-journal",
        "paper",
        "pde",
        "sensitivity",
        "topo-dataset",
        "topo-train",
        "topo-eval",
        "topo-report",
        "topo-all",
        "literature",
        "study",
        "refresh-help",
        "help-chat",
    ]
    return {
        "project_memory": project_memory,
        "project_summary": project_summary,
        "lab_journal": lab_journal,
        "topological_report": topo_report,
        "pde_report": pde_report,
        "gemma_analysis": gemma_analysis,
        "make_targets": _extract_make_targets(makefile_text),
        "commands": _extract_commands(makefile_text, interesting_targets),
        "core_modules": [_module_summary(path) for path in core_files],
        "analysis_modules": [_module_summary(path) for path in analysis_files],
    }


def _current_state_lines(inputs: Dict[str, Any]) -> List[str]:
    ga = inputs.get("gemma_analysis", {})
    aco = ga.get("aco", {})
    rl = ga.get("rl", {})
    operator = ga.get("operator", {})
    lines = [
        f"- Main issue: `{ga.get('main_issue', 'unknown')}`.",
        f"- ACO best-loss trend: `{aco.get('best_loss_trend', 'unknown')}`.",
        f"- RL mean-reward trend: `{rl.get('mean_reward_trend', 'unknown')}`.",
        f"- Operator spectral loss: `{operator.get('spectral_loss', 'unknown')}`.",
        f"- Operator eigensolver success: `{operator.get('eigh_success', 'unknown')}`.",
    ]
    topo_report = str(inputs.get("topological_report", ""))
    if "not yet better than random" in topo_report.lower():
        lines.append("- TopologicalLM is not yet better than random under the current evaluator.")
    elif "mean_reward=-34999" in topo_report or "≈ random" in topo_report.lower():
        lines.append("- TopologicalLM reporting still suggests no reliable improvement over random.")
    else:
        lines.append("- TopologicalLM status should be checked against the latest `runs/topological_lm/report.md`.")
    return lines


def _write_architecture_docs(root: Path, inputs: Dict[str, Any]) -> None:
    commands = inputs["commands"]
    overview = "\n".join(
        [
            "# Architecture Overview",
            "",
            "Ant-RH is a research codebase for Hilbert-Polya style operator experiments around zeta-zero structure.",
            "",
            "## Core Pipeline",
            "- Symbolic Artin words and geodesic features are generated first.",
            "- Selberg-style and spectral diagnostics score the symbolic/geometric candidates.",
            "- Operators are built and then used by ACO and RL search loops.",
            "- Stability, PDE-discovery, TopologicalLM, and Gemma analysis stages summarize the results.",
            "",
            "## Current State",
            *_current_state_lines(inputs),
            "",
            "## Primary Entrypoints",
            f"- `make run-v12`: `{commands.get('run-v12', 'configured in Makefile')}`",
            f"- `make topo-all`: `{commands.get('topo-all', 'configured in Makefile')}`",
            f"- `make study`: `{commands.get('study', 'configured in Makefile')}`",
            "",
        ]
    )
    pipeline = "\n".join(
        [
            "# Pipeline",
            "",
            "## Data Flow",
            "1. `artin` generates symbolic words/geodesics.",
            "2. `selberg` computes spectral-geometric consistency losses.",
            "3. `operator` builds the numeric operator from symbolic inputs.",
            "4. `aco` or `aco-gemma` searches symbolic candidates.",
            "5. `rl` trains a policy with operator-based feedback.",
            "6. `stability`, `pde`, `topo-eval`, and analysis agents inspect outcomes.",
            "",
            "## Main Outputs",
            "- `runs/artin_*`: symbolic and operator artifacts.",
            "- `runs/artin_rl/`: RL history.",
            "- `runs/operator_stability_report.json`: numerical stability diagnostics.",
            "- `runs/operator_pde_report.md`: PDE-style surrogate discovery report.",
            "- `runs/topological_lm/`: TopologicalLM model, eval, and report files.",
            "",
            "## Bottlenecks",
            "- ACO is not learning in the current run.",
            "- Alignment/scaling remains the main operator issue.",
            "- TopologicalLM should be treated as exploratory until it clearly beats random.",
            "",
        ]
    )
    core_modules = inputs["core_modules"]
    analysis_modules = inputs["analysis_modules"]
    modules = ["# Modules", "", "## core/"]
    for item in core_modules:
        modules.append(f"- `{Path(item['path']).name}`: {item['doc'] or 'Module for pipeline logic or operator/search utilities.'}")
    modules.extend(["", "## analysis/"])
    for item in analysis_modules:
        modules.append(f"- `{Path(item['path']).name}`: {item['doc'] or 'Analysis or report-generation utility.'}")
    modules.extend(
        [
            "",
            "## Notes",
            "- Module descriptions are generated from lightweight summaries, not full source reprints.",
            "- Use the source files for implementation details and exact APIs.",
            "",
        ]
    )
    _write_text(root / "Docs/Architecture/overview.md", overview)
    _write_text(root / "Docs/Architecture/pipeline.md", pipeline)
    _write_text(root / "Docs/Architecture/modules.md", "\n".join(modules))


def _write_concepts_docs(root: Path) -> None:
    docs = {
        "dtes.md": (
            "# DTES\n\n"
            "DTES means Deformable Tropical Energy Space. In this repository it is a modeling language for structured geometric or spectral states, not a proof object.\n\n"
            "## Role in Ant-RH\n"
            "- Provides a representation layer for operator-search experiments.\n"
            "- Appears in operator, geometry, and TopologicalLM-related code paths.\n"
        ),
        "artin_billiard.md": (
            "# Artin Billiard\n\n"
            "Artin billiard is the symbolic/geometric layer based on PSL(2,Z) words and associated hyperbolic data.\n\n"
            "## Role in Ant-RH\n"
            "- Generates symbolic words used by operator and search modules.\n"
            "- Connects geometric structure to Selberg-style constraints.\n"
        ),
        "selberg_trace.md": (
            "# Selberg Trace\n\n"
            "Selberg trace is used here as a spectral-geometric consistency constraint. It is a diagnostic and loss source, not a guarantee of correctness.\n\n"
            "## Role in Ant-RH\n"
            "- Compares symbolic/geometric information with spectral behavior.\n"
            "- Helps detect mismatch between geodesic structure and operator outputs.\n"
        ),
        "operator_search.md": (
            "# Operator Search\n\n"
            "Operator search is the process of building and adjusting candidate operators whose spectra are compared against zeta-zero-derived targets or proxy diagnostics.\n\n"
            "## Role in Ant-RH\n"
            "- Uses symbolic inputs, spectral losses, and stability checks.\n"
            "- Feeds ACO, RL, and PDE-style analysis stages.\n"
        ),
        "topological_llm.md": (
            "# Topological LLM\n\n"
            "Topological LLM is a next-token model over a braid/DTES DSL. It is an experimental candidate generator, not a validated mathematical solver.\n\n"
            "## Role in Ant-RH\n"
            "- Generates symbolic braid-like candidates.\n"
            "- Is currently useful only insofar as it outperforms random baselines under the evaluator.\n"
        ),
    }
    for name, text in docs.items():
        _write_text(root / "Docs/Concepts" / name, text)


def _write_experiment_docs(root: Path, inputs: Dict[str, Any]) -> None:
    ga = inputs["gemma_analysis"]
    topo_report = str(inputs.get("topological_report", ""))
    docs = {
        "v12.md": (
            "# V12\n\n"
            "## What was tried\n"
            "- End-to-end Artin -> Selberg -> operator -> ACO -> RL -> stability workflow.\n\n"
            "## Results\n"
            f"- ACO best-loss trend: `{ga.get('aco', {}).get('best_loss_trend', 'unknown')}`.\n"
            f"- RL mean-reward trend: `{ga.get('rl', {}).get('mean_reward_trend', 'unknown')}`.\n"
            f"- Operator spectral loss: `{ga.get('operator', {}).get('spectral_loss', 'unknown')}`.\n\n"
            "## Limitations\n"
            "- ACO is not learning in the current run.\n"
            "- Selberg relative error remains high.\n"
        ),
        "v13.md": (
            "# V13\n\n"
            "## What was tried\n"
            "- Gemma-assisted analysis, journaling, study, and help tooling were added around the main pipeline.\n\n"
            "## Results\n"
            "- Project memory and lab journal artifacts are now available in `runs/`.\n"
            "- Help answers can be grounded in current metrics and summaries.\n\n"
            "## Limitations\n"
            "- Reporting is ahead of consolidated experimental signal.\n"
            "- Better refresh loops are still needed after each run.\n"
        ),
        "v13_2_pde.md": (
            "# V13.2 PDE\n\n"
            "## What was tried\n"
            "- Sparse, interpretable operator-equation discovery over learned operator artifacts.\n\n"
            "## Results\n"
            "- `runs/operator_pde_report.md` reports a low normalized fit error on the current surrogate task.\n"
            "- The selected equation is a compact approximation, not an exact recovered law.\n\n"
            "## Limitations\n"
            "- The regression uses reconstructed geometry and a finite feature library.\n"
            "- Terms still require held-out validation and alternative-kernel checks.\n"
        ),
        "vnext_topological_llm.md": (
            "# VNext Topological LLM\n\n"
            "## What was tried\n"
            "- A small transformer was trained over a braid/DTES DSL to generate candidate symbolic sequences.\n\n"
            "## Results\n"
            + (
                "- Current reporting says the model is not yet better than random.\n"
                if "not yet better than random" in topo_report.lower() or "mean_reward=-34999" in topo_report
                else "- Check `runs/topological_lm/report.md` for the latest evaluator diagnosis.\n"
            )
            + "\n## Limitations\n"
            "- The model is heuristic and does not prove RH.\n"
            "- Evaluation quality depends strongly on the executor reward design and dataset quality.\n"
        ),
    }
    for name, text in docs.items():
        _write_text(root / "Docs/Experiments" / name, text)


def _write_guides(root: Path, inputs: Dict[str, Any]) -> None:
    commands = inputs["commands"]
    targets = inputs["make_targets"]
    setup = "\n".join(
        [
            "# Setup",
            "",
            "## Requirements",
            "- Python 3 with `torch`, `numpy`, `scipy`, `hydra-core`, `omegaconf`, `tqdm`, and `matplotlib`.",
            "- `llama-cli` for local Gemma-backed tools.",
            "",
            "## Install",
            "```bash",
            "make install",
            "```",
            "",
            "## Check local Gemma",
            "```bash",
            "make gemma-test",
            "```",
            "",
        ]
    )
    quickstart = "\n".join(
        [
            "# Quickstart",
            "",
            "## Main pipeline",
            "```bash",
            "make run-v12",
            "make analyze-gemma",
            "make lab-journal",
            "```",
            "",
            "## Experimental branches",
            "```bash",
            "make pde",
            "make topo-all",
            "make literature",
            "```",
            "",
        ]
    )
    command_lines = ["# Commands", "", "Commands below are copied from the current `Makefile` targets."]
    for name in targets:
        if name in commands:
            command_lines.extend(["", f"## `{name}`", "```bash", commands[name], "```"])
    debugging = "\n".join(
        [
            "# Debugging",
            "",
            "## Common issues",
            "- If `llama-cli` is missing, run `make gemma-test` and install `llama.cpp`.",
            "- If ACO quality is poor, accept that current runs show non-learning and retune exploration before overinterpreting results.",
            "- If operator metrics drift, inspect `runs/operator_stability_report.json` and rerun `make stability`.",
            "- If TopologicalLM does not beat random, treat it as exploratory and inspect `make topo-eval` outputs before claiming progress.",
            "",
            "## Useful commands",
            "```bash",
            "make aco-gemma",
            "make stability",
            "make topo-eval",
            "make study",
            "```",
            "",
        ]
    )
    _write_text(root / "Docs/Guides/setup.md", setup)
    _write_text(root / "Docs/Guides/quickstart.md", quickstart)
    _write_text(root / "Docs/Guides/commands.md", "\n".join(command_lines) + "\n")
    _write_text(root / "Docs/Guides/debugging.md", debugging)


def _write_reports(root: Path, inputs: Dict[str, Any]) -> None:
    text = "\n".join(
        [
            "# Latest Summary",
            "",
            "## Current State",
            *_current_state_lines(inputs),
            "",
            "## Bottlenecks",
            "- ACO non-learning.",
            "- Operator scaling/alignment issues.",
            "- TopologicalLM still needs a clear advantage over random baselines.",
            "",
            "## Next Actions",
            "- Retune ACO exploration and rerun a short smoke test.",
            "- Keep `make analyze-gemma`, `make lab-journal`, and `make pde` in the post-run loop.",
            "- Re-run `make topo-eval` after executor or dataset changes and check whether the model beats random.",
            "",
        ]
    )
    _write_text(root / "Docs/Reports/latest_summary.md", text)


def _write_docs_readme(root: Path) -> None:
    text = "\n".join(
        [
            "# Docs",
            "",
            "## Sections",
            "- `Architecture/`: overview, pipeline, and module summaries.",
            "- `Concepts/`: concise explanations of core ideas used by Ant-RH.",
            "- `Experiments/`: short summaries of major experiment tracks.",
            "- `Guides/`: setup, commands, quickstart, and debugging notes.",
            "- `Reports/`: latest project state summaries.",
            "",
            "## Entry points",
            "- Start with `Architecture/overview.md` for the system view.",
            "- Read `Guides/quickstart.md` for the main commands.",
            "- Read `Reports/latest_summary.md` for the current status.",
            "",
        ]
    )
    _write_text(root / "Docs/README.md", text)


def _polish_key_docs(root: Path, runner: LLMRunner | None, max_prompt_chars: int) -> None:
    targets = [
        root / "Docs/Architecture/overview.md",
        root / "Docs/Reports/latest_summary.md",
        root / "Docs/README.md",
    ]
    for path in targets:
        current = _read_text(path)
        if not current:
            continue
        prompt = (
            "You are editing Ant-RH technical documentation.\n"
            + GLOSSARY
            + "\nRewrite the markdown to be concise, technical, and honest.\n"
            + "Do not invent results. Keep ACO non-learning and TopologicalLM-vs-random caveats if present.\n\n"
            + current
        )
        polished = _llm_polish(runner, prompt, max_prompt_chars=max_prompt_chars, max_tokens=900)
        if polished.startswith("#"):
            _write_text(path, polished)


def main() -> None:
    ap = argparse.ArgumentParser(description="Gemma docs builder for Ant-RH")
    ap.add_argument("--backend", type=str, default="llama_cpp", choices=["llama_cpp", "rule_based"])
    ap.add_argument("--llama_cli", type=str, default="llama-cli")
    ap.add_argument("--model_path", type=str, default="/Users/machome/models/gemma/gemma-3-1b-it-Q5_K_M.gguf")
    ap.add_argument("--max_prompt_chars", type=int, default=9000)
    args = ap.parse_args()

    root = Path(ROOT)
    inputs = _build_inputs(root)
    runner: LLMRunner | None = None
    if args.backend == "llama_cpp":
        runner = LLMRunner(
            model_path=str(args.model_path),
            llama_cli=str(args.llama_cli),
            n_ctx=4096,
            n_threads=6,
            n_gpu_layers=0,
            timeout_s=90.0,
        )

    _write_architecture_docs(root, inputs)
    _write_concepts_docs(root)
    _write_experiment_docs(root, inputs)
    _write_guides(root, inputs)
    _write_reports(root, inputs)
    _write_docs_readme(root)
    _polish_key_docs(root, runner, int(args.max_prompt_chars))


if __name__ == "__main__":
    main()
