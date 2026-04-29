#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

import sys

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.llm_runner import LLMRunner


IGNORE_DIRS = {".git", "__pycache__", ".venv", "node_modules"}
IGNORE_SUFFIXES = {".npy", ".pt", ".gguf", ".png", ".pdf"}
DT = float
PROJECT_GLOSSARY = (
    "Project glossary:\n"
    "DTES = Deformable Tropical Energy Space.\n"
    "Ant-RH = research pipeline for zeta-zero / Hilbert-Polya style operator discovery.\n"
    "Artin billiard = symbolic/geometric layer using PSL(2,Z) words.\n"
    "ACO = ant colony optimization over Artin words/geodesics.\n"
    "RL = reinforcement learning policy over symbolic/geometric actions.\n"
    "Structured operator = H = sum_i w_i B_i.\n"
    "Gemma Planner = local LLM proposing Artin words.\n"
    "Gemma Analyzer = local LLM summarizing metrics/logs.\n"
    "Gemma Help = interactive local assistant.\n"
    "Do not reinterpret DTES as dynamic time warping. Use the glossary above.\n"
)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def fit_prompt(text, max_chars):
    text = str(text or "")
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n\n...[TRUNCATED]...\n\n" + text[-half:]


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _summarize_text_lines(text: str, head: int = 20, tail: int = 20) -> Dict[str, Any]:
    lines = text.splitlines()
    return {
        "line_count": len(lines),
        "head": lines[:head],
        "tail": lines[-tail:] if lines else [],
    }


def _summarize_csv(path: Path, head: int = 10, tail: int = 10) -> Dict[str, Any]:
    text = _read_text(path)
    base = _summarize_text_lines(text, head=head, tail=tail)
    stats: Dict[str, Any] = {"numeric_columns": {}}
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        numeric: Dict[str, List[float]] = {}
        for row in rows:
            for k, v in row.items():
                fv = _safe_float(v)
                if fv is None:
                    continue
                numeric.setdefault(k, []).append(fv)
        for k, vals in numeric.items():
            if not vals:
                continue
            stats["numeric_columns"][k] = {
                "count": len(vals),
                "min": min(vals),
                "max": max(vals),
                "last": vals[-1],
                "mean": sum(vals) / len(vals),
            }
    except Exception:
        pass
    base.update(stats)
    return base


def _summarize_json_file(path: Path, max_chars: int = 2500) -> Dict[str, Any]:
    obj = _read_json(path)
    if obj is None:
        txt = fit_prompt(_read_text(path), max_chars)
        return _summarize_text_lines(txt)
    keys = list(obj.keys())[:50]
    compact = {k: obj[k] for k in keys[:20]}
    preview = fit_prompt(json.dumps(compact, indent=2), max_chars)
    return {"keys": keys, "preview": preview}


def _python_file_summary(path: Path, max_chars: int = 2500) -> Dict[str, Any]:
    text = _read_text(path)
    lines = text.splitlines()
    defs = [ln.strip() for ln in lines if ln.strip().startswith(("def ", "class "))]
    imports = [ln.strip() for ln in lines if ln.strip().startswith(("import ", "from "))]
    has_main = "if __name__ == \"__main__\":" in text
    excerpt_lines = lines[:120] + (["...[TRUNCATED]..."] if len(lines) > 200 else []) + lines[-80:]
    excerpt = fit_prompt("\n".join(excerpt_lines), max_chars)
    return {
        "path": path.as_posix(),
        "line_count": len(lines),
        "imports": imports[:20],
        "symbols": defs[:30],
        "has_main": has_main,
        "excerpt": excerpt,
    }


def _yaml_file_summary(path: Path, max_chars: int = 2500) -> Dict[str, Any]:
    text = _read_text(path)
    lines = text.splitlines()
    keys = []
    for ln in lines:
        stripped = ln.strip()
        if stripped.endswith(":") and not stripped.startswith("-"):
            keys.append(stripped[:-1])
    return {"path": path.as_posix(), "keys": keys[:50], "excerpt": fit_prompt("\n".join(lines[:120]), max_chars)}


def _collect_files(root: Path, folder: str, pattern: str) -> List[Path]:
    target = root / folder
    if not target.exists():
        return []
    return sorted(target.glob(pattern))


def _should_ignore_file(path: Path) -> bool:
    if any(part in IGNORE_DIRS for part in path.parts):
        return True
    if path.suffix in IGNORE_SUFFIXES:
        return True
    try:
        size = path.stat().st_size
    except Exception:
        size = 0
    if path.suffix == ".csv" and size > 1024 * 1024:
        return True
    if path.suffix == ".log" and size > 1024 * 1024:
        return True
    return False


def _filtered_tree(root: Path) -> List[str]:
    out: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
        rel_dir = Path(dirpath).relative_to(root)
        for name in sorted(filenames):
            p = rel_dir / name
            if any(name.endswith(s) for s in IGNORE_SUFFIXES):
                continue
            if _should_ignore_file(root / p):
                continue
            out.append(p.as_posix())
    return out


def _folder_summary(root: Path, folder: str, patterns: Sequence[str], max_file_chars: int, max_files: int) -> Dict[str, Any]:
    files: List[Path] = []
    for pat in patterns:
        files.extend(_collect_files(root, folder, pat))
    files = sorted({p for p in files if not _should_ignore_file(p)})[: int(max_files)]
    items = []
    for path in files:
        if path.suffix == ".py":
            items.append(_python_file_summary(path, max_chars=max_file_chars))
        elif path.suffix in {".yaml", ".yml"}:
            items.append(_yaml_file_summary(path, max_chars=max_file_chars))
        elif path.suffix == ".json":
            items.append({"path": path.as_posix(), "summary": _summarize_json_file(path, max_chars=max_file_chars)})
        elif path.suffix == ".csv":
            items.append({"path": path.as_posix(), "summary": _summarize_csv(path)})
        else:
            text = fit_prompt(_read_text(path), max_file_chars)
            items.append({"path": path.as_posix(), "summary": _summarize_text_lines(text)})
    return {"folder": folder, "count": len(items), "items": items}


def _makefile_summary(root: Path) -> Dict[str, Any]:
    path = root / "Makefile"
    text = _read_text(path)
    targets = []
    for line in text.splitlines():
        if line and not line.startswith(("\t", "#", ".")) and ":" in line:
            name = line.split(":", 1)[0].strip()
            if name:
                targets.append(name)
    return {"path": path.as_posix(), "targets": targets}


def _readme_summary(root: Path, max_chars: int) -> Dict[str, Any]:
    path = root / "README.md"
    text = fit_prompt(_read_text(path), max_chars)
    return {"path": path.as_posix(), **_summarize_text_lines(text, head=40, tail=20)}


def _runs_summary(root: Path, max_file_chars: int, max_files: int) -> Dict[str, Any]:
    runs = root / "runs"
    if not runs.exists():
        return {"count": 0, "items": []}
    items = []
    for path in sorted(runs.glob("*")):
        if path.is_dir():
            continue
        if _should_ignore_file(path):
            continue
        if path.suffix == ".json":
            items.append({"path": path.as_posix(), "summary": _summarize_json_file(path, max_chars=max_file_chars)})
        elif path.suffix == ".csv":
            items.append({"path": path.as_posix(), "summary": _summarize_csv(path)})
    return {"count": len(items), "items": items[: int(max_files)]}


def _logs_summary(root: Path, max_files: int) -> Dict[str, Any]:
    logs = root / "logs/v12"
    if not logs.exists():
        return {"count": 0, "items": []}
    items = []
    for path in sorted(logs.glob("*.log"))[: int(max_files)]:
        if _should_ignore_file(path):
            continue
        items.append({"path": path.as_posix(), "summary": _summarize_text_lines(_read_text(path), head=0, tail=80)})
    return {"count": len(items), "items": items}


def _prompt_for_folder(folder_summary: Dict[str, Any]) -> str:
    return (
        "You are summarizing a code folder for Ant-RH.\n"
        + PROJECT_GLOSSARY
        + "\nReturn concise markdown bullets describing its main modules, entrypoints, and outputs.\n\n"
        f"{json.dumps(folder_summary, indent=2)}"
    )


def _prompt_global(index: Dict[str, Any], folder_notes: Dict[str, str]) -> str:
    payload = {
        "tree_sample": index.get("tree", [])[:120],
        "makefile": index.get("makefile"),
        "makefile_targets": index.get("makefile", {}).get("targets", []),
        "readme": index.get("readme"),
        "folder_notes": folder_notes,
        "runs_summary": {"count": index.get("runs_summary", {}).get("count"), "items": index.get("runs_summary", {}).get("items", [])[:20]},
        "logs_summary": {"count": index.get("logs_summary", {}).get("count"), "items": index.get("logs_summary", {}).get("items", [])[:10]},
    }
    return (
        "You are studying the Ant-RH codebase.\n"
        + PROJECT_GLOSSARY
        + "\nWrite concise markdown with sections:\n"
        "1. Project purpose\n"
        "2. Main pipeline\n"
        "3. Core modules\n"
        "4. Validation modules\n"
        "5. Analysis agents\n"
        "6. Help/voice interface\n"
        "7. Makefile commands\n"
        "8. Current bottlenecks\n"
        "9. Next recommended actions\n"
        "Be honest and practical.\n"
        "Extract Makefile targets explicitly.\n\n"
        f"{json.dumps(payload, indent=2)}"
    )


def _rule_based_global(index: Dict[str, Any], folder_notes: Dict[str, str]) -> Dict[str, str]:
    make_targets = index.get("makefile", {}).get("targets", [])
    summary = (
        "# Project Summary\n\n"
        "## Project Purpose\n"
        "Ant-RH is a zeta-zero / Hilbert-Polya style operator-discovery pipeline built around DTES "
        "(Deformable Tropical Energy Space), Artin symbolic geometry, operator construction, and local Gemma agents.\n\n"
        "## Main Pipeline\n"
        "- DTES/artin symbolic generation -> Selberg-style validation -> operator construction -> ACO -> RL -> stability -> analysis/help.\n\n"
        "## Core Modules\n"
        "- `core/`: Artin billiard, ACO, RL environment/policy, operator builders, stabilization, PDE feature library.\n\n"
        "## Validation Modules\n"
        "- `validation/`: Selberg loss, RL training, operator stability, experiment trainers and reconstructions.\n\n"
        "## Analysis Agents\n"
        "- `analysis/`: Gemma Analyzer, lab journal, paper writer, PDE discovery, project study.\n\n"
        "## Help/Voice Interface\n"
        "- `help/`: Gemma Help assistant, local macOS TTS, pyttsx3 TTS.\n\n"
        "## Makefile Commands\n"
        + "\n".join(f"- `{t}`" for t in make_targets[:30])
        + "\n\n## Current Bottlenecks\n"
        "- ACO learning remains weak in current runs.\n"
        "- Operator alignment/scaling remains a recurring issue.\n"
        "- Reporting/help stack is growing faster than consolidated project memory.\n\n"
        "## Next Recommended Actions\n"
        "- Run `make study` after each major change to refresh project memory.\n"
        "- Keep `make analyze-gemma`, `make lab-journal`, and `make pde` in the post-run loop.\n"
        "- Use the refreshed memory to tighten help-agent answers and pipeline triage.\n"
    )
    project_map = (
        "# Project Map\n\n"
        "## Entrypoints\n"
        + "\n".join(f"- `{t}`" for t in make_targets[:40])
        + "\n\n## Folder Notes\n"
        + "\n\n".join([f"### `{k}`\n{v}" for k, v in folder_notes.items()])
        + "\n"
    )
    memory = (
        "# Gemma Project Memory\n\n"
        "- Project: Ant-RH\n"
        "- Core loop: symbolic Artin features -> operator -> ACO/RL -> stability -> analysis\n"
        "- Key commands: `make run-v12`, `make analyze-gemma`, `make lab-journal`, `make paper`, `make pde`, `make help-chat`, `make study`\n"
        "- Current bottlenecks: ACO non-learning, operator alignment/scaling, fragmented experiment outputs\n"
        "- Important outputs live in `runs/` and `logs/v12/`\n"
    )
    return {"project_summary": summary, "project_map": project_map, "project_memory": memory}


def _llm_generate(runner: LLMRunner, prompt: str, max_tokens: int = 500) -> str:
    try:
        return runner.generate(prompt, max_tokens=max_tokens, temperature=0.2).strip()
    except Exception:
        return ""


def main() -> None:
    ap = argparse.ArgumentParser(description="Gemma project study agent for Ant-RH")
    ap.add_argument("--root", type=str, default=".")
    ap.add_argument("--backend", type=str, default="llama_cpp", choices=["llama_cpp", "rule_based"])
    ap.add_argument("--llama_cli", type=str, default="llama-cli")
    ap.add_argument("--model_path", type=str, default="/Users/machome/models/gemma/gemma-3-1b-it-Q5_K_M.gguf")
    ap.add_argument("--max_prompt_chars", type=int, default=9000)
    ap.add_argument("--max_file_chars", type=int, default=2500)
    ap.add_argument("--max_chunk_chars", type=int, default=6000)
    ap.add_argument("--max_files", type=int, default=80)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    runner = LLMRunner(
        model_path=str(args.model_path),
        llama_cli=str(args.llama_cli),
        n_ctx=4096,
        n_threads=6,
        n_gpu_layers=0,
        timeout_s=60.0,
    )

    folder_specs = {
        "core": ["*.py"],
        "validation": ["*.py"],
        "analysis": ["*.py"],
        "help": ["*.py"],
        "configs": ["*.yaml"],
    }

    folder_summaries: Dict[str, Any] = {}
    folder_notes: Dict[str, str] = {}
    for folder, patterns in folder_specs.items():
        summary = _folder_summary(
            root,
            folder,
            patterns,
            max_file_chars=int(args.max_file_chars),
            max_files=int(args.max_files),
        )
        folder_summaries[folder] = summary
        note = ""
        if args.backend == "llama_cpp":
            prompt = fit_prompt(_prompt_for_folder(summary), int(args.max_prompt_chars))
            note = _llm_generate(runner, prompt, max_tokens=350)
        if not note:
            note = f"- `{folder}` contains {summary['count']} summarized files.\n"
        note = fit_prompt(note, int(args.max_chunk_chars))
        folder_notes[folder] = note

    index = {
        "root": root.as_posix(),
        "tree": _filtered_tree(root),
        "makefile": _makefile_summary(root),
        "readme": _readme_summary(root, int(args.max_file_chars)) if (root / "README.md").exists() else {},
        "folders": folder_summaries,
        "folder_notes": folder_notes,
        "runs_summary": _runs_summary(root, int(args.max_file_chars), int(args.max_files)),
        "logs_summary": _logs_summary(root, int(args.max_files)),
    }

    project_summary = ""
    project_map = ""
    project_memory = ""
    if args.backend == "llama_cpp":
        global_prompt = fit_prompt(_prompt_global(index, folder_notes), int(args.max_prompt_chars))
        global_md = _llm_generate(runner, global_prompt, max_tokens=900)
        if global_md:
            global_md = fit_prompt(global_md, int(args.max_chunk_chars))
            project_summary = "# Project Summary\n\n" + global_md
            project_map = (
                "# Project Map\n\n## Folders\n\n"
                + "\n\n".join([f"### `{k}`\n{fit_prompt(v, int(args.max_chunk_chars))}" for k, v in folder_notes.items()])
                + "\n"
            )
            project_memory = (
                "# Gemma Project Memory\n\n"
                + "Use this as compact project context for future help answers.\n\n"
                + "\n".join(
                    [
                        "- Pipeline: artin -> selberg -> operator -> aco -> rl -> stability -> analysis/reporting/help.",
                        f"- Make targets: {', '.join(index['makefile'].get('targets', [])[:20])}",
                        "- Use `runs/` for experiment outputs and `logs/v12/` for stage logs.",
                    ]
                )
                + "\n\n"
                + fit_prompt(global_md, min(3000, int(args.max_chunk_chars)))
            )
    if not project_summary or not project_map or not project_memory:
        rb = _rule_based_global(index, folder_notes)
        project_summary = project_summary or rb["project_summary"]
        project_map = project_map or rb["project_map"]
        project_memory = project_memory or rb["project_memory"]

    _write_json(root / "runs/project_index.json", index)
    _write_text(root / "runs/project_summary.md", project_summary)
    _write_text(root / "runs/project_map.md", project_map)
    _write_text(root / "runs/gemma_project_memory.md", project_memory)


if __name__ == "__main__":
    main()

