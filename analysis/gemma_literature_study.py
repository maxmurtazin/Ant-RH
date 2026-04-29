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


def fit_prompt(text: str, max_chars: int) -> str:
    text = str(text or "")
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n\n...[TRUNCATED]...\n\n" + text[-half:]


def _read_text(path: Path, max_chars: int) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    return fit_prompt(text, max_chars)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _title_from_text(path: Path, text: str) -> str:
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("#"):
            return s.lstrip("#").strip()
        if s:
            return s
    return path.stem


def _extract_formulas(text: str) -> List[str]:
    formulas = re.findall(r"\$[^$\n]+\$|\\\[[\s\S]*?\\\]|\bH\s*=\s*[^\n]+|\bp\([^\n]+\)", text)
    out: List[str] = []
    for item in formulas:
        item = " ".join(str(item).split())
        if item and item not in out:
            out.append(item)
    return out[:12]


def _extract_keywords(text: str, keywords: List[str]) -> List[str]:
    low = text.lower()
    hits: List[str] = []
    for kw in keywords:
        if kw.lower() in low and kw not in hits:
            hits.append(kw)
    return hits


def _rule_based_file_summary(path: Path, text: str) -> Dict[str, Any]:
    keywords = [
        "Hilbert-Polya",
        "Riemann hypothesis",
        "Selberg trace",
        "Artin billiard",
        "DTES",
        "tropical geometry",
        "braid groups",
        "topological quantum computing",
        "random matrix theory",
        "quantum chaos",
        "operator theory",
        "spectral geometry",
        "PDE",
        "non-Abelian anyons",
    ]
    methods = [
        "trace formula",
        "operator construction",
        "spectral matching",
        "sparse regression",
        "ACO",
        "RL",
        "token modeling",
        "topological language model",
        "braid representation",
    ]
    title = _title_from_text(path, text)
    key_concepts = _extract_keywords(text, keywords)
    relevant_methods = _extract_keywords(text, methods)
    connection_parts: List[str] = []
    if any(k in key_concepts for k in ["Artin billiard", "Selberg trace", "Hilbert-Polya"]):
        connection_parts.append("Connects directly to the Artin/Selberg/operator side of Ant-RH.")
    if any(k in key_concepts for k in ["DTES", "tropical geometry"]):
        connection_parts.append("Relevant to DTES geometry and representation design.")
    if any(k in key_concepts for k in ["braid groups", "topological quantum computing"]):
        connection_parts.append("Relevant to braid/operator tokenization and Topological LLM ideas.")
    if not connection_parts:
        connection_parts.append("Background theory that may guide operator-search heuristics.")
    implementation = []
    if "braid groups" in key_concepts:
        implementation.append("Add braid-aware symbolic priors or token vocabularies.")
    if "Selberg trace" in key_concepts:
        implementation.append("Use trace-formula-inspired regularizers or validation constraints.")
    if "DTES" in key_concepts or "tropical geometry" in key_concepts:
        implementation.append("Tie DTES state variables to operator or episode serialization.")
    if "random matrix theory" in key_concepts or "quantum chaos" in key_concepts:
        implementation.append("Compare learned spectra against chaos/RMT diagnostics.")
    if not implementation:
        implementation.append("Use as conceptual guidance for future operator and DSL experiments.")
    open_questions = [
        "Which concepts here can become executable constraints rather than narrative motivation?",
        "How should these ideas connect to operator discovery or PDE discovery in Ant-RH?",
    ]
    return {
        "path": path.as_posix(),
        "title": title,
        "main_topic": key_concepts[0] if key_concepts else "literature review",
        "key_concepts": key_concepts[:10],
        "formulas": _extract_formulas(text),
        "relevant_methods": relevant_methods[:10],
        "connection_to_ant_rh": " ".join(connection_parts),
        "implementation_ideas": implementation[:6],
        "open_questions": open_questions,
    }


def _llm_file_summary(runner: LLMRunner, path: Path, text: str, max_prompt_chars: int) -> str:
    prompt = (
        "You are studying literature for Ant-RH.\n"
        + GLOSSARY
        + "\nSummarize this markdown file as compact JSON with keys:\n"
        + "title, main_topic, key_concepts, formulas, relevant_methods, connection_to_ant_rh, implementation_ideas, open_questions.\n"
        + "Be concrete and practical. Keep arrays short.\n\n"
        + f"FILE: {path.name}\n\n"
        + text
    )
    try:
        return runner.generate(fit_prompt(prompt, max_prompt_chars), max_tokens=600, temperature=0.2).strip()
    except Exception:
        return ""


def _parse_llm_summary(raw: str) -> Dict[str, Any]:
    if not raw:
        return {}
    i = raw.find("{")
    j = raw.rfind("}")
    if i == -1 or j == -1 or j <= i:
        return {}
    blob = raw[i : j + 1]
    try:
        obj = json.loads(blob)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _global_rule_based(file_summaries: List[Dict[str, Any]]) -> Dict[str, str]:
    concepts: List[str] = []
    methods: List[str] = []
    titles = [str(x.get("title", "")) for x in file_summaries]
    for item in file_summaries:
        for c in item.get("key_concepts", []):
            if c not in concepts:
                concepts.append(c)
        for m in item.get("relevant_methods", []):
            if m not in methods:
                methods.append(m)
    summary = (
        "# Literature Summary\n\n"
        "## Main Themes\n"
        + "\n".join(f"- {x}" for x in concepts[:10])
        + "\n\n## Relevant Concepts for Ant-RH\n"
        + "\n".join(f"- {x}" for x in concepts[:12])
        + "\n\n## Artin / Selberg / Hilbert-Polya Connections\n"
        + "- Literature repeatedly links spectral geometry, trace methods, and operator viewpoints that fit Ant-RH's Artin/Selberg/operator loop.\n"
        + "\n## DTES / Tropical Geometry Connections\n"
        + "- DTES and tropical geometry appear as representation and state-space ideas rather than proofs.\n"
        + "\n## Braid / Topological LLM Connections\n"
        + "- Braid-group and topological operator literature can guide DSL/token design and candidate generation.\n"
        + "\n## Operator Discovery Ideas\n"
        + "\n".join(f"- {x}" for x in methods[:8])
        + "\n\n## Implementation Opportunities\n"
        + "\n".join(f"- {idea}" for item in file_summaries for idea in item.get("implementation_ideas", [])[:2])
        + "\n\n## Risks and Limitations\n"
        + "- Literature inspiration does not itself validate the operator or prove RH.\n"
        + "- Many connections are heuristic and need executable tests.\n"
        + "\n## Next Research Actions\n"
        + "- Translate top literature concepts into measurable constraints, losses, or generators.\n"
        + "- Tie Artin/Selberg/braid ideas to operator sensitivity and PDE discovery tests.\n"
    )
    lit_map = "# Literature Map\n\n" + "\n\n".join(
        f"## {item.get('title','unknown')}\n"
        f"- Main topic: {item.get('main_topic','')}\n"
        f"- Ant-RH connection: {item.get('connection_to_ant_rh','')}\n"
        f"- Implementation ideas: {'; '.join(item.get('implementation_ideas', [])[:3])}\n"
        for item in file_summaries
    )
    memory = (
        "# Gemma Literature Memory\n\n"
        + f"- Literature set covers: {', '.join(titles[:6])}\n"
        + "- Best-linked themes: Hilbert-Polya spectral ideas, Artin/Selberg constraints, DTES geometry, braid/topological operator representations.\n"
        + "- Use literature as heuristic guidance for operator discovery, PDE discovery, and Topological LLM design.\n"
        + "- Do not overclaim: these reviews motivate experiments but do not prove RH.\n"
    )
    return {"summary": summary, "map": lit_map, "memory": fit_prompt(memory, 3000)}


def _global_llm(runner: LLMRunner, file_summaries: List[Dict[str, Any]], max_prompt_chars: int) -> str:
    payload = json.dumps(file_summaries, indent=2)
    prompt = (
        "You are synthesizing literature for Ant-RH.\n"
        + GLOSSARY
        + "\nWrite markdown with exactly these sections:\n"
        + "# Literature Summary\n"
        + "## Main Themes\n"
        + "## Relevant Concepts for Ant-RH\n"
        + "## Artin / Selberg / Hilbert-Polya Connections\n"
        + "## DTES / Tropical Geometry Connections\n"
        + "## Braid / Topological LLM Connections\n"
        + "## Operator Discovery Ideas\n"
        + "## Implementation Opportunities\n"
        + "## Risks and Limitations\n"
        + "## Next Research Actions\n\n"
        + payload
    )
    try:
        return runner.generate(fit_prompt(prompt, max_prompt_chars), max_tokens=1200, temperature=0.2).strip()
    except Exception:
        return ""


def _memory_from_summary(summary: str) -> str:
    lines = [ln.strip() for ln in summary.splitlines() if ln.strip()]
    keep: List[str] = ["# Gemma Literature Memory", ""]
    for ln in lines:
        if ln.startswith("## "):
            keep.append(ln)
            continue
        if ln.startswith("- "):
            keep.append(ln)
        if len("\n".join(keep)) > 2600:
            break
    return fit_prompt("\n".join(keep), 3000)


def main() -> None:
    ap = argparse.ArgumentParser(description="Gemma Literature Study Agent for Ant-RH")
    ap.add_argument("--literature_dir", type=str, default="Docs/Literature")
    ap.add_argument("--backend", type=str, default="llama_cpp")
    ap.add_argument("--llama_cli", type=str, default="llama-cli")
    ap.add_argument("--model_path", type=str, default="/Users/machome/models/gemma/gemma-3-1b-it-Q5_K_M.gguf")
    ap.add_argument("--max_file_chars", type=int, default=8000)
    ap.add_argument("--max_prompt_chars", type=int, default=9000)
    ap.add_argument("--max_files", type=int, default=80)
    args = ap.parse_args()

    root = Path(ROOT)
    literature_dir = (root / args.literature_dir).resolve()
    if not literature_dir.exists():
        raise SystemExit(f"Literature folder missing: {literature_dir}")

    paths = sorted(literature_dir.glob("*.md"))[: int(args.max_files)]
    if not paths:
        raise SystemExit(f"No markdown literature files found in: {literature_dir}")

    runner = LLMRunner(
        model_path=str(args.model_path),
        llama_cli=str(args.llama_cli),
        n_ctx=4096,
        n_threads=6,
        n_gpu_layers=0,
        timeout_s=120.0,
    )

    file_summaries: List[Dict[str, Any]] = []
    for path in paths:
        text = _read_text(path, int(args.max_file_chars))
        summary = _rule_based_file_summary(path, text)
        if str(args.backend).lower() == "llama_cpp":
            raw = _llm_file_summary(runner, path, text, int(args.max_prompt_chars))
            parsed = _parse_llm_summary(raw)
            if parsed:
                summary.update(
                    {
                        "title": parsed.get("title", summary["title"]),
                        "main_topic": parsed.get("main_topic", summary["main_topic"]),
                        "key_concepts": parsed.get("key_concepts", summary["key_concepts"]),
                        "formulas": parsed.get("formulas", summary["formulas"]),
                        "relevant_methods": parsed.get("relevant_methods", summary["relevant_methods"]),
                        "connection_to_ant_rh": parsed.get("connection_to_ant_rh", summary["connection_to_ant_rh"]),
                        "implementation_ideas": parsed.get("implementation_ideas", summary["implementation_ideas"]),
                        "open_questions": parsed.get("open_questions", summary["open_questions"]),
                    }
                )
        file_summaries.append(summary)

    global_summary = ""
    if str(args.backend).lower() == "llama_cpp":
        global_summary = _global_llm(runner, file_summaries, int(args.max_prompt_chars))

    rb = _global_rule_based(file_summaries)
    literature_summary = global_summary if global_summary.strip().startswith("# Literature Summary") else rb["summary"]
    literature_map = rb["map"]
    literature_memory = _memory_from_summary(literature_summary)
    if len(literature_memory.strip()) < 40:
        literature_memory = rb["memory"]

    index = {
        "literature_dir": literature_dir.as_posix(),
        "file_count": len(file_summaries),
        "files": file_summaries,
    }

    _write_json(root / "runs/literature_index.json", index)
    _write_text(root / "runs/literature_summary.md", literature_summary)
    _write_text(root / "runs/literature_map.md", literature_map)
    _write_text(root / "runs/gemma_literature_memory.md", fit_prompt(literature_memory, 3000))


if __name__ == "__main__":
    main()
