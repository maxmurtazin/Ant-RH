#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
import subprocess
from pathlib import Path
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.llm_runner import LLMRunner
from help.local_tts import speak_text, speak_with_say


def _read_text(path: Path) -> Optional[str]:
    try:
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _load_csv_rows(path: Path) -> List[Dict[str, str]]:
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


def _bool_arg(x: Any) -> bool:
    return str(x).lower() in {"1", "true", "yes", "y", "t"}


def _fit_text(text: str, max_chars: int) -> str:
    t = str(text or "")
    if len(t) <= max_chars:
        return t
    half = max_chars // 2
    return t[:half] + "\n\n...[TRUNCATED]...\n\n" + t[-half:]


def _fmt_metric(value: Optional[float]) -> str:
    if value is None:
        return "unknown"
    return f"{value:.6g}"


def _delete_if_exists(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


def _load_memory_entries(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    entries: List[Dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    entries.append(obj)
    except Exception:
        return []
    return entries


def _render_help_memory(turns: List[Dict[str, Any]], max_chars: int = 3000) -> str:
    if not turns:
        return "No prior help memory."
    blocks: List[str] = []
    for item in turns:
        q = str(item.get("question", "")).strip()
        a = str(item.get("answer", "")).strip()
        if not q and not a:
            continue
        blocks.append(f"Q: {q}\nA: {a}")
    text = "\n\n".join(blocks).strip()
    if len(text) <= max_chars:
        return text
    last_three = "\n\n".join(blocks[-3:]).strip()
    if len(last_three) <= max_chars:
        return last_three
    return _fit_text(last_three, max_chars)


def _append_help_memory(path: Path, question: str, answer: str, mode: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "timestamp": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
                        "question": str(question).strip(),
                        "answer": str(answer).strip(),
                        "mode": str(mode),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    except Exception:
        pass


def _memory_direct_answer(question: str, memory_turns: List[Dict[str, Any]]) -> str:
    q = str(question or "").lower()
    turns = [t for t in memory_turns if isinstance(t, dict)]
    if not turns:
        return ""
    if "что я спрашивал" in q or "спрашивал раньше" in q:
        recent_questions = [str(t.get("question", "")).strip() for t in turns[-6:] if str(t.get("question", "")).strip()]
        if not recent_questions:
            return "В памяти нет прошлых вопросов."
        return "Последние вопросы:\n" + "\n".join(f"- {item}" for item in recent_questions)
    if "какую команду" in q or "какие команды" in q or "ты предлагала" in q:
        commands: List[str] = []
        seen = set()
        for t in reversed(turns):
            ans = str(t.get("answer", ""))
            for cmd in re.findall(r"(?:make|python3)\s+[^\n]+", ans):
                cmd = cmd.strip().strip("`")
                if cmd and cmd not in seen:
                    seen.add(cmd)
                    commands.append(cmd)
        if not commands:
            return "В сохраненной памяти нет предложенных команд."
        return "Ранее я предлагала команды:\n" + "\n".join(f"- `{cmd}`" for cmd in commands[:5])
    if "summary" in q or "последних вопросов" in q or "наших последних вопросов" in q:
        return _rule_based_memory_summary(turns[-10:])
    return ""


def _rule_based_memory_summary(entries: List[Dict[str, Any]]) -> str:
    turns = entries[-10:]
    if not turns:
        return "# Help Memory Summary\n\nNo stored help memory.\n"
    questions = [str(t.get("question", "")).strip() for t in turns if str(t.get("question", "")).strip()]
    answers = [str(t.get("answer", "")).strip() for t in turns if str(t.get("answer", "")).strip()]
    topic = questions[-1] if questions else "recent Ant-RH troubleshooting"
    unresolved: List[str] = []
    for text in questions + answers:
        low = text.lower()
        if any(k in low for k in ["not learning", "не уч", "stuck", "hang", "plateau", "problem", "issue", "ошиб"]):
            unresolved.append(text)
    commands = []
    for text in answers:
        commands.extend(re.findall(r"(?:make|python3)\s+[^\n]+", text))
    seen = set()
    uniq_commands = []
    for cmd in commands:
        cmd = cmd.strip().strip("`")
        if cmd and cmd not in seen:
            seen.add(cmd)
            uniq_commands.append(cmd)
    next_actions = uniq_commands[-4:] if uniq_commands else ["make analyze-gemma", "make lab-journal"]
    lines = [
        "# Help Memory Summary",
        "",
        "## Current Topic",
        topic,
        "",
        "## Unresolved Issues",
    ]
    if unresolved:
        lines.extend(f"- {item}" for item in unresolved[-4:])
    else:
        lines.append("- No major unresolved issue captured.")
    lines.extend(["", "## Commands Tried"])
    if uniq_commands:
        lines.extend(f"- `{cmd}`" for cmd in uniq_commands[-6:])
    else:
        lines.append("- No commands captured.")
    lines.extend(["", "## Next Actions"])
    lines.extend(f"- `{cmd}`" for cmd in next_actions)
    lines.append("")
    return "\n".join(lines)


def _update_memory_summary(summary_path: Path, entries: List[Dict[str, Any]]) -> None:
    try:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(_rule_based_memory_summary(entries), encoding="utf-8")
    except Exception:
        pass


def chunk_text(text, max_chars=180):
    text = re.sub(r"\s+", " ", str(text or "").strip())
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks: List[str] = []
    current = ""
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        if len(current) + len(s) + 1 <= max_chars:
            current = (current + " " + s).strip()
        else:
            if current:
                chunks.append(current)
            if len(s) <= max_chars:
                current = s
            else:
                while len(s) > max_chars:
                    cut = s.rfind(" ", 0, max_chars)
                    if cut <= 0:
                        cut = max_chars
                    chunks.append(s[:cut].strip())
                    s = s[cut:].strip()
                current = s
    if current:
        chunks.append(current)
    return chunks


def prepare_spoken_chunk(chunk: str, prefix="Gemma says"):
    chunk = chunk.strip()
    chunk = chunk.replace("ACO", "A C O")
    chunk = chunk.replace("RL", "R L")
    return f"{prefix}... {chunk}"


def speak_chunk(chunk, voice="Samantha", rate=150):
    chunk_text_safe = str(chunk)
    if chunk_text_safe.startswith("-"):
        chunk_text_safe = " " + chunk_text_safe
    subprocess.run(
        [
            "say",
            "-v",
            str(voice),
            "-r",
            str(rate),
            chunk_text_safe,
        ],
        check=False,
    )


def _trend(values: List[float]) -> str:
    n = len(values)
    if n < 2:
        return "flat"
    xs = list(range(n))
    mx = sum(xs) / n
    my = sum(values) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, values))
    den = sum((x - mx) ** 2 for x in xs)
    slope = num / den if den else 0.0
    if slope > 0:
        return "increasing"
    if slope < 0:
        return "decreasing"
    return "flat"


def _extract_yaml_block(yaml_text: str, key: str) -> str:
    # minimal parser for `key: |` blocks
    pat = re.compile(rf"^{re.escape(key)}:\s*\|\s*$", re.M)
    m = pat.search(yaml_text)
    if not m:
        return ""
    rest = yaml_text[m.end() :].splitlines()
    lines: List[str] = []
    for ln in rest:
        if ln.startswith("  "):
            lines.append(ln[2:])
        elif ln.strip() == "":
            lines.append("")
        else:
            break
    return "\n".join(lines).strip()


def _load_prompt_config(path: Path) -> Dict[str, str]:
    txt = _read_text(path) or ""
    return {
        "system_prompt": _extract_yaml_block(txt, "system_prompt"),
        "fallback_answer_template": _extract_yaml_block(txt, "fallback_answer_template"),
    }


def _summarize_context(root: Path) -> Dict[str, Any]:
    ga = _read_json(root / "runs/gemma_analysis.json") or {}
    osr = _read_json(root / "runs/operator_stability_report.json") or {}
    journal = _read_text(root / "runs/lab_journal.md") or ""
    pde_report = _read_text(root / "runs/operator_pde_report.md") or ""
    paper_draft = _read_text(root / "runs/paper_draft.md") or ""
    gemma_project_memory = _read_text(root / "runs/gemma_project_memory.md") or ""
    gemma_literature_memory = _read_text(root / "runs/gemma_literature_memory.md") or ""
    project_summary = _read_text(root / "runs/project_summary.md") or ""
    project_map = _read_text(root / "runs/project_map.md") or ""
    aco_rows = _load_csv_rows(root / "runs/artin_aco_history.csv")
    rl_rows = _load_csv_rows(root / "runs/artin_rl/train_history.csv")

    aco_best = [v for v in (_to_float(r.get("best_loss")) for r in aco_rows) if v is not None]
    aco_mean = [v for v in (_to_float(r.get("mean_loss")) for r in aco_rows) if v is not None]
    rl_reward = [v for v in (_to_float(r.get("mean_reward")) for r in rl_rows) if v is not None]

    aco_trend = _trend(aco_best)
    aco_mean_trend = _trend(aco_mean)
    rl_trend = _trend(rl_reward)
    spectral_loss = _to_float(osr.get("spectral_loss"))
    eig_ok = bool(osr.get("eigh_success", osr.get("eigh_after", {}).get("eigh_success", False)))
    main_issue = str(ga.get("main_issue", "unknown"))

    next_command = "make run-v12"
    if aco_trend == "increasing":
        next_command = "make aco-gemma"
    elif spectral_loss is not None and spectral_loss > 1.0:
        next_command = "make operator && make stability"
    elif rl_trend == "increasing":
        next_command = "make paper"

    project_memory_parts = [gemma_project_memory.strip(), project_summary.strip(), project_map.strip()]
    project_memory_joined = "\n\n".join([p for p in project_memory_parts if p]).strip()
    project_memory_limited = _fit_text(project_memory_joined, 4000)
    literature_memory_limited = _fit_text(gemma_literature_memory.strip(), 3000)

    return {
        "project_memory": project_memory_limited,
        "literature_memory": literature_memory_limited,
        "aco_trend": aco_trend,
        "aco_best_loss_trend": aco_trend,
        "aco_mean_loss_trend": aco_mean_trend,
        "rl_trend": rl_trend,
        "aco_best_last": (aco_best[-1] if aco_best else None),
        "aco_best_min": (min(aco_best) if aco_best else None),
        "aco_mean_last": (aco_mean[-1] if aco_mean else None),
        "rl_reward_last": (rl_reward[-1] if rl_reward else None),
        "spectral_loss": spectral_loss,
        "eig_ok": eig_ok,
        "main_issue": main_issue,
        "journal_tail": "\n".join(journal.splitlines()[-40:]) if journal else "",
        "operator_pde_report_tail": "\n".join(pde_report.splitlines()[-30:]) if pde_report else "",
        "paper_draft_tail": "\n".join(paper_draft.splitlines()[-30:]) if paper_draft else "",
        "next_command": next_command,
    }


def _build_prompt(question: str, context: Dict[str, Any], prompt_cfg: Dict[str, str], help_memory: str = "") -> str:
    return (
        f"{prompt_cfg.get('system_prompt', '').strip()}\n\n"
        "Grounding rules:\n"
        "- Do not claim a specific function is broken unless its source code or traceback is present in context.\n"
        "- Prefer metrics from runs/*.csv and runs/*.json over guesses.\n"
        "- If ACO best_loss_trend == increasing, say: \"ACO is not learning in the current run because best_loss and mean_loss are increasing.\"\n"
        "- If loss is large, say: \"Loss scale is unstable or poorly normalized.\"\n"
        "- Do not say losses are zero unless metrics explicitly show zero.\n"
        "- Keep the answer concise.\n\n"
        "HELP MEMORY:\n"
        f"{help_memory or 'No prior help memory.'}\n\n"
        "LITERATURE MEMORY:\n"
        f"{context.get('literature_memory', '')}\n\n"
        "PROJECT MEMORY\n"
        f"{context.get('project_memory', '')}\n\n"
        "ACTUAL ACO METRICS:\n"
        f"best_loss_last = {_fmt_metric(context.get('aco_best_last'))}\n"
        f"best_loss_min = {_fmt_metric(context.get('aco_best_min'))}\n"
        f"best_loss_trend = {context.get('aco_best_loss_trend', 'unknown')}\n"
        f"mean_loss_last = {_fmt_metric(context.get('aco_mean_last'))}\n"
        f"mean_loss_trend = {context.get('aco_mean_loss_trend', 'unknown')}\n\n"
        "Context:\n"
        f"{json.dumps({k: v for k, v in context.items() if k not in {'project_memory', 'literature_memory'}}, indent=2)}\n\n"
        "CURRENT QUESTION:\n"
        f"{question.strip()}\n\n"
        "Respond in plain text with:\n"
        "1) short diagnosis\n"
        "2) concrete next command(s)\n"
        "3) one caution if relevant\n"
        "Keep the answer concise.\n"
        "If the user asks for commands, output exact shell commands.\n"
        "Understand both English and Russian questions naturally.\n"
    )


def _apply_grounding_correction(answer: str, context: Dict[str, Any]) -> str:
    corrected = str(answer or "").strip()
    best_loss_last = _to_float(context.get("aco_best_last"))
    best_loss_trend = str(context.get("aco_best_loss_trend", "unknown"))
    mean_loss_trend = str(context.get("aco_mean_loss_trend", "unknown"))

    contradiction_patterns = [
        "returning zero",
        "always zero",
        "zero loss",
    ]
    has_zero_claim = any(p in corrected.lower() for p in contradiction_patterns)

    if has_zero_claim and best_loss_last is not None and best_loss_last > 1.0:
        correction = (
            f"Correction: current logged ACO loss is not zero. The latest best_loss is "
            f"{_fmt_metric(best_loss_last)}, and the trend is {best_loss_trend}."
        )
        corrected = (corrected + "\n\n" + correction).strip()

    if (
        best_loss_trend == "increasing"
        and mean_loss_trend == "increasing"
        and "ACO is not learning in the current run because best_loss and mean_loss are increasing." not in corrected
    ):
        corrected = (
            "ACO is not learning in the current run because best_loss and mean_loss are increasing.\n\n"
            + corrected
        ).strip()

    if best_loss_last is not None and best_loss_last > 1.0:
        loss_scale_sentence = "Loss scale is unstable or poorly normalized."
        if loss_scale_sentence not in corrected:
            corrected = (corrected + "\n\n" + loss_scale_sentence).strip()

    if best_loss_trend == "increasing" and "make aco-gemma" not in corrected.lower():
        corrected = (corrected + "\n\nRun: make aco-gemma").strip()

    return corrected


def _fallback_answer(question: str, context: Dict[str, Any], prompt_cfg: Dict[str, str]) -> str:
    tmpl = prompt_cfg.get("fallback_answer_template", "").strip()
    if tmpl:
        try:
            base = tmpl.format(**context)
        except Exception:
            base = ""
    else:
        base = ""
    q = question.lower()
    if ("aco" in q and "learn" in q) or ("aco" in q and ("не уч" in q or "pочему" in q)):
        extra = (
            "ACO is likely stuck in a weak exploration-exploitation regime. "
            "Try increasing exploration or lowering pheromone lock-in. "
            "Run: make aco-gemma"
        )
    elif "spectral_loss" in q or "spectral loss" in q or "spectral loss" in q or "что значит spectral" in q:
        extra = (
            "spectral_loss measures mismatch between learned and target spectra. "
            "Lower is better; high values suggest poor operator alignment. "
            "Run: make operator && make stability"
        )
    elif "v12.6" in q or "v12 6" in q or "как запустить v12.6" in q:
        extra = (
            "Suggested V12.6-like command:\n"
            "python3 -m core.artin_operator --n_points 64 --sigma 0.3 --top_k_geodesics 50 "
            "--eps 0.6 --geodesics runs/artin_words.json --zeros data/zeta_zeros.txt --out_dir runs/"
        )
    elif "pde" in q and ("command" in q or "команд" in q):
        extra = "Run: make pde"
    elif "summary" in q or "summary" in q or "краткое" in q or "дальше" in q or "что делать" in q:
        extra = (
            "Short summary: ACO is not learning, RL is improving, and the main issue is scaling_and_alignment. "
            "Commands:\n- make aco-gemma\n- make operator && make stability\n- make pde"
        )
    else:
        extra = (
            "Recommended next step: run the full pipeline and then analysis.\n"
            "Commands:\n- make run-v12\n- make analyze-gemma\n- make lab-journal"
        )
    return (base + "\n\n" + extra).strip()


def _resolve_voice_name(voice_name: str) -> str:
    try:
        available = subprocess.run(
            ["say", "-v", "?"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
        if str(voice_name) not in available:
            return "Samantha"
        return str(voice_name)
    except Exception:
        return "Samantha"


def _generate_answer(
    question: str,
    args: argparse.Namespace,
    root: Path,
    memory_turns: Optional[List[Dict[str, Any]]] = None,
) -> str:
    prompt_cfg = _load_prompt_config(root / args.prompts)
    context = _summarize_context(root)
    if _bool_arg(getattr(args, "memory", False)):
        direct = _memory_direct_answer(question, memory_turns or [])
        if direct:
            return direct
    help_memory = ""
    if _bool_arg(getattr(args, "memory", False)):
        help_memory = _render_help_memory(memory_turns or [], max_chars=3000)
    prompt = _build_prompt(question, context, prompt_cfg, help_memory=help_memory)
    answer = ""
    try:
        print("[debug] new LLM call started", flush=True)
        runner = LLMRunner(
            model_path=str(args.model_path),
            llama_cli=str(args.llama_cli),
            n_ctx=4096,
            n_threads=6,
            n_gpu_layers=0,
            timeout_s=120.0,
        )
        answer = runner.generate(
            prompt,
            max_tokens=args.max_tokens,
            temperature=0.2,
        ).strip()
    except Exception:
        answer = ""
    if not answer:
        answer = _fallback_answer(question, context, prompt_cfg)
    return _apply_grounding_correction(answer, context)


def _build_runner(args: argparse.Namespace) -> LLMRunner:
    return LLMRunner(
        model_path=str(args.model_path),
        llama_cli=str(args.llama_cli),
        n_ctx=4096,
        n_threads=6,
        n_gpu_layers=0,
        timeout_s=60.0,
    )


def _build_help_prompt(
    question: str,
    args: argparse.Namespace,
    root: Path,
    memory_turns: Optional[List[Dict[str, Any]]] = None,
) -> tuple[str, Dict[str, str], Dict[str, Any]]:
    prompt_cfg = _load_prompt_config(root / args.prompts)
    context = _summarize_context(root)
    help_memory = ""
    if _bool_arg(getattr(args, "memory", False)):
        help_memory = _render_help_memory(memory_turns or [], max_chars=3000)
    prompt = _build_prompt(question, context, prompt_cfg, help_memory=help_memory)
    return prompt, prompt_cfg, context


def _real_stream_print_and_speak(
    question: str,
    args: argparse.Namespace,
    root: Path,
    voice_enabled: bool,
    memory_turns: Optional[List[Dict[str, Any]]] = None,
) -> bool:
    prompt, prompt_cfg, context = _build_help_prompt(question, args, root, memory_turns=memory_turns)
    runner = _build_runner(args)
    buffer = ""
    printed_any = False
    announced = False
    try:
        for frag in runner.generate_stream(
            prompt,
            max_tokens=args.max_tokens,
            temperature=0.2,
        ):
            if not frag:
                continue
            buffer += frag
            if re.search(r"[.!?\n]\s*$", buffer) or len(buffer) >= int(args.stream_chunk_chars):
                chunk = buffer.strip()
                buffer = ""
                if not chunk:
                    continue
                print(chunk, flush=True)
                printed_any = True
                if voice_enabled:
                    try:
                        if not announced:
                            subprocess.run(
                                [
                                    "say",
                                    "-v",
                                    args.voice_name,
                                    "-r",
                                    str(args.voice_rate),
                                    "Gemma says...",
                                ],
                                check=False,
                            )
                            announced = True
                        subprocess.run(
                            [
                                "say",
                                "-v",
                                args.voice_name,
                                "-r",
                                str(args.voice_rate),
                                chunk,
                            ],
                            check=False,
                        )
                    except Exception:
                        pass
        tail = buffer.strip()
        if tail:
            print(tail, flush=True)
            printed_any = True
            if voice_enabled:
                try:
                    if not announced:
                        subprocess.run(
                            [
                                "say",
                                "-v",
                                args.voice_name,
                                "-r",
                                str(args.voice_rate),
                                "Gemma says...",
                            ],
                            check=False,
                        )
                        announced = True
                    subprocess.run(
                        [
                            "say",
                            "-v",
                            args.voice_name,
                            "-r",
                            str(args.voice_rate),
                            tail,
                        ],
                        check=False,
                    )
                except Exception:
                    pass
        if printed_any:
            return True
    except Exception:
        pass

    answer = _fallback_answer(question, context, prompt_cfg)
    _stream_print_and_speak(answer, args, voice_enabled)
    return False


def _stream_print_and_speak(answer: str, args: argparse.Namespace, voice_enabled: bool) -> None:
    chunks = chunk_text(answer, max_chars=int(args.stream_chunk_chars))
    if voice_enabled:
        try:
            subprocess.run(
                [
                    "say",
                    "-v",
                    args.voice_name,
                    "-r",
                    str(args.voice_rate),
                    "Gemma says...",
                ],
                check=False,
            )
        except Exception:
            pass
    for chunk in chunks:
        print(chunk, flush=True)
        if voice_enabled:
            try:
                subprocess.run(
                    [
                        "say",
                        "-v",
                        args.voice_name,
                        "-r",
                        str(args.voice_rate),
                        chunk,
                    ],
                    check=False,
                )
            except Exception:
                pass
        time.sleep(float(args.stream_pause))


def main() -> None:
    ap = argparse.ArgumentParser(description="Local Gemma Help Agent for Ant-RH")
    ap.add_argument("--question", type=str, default="")
    ap.add_argument("--voice", type=str, default="False")
    ap.add_argument("--llama_cli", type=str, default="llama-cli")
    ap.add_argument("--model_path", type=str, default="/Users/machome/models/gemma/gemma-3-1b-it-Q4_K_M.gguf")
    ap.add_argument("--voice_name", type=str, default="Ava")
    ap.add_argument("--prompts", type=str, default="help/help_prompts.yaml")
    ap.add_argument("--tts_backend", type=str, default="say", choices=["say", "pyttsx3"])
    ap.add_argument("--voice_rate", type=int, default=145)
    ap.add_argument("--voice_volume", type=float, default=0.85)
    ap.add_argument("--soft_mode", type=str, default="True")
    ap.add_argument("--list_voices", type=str, default="False")
    ap.add_argument("--tts_max_chars", type=int, default=220)
    ap.add_argument("--stream", type=str, default="False")
    ap.add_argument("--stream_chunk_chars", type=int, default=180)
    ap.add_argument("--stream_pause", type=float, default=0.15)
    ap.add_argument("--interactive", type=str, default="False")
    ap.add_argument("--real_stream", type=str, default="False")
    ap.add_argument("--memory", type=str, default="True")
    ap.add_argument("--memory_turns", type=int, default=6)
    ap.add_argument("--memory_path", type=str, default="runs/help_memory.jsonl")
    ap.add_argument("--memory_summary_path", type=str, default="runs/help_memory_summary.md")
    ap.add_argument("--clear_memory", type=str, default="False")
    ap.add_argument(
        "--max_tokens",
        type=int,
        default=256,
        help="Max tokens for LLM generation",
    )
    args = ap.parse_args()

    root = Path(ROOT)
    memory_path = root / str(args.memory_path)
    memory_summary_path = root / str(args.memory_summary_path)
    if _bool_arg(args.clear_memory):
        _delete_if_exists(memory_path)
        _delete_if_exists(memory_summary_path)
    persisted_memory = _load_memory_entries(memory_path) if _bool_arg(args.memory) else []
    session_memory: List[Dict[str, Any]] = []

    if _bool_arg(args.list_voices):
        if str(args.tts_backend).lower() == "pyttsx3":
            try:
                from help.local_tts_pyttsx3 import LocalTTS

                tts = LocalTTS(
                    voice_hint="female",
                    rate=int(args.voice_rate),
                    volume=float(args.voice_volume),
                    soft_mode=_bool_arg(args.soft_mode),
                )
                tts.list_voices()
                return
            except Exception:
                print("python3 -m pip install pyttsx3")
                return
        print("Voice listing is available for --tts_backend pyttsx3")
        return

    args.voice_name = _resolve_voice_name(str(args.voice_name))
    voice_enabled = _bool_arg(args.voice)

    if _bool_arg(args.interactive):
        print("Gemma Help is ready. Type your question, or 'exit' to quit.")
        while True:
            try:
                question = input("you> ").strip()
            except EOFError:
                break
            if question.lower() in ["exit", "quit", "q"]:
                break
            if not question:
                continue
            recent_memory = (persisted_memory + session_memory)[-int(args.memory_turns) :]
            if _bool_arg(args.real_stream):
                _real_stream_print_and_speak(question, args, root, voice_enabled, memory_turns=recent_memory)
            else:
                answer = _generate_answer(question, args, root, memory_turns=recent_memory)
                if _bool_arg(args.stream):
                    _stream_print_and_speak(answer, args, voice_enabled)
                else:
                    print(answer)
                    if voice_enabled:
                        try:
                            speak_with_say(
                                answer,
                                voice=str(args.voice_name),
                                rate=int(args.voice_rate),
                                max_chars=int(args.tts_max_chars),
                                pause=0.35,
                            )
                        except Exception:
                            pass
                if _bool_arg(args.memory):
                    turn = {"question": question, "answer": answer, "mode": "interactive"}
                    session_memory.append(turn)
                    session_memory = session_memory[-int(args.memory_turns) :]
                    _append_help_memory(memory_path, question, answer, "interactive")
                    persisted_memory.append(turn)
                    if len(persisted_memory) % 5 == 0:
                        _update_memory_summary(memory_summary_path, persisted_memory)
                print("", flush=True)
        return

    if not str(args.question).strip():
        raise SystemExit("Please provide --question")

    if _bool_arg(args.real_stream):
        _real_stream_print_and_speak(args.question, args, root, voice_enabled)
        return

    single_memory = (persisted_memory + session_memory)[-int(args.memory_turns) :]
    answer = _generate_answer(args.question, args, root, memory_turns=single_memory)

    if _bool_arg(args.memory):
        turn = {"question": args.question, "answer": answer, "mode": "single"}
        _append_help_memory(memory_path, args.question, answer, "single")
        persisted_memory.append(turn)
        if len(persisted_memory) % 5 == 0:
            _update_memory_summary(memory_summary_path, persisted_memory)

    if _bool_arg(args.stream):
        _stream_print_and_speak(answer, args, voice_enabled)
    else:
        print(answer)

    if voice_enabled and not _bool_arg(args.stream):
        tts_backend = str(args.tts_backend).lower()
        if tts_backend == "pyttsx3":
            try:
                from help.local_tts_pyttsx3 import LocalTTS

                tts = LocalTTS(
                    voice_hint="female",
                    rate=int(args.voice_rate),
                    volume=float(args.voice_volume),
                    soft_mode=_bool_arg(args.soft_mode),
                )
                tts.speak(answer)
                return
            except Exception as e:
                print(f"[pyttsx3 unavailable: {e}]")
                print("python3 -m pip install pyttsx3")
        try:
            speak_with_say(
                answer,
                voice=str(args.voice_name),
                rate=int(args.voice_rate),
                max_chars=int(args.tts_max_chars),
                pause=0.35,
            )
        except Exception:
            ok, err = speak_text(answer, voice_name=str(args.voice_name))
            if not ok:
                print(f"[TTS disabled: {err}]")


if __name__ == "__main__":
    main()

