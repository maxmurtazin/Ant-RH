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


def _read_yaml_like(path: Path) -> Optional[str]:
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


def _linear_slope(values: List[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    xs = list(range(n))
    x_mean = sum(xs) / float(n)
    y_mean = sum(values) / float(n)
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, values))
    den = sum((x - x_mean) ** 2 for x in xs)
    if den <= 0.0:
        return 0.0
    return num / den


def _tail(path: Path, max_lines: int = 8) -> List[str]:
    try:
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return [ln.strip() for ln in lines[-max_lines:] if ln.strip()]
    except Exception:
        return []


def _git_commit_hash(repo_root: Path) -> Optional[str]:
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
        h = (p.stdout or "").strip()
        return h or None
    except Exception:
        return None


def _extract_json_blob(text: str) -> Optional[Dict[str, Any]]:
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _summarize_metrics(root: Path) -> Dict[str, Any]:
    aco_rows = _load_csv(root / "runs/artin_aco_history.csv")
    rl_rows = _load_csv(root / "runs/artin_rl/train_history.csv")
    op = _read_json(root / "runs/operator_stability_report.json") or {}
    sel = _read_json(root / "runs/selberg_trace_report.json") or {}
    structured = _read_json(root / "runs/artin_structured_report.json") or {}
    gemma_analysis = _read_json(root / "runs/gemma_analysis.json") or {}
    config_text = _read_yaml_like(root / "runs/v12_config_used.yaml")

    aco_best = [v for v in (_to_float(r.get("best_loss")) for r in aco_rows) if v is not None]
    aco_mean = [v for v in (_to_float(r.get("mean_loss")) for r in aco_rows) if v is not None]
    rl_reward = [v for v in (_to_float(r.get("mean_reward")) for r in rl_rows) if v is not None]
    rl_valid = [v for v in (_to_float(r.get("valid_rate")) for r in rl_rows) if v is not None]

    aco_slope = _linear_slope(aco_best) if aco_best else 0.0
    reward_slope = _linear_slope(rl_reward) if rl_reward else 0.0
    aco_trend = "increasing" if aco_slope > 0 else ("decreasing" if aco_slope < 0 else "flat")
    rl_trend = "increasing" if reward_slope > 0 else ("decreasing" if reward_slope < 0 else "flat")

    spectral_loss = _to_float(op.get("spectral_loss"))
    spacing_loss = _to_float(op.get("spacing_loss"))
    eig_ok = bool(op.get("eigh_success", op.get("eigh_after", {}).get("eigh_success", False)))

    logs_dir = root / "logs/v12"
    log_snippets: Dict[str, List[str]] = {}
    if logs_dir.exists():
        for p in sorted(logs_dir.glob("*.log")):
            log_snippets[p.name] = _tail(p, max_lines=4)

    return {
        "aco": {
            "n": len(aco_rows),
            "best_loss_last": aco_best[-1] if aco_best else None,
            "mean_loss_last": aco_mean[-1] if aco_mean else None,
            "best_loss_trend": aco_trend,
        },
        "rl": {
            "n": len(rl_rows),
            "mean_reward_last": rl_reward[-1] if rl_reward else None,
            "mean_reward_trend": rl_trend,
            "valid_rate_last": rl_valid[-1] if rl_valid else None,
        },
        "operator": {
            "spectral_loss": spectral_loss,
            "spacing_loss": spacing_loss,
            "eigensolver_succeeded": eig_ok,
            "condition_proxy_after": _to_float(op.get("condition_proxy_after")),
        },
        "selberg": {
            "loss": _to_float(sel.get("loss")),
            "relative_error": _to_float(sel.get("relative_error")),
        },
        "structured": {
            "learning": structured.get("learning"),
            "operator_quality": structured.get("operator_quality"),
            "main_issue": structured.get("main_issue"),
        },
        "gemma_analysis": {
            "learning": gemma_analysis.get("learning"),
            "operator_quality": gemma_analysis.get("operator_quality"),
            "main_issue": gemma_analysis.get("main_issue"),
        },
        "config_excerpt": (config_text[:1200] if config_text else None),
        "logs_tail": log_snippets,
    }


def _rule_based_sections(metrics: Dict[str, Any]) -> Dict[str, str]:
    aco = metrics.get("aco", {})
    rl = metrics.get("rl", {})
    op = metrics.get("operator", {})
    sel = metrics.get("selberg", {})
    structured = metrics.get("structured", {})

    worked: List[str] = []
    failed: List[str] = []

    if rl.get("mean_reward_trend") == "increasing":
        worked.append("RL mean reward increased across updates.")
    else:
        failed.append("RL reward trend did not improve.")

    if op.get("eigensolver_succeeded"):
        worked.append("Stability check passed; eigensolver succeeded.")
    else:
        failed.append("Stability check failed; eigensolver did not succeed.")

    if aco.get("best_loss_trend") == "increasing":
        failed.append("ACO is not learning in this run.")
    elif aco.get("best_loss_trend") == "decreasing":
        worked.append("ACO best loss trended downward.")

    spectral_loss = _to_float(op.get("spectral_loss"))
    if spectral_loss is not None and spectral_loss > 1e4:
        failed.append("Operator/spectrum alignment is poor.")
    elif spectral_loss is not None:
        worked.append(f"Spectral loss remains bounded ({spectral_loss:.6g}).")

    rel_err = _to_float(sel.get("relative_error"))
    if rel_err is not None and rel_err > 1.0:
        failed.append("Selberg relative error remains high.")

    interpretation = "Learning signal is mixed across modules."
    if failed and not worked:
        interpretation = "Current run is unstable and needs intervention before scaling."
    elif worked and not failed:
        interpretation = "Current run is stable enough for controlled scaling."
    elif structured.get("main_issue"):
        interpretation = f"Primary bottleneck appears to be {structured.get('main_issue')}."

    next_action = "Increase diagnostics and run a short ablation."
    if "ACO is not learning in this run." in failed:
        next_action = "Retune ACO exploration parameters (beta, rho) and rerun a 10-iteration smoke test."
    elif rel_err is not None and rel_err > 1.0:
        next_action = "Prioritize Selberg alignment and normalize trace terms before the next full run."
    elif rl.get("mean_reward_trend") == "increasing" and op.get("eigensolver_succeeded"):
        next_action = "Scale to a longer run with unchanged stability settings and monitor spectral drift."

    return {
        "what_worked": " ".join(worked) if worked else "No strong positive signal identified.",
        "what_failed": " ".join(failed) if failed else "No major failure flags in this run.",
        "interpretation": interpretation,
        "next_action": next_action,
    }


def _build_prompt(payload: Dict[str, Any]) -> str:
    return (
        "You are the Ant-RH lab journal assistant. Write concise scientific prose.\n"
        "Return strict JSON with keys: what_worked, what_failed, interpretation, next_action.\n"
        "Required hard rules:\n"
        '- If ACO best-loss trend is increasing, include exact sentence: "ACO is not learning in this run."\n'
        '- If spectral_loss > 1e4, include exact sentence: "Operator/spectrum alignment is poor."\n'
        "- If eigensolver succeeded, explicitly mention stability passed.\n\n"
        "Experiment summary data:\n"
        f"{json.dumps(payload, indent=2)}\n"
    )


def _llm_sections(metrics: Dict[str, Any], llama_cli: str, model_path: str) -> Optional[Dict[str, str]]:
    try:
        runner = LLMRunner(
            model_path=str(model_path),
            llama_cli=str(llama_cli),
            n_ctx=4096,
            n_threads=6,
            n_gpu_layers=0,
            timeout_s=60.0,
        )
        raw = runner.generate(_build_prompt(metrics), max_tokens=512, temperature=0.2)
        parsed = _extract_json_blob(raw or "")
        if not parsed:
            return None
        required = {"what_worked", "what_failed", "interpretation", "next_action"}
        if not required.issubset(set(parsed.keys())):
            return None
        return {k: str(parsed.get(k, "")).strip() for k in required}
    except Exception:
        return None


def _enforce_required_sentences(sections: Dict[str, str], metrics: Dict[str, Any]) -> Dict[str, str]:
    out = dict(sections)
    aco_trend = metrics.get("aco", {}).get("best_loss_trend")
    spec = _to_float(metrics.get("operator", {}).get("spectral_loss"))
    eig_ok = bool(metrics.get("operator", {}).get("eigensolver_succeeded"))

    if aco_trend == "increasing":
        s = "ACO is not learning in this run."
        if s not in out.get("what_failed", ""):
            out["what_failed"] = (out.get("what_failed", "") + " " + s).strip()
    if spec is not None and spec > 1e4:
        s = "Operator/spectrum alignment is poor."
        if s not in out.get("what_failed", ""):
            out["what_failed"] = (out.get("what_failed", "") + " " + s).strip()
    if eig_ok:
        s = "Stability check passed; eigensolver succeeded."
        if "stability passed" not in out.get("what_worked", "").lower() and s not in out.get("what_worked", ""):
            out["what_worked"] = (out.get("what_worked", "") + " " + s).strip()
    return out


def _append_jsonl(path: Path, entry: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=True) + "\n")


def _append_markdown(path: Path, entry: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"## Lab Entry {entry['timestamp_utc']}",
        f"- Commit: `{entry.get('git_commit') or 'N/A'}`",
        f"- Backend: `{entry.get('backend')}`",
        f"- Model: `{entry.get('model_path')}`",
        f"- Command/config used: {entry.get('command_or_config_used')}",
        "- Key metrics:",
    ]
    for k, v in (entry.get("key_metrics") or {}).items():
        lines.append(f"  - {k}: {v}")
    lines.extend(
        [
            f"- What worked: {entry.get('what_worked')}",
            f"- What failed: {entry.get('what_failed')}",
            f"- Interpretation: {entry.get('interpretation')}",
            f"- Next suggested action: {entry.get('next_suggested_action')}",
            "",
        ]
    )
    with path.open("a", encoding="utf-8") as f:
        if path.stat().st_size == 0:
            f.write("# Ant-RH Gemma Lab Journal\n\n")
        f.write("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Append-only Gemma lab journal agent for Ant-RH")
    ap.add_argument("--backend", type=str, default="llama_cpp", choices=["llama_cpp", "rule_based"])
    ap.add_argument("--llama_cli", type=str, default="llama-cli")
    ap.add_argument(
        "--model_path",
        type=str,
        default="/Users/machome/models/gemma/gemma-3-1b-it-Q5_K_M.gguf",
    )
    ap.add_argument("--out_md", type=str, default="runs/lab_journal.md")
    ap.add_argument("--out_jsonl", type=str, default="runs/lab_journal.jsonl")
    args = ap.parse_args()

    root = Path(ROOT)
    metrics = _summarize_metrics(root)
    timestamp_utc = dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()
    commit_hash = _git_commit_hash(root)

    sections = None
    if str(args.backend).lower() == "llama_cpp":
        sections = _llm_sections(metrics, args.llama_cli, args.model_path)
    if not sections:
        sections = _rule_based_sections(metrics)
        backend_used = "rule_based"
    else:
        backend_used = "llama_cpp"
    sections = _enforce_required_sentences(sections, metrics)

    key_metrics = {
        "aco_best_loss_last": metrics.get("aco", {}).get("best_loss_last"),
        "aco_best_loss_trend": metrics.get("aco", {}).get("best_loss_trend"),
        "rl_mean_reward_last": metrics.get("rl", {}).get("mean_reward_last"),
        "rl_mean_reward_trend": metrics.get("rl", {}).get("mean_reward_trend"),
        "operator_spectral_loss": metrics.get("operator", {}).get("spectral_loss"),
        "operator_spacing_loss": metrics.get("operator", {}).get("spacing_loss"),
        "operator_eigensolver_succeeded": metrics.get("operator", {}).get("eigensolver_succeeded"),
        "selberg_relative_error": metrics.get("selberg", {}).get("relative_error"),
    }
    entry = {
        "timestamp_utc": timestamp_utc,
        "git_commit": commit_hash,
        "backend": backend_used,
        "model_path": str(args.model_path),
        "command_or_config_used": "runs/v12_config_used.yaml (if present) + logs/v12/*.log tails",
        "key_metrics": key_metrics,
        "what_worked": sections.get("what_worked", ""),
        "what_failed": sections.get("what_failed", ""),
        "interpretation": sections.get("interpretation", ""),
        "next_suggested_action": sections.get("next_action", ""),
        "inputs_seen": {
            "runs/artin_aco_history.csv": (root / "runs/artin_aco_history.csv").exists(),
            "runs/artin_rl/train_history.csv": (root / "runs/artin_rl/train_history.csv").exists(),
            "runs/operator_stability_report.json": (root / "runs/operator_stability_report.json").exists(),
            "runs/selberg_trace_report.json": (root / "runs/selberg_trace_report.json").exists(),
            "runs/artin_structured_report.json": (root / "runs/artin_structured_report.json").exists(),
            "runs/gemma_analysis.json": (root / "runs/gemma_analysis.json").exists(),
            "runs/v12_config_used.yaml": (root / "runs/v12_config_used.yaml").exists(),
            "logs/v12/*.log": bool(list((root / "logs/v12").glob("*.log"))),
        },
    }

    _append_jsonl(root / args.out_jsonl, entry)
    _append_markdown(root / args.out_md, entry)


if __name__ == "__main__":
    main()

