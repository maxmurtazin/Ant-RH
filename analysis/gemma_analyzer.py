#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.llm_runner import LLMRunner

DTYPE = np.float64


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _write_text(path: Path, s: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(s, encoding="utf-8")


def _load_csv_columns(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        cols: Dict[str, List[float]] = {}
        for row in reader:
            for k, v in row.items():
                if v is None:
                    continue
                vv = v.strip()
                if vv == "":
                    continue
                try:
                    x = float(vv)
                except Exception:
                    continue
                cols.setdefault(k, []).append(x)
    return {k: np.asarray(v, dtype=DTYPE) for k, v in cols.items()}


def _slope(y: np.ndarray) -> float:
    y = np.asarray(y, dtype=DTYPE).reshape(-1)
    y = y[np.isfinite(y)]
    n = int(y.size)
    if n < 3:
        return 0.0
    x = np.arange(n, dtype=DTYPE)
    x = x - x.mean()
    y = y - y.mean()
    denom = float(np.dot(x, x)) + 1e-12
    return float(np.dot(x, y) / denom)


def _trend_label(slope: float, scale: float) -> str:
    if scale <= 0:
        scale = 1.0
    s = slope / (scale + 1e-12)
    if s < -1e-3:
        return "decreasing"
    if s > 1e-3:
        return "increasing"
    return "flat"


def _robust_tail_stats(x: np.ndarray, tail: int = 10) -> Dict[str, float]:
    x = np.asarray(x, dtype=DTYPE).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"last": float("nan"), "mean_tail": float("nan"), "min": float("nan")}
    t = x[-min(int(tail), int(x.size)) :]
    return {"last": float(x[-1]), "mean_tail": float(np.mean(t)), "min": float(np.min(x))}


def _markdown_report(summary: Dict[str, Any]) -> str:
    j = json.dumps(summary, indent=2)
    return (
        "# Ant-RH Analysis\n\n"
        "## Learning\n"
        f"- learning: {summary.get('learning')}\n"
        f"- aco_best_trend: {summary.get('aco', {}).get('best_loss_trend')}\n"
        f"- rl_reward_trend: {summary.get('rl', {}).get('mean_reward_trend')}\n\n"
        "## Operator quality\n"
        f"- operator_quality: {summary.get('operator_quality')}\n"
        f"- spectral_loss: {summary.get('operator', {}).get('spectral_loss')}\n"
        f"- spacing_loss: {summary.get('operator', {}).get('spacing_loss')}\n"
        f"- stability_flags: {summary.get('operator', {}).get('stability_flags')}\n\n"
        "## Issues\n"
        f"- main_issue: {summary.get('main_issue')}\n"
        f"- warnings: {summary.get('warnings')}\n\n"
        "## Recommendations\n"
        + "\n".join([f"- {s}" for s in (summary.get("suggestions") or [])])
        + "\n\n"
        "## Raw JSON\n\n"
        "```json\n"
        f"{j}\n"
        "```\n"
    )


def _build_prompt(aco_summary: Dict[str, Any], rl_summary: Dict[str, Any], op_summary: Dict[str, Any], sel_summary: Dict[str, Any]) -> str:
    return (
        "You are analyzing Ant-RH results.\n\n"
        f"ACO:\n{json.dumps(aco_summary, indent=2)}\n\n"
        f"RL:\n{json.dumps(rl_summary, indent=2)}\n\n"
        f"Operator:\n{json.dumps(op_summary, indent=2)}\n\n"
        f"Selberg:\n{json.dumps(sel_summary, indent=2)}\n\n"
        "Answer in JSON:\n\n"
        "{\n"
        ' "learning": true/false,\n'
        ' "operator_quality": "low|medium|high",\n'
        ' "main_issue": "...",\n'
        ' "suggestions": [...]\n'
        "}\n"
    )


def _extract_json_blob(text: str) -> Optional[Dict[str, Any]]:
    # Find first {...} block
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    blob = m.group(0)
    try:
        return json.loads(blob)
    except Exception:
        return None


def _heuristic_analysis(aco: Dict[str, Any], rl: Dict[str, Any], op: Dict[str, Any], sel: Dict[str, Any]) -> Dict[str, Any]:
    warnings: List[str] = []
    suggestions: List[str] = []

    aco_trend = str(aco.get("best_loss_trend", "unknown"))
    rl_trend = str(rl.get("mean_reward_trend", "unknown"))

    learning = False
    if aco_trend == "decreasing" or rl_trend == "increasing":
        learning = True

    spec = float(op.get("spectral_loss", float("inf")))
    spacing = float(op.get("spacing_loss", float("inf")))
    cond = float(op.get("condition_proxy_after", float("inf")))
    eig_ok = bool(op.get("eigh_success", True))

    if not eig_ok:
        warnings.append("eigensolver_failed")
    if not np.isfinite(spec):
        warnings.append("spectral_loss_nonfinite")
    if cond > 1e10:
        warnings.append("operator_ill_conditioned")

    if spec < 0.5 and spacing < 0.5:
        opq = "high"
    elif spec < 2.0 and spacing < 2.0:
        opq = "medium"
    else:
        opq = "low"

    main_issue = "unknown"
    if opq == "low":
        main_issue = "operator_spectrum_mismatch"
        suggestions.extend(
            [
                "Increase operator stabilization (diagonal shift/jitter) and ensure self-adjoint projection before eigensolve.",
                "Use structured operator basis expansion (V12.6) and optimize weights instead of arbitrary kernels.",
                "Tune geodesic selection: increase top-K geodesics and filter by primitive/hyperbolic length caps.",
            ]
        )
    elif not learning:
        main_issue = "learning_stalled"
        suggestions.extend(
            [
                "Increase exploration (ACO beta/pheromone evaporation) or add planner-guided proposals.",
                "For RL, increase steps_per_update or entropy_coef to avoid premature collapse.",
                "Use a smaller operator size during early training to reduce evaluation noise/cost.",
            ]
        )
    else:
        main_issue = "scaling_and_alignment"
        suggestions.extend(
            [
                "Normalize spectra (z-score) consistently and compare spacing statistics after normalization.",
                "Log per-stage contributions (Selberg vs spectral vs spacing) to identify dominating term.",
                "Cache operator builds and reuse geodesic banks between ACO and RL updates.",
            ]
        )

    if sel.get("relative_error", 0.0) is not None:
        try:
            rel = float(sel.get("relative_error", 0.0))
            if rel > 1.0:
                warnings.append("selberg_relative_error_large")
        except Exception:
            pass

    return {
        "learning": bool(learning),
        "operator_quality": opq,
        "main_issue": str(main_issue),
        "suggestions": suggestions[:3],
        "warnings": warnings,
        "aco": aco,
        "rl": rl,
        "operator": op,
        "selberg": sel,
    }


class GemmaAnalyzer:
    def __init__(self) -> None:
        self.runner = LLMRunner(
            model_path="/Users/machome/models/gemma/gemma-3-1b-it-Q5_K_M.gguf",
            n_ctx=2048,
            n_threads=6,
            n_gpu_layers=0,
            timeout_s=60.0,
        )

    def analyze(self, metrics_dict: Dict[str, Any]) -> Dict[str, Any]:
        aco_summary = metrics_dict.get("aco_summary", {}) or {}
        rl_summary = metrics_dict.get("rl_summary", {}) or {}
        op_summary = metrics_dict.get("operator_summary", {}) or {}
        sel_summary = metrics_dict.get("selberg_summary", {}) or {}

        prompt = _build_prompt(aco_summary, rl_summary, op_summary, sel_summary)
        try:
            resp = self.runner.generate(prompt, max_tokens=256, temperature=0.3)
            parsed = _extract_json_blob(resp)
            if isinstance(parsed, dict) and {"learning", "operator_quality", "main_issue", "suggestions"} <= set(parsed.keys()):
                out = dict(parsed)
                out.setdefault("warnings", [])
                return out
        except Exception:
            pass

        # Fallback heuristic: never crash pipeline
        return _heuristic_analysis(aco_summary, rl_summary, op_summary, sel_summary)


def main() -> None:
    ap = argparse.ArgumentParser(description="Gemma Analyzer (logs -> structured insight)")
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--backend", type=str, default="llama_cpp", choices=["llama_cpp", "mock"])
    ap.add_argument("--llama_cli", type=str, default="/Users/machome/llama.cpp/llama-cli")
    ap.add_argument("--model_path", type=str, default="/Users/machome/models/gemma/gemma-3-1b-it-Q5_K_M.gguf")
    ap.add_argument("--out_json", type=str, default="runs/gemma_analysis.json")
    ap.add_argument("--out_md", type=str, default="runs/gemma_analysis.md")
    args = ap.parse_args()

    if args.test:
        analyzer = GemmaAnalyzer()
        if str(args.backend).lower() == "llama_cpp":
            analyzer.runner = LLMRunner(
                model_path=str(args.model_path),
                llama_cli=str(args.llama_cli),
                n_ctx=2048,
                n_threads=6,
                n_gpu_layers=0,
                timeout_s=60.0,
            )
        metrics = {
            "aco_summary": {"best_loss_trend": "decreasing", "best_loss_stats": {"last": 0.5}},
            "rl_summary": {"mean_reward_trend": "increasing", "mean_reward_stats": {"last": 1.2}},
            "operator_summary": {"spectral_loss": 0.9, "spacing_loss": 0.7, "eigh_success": True, "condition_proxy_after": 1e6},
            "selberg_summary": {"relative_error": 0.2, "loss": 0.1},
        }
        out = analyzer.analyze(metrics)
        _write_json(Path(args.out_json), out)
        _write_text(Path(args.out_md), _markdown_report(out))
        return

    aco_csv = Path("runs/artin_aco_history.csv")
    rl_csv = Path("runs/artin_rl/train_history.csv")
    op_json = Path("runs/operator_stability_report.json")
    sel_json = Path("runs/selberg_trace_report.json")
    structured_json = Path("runs/artin_structured_report.json")

    aco_cols = _load_csv_columns(aco_csv)
    rl_cols = _load_csv_columns(rl_csv)
    op = _read_json(op_json) or {}
    sel = _read_json(sel_json) or {}
    structured = _read_json(structured_json) or {}

    aco_best = aco_cols.get("best_loss", np.array([], dtype=DTYPE))
    aco_mean = aco_cols.get("mean_loss", np.array([], dtype=DTYPE))
    rl_mean_reward = rl_cols.get("mean_reward", np.array([], dtype=DTYPE))

    aco_best_slope = _slope(aco_best)
    aco_mean_slope = _slope(aco_mean)
    rl_reward_slope = _slope(rl_mean_reward)

    aco_summary = {
        "path": str(aco_csv),
        "n": int(aco_best.size) if aco_best.size else 0,
        "best_loss_stats": _robust_tail_stats(aco_best),
        "mean_loss_stats": _robust_tail_stats(aco_mean),
        "best_loss_trend": _trend_label(aco_best_slope, float(np.std(aco_best) + 1e-12) if aco_best.size else 1.0),
        "mean_loss_trend": _trend_label(aco_mean_slope, float(np.std(aco_mean) + 1e-12) if aco_mean.size else 1.0),
    }

    rl_summary = {
        "path": str(rl_csv),
        "n": int(rl_mean_reward.size) if rl_mean_reward.size else 0,
        "mean_reward_stats": _robust_tail_stats(rl_mean_reward),
        "mean_reward_trend": _trend_label(rl_reward_slope, float(np.std(rl_mean_reward) + 1e-12) if rl_mean_reward.size else 1.0),
    }

    op_summary = {
        "path": str(op_json),
        "eigh_success": bool(op.get("eigh_success", op.get("eigh_after", {}).get("eigh_success", True))),
        "spectral_loss": float(op.get("spectral_loss", op.get("spectral_loss", float("nan")))),
        "spacing_loss": float(op.get("spacing_loss", op.get("spacing_loss", float("nan")))),
        "condition_proxy_after": float(op.get("condition_proxy_after", op.get("condition_proxy_after", float("nan")))),
        "stability_flags": {
            "nan_count": int(op.get("nan_count", 0)) if isinstance(op.get("nan_count", 0), (int, float)) else 0,
            "inf_count": int(op.get("inf_count", 0)) if isinstance(op.get("inf_count", 0), (int, float)) else 0,
        },
    }
    if isinstance(op.get("eigh_after"), dict):
        op_summary["eigh_backend_used"] = op["eigh_after"].get("eigh_backend_used")

    sel_summary = {
        "path": str(sel_json),
        "S_spec": sel.get("S_spec"),
        "S_geo": sel.get("S_geo"),
        "loss": sel.get("loss"),
        "relative_error": sel.get("relative_error"),
    }

    if structured:
        op_summary["structured"] = {
            "path": str(structured_json),
            "final_loss": structured.get("final_loss"),
            "spectral_loss": structured.get("spectral_loss"),
            "spacing_loss": structured.get("spacing_loss"),
            "weight_norm": structured.get("weight_norm"),
        }

    analyzer = GemmaAnalyzer()
    if str(args.backend).lower() == "mock":
        def _no_llm(_: str, max_tokens: int = 256, temperature: float = 0.7) -> str:
            return ""
        analyzer.runner.generate = _no_llm  # type: ignore
    else:
        analyzer.runner = LLMRunner(
            model_path=str(args.model_path),
            llama_cli=str(args.llama_cli),
            n_ctx=2048,
            n_threads=6,
            n_gpu_layers=0,
            timeout_s=60.0,
        )
    metrics = {
        "aco_summary": aco_summary,
        "rl_summary": rl_summary,
        "operator_summary": op_summary,
        "selberg_summary": sel_summary,
    }
    final = analyzer.analyze(metrics)

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    _write_json(out_json, final)
    _write_text(out_md, _markdown_report(final))


if __name__ == "__main__":
    main()

