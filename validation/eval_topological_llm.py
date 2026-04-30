#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.braid_tokenizer import BraidTokenizer
from core.dtes_braid_executor import evaluate_braid_candidate
from core.topological_llm import generate
from core.topological_ppo import (
    build_prefix_ids,
    get_device,
    load_ppo_policy,
    load_pretrained_lm,
    sample_policy_episode,
)


def _random_word(max_sigma: int = 6) -> List[int]:
    vals = [a for a in range(-max_sigma, max_sigma + 1) if a != 0]
    return [random.choice(vals) for _ in range(random.randint(3, 8))]


def _bool_arg(x: Any) -> bool:
    return str(x).lower() in {"1", "true", "yes", "y", "t"}


def _safe_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {
            "reward_mean": 0.0,
            "reward_median": 0.0,
            "reward_std": 0.0,
            "reward_min": 0.0,
            "reward_max": 0.0,
        }
    return {
        "reward_mean": float(sum(values) / len(values)),
        "reward_median": float(statistics.median(values)),
        "reward_std": float(statistics.pstdev(values)) if len(values) > 1 else 0.0,
        "reward_min": float(min(values)),
        "reward_max": float(max(values)),
    }


def _mode(items: List[str]) -> str:
    if not items:
        return "n/a"
    counts: Dict[str, int] = {}
    for x in items:
        k = str(x)
        counts[k] = counts.get(k, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def _physics_summary(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [r for r in results if r.get("valid")]
    pool = valid if valid else results

    sa_errors = []
    sa_statuses = []
    spectral_statuses = []
    r_means = []
    for r in pool:
        try:
            sa_errors.append(float(r.get("self_adjoint_error", float("inf"))))
        except Exception:
            pass
        if r.get("self_adjoint_status") is not None:
            sa_statuses.append(str(r.get("self_adjoint_status")))
        if r.get("spectral_status") is not None:
            spectral_statuses.append(str(r.get("spectral_status")))
        try:
            rv = r.get("r_mean", None)
            if rv is not None:
                r_means.append(float(rv))
        except Exception:
            pass

    sa_err_mean = float(sum(sa_errors) / len(sa_errors)) if sa_errors else float("inf")
    if sa_err_mean < 1e-6:
        sa_status = "ok"
    elif sa_err_mean < 1e-3:
        sa_status = "approx"
    else:
        sa_status = "broken"

    spectral_status = _mode(spectral_statuses) if spectral_statuses else "degenerate"

    r_mean = float(sum(r_means) / len(r_means)) if r_means else None
    if r_mean is None:
        otoc = "integrable"
    elif r_mean < 0.4:
        otoc = "integrable"
    elif r_mean < 0.5:
        otoc = "intermediate"
    else:
        otoc = "chaotic"

    return {
        "self_adjoint_status": sa_status,
        "spectral_status": spectral_status,
        "otoc_indicator": otoc,
        "r_mean": r_mean,
    }


def _summarize_one(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [r for r in results if r.get("valid")]
    rejected = [r for r in results if not r.get("valid")]
    rewards = [float(r.get("reward", -10.0)) for r in results]
    spec = [float(r.get("spectral_error", float("inf"))) for r in valid]
    component_names = [
        "validity_score",
        "length_score",
        "spectral_score",
        "stability_score",
        "diversity_score",
    ]
    component_means = {
        name: float(sum(float(r.get(name, 0.0)) for r in results) / max(1, len(results))) for name in component_names
    }
    top_10_candidates = sorted(results, key=lambda r: float(r.get("reward", -10.0)), reverse=True)[:10]
    summary = {
        "valid_braid_ratio": len(valid) / max(1, len(results)),
        "executor_rejection_rate": len(rejected) / max(1, len(results)),
        "best_reward": max(rewards) if rewards else float("-inf"),
        "mean_reward": sum(rewards) / max(1, len(rewards)),
        "best_spectral_error": min(spec) if spec else float("inf"),
        "sample_efficiency_proxy": (len(valid) / max(1, len(results))) * (max(rewards) if rewards else 0.0),
        "component_means": component_means,
        "top_10_candidates": top_10_candidates,
    }
    summary.update(_physics_summary(results))
    summary.update(_safe_stats(rewards))
    return summary


def _deduplicate_results(results: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    seen = set()
    deduped: List[Dict[str, Any]] = []
    duplicate_count = 0
    for item in results:
        try:
            key = tuple(int(x) for x in item.get("word", []))
        except Exception:
            key = tuple()
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)
        deduped.append(item)
    return deduped, duplicate_count


def _summarize(results: List[Dict[str, Any]], attempts_used: int | None = None) -> Dict[str, Any]:
    deduped, duplicate_count = _deduplicate_results(results)
    raw_summary = _summarize_one(results)
    dedup_summary = _summarize_one(deduped)
    denom = max(1, attempts_used if attempts_used is not None else len(results))
    unique_candidate_ratio = len(deduped) / denom
    return {
        "raw": raw_summary,
        "dedup": dedup_summary,
        "unique_candidate_ratio": unique_candidate_ratio,
        "duplicate_count": duplicate_count,
        "top_unique_candidates": sorted(deduped, key=lambda r: float(r.get("reward", -10.0)), reverse=True)[:10],
    }


def _normalize_baseline_structure(baseline: Dict[str, Any]) -> Dict[str, Any]:
    if "raw" in baseline or "dedup" in baseline:
        out = dict(baseline)
        if "deduplicated" in out and "dedup" not in out:
            out["dedup"] = out.pop("deduplicated")
        return out
    return {
        "raw": dict(baseline),
        "dedup": dict(baseline),
        "unique_candidate_ratio": 1.0,
        "duplicate_count": 0,
        "top_unique_candidates": list(baseline.get("top_10_candidates", []))[:10],
    }


def _collect_pretrained_results(
    model,
    tokenizer: BraidTokenizer,
    num_candidates: int,
    max_attempts: int,
    args: argparse.Namespace,
) -> tuple[List[Dict[str, Any]], int]:
    prefix_ids = build_prefix_ids(tokenizer)
    results: List[Dict[str, Any]] = []
    seen_words = set()
    attempts_used = 0
    while len(results) < int(num_candidates) and attempts_used < max_attempts:
        attempts_used += 1
        gen_ids = generate(
            model,
            tokenizer,
            prefix_ids,
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_k=int(args.top_k),
            top_p=float(args.top_p),
            no_repeat_ngram_size=int(args.no_repeat_ngram_size),
            repetition_penalty=float(args.repetition_penalty),
        )
        decoded = tokenizer.decode(gen_ids)
        word = []
        for tok in decoded.split():
            if tok.startswith("SIGMA_") and (tok.endswith("_PLUS") or tok.endswith("_MINUS")):
                parts = tok.split("_")
                if len(parts) == 3:
                    try:
                        idx = int(parts[1])
                    except Exception:
                        continue
                    sign = 1 if parts[2] == "PLUS" else -1
                    word.append(sign * idx)
        key = tuple(word)
        if _bool_arg(args.dedup) and key in seen_words:
            continue
        seen_words.add(key)
        results.append(evaluate_braid_candidate(word))
    return results, attempts_used


def _collect_ppo_results(
    policy,
    ref_model,
    tokenizer: BraidTokenizer,
    num_candidates: int,
    max_attempts: int,
    args: argparse.Namespace,
) -> tuple[List[Dict[str, Any]], int]:
    prefix_ids = build_prefix_ids(tokenizer)
    results: List[Dict[str, Any]] = []
    seen_words = set()
    attempts_used = 0
    while len(results) < int(num_candidates) and attempts_used < max_attempts:
        attempts_used += 1
        sample = sample_policy_episode(
            policy,
            ref_model,
            tokenizer,
            prefix_ids,
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_k=int(args.top_k),
            top_p=float(args.top_p),
            no_repeat_ngram_size=int(args.no_repeat_ngram_size),
            repetition_penalty=float(args.repetition_penalty),
        )
        word = [int(x) for x in sample.get("word", [])]
        key = tuple(word)
        if _bool_arg(args.dedup) and key in seen_words:
            continue
        seen_words.add(key)
        results.append(evaluate_braid_candidate(word))
    return results, attempts_used


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate topological tiny LM")
    ap.add_argument("--model", type=str, default="runs/topological_lm/model.pt")
    ap.add_argument("--tokenizer", type=str, default="runs/topological_lm/tokenizer.json")
    ap.add_argument("--ppo_model", type=str, default="")
    ap.add_argument("--num_candidates", type=int, default=200)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=1.1)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--dedup", type=str, default="True")
    ap.add_argument("--no_repeat_ngram_size", type=int, default=3)
    ap.add_argument("--repetition_penalty", type=float, default=1.2)
    args = ap.parse_args()

    root = Path(ROOT)
    model_path = root / str(args.model)
    tokenizer_path = root / str(args.tokenizer)
    ppo_path = root / str(args.ppo_model) if str(args.ppo_model).strip() else None
    out_dir = ppo_path.parent if ppo_path else model_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = BraidTokenizer.load(tokenizer_path)
    device = get_device()
    pretrained_model, _ = load_pretrained_lm(model_path, tokenizer, device)
    pretrained_model.eval()

    random_results = [evaluate_braid_candidate(_random_word()) for _ in range(int(args.num_candidates))]

    best_path = root / "runs/artin_aco_best.json"
    try:
        best_payload = json.loads(best_path.read_text(encoding="utf-8"))
        aco_best_results = [evaluate_braid_candidate(word) for word in best_payload.get("best_words", [])[:1]]
    except Exception:
        aco_best_results = [evaluate_braid_candidate([4, -2, -1, -1])]

    max_attempts = int(args.num_candidates) * 10
    pretrained_results, pretrained_attempts = _collect_pretrained_results(
        pretrained_model, tokenizer, int(args.num_candidates), max_attempts, args
    )

    baselines: Dict[str, Dict[str, Any]] = {
        "random": _summarize(random_results, attempts_used=int(args.num_candidates)),
        "ACO-only existing best": _summarize(aco_best_results, attempts_used=len(aco_best_results)),
        "TopologicalLM-only": _summarize(pretrained_results, attempts_used=pretrained_attempts),
    }

    candidates_payload: List[Dict[str, Any]] = [{"baseline": "TopologicalLM-only", "word": r.get("word", []), "reward": r.get("reward")} for r in pretrained_results[:50]]
    attempts_used = {"TopologicalLM-only": pretrained_attempts}

    if ppo_path:
        ppo_policy, _ = load_ppo_policy(ppo_path, device)
        ppo_results, ppo_attempts = _collect_ppo_results(
            ppo_policy, pretrained_model, tokenizer, int(args.num_candidates), max_attempts, args
        )
        ppo_refine_results = []
        for item in ppo_results:
            word = list(item.get("word", []))
            if len(word) >= 3:
                word = word[: min(len(word), 6)]
                if len(word) >= 2 and word[-1] == word[-2]:
                    word[-1] = -word[-1]
            ppo_refine_results.append(evaluate_braid_candidate(word))
        baselines["PPO TopologicalLM"] = _summarize(ppo_results, attempts_used=ppo_attempts)
        baselines["PPO + ACO placeholder"] = _summarize(ppo_refine_results, attempts_used=ppo_attempts)
        attempts_used["PPO TopologicalLM"] = ppo_attempts
        attempts_used["PPO + ACO placeholder"] = ppo_attempts
        candidates_payload.extend(
            [{"baseline": "PPO TopologicalLM", "word": r.get("word", []), "reward": r.get("reward")} for r in ppo_results[:50]]
        )

    baselines = {name: _normalize_baseline_structure(stats) for name, stats in baselines.items()}
    comparison_target = "PPO TopologicalLM" if "PPO TopologicalLM" in baselines else "TopologicalLM-only"
    advantage_over_random = float(
        baselines[comparison_target]["dedup"]["reward_mean"] - baselines["random"]["dedup"]["reward_mean"]
    )
    diagnosis = (
        f"{comparison_target} is not yet better than random."
        if advantage_over_random <= 0.0
        else f"{comparison_target} improves mean reward over random."
    )
    if baselines[comparison_target]["unique_candidate_ratio"] <= 0.5:
        diagnosis += " Candidate diversity is still low."

    eval_report = {
        "num_candidates": int(args.num_candidates),
        "device": str(device),
        "temperature": float(args.temperature),
        "top_k": int(args.top_k),
        "top_p": float(args.top_p),
        "dedup": bool(_bool_arg(args.dedup)),
        "comparison_mode": "dedup",
        "no_repeat_ngram_size": int(args.no_repeat_ngram_size),
        "repetition_penalty": float(args.repetition_penalty),
        "attempts_used": attempts_used,
        "baselines": baselines,
        "advantage_over_random": advantage_over_random,
        "diagnosis": diagnosis,
    }

    (out_dir / "eval_report.json").write_text(json.dumps(eval_report, indent=2), encoding="utf-8")
    (out_dir / "generated_candidates.json").write_text(json.dumps(candidates_payload, indent=2), encoding="utf-8")
    with (out_dir / "baseline_comparison.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "baseline",
                "mode",
                "reward_mean",
                "reward_median",
                "reward_max",
                "valid_ratio",
                "unique_ratio",
                "duplicate_count",
                "advantage_over_random",
            ]
        )
        for name, stats in baselines.items():
            for mode in ["raw", "dedup"]:
                mode_stats = stats.get(mode, {})
                writer.writerow(
                    [
                        name,
                        mode,
                        mode_stats.get("reward_mean", 0.0),
                        mode_stats.get("reward_median", 0.0),
                        mode_stats.get("reward_max", 0.0),
                        mode_stats.get("valid_braid_ratio", 0.0),
                        stats.get("unique_candidate_ratio", 1.0),
                        stats.get("duplicate_count", 0),
                        advantage_over_random if name == comparison_target else "",
                    ]
                )


if __name__ == "__main__":
    main()
