#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _normalize_baseline(stats: Dict[str, Any]) -> Dict[str, Any]:
    if "raw" in stats or "dedup" in stats or "deduplicated" in stats:
        out = dict(stats)
        if "deduplicated" in out and "dedup" not in out:
            out["dedup"] = out.pop("deduplicated")
        return out
    return {
        "raw": dict(stats),
        "dedup": dict(stats),
        "unique_candidate_ratio": 1.0,
        "duplicate_count": 0,
        "top_unique_candidates": list(stats.get("top_10_candidates", []))[:10],
    }


def _metric(stats: Dict[str, Any], key: str) -> Any:
    return stats.get(key, "n/a")


def main() -> None:
    root = Path(ROOT)
    out_dir = root / "runs/topological_lm"
    train_history = _read_csv(out_dir / "train_history.csv")
    baseline_rows = _read_csv(out_dir / "baseline_comparison.csv")
    try:
        eval_report = json.loads((out_dir / "eval_report.json").read_text(encoding="utf-8"))
    except Exception:
        eval_report = {}

    final_loss = train_history[-1]["train_loss"] if train_history else "n/a"
    final_ppl = train_history[-1]["perplexity"] if train_history else "n/a"
    baselines = {name: _normalize_baseline(stats) for name, stats in eval_report.get("baselines", {}).items()}
    topo = baselines.get("TopologicalLM-only", {"raw": {}, "dedup": {}, "unique_candidate_ratio": "n/a", "duplicate_count": "n/a"})
    topo_raw = topo.get("raw", {})
    topo_dedup = topo.get("dedup", {})
    advantage = eval_report.get("advantage_over_random", "n/a")
    diagnosis = eval_report.get("diagnosis", "No diagnosis available.")
    if isinstance(_metric(topo_dedup, "mean_reward"), (int, float)) and isinstance(_metric(topo_raw, "mean_reward"), (int, float)):
        if float(_metric(topo_dedup, "mean_reward")) < float(_metric(topo_raw, "mean_reward")):
            diagnosis += " Deduplication reduces reward, indicating mode collapse around a few high-reward candidates."
    unique_ratio = topo.get("unique_candidate_ratio", 1.0)
    if isinstance(unique_ratio, (int, float)) and float(unique_ratio) < 0.7:
        diagnosis += " Candidate diversity remains low."

    lines = [
        "# Topological LLM Report",
        "",
        "## 1. Goal",
        "Train a small topological next-token model over DTES-Braid episodes to heuristically generate braid/operator candidates.",
        "",
        "## 2. Dataset",
        "Dataset mixes ACO-derived DTES-Braid episodes, optional lab journal signals, and synthetic fallback episodes.",
        "",
        "## 3. Training",
        f"- Final train loss: `{final_loss}`",
        f"- Final perplexity: `{final_ppl}`",
        f"- Epochs logged: `{len(train_history)}`",
        "",
        "## 4. Candidate Validity",
    ]
    lines.extend(
        [
            "TopologicalLM raw:",
            f"- valid_braid_ratio: `{_metric(topo_raw, 'valid_braid_ratio')}`",
            f"- rejection_rate: `{_metric(topo_raw, 'executor_rejection_rate')}`",
            f"- reward_mean: `{_metric(topo_raw, 'reward_mean')}`",
            f"- reward_median: `{_metric(topo_raw, 'reward_median')}`",
            f"- reward_std: `{_metric(topo_raw, 'reward_std')}`",
            f"- unique_candidate_ratio: `{topo.get('unique_candidate_ratio', 'n/a')}`",
            f"- duplicate_count: `{topo.get('duplicate_count', 'n/a')}`",
            "",
            "TopologicalLM dedup:",
            f"- valid_braid_ratio: `{_metric(topo_dedup, 'valid_braid_ratio')}`",
            f"- rejection_rate: `{_metric(topo_dedup, 'executor_rejection_rate')}`",
            f"- reward_mean: `{_metric(topo_dedup, 'reward_mean')}`",
            f"- reward_median: `{_metric(topo_dedup, 'reward_median')}`",
            f"- reward_std: `{_metric(topo_dedup, 'reward_std')}`",
            "",
            "## 5. Baseline Comparison",
        ]
    )
    baseline_variants = [
        ("random", "raw", "random"),
        ("ACO-only existing best", "raw", "ACO-only existing best"),
        ("TopologicalLM-only", "raw", "TopologicalLM-only raw"),
        ("TopologicalLM-only", "dedup", "TopologicalLM-only dedup"),
        ("TopologicalLM + ACO refinement placeholder", "raw", "TopologicalLM + ACO raw"),
        ("TopologicalLM + ACO refinement placeholder", "dedup", "TopologicalLM + ACO dedup"),
    ]
    printed_any = False
    for baseline_name, mode_alias, label in baseline_variants:
        stats = baselines.get(baseline_name, {})
        mode_key = "dedup" if mode_alias == "dedup" else "raw"
        mode_stats = stats.get(mode_key, {})
        if not mode_stats:
            continue
        printed_any = True
        lines.append(
            f"- `{label}`: best_reward={_metric(mode_stats, 'best_reward')} "
            f"mean_reward={_metric(mode_stats, 'reward_mean')} median={_metric(mode_stats, 'reward_median')} "
            f"std={_metric(mode_stats, 'reward_std')} valid_ratio={_metric(mode_stats, 'valid_braid_ratio')}"
        )
    if not printed_any:
        lines.append("- No baseline comparison available.")
    lines.extend(
        [
            "",
            "## 6. Reward Diagnosis",
            f"- advantage_over_random: `{advantage}`",
            f"- Diagnosis: {diagnosis}",
            "- Report this honestly: if the mean-reward advantage is near zero, the model is not yet better than random under the current executor.",
            "",
            "## 7. Component Trends",
        ]
    )
    if baselines:
        for name, stats in baselines.items():
            cm = stats.get("dedup", stats.get("raw", {})).get("component_means", {})
            lines.append(
                f"- `{name}` component means: validity={cm.get('validity_score', 'n/a')}, "
                f"length={cm.get('length_score', 'n/a')}, spectral={cm.get('spectral_score', 'n/a')}, "
                f"stability={cm.get('stability_score', 'n/a')}, diversity={cm.get('diversity_score', 'n/a')}"
            )
    else:
        lines.append("- No component summaries available.")
    lines.extend(
        [
            "",
            "## 8. Top Candidates",
        ]
    )
    top_candidates = topo.get("top_unique_candidates", []) or topo_dedup.get("top_10_candidates", []) or topo_raw.get("top_10_candidates", [])
    if top_candidates:
        for item in top_candidates[:10]:
            lines.append(
                f"- word={item.get('word', [])} reward={item.get('reward', 'n/a')} "
                f"valid={item.get('valid', 'n/a')} spectral_error={item.get('spectral_error', 'n/a')}"
            )
    else:
        lines.append("- No top candidates recorded.")
    lines.extend(["", "## Physics Diagnostics", "", "For each baseline (dedup if available):"])
    if baselines:
        for name, stats in baselines.items():
            mode_stats = stats.get("dedup", stats.get("raw", {})) or {}
            lines.append(
                f"- `{name}`: self_adjoint_status=`{mode_stats.get('self_adjoint_status', 'n/a')}` "
                f"spectral_status=`{mode_stats.get('spectral_status', 'n/a')}` "
                f"otoc_indicator=`{mode_stats.get('otoc_indicator', 'n/a')}` "
                f"r_mean=`{mode_stats.get('r_mean', 'n/a')}`"
            )
    else:
        lines.append("- No physics diagnostics available.")

    lines.extend(
        [
            "",
            "## 9. Limitations",
            "This is a heuristic generator and does not prove RH.",
            "The MVP uses symbolic token bins and proxy executor rewards; it is not a mathematically validated proof engine.",
            "If TopologicalLM and random remain close under bounded rewards, the current model or dataset is not yet carrying useful structure.",
            "",
            "## 10. Next steps",
            "- Add real DTES-braid trajectories instead of synthetic fallbacks.",
            "- Replace proxy executor terms with full structured operator evaluation.",
            "- Add ACO/RL refinement loop over generated candidates.",
            "- Track whether advantage_over_random improves after each executor or dataset revision.",
            "- Track physics diagnostics (self-adjointness, spectral non-degeneracy, r-statistic) across baselines and over time.",
            "",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
