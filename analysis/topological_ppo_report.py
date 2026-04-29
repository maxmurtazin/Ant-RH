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


def _metric(row: Dict[str, Any], key: str, default: str = "missing") -> str:
    return str(row.get(key, default))


def main() -> None:
    root = Path(ROOT)
    out_dir = root / "runs/topological_ppo"
    train_history = _read_csv(out_dir / "train_history.csv")
    eval_report = {}
    try:
        eval_report = json.loads((out_dir / "eval_report.json").read_text(encoding="utf-8"))
    except Exception:
        eval_report = {}
    try:
        best_candidates = json.loads((out_dir / "best_candidates.json").read_text(encoding="utf-8"))
    except Exception:
        best_candidates = []

    final = train_history[-1] if train_history else {}
    baselines = eval_report.get("baselines", {})
    ppo_stats = baselines.get("PPO TopologicalLM", {})
    pretrained_stats = baselines.get("TopologicalLM-only", {})
    random_stats = baselines.get("random", {})
    comparison_mode = str(eval_report.get("comparison_mode", "dedup"))
    ppo_mode = ppo_stats.get(comparison_mode, {})
    pretrained_mode = pretrained_stats.get(comparison_mode, {})
    random_mode = random_stats.get(comparison_mode, {})

    lines = [
        "# Topological PPO Report",
        "",
        "## 1. Goal",
        "Fine-tune the TopologicalLM policy with PPO using executor rewards plus validity and uniqueness bonuses.",
        "",
        "## 2. Training Reward Curve",
        f"- Updates logged: `{len(train_history)}`",
        f"- Final reward_mean: `{_metric(final, 'reward_mean')}`",
        f"- Final reward_median: `{_metric(final, 'reward_median')}`",
        f"- Final reward_max: `{_metric(final, 'reward_max')}`",
        "",
        "## 3. Validity and Diversity",
        f"- Final valid_ratio: `{_metric(final, 'valid_ratio')}`",
        f"- Final unique_ratio: `{_metric(final, 'unique_ratio')}`",
        f"- Final duplicate_count: `{_metric(final, 'duplicate_count')}`",
        "",
        "## 4. KL Trend",
        f"- Final kl_mean: `{_metric(final, 'kl_mean')}`",
        f"- Final entropy: `{_metric(final, 'entropy')}`",
        "",
        "## 5. Comparison",
        f"- Comparison mode: `{comparison_mode}`",
        f"- PPO reward_mean: `{ppo_mode.get('reward_mean', 'missing')}`",
        f"- Pretrained reward_mean: `{pretrained_mode.get('reward_mean', 'missing')}`",
        f"- Random reward_mean: `{random_mode.get('reward_mean', 'missing')}`",
        f"- advantage_over_random: `{eval_report.get('advantage_over_random', 'missing')}`",
        "",
        "## 6. Best Candidates",
    ]
    if best_candidates:
        for item in best_candidates[:10]:
            lines.append(
                f"- update={item.get('update', 'missing')} word={item.get('word', [])} "
                f"reward={item.get('reward', 'missing')} valid={item.get('executor_result', {}).get('valid', 'missing')}"
            )
    else:
        lines.append("- No best candidates saved.")

    lines.extend(
        [
            "",
            "## 7. Limitations",
            "- PPO is optimizing a proxy executor reward, not a full operator-level objective.",
            "- If PPO reward_mean does not exceed pretrained and random baselines, the result should be treated as a failed optimization attempt.",
            "- Low unique_ratio indicates residual collapse even with KL regularization.",
            "- Empty or invalid generated words are skipped, which can reduce effective batch size.",
            "",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
