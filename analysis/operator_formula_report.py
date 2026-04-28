#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _write_text(path: Path, s: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(s, encoding="utf-8")


def _md(report: Dict[str, Any]) -> str:
    latex = report.get("latex", "")
    expr = report.get("expression_simplified", "")
    mse = report.get("mse")
    spec = report.get("spectral_error")
    fit = report.get("fit_info", {})

    return (
        "# Operator Reconstruction\n\n"
        "## Learned form\n\n"
        "### LaTeX\n\n"
        "```tex\n"
        f"{latex}\n"
        "```\n\n"
        "### Simplified Python form\n\n"
        "```text\n"
        f"{expr}\n"
        "```\n\n"
        "## Interpretation\n"
        "- kernel part: terms involving `exp(-d2 / k^2)` and `exp(-d2)`\n"
        "- Laplacian part: terms involving `d2`, `d`, and local_density proxies\n"
        "- potential: diagonal term `diag`\n\n"
        "## Accuracy\n"
        f"- MSE: {mse}\n"
        f"- spectral error: {spec}\n\n"
        "## Comparison\n"
        "- Original vs symbolic: reported via MSE and normalized spectrum error.\n\n"
        "## Fit metadata\n\n"
        "```json\n"
        f"{json.dumps(fit, indent=2)}\n"
        "```\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate operator formula report markdown")
    ap.add_argument("--in_json", type=str, default="runs/operator_symbolic.json")
    ap.add_argument("--out_md", type=str, default="runs/operator_formula_report.md")
    args = ap.parse_args()

    rep = _read_json(Path(args.in_json))
    if rep is None:
        raise FileNotFoundError(f"missing input json: {args.in_json}")

    _write_text(Path(args.out_md), _md(rep))


if __name__ == "__main__":
    main()

