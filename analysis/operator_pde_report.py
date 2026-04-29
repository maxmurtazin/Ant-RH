#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _write_text(path: Path, s: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(s, encoding="utf-8")


def _build_md(obj: Dict[str, Any]) -> str:
    return (
        "# Operator PDE Discovery Report\n\n"
        "## Inputs\n\n"
        "```json\n"
        f"{json.dumps(obj.get('inputs', {}), indent=2)}\n"
        "```\n\n"
        "## Eigenpair quality\n\n"
        "```json\n"
        f"{json.dumps(obj.get('eigenpair_quality', {}), indent=2)}\n"
        "```\n\n"
        "## Candidate term library\n\n"
        "- psi\n- laplacian_psi\n- rbf_kernel_psi\n- inv_distance_kernel_psi\n- potential_psi\n"
        "- density_psi\n- mean_length_psi\n- std_length_psi\n- psi_cubed\n- abs_psi_times_psi\n"
        "- inv_im_psi\n- log_im_psi\n\n"
        "## Selected equation\n\n"
        "```tex\n"
        f"{obj.get('equation_latex', '')}\n"
        "```\n\n"
        "## Coefficients\n\n"
        "```json\n"
        f"{json.dumps(obj.get('terms', []), indent=2)}\n"
        "```\n\n"
        "## Fit quality\n\n"
        "```json\n"
        f"{json.dumps(obj.get('fit_quality', {}), indent=2)}\n"
        "```\n\n"
        "## Interpretation\n\n"
        f"{obj.get('interpretation', 'LLM interpretation unavailable.')}\n\n"
        "## Limitations\n\n"
        + "\n".join(f"- {x}" for x in obj.get("limitations", []))
        + "\n\n## Next steps\n\n"
        + "\n".join(f"- {x}" for x in obj.get("next_steps", []))
        + "\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Render markdown report for operator PDE discovery")
    ap.add_argument("--in_json", type=str, default="runs/operator_pde_discovery.json")
    ap.add_argument("--out_md", type=str, default="runs/operator_pde_report.md")
    args = ap.parse_args()

    obj = _read_json(Path(args.in_json))
    if obj is None:
        raise FileNotFoundError(f"Missing or invalid input JSON: {args.in_json}")
    _write_text(Path(args.out_md), _build_md(obj))


if __name__ == "__main__":
    main()

