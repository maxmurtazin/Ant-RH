#!/usr/bin/env python3
"""
V13G: validate full-run top-5 from ``v13_top_candidates.json`` (word-sensitive path), then run Pareto selection.

  python3 scripts/validate_top5_v13.py \\
    --input_json runs/v13_formula_report/v13_top_candidates.json \\
    --out_dir runs/v13_top5_validation \\
    --zeros 128 --dim 128

  python3 scripts/pareto_select_v13.py \\
    --input_json runs/v13_top5_validation/validation_results.json \\
    --out_dir runs/v13_top5_pareto
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from typing import Any, Dict, List, Optional

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

if not os.environ.get("MPLCONFIGDIR"):
    _mpl_cfg = Path(ROOT) / ".mpl_cache"
    try:
        _mpl_cfg.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(_mpl_cfg)
    except OSError:
        pass


def _load_validate_module() -> Any:
    path = Path(ROOT) / "scripts" / "validate_candidates_v13.py"
    spec = importlib.util.spec_from_file_location("_validate_candidates_v13", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_validate_candidates_v13"] = mod
    spec.loader.exec_module(mod)
    return mod


def load_top_candidates(path: Path) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and isinstance(data.get("top_candidates"), list):
        return list(data["top_candidates"])
    if isinstance(data, list):
        return list(data)
    raise ValueError(f"expected object with 'top_candidates' or a list in {path}")


def rows_to_validation_candidates(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    out: List[Dict[str, Any]] = []
    warnings: List[str] = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            warnings.append(f"row_{i}:not_a_dict")
            continue
        seed = row.get("seed")
        sid = f"seed_{int(seed)}" if seed is not None else f"seed_unknown_{i}"
        bw = row.get("best_words")
        if isinstance(bw, str):
            try:
                bw = json.loads(bw)
            except json.JSONDecodeError:
                bw = None
        word: Optional[List[int]] = None
        if isinstance(bw, list) and bw:
            w0 = bw[0]
            if isinstance(w0, list):
                try:
                    word = [int(x) for x in w0]
                except (TypeError, ValueError):
                    word = None
        if not word:
            warnings.append(f"{sid}:missing_or_invalid_best_words")
            continue
        out.append({"id": sid, "word": word})
    return out, warnings


def run_validation(
    *,
    candidates: List[Dict[str, Any]],
    out_dir: Path,
    zeros_n: int,
    dim: int,
    eps: float,
    geo_weight: float,
    geo_sigma: float,
    potential_weight: float,
    seed: int,
    save_plots: bool,
) -> None:
    v = _load_validate_module()
    if v.build_word_sensitive_operator is None:
        raise SystemExit("word_sensitive builder unavailable (import core.artin_operator_word_sensitive failed)")

    out_dir.mkdir(parents=True, exist_ok=True)
    zeros = v._load_zeros(int(zeros_n))
    if zeros.size < int(dim):
        raise ValueError(f"need at least --dim zeros ({dim}), got {zeros.size}")

    results: List[Dict[str, Any]] = []
    H_mats: List[Optional[Any]] = []
    for c in candidates:
        r, Hm = v.validate_one(
            c,
            zeros=zeros,
            dim=int(dim),
            eps=float(eps),
            seed=int(seed),
            geo_weight=float(geo_weight),
            geo_sigma=float(geo_sigma),
            potential_weight=float(potential_weight),
        )
        results.append(r)
        H_mats.append(Hm)

    ids_list = [str(r["id"]) for r in results]
    pairwise = v.pairwise_operator_fro_dists(ids_list, H_mats)
    v.warn_identical_operator_hashes(results)
    results_sorted = sorted(results, key=v._sort_key)

    payload = {
        "zeros_n": int(zeros.size),
        "dim": int(dim),
        "operator_builder": "word_sensitive",
        "source": "validate_top5_v13",
        "eps": float(eps),
        "geo_weight": float(geo_weight),
        "geo_sigma": float(geo_sigma),
        "potential_weight": float(potential_weight),
        "seed": int(seed),
        "pairwise_operator_distances": pairwise,
        "candidates": results_sorted,
    }
    with open(out_dir / "validation_results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=True)

    if save_plots:
        for r in results:
            cid = str(r["id"])
            safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", cid)
            word = r.get("word_used") or r.get("word") or []
            try:
                H, _ = v.build_operator_aco_word_sensitive(
                    [int(x) for x in word],
                    n_points=int(dim),
                    eps=float(eps),
                    geo_weight=float(geo_weight),
                    geo_sigma=float(geo_sigma),
                    potential_weight=float(potential_weight),
                    seed=int(seed),
                )
                Ht = v._symmetrize_torch(H)
                eig, _ = v._eigvalsh_safe(Ht)
                k = int(r.get("k_align") or 0)
                if eig.size >= 2 and k >= 2:
                    v._plot_spacing(
                        out_dir / f"spacing_{safe}.png",
                        np.sort(eig[:k]),
                        title=f"{cid}: normalized eig spacings (k={k})",
                    )
            except Exception:
                pass


def run_pareto_subprocess(*, validation_json: Path, pareto_out: Path) -> int:
    script = Path(ROOT) / "scripts" / "pareto_select_v13.py"
    cmd = [
        sys.executable,
        str(script),
        "--input_json",
        str(validation_json.resolve()),
        "--out_dir",
        str(pareto_out.resolve() if pareto_out.is_absolute() else (Path(ROOT) / pareto_out).resolve()),
    ]
    print("Running:", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=ROOT)
    return int(r.returncode)


def main() -> None:
    ap = argparse.ArgumentParser(description="V13G: validate top-5 from formula report JSON; optional Pareto pass.")
    ap.add_argument(
        "--input_json",
        type=str,
        default="runs/v13_formula_report/v13_top_candidates.json",
        help="v13_top_candidates.json (object with top_candidates).",
    )
    ap.add_argument("--out_dir", type=str, default="runs/v13_top5_validation")
    ap.add_argument("--zeros", type=int, default=128)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--eps", type=float, default=0.6)
    ap.add_argument("--geo_weight", type=float, default=10.0)
    ap.add_argument("--geo_sigma", type=float, default=0.6)
    ap.add_argument("--potential_weight", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=42, help="sample_domain / operator build seed (V13C default).")
    ap.add_argument("--save_plots", type=str, default="false", help="true/false: spacing PNGs (matplotlib).")
    ap.add_argument(
        "--run_pareto",
        type=str,
        default="true",
        help="If true, run scripts/pareto_select_v13.py after validation.",
    )
    ap.add_argument("--pareto_out_dir", type=str, default="runs/v13_top5_pareto")
    args = ap.parse_args()

    in_path = Path(args.input_json)
    if not in_path.is_absolute():
        in_path = Path(ROOT) / in_path
    if not in_path.is_file():
        raise FileNotFoundError(f"input_json not found: {in_path}")

    rows = load_top_candidates(in_path)
    candidates, warns = rows_to_validation_candidates(rows)
    for w in warns:
        print(f"[validate-top5-warning] {w}", flush=True)
    if not candidates:
        raise SystemExit("no valid candidates extracted (check best_words in input JSON)")

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = Path(ROOT) / out_dir

    save_plots = str(args.save_plots).lower() in ("1", "true", "yes", "y")
    run_validation(
        candidates=candidates,
        out_dir=out_dir,
        zeros_n=int(args.zeros),
        dim=int(args.dim),
        eps=float(args.eps),
        geo_weight=float(args.geo_weight),
        geo_sigma=float(args.geo_sigma),
        potential_weight=float(args.potential_weight),
        seed=int(args.seed),
        save_plots=save_plots,
    )

    vjson = out_dir / "validation_results.json"
    print(f"Wrote {vjson}", flush=True)

    if str(args.run_pareto).lower() in ("1", "true", "yes", "y"):
        pareto_dir = Path(args.pareto_out_dir)
        if not pareto_dir.is_absolute():
            pareto_dir = Path(ROOT) / pareto_dir
        code = run_pareto_subprocess(validation_json=vjson, pareto_out=pareto_dir)
        if code != 0:
            raise SystemExit(f"pareto_select_v13.py exited with code {code}")
        print(f"Pareto results under {pareto_dir}", flush=True)


if __name__ == "__main__":
    main()
