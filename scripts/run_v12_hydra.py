#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _require_hydra():
    try:
        import hydra  # noqa: F401
        from omegaconf import OmegaConf  # noqa: F401

        return True
    except Exception:
        print("Missing Hydra dependencies.", file=sys.stderr)
        print("Install with:", file=sys.stderr)
        print("  pip install hydra-core omegaconf", file=sys.stderr)
        return False


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _write_text(p: Path, s: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8")


def _run_stage(name: str, cmd: List[str], logs_dir: Path) -> float:
    _ensure_dir(logs_dir)
    log_path = logs_dir / f"{name}.log"
    print(f"\n=== stage: {name} ===", flush=True)
    print("cmd:", " ".join(cmd), flush=True)
    t0 = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    dt = time.perf_counter() - t0
    if p.returncode != 0:
        raise SystemExit(f"stage {name!r} failed (code={p.returncode}). See log: {log_path}")
    return dt


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V12 pipeline via Hydra configs")
    parser.add_argument("--config-name", type=str, default="config")
    args, unknown = parser.parse_known_args()

    if not _require_hydra():
        raise SystemExit(1)

    import hydra
    from omegaconf import OmegaConf

    root = _repo_root()
    os.chdir(root)
    if root not in sys.path:
        sys.path.insert(0, root)

    cfg_dir = os.path.join(root, "configs")
    with hydra.initialize_config_dir(config_dir=cfg_dir, job_name="v12", version_base=None):
        cfg = hydra.compose(config_name=str(args.config_name), overrides=unknown)

    runs_dir = Path(str(cfg.paths.runs_dir))
    logs_dir = Path(str(cfg.paths.logs_dir))
    _ensure_dir(runs_dir)
    _ensure_dir(logs_dir)

    OmegaConf.save(cfg, runs_dir / "v12_config_used.yaml")

    zeros_path = str(cfg.paths.zeros)
    stages = list(cfg.run.stages)

    timings: Dict[str, float] = {}

    for stage in stages:
        stage = str(stage)
        py = sys.executable

        if stage == "artin":
            c = cfg.artin
            cmd = [
                py,
                "-m",
                "core.artin_symbolic_billiard",
                "--num_samples",
                str(int(c.num_samples)),
                "--max_length",
                str(int(c.max_length)),
                "--max_power",
                str(int(c.max_power)),
                "--out_dir",
                str(c.out_dir),
            ]
            timings[stage] = _run_stage(stage, cmd, logs_dir)

        elif stage == "selberg":
            c = cfg.selberg
            cmd = [
                py,
                "-m",
                "validation.selberg_trace_loss",
                "--lengths",
                str(c.lengths),
                "--zeros",
                str(c.zeros),
                "--sigma",
                str(float(c.sigma)),
                "--m_max",
                str(int(c.m_max)),
                "--out_dir",
                str(c.out_dir),
            ]
            timings[stage] = _run_stage(stage, cmd, logs_dir)

        elif stage == "operator":
            c = cfg.operator
            cmd = [
                py,
                "-m",
                "core.artin_operator",
                "--n_points",
                str(int(c.n_points)),
                "--sigma",
                str(float(c.sigma)),
                "--top_k_geodesics",
                str(int(c.top_k_geodesics)),
                "--eps",
                str(float(c.eps)),
                "--geodesics",
                str(c.geodesics),
                "--zeros",
                str(c.zeros),
                "--out_dir",
                str(c.out_dir),
            ]
            timings[stage] = _run_stage(stage, cmd, logs_dir)

        elif stage == "aco":
            c = cfg.aco
            cmd = [
                py,
                "-m",
                "core.artin_aco",
                "--num_ants",
                str(int(c.num_ants)),
                "--num_iters",
                str(int(c.num_iters)),
                "--max_length",
                str(int(c.max_length)),
                "--max_power",
                str(int(c.max_power)),
                "--alpha",
                str(float(c.alpha)),
                "--beta",
                str(float(c.beta)),
                "--rho",
                str(float(c.rho)),
                "--seed",
                str(int(c.seed)),
                "--zeros",
                str(c.zeros),
                "--out_dir",
                str(c.out_dir),
            ]
            timings[stage] = _run_stage(stage, cmd, logs_dir)

        elif stage == "rl":
            c = cfg.rl
            cmd = [
                py,
                "-m",
                "validation.artin_rl_train",
                "--num_updates",
                str(int(c.num_updates)),
                "--steps_per_update",
                str(int(c.steps_per_update)),
                "--max_length",
                str(int(c.max_length)),
                "--max_power",
                str(int(c.max_power)),
                "--lr",
                str(float(c.lr)),
                "--pheromone_bias",
                str(float(c.pheromone_bias)),
                "--eval_operator_every",
                str(int(c.eval_operator_every)),
                "--target_zeros_path",
                str(c.target_zeros_path),
                "--out_dir",
                str(c.out_dir),
                "--seed",
                str(int(c.seed)),
            ]
            timings[stage] = _run_stage(stage, cmd, logs_dir)

        elif stage == "stability":
            c = cfg.stability
            cmd = [
                py,
                "-m",
                "validation.operator_stability_report",
                "--operator",
                str(c.operator_path),
                "--zeros",
                str(c.zeros),
                "--k",
                str(int(c.k)),
                "--out",
                str(c.out),
            ]
            timings[stage] = _run_stage(stage, cmd, logs_dir)

        elif stage == "analysis":
            cmd = [py, "analysis/gemma_analyzer.py"]
            try:
                timings[stage] = _run_stage(stage, cmd, logs_dir)
            except Exception:
                # Never crash pipeline on analysis.
                timings[stage] = 0.0

        elif stage == "analysis_journal":
            cmd = [py, "analysis/gemma_lab_journal.py"]
            try:
                timings[stage] = _run_stage(stage, cmd, logs_dir)
            except Exception:
                # Never crash pipeline on journal analysis.
                timings[stage] = 0.0

        elif stage == "paper":
            cmd = [py, "analysis/gemma_paper_writer.py"]
            try:
                timings[stage] = _run_stage(stage, cmd, logs_dir)
            except Exception:
                # Never crash pipeline on paper writing.
                timings[stage] = 0.0

        elif stage == "pde":
            cmd = [py, "analysis/operator_pde_discovery.py"]
            try:
                timings[stage] = _run_stage(stage, cmd, logs_dir)
            except Exception:
                # Never crash pipeline on PDE discovery.
                timings[stage] = 0.0

        else:
            raise SystemExit(f"unknown stage: {stage}")

    # AUTO ANALYSIS: always attempt at end (never crash)
    try:
        py = sys.executable
        _run_stage("analysis_auto", [py, "analysis/gemma_analyzer.py"], logs_dir)
    except Exception:
        pass

    lines = ["stage,time_s"] + [f"{k},{v:.6g}" for k, v in timings.items()]
    _write_text(logs_dir / "timings.csv", "\n".join(lines) + "\n")
    print("\nall stages complete", flush=True)


if __name__ == "__main__":
    main()

