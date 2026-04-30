from __future__ import annotations

import csv
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs"

LOCALHOSTS = {"127.0.0.1", "::1", "localhost"}

DEFAULT_SUBPROCESS_TIMEOUT_S = 300


class ApiError(RuntimeError):
    pass


def is_local_client(host: Optional[str]) -> bool:
    if not host:
        return False
    if host in LOCALHOSTS:
        return True
    # Allow IPv6 localhost forms (e.g. "::1%lo0")
    if host.startswith("::1"):
        return True
    return False


def _safe_path_under(base: Path, rel: str) -> Path:
    p = (base / rel).resolve()
    if base.resolve() not in p.parents and p != base.resolve():
        raise ApiError("Refusing to access path outside allowed directory.")
    return p


def read_text_rel_runs(rel_path: str, max_chars: int = 200_000) -> Optional[str]:
    path = _safe_path_under(RUNS_DIR, rel_path)
    try:
        if not path.exists():
            return None
        txt = path.read_text(encoding="utf-8")
        if len(txt) > max_chars:
            return txt[:max_chars] + "\n...[TRUNCATED]...\n"
        return txt
    except Exception:
        return None


def read_json_rel_runs(rel_path: str) -> Optional[Dict[str, Any]]:
    path = _safe_path_under(RUNS_DIR, rel_path)
    try:
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def read_csv_rows_rel_runs(rel_path: str, max_rows: int = 50_000) -> List[Dict[str, str]]:
    path = _safe_path_under(RUNS_DIR, rel_path)
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows: List[Dict[str, str]] = []
            for i, row in enumerate(reader):
                if i >= max_rows:
                    break
                rows.append(dict(row))
            return rows
    except Exception:
        return []


def to_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, bool):
            return float(int(x))
        return float(x)
    except Exception:
        return None


def _linear_trend(values: Sequence[float]) -> str:
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


def aco_metrics_from_history(rows: Sequence[Mapping[str, str]]) -> Dict[str, Any]:
    best_vals = [to_float(r.get("best_loss")) for r in rows]
    mean_vals = [to_float(r.get("mean_loss")) for r in rows]
    best = [v for v in best_vals if v is not None]
    mean = [v for v in mean_vals if v is not None]
    tail_best = best[-20:] if len(best) >= 2 else best
    trend = _linear_trend(tail_best) if tail_best else "flat"
    return {
        "best_loss": (best[-1] if best else None),
        "mean_loss": (mean[-1] if mean else None),
        "trend": trend,
        "n_rows": len(rows),
    }


def topological_lm_metrics_from_eval_report(report: Mapping[str, Any]) -> Dict[str, Any]:
    baselines = report.get("baselines", {}) if isinstance(report.get("baselines"), dict) else {}

    def _mean_reward(name: str) -> Optional[float]:
        node = baselines.get(name, {}) if isinstance(baselines.get(name), dict) else {}
        dedup = node.get("dedup", {}) if isinstance(node.get("dedup"), dict) else {}
        raw = node.get("raw", {}) if isinstance(node.get("raw"), dict) else {}
        v = to_float(dedup.get("mean_reward"))
        if v is None:
            v = to_float(raw.get("mean_reward"))
        return v

    random_mean = _mean_reward("random")
    topo_mean = _mean_reward("TopologicalLM-only")
    unique_ratio = None
    valid_ratio = None

    topo_node = baselines.get("TopologicalLM-only", {}) if isinstance(baselines.get("TopologicalLM-only"), dict) else {}
    if isinstance(topo_node.get("dedup"), dict):
        unique_ratio = to_float(topo_node.get("dedup", {}).get("unique_candidate_ratio"))
    if unique_ratio is None:
        unique_ratio = to_float(topo_node.get("unique_candidate_ratio"))

    topo_dedup = topo_node.get("dedup", {}) if isinstance(topo_node.get("dedup"), dict) else {}
    topo_raw = topo_node.get("raw", {}) if isinstance(topo_node.get("raw"), dict) else {}
    valid_ratio = to_float(topo_dedup.get("valid_braid_ratio"))
    if valid_ratio is None:
        valid_ratio = to_float(topo_raw.get("valid_braid_ratio"))

    advantage = None
    if topo_mean is not None and random_mean is not None:
        advantage = topo_mean - random_mean

    return {
        "random_mean_reward": random_mean,
        "topological_lm_mean_reward": topo_mean,
        "advantage_over_random": advantage,
        "unique_candidate_ratio": unique_ratio,
        "valid_braid_ratio": valid_ratio,
    }


@dataclass(frozen=True)
class MakeResult:
    target: str
    returncode: int
    duration_s: float
    stdout: str
    stderr: str
    timed_out: bool


def run_make_target(
    target: str,
    *,
    timeout_s: int = DEFAULT_SUBPROCESS_TIMEOUT_S,
    env_extra: Optional[Mapping[str, str]] = None,
) -> MakeResult:
    if not target or not isinstance(target, str):
        raise ApiError("Invalid make target.")
    start = time.time()
    env = os.environ.copy()
    if env_extra:
        env.update({str(k): str(v) for k, v in env_extra.items()})
    try:
        p = subprocess.run(
            ["make", target],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=int(timeout_s),
            check=False,
            env=env,
        )
        dur = time.time() - start
        return MakeResult(
            target=target,
            returncode=int(p.returncode),
            duration_s=dur,
            stdout=p.stdout or "",
            stderr=p.stderr or "",
            timed_out=False,
        )
    except subprocess.TimeoutExpired as e:
        dur = time.time() - start
        return MakeResult(
            target=target,
            returncode=124,
            duration_s=dur,
            stdout=(e.stdout or "") if isinstance(e.stdout, str) else "",
            stderr=(e.stderr or "") if isinstance(e.stderr, str) else "",
            timed_out=True,
        )


def run_gemma_help(question: str, *, voice: bool = False, timeout_s: int = 300) -> str:
    q = str(question or "").strip()
    if not q:
        raise ApiError("Question must be non-empty.")
    cmd = [
        "python3",
        "help/gemma_help_agent.py",
        "--question",
        q,
        "--voice",
        "True" if bool(voice) else "False",
        "--stream",
        "False",
        "--interactive",
        "False",
    ]
    try:
        p = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=int(timeout_s),
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise ApiError(f"Gemma help timed out after {timeout_s}s.")

    out = (p.stdout or "").strip()
    err = (p.stderr or "").strip()
    # Filter noisy debug lines from stdout, if present.
    out_lines = [ln for ln in out.splitlines() if not ln.strip().startswith("[debug]")]
    out = "\n".join(out_lines).strip()

    if p.returncode != 0 and not out:
        raise ApiError(err or f"Gemma help failed with exit code {p.returncode}.")
    if not out and err:
        # Some environments may write answer to stderr; fallback to returning it.
        out = err
    return out.strip()

