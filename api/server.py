from __future__ import annotations

import os
import json
import csv
import time
import uuid
import threading
import subprocess
import math
import tempfile
import zipfile
import shutil
import re
from datetime import datetime
from collections import deque
from pathlib import Path
from typing import Any, Dict, Optional, List, Mapping

from fastapi import Body, FastAPI, HTTPException, Request, Response, Query, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from api.schemas import (
    AcoMetricsResponse,
    ErrorResponse,
    GemmaHelpRequest,
    GemmaHelpResponse,
    HealthResponse,
    RunResultResponse,
    RunStageRequest,
    StatusResponse,
    TopologicalLmMetricsResponse,
)
from api.utils import (
    ApiError,
    aco_metrics_from_history,
    is_local_client,
    REPO_ROOT,
    read_csv_rows_rel_runs,
    read_json_rel_runs,
    read_text_rel_runs,
    run_gemma_help,
    run_make_target,
    topological_lm_metrics_from_eval_report,
)

def _env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return bool(default)
    s = str(v).strip().lower()
    return s in {"1", "true", "t", "yes", "y", "on"}


LOW_RESOURCE_MODE = _env_flag("LOW_RESOURCE_MODE", False)
MAX_CONCURRENT_JOBS = 1 if LOW_RESOURCE_MODE else 2

# Small in-process cache for lightweight endpoints to reduce disk churn / CPU.
_RESP_CACHE: Dict[str, Any] = {}
_RESP_CACHE_TS: Dict[str, float] = {}
CACHE_TTL_S = 8.0 if LOW_RESOURCE_MODE else 5.0


def _cache_get(key: str) -> Optional[Any]:
    try:
        ts = float(_RESP_CACHE_TS.get(key, 0.0))
        if (time.time() - ts) <= CACHE_TTL_S and key in _RESP_CACHE:
            return _RESP_CACHE.get(key)
    except Exception:
        return None
    return None


def _cache_set(key: str, val: Any) -> Any:
    _RESP_CACHE[key] = val
    _RESP_CACHE_TS[key] = time.time()
    return val


app = FastAPI(title="Ant-RH Control API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

# -----------------------------------------------------------------------------
# Jobs (whitelisted commands only)
# -----------------------------------------------------------------------------

LOGS_DIR = (REPO_ROOT / "logs" / "jobs").resolve()
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Exports / snapshots
# -----------------------------------------------------------------------------

EXPORTS_DIR = (REPO_ROOT / "runs" / "exports").resolve()
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
EXPORTS_INDEX_PATH = EXPORTS_DIR / "index.json"

# In-memory job store (simple, local-only API).
JOBS: Dict[str, Dict[str, Any]] = {}

ALLOWED_SIMPLE_JOBS = {
    "topo-train": {"name": "TopologicalLM Train", "kind": "make", "target": "topo-train"},
    "topo-eval": {"name": "TopologicalLM Eval", "kind": "make", "target": "topo-eval"},
    "topo-ppo": {"name": "TopologicalLM PPO", "kind": "make", "target": "topo-ppo"},
    "topo-report": {"name": "TopologicalLM Report", "kind": "make", "target": "topo-report"},
    "gemma-health": {"name": "Gemma Health", "kind": "make", "target": "gemma-health"},
    "docs": {"name": "Docs", "kind": "make", "target": "docs"},
    "study": {"name": "Study", "kind": "make", "target": "study"},
    "pde": {"name": "Operator PDE discovery", "kind": "cmd", "command": ["make", "pde"]},
    "sensitivity": {"name": "Operator sensitivity", "kind": "cmd", "command": ["make", "sensitivity"]},
    "stability": {
        "name": "Operator stability",
        "kind": "cmd",
        "command": [
            "python3",
            "-m",
            "validation.operator_stability_report",
            "--operator",
            "runs/artin_operator.npy",
            "--zeros",
            "data/zeta_zeros.txt",
            "--k",
            "64",
            "--out",
            "runs/operator_stability_report.json",
        ],
    },
    "full-pipeline": {"name": "Full pipeline", "kind": "pipeline"},
}

ALLOWED_JOBS = set(ALLOWED_SIMPLE_JOBS.keys()) | {"aco"}


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _load_exports_index() -> Dict[str, Any]:
    try:
        if not EXPORTS_INDEX_PATH.exists():
            return {"exports": []}
        obj = json.loads(EXPORTS_INDEX_PATH.read_text(encoding="utf-8", errors="replace"))
        if isinstance(obj, dict) and isinstance(obj.get("exports"), list):
            return obj
    except Exception:
        pass
    return {"exports": []}


def _save_exports_index(idx: Dict[str, Any]) -> None:
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = EXPORTS_INDEX_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(idx, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(EXPORTS_INDEX_PATH)


def _safe_rel_from_repo(p: Path) -> Optional[Path]:
    try:
        rp = p.resolve()
        if REPO_ROOT.resolve() not in rp.parents and rp != REPO_ROOT.resolve():
            return None
        return rp.relative_to(REPO_ROOT)
    except Exception:
        return None


def _metrics_summary() -> Dict[str, Any]:
    # Best-effort extraction of a few high-signal summary metrics.
    out: Dict[str, Any] = {"aco_best_loss": None, "topo_reward_mean": None, "spectral_loss": None}
    try:
        p = REPO_ROOT / "runs" / "artin_aco_best.json"
        if p.exists():
            obj = json.loads(p.read_text(encoding="utf-8", errors="replace"))
            if isinstance(obj, dict):
                for k in ("best_loss", "best_loss_value", "loss"):
                    v = obj.get(k)
                    try:
                        if v is not None:
                            out["aco_best_loss"] = float(v)
                            break
                    except Exception:
                        pass
    except Exception:
        pass

    try:
        p = REPO_ROOT / "runs" / "topological_lm" / "eval_report.json"
        if p.exists():
            obj = json.loads(p.read_text(encoding="utf-8", errors="replace"))
            if isinstance(obj, dict):
                baselines = obj.get("baselines") if isinstance(obj.get("baselines"), dict) else {}
                topo = baselines.get("TopologicalLM-only") if isinstance(baselines.get("TopologicalLM-only"), dict) else {}
                dedup = topo.get("dedup") if isinstance(topo.get("dedup"), dict) else {}
                raw = topo.get("raw") if isinstance(topo.get("raw"), dict) else {}
                v = dedup.get("mean_reward", None)
                if v is None:
                    v = raw.get("mean_reward", None)
                if v is not None:
                    out["topo_reward_mean"] = float(v)
    except Exception:
        pass

    try:
        p = REPO_ROOT / "runs" / "operator_stability_report.json"
        if p.exists():
            obj = json.loads(p.read_text(encoding="utf-8", errors="replace"))
            if isinstance(obj, dict) and obj.get("spectral_loss") is not None:
                out["spectral_loss"] = float(obj.get("spectral_loss"))
    except Exception:
        pass
    return out


def create_export_snapshot(
    name: str,
    reason: str,
    include_patterns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Create a versioned snapshot zip under runs/exports and update index.json.

    Excludes: *.gguf, *.pt, *.pth, *.npy, *.npz and files > 50MB.
    """

    snap_name = str(name or "").strip() or "snapshot"
    snap_reason = str(reason or "").strip() or "manual"
    ts = _now_ts()
    export_id = uuid.uuid4().hex[:12]

    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    zip_rel = Path("runs") / "exports" / f"{ts}_{snap_name}.zip"
    zip_path = (REPO_ROOT / zip_rel).resolve()

    default_patterns = [
        "runs/artin_aco_history.csv",
        "runs/artin_aco_best.json",
        "runs/topological_lm/report.md",
        "runs/topological_lm/eval_report.json",
        "runs/topological_ppo/**",
        "runs/operator_pde_report.md",
        "runs/operator_pde_formula.tex",
        "runs/operator_sensitivity_report.json",
        "runs/operator_stability_report.json",
        "runs/gemma_analysis.json",
        "runs/lab_journal.md",
        "runs/project_summary.md",
        "configs/*.yaml",
        "Makefile",
    ]
    pats = include_patterns if include_patterns else default_patterns

    excluded_ext = {".gguf", ".pt", ".pth", ".npy", ".npz"}
    max_bytes = 50 * 1024 * 1024

    def _iter_paths() -> List[Path]:
        out: List[Path] = []
        for pat in pats:
            pat = str(pat)
            if pat.endswith("/**"):
                root = (REPO_ROOT / pat[:-3]).resolve()
                if root.exists():
                    out.extend([p for p in root.rglob("*") if p.is_file()])
            elif "**" in pat:
                base = pat.split("/")[0]
                root = (REPO_ROOT / base).resolve()
                if root.exists():
                    out.extend([p for p in root.rglob("*") if p.is_file()])
            else:
                out.extend([p for p in (REPO_ROOT / ".").glob(pat) if p.is_file()])
        seen = set()
        uniq: List[Path] = []
        for p in out:
            rp = p.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            uniq.append(rp)
        return uniq

    files_added = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in _iter_paths():
            rel = _safe_rel_from_repo(p)
            if rel is None:
                continue
            rel_s = rel.as_posix()
            if "node_modules" in rel_s or "__pycache__" in rel_s or "/.venv/" in rel_s:
                continue
            if p.suffix.lower() in excluded_ext:
                continue
            try:
                if p.stat().st_size > max_bytes:
                    continue
            except Exception:
                continue
            z.write(p, arcname=rel_s)
            files_added += 1

    size_bytes = zip_path.stat().st_size if zip_path.exists() else 0
    entry = {
        "id": export_id,
        "timestamp": _now_iso(),
        "name": snap_name,
        "reason": snap_reason,
        "path": zip_rel.as_posix(),
        "size_bytes": int(size_bytes),
        "files_count": int(files_added),
        "metrics_summary": _metrics_summary(),
    }

    idx = _load_exports_index()
    idx.setdefault("exports", [])
    idx["exports"].append(entry)

    # Keep latest 20 auto snapshots; don't delete manual ones automatically.
    try:
        exports = [e for e in idx.get("exports", []) if isinstance(e, dict)]
        autos = [e for e in exports if e.get("reason") == "auto_after_job"]
        autos_sorted = sorted(autos, key=lambda e: str(e.get("timestamp") or ""), reverse=True)
        autos_keep = set((e.get("id") for e in autos_sorted[:20]))
        to_prune = [e for e in autos if e.get("id") not in autos_keep]
        if to_prune:
            kept = []
            for e in exports:
                if e.get("reason") == "auto_after_job" and e.get("id") in {x.get("id") for x in to_prune}:
                    # delete file
                    try:
                        p = (REPO_ROOT / str(e.get("path") or "")).resolve()
                        if p.exists() and REPO_ROOT.resolve() in p.parents:
                            p.unlink()
                    except Exception:
                        pass
                    continue
                kept.append(e)
            idx["exports"] = kept
    except Exception:
        pass

    _save_exports_index(idx)
    return entry


def _running_jobs_count() -> int:
    return sum(1 for j in JOBS.values() if j.get("status") == "running")


def _running_jobs_summary() -> List[Dict[str, Any]]:
    out = []
    for j in JOBS.values():
        if j.get("status") not in {"running", "paused"}:
            continue
        out.append(
            {
                "id": j.get("id"),
                "job": j.get("job"),
                "name": j.get("name"),
                "status": j.get("status"),
                "started_at": j.get("started_at"),
            }
        )
    out.sort(key=lambda x: str(x.get("started_at") or ""), reverse=True)
    return out


JOB_DEFAULT_ETA_S = {
    "aco": 120,
    "topo-train": 300,
    "topo-eval": 60,
    "topo-ppo": 600,
    "topo-report": 20,
    "gemma-health": 30,
    "study": 60,
    "docs": 120,
    "pde": 180,
    "sensitivity": 120,
    "stability": 60,
    "full-pipeline": 900,
    "checkpoint": 20,
}

JOB_TIMEOUT_S = (
    {
        "gemma-health": 90,
        "aco": 300,
        "topo-ppo": 900,
        "full-pipeline": 1800,
    }
    if LOW_RESOURCE_MODE
    else {
        "gemma-health": 180,
        "aco": 600,
        "topo-ppo": 1800,
        "full-pipeline": 3600,
    }
)


def _job_elapsed_seconds(job: Mapping[str, Any]) -> float:
    # Prefer monotonic-ish timestamps stored on job record.
    try:
        st = float(job.get("started_ts")) if job.get("started_ts") is not None else None
        et = float(job.get("ended_ts")) if job.get("ended_ts") is not None else None
        if st is not None:
            now = time.time()
            if job.get("status") == "running":
                return max(0.0, now - st)
            if et is not None:
                return max(0.0, et - st)
    except Exception:
        pass

    # Fallback parse ISO-ish timestamps
    def parse(s: Any) -> Optional[float]:
        try:
            if not isinstance(s, str) or not s:
                return None
            dt = datetime.strptime(s, "%Y-%m-%dT%H:%M:%S%z")
            return float(dt.timestamp())
        except Exception:
            return None

    st = parse(job.get("started_at"))
    if st is None:
        return 0.0
    if job.get("status") == "running":
        return max(0.0, time.time() - st)
    et = parse(job.get("ended_at"))
    if et is None:
        return 0.0
    return max(0.0, et - st)


def _job_eta_seconds(job_key: str, job: Mapping[str, Any]) -> int:
    default_eta = int(JOB_DEFAULT_ETA_S.get(job_key, 0))
    if str(job.get("status")) != "running":
        return 0
    elapsed = _job_elapsed_seconds(job)
    return int(max(0, default_eta - elapsed))


def _job_progress_percent(job_key: str, job: Mapping[str, Any]) -> float:
    default_eta = float(JOB_DEFAULT_ETA_S.get(job_key, 0))
    st = str(job.get("status") or "pending")
    if st == "done":
        return 100.0
    if st == "failed":
        return 100.0
    if st != "running" or default_eta <= 0:
        return 0.0
    elapsed = _job_elapsed_seconds(job)
    return float(max(0.0, min(100.0, (elapsed / default_eta) * 100.0)))


def _job_public(job: Mapping[str, Any]) -> Dict[str, Any]:
    job_key = str(job.get("job") or job.get("name") or "")
    out = dict(job)
    out.pop("_log_fh", None)
    out.pop("_proc", None)
    # normalize name -> job key for UI
    out["name"] = job_key
    out["elapsed_seconds"] = float(_job_elapsed_seconds(job))
    out["eta_seconds"] = int(_job_eta_seconds(job_key, job))
    out["progress_percent"] = float(_job_progress_percent(job_key, job))
    return out


def _close_job_log(job: Dict[str, Any]) -> None:
    fh = job.pop("_log_fh", None)
    try:
        if fh:
            fh.flush()
            fh.close()
    except Exception:
        pass


def _mark_job_ended(job: Dict[str, Any], *, status: str, returncode: Optional[int] = None, error: Optional[str] = None) -> None:
    job["ended_at"] = _now_iso()
    job["ended_ts"] = time.time()
    job["status"] = status
    if returncode is not None:
        job["returncode"] = int(returncode)
    if error is not None:
        job["error"] = str(error)
    _close_job_log(job)
    job.pop("_proc", None)


def _jobs_watchdog() -> None:
    """
    Best-effort reconciliation for running jobs:
    - If process already exited, mark job done/failed and set ended_at/returncode.
    - If elapsed exceeds timeout_seconds, terminate and mark failed with error=timeout.
    Runs on read endpoints to avoid needing a dedicated scheduler.
    """
    try:
        import signal
    except Exception:
        signal = None  # type: ignore

    for job in list(JOBS.values()):
        try:
            if str(job.get("status")) != "running":
                continue

            timeout_seconds = job.get("timeout_seconds")
            try:
                timeout_s = float(timeout_seconds) if timeout_seconds is not None else None
            except Exception:
                timeout_s = None

            proc = job.get("_proc")
            if proc is not None:
                try:
                    rc = proc.poll()
                except Exception:
                    rc = None
                if rc is not None:
                    _mark_job_ended(job, status=("done" if int(rc) == 0 else "failed"), returncode=int(rc))
                    continue

            if timeout_s is not None and timeout_s > 0:
                elapsed = float(_job_elapsed_seconds(job))
                if elapsed > timeout_s:
                    # Timeout: terminate the underlying process if we can.
                    try:
                        if proc is not None:
                            try:
                                proc.terminate()
                                try:
                                    proc.wait(timeout=5)
                                except Exception:
                                    proc.kill()
                            except Exception:
                                pass
                        else:
                            pid = job.get("pid")
                            if pid and signal is not None:
                                try:
                                    os.kill(int(pid), signal.SIGTERM)
                                except Exception:
                                    pass
                    finally:
                        _mark_job_ended(job, status="failed", returncode=-9, error="timeout")
        except Exception:
            # Never let watchdog break API calls.
            continue


def _to_int(v: Any, *, min_v: int, max_v: int) -> int:
    try:
        iv = int(v)
    except Exception:
        raise HTTPException(status_code=400, detail=f"Invalid int: {v!r}")
    if iv < min_v or iv > max_v:
        raise HTTPException(status_code=400, detail=f"Out of range [{min_v},{max_v}]: {iv}")
    return iv


def _to_float(v: Any, *, min_v: float, max_v: float) -> float:
    try:
        fv = float(v)
    except Exception:
        raise HTTPException(status_code=400, detail=f"Invalid float: {v!r}")
    if fv < min_v or fv > max_v:
        raise HTTPException(status_code=400, detail=f"Out of range [{min_v},{max_v}]: {fv}")
    return float(fv)


def _to_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return bool(v)
    if isinstance(v, (int, float)) and v in (0, 1):
        return bool(int(v))
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "f", "no", "n", "off"}:
            return False
    raise HTTPException(status_code=400, detail=f"Invalid bool: {v!r}")


def _build_aco_cmd(params: Mapping[str, Any]) -> List[str]:
    # Only allow a fixed set of parameters; never accept arbitrary command strings.
    p = dict(params or {})

    cmd = ["python3", "-m", "core.artin_aco"]

    # Safe numeric params with reasonable bounds.
    num_ants = _to_int(p.get("num_ants", 32), min_v=1, max_v=10_000)
    num_iters = _to_int(p.get("num_iters", 80), min_v=1, max_v=1_000_000)
    max_length = _to_int(p.get("max_length", 6), min_v=1, max_v=64)
    max_power = _to_int(p.get("max_power", 4), min_v=1, max_v=64)
    cmd += ["--num_ants", str(num_ants), "--num_iters", str(num_iters), "--max_length", str(max_length), "--max_power", str(max_power)]

    # Optional planner
    if "use_planner" in p:
        cmd += ["--use_planner", "True" if _to_bool(p["use_planner"]) else "False"]

    # Optional loss weights
    if "lambda_selberg" in p:
        cmd += ["--lambda_selberg", str(_to_float(p["lambda_selberg"], min_v=0.0, max_v=1e6))]
    if "lambda_spec" in p:
        cmd += ["--lambda_spec", str(_to_float(p["lambda_spec"], min_v=0.0, max_v=1e6))]
    if "lambda_spacing" in p:
        cmd += ["--lambda_spacing", str(_to_float(p["lambda_spacing"], min_v=0.0, max_v=1e6))]

    # Optional reward mode (string whitelist)
    if "reward_mode" in p:
        rm = str(p["reward_mode"]).strip().lower()
        if rm not in {"rank", "raw", "hybrid"}:
            raise HTTPException(status_code=400, detail="Invalid reward_mode (allowed: rank, raw, hybrid).")
        cmd += ["--reward_mode", rm]

    # Reject any unknown keys (hard safety).
    allowed_keys = {
        "num_ants",
        "num_iters",
        "max_length",
        "max_power",
        "use_planner",
        "lambda_selberg",
        "lambda_spec",
        "lambda_spacing",
        "reward_mode",
    }
    unknown = sorted(set(p.keys()) - allowed_keys)
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown ACO params: {unknown}")

    return cmd


def _start_job_process(job_id: str, cmd: List[str], log_path: Path) -> subprocess.Popen:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Merge stdout/stderr into one log file.
    f = open(log_path, "a", encoding="utf-8", buffering=1)  # line-buffered
    env = os.environ.copy()
    p = subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        stdout=f,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
    )
    # Keep the file handle reachable so it isn't GC'd early.
    JOBS[job_id]["_log_fh"] = f
    JOBS[job_id]["_proc"] = p
    JOBS[job_id]["pid"] = int(p.pid)
    return p


def _watch_job(job_id: str, proc: subprocess.Popen) -> None:
    rc = None
    try:
        rc = proc.wait()
    except Exception:
        rc = -1
    job = JOBS.get(job_id)
    if not job:
        return
    job["returncode"] = int(rc) if rc is not None else -1
    job["ended_at"] = _now_iso()
    job["ended_ts"] = time.time()
    job["status"] = "done" if job["returncode"] == 0 else "failed"
    _close_job_log(job)
    job.pop("_proc", None)

    # Auto-snapshot after successful key jobs.
    try:
        job_key = str(job.get("job") or "")
        if job.get("status") == "done" and job_key in {"aco", "topo-ppo", "topo-eval", "pde", "sensitivity"}:
            try:
                entry = create_export_snapshot(name=f"{job_key}_{job_id}", reason="auto_after_job")
                job["snapshot_id"] = entry.get("id")
            except Exception as e:
                job["snapshot_warning"] = str(e)
    except Exception:
        pass


def _read_last_lines(path: Path, n: int) -> List[str]:
    try:
        if not path.exists():
            return []
        dq: deque[str] = deque(maxlen=max(1, int(n)))
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for ln in f:
                dq.append(ln.rstrip("\n"))
        return list(dq)
    except Exception:
        return []


def _sse(data: Dict[str, Any]) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _stream_log(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        yield _sse({"error": "job not found"})
        return

    log_path = Path(str(job.get("log_path") or ""))
    # Send a short backlog first.
    for ln in _read_last_lines(log_path, 200):
        yield _sse({"line": ln})

    last_keepalive = time.time()
    last_emit = 0.0
    pending: deque[str] = deque()
    pos = 0
    try:
        if log_path.exists():
            with log_path.open("r", encoding="utf-8", errors="replace") as f:
                f.seek(0, 2)
                pos = f.tell()

        while True:
            job = JOBS.get(job_id)
            if not job:
                yield _sse({"status": "gone"})
                return

            if log_path.exists():
                with log_path.open("r", encoding="utf-8", errors="replace") as f:
                    f.seek(pos)
                    chunk = f.read()
                    if chunk:
                        pos = f.tell()
                        for ln in chunk.splitlines():
                            pending.append(ln)

            # Throttle output to reduce CPU: max 1 line per 0.5s.
            now = time.time()
            if pending and (now - last_emit) >= 0.5:
                yield _sse({"line": pending.popleft()})
                last_emit = now

            if time.time() - last_keepalive > 10:
                yield ": ping\n\n"
                last_keepalive = time.time()

            if job.get("status") in {"done", "failed"}:
                payload: Dict[str, Any] = {"status": job.get("status"), "returncode": job.get("returncode")}
                if job.get("error") is not None:
                    payload["error"] = job.get("error")
                yield _sse(payload)
                return

            time.sleep(0.25)
    except Exception:
        yield _sse({"status": "error"})


@app.get("/")
def dashboard() -> FileResponse:
    return FileResponse(str(_STATIC_DIR / "index.html"))


@app.head("/")
def dashboard_head() -> FileResponse:
    # Some clients probe with HEAD; mirror GET behavior.
    return FileResponse(str(_STATIC_DIR / "index.html"))


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/apple-touch-icon.png", include_in_schema=False)
def apple_touch_icon() -> Response:
    return Response(status_code=204)


@app.get("/apple-touch-icon-precomposed.png", include_in_schema=False)
def apple_touch_icon_precomposed() -> Response:
    return Response(status_code=204)


@app.exception_handler(ApiError)
def _api_error_handler(_: Request, exc: ApiError) -> JSONResponse:
    return JSONResponse(status_code=400, content=ErrorResponse(detail=str(exc)).model_dump())


@app.middleware("http")
async def _local_only(request: Request, call_next):
    client_host = getattr(getattr(request, "client", None), "host", None)
    if not is_local_client(client_host):
        return JSONResponse(status_code=403, content=ErrorResponse(detail="Localhost only.").model_dump())
    return await call_next(request)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/debug/cors")
def debug_cors() -> JSONResponse:
    return JSONResponse(status_code=200, content={"cors": "enabled"})


@app.get("/docs/playbook")
def docs_playbook() -> JSONResponse:
    path = (REPO_ROOT / "Docs" / "Guides" / "experiment_playbook.md").resolve()
    fallback = (
        "# Ant-RH Experiment Playbook\n\n"
        "Playbook not found. Create `Docs/Guides/experiment_playbook.md`.\n"
        "Recommended start:\n\n"
        "- `make dashboard-next`\n"
        "- Run `Gemma Health`\n"
        "- Run `ACO`\n"
    )
    try:
        if not path.exists():
            return JSONResponse(status_code=200, content={"content": fallback})
        txt = path.read_text(encoding="utf-8", errors="replace")
        return JSONResponse(status_code=200, content={"content": txt})
    except Exception:
        return JSONResponse(status_code=200, content={"content": fallback})


@app.get("/exports")
def exports_list() -> JSONResponse:
    return JSONResponse(status_code=200, content=_load_exports_index())


@app.get("/experiment/progress")
def experiment_progress() -> JSONResponse:
    """
    Checklist-style progress derived from runs/ artifacts + currently running jobs.
    """

    estimates = {
        "health": 30,
        "aco": 120,
        "analyze": 60,
        "topo_lm": 300,
        "ppo": 600,
        "physics": 60,
        "operator_analysis": 180,
        "checkpoint": 20,
        "export": 20,
    }

    def _parse_started_at(s: Any) -> Optional[float]:
        try:
            if not isinstance(s, str) or not s:
                return None
            # format from _now_iso(): "%Y-%m-%dT%H:%M:%S%z"
            tt = time.strptime(s, "%Y-%m-%dT%H:%M:%S%z")
            return time.mktime(tt)
        except Exception:
            return None

    def _job_running(job_key: str) -> Optional[Dict[str, Any]]:
        for j in JOBS.values():
            if j.get("status") == "running" and str(j.get("job")) == job_key:
                return j
        return None

    def _file_exists(rel: str) -> bool:
        try:
            return (REPO_ROOT / rel).exists()
        except Exception:
            return False

    def _json_ok(rel: str) -> Optional[Dict[str, Any]]:
        p = (REPO_ROOT / rel)
        try:
            if not p.exists():
                return None
            obj = json.loads(p.read_text(encoding="utf-8", errors="replace"))
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    def _count_csv_rows(rel: str, min_rows: int) -> Optional[bool]:
        p = (REPO_ROOT / rel)
        try:
            if not p.exists():
                return None
            with p.open("r", encoding="utf-8", errors="replace", newline="") as f:
                reader = csv.reader(f)
                n = 0
                for row in reader:
                    if not row:
                        continue
                    n += 1
                    if n >= min_rows + 1:  # include header if present
                        return True
            return n >= min_rows
        except Exception:
            return False

    def _exports_has_entries() -> Optional[bool]:
        try:
            idx = _load_exports_index()
            ex = idx.get("exports", [])
            if not isinstance(ex, list):
                return False
            return len(ex) > 0
        except Exception:
            return False

    def _exports_has_zip() -> bool:
        try:
            return any(p.suffix.lower() == ".zip" for p in EXPORTS_DIR.glob("*.zip"))
        except Exception:
            return False

    steps = []

    def add_step(step_id: str, title: str, evidence: str, command: str, job_key: Optional[str], done: Optional[bool], done_failed: bool = False):
        est = int(estimates.get(step_id, 0))
        status = "pending"
        eta = est

        if done is True:
            status = "done"
            eta = 0
        elif done is False:
            status = "failed" if done_failed else "pending"
            eta = est

        # Override to running if job currently running
        if job_key:
            j = _job_running(job_key)
            if j:
                status = "running"
                started = _parse_started_at(j.get("started_at"))
                if started:
                    elapsed = max(0.0, time.time() - started)
                    eta = max(0, int(est - elapsed))
                else:
                    eta = est

        steps.append(
            {
                "id": step_id,
                "title": title,
                "status": status,
                "eta_seconds": int(max(0, eta)),
                "evidence": evidence,
                "command": command,
            }
        )

    # 0 setup_dashboard (dashboard already running if this endpoint is hit)
    steps.append(
        {
            "id": "setup_dashboard",
            "title": "Setup Dashboard",
            "status": "done",
            "eta_seconds": 0,
            "evidence": "make dashboard-next",
            "command": "make dashboard-next",
        }
    )

    # 1 health
    health_obj = _json_ok("runs/gemma_health_check.json")
    health_done = None
    health_failed = False
    if health_obj is None:
        health_done = None
    elif health_obj == {}:
        health_done = False
        health_failed = True
    else:
        overall = health_obj.get("overall_status")
        health_done = overall in {"ok", "degraded"}
    add_step("health", "Health Check", "runs/gemma_health_check.json", "make gemma-health", "gemma-health", health_done, done_failed=health_failed)

    # 2 aco
    aco_rows = _count_csv_rows("runs/artin_aco_history.csv", 10)
    add_step("aco", "ACO", "runs/artin_aco_history.csv", "make aco", "aco", (True if aco_rows else None) if aco_rows is not False else False)

    # 3 analyze
    ga = _json_ok("runs/gemma_analysis.json")
    analyze_done = None if ga is None else (False if ga == {} else True)
    add_step("analyze", "Analyze Results", "runs/gemma_analysis.json", "make analyze-gemma", "analyze", analyze_done, done_failed=(ga == {}))

    # 4 topo_lm
    topo_eval = _json_ok("runs/topological_lm/eval_report.json")
    topo_done = None if topo_eval is None else (False if topo_eval == {} else True)
    add_step("topo_lm", "TopologicalLM", "runs/topological_lm/eval_report.json", "make topo-eval", "topo-eval", topo_done, done_failed=(topo_eval == {}))

    # 5 ppo
    ppo_done = _file_exists("runs/topological_ppo/train_history.csv")
    add_step("ppo", "Topological PPO", "runs/topological_ppo/train_history.csv", "make topo-ppo", "topo-ppo", True if ppo_done else None)

    # 6 physics (using stability report as proxy)
    phys_obj = _json_ok("runs/operator_stability_report.json")
    phys_done = None if phys_obj is None else (False if phys_obj == {} else True)
    add_step("physics", "Physics Diagnostics", "runs/operator_stability_report.json", "make stability", "stability", phys_done, done_failed=(phys_obj == {}))

    # 7 operator_analysis
    op_done = _file_exists("runs/operator_pde_report.md") or _file_exists("runs/operator_sensitivity_report.json")
    add_step("operator_analysis", "Operator Analysis", "runs/operator_pde_report.md / runs/operator_sensitivity_report.json", "make pde", "pde", True if op_done else None)

    # 8 checkpoint
    ck_done = _exports_has_entries()
    add_step("checkpoint", "Checkpoint", "runs/exports/index.json", "make checkpoint", None, True if ck_done else None)

    # 9 export
    ex_done = _exports_has_zip()
    add_step("export", "Export", "runs/exports/*.zip", "GET /export", None, True if ex_done else None)

    # overall progress/ETA
    total = len(steps)
    done_n = sum(1 for s in steps if s.get("status") == "done")
    overall = float(done_n / total) if total else 0.0
    remaining = sum(int(s.get("eta_seconds") or 0) for s in steps if s.get("status") in {"pending", "running"})

    return JSONResponse(
        status_code=200,
        content={
            "steps": steps,
            "overall_progress": overall,
            "estimated_remaining_seconds": int(max(0, remaining)),
        },
    )


def _json_from_response(resp: JSONResponse) -> Dict[str, Any]:
    try:
        body = getattr(resp, "body", b"") or b""
        if isinstance(body, (bytes, bytearray)):
            return json.loads(body.decode("utf-8"))
        return {}
    except Exception:
        return {}


def _extract_first_json_obj(txt: str) -> Optional[Dict[str, Any]]:
    try:
        m = re.search(r"\{[\s\S]*\}", txt)
        if not m:
            return None
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


@app.get("/recommend/next-step")
def recommend_next_step(use_gemma: bool = Query(False)) -> JSONResponse:
    # Build concise context from progress + a few key files.
    progress = _json_from_response(experiment_progress())
    ctx_parts = [f"progress={json.dumps(progress, ensure_ascii=False)}"]

    def _read(path: str, limit: int = 6000) -> str:
        try:
            p = (REPO_ROOT / path).resolve()
            if not p.exists():
                return ""
            txt = p.read_text(encoding="utf-8", errors="replace")
            return txt[:limit]
        except Exception:
            return ""

    for p in [
        "runs/gemma_analysis.json",
        "runs/gemma_project_memory.md",
        "runs/topological_lm/report.md",
        "runs/operator_sensitivity_report.json",
        "runs/operator_stability_report.json",
    ]:
        txt = _read(p, 6000)
        if txt:
            ctx_parts.append(f"\n[{p}]\n{txt}")

    context = "\n".join(ctx_parts)

    # Rule-based fallback.
    def rule_fallback() -> Dict[str, Any]:
        steps = progress.get("steps", []) if isinstance(progress.get("steps"), list) else []
        by_id = {str(s.get("id")): s for s in steps if isinstance(s, dict)}

        def st(i: str) -> str:
            return str(by_id.get(i, {}).get("status") or "pending")

        # health missing
        if st("health") != "done":
            return {
                "source": "rules",
                "next_step": "Run Gemma Health",
                "reason": "Health check not completed yet.",
                "command": "make gemma-health",
                "dashboard_action": "gemma-health",
                "priority": "high",
            }
        # ACO missing
        if st("aco") != "done":
            return {
                "source": "rules",
                "next_step": "Run ACO baseline",
                "reason": "ACO history missing or too short.",
                "command": "make aco",
                "dashboard_action": "aco",
                "priority": "high",
            }

        # ACO plateau heuristic (best_loss not improving over last 20 points)
        plateau = False
        try:
            p = (REPO_ROOT / "runs/artin_aco_history.csv")
            if p.exists():
                rows = read_csv_rows_rel_runs("artin_aco_history.csv")
                best = []
                for r in rows[-60:]:
                    v = r.get("best_loss") if isinstance(r, dict) else None
                    try:
                        if v is not None:
                            best.append(float(v))
                    except Exception:
                        pass
                if len(best) >= 40:
                    tail = best[-20:]
                    head = best[-40:-20]
                    if head and tail and min(tail) >= (min(head) * 0.995):
                        plateau = True
        except Exception:
            pass
        if plateau:
            return {
                "source": "rules",
                "next_step": "Run sensitivity test",
                "reason": "ACO appears to be plateauing; probe operator sensitivity.",
                "command": "make sensitivity",
                "dashboard_action": "sensitivity",
                "priority": "medium",
            }

        # operator insensitive heuristic
        try:
            sens = read_json_rel_runs("operator_sensitivity_report.json") or {}
            od = sens.get("operator_distance_mean")
            if od is not None and float(od) < 1e-3:
                return {
                    "source": "rules",
                    "next_step": "Run PDE discovery",
                    "reason": "Operator sensitivity looks weak; try PDE discovery to refine operator formula.",
                    "command": "make pde",
                    "dashboard_action": "pde",
                    "priority": "medium",
                }
        except Exception:
            pass

        # TopologicalLM worse than random
        try:
            report = read_json_rel_runs("topological_lm/eval_report.json") or {}
            baselines = report.get("baselines", {}) if isinstance(report.get("baselines"), dict) else {}
            rand = baselines.get("random", {}) if isinstance(baselines.get("random"), dict) else {}
            topo = baselines.get("TopologicalLM-only", {}) if isinstance(baselines.get("TopologicalLM-only"), dict) else {}
            def _mr(node: dict) -> Optional[float]:
                d = node.get("dedup", {}) if isinstance(node.get("dedup"), dict) else {}
                r = node.get("raw", {}) if isinstance(node.get("raw"), dict) else {}
                v = d.get("mean_reward", None)
                if v is None:
                    v = r.get("mean_reward", None)
                try:
                    return float(v) if v is not None else None
                except Exception:
                    return None
            r_m = _mr(rand)
            t_m = _mr(topo)
            if r_m is not None and t_m is not None and t_m <= r_m:
                return {
                    "source": "rules",
                    "next_step": "Improve TopologicalLM (train or PPO)",
                    "reason": "TopologicalLM mean reward is not better than random baseline.",
                    "command": "make topo-train && make topo-ppo",
                    "dashboard_action": "topo-train",
                    "priority": "high",
                }
        except Exception:
            pass

        # no checkpoint
        try:
            idx = _load_exports_index()
            if not (isinstance(idx.get("exports"), list) and idx["exports"]):
                return {
                    "source": "rules",
                    "next_step": "Create checkpoint",
                    "reason": "No checkpoints found yet.",
                    "command": "make checkpoint",
                    "dashboard_action": "checkpoint",
                    "priority": "medium",
                }
        except Exception:
            pass

        return {
            "source": "rules",
            "next_step": "Update docs",
            "reason": "Core steps look complete; capture findings and next actions.",
            "command": "make docs",
            "dashboard_action": "docs",
            "priority": "low",
        }

    # In LOW_RESOURCE_MODE, only use Gemma when explicitly requested (?use_gemma=true).
    use_gemma = bool(use_gemma) if LOW_RESOURCE_MODE else True

    # Try Gemma via llama-cli (optional).
    llama_cli = shutil.which("llama-cli")
    model_path = Path("/Users/machome/models/gemma/gemma-3-1b-it-Q5_K_M.gguf")
    if use_gemma and llama_cli and model_path.exists():
        prompt = (
            "You are Ant-RH experiment coordinator.\n\n"
            f"Given project state:\n{context}\n\n"
            "Recommend ONE next action.\n\n"
            "Return JSON only:\n"
            '{\n'
            '  "next_step": "...",\n'
            '  "reason": "...",\n'
            '  "command": "...",\n'
            '  "dashboard_action": "...",\n'
            '  "priority": "low|medium|high"\n'
            "}\n"
        )
        try:
            p = subprocess.run(
                [llama_cli, "-m", str(model_path), "-p", prompt, "-n", "256"],
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            txt = (p.stdout or "").strip()
            obj = _extract_first_json_obj(txt)
            if obj and all(k in obj for k in ("next_step", "reason", "command", "dashboard_action", "priority")):
                obj["source"] = "gemma"
                return JSONResponse(status_code=200, content=obj)
        except Exception:
            pass

    return JSONResponse(status_code=200, content=rule_fallback())


@app.post("/jobs/start-full-pipeline")
def jobs_start_full_pipeline(payload: Dict[str, Any] = Body(...)) -> JSONResponse:
    if _running_jobs_count() >= MAX_CONCURRENT_JOBS:
        raise HTTPException(status_code=429, detail=f"Too many running jobs (max {MAX_CONCURRENT_JOBS}).")

    include_ppo = bool((payload or {}).get("include_ppo", False))
    continue_on_failure = bool((payload or {}).get("continue_on_failure", True))

    stages: List[Dict[str, Any]] = [
        {"name": "gemma-health", "cmd": ["make", "gemma-health"]},
        {"name": "aco", "cmd": ["make", "aco"]},
        {"name": "analyze", "cmd": ["make", "analyze-gemma"]},
        {"name": "lab-journal", "cmd": ["make", "lab-journal"]},
        {"name": "topo-eval", "cmd": ["make", "topo-eval"]},
    ]
    if include_ppo:
        stages.append({"name": "topo-ppo", "cmd": ["make", "topo-ppo"]})
    stages += [
        {"name": "pde", "cmd": ["make", "pde"]},
        {"name": "sensitivity", "cmd": ["make", "sensitivity"]},
        {"name": "docs", "cmd": ["make", "docs"]},
    ]

    job_id = uuid.uuid4().hex[:12]
    log_path = LOGS_DIR / f"{job_id}.log"
    JOBS[job_id] = {
        "id": job_id,
        "job": "full-pipeline",
        "name": "Full pipeline",
        "command": [s["name"] for s in stages],
        "status": "running",
        "started_at": _now_iso(),
        "started_ts": time.time(),
        "timeout_seconds": int(JOB_TIMEOUT_S.get("full-pipeline", 0) or 0),
        "ended_at": None,
        "ended_ts": None,
        "log_path": str(log_path),
        "returncode": None,
    }

    def _runner():
        rc_total = 0
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8", buffering=1) as f:
            for i, st in enumerate(stages, start=1):
                f.write(f"\n=== [{i}/{len(stages)}] {st['name']} ===\n")
                f.flush()
                try:
                    p = subprocess.run(
                        list(st["cmd"]),
                        cwd=str(REPO_ROOT),
                        text=True,
                        capture_output=True,
                        timeout=(int(JOB_TIMEOUT_S.get(str(st.get("name") or ""), 0) or 0) or None),
                        check=False,
                    )
                    if p.stdout:
                        f.write(p.stdout)
                    if p.stderr:
                        f.write(p.stderr)
                    f.write(f"\n--- returncode={p.returncode} ---\n")
                    f.flush()
                    if p.returncode != 0:
                        rc_total = p.returncode or 1
                        if not continue_on_failure:
                            f.write("Stopping pipeline (continue_on_failure=false).\n")
                            break
                except Exception as e:
                    f.write(f"Stage failed with exception: {e}\n")
                    f.flush()
                    rc_total = 1
                    if not continue_on_failure:
                        break

            # checkpoint at end (best-effort)
            try:
                f.write("\n=== [checkpoint] create_export_snapshot ===\n")
                f.flush()
                entry = create_export_snapshot(name=f"full_pipeline_{job_id}", reason="auto_after_job")
                JOBS[job_id]["snapshot_id"] = entry.get("id")
                f.write(f"snapshot_id={entry.get('id')}\n")
                f.flush()
            except Exception as e:
                JOBS[job_id]["snapshot_warning"] = str(e)
                f.write(f"snapshot_warning={e}\n")
                f.flush()

        job = JOBS.get(job_id)
        if job:
            job["returncode"] = int(rc_total)
            job["ended_at"] = _now_iso()
            job["ended_ts"] = time.time()
            job["status"] = "done" if rc_total == 0 else "failed"

    t = threading.Thread(target=_runner, daemon=True)
    t.start()

    return JSONResponse(status_code=200, content={"job_id": job_id})


@app.post("/exports/create")
def exports_create(payload: Dict[str, Any] = Body(...)) -> JSONResponse:
    name = str((payload or {}).get("name") or "").strip() or "manual_checkpoint"
    reason = str((payload or {}).get("reason") or "").strip() or "manual"
    entry = create_export_snapshot(name=name, reason=reason)
    return JSONResponse(status_code=200, content=entry)


@app.get("/exports/{export_id}/download")
def exports_download(export_id: str) -> FileResponse:
    idx = _load_exports_index()
    items = [e for e in idx.get("exports", []) if isinstance(e, dict)]
    entry = next((e for e in items if str(e.get("id")) == str(export_id)), None)
    if not entry:
        raise HTTPException(status_code=404, detail="Export not found.")
    rel = str(entry.get("path") or "")
    p = (REPO_ROOT / rel).resolve()
    if not p.exists() or REPO_ROOT.resolve() not in p.parents:
        raise HTTPException(status_code=404, detail="Export file missing.")
    return FileResponse(path=str(p), filename=Path(rel).name, media_type="application/zip")


@app.delete("/exports/{export_id}")
def exports_delete(export_id: str) -> JSONResponse:
    idx = _load_exports_index()
    items = [e for e in idx.get("exports", []) if isinstance(e, dict)]
    kept = []
    removed = None
    for e in items:
        if str(e.get("id")) == str(export_id):
            removed = e
            continue
        kept.append(e)
    if not removed:
        raise HTTPException(status_code=404, detail="Export not found.")
    idx["exports"] = kept
    _save_exports_index(idx)
    try:
        rel = str(removed.get("path") or "")
        p = (REPO_ROOT / rel).resolve()
        if p.exists() and REPO_ROOT.resolve() in p.parents:
            p.unlink()
    except Exception:
        pass
    return JSONResponse(status_code=200, content={"status": "ok"})


def _safe_zip_members(zf: zipfile.ZipFile) -> List[zipfile.ZipInfo]:
    out: List[zipfile.ZipInfo] = []
    for info in zf.infolist():
        name = info.filename.replace("\\", "/")
        if name.startswith("/") or name.startswith("../") or "/../" in name:
            continue
        out.append(info)
    return out


@app.get("/export")
def export_data(background: BackgroundTasks) -> FileResponse:
    """
    Export a safe subset of experiment artifacts as a zip.
    Excludes large files (>50MB) and model weights (*.gguf, *.pt).
    """

    runs_dir = (REPO_ROOT / "runs").resolve()
    if not runs_dir.exists():
        runs_dir.mkdir(parents=True, exist_ok=True)

    include_patterns = [
        "artin_aco_history.csv",
        "artin_aco_best.json",
        "topological_lm/**",
        "topological_ppo/**",
        "operator_*.json",
        "operator_*.md",
        "gemma_*.md",
        "gemma_*.json",
    ]

    def iter_included_files() -> List[Path]:
        files: List[Path] = []
        for pat in include_patterns:
            if pat.endswith("/**"):
                root = runs_dir / pat[:-3]
                if root.exists():
                    files.extend([p for p in root.rglob("*") if p.is_file()])
            elif "**" in pat:
                # e.g. topological_lm/**
                base = pat.split("/")[0]
                root = runs_dir / base
                if root.exists():
                    files.extend([p for p in root.rglob("*") if p.is_file()])
            else:
                files.extend([p for p in runs_dir.glob(pat) if p.is_file()])
        # de-dup while preserving order
        seen = set()
        out: List[Path] = []
        for p in files:
            rp = p.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            out.append(rp)
        return out

    def allowed_file(p: Path) -> bool:
        try:
            if not p.is_file():
                return False
            if runs_dir not in p.parents and p != runs_dir:
                return False
            if p.suffix.lower() in {".gguf", ".pt"}:
                return False
            if p.stat().st_size > 50 * 1024 * 1024:
                return False
            return True
        except Exception:
            return False

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_name = f"ant_rh_export_{ts}.zip"
    tmp_dir = Path(tempfile.mkdtemp(prefix="ant_rh_export_"))
    zip_path = tmp_dir / out_name

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in iter_included_files():
            if not allowed_file(p):
                continue
            rel = p.relative_to(runs_dir)
            z.write(p, arcname=str(Path("runs") / rel))

    def _cleanup():
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass

    background.add_task(_cleanup)
    return FileResponse(path=str(zip_path), filename=out_name, media_type="application/zip")


@app.post("/import")
async def import_data(file: UploadFile = File(...)) -> JSONResponse:
    """
    Import a previously exported zip into runs/ (safe allowlist).
    """

    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip uploads are supported.")

    # Save upload to temp file with size limit (~100MB)
    max_bytes = 100 * 1024 * 1024
    tmp_dir = Path(tempfile.mkdtemp(prefix="ant_rh_import_"))
    zip_path = tmp_dir / "upload.zip"
    written = 0
    try:
        with zip_path.open("wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > max_bytes:
                    raise HTTPException(status_code=413, detail="Upload too large (max ~100MB).")
                f.write(chunk)

        # Extract to temp dir
        extract_dir = tmp_dir / "unzipped"
        extract_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zf:
            members = _safe_zip_members(zf)
            # Only allow runs/ subtree
            for info in members:
                name = info.filename.replace("\\", "/")
                if not name.startswith("runs/"):
                    continue
                # Block models and large binaries by name
                low = name.lower()
                if low.endswith(".gguf") or low.endswith(".pt"):
                    continue
                zf.extract(info, path=str(extract_dir))

        runs_dir = (REPO_ROOT / "runs").resolve()
        runs_dir.mkdir(parents=True, exist_ok=True)

        # Allowlist copy rules
        allowed_prefixes = [
            "runs/topological_lm/",
            "runs/topological_ppo/",
        ]
        allowed_globs = [
            "runs/artin_aco_history.csv",
            "runs/artin_aco_best.json",
            "runs/operator_*.json",
            "runs/operator_*.md",
            "runs/gemma_*.md",
            "runs/gemma_*.json",
        ]

        def is_allowed(rel: str) -> bool:
            r = rel.replace("\\", "/")
            if any(r.startswith(p) for p in allowed_prefixes):
                return True
            for g in allowed_globs:
                # simple glob matching on the runs-relative path
                if Path(r).match(g):
                    return True
            return False

        imported = 0
        for p in extract_dir.rglob("*"):
            if not p.is_file():
                continue
            # Path traversal already blocked; compute relative to extract_dir
            rel = p.relative_to(extract_dir).as_posix()
            if not rel.startswith("runs/"):
                continue
            if not is_allowed(rel):
                continue
            # Enforce size limit per file and model exclusions
            if p.stat().st_size > 50 * 1024 * 1024:
                continue
            if p.suffix.lower() in {".gguf", ".pt"}:
                continue

            dest_rel = Path(rel).relative_to("runs")
            dest = (runs_dir / dest_rel).resolve()
            if runs_dir not in dest.parents and dest != runs_dir:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
            imported += 1

        return JSONResponse(status_code=200, content={"status": "ok", "files_imported": imported})
    finally:
        try:
            shutil.rmtree(tmp_dir, ignore_errors=True)
        except Exception:
            pass


@app.get("/operator/analysis")
def operator_analysis() -> JSONResponse:
    """
    Aggregate operator formula/analysis artifacts from runs/.
    Missing files are tolerated (null/empty fields).
    """

    def _read_text(path: str, *, limit: int) -> Optional[str]:
        try:
            p = Path(path)
            if not p.exists():
                return None
            txt = p.read_text(encoding="utf-8", errors="replace")
            if len(txt) > int(limit):
                return txt[: int(limit)] + "\n...[TRUNCATED]...\n"
            return txt
        except Exception:
            return None

    def _read_json(path: str) -> Optional[Dict[str, Any]]:
        try:
            p = Path(path)
            if not p.exists():
                return None
            obj = json.loads(p.read_text(encoding="utf-8", errors="replace"))
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    def _to_float_safe(v: Any) -> Optional[float]:
        try:
            if v is None:
                return None
            if isinstance(v, bool):
                return float(int(v))
            f = float(v)
            if f != f or abs(f) == float("inf"):
                return None
            return float(f)
        except Exception:
            return None

    candidates = [
        "runs/operator_pde_report.md",
        "runs/operator_pde_formula.tex",
        "runs/operator_pde_terms.csv",
        "runs/operator_symbolic.json",
        "runs/operator_stability_report.json",
        "runs/operator_sensitivity_report.json",
        "runs/artin_structured_report.json",
    ]
    source_files = [p for p in candidates if Path(p).exists()]

    formula_tex = _read_text("runs/operator_pde_formula.tex", limit=2000)
    pde_report_excerpt = _read_text("runs/operator_pde_report.md", limit=4000)

    symbolic = _read_json("runs/operator_symbolic.json") or {}
    formula_text = None
    if isinstance(symbolic, dict):
        for k in ("formula_text", "formula", "text", "operator_text"):
            v = symbolic.get(k)
            if isinstance(v, str) and v.strip():
                formula_text = v.strip()
                break
    if formula_text and len(formula_text) > 2000:
        formula_text = formula_text[:2000] + "\n...[TRUNCATED]...\n"
    if not formula_text:
        formula_text = (formula_tex or None)
        if formula_text and len(formula_text) > 2000:
            formula_text = formula_text[:2000] + "\n...[TRUNCATED]...\n"

    active_terms: List[Dict[str, Any]] = []
    terms_path = Path("runs/operator_pde_terms.csv")
    if terms_path.exists():
        try:
            with terms_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if not isinstance(row, dict):
                        continue
                    term = row.get("term") or row.get("Term") or row.get("name") or row.get("feature")
                    coef = row.get("coefficient") or row.get("coef") or row.get("weight") or row.get("Coefficient")
                    if not term:
                        continue
                    c = _to_float_safe(coef)
                    if c is None:
                        continue
                    active_terms.append(
                        {
                            "term": str(term).strip(),
                            "coefficient": float(c),
                            "abs_coefficient": float(abs(c)),
                        }
                    )
        except Exception:
            active_terms = []

    active_terms.sort(key=lambda x: float(x.get("abs_coefficient") or 0.0), reverse=True)
    if len(active_terms) > 200:
        active_terms = active_terms[:200]

    stability_report = _read_json("runs/operator_stability_report.json") or {}
    stability = {
        "self_adjoint_error": _to_float_safe(stability_report.get("self_adjoint_error_after"))
        or _to_float_safe(stability_report.get("symmetry_error_after")),
        "eigh_success": stability_report.get("eigh_success"),
        "spectral_loss": _to_float_safe(stability_report.get("spectral_loss")),
    }

    sensitivity_report = _read_json("runs/operator_sensitivity_report.json") or {}
    sensitivity = {
        "operator_distance_mean": _to_float_safe(sensitivity_report.get("operator_distance_mean")),
        "spectrum_distance_mean": _to_float_safe(sensitivity_report.get("spectrum_distance_mean")),
        "loss_std": _to_float_safe(sensitivity_report.get("loss_std")),
        "diagnosis": sensitivity_report.get("diagnosis") if isinstance(sensitivity_report.get("diagnosis"), str) else None,
    }

    structured = _read_json("runs/artin_structured_report.json") or {}
    structured_operator = {
        "final_loss": _to_float_safe(structured.get("final_loss") or structured.get("loss") or structured.get("total_loss")),
        "spectral_loss": _to_float_safe(structured.get("spectral_loss")),
        "spacing_loss": _to_float_safe(structured.get("spacing_loss")),
        "top_weights": structured.get("top_weights") if isinstance(structured.get("top_weights"), list) else [],
    }

    return JSONResponse(
        status_code=200,
        content={
            "formula_tex": formula_tex,
            "formula_text": formula_text,
            "pde_report_excerpt": pde_report_excerpt,
            "active_terms": active_terms,
            "stability": stability,
            "sensitivity": sensitivity,
            "structured_operator": structured_operator,
            "source_files": source_files,
        },
    )


@app.post("/jobs/start")
def jobs_start(payload: Dict[str, Any] = Body(...)) -> JSONResponse:
    job_key = str((payload or {}).get("job") or "").strip()
    params = (payload or {}).get("params") or {}
    if job_key not in ALLOWED_JOBS:
        raise HTTPException(status_code=400, detail="Job not allowed.")

    # Prevent duplicate starts for the same job key.
    if any(j.get("status") in {"running", "paused"} and str(j.get("job")) == job_key for j in JOBS.values()):
        return JSONResponse(
            status_code=409,
            content={
                "error": "job_conflict",
                "message": "A conflicting job is already running or paused.",
                "running_jobs": _running_jobs_summary(),
            },
        )

    if _running_jobs_count() >= MAX_CONCURRENT_JOBS:
        return JSONResponse(
            status_code=429,
            content={
                "error": "too_many_jobs",
                "message": "Too many jobs are already running. Wait for current jobs to finish.",
                "running_jobs": _running_jobs_summary(),
            },
        )

    job_id = uuid.uuid4().hex[:12]
    log_path = LOGS_DIR / f"{job_id}.log"

    if job_key == "aco":
        cmd = _build_aco_cmd(params if isinstance(params, dict) else {})
        name = "ACO"
    else:
        spec = ALLOWED_SIMPLE_JOBS[job_key]
        name = str(spec["name"])
        if spec.get("kind") == "cmd":
            cmd = list(spec.get("command") or [])
            if not cmd:
                raise HTTPException(status_code=500, detail="Invalid job spec.")
        else:
            cmd = ["make", str(spec["target"])]

    JOBS[job_id] = {
        "id": job_id,
        "job": job_key,
        "name": name,
        "command": cmd,
        "status": "running",
        "started_at": _now_iso(),
        "started_ts": time.time(),
        "timeout_seconds": int(JOB_TIMEOUT_S.get(job_key, 0) or 0),
        "ended_at": None,
        "ended_ts": None,
        "log_path": str(log_path),
        "returncode": None,
    }

    proc = _start_job_process(job_id, cmd, log_path)
    t = threading.Thread(target=_watch_job, args=(job_id, proc), daemon=True)
    t.start()

    return JSONResponse(status_code=200, content={"job_id": job_id})


@app.get("/jobs")
def jobs_list() -> JSONResponse:
    _jobs_watchdog()
    items = [_job_public(j) for j in JOBS.values()]
    items.sort(key=lambda j: str(j.get("started_at") or ""), reverse=True)
    return JSONResponse(status_code=200, content={"jobs": items})


@app.get("/jobs/{job_id}")
def jobs_get(job_id: str) -> JSONResponse:
    job = JOBS.get(str(job_id))
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return JSONResponse(status_code=200, content=_job_public(job))


@app.get("/jobs/summary")
def jobs_summary() -> JSONResponse:
    _jobs_watchdog()
    items = [_job_public(j) for j in JOBS.values()]
    running = [j for j in items if j.get("status") == "running"]
    latest_by_name: Dict[str, Any] = {}
    for j in items:
        name = str(j.get("name") or "")
        if not name:
            continue
        prev = latest_by_name.get(name)
        if not prev:
            latest_by_name[name] = j
            continue
        if str(j.get("started_at") or "") > str(prev.get("started_at") or ""):
            latest_by_name[name] = j
    return JSONResponse(status_code=200, content={"running_count": len(running), "latest_by_name": latest_by_name})


@app.post("/jobs/{job_id}/stop")
def jobs_stop(job_id: str) -> JSONResponse:
    job = JOBS.get(str(job_id))
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    pid = job.get("pid")
    if not pid:
        raise HTTPException(status_code=400, detail="Job has no PID (cannot stop).")
    try:
        import signal

        os.kill(int(pid), signal.SIGTERM)
        job["status"] = "failed"
        job["ended_at"] = _now_iso()
        job["ended_ts"] = time.time()
        job["returncode"] = -15
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse(status_code=200, content={"status": "ok"})


@app.post("/jobs/{job_id}/resume")
def jobs_resume(job_id: str) -> JSONResponse:
    job = JOBS.get(str(job_id))
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    pid = job.get("pid")
    if not pid:
        raise HTTPException(status_code=400, detail="Job has no PID (cannot resume).")
    try:
        import signal

        os.kill(int(pid), signal.SIGCONT)
        if job.get("status") == "paused":
            job["status"] = "running"
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse(status_code=200, content={"status": "ok"})


@app.get("/jobs/{job_id}/log")
def jobs_log(job_id: str, lines: int = Query(200, ge=1, le=5000)) -> JSONResponse:
    job = JOBS.get(str(job_id))
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    log_path = Path(str(job.get("log_path") or ""))
    out_lines = _read_last_lines(log_path, int(lines))
    return JSONResponse(status_code=200, content={"job_id": job_id, "lines": out_lines})


@app.get("/jobs/{job_id}/stream")
def jobs_stream(job_id: str):
    job = JOBS.get(str(job_id))
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return StreamingResponse(_stream_log(str(job_id)), media_type="text/event-stream")


@app.get("/health/gemma")
def health_gemma() -> JSONResponse:
    path = Path("runs/gemma_health_check.json")
    try:
        if not path.exists():
            return JSONResponse(
                status_code=200,
                content={"status": "unknown", "message": "Run make gemma-health first."},
            )
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            return JSONResponse(status_code=200, content={"status": "unknown", "message": "Invalid health JSON."})
        return JSONResponse(status_code=200, content=loaded)
    except Exception:
        return JSONResponse(status_code=200, content={"status": "unknown", "message": "Failed to read health report."})


@app.get("/status", response_model=StatusResponse)
def status() -> StatusResponse:
    cached = _cache_get("status")
    if cached is not None:
        return cached
    missing = []

    mem = read_text_rel_runs("gemma_project_memory.md", max_chars=20_000)
    if mem is None:
        missing.append("runs/gemma_project_memory.md")

    ga = read_json_rel_runs("gemma_analysis.json")
    if ga is None:
        missing.append("runs/gemma_analysis.json")

    osr = read_json_rel_runs("operator_stability_report.json")
    if osr is None:
        missing.append("runs/operator_stability_report.json")

    topo_md = read_text_rel_runs("topological_lm/report.md", max_chars=60_000)
    if topo_md is None:
        missing.append("runs/topological_lm/report.md")

    topo_eval = read_json_rel_runs("topological_lm/eval_report.json")
    if topo_eval is None:
        missing.append("runs/topological_lm/eval_report.json")

    # Compact extraction (avoid dumping entire files)
    aco = (ga or {}).get("aco", {}) if isinstance((ga or {}).get("aco"), dict) else {}
    aco_best_stats = aco.get("best_loss_stats", {}) if isinstance(aco.get("best_loss_stats"), dict) else {}
    aco_mean_stats = aco.get("mean_loss_stats", {}) if isinstance(aco.get("mean_loss_stats"), dict) else {}
    operator = (ga or {}).get("operator", {}) if isinstance((ga or {}).get("operator"), dict) else {}

    topo_metrics = topological_lm_metrics_from_eval_report(topo_eval or {})
    physics = _physics_metrics()

    head = None
    if mem:
        head = "\n".join(mem.strip().splitlines()[:12]).strip()

    return _cache_set(
        "status",
        StatusResponse(
        ok=True,
        missing=missing,
        gemma_main_issue=(ga or {}).get("main_issue") if isinstance(ga, dict) else None,
        gemma_learning=(ga or {}).get("learning") if isinstance(ga, dict) else None,
        aco_best_loss_last=(aco_best_stats.get("last") if isinstance(aco_best_stats, dict) else None),
        aco_best_loss_trend=(aco.get("best_loss_trend") if isinstance(aco, dict) else None),
        aco_mean_loss_last=(aco_mean_stats.get("last") if isinstance(aco_mean_stats, dict) else None),
        aco_mean_loss_trend=(aco.get("mean_loss_trend") if isinstance(aco, dict) else None),
        operator_spectral_loss=operator.get("spectral_loss") if isinstance(operator, dict) else None,
        operator_spacing_loss=operator.get("spacing_loss") if isinstance(operator, dict) else None,
        operator_total_loss=(osr or {}).get("total_loss") if isinstance(osr, dict) else None,
        operator_eigh_success=bool(operator.get("eigh_success")) if isinstance(operator, dict) and "eigh_success" in operator else None,
        topo_advantage_over_random=topo_metrics.get("advantage_over_random"),
        topo_random_mean_reward=topo_metrics.get("random_mean_reward"),
        topo_topological_lm_mean_reward=topo_metrics.get("topological_lm_mean_reward"),
        topo_unique_candidate_ratio=topo_metrics.get("unique_candidate_ratio"),
        topo_valid_braid_ratio=topo_metrics.get("valid_braid_ratio"),
        physics_self_adjoint_status=physics.get("self_adjoint_status"),
        physics_spectral_status=physics.get("spectral_status"),
        physics_otoc_indicator=physics.get("otoc_indicator"),
        physics_r_mean=physics.get("r_mean"),
        physics_source=physics.get("source"),
        project_memory_head=head,
        ),
    )


def _unknown_physics(source: str = "unknown") -> dict:
    return {
        "self_adjoint_status": "unknown",
        "self_adjoint_error": None,
        "spectral_status": "unknown",
        "otoc_indicator": "unknown",
        "r_mean": None,
        "spectrum_real": None,
        "spacing_std": None,
        "source": source,
    }


def _physics_metrics() -> dict:
    # Priority:
    # 1) runs/topological_lm/eval_report.json
    # 2) runs/operator_stability_report.json
    # 3) runs/operator_sensitivity_report.json

    # 1) TopologicalLM eval report (best available for r-statistic / otoc proxy)
    eval_report = read_json_rel_runs("topological_lm/eval_report.json")
    if isinstance(eval_report, dict):
        baselines = eval_report.get("baselines", {}) if isinstance(eval_report.get("baselines"), dict) else {}
        topo = baselines.get("TopologicalLM-only", {}) if isinstance(baselines.get("TopologicalLM-only"), dict) else {}
        dedup = topo.get("dedup", {}) if isinstance(topo.get("dedup"), dict) else {}
        raw = topo.get("raw", {}) if isinstance(topo.get("raw"), dict) else {}
        src = dedup if dedup else raw
        out = _unknown_physics("runs/topological_lm/eval_report.json")
        if isinstance(src, dict):
            out["self_adjoint_status"] = str(src.get("self_adjoint_status") or "unknown")
            out["spectral_status"] = str(src.get("spectral_status") or "unknown")
            out["otoc_indicator"] = str(src.get("otoc_indicator") or "unknown")
            out["r_mean"] = src.get("r_mean", None)

            # self_adjoint_error: estimate from top candidates (mean of finite values)
            cand = src.get("top_unique_candidates") or src.get("top_10_candidates") or []
            if isinstance(cand, list):
                vals = []
                for item in cand:
                    if not isinstance(item, dict):
                        continue
                    v = item.get("self_adjoint_error", None)
                    try:
                        if v is not None:
                            fv = float(v)
                            if fv == fv and abs(fv) != float("inf"):
                                vals.append(fv)
                    except Exception:
                        pass
                if vals:
                    out["self_adjoint_error"] = float(sum(vals) / len(vals))

            # spectrum_real / spacing_std are not explicitly logged in eval_report; infer minimally.
            if out["spectral_status"] == "ok":
                out["spectrum_real"] = True
            return out

    # 2) Operator stability report
    osr = read_json_rel_runs("operator_stability_report.json")
    if isinstance(osr, dict):
        out = _unknown_physics("runs/operator_stability_report.json")
        try:
            sym = osr.get("symmetry_error_after", None)
            fro = osr.get("fro_norm_after", None)
            if sym is not None and fro is not None:
                sym_f = float(sym)
                fro_f = float(fro)
                hdiff = sym_f / (fro_f + 1e-8)
                out["self_adjoint_error"] = float(hdiff)
                if hdiff < 1e-6:
                    out["self_adjoint_status"] = "ok"
                elif hdiff < 1e-3:
                    out["self_adjoint_status"] = "approx"
                else:
                    out["self_adjoint_status"] = "broken"
                out["spectrum_real"] = True
        except Exception:
            pass
        return out

    # 3) Operator sensitivity report (no detailed spectral structure; keep unknowns)
    sens = read_json_rel_runs("operator_sensitivity_report.json")
    if isinstance(sens, dict):
        return _unknown_physics("runs/operator_sensitivity_report.json")

    return _unknown_physics("none")


@app.get("/metrics/physics")
def metrics_physics() -> JSONResponse:
    cached = _cache_get("metrics_physics")
    if cached is not None:
        return JSONResponse(status_code=200, content=cached)
    val = _physics_metrics()
    _cache_set("metrics_physics", val)
    return JSONResponse(status_code=200, content=val)


@app.get("/metrics/aco", response_model=AcoMetricsResponse)
def metrics_aco() -> AcoMetricsResponse:
    cached = _cache_get("metrics_aco")
    if cached is not None:
        return cached
    rows = read_csv_rows_rel_runs("artin_aco_history.csv")
    m = aco_metrics_from_history(rows)
    return _cache_set("metrics_aco", AcoMetricsResponse(**m))


@app.get("/metrics/aco/history")
def metrics_aco_history(limit: int = Query(300, ge=1, le=50_000)) -> JSONResponse:
    """
    Return last N rows from runs/artin_aco_history.csv.

    Supports both CSV formats:
      - iter,best_loss,mean_loss
      - iter,best_loss,mean_loss,best_reward,mean_reward,reward_mode

    Missing columns become null. Malformed rows are skipped.
    """

    source = "runs/artin_aco_history.csv"
    path = Path(source)
    if not path.exists():
        return JSONResponse(status_code=200, content={"points": [], "n": 0, "source": source})

    def _to_float(v):
        if v is None:
            return None
        try:
            s = str(v).strip()
            if s == "" or s.lower() == "nan":
                return None
            return float(s)
        except Exception:
            return None

    def _to_int(v):
        if v is None:
            return None
        try:
            s = str(v).strip()
            if s == "" or s.lower() == "nan":
                return None
            return int(float(s))
        except Exception:
            return None

    def _is_numeric(s: str) -> bool:
        try:
            ss = str(s).strip()
            if ss == "" or ss.lower() == "nan":
                return False
            float(ss)
            return True
        except Exception:
            return False

    points = []
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            # Detect header vs no-header by inspecting first non-empty row.
            first_row = None
            while True:
                pos = f.tell()
                line = f.readline()
                if not line:
                    break
                if line.strip() == "":
                    continue
                first_row = line
                # rewind to beginning for the actual parser
                f.seek(pos)
                break

            if first_row is None:
                return JSONResponse(status_code=200, content={"points": [], "n": 0, "source": source})

            first_cells = [c.strip() for c in first_row.split(",")]
            no_header = len(first_cells) >= 3 and _is_numeric(first_cells[0]) and _is_numeric(first_cells[1]) and _is_numeric(first_cells[2])

            if no_header:
                reader = csv.reader(f)
                for row in reader:
                    if not row or len(row) < 3:
                        continue
                    it = _to_int(row[0])
                    if it is None:
                        continue
                    best_loss = _to_float(row[1])
                    mean_loss = _to_float(row[2])
                    best_reward = _to_float(row[3]) if len(row) > 3 else None
                    mean_reward = _to_float(row[4]) if len(row) > 4 else None
                    reward_mode = (str(row[5]).strip() or None) if len(row) > 5 and row[5] is not None else None
                    points.append(
                        {
                            "iter": it,
                            "best_loss": best_loss,
                            "mean_loss": mean_loss,
                            "best_reward": best_reward,
                            "mean_reward": mean_reward,
                            "reward_mode": reward_mode,
                        }
                    )
            else:
                reader = csv.DictReader(f)
                for row in reader:
                    if not isinstance(row, dict):
                        continue
                    it = _to_int(row.get("iter"))
                    if it is None:
                        continue
                    reward_mode = row.get("reward_mode", None)
                    reward_mode = (str(reward_mode).strip() or None) if reward_mode is not None else None
                    points.append(
                        {
                            "iter": it,
                            "best_loss": _to_float(row.get("best_loss")),
                            "mean_loss": _to_float(row.get("mean_loss")),
                            "best_reward": _to_float(row.get("best_reward")),
                            "mean_reward": _to_float(row.get("mean_reward")),
                            "reward_mode": reward_mode,
                        }
                    )
    except Exception:
        # Be forgiving: never crash the dashboard due to malformed CSV.
        return JSONResponse(status_code=200, content={"points": [], "n": 0, "source": source})

    if len(points) > int(limit):
        points = points[-int(limit) :]

    return JSONResponse(status_code=200, content={"points": points, "n": len(points), "source": source})


@app.get("/metrics/topological-lm", response_model=TopologicalLmMetricsResponse)
def metrics_topological_lm() -> TopologicalLmMetricsResponse:
    cached = _cache_get("metrics_topological_lm")
    if cached is not None:
        return cached
    report = read_json_rel_runs("topological_lm/eval_report.json") or {}
    m = topological_lm_metrics_from_eval_report(report)
    return _cache_set(
        "metrics_topological_lm",
        TopologicalLmMetricsResponse(
        random_mean_reward=m.get("random_mean_reward"),
        topological_lm_mean_reward=m.get("topological_lm_mean_reward"),
        advantage_over_random=m.get("advantage_over_random"),
        unique_candidate_ratio=m.get("unique_candidate_ratio"),
        valid_braid_ratio=m.get("valid_braid_ratio"),
        self_adjoint_status=m.get("self_adjoint_status"),
        spectral_status=m.get("spectral_status"),
        otoc_indicator=m.get("otoc_indicator"),
        r_mean=m.get("r_mean"),
        ),
    )


@app.get("/metrics/topological-lm/history")
def metrics_topological_lm_history(limit: int = Query(300, ge=1, le=50_000)) -> JSONResponse:
    """
    Best-effort history for TopologicalLM training/eval metrics.

    Reads the first existing CSV from:
      - runs/topological_lm/history.csv
      - runs/topological_lm/metrics_history.csv
      - runs/topo_history.csv

    Supported columns (missing -> null):
      - iter (or step)
      - mean_reward
      - unique_candidate_ratio
      - rewards_json (optional JSON list for histogram)
    """

    sources = [
        "runs/topological_lm/history.csv",
        "runs/topological_lm/metrics_history.csv",
        "runs/topo_history.csv",
    ]
    src = next((p for p in sources if Path(p).exists()), None)
    if not src:
        return JSONResponse(status_code=200, content={"points": [], "reward_samples": [], "n": 0, "source": None})

    def _to_float(v: Any) -> Optional[float]:
        try:
            if v is None:
                return None
            s = str(v).strip()
            if s == "" or s.lower() == "nan":
                return None
            f = float(s)
            if f != f or abs(f) == float("inf"):
                return None
            return float(f)
        except Exception:
            return None

    def _to_int(v: Any) -> Optional[int]:
        try:
            if v is None:
                return None
            s = str(v).strip()
            if s == "" or s.lower() == "nan":
                return None
            return int(float(s))
        except Exception:
            return None

    points: List[Dict[str, Any]] = []
    reward_samples: List[float] = []
    try:
        with Path(src).open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not isinstance(row, dict):
                    continue
                it = _to_int(row.get("iter")) or _to_int(row.get("step"))
                if it is None:
                    continue
                mr = _to_float(row.get("mean_reward"))
                ucr = _to_float(row.get("unique_candidate_ratio"))
                points.append({"iter": it, "mean_reward": mr, "unique_candidate_ratio": ucr})

                # Optional rewards list
                rj = row.get("rewards_json") or row.get("rewards") or None
                if rj and not reward_samples:
                    try:
                        arr = json.loads(str(rj))
                        if isinstance(arr, list):
                            for x in arr[:5000]:
                                fx = _to_float(x)
                                if fx is not None:
                                    reward_samples.append(fx)
                    except Exception:
                        pass
    except Exception:
        return JSONResponse(status_code=200, content={"points": [], "reward_samples": [], "n": 0, "source": src})

    if len(points) > int(limit):
        points = points[-int(limit) :]

    return JSONResponse(
        status_code=200,
        content={"points": points, "reward_samples": reward_samples, "n": len(points), "source": src},
    )


@app.get("/metrics/physics/history")
def metrics_physics_history(limit: int = Query(300, ge=1, le=50_000)) -> JSONResponse:
    """
    Best-effort history for physics diagnostics.

    Reads the first existing CSV from:
      - runs/physics_history.csv
      - runs/topological_lm/physics_history.csv

    Supported columns (missing -> null):
      - iter (or step)
      - r_mean
      - self_adjoint_error
    """

    sources = [
        "runs/physics_history.csv",
        "runs/topological_lm/physics_history.csv",
    ]
    src = next((p for p in sources if Path(p).exists()), None)
    if not src:
        return JSONResponse(status_code=200, content={"points": [], "n": 0, "source": None})

    def _to_float(v: Any) -> Optional[float]:
        try:
            if v is None:
                return None
            s = str(v).strip()
            if s == "" or s.lower() == "nan":
                return None
            f = float(s)
            if f != f or abs(f) == float("inf"):
                return None
            return float(f)
        except Exception:
            return None

    def _to_int(v: Any) -> Optional[int]:
        try:
            if v is None:
                return None
            s = str(v).strip()
            if s == "" or s.lower() == "nan":
                return None
            return int(float(s))
        except Exception:
            return None

    points: List[Dict[str, Any]] = []
    try:
        with Path(src).open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if not isinstance(row, dict):
                    continue
                it = _to_int(row.get("iter")) or _to_int(row.get("step"))
                if it is None:
                    continue
                points.append(
                    {
                        "iter": it,
                        "r_mean": _to_float(row.get("r_mean")),
                        "self_adjoint_error": _to_float(row.get("self_adjoint_error")),
                    }
                )
    except Exception:
        return JSONResponse(status_code=200, content={"points": [], "n": 0, "source": src})

    if len(points) > int(limit):
        points = points[-int(limit) :]

    return JSONResponse(status_code=200, content={"points": points, "n": len(points), "source": src})


@app.get("/metrics/spectral/spacing")
def metrics_spectral_spacing() -> JSONResponse:
    """
    Overlay operator spacing vs zeta spacing vs GUE vs Poisson baselines.
    """

    spectrum_sources = [
        "runs/artin_operator_spectrum.csv",
        "runs/artin_structured_spectrum.csv",
        "runs/stable_eigenvalues.csv",
    ]
    spectrum_src = next((p for p in spectrum_sources if Path(p).exists()), None)

    def _to_float(v: Any) -> Optional[float]:
        try:
            if v is None:
                return None
            s = str(v).strip()
            if s == "" or s.lower() == "nan":
                return None
            f = float(s)
            if f != f or abs(f) == float("inf"):
                return None
            return float(f)
        except Exception:
            return None

    def _read_eigenvalues(path: str) -> List[float]:
        vals: List[float] = []
        p = Path(path)
        if not p.exists():
            return vals
        try:
            # Try DictReader first (headered CSV)
            with p.open("r", encoding="utf-8", errors="replace", newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames:
                    keys = [k.lower() for k in reader.fieldnames if isinstance(k, str)]
                    cand_keys = []
                    for k in reader.fieldnames:
                        if not isinstance(k, str):
                            continue
                        kl = k.lower()
                        if any(t in kl for t in ("eigen", "lambda", "value", "eval", "eig")):
                            cand_keys.append(k)
                    use_key = cand_keys[0] if cand_keys else (reader.fieldnames[0] if reader.fieldnames else None)
                    for row in reader:
                        if not isinstance(row, dict):
                            continue
                        v = _to_float(row.get(use_key)) if use_key else None
                        if v is None:
                            # fallback: first parsable in row values
                            for vv in row.values():
                                v = _to_float(vv)
                                if v is not None:
                                    break
                        if v is not None:
                            vals.append(v)
                    return vals
        except Exception:
            pass

        # Fallback: csv.reader (no header)
        try:
            with p.open("r", encoding="utf-8", errors="replace", newline="") as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row:
                        continue
                    v = None
                    for cell in row:
                        v = _to_float(cell)
                        if v is not None:
                            break
                    if v is not None:
                        vals.append(v)
        except Exception:
            return []

        return vals

    def _spacings_from_sorted(vals: List[float]) -> List[float]:
        xs = [x for x in vals if isinstance(x, (int, float)) and float(x) == float(x) and abs(float(x)) != float("inf")]
        xs.sort()
        ss: List[float] = []
        for a, b in zip(xs, xs[1:]):
            d = float(b) - float(a)
            if d > 0 and d == d and abs(d) != float("inf"):
                ss.append(d)
        if not ss:
            return []
        m = sum(ss) / len(ss)
        if not m or m <= 0:
            return []
        return [s / m for s in ss]

    def _r_mean(spacings: List[float]) -> Optional[float]:
        if len(spacings) < 3:
            return None
        rs = []
        for s1, s2 in zip(spacings, spacings[1:]):
            if s1 <= 0 or s2 <= 0:
                continue
            rs.append(min(s1, s2) / max(s1, s2))
        if not rs:
            return None
        return float(sum(rs) / len(rs))

    def _hist_density(spacings: List[float], *, lo: float, hi: float, bins: int) -> Dict[str, Any]:
        if not spacings:
            edges = [lo + (hi - lo) * i / bins for i in range(bins + 1)]
            centers = [(edges[i] + edges[i + 1]) / 2 for i in range(bins)]
            return {"bins": centers, "hist": [0.0 for _ in range(bins)]}
        bw = (hi - lo) / bins
        counts = [0 for _ in range(bins)]
        kept = 0
        for s in spacings:
            if s < lo or s > hi:
                continue
            # Put hi into last bin
            idx = int((s - lo) / bw) if bw > 0 else 0
            if idx >= bins:
                idx = bins - 1
            if idx < 0:
                continue
            counts[idx] += 1
            kept += 1
        if kept == 0:
            edges = [lo + (hi - lo) * i / bins for i in range(bins + 1)]
            centers = [(edges[i] + edges[i + 1]) / 2 for i in range(bins)]
            return {"bins": centers, "hist": [0.0 for _ in range(bins)]}
        dens = [c / (kept * bw) for c in counts]
        edges = [lo + (hi - lo) * i / bins for i in range(bins + 1)]
        centers = [(edges[i] + edges[i + 1]) / 2 for i in range(bins)]
        return {"bins": centers, "hist": dens}

    operator_spacings: List[float] = []
    if spectrum_src:
        operator_spacings = _spacings_from_sorted(_read_eigenvalues(spectrum_src))

    zeta_src = "data/zeta_zeros.txt"
    zeta_spacings: List[float] = []
    if Path(zeta_src).exists():
        try:
            zeros: List[float] = []
            with Path(zeta_src).open("r", encoding="utf-8", errors="replace") as f:
                for ln in f:
                    s = ln.strip()
                    if not s or s.startswith("#"):
                        continue
                    # take first float on the line
                    tok = s.split()[0]
                    v = _to_float(tok)
                    if v is not None:
                        zeros.append(v)
            zeta_spacings = _spacings_from_sorted(zeros)
        except Exception:
            zeta_spacings = []

    # Histogram and curves
    lo, hi, nb = 0.0, 4.0, 50
    op_hist = _hist_density(operator_spacings, lo=lo, hi=hi, bins=nb)
    zz_hist = _hist_density(zeta_spacings, lo=lo, hi=hi, bins=nb)
    bins_centers = op_hist["bins"]

    def p_gue(s: float) -> float:
        # Wigner surmise (GUE): (32/pi^2) s^2 exp(-4 s^2 / pi)
        return (32.0 / (math.pi ** 2)) * (s ** 2) * math.exp((-4.0 * (s ** 2)) / math.pi)

    def p_poisson(s: float) -> float:
        return math.exp(-s)

    gue_curve = [p_gue(float(s)) for s in bins_centers]
    poisson_curve = [p_poisson(float(s)) for s in bins_centers]

    return JSONResponse(
        status_code=200,
        content={
            "hist_bins": bins_centers,
            "operator_spacing_hist": op_hist["hist"],
            "zeta_spacing_hist": (zz_hist["hist"] if zeta_spacings else []),
            "gue_curve": gue_curve,
            "poisson_curve": poisson_curve,
            "operator_r_mean": _r_mean(operator_spacings),
            "zeta_r_mean": _r_mean(zeta_spacings),
            "source": spectrum_src,
        },
    )


@app.post("/gemma/help", response_model=GemmaHelpResponse)
def gemma_help(req: GemmaHelpRequest = Body(...)) -> GemmaHelpResponse:
    try:
        answer = run_gemma_help(req.question, voice=req.voice, timeout_s=300)
    except ApiError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return GemmaHelpResponse(answer=answer)


STAGE_TO_MAKE_TARGET = {
    "study": "study",
    "analyze": "analyze-gemma",
    "journal": "lab-journal",
    "docs": "docs",
    "topo-eval": "topo-eval",
    "topo-report": "topo-report",
    "gemma-health": "gemma-health",
    "stability": "stability",
    "pde": "pde",
    "sensitivity": "sensitivity",
}


@app.post("/run/stage", response_model=RunResultResponse)
def run_stage(req: RunStageRequest = Body(...)) -> RunResultResponse:
    target = STAGE_TO_MAKE_TARGET.get(req.stage)
    if not target:
        raise HTTPException(status_code=400, detail="Invalid stage.")
    res = run_make_target(target, timeout_s=300)
    return RunResultResponse(
        ok=(res.returncode == 0 and not res.timed_out),
        target=f"make {target}",
        returncode=res.returncode,
        duration_s=res.duration_s,
        timed_out=res.timed_out,
        stdout=res.stdout,
        stderr=res.stderr,
    )


@app.post("/docs/update", response_model=RunResultResponse)
def docs_update() -> RunResultResponse:
    res = run_make_target("docs", timeout_s=300)
    return RunResultResponse(
        ok=(res.returncode == 0 and not res.timed_out),
        target="make docs",
        returncode=res.returncode,
        duration_s=res.duration_s,
        timed_out=res.timed_out,
        stdout=res.stdout,
        stderr=res.stderr,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8080)

