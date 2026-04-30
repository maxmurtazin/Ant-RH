#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple


Status = Literal["ok", "failed", "missing_optional", "timeout"]


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = REPO_ROOT / "runs"


@dataclass
class CheckResult:
    name: str
    status: Status
    latency_s: float
    message: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "latency_s": round(float(self.latency_s), 4),
            "message": self.message,
        }


def _now() -> float:
    return time.perf_counter()


def _run(
    cmd: Sequence[str],
    *,
    timeout_s: int,
    cwd: Optional[Path] = None,
    env: Optional[Dict[str, str]] = None,
) -> Tuple[int, float, str, str, bool]:
    start = _now()
    try:
        p = subprocess.run(
            list(cmd),
            cwd=str(cwd) if cwd else None,
            env=env,
            capture_output=True,
            text=True,
            timeout=int(timeout_s),
            check=False,
        )
        dur = _now() - start
        return int(p.returncode), dur, (p.stdout or ""), (p.stderr or ""), False
    except subprocess.TimeoutExpired as e:
        dur = _now() - start
        out = (e.stdout or "") if isinstance(e.stdout, str) else ""
        err = (e.stderr or "") if isinstance(e.stderr, str) else ""
        return 124, dur, out, err, True


def _truncate(text: str, max_chars: int = 500) -> str:
    t = (text or "").strip()
    if len(t) <= max_chars:
        return t
    return t[:max_chars] + " …[truncated]"


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _fmt_row_csv(name: str, status: str, latency_s: float, message: str) -> str:
    # print a CSV row without external deps; quote only when needed
    def q(s: str) -> str:
        s = str(s)
        if any(ch in s for ch in [",", "\n", '"']):
            return '"' + s.replace('"', '""') + '"'
        return s

    return ",".join([q(name), q(status), q(f"{latency_s:.3f}"), q(message)])


def check_llama_cli(which_name: str, timeout_s: int) -> CheckResult:
    start = _now()
    path = shutil.which(which_name)
    dur = _now() - start
    if not path:
        return CheckResult("llama-cli", "failed", dur, f"not found in PATH: {which_name}")
    # sanity: also run `which`
    rc, dur2, out, err, to = _run(["which", which_name], timeout_s=timeout_s, cwd=REPO_ROOT)
    if to:
        return CheckResult("llama-cli", "timeout", dur2, "timeout running `which`")
    if rc != 0:
        return CheckResult("llama-cli", "failed", dur2, _truncate(err or out or "which failed"))
    return CheckResult("llama-cli", "ok", dur2, f"found: {out.strip()}")


def check_model_file(name: str, path: str) -> CheckResult:
    start = _now()
    p = Path(path)
    exists = p.exists()
    dur = _now() - start
    if exists:
        try:
            size_mb = p.stat().st_size / (1024 * 1024)
            return CheckResult(name, "ok", dur, f"exists ({size_mb:.1f} MB)")
        except Exception:
            return CheckResult(name, "ok", dur, "exists")
    return CheckResult(name, "failed", dur, "missing")


def check_direct_llama(llama_cli: str, model_path: str, timeout_s: int) -> CheckResult:
    cmd = [llama_cli, "-m", model_path, "-p", "Return OK", "-n", "8"]
    rc, dur, out, err, to = _run(cmd, timeout_s=timeout_s, cwd=REPO_ROOT)
    if to:
        return CheckResult("direct_llama", "timeout", dur, "timed out")
    combined = (out + "\n" + err).strip()
    if rc != 0:
        return CheckResult("direct_llama", "failed", dur, _truncate(combined or f"exit code {rc}"))
    if "OK" not in combined.upper():
        return CheckResult("direct_llama", "failed", dur, _truncate("did not find 'OK' in output"))
    return CheckResult("direct_llama", "ok", dur, "OK")


def check_planner(llama_cli: str, planner_model: str, timeout_s: int) -> CheckResult:
    cmd = ["python3", "-m", "core.gemma_planner", "--test", "--llama_cli", llama_cli, "--model_path", planner_model]
    rc, dur, out, err, to = _run(cmd, timeout_s=timeout_s, cwd=REPO_ROOT)
    if to:
        return CheckResult("planner", "timeout", dur, "timed out")
    msg = _truncate(err or out)
    return CheckResult("planner", "ok" if rc == 0 else "failed", dur, msg or f"exit code {rc}")


def check_analyzer(llama_cli: str, analyzer_model: str, timeout_s: int) -> CheckResult:
    cmd = ["python3", "analysis/gemma_analyzer.py", "--test", "--llama_cli", llama_cli, "--model_path", analyzer_model]
    rc, dur, out, err, to = _run(cmd, timeout_s=timeout_s, cwd=REPO_ROOT)
    if to:
        return CheckResult("analyzer", "timeout", dur, "timed out")
    msg = _truncate(err or out)
    return CheckResult("analyzer", "ok" if rc == 0 else "failed", dur, msg or f"exit code {rc}")


def check_help(llama_cli: str, planner_model: str, timeout_s: int) -> CheckResult:
    cmd = [
        "python3",
        "help/gemma_help_agent.py",
        "--question",
        "Return OK only.",
        "--voice",
        "False",
        "--stream",
        "False",
        "--llama_cli",
        llama_cli,
        "--model_path",
        planner_model,
        "--interactive",
        "False",
        "--memory",
        "False",
        "--max_tokens",
        "16",
    ]
    rc, dur, out, err, to = _run(cmd, timeout_s=timeout_s, cwd=REPO_ROOT)
    if to:
        return CheckResult("help", "timeout", dur, "timed out")
    combined = (out + "\n" + err).strip()
    if rc != 0:
        return CheckResult("help", "failed", dur, _truncate(combined or f"exit code {rc}"))
    if "OK" not in combined.upper():
        return CheckResult("help", "failed", dur, _truncate("did not find 'OK' in output"))
    return CheckResult("help", "ok", dur, "OK")


def _optional_agent_check(name: str, cmd: Sequence[str], timeout_s: int) -> CheckResult:
    rc, dur, out, err, to = _run(cmd, timeout_s=timeout_s, cwd=REPO_ROOT)
    if to:
        return CheckResult(name, "timeout", dur, "timed out")
    combined = (out + "\n" + err).strip()
    if rc == 0:
        return CheckResult(name, "ok", dur, _truncate(combined) or "ok")
    low = combined.lower()
    if "unrecognized arguments" in low and "--dry_run" in low:
        return CheckResult(name, "missing_optional", dur, "dry_run not supported (non-fatal)")
    if "no such file" in low or "cannot open file" in low or "module not found" in low:
        return CheckResult(name, "missing_optional", dur, _truncate(combined) or "missing (non-fatal)")
    return CheckResult(name, "missing_optional", dur, _truncate(combined) or f"exit code {rc} (non-fatal)")


def overall_status(checks: List[CheckResult]) -> Literal["ok", "degraded", "failed"]:
    # Per spec:
    # - "failed" only when llama-cli is missing OR both models are missing.
    # - Everything else (timeouts, partial agent failures, one model missing) is "degraded".
    by_name = {c.name: c for c in checks}
    llama = by_name.get("llama-cli")
    planner = by_name.get("planner_model")
    analyzer = by_name.get("analyzer_model")

    llama_missing = llama is not None and llama.status != "ok"
    planner_missing = planner is not None and planner.status != "ok"
    analyzer_missing = analyzer is not None and analyzer.status != "ok"

    if llama_missing or (planner_missing and analyzer_missing):
        return "failed"
    any_failed = any(c.status in {"failed", "timeout"} for c in checks)
    if any_failed:
        return "degraded"
    return "ok"


def render_md(overall: str, checks: List[CheckResult]) -> str:
    lines = []
    lines.append("# Gemma Agents Health Check")
    lines.append("")
    lines.append(f"- overall_status: **{overall}**")
    lines.append("")
    lines.append("## Checks")
    lines.append("")
    lines.append("| name | status | latency_s | message |")
    lines.append("|---|---:|---:|---|")
    for c in checks:
        msg = (c.message or "").replace("\n", " ")
        lines.append(f"| `{c.name}` | **{c.status}** | {c.latency_s:.3f} | {msg} |")
    lines.append("")
    lines.append("Core failure criteria: `llama-cli` missing, model(s) missing, or direct llama test fails.")
    lines.append("Optional checks may show as `missing_optional` without failing the run.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Gemma agents health check for Ant-RH")
    ap.add_argument("--llama_cli", type=str, default="llama-cli")
    ap.add_argument("--planner_model", type=str, required=True)
    ap.add_argument("--analyzer_model", type=str, required=True)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--global_timeout", type=int, default=180)
    args = ap.parse_args()

    timeout_s = int(args.timeout)
    global_timeout_s = int(args.global_timeout)
    llama_cli = str(args.llama_cli)
    planner_model = str(args.planner_model)
    analyzer_model = str(args.analyzer_model)

    checks: List[CheckResult] = []
    global_start = time.time()
    hit_global_timeout = False

    def _global_ok() -> bool:
        return (time.time() - global_start) <= global_timeout_s

    def _stop_if_global_timeout() -> bool:
        nonlocal hit_global_timeout
        if _global_ok():
            return False
        hit_global_timeout = True
        return True

    def _write_and_exit_partial() -> None:
        overall = "degraded"
        payload = {"overall_status": overall, "checks": [c.to_dict() for c in checks]}
        _write_json(RUNS_DIR / "gemma_health_check.json", payload)
        _write_text(RUNS_DIR / "gemma_health_check.md", render_md(overall, checks))
        raise SystemExit(0)

    checks.append(check_llama_cli(llama_cli, timeout_s=timeout_s))
    checks.append(check_model_file("planner_model", planner_model))
    checks.append(check_model_file("analyzer_model", analyzer_model))

    planner_exists = Path(planner_model).exists()
    analyzer_exists = Path(analyzer_model).exists()
    cli_ok = next((c for c in checks if c.name == "llama-cli"), None)

    if _stop_if_global_timeout():
        _write_and_exit_partial()

    if cli_ok and cli_ok.status == "ok" and planner_exists:
        checks.append(check_direct_llama(llama_cli, planner_model, timeout_s=timeout_s))
    else:
        checks.append(CheckResult("direct_llama", "failed", 0.0, "skipped (llama-cli or planner model missing)"))

    if _stop_if_global_timeout():
        _write_and_exit_partial()

    if cli_ok and cli_ok.status == "ok" and planner_exists:
        checks.append(check_planner(llama_cli, planner_model, timeout_s=timeout_s))
    else:
        checks.append(CheckResult("planner", "failed", 0.0, "skipped (llama-cli or planner model missing)"))

    if _stop_if_global_timeout():
        _write_and_exit_partial()

    if cli_ok and cli_ok.status == "ok" and analyzer_exists:
        checks.append(check_analyzer(llama_cli, analyzer_model, timeout_s=timeout_s))
    else:
        checks.append(CheckResult("analyzer", "failed", 0.0, "skipped (llama-cli or analyzer model missing)"))

    if _stop_if_global_timeout():
        _write_and_exit_partial()

    if cli_ok and cli_ok.status == "ok" and planner_exists:
        checks.append(check_help(llama_cli, planner_model, timeout_s=timeout_s))
    else:
        checks.append(CheckResult("help", "failed", 0.0, "skipped (llama-cli or planner model missing)"))

    # Optional checks as requested (non-fatal if unsupported)
    if not _stop_if_global_timeout():
        checks.append(_optional_agent_check("lab_journal", ["python3", "analysis/gemma_lab_journal.py", "--dry_run"], timeout_s=timeout_s))
    if not _stop_if_global_timeout():
        checks.append(_optional_agent_check("paper_writer", ["python3", "analysis/gemma_paper_writer.py", "--dry_run"], timeout_s=timeout_s))
    if not _stop_if_global_timeout():
        checks.append(_optional_agent_check("docs_builder", ["python3", "analysis/gemma_docs_builder.py", "--dry_run"], timeout_s=timeout_s))
    if not _stop_if_global_timeout():
        checks.append(_optional_agent_check("literature_study", ["python3", "analysis/gemma_literature_study.py", "--dry_run"], timeout_s=timeout_s))

    if hit_global_timeout:
        overall = "degraded"
    else:
        overall = overall_status(checks)

    # Print CSV table
    print("agent,status,latency_s,message")
    for c in checks:
        print(_fmt_row_csv(c.name, c.status, c.latency_s, c.message))

    payload = {"overall_status": overall, "checks": [c.to_dict() for c in checks]}
    _write_json(RUNS_DIR / "gemma_health_check.json", payload)
    _write_text(RUNS_DIR / "gemma_health_check.md", render_md(overall, checks))

    # Exit code: 1 only if llama-cli missing OR both models missing.
    if overall == "failed":
        raise SystemExit(1)
    raise SystemExit(0)


if __name__ == "__main__":
    main()

