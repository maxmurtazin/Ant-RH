#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from typing import Iterable, Tuple


REQUIRED_ENDPOINTS = [
    "/health",
    "/debug/cors",
    "/status",
    "/metrics/aco",
    "/metrics/aco/history",
    "/metrics/topological-lm",
    "/metrics/physics",
    "/system/metrics",
    "/jobs",
    "/jobs/summary",
    "/jobs/queue",
    "/debug/routes",
    "/experiment/progress",
    "/metrics/spectral/spacing",
    "/operator/analysis",
    "/docs/playbook",
    "/exports",
    "/runs/compare",
]

OPTIONAL_ENDPOINTS = [
    "/health/gemma",
    "/metrics/topological-lm/history",
    "/metrics/physics/history",
]


def _get(base: str, path: str, timeout_s: float = 3.0, *, method: str = "GET", headers: dict | None = None) -> Tuple[int, str]:
    url = base.rstrip("/") + path
    req = urllib.request.Request(url, method=str(method or "GET"))
    if headers:
        for k, v in headers.items():
            req.add_header(str(k), str(v))
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return int(resp.status), "ok"
    except urllib.error.HTTPError as e:
        return int(getattr(e, "code", 0) or 0), "http_error"
    except Exception as e:
        return 0, f"error: {e.__class__.__name__}"


def _print_rows(rows: Iterable[Tuple[str, int, bool, str]]) -> None:
    print("endpoint\tstatus\tok\tinfo")
    for ep, st, ok, info in rows:
        print(f"{ep}\t{st}\t{('ok' if ok else 'fail')}\t{info}")


def check_endpoints(base_url: str) -> dict:
    base = str(base_url).strip()
    req_rows = []
    any_req_fail = False
    for ep in REQUIRED_ENDPOINTS:
        st, info = _get(base, ep)
        ok = st == 200
        if not ok:
            any_req_fail = True
        req_rows.append((ep, st, ok, info))

    opt_rows = []
    for ep in OPTIONAL_ENDPOINTS:
        st, info = _get(base, ep)
        ok = st == 200
        opt_rows.append((ep, st, ok, info))

    # CORS check: preflight OPTIONS /status from localhost:3000 should include allow-origin.
    cors_ok = True
    cors_info = "ok"
    try:
        url = base.rstrip("/") + "/status"
        req = urllib.request.Request(
            url,
            method="OPTIONS",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        with urllib.request.urlopen(req, timeout=3.0) as resp:
            acao = resp.headers.get("Access-Control-Allow-Origin")
            if not acao:
                cors_ok = False
                cors_info = "missing access-control-allow-origin"
    except Exception as e:
        cors_ok = False
        cors_info = f"cors_error: {e.__class__.__name__}"

    if not cors_ok:
        any_req_fail = True
        req_rows.append(("/cors(preflight /status)", 0, False, cors_info))

    return {"required": req_rows, "optional": opt_rows, "required_ok": (not any_req_fail)}


def main() -> int:
    ap = argparse.ArgumentParser(description="Preflight check Ant-RH dashboard backend endpoints")
    ap.add_argument("--base", type=str, required=True, help="Base URL, e.g. http://127.0.0.1:8084")
    args = ap.parse_args()

    base = str(args.base).strip()
    if not base.startswith("http://") and not base.startswith("https://"):
        print(f"ERROR: invalid base URL: {base}", file=sys.stderr)
        return 2

    res = check_endpoints(base)
    req_rows = res["required"]
    opt_rows = res["optional"]

    print("== Required endpoints ==")
    _print_rows(req_rows)
    print("")
    print("== Optional endpoints ==")
    _print_rows(opt_rows)
    print("")

    if not res["required_ok"]:
        bad = [ep for ep, st, ok, _ in req_rows if not ok]
        print(f"ERROR: preflight failed. Missing/failed required endpoints: {bad}", file=sys.stderr)
        return 1

    print("OK: preflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

