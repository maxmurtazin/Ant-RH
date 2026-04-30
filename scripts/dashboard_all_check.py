#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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
    "/experiment/progress",
    "/metrics/spectral/spacing",
    "/operator/analysis",
    "/docs/playbook",
    "/exports",
    "/runs/compare",
    "/debug/routes",
]


def _get(base: str, path: str, timeout_s: float = 3.0, *, method: str = "GET", headers: dict | None = None) -> Tuple[int, bytes, dict]:
    url = base.rstrip("/") + path
    req = urllib.request.Request(url, method=str(method or "GET"))
    if headers:
        for k, v in headers.items():
            req.add_header(str(k), str(v))
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            return int(resp.status), resp.read() or b"", dict(resp.headers.items())
    except urllib.error.HTTPError as e:
        try:
            body = e.read() or b""
        except Exception:
            body = b""
        return int(getattr(e, "code", 0) or 0), body, dict(getattr(e, "headers", {}) or {})
    except Exception:
        return 0, b"", {}


def _print_rows(rows: Iterable[Tuple[str, int, bool, str]]) -> None:
    print("endpoint\tstatus\tok\tinfo")
    for ep, st, ok, info in rows:
        print(f"{ep}\t{st}\t{('ok' if ok else 'fail')}\t{info}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Unified runtime check for Ant-RH dashboard backend")
    ap.add_argument("--base", type=str, required=True, help="Base URL, e.g. http://127.0.0.1:8084")
    args = ap.parse_args()

    base = str(args.base).strip()
    if not base.startswith("http://") and not base.startswith("https://"):
        print(f"ERROR: invalid base URL: {base}", file=sys.stderr)
        return 2

    rows: list[Tuple[str, int, bool, str]] = []
    any_fail = False

    for ep in REQUIRED_ENDPOINTS:
        st, body, _hdrs = _get(base, ep)
        ok = st == 200
        if not ok:
            any_fail = True
        rows.append((ep, st, ok, "ok" if ok else "failed"))

        if ep == "/debug/routes" and ok:
            try:
                routes = json.loads(body.decode("utf-8", errors="replace"))
                if not isinstance(routes, list):
                    raise ValueError("routes not list")
                for must in ("/jobs/summary", "/jobs/queue"):
                    if must not in routes:
                        any_fail = True
                        rows.append((f"{ep} contains {must}", 200, False, "missing"))
            except Exception as e:
                any_fail = True
                rows.append((f"{ep} parse", 200, False, f"error: {e.__class__.__name__}"))

        if ep == "/metrics/aco/history" and ok:
            try:
                payload = json.loads(body.decode("utf-8", errors="replace"))
                pts = payload.get("points") if isinstance(payload, dict) else None
                if pts is None:
                    raise ValueError("missing points")
            except Exception as e:
                any_fail = True
                rows.append((f"{ep} shape", 200, False, f"error: {e.__class__.__name__}"))

    # CORS check: preflight OPTIONS /status from localhost:3000 should include allow-origin.
    try:
        st, _body, hdrs = _get(
            base,
            "/status",
            method="OPTIONS",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        acao = ""
        for k, v in hdrs.items():
            if str(k).lower() == "access-control-allow-origin":
                acao = str(v)
                break
        ok = (st in (200, 204)) and bool(acao)
        if not ok:
            any_fail = True
        rows.append(("/cors(preflight /status)", st, ok, "ok" if ok else "missing allow-origin"))
    except Exception as e:
        any_fail = True
        rows.append(("/cors(preflight /status)", 0, False, f"error: {e.__class__.__name__}"))

    print("== Dashboard all-check (live) ==")
    _print_rows(rows)

    if any_fail:
        print("ERROR: dashboard all-check failed.", file=sys.stderr)
        return 1

    print("OK: dashboard all-check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

