#!/usr/bin/env python3
from __future__ import annotations

import argparse
import socket
import sys
from pathlib import Path
from typing import Optional

import uvicorn


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def is_port_free(host: str, port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, int(port)))
            return True
    except OSError:
        return False


def pick_port(host: str, preferred_port: int, max_port: int) -> Optional[int]:
    for p in range(int(preferred_port), int(max_port) + 1):
        if is_port_free(host, p):
            return p
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Ant-RH dashboard with auto port selection.")
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--max_port", type=int, default=8099)
    ap.add_argument("--reload", type=str, default="True")
    ap.add_argument(
        "--write_port_file",
        type=str,
        default=None,
        help="If set, write the selected port to this file (e.g. runs/dashboard_port.txt).",
    )
    args = ap.parse_args()

    host = str(args.host)
    if host != "127.0.0.1":
        raise SystemExit("Refusing to bind non-localhost by default. Use --host 127.0.0.1.")

    port = pick_port(host, int(args.port), int(args.max_port))
    if port is None:
        raise SystemExit(f"No free ports in range {args.port}-{args.max_port} on {host}.")

    reload_flag = str(args.reload).lower() in {"1", "true", "yes", "y", "t"}

    if args.write_port_file:
        out_path = Path(str(args.write_port_file)).expanduser()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(str(int(port)))
            f.flush()

    print("")
    print("Dashboard running:")
    print(f"http://{host}:{port}")
    print("")
    print(f"Dashboard API running on selected port: {port}")
    print("")

    if reload_flag:
        # Uvicorn reload requires an import string.
        uvicorn.run(
            "api.server:app",
            host=host,
            port=int(port),
            reload=True,
        )
    else:
        from api.server import app  # noqa: PLC0415

        uvicorn.run(
            app,
            host=host,
            port=int(port),
            reload=False,
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)

