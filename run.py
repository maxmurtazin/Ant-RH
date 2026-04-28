#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", type=str, required=True)
    args = parser.parse_args()

    if args.module == "artin_operator":
        from core.artin_operator import main as _main

        _main()
    elif args.module == "aco":
        from core.artin_aco import main as _main

        _main()
    elif args.module == "rl":
        from validation.artin_rl_train import main as _main

        _main()
    elif args.module == "selberg":
        from validation.selberg_trace_loss import main as _main

        _main()
    elif args.module == "stability":
        from validation.operator_stability_report import main as _main

        _main()
    else:
        raise SystemExit(f"unknown --module {args.module!r}")


if __name__ == "__main__":
    main()

