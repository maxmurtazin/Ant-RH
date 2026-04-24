#!/usr/bin/env python3
"""
gap_detector.py

Find large gaps in DTES candidate coverage.

Ant-RH refinement:
    Use after a candidate JSON is produced to see where the sampler may need
    wider windows or extra anchors before hybrid scanning.
"""

from __future__ import annotations

import argparse, json
from pathlib import Path


def load_ts(path: Path):
    data = json.loads(path.read_text(encoding="utf-8"))
    seq = data.get("candidates", data if isinstance(data, list) else [])
    ts = []
    for x in seq:
        if isinstance(x, dict):
            ts.append(float(x["t"]))
        else:
            ts.append(float(x))
    return sorted(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--threshold", type=float, default=1.0)
    ap.add_argument("--out", default="gaps.json")
    args = ap.parse_args()
    ts = load_ts(Path(args.candidates))
    gaps = []
    for a, b in zip(ts[:-1], ts[1:]):
        if b - a >= args.threshold:
            gaps.append({"left": a, "right": b, "gap": b - a, "mid": 0.5 * (a + b)})
    Path(args.out).write_text(json.dumps({"gaps": gaps, "count": len(gaps)}, indent=2), encoding="utf-8")
    print(f"[SAVE] {args.out} | gaps={len(gaps)}")


if __name__ == "__main__":
    main()
