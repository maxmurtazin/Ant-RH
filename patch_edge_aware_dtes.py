#!/usr/bin/env python3

from __future__ import annotations

import re
import sys
from pathlib import Path

HELPER = r"""
def _edge_aware_candidates(candidates, t_min, t_max, edge_padding=2.5, edge_step=0.05):
    import math

    cleaned = []
    for t in candidates:
        try:
            tt = float(t)
        except:
            continue
        if t_min <= tt <= t_max:
            cleaned.append(tt)

    anchors = []
    n = int(math.ceil(edge_padding / edge_step))
    for i in range(n + 1):
        d = i * edge_step
        anchors.append(t_min + d)
        anchors.append(t_max - d)

    merged = cleaned + anchors
    merged = sorted(set(round(float(t), 12) for t in merged))
    return merged


def _autosave_dtes_candidates(candidates, path="dtes_candidates.json"):
    import json
    rows = []
    for i, t in enumerate(candidates, start=1):
        try:
            tt = float(t)
        except:
            continue
        rows.append({
            "rank": i,
            "t": tt
        })

    with open(path, "w", encoding="utf-8") as f:
        json.dump({"candidates": rows}, f, indent=2)

    print(f"[SAVE] {path} | count={len(rows)}")
"""


def patch_file(path: Path):
    text = path.read_text()

    if "_edge_aware_candidates" not in text:
        text = HELPER + "\n" + text

    if "_edge_aware_candidates(candidates" not in text:
        text = re.sub(
            r"(candidates\s*=\s*searcher\.run\(\))",
            r"\1\n    candidates = _edge_aware_candidates(candidates, args.t_min, args.t_max)\n    _autosave_dtes_candidates(candidates)",
            text,
        )

    path.write_text(text)
    print("[OK] patched")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 patch_edge_aware_dtes.py file.py")
        exit(1)

    patch_file(Path(sys.argv[1]))
