#!/usr/bin/env python3
"""
patch_compare_all_suffix_fix.py

Fixes invalid Path.with_suffix("_name.ext") calls in compare_dtes_vs_truth.py.

Problem:
    prefix.with_suffix("_comparison.json")  # invalid
    prefix.with_suffix("_timeline.png")     # invalid

Correct:
    prefix.with_name(prefix.name + "_comparison.json")
    prefix.with_name(prefix.name + "_timeline.png")

Usage:
    python3 patch_compare_all_suffix_fix.py compare_dtes_vs_truth.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


BAD_SUFFIXES = [
    "_comparison.json",
    "_matches.csv",
    "_false_positives.csv",
    "_missed.csv",
    "_precision_recall_by_tol.csv",
    "_summary.md",
    "_timeline.png",
    "_match_error_hist.png",
    "_nearest_distance.png",
    "_precision_recall_by_tol.png",
    "_spacing_overlay.png",
]


def patch_file(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    original = text

    for suffix in BAD_SUFFIXES:
        text = text.replace(
            f'prefix.with_suffix("{suffix}")',
            f'prefix.with_name(prefix.name + "{suffix}")',
        )
        text = text.replace(
            f"prefix.with_suffix('{suffix}')",
            f'prefix.with_name(prefix.name + "{suffix}")',
        )

    # Generic fallback: fix any remaining prefix.with_suffix("_...") pattern.
    text = re.sub(
        r'prefix\.with_suffix\((["\'])(_[^"\']+\.[^"\']+)\1\)',
        r'prefix.with_name(prefix.name + "\2")',
        text,
    )

    if text == original:
        print("[OK] No changes needed or file already patched.")
    else:
        backup = path.with_suffix(path.suffix + ".bak")
        backup.write_text(original, encoding="utf-8")
        path.write_text(text, encoding="utf-8")
        print(f"[OK] patched: {path}")
        print(f"[BACKUP] {backup}")

    remaining = re.findall(r'prefix\.with_suffix\((["\'])_[^"\']+\1\)', text)
    if remaining:
        print("[WARN] Some invalid prefix.with_suffix calls may remain. Run:")
        print("       grep -n 'with_suffix(\"_' compare_dtes_vs_truth.py")
    else:
        print("[CHECK] No invalid prefix.with_suffix(\"_...\") calls remain.")


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python3 patch_compare_all_suffix_fix.py compare_dtes_vs_truth.py")
        raise SystemExit(2)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"[ERROR] File not found: {path}")
        raise SystemExit(1)

    patch_file(path)


if __name__ == "__main__":
    main()
