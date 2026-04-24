#!/usr/bin/env python3

from pathlib import Path
import re
import sys

REPLACEMENTS = [
    ("_comparison.json", '_comparison.json'),
    ("_matches.csv", '_matches.csv'),
    ("_false_positives.csv", '_false_positives.csv'),
    ("_missed.csv", '_missed.csv'),
    ("_precision_recall_by_tol.csv", '_precision_recall_by_tol.csv'),
    ("_summary.md", '_summary.md'),
]


def patch_file(path: Path):
    text = path.read_text()

    for suffix, name in REPLACEMENTS:
        text = re.sub(
            rf'prefix\.with_suffix\("{re.escape(suffix)}"\)',
            f'prefix.with_name(prefix.name + "{name}")',
            text
        )

    path.write_text(text)
    print(f"[OK] patched: {path}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 patch_compare_suffix_fix.py compare_dtes_vs_truth.py")
        return

    file_path = Path(sys.argv[1])

    if not file_path.exists():
        print("File not found")
        return

    patch_file(file_path)


if __name__ == "__main__":
    main()