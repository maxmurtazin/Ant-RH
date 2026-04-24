#!/usr/bin/env python3
"""
fix_future_import_order.py

Fixes Python files broken by prepending code before:
    from __future__ import annotations

Usage:
    python3 fix_future_import_order.py fractal_dtes_aco_zeta_crosschannel.py
"""

from __future__ import annotations

import sys
from pathlib import Path


def fix(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    original = text

    future_line = "from __future__ import annotations"
    if future_line not in text:
        print("[INFO] no future import found; nothing to fix")
        return

    # Remove all occurrences.
    text = text.replace(future_line + "\n", "")
    text = text.replace(future_line, "")

    lines = text.splitlines()
    insert_at = 0

    # Keep shebang first.
    if lines and lines[0].startswith("#!"):
        insert_at = 1

    # Keep module docstring before future import if present.
    i = insert_at
    while i < len(lines) and not lines[i].strip():
        i += 1

    if i < len(lines):
        stripped = lines[i].lstrip()
        quote = None
        if stripped.startswith(chr(34) * 3):
            quote = chr(34) * 3
        elif stripped.startswith(chr(39) * 3):
            quote = chr(39) * 3

        if quote is not None:
            if lines[i].count(quote) >= 2:
                insert_at = i + 1
            else:
                j = i + 1
                while j < len(lines):
                    if quote in lines[j]:
                        insert_at = j + 1
                        break
                    j += 1

    lines.insert(insert_at, future_line)
    fixed = "\n".join(lines) + "\n"

    backup = path.with_suffix(path.suffix + ".futurefix.bak")
    backup.write_text(original, encoding="utf-8")
    path.write_text(fixed, encoding="utf-8")

    print(f"[OK] fixed future import order: {path}")
    print(f"[BACKUP] {backup}")


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python3 fix_future_import_order.py file.py")
        raise SystemExit(2)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"[ERROR] file not found: {path}")
        raise SystemExit(1)

    fix(path)


if __name__ == "__main__":
    main()
