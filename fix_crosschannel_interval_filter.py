#!/usr/bin/env python3
"""
Patch fractal_dtes_aco_zeta_crosschannel.py so refined candidates are filtered
by the configured interval [t_min, t_max].

Usage:
    python3 fix_crosschannel_interval_filter.py
    python3 fix_crosschannel_interval_filter.py path/to/fractal_dtes_aco_zeta_crosschannel.py

The script creates a .bak backup before modifying the file.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path


FILTER_METHOD = r'''
    def filter_candidates_to_interval(self, candidates):
        """Keep only refined candidates inside the configured search interval.

        This prevents local refinement from returning zeros outside [t_min, t_max],
        e.g. returning t≈21 when the requested interval is [100, 400].
        """
        lo = float(self.cfg.t_min)
        hi = float(self.cfg.t_max)
        filtered = []
        rejected = []
        for t in candidates:
            tf = float(t)
            if lo <= tf <= hi:
                filtered.append(tf)
            else:
                rejected.append(tf)
        if rejected:
            print(
                f"[FILTER] rejected {len(rejected)} out-of-interval candidates "
                f"outside [{lo}, {hi}]: "
                + ", ".join(f"{x:.12f}" for x in rejected[:8])
                + (" ..." if len(rejected) > 8 else "")
            )
        return filtered
'''


def patch_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    text = path.read_text(encoding="utf-8")
    original = text

    if "def filter_candidates_to_interval" not in text:
        marker = "\ndef default_demo()"
        if marker in text:
            text = text.replace(marker, FILTER_METHOD + marker, 1)
        else:
            marker = "\nif __name__ == \"__main__\""
            if marker in text:
                text = text.replace(marker, FILTER_METHOD + marker, 1)
            else:
                # Last resort: append method text; user can still manually move it into the class.
                raise RuntimeError(
                    "Could not find insertion point. Please insert FILTER_METHOD inside "
                    "class FractalDTESACOZeta before default_demo()."
                )

    replacements = [
        (
            "return self.merge_close_candidates(candidates)",
            "merged = self.merge_close_candidates(candidates)\n        return self.filter_candidates_to_interval(merged)",
        ),
        (
            "candidates = self.merge_close_candidates(candidates)\n        return candidates",
            "candidates = self.merge_close_candidates(candidates)\n        candidates = self.filter_candidates_to_interval(candidates)\n        return candidates",
        ),
        (
            "merged = self.merge_close_candidates(candidates)\n        self.metrics = self.compute_metrics(candidate_nodes, merged)",
            "merged = self.merge_close_candidates(candidates)\n        merged = self.filter_candidates_to_interval(merged)\n        self.metrics = self.compute_metrics(candidate_nodes, merged)",
        ),
    ]

    changed_replacement = False
    for old, new in replacements:
        if old in text and new not in text:
            text = text.replace(old, new, 1)
            changed_replacement = True

    # Additional safety: patch local_refinement return if present.
    # This makes every refinement itself interval-safe.
    if "return best_t" in text and "if self.cfg.t_min <= best_t <= self.cfg.t_max:" not in text:
        text = text.replace(
            "return best_t",
            "if self.cfg.t_min <= best_t <= self.cfg.t_max:\n            return best_t\n        return None",
            1,
        )
        changed_replacement = True

    if text == original:
        print("[INFO] No changes made. File may already be patched.")
        return

    backup = path.with_suffix(path.suffix + ".bak")
    shutil.copy2(path, backup)
    path.write_text(text, encoding="utf-8")
    print(f"[OK] patched: {path}")
    print(f"[OK] backup:  {backup}")
    if not changed_replacement:
        print("[WARN] Inserted filter method, but did not find the expected return pattern.")
        print("       Manually ensure run() calls: candidates = self.filter_candidates_to_interval(candidates)")


def main() -> None:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("fractal_dtes_aco_zeta_crosschannel.py")
    patch_file(target)


if __name__ == "__main__":
    main()
