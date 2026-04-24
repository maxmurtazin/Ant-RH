#!/usr/bin/env python3
"""
validate_zeros_and_spacing_eta.py

ETA-enabled validator for zeta zeros.

It reuses validate_zeros_and_spacing.py and adds:
  - --progress_every
  - --skip_full_scan
  - --scan_only_candidates
  - ETA for candidate verification
  - ETA for independent Hardy-Z scan
"""
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence

import mpmath as mp

import validate_zeros_and_spacing as base


def format_seconds(x: float) -> str:
    x = max(0.0, float(x))
    if x < 60:
        return f"{x:.1f}s"
    if x < 3600:
        return f"{x / 60:.1f}m"
    return f"{x / 3600:.2f}h"


class ETAState:
    def __init__(self, total: int, label: str, progress_every: int = 500, alpha: float = 0.2):
        self.total = max(1, int(total))
        self.label = label
        self.progress_every = max(1, int(progress_every))
        self.alpha = float(alpha)
        self.start = time.time()
        self.last = self.start
        self.ema_iter: Optional[float] = None

    def tick(self, done: int, extra: str = "") -> None:
        done = max(1, int(done))
        now = time.time()
        dt = now - self.last
        self.last = now
        if self.ema_iter is None:
            self.ema_iter = (now - self.start) / done
        else:
            self.ema_iter = self.alpha * dt + (1.0 - self.alpha) * self.ema_iter

        if done % self.progress_every != 0 and done < self.total:
            return

        elapsed = now - self.start
        remaining = max(0, self.total - done)
        eta = remaining * (self.ema_iter or elapsed / done)
        speed = done / elapsed if elapsed > 0 else 0.0
        msg = (
            f"[{self.label}] {done}/{self.total} "
            f"| elapsed={format_seconds(elapsed)} "
            f"| ETA={format_seconds(eta)} "
            f"| speed={speed:.1f}/s"
        )
        if extra:
            msg += f" | {extra}"
        print(msg, flush=True)


def verify_candidates_eta(candidates: Sequence[float], abs_tol: float, progress_every: int = 100) -> List[base.CandidateCheck]:
    checks: List[base.CandidateCheck] = []
    eta = ETAState(total=len(candidates), label="VERIFY", progress_every=progress_every)
    verified_count = 0
    for i, t in enumerate(candidates, start=1):
        tt = mp.mpf(str(float(t)))
        az = float(base.abs_zeta_critical(tt))
        hz = float(base.hardy_z(tt))
        ok = az <= abs_tol
        if ok:
            verified_count += 1
        checks.append(base.CandidateCheck(t=float(t), abs_zeta=az, hardy_z=hz, verified=ok))
        eta.tick(i, extra=f"verified={verified_count}")
    return checks


def scan_hardy_sign_changes_eta(
    t_min: float,
    t_max: float,
    step: float,
    max_iter: int = 80,
    progress_every: int = 500,
) -> List[float]:
    if step <= 0:
        raise ValueError("scan_step must be positive")

    zeros: List[float] = []
    n = int(math.ceil((t_max - t_min) / step))
    eta = ETAState(total=n, label="SCAN", progress_every=progress_every)

    prev_t = mp.mpf(str(t_min))
    prev_z = base.hardy_z(prev_t)

    for i in range(1, n + 1):
        cur = min(t_min + i * step, t_max)
        cur_t = mp.mpf(str(cur))
        cur_z = base.hardy_z(cur_t)

        found = False
        if prev_z == 0:
            zeros.append(float(prev_t))
            found = True
        elif cur_z == 0:
            zeros.append(float(cur_t))
            found = True
        elif prev_z * cur_z < 0:
            zeros.append(base.bisect_hardy_root(float(prev_t), float(cur_t), max_iter=max_iter))
            found = True

        prev_t, prev_z = cur_t, cur_z
        eta.tick(i, extra=f"zeros={len(zeros)}" + (" found" if found else ""))

    zeros, _ = base.deduplicate(zeros, tol=max(step * 0.25, 1e-10))
    print(f"[SCAN] done | zeros={len(zeros)}", flush=True)
    return zeros


def validate_pipeline_eta(
    cfg: base.ValidationConfig,
    progress_every: int = 500,
    skip_full_scan: bool = False,
    scan_only_candidates: bool = False,
) -> Dict[str, Any]:
    mp.mp.dps = cfg.dps

    raw = base.read_zeros(cfg.input_path)
    if not raw:
        raise ValueError("No zeros found in input file.")

    t_min = cfg.t_min if cfg.t_min is not None else min(raw)
    t_max = cfg.t_max if cfg.t_max is not None else max(raw)
    if t_min >= t_max:
        raise ValueError("t_min must be smaller than t_max")

    filtered = [z for z in raw if t_min <= z <= t_max]
    input_zeros, duplicate_groups = base.deduplicate(filtered, cfg.duplicate_tol)
    print(f"[LOAD] raw={len(raw)} filtered={len(filtered)} dedup={len(input_zeros)} interval=[{t_min}, {t_max}]", flush=True)

    candidate_checks = verify_candidates_eta(
        input_zeros,
        cfg.verify_abs_tol,
        progress_every=max(1, progress_every // 5),
    )

    if skip_full_scan or scan_only_candidates:
        print("[SCAN] skipped full Hardy-Z grid scan; using input zeros as scan reference", flush=True)
        scan_zeros = list(input_zeros)
    else:
        scan_zeros = scan_hardy_sign_changes_eta(
            t_min,
            t_max,
            cfg.scan_step,
            max_iter=cfg.max_bisect_iter,
            progress_every=progress_every,
        )

    matches = base.match_zeros(input_zeros, scan_zeros, cfg.match_tol)
    match_counts: Dict[str, int] = {}
    for m in matches:
        match_counts[m.status] = match_counts.get(m.status, 0) + 1

    spacing_source = scan_zeros if scan_zeros else input_zeros
    spacing = base.spacing_analysis(spacing_source)
    c_report = base.count_report(t_min, t_max, len(scan_zeros))

    report: Dict[str, Any] = {
        "config": asdict(cfg),
        "progress_every": progress_every,
        "skip_full_scan": skip_full_scan,
        "scan_only_candidates": scan_only_candidates,
        "interval": [t_min, t_max],
        "raw_input_count": len(raw),
        "filtered_input_count": len(filtered),
        "input_count": len(input_zeros),
        "scan_count": len(scan_zeros),
        "duplicate_groups": duplicate_groups,
        "candidate_checks": [asdict(x) for x in candidate_checks],
        "scan_zeros": scan_zeros,
        "matches": [asdict(x) for x in matches],
        "match_counts": match_counts,
        "spacing_analysis": spacing,
        "count_report": c_report,
    }

    if cfg.make_plots:
        report["plots"] = base.make_plots(cfg.out_prefix, spacing_source, spacing, matches)
    else:
        report["plots"] = []

    json_path = f"{cfg.out_prefix}_validation.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    base.write_csv(f"{cfg.out_prefix}_candidate_checks.csv", [asdict(x) for x in candidate_checks])
    base.write_csv(f"{cfg.out_prefix}_matches.csv", [asdict(x) for x in matches])

    gap_rows = []
    src = sorted(spacing_source)
    gaps = spacing.get("gaps", [])
    unfolded = spacing.get("unfolded_spacings", [])
    for i, (g, u) in enumerate(zip(gaps, unfolded)):
        gap_rows.append({
            "i": i,
            "t_left": src[i],
            "t_right": src[i + 1],
            "gap": g,
            "unfolded_spacing": u,
        })
    base.write_csv(f"{cfg.out_prefix}_spacing.csv", gap_rows)
    base.write_text_summary(f"{cfg.out_prefix}_summary.md", report)
    return report


def parse_args():
    ap = argparse.ArgumentParser(description="Validate zeta zeros with ETA/progress output.")
    ap.add_argument("--input", required=True, help="Input zeros file: JSON, CSV, or TXT.")
    ap.add_argument("--t_min", type=float, default=None, help="Lower bound of validation interval.")
    ap.add_argument("--t_max", type=float, default=None, help="Upper bound of validation interval.")
    ap.add_argument("--scan_step", type=float, default=0.01, help="Hardy Z sign-change scan step.")
    ap.add_argument("--dps", type=int, default=80, help="mpmath decimal precision.")
    ap.add_argument("--verify_abs_tol", type=float, default=1e-8, help="Candidate |zeta| verification tolerance.")
    ap.add_argument("--match_tol", type=float, default=1e-5, help="Tolerance for matching input zeros to scan zeros.")
    ap.add_argument("--duplicate_tol", type=float, default=1e-7, help="Tolerance for deduplicating input zeros.")
    ap.add_argument("--out", default="zero_validation", help="Output prefix.")
    ap.add_argument("--no_plots", action="store_true", help="Disable PNG plot generation.")
    ap.add_argument("--max_bisect_iter", type=int, default=90, help="Bisection iterations for Hardy Z roots.")
    ap.add_argument("--min_bracket_abs", type=float, default=0.0, help="Reserved for future use.")
    ap.add_argument("--progress_every", type=int, default=500, help="Print ETA every N scan steps.")
    ap.add_argument("--skip_full_scan", action="store_true", help="Skip independent full Hardy-Z scan; validate only candidates.")
    ap.add_argument("--scan_only_candidates", action="store_true", help="Alias: use candidates as scan reference for spacing/matching.")
    args = ap.parse_args()

    cfg = base.ValidationConfig(
        input_path=args.input,
        t_min=args.t_min,
        t_max=args.t_max,
        scan_step=args.scan_step,
        dps=args.dps,
        verify_abs_tol=args.verify_abs_tol,
        match_tol=args.match_tol,
        duplicate_tol=args.duplicate_tol,
        out_prefix=args.out,
        make_plots=not args.no_plots,
        max_bisect_iter=args.max_bisect_iter,
        min_bracket_abs=args.min_bracket_abs,
    )
    return cfg, args.progress_every, args.skip_full_scan, args.scan_only_candidates


def main() -> None:
    cfg, progress_every, skip_full_scan, scan_only_candidates = parse_args()
    report = validate_pipeline_eta(cfg, progress_every, skip_full_scan, scan_only_candidates)
    print("Validation complete.")
    print(f"Input zeros: {report['input_count']}")
    print(f"Independent scan zeros: {report['scan_count']}")
    print(f"Matches: {report['match_counts'].get('matched', 0)}")
    print(f"Missing from input: {report['match_counts'].get('missing_from_input', 0)}")
    print(f"Unmatched input: {report['match_counts'].get('unmatched_input', 0)}")
    print(f"Report: {cfg.out_prefix}_validation.json")
    print(f"Summary: {cfg.out_prefix}_summary.md")


if __name__ == "__main__":
    main()
