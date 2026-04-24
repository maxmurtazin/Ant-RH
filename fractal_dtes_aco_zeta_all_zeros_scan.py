#!/usr/bin/env python3
"""
Find ALL Riemann zeta zeros on the critical line in a given t-range.

This file is meant as a robust companion to the Fractal-DTES-ACO-Zeta pack.
Instead of returning only the best candidate, it scans the full interval and
returns every bracketed zero it can verify.

Method:
  1. Work on the critical line s = 1/2 + i t.
  2. Use the real Hardy Z-function Z(t), whose real zeros correspond to
     zeta(1/2 + i t) zeros on the critical line.
  3. Scan [t_min, t_max] on a grid and find sign changes of Z(t).
  4. Refine each bracket by bisection.
  5. Verify by |zeta(1/2 + i t)| and save JSON/CSV.

Dependencies:
  pip install mpmath numpy

Examples:
  python3 fractal_dtes_aco_zeta_all_zeros_scan.py --t_min 100 --t_max 400 --step 0.02
  python3 fractal_dtes_aco_zeta_all_zeros_scan.py --t_min 20 --t_max 30 --step 0.005 --dps 80
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import mpmath as mp


@dataclass
class ZeroRecord:
    index_in_run: int
    t: float
    abs_zeta: float
    hardy_z: float
    bracket_left: float
    bracket_right: float
    bracket_width: float
    iterations: int
    method: str


@dataclass
class ScanConfig:
    t_min: float
    t_max: float
    step: float = 0.02
    dps: int = 60
    tol_t: float = 1e-12
    verify_abs_zeta: float = 1e-8
    zero_value_tol: float = 1e-30
    max_bisect_iter: int = 120
    output_prefix: str = "zeta_zeros_scan"
    progress_every: int = 500
    refine_midpoint_minima: bool = True


def hardy_z(t: float) -> mp.mpf:
    """Hardy Z(t), real-valued on real t.

    mpmath exposes siegelz(t). If not available in a local version, this falls
    back to a Riemann-Siegel theta expression.
    """
    tt = mp.mpf(str(t))
    if hasattr(mp, "siegelz"):
        return mp.re(mp.siegelz(tt))
    # fallback: Z(t) = exp(i theta(t)) zeta(1/2 + i t), real on real t
    theta = mp.siegeltheta(tt)
    z = mp.e ** (1j * theta) * mp.zeta(mp.mpf("0.5") + 1j * tt)
    return mp.re(z)


def zeta_abs_on_critical(t: float) -> mp.mpf:
    tt = mp.mpf(str(t))
    return abs(mp.zeta(mp.mpf("0.5") + 1j * tt))


def sign_of(x: mp.mpf, zero_tol: mp.mpf) -> int:
    if abs(x) <= zero_tol:
        return 0
    return 1 if x > 0 else -1


def frange_inclusive(a: float, b: float, step: float) -> Iterable[float]:
    n = int(math.floor((b - a) / step))
    for k in range(n + 1):
        yield a + k * step
    last = a + (n + 1) * step
    if last < b + 0.5 * step:
        yield b


def bisect_zero(left: float, right: float, cfg: ScanConfig) -> Tuple[float, int]:
    """Refine a sign-change bracket for Hardy Z(t)."""
    zl = hardy_z(left)
    zr = hardy_z(right)
    zero_tol = mp.mpf(str(cfg.zero_value_tol))

    if abs(zl) <= zero_tol:
        return left, 0
    if abs(zr) <= zero_tol:
        return right, 0
    if zl * zr > 0:
        # The caller should avoid this, but keep a safe fallback.
        mid = 0.5 * (left + right)
        return mid, 0

    lo, hi = float(left), float(right)
    flo, fhi = zl, zr
    for it in range(1, cfg.max_bisect_iter + 1):
        mid = 0.5 * (lo + hi)
        fm = hardy_z(mid)
        if abs(fm) <= zero_tol or abs(hi - lo) <= cfg.tol_t:
            return mid, it
        if flo * fm <= 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return 0.5 * (lo + hi), cfg.max_bisect_iter


def local_minimum_candidates(samples: List[Tuple[float, mp.mpf]], cfg: ScanConfig) -> List[Tuple[float, float]]:
    """Optional safety net for near-tangent/missed sign-change regions.

    Zeta zeros on the critical line should be simple in tested ranges, so sign
    changes are the main mechanism. This extra pass just flags local |Z| minima
    below a threshold and creates a tiny local bracket around them.
    """
    out: List[Tuple[float, float]] = []
    if len(samples) < 3:
        return out
    threshold = mp.mpf(str(max(cfg.verify_abs_zeta * 10, 1e-10)))
    for i in range(1, len(samples) - 1):
        t, z = samples[i]
        az = abs(z)
        if az <= abs(samples[i - 1][1]) and az <= abs(samples[i + 1][1]) and az < threshold:
            out.append((samples[i - 1][0], samples[i + 1][0]))
    return out


def merge_brackets(brackets: List[Tuple[float, float]], tol: float) -> List[Tuple[float, float]]:
    if not brackets:
        return []
    brackets = sorted((min(a, b), max(a, b)) for a, b in brackets)
    merged = [brackets[0]]
    for a, b in brackets[1:]:
        pa, pb = merged[-1]
        if a <= pb + tol:
            merged[-1] = (pa, max(pb, b))
        else:
            merged.append((a, b))
    return merged


def scan_all_zeros(cfg: ScanConfig) -> List[ZeroRecord]:
    if cfg.t_max <= cfg.t_min:
        raise ValueError("t_max must be greater than t_min")
    if cfg.step <= 0:
        raise ValueError("step must be positive")

    mp.mp.dps = cfg.dps
    zero_tol = mp.mpf(str(cfg.zero_value_tol))

    start = time.time()
    samples: List[Tuple[float, mp.mpf]] = []
    brackets: List[Tuple[float, float]] = []

    prev_t: Optional[float] = None
    prev_z: Optional[mp.mpf] = None
    prev_s: Optional[int] = None

    points = list(frange_inclusive(cfg.t_min, cfg.t_max, cfg.step))
    total = len(points)

    for i, t in enumerate(points, start=1):
        z = hardy_z(t)
        s = sign_of(z, zero_tol)
        samples.append((t, z))

        if prev_t is not None and prev_z is not None and prev_s is not None:
            if s == 0:
                brackets.append((max(cfg.t_min, t - cfg.step), min(cfg.t_max, t + cfg.step)))
            elif prev_s == 0:
                brackets.append((max(cfg.t_min, prev_t - cfg.step), min(cfg.t_max, t)))
            elif s != prev_s:
                brackets.append((prev_t, t))

        prev_t, prev_z, prev_s = t, z, s

        if cfg.progress_every > 0 and (i % cfg.progress_every == 0 or i == total):
            elapsed = time.time() - start
            rate = i / max(elapsed, 1e-9)
            eta = (total - i) / max(rate, 1e-9)
            print(
                f"[SCAN] {i}/{total} | brackets={len(brackets)} | "
                f"elapsed={elapsed:.1f}s | ETA={eta:.1f}s",
                flush=True,
            )

    if cfg.refine_midpoint_minima:
        brackets.extend(local_minimum_candidates(samples, cfg))

    brackets = merge_brackets(brackets, tol=max(cfg.step * 1.5, cfg.tol_t * 10))

    records: List[ZeroRecord] = []
    seen: List[float] = []
    for left, right in brackets:
        t_star, n_iter = bisect_zero(left, right, cfg)
        if not (cfg.t_min <= t_star <= cfg.t_max):
            continue
        # Deduplicate roots from overlapping brackets.
        if any(abs(t_star - old) <= max(cfg.tol_t * 20, cfg.step * 0.25) for old in seen):
            continue
        abs_z = float(zeta_abs_on_critical(t_star))
        hz = float(hardy_z(t_star))
        if abs_z <= cfg.verify_abs_zeta:
            seen.append(t_star)
            records.append(
                ZeroRecord(
                    index_in_run=len(records) + 1,
                    t=float(t_star),
                    abs_zeta=abs_z,
                    hardy_z=hz,
                    bracket_left=float(left),
                    bracket_right=float(right),
                    bracket_width=float(abs(right - left)),
                    iterations=int(n_iter),
                    method="hardy_z_sign_scan_bisection",
                )
            )

    records.sort(key=lambda r: r.t)
    for i, r in enumerate(records, start=1):
        r.index_in_run = i
    return records


def save_outputs(records: List[ZeroRecord], cfg: ScanConfig) -> Dict[str, str]:
    prefix = Path(cfg.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True) if prefix.parent != Path('.') else None

    json_path = prefix.with_suffix(".json")
    csv_path = prefix.with_suffix(".csv")
    txt_path = prefix.with_suffix(".txt")

    payload = {
        "config": asdict(cfg),
        "count": len(records),
        "zeros": [asdict(r) for r in records],
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "index_in_run", "t", "abs_zeta", "hardy_z", "bracket_left",
                "bracket_right", "bracket_width", "iterations", "method",
            ],
        )
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))

    lines = [f"Found {len(records)} zeros in [{cfg.t_min}, {cfg.t_max}]", ""]
    for r in records:
        lines.append(f"{r.index_in_run:04d}  t={r.t:.15f}  |zeta|={r.abs_zeta:.3e}")
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {"json": str(json_path), "csv": str(csv_path), "txt": str(txt_path)}


def parse_args() -> ScanConfig:
    p = argparse.ArgumentParser(description="Find all zeta zeros on the critical line in a t-range.")
    p.add_argument("--t_min", "--t-min", dest="t_min", type=float, required=True)
    p.add_argument("--t_max", "--t-max", dest="t_max", type=float, required=True)
    p.add_argument("--step", type=float, default=0.02, help="Scan step in t. Smaller is safer but slower.")
    p.add_argument("--dps", type=int, default=60, help="mpmath decimal precision")
    p.add_argument("--tol_t", "--tol-t", dest="tol_t", type=float, default=1e-12)
    p.add_argument("--verify_abs_zeta", "--verify-abs-zeta", dest="verify_abs_zeta", type=float, default=1e-8)
    p.add_argument("--output", "-o", dest="output_prefix", default="zeta_zeros_scan")
    p.add_argument("--progress_every", "--progress-every", dest="progress_every", type=int, default=500)
    p.add_argument("--no_minima_safety", "--no-minima-safety", dest="refine_midpoint_minima", action="store_false")
    ns = p.parse_args()
    return ScanConfig(**vars(ns))


def main() -> None:
    cfg = parse_args()
    print(f"[START] scanning all zeros in [{cfg.t_min}, {cfg.t_max}] step={cfg.step} dps={cfg.dps}")
    t0 = time.time()
    records = scan_all_zeros(cfg)
    out = save_outputs(records, cfg)
    elapsed = time.time() - t0
    print(f"[DONE] found {len(records)} zeros | elapsed={elapsed:.2f}s")
    for r in records:
        print(f"t = {r.t:.15f} |zeta| = {r.abs_zeta:.3e}")
    print("Outputs:")
    for k, v in out.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
