#!/usr/bin/env python3
from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np


def safe_sinh(x: np.ndarray | float) -> np.ndarray | float:
    xa = np.asarray(x, dtype=np.float64)
    out = np.where(xa < 20.0, np.sinh(xa), 0.5 * np.exp(xa))
    return float(out) if np.ndim(x) == 0 else out


def _safe_inv_2sinh(x: np.ndarray) -> np.ndarray:
    """
    Compute 1 / (2*sinh(x)) without overflow:
      if x < 20: 1/(2*sinh(x))
      else:      exp(-x)  (since 2*sinh(x) ~ exp(x))
    """
    x = np.asarray(x, dtype=np.float64)
    small = x < 20.0
    out = np.empty_like(x)
    out[small] = 1.0 / (2.0 * np.sinh(x[small]))
    out[~small] = np.exp(-x[~small])
    return out


def h(r: np.ndarray, r0: np.ndarray, sigma: float) -> np.ndarray:
    r = np.asarray(r, dtype=np.float64)
    r0 = np.asarray(r0, dtype=np.float64)
    return np.exp(-(sigma * sigma) * (r - r0) * (r - r0))


def g(l: np.ndarray, r0: np.ndarray, sigma: float) -> np.ndarray:
    l = np.asarray(l, dtype=np.float64)
    r0 = np.asarray(r0, dtype=np.float64)
    return np.exp(-(l * l) / (4.0 * sigma * sigma)) * np.cos(r0 * l)


def _load_lengths_csv(path: str | Path) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"lengths file not found: {p}")
    arr = np.loadtxt(str(path), delimiter=",", skiprows=1, usecols=(0,), dtype=np.float64)
    arr = np.asarray(arr, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr > 0.0]
    return arr


def _load_zeros(path: str | Path) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"zeros file not found: {p}")
    z = np.loadtxt(str(path), dtype=np.float64)
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    z = z[np.isfinite(z)]
    return z


def _chunk_iter(n: int, chunk_size: int) -> Iterable[Tuple[int, int]]:
    i = 0
    while i < n:
        j = min(n, i + chunk_size)
        yield i, j
        i = j


def selberg_sums(
    lengths: np.ndarray,
    zeros: np.ndarray,
    sigma: float,
    m_max: int,
    r0_values: np.ndarray,
    chunk_size: int = 1000,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (S_spec[r0], S_geo[r0]) for each r0 in r0_values.
    """
    lengths = np.asarray(lengths, dtype=np.float64).reshape(-1)
    zeros = np.asarray(zeros, dtype=np.float64).reshape(-1)
    r0_values = np.asarray(r0_values, dtype=np.float64).reshape(-1)

    if lengths.size == 0 or zeros.size == 0 or r0_values.size == 0:
        z = np.zeros((r0_values.size,), dtype=np.float64)
        return z, z

    # Spectral sum: S_spec(r0) = sum_n exp(-sigma^2 (r_n - r0)^2)
    # Broadcasting: (N,1) - (1,R)
    S_spec = np.sum(h(zeros[:, None], r0_values[None, :], sigma), axis=0)

    # Geometric sum: chunk lengths if large
    m = np.arange(1, int(m_max) + 1, dtype=np.float64)[:, None]  # (M,1)
    S_geo = np.zeros((r0_values.size,), dtype=np.float64)

    for i, j in _chunk_iter(lengths.size, chunk_size):
        ell = lengths[i:j][None, :]  # (1,Nc)
        ml = m * ell  # (M,Nc)

        inv_2sinh = _safe_inv_2sinh(ml * 0.5)  # (M,Nc)
        amp = (ell * inv_2sinh) * np.exp(-(ml * ml) / (4.0 * sigma * sigma))  # (M,Nc)

        # cos(r0 * ml): (R,1,1) * (1,M,Nc) => (R,M,Nc)
        cos_term = np.cos(r0_values[:, None, None] * ml[None, :, :])
        S_geo += np.sum(amp[None, :, :] * cos_term, axis=(1, 2))

    return S_spec, S_geo


def compute_selberg_loss(lengths: np.ndarray, zeros: np.ndarray, sigma: float = 0.5, m_max: int = 6) -> float:
    zeros = np.asarray(zeros, dtype=np.float64).reshape(-1)
    if zeros.size == 0:
        return float("nan")
    r0 = float(np.median(zeros))
    S_spec, S_geo = selberg_sums(lengths, zeros, sigma=sigma, m_max=m_max, r0_values=np.array([r0]))
    diff = float(S_spec[0] - S_geo[0])
    return diff * diff


def main() -> None:
    p = argparse.ArgumentParser(description="Selberg trace loss from Artin geodesic lengths")
    p.add_argument("--lengths", type=str, required=True)
    p.add_argument("--zeros", type=str, required=True)
    p.add_argument("--sigma", type=float, default=0.5)
    p.add_argument("--m_max", type=int, default=6)
    p.add_argument("--out_dir", type=str, default="runs/")
    args = p.parse_args()

    lengths = _load_lengths_csv(args.lengths)
    zeros = _load_zeros(args.zeros)
    sigma = float(args.sigma)
    m_max = int(args.m_max)

    # Curve evaluated on r0 = zeros
    r0_values = zeros.copy()
    S_spec_curve, S_geo_curve = selberg_sums(
        lengths=lengths,
        zeros=zeros,
        sigma=sigma,
        m_max=m_max,
        r0_values=r0_values,
        chunk_size=1000 if lengths.size > 10_000 else max(1000, lengths.size),
    )

    diff_curve = S_spec_curve - S_geo_curve
    loss_curve = diff_curve * diff_curve
    rel_curve = np.abs(diff_curve) / (np.abs(S_spec_curve) + 1e-8)

    # Report at r0 = median(zeros)
    r0_report = float(np.median(zeros)) if zeros.size else 0.0
    S_spec_rep, S_geo_rep = selberg_sums(
        lengths=lengths,
        zeros=zeros,
        sigma=sigma,
        m_max=m_max,
        r0_values=np.array([r0_report], dtype=np.float64),
        chunk_size=1000 if lengths.size > 10_000 else max(1000, lengths.size),
    )
    S_spec = float(S_spec_rep[0]) if S_spec_rep.size else 0.0
    S_geo = float(S_geo_rep[0]) if S_geo_rep.size else 0.0
    diff = S_spec - S_geo
    loss = float(diff * diff)
    relative_error = float(abs(diff) / (abs(S_spec) + 1e-8))

    print(f"spectral sum: {S_spec:.12g}", flush=True)
    print(f"geometric sum: {S_geo:.12g}", flush=True)
    print(f"loss: {loss:.12g}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "S_spec": S_spec,
        "S_geo": S_geo,
        "loss": loss,
        "relative_error": relative_error,
    }
    with open(out_dir / "selberg_trace_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with open(out_dir / "trace_loss_curve.csv", "w", encoding="utf-8") as f:
        f.write("r0,S_spec,S_geo,loss,relative_error\n")
        for r0, ss, sg, ll, rr in zip(r0_values, S_spec_curve, S_geo_curve, loss_curve, rel_curve):
            f.write(f"{float(r0)},{float(ss)},{float(sg)},{float(ll)},{float(rr)}\n")


if __name__ == "__main__":
    main()

