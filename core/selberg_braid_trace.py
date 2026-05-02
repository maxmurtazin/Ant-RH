"""
Selberg-style trace alignment for braid-word candidates (experimental).

Relates a geometric side built from primitive braid cycles (with weights
inspired by prime-length orbit terms) to a spectral side built from
eigenvalues of an associated operator. This is not a proof of the
Riemann Hypothesis — only a numerical consistency hook for Ant-RH V13.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

import torch

from core.ncg_braid_spectral import BraidWord, reduce_free_inverse, safe_eigvalsh


def _primitive_period_word(w: BraidWord) -> BraidWord:
    """Smallest repeating unit if w = chunk^k (cyclic repetition)."""
    L = len(w)
    if L == 0:
        return w
    for d in range(1, L + 1):
        if L % d != 0:
            continue
        chunk = w[:d]
        ok = True
        for i in range(L):
            if w[i] != chunk[i % d]:
                ok = False
                break
        if ok:
            return chunk
    return w


def collect_primitive_braid_cycles(word: BraidWord, max_cycles: int = 16) -> List[BraidWord]:
    """
    Collect cyclic rotations of the primitive root of the freely reduced word.

    Each distinct rotation is treated as a primitive braid cycle representative
    for the trace-formula-style geometric side (finite truncation heuristic).
    """
    w = reduce_free_inverse(word)
    if not w:
        return []
    root = _primitive_period_word(w)
    L = len(root)
    seen = set()
    out: List[BraidWord] = []
    for s in range(L):
        rot = root[s:] + root[:s]
        if rot in seen:
            continue
        seen.add(rot)
        out.append(rot)
        if len(out) >= max_cycles:
            break
    return out


def combinatorial_cycle_length(beta: BraidWord) -> float:
    """Positive braid-letter complexity proxy for ell(beta)."""
    if not beta:
        return 0.0
    return float(sum(max(1, int(g[0])) for g in beta))


def match_prime_orbit(
    ell_beta: float,
    *,
    primes: Tuple[int, ...] = (2, 3, 5, 7, 11, 13, 17, 19),
    k_max: int = 6,
) -> Tuple[int, int]:
    """Pick (p, k) so k log p is closest to ell_beta (Selberg-style length bookkeeping)."""
    best_p, best_k = 2, 1
    best_d = float("inf")
    for p in primes:
        if p < 2:
            continue
        lp = math.log(float(p))
        for k in range(1, int(k_max) + 1):
            t = float(k) * lp
            d = abs(float(ell_beta) - t)
            if d < best_d:
                best_d = d
                best_p, best_k = int(p), int(k)
    return best_p, best_k


def selberg_cycle_weight(p: int, k: int) -> float:
    """Approximate |log(p)| / p^(k/2) orbit weight (finite model)."""
    if p < 2 or k < 1:
        return 0.0
    return float(math.log(float(p)) / (float(p) ** (float(k) / 2.0)))


def geometric_selberg_side(
    ell_betas: List[float],
    *,
    sigma: float,
    target_length: float,
) -> float:
    """
    Geometric side: sum_beta A_beta * exp(-(ell_beta - target_length)^2 / (2 sigma^2)).
    A_beta uses prime-orbit weights matched to each ell_beta.
    """
    if not ell_betas:
        return 0.0
    sig = max(float(sigma), 1e-8)
    total = 0.0
    for ell_beta in ell_betas:
        p, k = match_prime_orbit(float(ell_beta))
        a_beta = selberg_cycle_weight(p, k)
        diff = float(ell_beta) - float(target_length)
        total += a_beta * math.exp(-(diff * diff) / (2.0 * sig * sig))
    return float(total)


def spectral_selberg_side(
    eigs: torch.Tensor,
    *,
    sigma: float,
    target_length: float,
    scale_to_target: bool = True,
) -> float:
    """
    Spectral side: sum_i h(lambda_i) with Gaussian test function aligned to target_length.

    Eigenvalues are affine-mapped into the same length scale as braid lengths when
    scale_to_target is True (experimental calibration).
    """
    vals = torch.sort(eigs.reshape(-1)).values.detach().cpu().float().numpy()
    vals = vals[::-1]
    if vals.size == 0:
        return 0.0
    vmin, vmax = float(vals.min()), float(vals.max())
    sig = max(float(sigma), 1e-8)
    tgt = float(target_length)
    if scale_to_target and abs(vmax - vmin) > 1e-12 and abs(tgt) > 1e-12:
        scale = abs(tgt) / max(abs(vmax - vmin), 1e-12)
        mid = 0.5 * (vmin + vmax)
        vals = (vals - mid) * scale + tgt
    s = 0.0
    for lam in vals.reshape(-1):
        d = float(lam) - tgt
        s += math.exp(-(d * d) / (2.0 * sig * sig))
    return float(s)


def compute_selberg_braid_trace_metrics(
    H: torch.Tensor,
    braid_word: BraidWord,
    hyperbolic_length: float,
    *,
    sigma: float = 0.75,
    max_cycles: int = 16,
) -> Dict[str, float]:
    """
    Selberg-style trace consistency between braid cycles andspectrum(H).

    Returns trace_loss, spectral_side, geometric_side, primitive_braid_count,
    mean_braid_length.
    """
    cycles = collect_primitive_braid_cycles(braid_word, max_cycles=int(max_cycles))
    if not cycles:
        return {
            "trace_loss": 0.0,
            "spectral_side": 0.0,
            "geometric_side": 0.0,
            "primitive_braid_count": 0.0,
            "mean_braid_length": 0.0,
        }

    combs = [combinatorial_cycle_length(b) for b in cycles]
    total_comb = float(sum(combs)) if combs else 1.0
    ell_hyp = float(hyperbolic_length) if math.isfinite(float(hyperbolic_length)) else float(sum(combs))
    ell_hyp = max(ell_hyp, 1e-8)
    ell_betas = [max(1e-8, c * (ell_hyp / max(total_comb, 1e-8))) for c in combs]

    mean_ell = float(sum(ell_betas) / max(len(ell_betas), 1))
    g_side = geometric_selberg_side(ell_betas, sigma=sigma, target_length=mean_ell)

    eigs = safe_eigvalsh(H)
    sp_side = spectral_selberg_side(eigs, sigma=sigma, target_length=mean_ell, scale_to_target=True)

    diff = float(sp_side) - float(g_side)
    trace_loss = diff * diff

    return {
        "trace_loss": float(trace_loss),
        "spectral_side": float(sp_side),
        "geometric_side": float(g_side),
        "primitive_braid_count": float(len(cycles)),
        "mean_braid_length": float(mean_ell),
    }
