"""
Ramsey-style word scoring and a finite-dimensional Nijenhuis-torsion proxy (V13; not an RH proof).
"""

from __future__ import annotations

import math
from collections import Counter
from typing import List, Optional

import torch


def ramsey_score_word(word: List[int], min_block: int = 2) -> float:
    """
    Reward repeated homogeneous subpatterns in a braid/DTES word.
    Score is finite, normalized roughly in [0, 1].
    """
    mb = max(2, int(min_block))
    if not word:
        return 0.0
    w = [int(x) for x in word]
    n = len(w)
    # Homogeneous runs of length >= mb
    score_runs = 0.0
    i = 0
    while i < n:
        j = i + 1
        while j < n and w[j] == w[i]:
            j += 1
        L = j - i
        if L >= mb:
            score_runs += float(L - mb + 1) / float(max(n, mb))
        i = j
    score_runs = min(1.0, score_runs)
    # Bigram repetition (fewer unique bigrams → more structure)
    if n >= 2:
        pairs = [(w[k], w[k + 1]) for k in range(n - 1)]
        nu = len(set(pairs))
        rep = 1.0 - float(nu) / float(max(1, len(pairs)))
    else:
        rep = 0.0
    rep = max(0.0, min(1.0, rep))
    # Normalized entropy of |generator| histogram (balanced reuse)
    modes = [abs(x) for x in w]
    ctr = Counter(modes)
    if len(ctr) <= 1:
        ent_n = 0.0
    else:
        ent = 0.0
        for v in ctr.values():
            p = float(v) / float(n)
            ent -= p * math.log(p + 1e-15)
        ent_n = float(ent) / math.log(float(len(ctr)))
    ent_n = max(0.0, min(1.0, ent_n))
    out = 0.45 * score_runs + 0.28 * rep + 0.27 * ent_n
    return float(max(0.0, min(1.0, out)))


def _circ_shift_matrix(
    dim: int,
    k: int,
    *,
    forward: bool,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Orthogonal circular shift on columns: M @ e_j = e_{(j±k)%dim}."""
    d = int(dim)
    kk = int(k) % d
    if kk == 0:
        kk = 1
    idx = torch.arange(d, device=device, dtype=torch.long)
    if forward:
        src = (idx + kk) % d
    else:
        src = (idx - kk) % d
    P = torch.zeros((d, d), dtype=dtype, device=device)
    P[idx, src] = 1.0
    return P


def word_to_shift_operator(word: List[int], dim: int, device: str = "cpu") -> torch.Tensor:
    """
    Build finite-dimensional operator N_w from word via products of signed cyclic shifts.
    Returns shape [dim, dim].
    """
    d = max(1, int(dim))
    dev = torch.device(device)
    dtype = torch.float64
    I = torch.eye(d, dtype=dtype, device=dev)
    if not word:
        return I
    N = I.clone()
    for t, g in enumerate(word):
        gi = int(g)
        kk = abs(gi) % d
        if kk == 0:
            kk = max(1, d // 2)
        forward = gi >= 0
        P = _circ_shift_matrix(d, kk, forward=forward, dtype=dtype, device=dev)
        # Tiny skew tail so composed operators are not pure permutations; torsion proxy is finite-safe
        # and typically nonzero (V13 diagnostic; not an RH proof).
        eps = 0.04 + 0.06 * float((abs(gi) + t) % 11) / 11.0
        S = torch.zeros((d, d), dtype=dtype, device=dev)
        if d > 1:
            a = int(abs(gi) + t) % d
            b = (a + 1 + (abs(gi) % max(1, d - 1))) % d
            if a != b:
                S[a, b] = 1.0
                S[b, a] = -1.0
        Q = I + eps * S
        N = N @ P @ Q
    if not torch.isfinite(N).all():
        N = torch.nan_to_num(N, nan=0.0, posinf=0.0, neginf=0.0)
    return N


def _comm(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return A @ B - B @ A


def _torsion_N(N: torch.Tensor, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """T_N(X,Y) = [NX,NY] - N([NX,Y]+[X,NY]) + N^2[X,Y]."""
    NX = N @ X
    NY = N @ Y
    t1 = _comm(NX, NY)
    inner = _comm(NX, Y) + _comm(X, NY)
    t2 = N @ inner
    XY = _comm(X, Y)
    t3 = (N @ N) @ XY
    return t1 - t2 + t3


def nijenhuis_defect(
    N: torch.Tensor,
    probes: Optional[List[torch.Tensor]] = None,
) -> torch.Tensor:
    """
    Mean Frobenius norm of the Nijenhuis torsion ``T_N`` on probe pairs, plus a small
    auxiliary commutator term ``||[N,P]||_F`` on the same probes. For products of
    nearly-normal shifts, ``T_N`` often cancels in float64; the commutator term keeps
    the defect finite-safe and discriminative across words (V13 diagnostic; not an RH proof).
    """
    d = int(N.shape[0])
    dev = N.device
    dt = N.dtype
    singles: List[torch.Tensor] = []
    if probes is None:
        I = torch.eye(d, dtype=dt, device=dev)
        g = torch.Generator(device=dev)
        g.manual_seed(9001 + d)
        X1 = torch.randn((d, d), generator=g, dtype=dt, device=dev) * 0.25
        Y1 = torch.randn((d, d), generator=g, dtype=dt, device=dev) * 0.25
        X2 = I + 0.15 * torch.randn((d, d), generator=g, dtype=dt, device=dev)
        Y2 = 0.5 * I + 0.12 * torch.randn((d, d), generator=g, dtype=dt, device=dev)
        probe_pairs = [(X1, Y1), (X2, Y2)]
        singles = [X1, Y1, X2, Y2]
    else:
        probe_pairs = []
        it = iter(probes)
        for X in it:
            try:
                Y = next(it)
            except StopIteration:
                break
            probe_pairs.append((X, Y))
            singles.append(X)
            singles.append(Y)
    tor_vals: List[torch.Tensor] = []
    for X, Y in probe_pairs:
        if X.shape != N.shape or Y.shape != N.shape:
            continue
        T = _torsion_N(N, X, Y)
        if not torch.isfinite(T).all():
            T = torch.nan_to_num(T, nan=0.0, posinf=0.0, neginf=0.0)
        tor_vals.append(torch.linalg.matrix_norm(T, ord="fro"))
    com_vals: List[torch.Tensor] = []
    for P in singles:
        if P.shape != N.shape:
            continue
        C = _comm(N, P)
        if not torch.isfinite(C).all():
            C = torch.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
        com_vals.append(torch.linalg.matrix_norm(C, ord="fro"))
    parts: List[torch.Tensor] = []
    if tor_vals:
        parts.append(torch.stack(tor_vals).mean())
    if com_vals:
        parts.append(torch.stack(com_vals).mean())
    if not parts:
        return torch.zeros((), dtype=dt, device=dev)
    m = torch.stack(parts).sum()
    if not torch.isfinite(m):
        m = torch.zeros((), dtype=dt, device=dev)
    return m


def ramsey_nijenhuis_loss(
    word: List[int],
    dim: int,
    lambda_ramsey: float = 0.0,
    lambda_nijenhuis: float = 0.0,
    device: str = "cpu",
    *,
    min_block: int = 2,
) -> dict:
    """
    Return dict with ramsey_score, nijenhuis_defect, and combined loss:
    loss = lambda_nijenhuis * defect - lambda_ramsey * score
    """
    lr = float(lambda_ramsey)
    ln = float(lambda_nijenhuis)
    r = ramsey_score_word(list(word), min_block=int(min_block))
    if ln == 0.0:
        d = 0.0
    else:
        N = word_to_shift_operator(list(word), int(dim), device=str(device))
        d = float(nijenhuis_defect(N).detach().cpu().item())
        if not math.isfinite(d):
            d = 0.0
    if not math.isfinite(r):
        r = 0.0
    loss = ln * d - lr * r
    if not math.isfinite(loss):
        loss = 0.0
    return {"ramsey_score": float(r), "nijenhuis_defect": float(d), "loss": float(loss)}
