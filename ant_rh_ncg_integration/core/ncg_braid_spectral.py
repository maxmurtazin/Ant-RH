"""
NCG + Braid spectral layer for Ant-RH.

Goal:
    Build a small, numerically stable spectral-triple-inspired operator from
    braid/DTES trajectories and train/score it against zeta-zero ordinates.

This is NOT a proof of RH. It is an experimental operator-search module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Dict, Optional
import math
import json
import torch


BraidGen = Tuple[int, int]  # (i, sign), represents sigma_i^{sign}, sign in {-1,+1}
BraidWord = Tuple[BraidGen, ...]


@dataclass(frozen=True)
class BraidNCGConfig:
    n_strands: int = 4
    max_word_len: int = 8
    dim: int = 128
    dtype: str = "float64"
    device: str = "cpu"
    diagonal_growth: str = "exp"  # linear | quadratic | exp
    growth_alpha: float = 0.15
    edge_scale: float = 0.05
    commutator_weight: float = 1e-3
    selfadjoint_weight: float = 1e-2
    zeta_weight: float = 1e-2
    spacing_weight: float = 1e-2
    eps: float = 1e-8

    @property
    def torch_dtype(self):
        return torch.float64 if self.dtype == "float64" else torch.float32


def parse_braid_word(text: str) -> BraidWord:
    """
    Parse strings like: "s1 s2^-1 s3" or "sigma1 sigma2- sigma3+".
    """
    out: List[BraidGen] = []
    for tok in text.replace(",", " ").split():
        tok = tok.strip().lower().replace("sigma", "s")
        if not tok:
            continue
        sign = -1 if ("^-1" in tok or tok.endswith("-") or tok.endswith("-1")) else +1
        digits = "".join(ch for ch in tok if ch.isdigit())
        if not digits:
            raise ValueError(f"Cannot parse braid token: {tok!r}")
        out.append((int(digits), sign))
    return tuple(out)


def braid_word_length(word: BraidWord) -> int:
    return len(word)


def reduce_free_inverse(word: BraidWord) -> BraidWord:
    """Simple cancellation sigma_i sigma_i^{-1} -> e. Does not solve full braid normal form."""
    stack: List[BraidGen] = []
    for g in word:
        if stack and stack[-1][0] == g[0] and stack[-1][1] == -g[1]:
            stack.pop()
        else:
            stack.append(g)
    return tuple(stack)


def enumerate_braid_words(n_strands: int, max_len: int, limit: int) -> List[BraidWord]:
    """
    Deterministic BFS-like enumeration with inverse cancellations removed.
    Useful for smoke tests and finite truncations of l2(B_n).
    """
    gens: List[BraidGen] = []
    for i in range(1, n_strands):
        gens.append((i, +1))
        gens.append((i, -1))

    words: List[BraidWord] = [tuple()]
    frontier: List[BraidWord] = [tuple()]
    seen = {tuple()}

    while frontier and len(words) < limit:
        new_frontier: List[BraidWord] = []
        for w in frontier:
            if len(w) >= max_len:
                continue
            for g in gens:
                nw = reduce_free_inverse(w + (g,))
                if len(nw) <= max_len and nw not in seen:
                    seen.add(nw)
                    words.append(nw)
                    new_frontier.append(nw)
                    if len(words) >= limit:
                        break
            if len(words) >= limit:
                break
        frontier = new_frontier
    return words[:limit]


def length_spectrum(words: Sequence[BraidWord], cfg: BraidNCGConfig) -> torch.Tensor:
    lengths = torch.tensor([max(1, braid_word_length(w)) for w in words], dtype=cfg.torch_dtype, device=cfg.device)
    if cfg.diagonal_growth == "linear":
        return lengths
    if cfg.diagonal_growth == "quadratic":
        return lengths * lengths
    if cfg.diagonal_growth == "exp":
        return torch.exp(cfg.growth_alpha * lengths)
    raise ValueError(f"Unknown diagonal_growth={cfg.diagonal_growth}")


def build_braid_adjacency(words: Sequence[BraidWord], cfg: BraidNCGConfig) -> torch.Tensor:
    """
    Finite truncation adjacency: connect words differing by one generator action.
    Symmetrized to make the final operator self-adjoint.
    """
    idx: Dict[BraidWord, int] = {w: k for k, w in enumerate(words)}
    n = len(words)
    A = torch.zeros((n, n), dtype=cfg.torch_dtype, device=cfg.device)
    gens = [(i, s) for i in range(1, cfg.n_strands) for s in (+1, -1)]
    for w, a in idx.items():
        for g in gens:
            nw = reduce_free_inverse(w + (g,))
            b = idx.get(nw)
            if b is not None and a != b:
                A[a, b] = 1.0
    A = 0.5 * (A + A.T)
    deg = A.sum(dim=1).clamp_min(1.0)
    A = A / torch.sqrt(torch.outer(deg, deg))
    return A


def build_dirac_operator(words: Sequence[BraidWord], cfg: BraidNCGConfig, learnable_edges: Optional[torch.Tensor] = None) -> torch.Tensor:
    diag = length_spectrum(words, cfg)
    A = build_braid_adjacency(words, cfg)
    if learnable_edges is not None:
        E = 0.5 * (learnable_edges + learnable_edges.T)
        A = A * torch.tanh(E)
    D = torch.diag(diag) + cfg.edge_scale * A
    return 0.5 * (D + D.T)


def safe_eigvalsh(H: torch.Tensor, jitter: float = 1e-7) -> torch.Tensor:
    H = 0.5 * (H + H.T)
    eye = torch.eye(H.shape[0], dtype=H.dtype, device=H.device)
    for k in range(6):
        try:
            return torch.linalg.eigvalsh(H + (10**k) * jitter * eye)
        except RuntimeError:
            continue
    return torch.linalg.eigvalsh((H + 1e-2 * eye).to(torch.float64))


def spectral_zeta_from_eigs(eigs: torch.Tensor, s: float, eps: float = 1e-8) -> torch.Tensor:
    vals = eigs.abs().clamp_min(eps)
    return torch.sum(vals.pow(-s))


def heat_trace(eigs: torch.Tensor, t: float) -> torch.Tensor:
    return torch.sum(torch.exp(-t * eigs * eigs))


def spacing_stats(eigs: torch.Tensor, k: int = 64, eps: float = 1e-8) -> Dict[str, float]:
    vals = torch.sort(eigs.detach().float().cpu()).values
    vals = vals[: min(k + 1, len(vals))]
    if len(vals) < 3:
        return {"spacing_mean": 0.0, "spacing_std": 0.0, "r_stat_mean": 0.0}
    sp = torch.diff(vals).abs().clamp_min(eps)
    sp = sp / sp.mean().clamp_min(eps)
    r = torch.minimum(sp[:-1], sp[1:]) / torch.maximum(sp[:-1], sp[1:]).clamp_min(eps)
    return {
        "spacing_mean": float(sp.mean()),
        "spacing_std": float(sp.std(unbiased=False)),
        "r_stat_mean": float(r.mean()),
    }


def zeta_zero_loss(eigs: torch.Tensor, zeros: torch.Tensor, k: int = 64) -> torch.Tensor:
    """
    Match first k positive eigenvalues to zeta zero ordinates after affine normalization.
    This avoids scale-dominating early experiments.
    """
    ev = torch.sort(eigs.abs()).values
    ev = ev[: min(k, len(ev), len(zeros))]
    zz = zeros[: len(ev)].to(ev.device, ev.dtype)
    if len(ev) == 0:
        return torch.tensor(0.0, dtype=eigs.dtype, device=eigs.device)
    evn = (ev - ev.mean()) / ev.std(unbiased=False).clamp_min(1e-8)
    zzn = (zz - zz.mean()) / zz.std(unbiased=False).clamp_min(1e-8)
    return torch.mean((evn - zzn) ** 2)


def selfadjoint_loss(H: torch.Tensor) -> torch.Tensor:
    return torch.mean((H - H.T.conj()).abs() ** 2)


def commutator_loss(D: torch.Tensor, generators: Sequence[torch.Tensor]) -> torch.Tensor:
    if not generators:
        return torch.tensor(0.0, dtype=D.dtype, device=D.device)
    total = torch.tensor(0.0, dtype=D.dtype, device=D.device)
    for A in generators:
        C = D @ A - A @ D
        total = total + torch.mean(C.abs() ** 2)
    return total / len(generators)


def build_left_action_matrices(words: Sequence[BraidWord], cfg: BraidNCGConfig) -> List[torch.Tensor]:
    idx = {w: k for k, w in enumerate(words)}
    mats = []
    for gen in [(i, +1) for i in range(1, cfg.n_strands)]:
        M = torch.zeros((len(words), len(words)), dtype=cfg.torch_dtype, device=cfg.device)
        for w, col in idx.items():
            nw = reduce_free_inverse((gen,) + w)
            row = idx.get(nw)
            if row is not None:
                M[row, col] = 1.0
        mats.append(M)
    return mats


def full_ncg_loss(D: torch.Tensor, zeros: torch.Tensor, cfg: BraidNCGConfig) -> Tuple[torch.Tensor, Dict[str, float]]:
    eigs = safe_eigvalsh(D)
    l_spec = zeta_zero_loss(eigs, zeros, k=min(64, len(eigs), len(zeros)))
    l_self = selfadjoint_loss(D)
    # Use heat-trace stability as a soft zeta/partition regularizer.
    ht = heat_trace(eigs, t=0.01)
    l_zeta = torch.log1p(ht.abs())
    loss = l_spec + cfg.selfadjoint_weight * l_self + cfg.zeta_weight * l_zeta
    stats = spacing_stats(eigs)
    stats.update({
        "loss": float(loss.detach().cpu()),
        "spectral_loss": float(l_spec.detach().cpu()),
        "selfadjoint_loss": float(l_self.detach().cpu()),
        "heat_trace_log": float(l_zeta.detach().cpu()),
        "eig_min": float(eigs.min().detach().cpu()),
        "eig_max": float(eigs.max().detach().cpu()),
    })
    return loss, stats


def load_zeros(path: Optional[str], cfg: BraidNCGConfig, count: int = 128) -> torch.Tensor:
    """Load zeta ordinates from txt/csv/json, or fallback to first known values."""
    fallback = [
        14.134725141734693, 21.022039638771555, 25.010857580145688,
        30.424876125859513, 32.935061587739190, 37.586178158825671,
        40.918719012147495, 43.327073280914999, 48.005150881167159,
        49.773832477672302, 52.970321477714460, 56.446247697063394,
        59.347044002602353, 60.831778524609809, 65.112544048081651,
        67.079810529494173, 69.546401711173979, 72.067157674481907,
        75.704690699083933, 77.144840068874805, 79.337375020249367,
        82.910380854086030, 84.735492980517050, 87.425274613125229,
    ]
    vals: List[float] = []
    if path:
        text = open(path, "r", encoding="utf-8").read().strip()
        if path.endswith(".json"):
            data = json.loads(text)
            if isinstance(data, dict):
                data = data.get("zeros") or data.get("ordinates") or data.get("gamma") or []
            vals = [float(x) for x in data]
        else:
            for line in text.splitlines():
                for part in line.replace(",", " ").split():
                    try:
                        vals.append(float(part))
                    except ValueError:
                        pass
    if not vals:
        vals = fallback
    while len(vals) < count:
        # crude continuation for smoke tests only
        vals.append(vals[-1] + math.log(max(vals[-1], 10.0)) * 2.0)
    return torch.tensor(vals[:count], dtype=cfg.torch_dtype, device=cfg.device)
