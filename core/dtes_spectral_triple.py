"""
DTES / Artin spectral-triple style abstraction (finite-dimensional).

Represents candidate operators in an NCG-inspired spectral triple (A, H, D):

  - A: finite-dimensional algebra probe built from braid / Artin word generators
       (``algebra_probe_from_word``).
  - H: finite-dimensional Hilbert space (implicitly C^n where n = matrix dimension).
  - D: Dirac-like operator from the learned matrix H via symmetrization
       (``dirac_from_operator``).

This is a **finite-dimensional NCG-inspired regularizer** used to discourage
operator / geometric collapse during ACO search. It is **not** a proof of the
Riemann Hypothesis.
"""

from __future__ import annotations

from typing import Any, List, Sequence, Union

import torch


class DTESSpectralTriple:
    """Finite-dimensional packaging for (algebra probe A, Hilbert space dimension, Dirac D)."""

    def __init__(self, dim: int, device: str = "cpu", *, dtype: str = "float64"):
        self.dim = int(dim)
        self.device = torch.device(device)
        self.dtype = torch.float64 if str(dtype).lower() in ("float64", "double") else torch.float32

    def algebra_probe_from_word(self, word: Union[List[int], Sequence[int]]) -> torch.Tensor:
        """
        Build a finite-dimensional algebra element A_w from a braid / DTES word.

        Uses a diagonal + symmetric tridiagonal band: normalized positions and generator
        indices discourage trivial commutative alignment with arbitrary D.
        """
        n = self.dim
        A = torch.zeros(n, n, dtype=self.dtype, device=self.device)
        if n <= 0:
            return A
        w = [int(x) for x in word]
        if not w:
            A[0, 0] = torch.tensor(1.0, dtype=self.dtype, device=self.device)
            return A

        L = len(w)
        for i, g in enumerate(w):
            j = i % n
            pos = float(i + 1) / float(max(L, 1))
            gv = float(g)
            scale = gv / (1.0 + abs(gv))
            A[j, j] = A[j, j] + pos + 0.25 * scale

        for i in range(min(len(w) - 1, n - 1)):
            c = 0.5 * (float(w[i]) + float(w[i + 1]))
            c = c / (1.0 + max(abs(float(w[i])), abs(float(w[i + 1])), 1.0))
            A[i, i + 1] = A[i, i + 1] + c
            A[i + 1, i] = A[i + 1, i] + c

        d = torch.diagonal(A)
        nf = torch.linalg.norm(torch.abs(d))
        if float(nf) > 1e-12:
            A = A / nf
        return A

    def dirac_from_operator(self, H: torch.Tensor) -> torch.Tensor:
        """
        Treat the given operator matrix as a Dirac-like operator D.
        Symmetrize: D = 0.5 * (H + H^T).
        """
        H = H.to(device=self.device, dtype=self.dtype)
        return 0.5 * (H + H.mT)

    def commutator(self, D: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        return D @ A - A @ D

    def commutator_norm(self, D: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
        return torch.linalg.norm(self.commutator(D, A), ord="fro")

    def spectral_distance(self, eig1: Any, eig2: Any) -> torch.Tensor:
        """
        Euclidean distance between two spectra (truncated to common length, sorted order assumed).
        """
        e1 = torch.as_tensor(eig1, dtype=self.dtype, device=self.device).reshape(-1)
        e2 = torch.as_tensor(eig2, dtype=self.dtype, device=self.device).reshape(-1)
        m = min(e1.numel(), e2.numel())
        if m == 0:
            return torch.tensor(0.0, dtype=self.dtype, device=self.device)
        return torch.linalg.norm(e1[:m] - e2[:m])

    def ncg_collapse_loss(self, D: torch.Tensor, A: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        Penalize near-commutative collapse (large when [D,A] is small in Frobenius norm):
            1 / (eps + ||[D,A]||_F)
        """
        n = self.commutator_norm(D, A)
        e = torch.as_tensor(float(eps), dtype=n.dtype, device=n.device)
        return 1.0 / (e + n)
