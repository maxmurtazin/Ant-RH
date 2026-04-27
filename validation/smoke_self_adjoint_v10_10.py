from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.operator_health import validate_operator_health


def check(name, H, should_pass_raw):
    _, eig, rep = validate_operator_health(H)
    print(name, rep)
    assert rep["finite_outputs"]
    assert eig is not None
    assert rep["hermitian_error_projected"] <= 1e-8
    assert bool(rep["hermitian_error_raw"] <= 1e-8) == bool(should_pass_raw)


def main():
    A = torch.randn(8, 8)
    check("real_symmetric", 0.5 * (A + A.T), True)

    B = torch.randn(8, 8)
    check("real_nonsymmetric", B, False)

    C = torch.randn(8, 8) + 1j * torch.randn(8, 8)
    Hc = 0.5 * (C + C.conj().T)
    check("complex_hermitian", Hc, True)

    D = torch.randn(8, 8) + 1j * torch.randn(8, 8)
    check("complex_nonhermitian", D, False)

    L = torch.diag(torch.linspace(0.1, 3.0, 8))
    alpha = 0.7
    Lfrac = L**alpha
    check("fractional_laplacian", Lfrac, True)

    print("[OK] V10.10 self-adjoint smoke tests passed")


if __name__ == "__main__":
    main()
