from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.dtes_spectral_operator import (  # noqa: E402
    build_dtes_operator,
    compare_spectral_statistics,
    compute_spectrum,
    spectral_alignment_loss,
)


def _fixture():
    t_grid = np.linspace(10.0, 40.0, 16)
    zeta_abs = np.abs(np.sin(t_grid)) + 0.05
    pheromone = np.zeros((t_grid.size, t_grid.size), dtype=float)
    for i in range(t_grid.size - 1):
        pheromone[i, i + 1] = 1.0 + 0.1 * i
        pheromone[i + 1, i] = 0.5 + 0.05 * i
    zeros = np.array([14.134725141734693, 21.022039638771554, 25.010857580145688])
    return t_grid, zeta_abs, pheromone, zeros


def test_operator_is_symmetric():
    t_grid, zeta_abs, pheromone, _zeros = _fixture()
    H = build_dtes_operator(t_grid, zeta_abs, pheromone)
    assert np.allclose(H, H.T)


def test_eigenvalues_are_real():
    t_grid, zeta_abs, pheromone, _zeros = _fixture()
    H = build_dtes_operator(t_grid, zeta_abs, pheromone)
    eigvals = compute_spectrum(H, k=8)
    assert np.isrealobj(eigvals)


def test_no_nan_or_inf():
    t_grid, zeta_abs, pheromone, _zeros = _fixture()
    H = build_dtes_operator(t_grid, zeta_abs, pheromone, potential_mode="inverse")
    eigvals = compute_spectrum(H, k=8)
    assert np.all(np.isfinite(H))
    assert np.all(np.isfinite(eigvals))


def test_spectral_loss_returns_finite_values():
    t_grid, zeta_abs, pheromone, zeros = _fixture()
    H = build_dtes_operator(t_grid, zeta_abs, pheromone)
    eigvals = compute_spectrum(H, k=3)
    report = spectral_alignment_loss(eigvals, zeros)
    assert report["n_compared"] == 3
    assert np.isfinite(report["mae"])
    assert np.isfinite(report["rmse"])
    stats = compare_spectral_statistics(eigvals, zeros)
    assert np.isfinite(stats["mae"])


def test_normalized_laplacian_works():
    t_grid, zeta_abs, pheromone, _zeros = _fixture()
    H = build_dtes_operator(
        t_grid,
        zeta_abs,
        pheromone,
        normalize_laplacian=True,
    )
    eigvals = compute_spectrum(H, k=8)
    assert np.allclose(H, H.T)
    assert np.all(np.isfinite(eigvals))


if __name__ == "__main__":
    test_operator_is_symmetric()
    test_eigenvalues_are_real()
    test_no_nan_or_inf()
    test_spectral_loss_returns_finite_values()
    test_normalized_laplacian_works()
    print("[OK] dtes spectral operator tests passed")
