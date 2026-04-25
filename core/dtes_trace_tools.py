from __future__ import annotations

"""Trace-formula diagnostics for experimental DTES spectral operators.

These helpers compare finite DTES spectra with zeta-inspired diagnostics. They
are numerical research signals for the ACO feedback loop, not RH proof tools.
"""

import numpy as np


def _finite_real(values) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr)]


def heat_trace(eigvals, t):
    eigvals = _finite_real(eigvals)
    if eigvals.size == 0:
        return 0.0
    trace = np.sum(np.exp(-float(t) * eigvals))
    return float(np.nan_to_num(trace, nan=0.0, posinf=0.0, neginf=0.0))


def heat_trace_curve(eigvals, t_grid):
    eigvals = _finite_real(eigvals)
    t_grid = _finite_real(t_grid)
    if eigvals.size == 0:
        return np.zeros_like(t_grid, dtype=float)
    return np.array(
        [np.sum(np.exp(-float(t) * eigvals)) for t in t_grid],
        dtype=float,
    )


def spectral_density(eigvals, grid, bandwidth=0.5):
    eigvals = _finite_real(eigvals)
    grid = np.asarray(grid, dtype=np.float64)
    density = np.zeros_like(grid, dtype=float)
    if eigvals.size == 0:
        return density

    bandwidth = max(float(bandwidth), 1e-12)
    for lmbda in eigvals:
        density += np.exp(-0.5 * ((grid - lmbda) / bandwidth) ** 2)
    density /= bandwidth * np.sqrt(2 * np.pi) * len(eigvals) + 1e-12
    return np.nan_to_num(density, nan=0.0, posinf=0.0, neginf=0.0)


def riemann_counting(T):
    T = np.asarray(T, dtype=np.float64)
    T_safe = np.maximum(T, 1e-12)
    count = (
        (T_safe / (2 * np.pi)) * np.log(T_safe / (2 * np.pi))
        - T_safe / (2 * np.pi)
        + 7 / 8
    )
    return np.nan_to_num(count, nan=0.0, posinf=0.0, neginf=0.0)


def empirical_count(eigvals, T_grid):
    eigvals = _finite_real(eigvals)
    T_grid = np.asarray(T_grid, dtype=np.float64)
    return np.array([np.sum(eigvals <= T) for T in T_grid], dtype=float)


def _normalize(x):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x
    return (x - x.mean()) / (x.std() + 1e-12)


def counting_loss(eigvals, T_grid):
    emp = empirical_count(eigvals, T_grid)
    theo = riemann_counting(T_grid)
    if emp.size == 0 or theo.size == 0:
        return float("inf")
    theo = _normalize(theo)
    emp = _normalize(emp)
    return float(np.mean((emp - theo) ** 2))


def spectral_logdet(eigvals, s):
    eigvals = np.asarray(eigvals, dtype=np.complex128).reshape(-1)
    eigvals = eigvals[np.isfinite(eigvals.real) & np.isfinite(eigvals.imag)]
    if eigvals.size == 0:
        return complex(0.0)
    return np.sum(np.log(complex(s) - eigvals + 1e-12))


def xi_approx(s, zeta_fn):
    import mpmath as mp

    s_mp = mp.mpc(complex(s))
    return (
        mp.mpf("0.5")
        * s_mp
        * (s_mp - 1)
        * (mp.pi ** (-s_mp / 2))
        * mp.gamma(s_mp / 2)
        * zeta_fn(s_mp)
    )


def determinant_loss(eigvals, t_values, zeta_fn):
    losses = []
    for t in _finite_real(t_values):
        s = 0.5 + 1j * float(t)
        try:
            logdet = spectral_logdet(eigvals, s)
            xi_val = xi_approx(s, zeta_fn)
            xi_abs = abs(complex(xi_val))
        except Exception:
            continue

        if not np.isfinite(xi_abs) or xi_abs <= 0.0:
            continue

        val = np.log(xi_abs + 1e-12)
        delta = abs(float(np.real(logdet)) - float(val))
        if np.isfinite(delta):
            losses.append(delta)

    return float(np.mean(losses)) if losses else float("inf")


def v5_loss_components(eigvals, zeta_zeros, T_grid, zeta_fn):
    from core.dtes_spectral_learning import total_loss

    eigvals = np.sort(_finite_real(eigvals))[:100]
    spec = float(total_loss(eigvals, zeta_zeros))
    count = float(counting_loss(eigvals, T_grid))
    det = float(determinant_loss(eigvals, _finite_real(T_grid)[:20], zeta_fn))
    total = spec + 0.5 * count + 0.1 * det
    return {
        "spectral": spec,
        "counting": count,
        "determinant": det,
        "total": float(total),
    }


def v5_total_loss(eigvals, zeta_zeros, T_grid, zeta_fn):
    return v5_loss_components(eigvals, zeta_zeros, T_grid, zeta_fn)["total"]
