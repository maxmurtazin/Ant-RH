from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from core.artin_operator import hyperbolic_distance_matrix, sample_domain

DT = np.float64


def reconstruct_points(n: int, seed: int = 42) -> np.ndarray:
    return sample_domain(int(n), seed=int(seed)).astype(np.complex128, copy=False)


def pairwise_hyperbolic_distance(z: np.ndarray) -> np.ndarray:
    return hyperbolic_distance_matrix(z, z).astype(DT, copy=False)


@dataclass
class FeatureContext:
    d: np.ndarray
    d2: np.ndarray
    sigma: float
    V: np.ndarray
    rho: np.ndarray
    inv_im: np.ndarray
    log_im: np.ndarray
    mean_length: float
    std_length: float
    laplacian: np.ndarray
    kernel_rbf: np.ndarray
    kernel_inv: np.ndarray


def build_feature_context(H: np.ndarray, z: np.ndarray, lengths: np.ndarray, sigma: float = 0.6) -> FeatureContext:
    H = np.asarray(H, dtype=DT)
    z = np.asarray(z, dtype=np.complex128)
    d = pairwise_hyperbolic_distance(z)
    d2 = d * d
    sigma2 = float(sigma) ** 2 + 1e-12
    kernel_rbf = np.exp(-d2 / sigma2, dtype=DT)
    np.fill_diagonal(kernel_rbf, 0.0)
    kernel_inv = 1.0 / (1.0 + d)
    np.fill_diagonal(kernel_inv, 0.0)
    rho = np.sum(kernel_rbf, axis=1).astype(DT, copy=False)
    V = np.sum(H, axis=1).astype(DT, copy=False)
    y = np.maximum(np.imag(z).astype(DT, copy=False), 1e-9)
    inv_im = 1.0 / y
    log_im = np.log(y)
    A = kernel_rbf
    D = np.diag(np.sum(A, axis=1))
    lap = D - A

    lengths = np.asarray(lengths, dtype=DT).reshape(-1)
    lengths = lengths[np.isfinite(lengths)]
    if lengths.size == 0:
        mean_length = 0.0
        std_length = 0.0
    else:
        mean_length = float(np.mean(lengths))
        std_length = float(np.std(lengths))

    return FeatureContext(
        d=d,
        d2=d2,
        sigma=float(sigma),
        V=V,
        rho=rho,
        inv_im=inv_im,
        log_im=log_im,
        mean_length=mean_length,
        std_length=std_length,
        laplacian=lap,
        kernel_rbf=kernel_rbf,
        kernel_inv=kernel_inv,
    )


def feature_matrix_for_psi(psi: np.ndarray, ctx: FeatureContext) -> Tuple[np.ndarray, List[str]]:
    psi = np.asarray(psi, dtype=DT).reshape(-1)
    k_rbf = ctx.kernel_rbf @ psi
    k_inv = ctx.kernel_inv @ psi
    lap = ctx.laplacian @ psi
    feats = [
        psi,  # identity
        lap,  # graph laplacian
        k_rbf,  # hyperbolic radial kernel
        k_inv,  # inverse-distance kernel
        ctx.V * psi,  # diagonal potential
        ctx.rho * psi,  # local density
        (ctx.mean_length * np.ones_like(psi)) * psi,  # global mean length term
        (ctx.std_length * np.ones_like(psi)) * psi,  # global std length term
        psi**3,  # nonlinear cubic
        np.abs(psi) * psi,  # nonlinear abs
        ctx.inv_im * psi,  # cusp proxy inverse-im
        ctx.log_im * psi,  # cusp proxy log-im
    ]
    names = [
        "psi",
        "laplacian_psi",
        "rbf_kernel_psi",
        "inv_distance_kernel_psi",
        "potential_psi",
        "density_psi",
        "mean_length_psi",
        "std_length_psi",
        "psi_cubed",
        "abs_psi_times_psi",
        "inv_im_psi",
        "log_im_psi",
    ]
    X = np.stack(feats, axis=1).astype(DT, copy=False)
    return X, names


def stack_eigenpair_system(eigvals: np.ndarray, eigvecs: np.ndarray, ctx: FeatureContext, k_use: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    eigvals = np.asarray(eigvals, dtype=DT).reshape(-1)
    eigvecs = np.asarray(eigvecs, dtype=DT)
    n = eigvecs.shape[0]
    kk = min(int(k_use), int(eigvals.size), int(eigvecs.shape[1]))
    X_blocks: List[np.ndarray] = []
    y_blocks: List[np.ndarray] = []
    names: List[str] = []
    for i in range(kk):
        psi = eigvecs[:, i].reshape(n)
        X_i, names_i = feature_matrix_for_psi(psi, ctx)
        if not names:
            names = names_i
        y_i = eigvals[i] * psi
        X_blocks.append(X_i)
        y_blocks.append(y_i)
    if not X_blocks:
        return np.zeros((0, 0), dtype=DT), np.zeros((0,), dtype=DT), []
    return np.vstack(X_blocks), np.concatenate(y_blocks), names

