from __future__ import annotations

"""Physics-constrained DTES operator learning.

This module learns a Schrödinger-type finite operator H = -Delta + V(x) with a
fixed local Laplacian and a learnable diagonal potential. It is an experimental
spectral fitting diagnostic, not a proof method.
"""

import torch
import torch.nn as nn


def build_1d_laplacian(n, dx, device="cpu"):
    L = torch.zeros((int(n), int(n)), dtype=torch.float32, device=device)

    for i in range(int(n)):
        L[i, i] = 2.0
        if i > 0:
            L[i, i - 1] = -1.0
        if i < int(n) - 1:
            L[i, i + 1] = -1.0

    return L / (float(dx) ** 2)


class LearnablePotential(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.raw = nn.Parameter(torch.zeros(int(n)))

    def forward(self):
        return self.raw


class SchrodingerOperator(nn.Module):
    def __init__(self, n, dx):
        super().__init__()
        self.n = int(n)
        self.dx = float(dx)
        self.V = LearnablePotential(self.n)

    def forward(self):
        L = build_1d_laplacian(self.n, self.dx, device=self.V.raw.device)
        V_diag = torch.diag(self.V())
        H = L + V_diag
        H = 0.5 * (H + H.T)
        return torch.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)


def smoothness_loss(V):
    if V.numel() < 2:
        return V.new_tensor(0.0)
    return torch.mean((V[1:] - V[:-1]) ** 2)


def curvature_loss(V):
    if V.numel() < 3:
        return V.new_tensor(0.0)
    return torch.mean((V[2:] - 2.0 * V[1:-1] + V[:-2]) ** 2)


def total_variation_loss(V):
    if V.numel() < 2:
        return V.new_tensor(0.0)
    return torch.mean(torch.abs(V[1:] - V[:-1]))


def amplitude_loss(V):
    return torch.mean(V ** 2)


def locality_penalty(H):
    off_band = H.clone()
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            if abs(i - j) > 2:
                off_band[i, j] = 0
    return torch.mean((H - off_band) ** 2)


def _normalize(x):
    return (x - x.mean()) / (x.std(unbiased=False) + 1e-8)


def _spectral_losses(eigvals_k, zeta_zeros_k):
    e = _normalize(eigvals_k)
    z = _normalize(zeta_zeros_k)
    loss_spec = torch.mean((e - z) ** 2)

    if eigvals_k.numel() < 2 or zeta_zeros_k.numel() < 2:
        return loss_spec, eigvals_k.new_tensor(0.0)

    e_diff = torch.diff(eigvals_k)
    z_diff = torch.diff(zeta_zeros_k)
    e_diff = _normalize(e_diff)
    z_diff = _normalize(z_diff)
    loss_spacing = torch.mean(torch.abs(e_diff - z_diff))
    return loss_spec, loss_spacing


def train_physics_operator(
    t_grid,
    zeta_zeros,
    steps=2000,
    lr=1e-3,
    k=50,
    prime_weight=0.0,
    smooth_weight=0.01,
    curvature_weight=0.1,
    tv_weight=0.01,
    amplitude_weight=0.001,
    device="cpu",
):
    import numpy as np

    t_grid = np.asarray(t_grid, dtype=float).reshape(-1)
    if t_grid.size < 2:
        raise ValueError("t_grid must contain at least two points")
    if not np.all(np.isfinite(t_grid)):
        raise ValueError("t_grid contains NaN or inf")

    n = len(t_grid)
    dx = float(t_grid[1] - t_grid[0])
    if dx <= 0:
        raise ValueError("t_grid must be strictly increasing")

    model = SchrodingerOperator(n, dx).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    zeta_zeros = torch.tensor(zeta_zeros, dtype=torch.float32, device=device)
    zeta_zeros = zeta_zeros[torch.isfinite(zeta_zeros)]
    if zeta_zeros.numel() == 0:
        raise ValueError("zeta_zeros required for physics-constrained operator learning")

    history = []

    for step in range(int(steps)):
        opt.zero_grad()

        H = model()
        eigvals = torch.linalg.eigvalsh(H)
        eigvals = torch.nan_to_num(eigvals, nan=0.0, posinf=0.0, neginf=0.0)
        kk = min(int(k), eigvals.numel(), zeta_zeros.numel())
        if kk == 0:
            raise ValueError("k and zeta_zeros must provide at least one comparable eigenvalue")
        eigvals_k = eigvals[:kk]
        zeta_zeros_k = zeta_zeros[:kk]

        loss_spec, loss_spacing = _spectral_losses(eigvals_k, zeta_zeros_k)
        V = model.V()
        loss_smooth = smoothness_loss(V)
        loss_curvature = curvature_loss(V)
        loss_tv = total_variation_loss(V)
        loss_amp = amplitude_loss(V)
        loss_prime = torch.tensor(0.0, device=device)

        if prime_weight > 0:
            try:
                from core.dtes_operator_learning import prime_reconstruction_loss_torch

                loss_prime = prime_reconstruction_loss_torch(
                    eigvals,
                    zeta_zeros,
                    x_max=500,
                    k=kk,
                    device=device,
                )
                if not torch.isfinite(loss_prime):
                    print("[WARN] invalid V8 prime loss, skipping prime-aware term")
                    loss_prime = torch.tensor(0.0, device=device)
            except Exception as exc:
                print(f"[WARN] V8 prime loss failed, skipping: {exc}")
                loss_prime = torch.tensor(0.0, device=device)

        loss = (
            loss_spec
            + 0.5 * loss_spacing
            + float(smooth_weight) * loss_smooth
            + float(curvature_weight) * loss_curvature
            + float(tv_weight) * loss_tv
            + float(amplitude_weight) * loss_amp
            + float(prime_weight) * loss_prime
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("physics operator loss became NaN or inf")

        loss.backward()
        opt.step()

        row = {
            "step": int(step),
            "loss": float(loss.detach().cpu()),
            "spectral_loss": float(loss_spec.detach().cpu()),
            "spacing_loss": float(loss_spacing.detach().cpu()),
            "smoothness_loss": float(loss_smooth.detach().cpu()),
            "curvature_loss": float(loss_curvature.detach().cpu()),
            "tv_loss": float(loss_tv.detach().cpu()),
            "amplitude_loss": float(loss_amp.detach().cpu()),
            "prime_loss": float(loss_prime.detach().cpu()),
        }
        history.append(row)

        if step % 50 == 0 or step == int(steps) - 1:
            print(
                f"[V8.1] step={step} loss={row['loss']:.6f} "
                f"spec={row['spectral_loss']:.6f} "
                f"spacing={row['spacing_loss']:.6f} "
                f"curv={row['curvature_loss']:.6f}"
            )

    return model, history
