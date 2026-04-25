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


class LowFrequencyPotential(nn.Module):
    def __init__(self, n, n_modes=12):
        super().__init__()
        self.n = int(n)
        self.n_modes = int(n_modes)
        self.a0 = nn.Parameter(torch.zeros(1))
        self.cos_coeff = nn.Parameter(torch.zeros(self.n_modes))
        self.sin_coeff = nn.Parameter(torch.zeros(self.n_modes))

    def forward(self):
        device = self.a0.device
        x = torch.linspace(0.0, 1.0, self.n, device=device)

        V = self.a0 * torch.ones_like(x)

        for k in range(1, self.n_modes + 1):
            V = V + self.cos_coeff[k - 1] * torch.cos(2.0 * torch.pi * k * x)
            V = V + self.sin_coeff[k - 1] * torch.sin(2.0 * torch.pi * k * x)

        return V


class SchrodingerOperator(nn.Module):
    def __init__(self, n, dx, potential_type="free", n_modes=12):
        super().__init__()
        self.n = int(n)
        self.dx = float(dx)
        self.potential_type = potential_type

        if potential_type == "lowfreq":
            self.V = LowFrequencyPotential(self.n, n_modes=n_modes)
        else:
            self.V = LearnablePotential(self.n)

    def forward(self):
        L = build_1d_laplacian(self.n, self.dx, device=next(self.parameters()).device)
        V_diag = torch.diag(self.V())
        H = L + V_diag
        H = 0.5 * (H + H.T)
        return torch.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)


class SpectralCalibration(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(1))
        self.shift = nn.Parameter(torch.zeros(1))

    def forward(self, eigvals):
        scale = torch.exp(self.log_scale)
        return scale * eigvals + self.shift, scale, self.shift


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


def frequency_loss(V, cutoff_ratio=0.20):
    fft = torch.fft.fft(V)
    power = torch.abs(fft) ** 2

    n = V.numel()
    cutoff = max(1, int(n * float(cutoff_ratio)))
    high_freq_power = power[cutoff : n - cutoff]

    if high_freq_power.numel() == 0:
        return torch.tensor(0.0, device=V.device)

    return torch.mean(high_freq_power)


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
    frequency_weight=0.1,
    frequency_cutoff_ratio=0.20,
    potential_type="free",
    n_modes=12,
    spectral_calibration=False,
    calibration_weight=1e-4,
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

    model = SchrodingerOperator(
        n,
        dx,
        potential_type=potential_type,
        n_modes=n_modes,
    ).to(device)
    calibration = SpectralCalibration().to(device) if spectral_calibration else None
    opt_params = list(model.parameters())
    if calibration is not None:
        opt_params += list(calibration.parameters())
    opt = torch.optim.Adam(opt_params, lr=lr)
    model.spectral_calibration = calibration
    zeta_zeros = torch.tensor(zeta_zeros, dtype=torch.float32, device=device)
    zeta_zeros = zeta_zeros[torch.isfinite(zeta_zeros)]
    if zeta_zeros.numel() == 0:
        raise ValueError("zeta_zeros required for physics-constrained operator learning")

    history = []

    for step in range(int(steps)):
        opt.zero_grad()

        H = model()
        eigvals_raw = torch.linalg.eigvalsh(H)
        eigvals_raw = torch.nan_to_num(eigvals_raw, nan=0.0, posinf=0.0, neginf=0.0)
        if calibration is not None:
            eigvals_scaled, spectral_scale, spectral_shift = calibration(eigvals_raw)
        else:
            eigvals_scaled = eigvals_raw
            spectral_scale = torch.ones(1, dtype=torch.float32, device=device)
            spectral_shift = torch.zeros(1, dtype=torch.float32, device=device)
        eigvals_scaled = torch.nan_to_num(eigvals_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        kk = min(int(k), eigvals_scaled.numel(), zeta_zeros.numel())
        if kk == 0:
            raise ValueError("k and zeta_zeros must provide at least one comparable eigenvalue")
        eigvals_raw_k = eigvals_raw[:kk]
        eigvals_k = eigvals_scaled[:kk]
        zeta_zeros_k = zeta_zeros[:kk]

        loss_spec, loss_spacing = _spectral_losses(eigvals_k, zeta_zeros_k)
        V = model.V()
        loss_smooth = smoothness_loss(V)
        loss_curvature = curvature_loss(V)
        loss_tv = total_variation_loss(V)
        loss_amp = amplitude_loss(V)
        loss_freq = frequency_loss(V, cutoff_ratio=frequency_cutoff_ratio)
        calibration_reg = (torch.log(spectral_scale) ** 2) + 1e-4 * (spectral_shift ** 2)
        calibration_reg = torch.mean(calibration_reg)
        loss_prime = torch.tensor(0.0, device=device)

        if prime_weight > 0:
            try:
                from core.dtes_operator_learning import prime_reconstruction_loss_torch

                loss_prime = prime_reconstruction_loss_torch(
                    eigvals_scaled,
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
            + float(frequency_weight) * loss_freq
            + float(calibration_weight) * calibration_reg
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
            "frequency_loss": float(loss_freq.detach().cpu()),
            "spectral_scale": float(spectral_scale.detach().cpu().reshape(-1)[0]),
            "spectral_shift": float(spectral_shift.detach().cpu().reshape(-1)[0]),
            "calibration_reg": float(calibration_reg.detach().cpu()),
            "raw_eig_min": float(eigvals_raw_k.min().detach().cpu()),
            "raw_eig_max": float(eigvals_raw_k.max().detach().cpu()),
            "scaled_eig_min": float(eigvals_k.min().detach().cpu()),
            "scaled_eig_max": float(eigvals_k.max().detach().cpu()),
            "prime_loss": float(loss_prime.detach().cpu()),
        }
        history.append(row)

        if step % 50 == 0 or step == int(steps) - 1:
            print(
                f"[V8.3] step={step} loss={row['loss']:.6f} "
                f"spec={row['spectral_loss']:.6f} "
                f"spacing={row['spacing_loss']:.6f} "
                f"scale={row['spectral_scale']:.6f} "
                f"shift={row['spectral_shift']:.6f}"
            )

    return model, history
