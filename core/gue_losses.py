from __future__ import annotations

import torch


def unfolded_spacings(eig, k=50, eps=1e-8):
    k = min(int(k), eig.numel())
    if k < 2:
        return eig.new_zeros(0)
    vals = torch.sort(eig[:k]).values
    spacings = vals[1:] - vals[:-1]
    mean_spacing = torch.mean(spacings) + eps
    return spacings / mean_spacing


def gue_wigner_surmise_loss(eig, k=50, n_bins=32):
    """
    Match spacing distribution to approximate GUE Wigner surmise:
    P(s) = (32/pi^2) s^2 exp(-4s^2/pi)
    """
    s = unfolded_spacings(eig, k=k)
    if s.numel() == 0:
        return eig.new_tensor(0.0)
    s = torch.clamp(s, 0.0, 5.0)

    hist = torch.histc(s, bins=int(n_bins), min=0.0, max=5.0)
    hist = hist / (hist.sum() + 1e-8)

    centers = torch.linspace(0.0, 5.0, int(n_bins), device=eig.device)
    target = (32.0 / torch.pi**2) * centers**2 * torch.exp(
        -4.0 * centers**2 / torch.pi
    )
    target = target / (target.sum() + 1e-8)

    return torch.mean((hist - target) ** 2)


def log_gas_loss(eig, k=50, confinement=1e-3, eps=1e-6):
    """
    Dyson log-gas energy:
    E = confinement * sum(lambda_i^2) - sum_{i<j} log |lambda_i - lambda_j|
    """
    k = min(int(k), eig.numel())
    if k < 2:
        return eig.new_tensor(0.0)
    vals = torch.sort(eig[:k]).values
    vals = (vals - vals.mean()) / (vals.std() + eps)

    diff = torch.abs(vals[:, None] - vals[None, :]) + eps
    mask = torch.triu(torch.ones_like(diff), diagonal=1)

    repulsion = -torch.sum(mask * torch.log(diff)) / (k * k)
    conf = confinement * torch.mean(vals**2)

    return conf + repulsion


def spacing_variance_loss(eig, k=50):
    s = unfolded_spacings(eig, k=k)
    if s.numel() < 2:
        return eig.new_tensor(0.0)
    return torch.var(s)


def raw_spectral_loss(eig, zeta, k=50):
    k = min(int(k), eig.numel(), zeta.numel())
    if k <= 0:
        return eig.new_tensor(float("inf"))
    return torch.mean((eig[:k] - zeta[:k]) ** 2)


def range_loss(eig, zeta, k=50):
    k = min(int(k), eig.numel(), zeta.numel())
    if k <= 0:
        return eig.new_tensor(float("inf"))
    er = eig[:k].max() - eig[:k].min()
    zr = zeta[:k].max() - zeta[:k].min()
    return ((er - zr) / (zr + 1e-8)) ** 2
