from __future__ import annotations

"""Direct learning of self-adjoint DTES operators.

This module is an experimental Hilbert-Polya-inspired numerical fitting tool.
It learns a symmetric finite operator whose spectrum is compared with zeta-zero
ordinates; it is not a proof method.
"""

import numpy as np
import torch
import torch.nn as nn


class LearnableDTESOperator(nn.Module):
    def __init__(self, n, init_operator=None):
        super().__init__()
        self.n = int(n)

        if init_operator is None:
            A = 0.01 * torch.randn(self.n, self.n)
        else:
            A = torch.tensor(init_operator, dtype=torch.float32)

        if A.shape != (self.n, self.n):
            raise ValueError("init_operator must have shape (n, n)")

        self.raw = nn.Parameter(A)

    def forward(self):
        H = 0.5 * (self.raw + self.raw.T)
        H = torch.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
        H = H - torch.diag(torch.diag(H)) + torch.diag(torch.diag(H))
        return H


def normalize(x):
    return (x - x.mean()) / (x.std(unbiased=False) + 1e-8)


def spectral_loss_torch(eigvals, zeta_zeros):
    k = min(eigvals.numel(), zeta_zeros.numel())
    if k == 0:
        return eigvals.new_tensor(float("inf"))
    e = normalize(eigvals[:k])
    z = normalize(zeta_zeros[:k])
    return torch.mean((e - z) ** 2)


def spacing_loss_torch(eigvals, zeta_zeros):
    e = torch.diff(torch.sort(eigvals).values)
    z = torch.diff(torch.sort(zeta_zeros).values)

    k = min(e.numel(), z.numel())
    if k == 0:
        return eigvals.new_tensor(0.0)
    e = normalize(e[:k])
    z = normalize(z[:k])
    return torch.mean(torch.abs(e - z))


def operator_regularization(H):
    offdiag = H - torch.diag(torch.diag(H))
    symmetry_error = torch.mean((H - H.T) ** 2)
    scale_penalty = torch.mean(H ** 2)
    sparsity_penalty = torch.mean(torch.abs(offdiag))
    return symmetry_error + 1e-4 * scale_penalty + 1e-4 * sparsity_penalty


def rescale_eigvals_to_zeros_torch(eigvals, zeta_zeros, k):
    k = min(int(k), eigvals.numel(), zeta_zeros.numel())
    e = eigvals[:k]
    z = zeta_zeros[:k]
    e_norm = (e - e.mean()) / (e.std(unbiased=False) + 1e-8)
    return e_norm * (z.std(unbiased=False) + 1e-8) + z.mean()


def chebyshev_psi_torch(x_grid, gammas):
    x = x_grid[:, None]
    g = gammas[None, :]

    logx = torch.log(x)
    sqrt_x = torch.sqrt(x)
    cos_part = torch.cos(g * logx)
    sin_part = torch.sin(g * logx)

    denom = 0.25 + g**2
    real_part = sqrt_x * (0.5 * cos_part + g * sin_part) / denom

    psi = x_grid - 2.0 * torch.sum(real_part, dim=1)
    return torch.nan_to_num(psi, nan=0.0, posinf=0.0, neginf=0.0)


def pi_from_psi_soft_torch(psi, x_grid, temperature=0.2):
    lam = psi[1:] - psi[:-1]
    n = x_grid[1:]

    target = torch.log(n)
    temp = max(float(temperature), 1e-8)
    score = torch.exp(-((lam - target) ** 2) / (2 * temp**2))
    score = torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)

    pi_pred = torch.cumsum(score, dim=0)
    return pi_pred, score


def sieve_primes_bool(n):
    arr = [True] * (n + 1)
    if n >= 0:
        arr[0] = False
    if n >= 1:
        arr[1] = False
    p = 2
    while p * p <= n:
        if arr[p]:
            for q in range(p * p, n + 1, p):
                arr[q] = False
        p += 1
    return arr


def true_prime_count_torch(x_max, device):
    flags = sieve_primes_bool(int(x_max))
    vals = []
    c = 0
    for n in range(2, int(x_max) + 1):
        if flags[n]:
            c += 1
        vals.append(float(c))
    return torch.tensor(vals, dtype=torch.float32, device=device)


def prime_reconstruction_loss_torch(
    eigvals,
    zeta_zeros,
    x_max=500,
    k=50,
    temperature=0.25,
    device="cpu",
):
    k = min(int(k), eigvals.numel(), zeta_zeros.numel())
    if k == 0 or int(x_max) < 2:
        return eigvals.new_tensor(0.0)

    gammas = rescale_eigvals_to_zeros_torch(eigvals, zeta_zeros, k)
    x_grid = torch.arange(1, int(x_max) + 1, dtype=torch.float32, device=device)

    psi = chebyshev_psi_torch(x_grid, gammas)
    pi_pred, score = pi_from_psi_soft_torch(
        psi,
        x_grid,
        temperature=temperature,
    )

    pi_true = true_prime_count_torch(int(x_max), device)
    pi_true = pi_true[: pi_pred.numel()]

    pi_pred_n = (pi_pred - pi_pred.mean()) / (pi_pred.std(unbiased=False) + 1e-8)
    pi_true_n = (pi_true - pi_true.mean()) / (pi_true.std(unbiased=False) + 1e-8)
    loss_pi = torch.mean((pi_pred_n - pi_true_n) ** 2)

    score_mean = score.mean()
    density_target = float(len([p for p in sieve_primes_bool(int(x_max)) if p])) / max(1, int(x_max))
    loss_density = (score_mean - density_target) ** 2

    loss = loss_pi + 0.1 * loss_density
    return torch.nan_to_num(loss, nan=float("inf"), posinf=float("inf"), neginf=float("inf"))


def _loss_parts(
    model,
    zeta_zeros,
    k,
    spacing_weight,
    reg_weight,
    prime_weight,
    prime_x_max,
    prime_temperature,
    device,
):
    H = model()
    eigvals = torch.linalg.eigvalsh(H)
    eigvals_k = eigvals[: min(int(k), eigvals.numel(), zeta_zeros.numel())]

    loss_spec = spectral_loss_torch(eigvals_k, zeta_zeros[: eigvals_k.numel()])
    loss_spacing = spacing_loss_torch(eigvals_k, zeta_zeros[: eigvals_k.numel()])
    loss_reg = operator_regularization(H)
    loss_prime = torch.tensor(0.0, device=device)

    if prime_weight > 0:
        loss_prime = prime_reconstruction_loss_torch(
            eigvals,
            zeta_zeros,
            x_max=prime_x_max,
            k=k,
            temperature=prime_temperature,
            device=device,
        )
        if not torch.isfinite(loss_prime):
            print("[WARN] invalid prime reconstruction loss, skipping prime-aware term")
            loss_prime = torch.tensor(0.0, device=device)

    loss = (
        loss_spec
        + spacing_weight * loss_spacing
        + reg_weight * loss_reg
        + prime_weight * loss_prime
    )
    return H, eigvals, loss, loss_spec, loss_spacing, loss_reg, loss_prime


def train_operator(
    init_operator,
    zeta_zeros,
    steps=2000,
    lr=1e-3,
    k=50,
    spacing_weight=0.5,
    reg_weight=1e-3,
    prime_weight=0.0,
    prime_x_max=500,
    prime_temperature=0.25,
    device="cpu",
):
    init_operator = np.asarray(init_operator, dtype=np.float32)
    if init_operator.ndim != 2 or init_operator.shape[0] != init_operator.shape[1]:
        raise ValueError("init_operator must be a square matrix")
    if not np.all(np.isfinite(init_operator)):
        raise ValueError("init_operator contains NaN or inf")

    model = LearnableDTESOperator(init_operator.shape[0], init_operator).to(device)
    zeta_zeros = torch.tensor(zeta_zeros, dtype=torch.float32, device=device)
    if zeta_zeros.numel() == 0:
        raise ValueError("zeta_zeros required for direct operator learning")

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    for step in range(int(steps)):
        opt.zero_grad()

        H, eigvals, loss, loss_spec, loss_spacing, loss_reg, loss_prime = _loss_parts(
            model,
            zeta_zeros,
            k,
            spacing_weight,
            reg_weight,
            prime_weight,
            prime_x_max,
            prime_temperature,
            device,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("operator learning loss became NaN or inf")

        loss.backward()
        opt.step()

        if step % 50 == 0 or step == int(steps) - 1:
            history.append({
                "step": step,
                "loss": float(loss.detach().cpu()),
                "spectral_loss": float(loss_spec.detach().cpu()),
                "spacing_loss": float(loss_spacing.detach().cpu()),
                "prime_loss": float(loss_prime.detach().cpu()),
                "regularization": float(loss_reg.detach().cpu()),
            })
            print(
                f"[operator-learning] step={step} "
                f"loss={history[-1]['loss']:.6f} "
                f"spec={history[-1]['spectral_loss']:.6f} "
                f"spacing={history[-1]['spacing_loss']:.6f} "
                f"prime={history[-1]['prime_loss']:.6f}"
            )

    H_final = model().detach().cpu().numpy()
    eig_final = torch.linalg.eigvalsh(model()).detach().cpu().numpy()

    return H_final, eig_final, history
