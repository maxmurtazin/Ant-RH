from __future__ import annotations

"""Graph-DTES operator learning utilities.

This module learns a finite self-adjoint DTES graph operator H = L_DTES + V.
It is a numerical inverse spectral fitting experiment, not a proof method.
"""

import numpy as np
import torch
import torch.nn as nn

from core.spectral_stabilization import safe_torch_eigh


def safe_eigvalsh(H, jitter=1e-6, max_tries=6):
    # Prefer differentiable torch path; fallback to stabilized CPU/numpy if needed.
    eigvals, _, _ = safe_torch_eigh(
        H,
        k=None,
        return_eigenvectors=False,
        stabilize=True,
        seed=42,
    )
    return eigvals


def build_dtes_graph_weights(
    t_grid,
    zeta_abs,
    sigma_t=5.0,
    energy_beta=1.0,
    k_neighbors=12,
    eps=1e-12,
):
    t = np.asarray(t_grid, dtype=float).reshape(-1)
    z = np.maximum(np.asarray(zeta_abs, dtype=float).reshape(-1), eps)
    if t.size == 0:
        raise ValueError("t_grid must be non-empty")
    if z.size != t.size:
        raise ValueError("t_grid and zeta_abs must have the same length")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(z)):
        raise ValueError("t_grid and zeta_abs must be finite")

    n = len(t)
    sigma_t = max(float(sigma_t), eps)
    W = np.zeros((n, n), dtype=float)

    for i in range(n):
        dt = np.abs(t - t[i])
        local = np.exp(-(dt ** 2) / (2.0 * sigma_t ** 2))
        energy = np.exp(-float(energy_beta) * (z + z[i]))
        weights = local * energy
        weights[i] = 0.0

        if k_neighbors is not None and int(k_neighbors) > 0:
            kk = min(int(k_neighbors), n - 1)
            idx = np.argsort(weights)[-kk:]
            row = np.zeros(n, dtype=float)
            row[idx] = weights[idx]
            W[i] = row
        else:
            W[i] = weights

    W = 0.5 * (W + W.T)
    return np.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)


def normalized_graph_laplacian(W, eps=1e-8):
    W = np.asarray(W, dtype=float)
    d = W.sum(axis=1)
    inv_sqrt_d = 1.0 / np.sqrt(d + eps)
    D_inv = np.diag(inv_sqrt_d)
    L = np.eye(W.shape[0]) - D_inv @ W @ D_inv
    L = 0.5 * (L + L.T)
    L = np.nan_to_num(L, nan=0.0, posinf=1e6, neginf=-1e6)
    L = L + 1e-8 * np.eye(L.shape[0])
    return L


def graph_laplacian(W):
    W = np.asarray(W, dtype=float)
    return np.diag(W.sum(axis=1)) - W


class LearnableGraphDTESOperator(nn.Module):
    def __init__(
        self,
        L_init,
        potential_init=None,
        learn_edge_scale=True,
    ):
        super().__init__()

        L = torch.tensor(L_init, dtype=torch.float32)
        self.register_buffer("L_base", L)
        n = L.shape[0]

        if potential_init is None:
            potential_init = torch.zeros(n)
        else:
            potential_init = torch.tensor(potential_init, dtype=torch.float32)

        self.V = nn.Parameter(potential_init)

        if learn_edge_scale:
            self.log_edge_scale = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer("log_edge_scale", torch.zeros(1))

    def forward(self):
        edge_scale = torch.exp(self.log_edge_scale).clamp(1e-6, 1e6)
        V_safe = torch.nan_to_num(self.V, nan=0.0, posinf=1e3, neginf=-1e3)
        V_safe = torch.clamp(V_safe, -100.0, 100.0)

        H = edge_scale * self.L_base + torch.diag(V_safe)
        H = 0.5 * (H + H.T)
        n = H.shape[0]
        diag_jitter = 1e-6 * torch.linspace(0.0, 1.0, n, device=H.device)
        H = H + torch.diag(diag_jitter)
        H = torch.nan_to_num(H, nan=0.0, posinf=1e6, neginf=-1e6)
        return H, edge_scale


class SpectralCalibration(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(1))
        self.shift = nn.Parameter(torch.zeros(1))

    def forward(self, eigvals):
        scale = torch.exp(self.log_scale)
        return scale * eigvals + self.shift, scale, self.shift


def normalize_torch(x):
    return (x - x.mean()) / (x.std(unbiased=False) + 1e-8)


def spectral_loss(eigvals, zeta_zeros, k):
    k = min(int(k), eigvals.numel(), zeta_zeros.numel())
    if k == 0:
        return eigvals.new_tensor(float("inf"))
    e = normalize_torch(eigvals[:k])
    z = normalize_torch(zeta_zeros[:k])
    return torch.mean((e - z) ** 2)


def spacing_loss(eigvals, zeta_zeros, k):
    k = min(int(k), eigvals.numel(), zeta_zeros.numel())
    if k < 2:
        return eigvals.new_tensor(0.0)
    e = torch.diff(torch.sort(eigvals[:k]).values)
    z = torch.diff(torch.sort(zeta_zeros[:k]).values)
    m = min(e.numel(), z.numel())
    if m <= 1:
        return torch.tensor(0.0, device=eigvals.device)
    return torch.mean(torch.abs(normalize_torch(e[:m]) - normalize_torch(z[:m])))


def potential_smoothness_on_graph(V, W, device):
    Wt = torch.tensor(W, dtype=torch.float32, device=device)
    diff = V[:, None] - V[None, :]
    return torch.sum(Wt * diff**2) / (torch.sum(Wt) + 1e-8)


def potential_amplitude(V):
    return torch.mean(V**2)


def train_graph_dtes_operator(
    t_grid,
    zeta_abs,
    zeta_zeros,
    steps=3000,
    lr=1e-3,
    k=50,
    sigma_t=5.0,
    energy_beta=1.0,
    k_neighbors=12,
    laplacian_type="normalized",
    spectral_calibration=True,
    smooth_weight=0.01,
    amplitude_weight=0.001,
    device="cpu",
):
    W = build_dtes_graph_weights(
        t_grid,
        zeta_abs,
        sigma_t=sigma_t,
        energy_beta=energy_beta,
        k_neighbors=k_neighbors,
    )

    if laplacian_type == "normalized":
        L = normalized_graph_laplacian(W)
    elif laplacian_type == "standard":
        L = graph_laplacian(W)
    else:
        raise ValueError("laplacian_type must be 'normalized' or 'standard'")

    z = np.maximum(np.asarray(zeta_abs, dtype=float).reshape(-1), 1e-12)
    V0 = -np.log(z)
    V0 = (V0 - V0.mean()) / (V0.std() + 1e-12)

    model = LearnableGraphDTESOperator(L, potential_init=V0).to(device)
    zeta_zeros_t = torch.tensor(zeta_zeros, dtype=torch.float32, device=device)
    zeta_zeros_t = zeta_zeros_t[torch.isfinite(zeta_zeros_t)]
    if zeta_zeros_t.numel() == 0:
        raise ValueError("zeta_zeros required for Graph-DTES operator learning")

    calibration = SpectralCalibration().to(device) if spectral_calibration else None
    params = list(model.parameters())
    if calibration is not None:
        params += list(calibration.parameters())

    opt = torch.optim.Adam(params, lr=lr)
    history = []

    for step in range(int(steps)):
        opt.zero_grad()

        H, edge_scale = model()
        eig_raw = safe_eigvalsh(H)
        eig_raw = torch.nan_to_num(eig_raw, nan=0.0, posinf=0.0, neginf=0.0)

        if calibration is not None:
            eig, spec_scale, spec_shift = calibration(eig_raw)
        else:
            eig = eig_raw
            spec_scale = torch.tensor(1.0, device=device)
            spec_shift = torch.tensor(0.0, device=device)
        eig = torch.nan_to_num(eig, nan=0.0, posinf=0.0, neginf=0.0)

        loss_spec = spectral_loss(eig, zeta_zeros_t, k)
        loss_spacing = spacing_loss(eig, zeta_zeros_t, k)
        loss_smooth = potential_smoothness_on_graph(model.V, W, device)
        loss_amp = potential_amplitude(model.V)

        loss = (
            loss_spec
            + 0.5 * loss_spacing
            + float(smooth_weight) * loss_smooth
            + float(amplitude_weight) * loss_amp
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("Graph-DTES operator loss became NaN or inf")

        loss.backward()
        opt.step()

        if step % 50 == 0 or step == int(steps) - 1:
            rec = {
                "step": int(step),
                "loss": float(loss.detach().cpu()),
                "spectral_loss": float(loss_spec.detach().cpu()),
                "spacing_loss": float(loss_spacing.detach().cpu()),
                "smoothness_loss": float(loss_smooth.detach().cpu()),
                "amplitude_loss": float(loss_amp.detach().cpu()),
                "edge_scale": float(edge_scale.detach().cpu().reshape(-1)[0]),
                "spectral_scale": float(spec_scale.detach().cpu().reshape(-1)[0]),
                "spectral_shift": float(spec_shift.detach().cpu().reshape(-1)[0]),
            }
            history.append(rec)
            print(
                f"[V9] step={step} loss={rec['loss']:.6f} "
                f"spec={rec['spectral_loss']:.6f} "
                f"spacing={rec['spacing_loss']:.6f} "
                f"edge={rec['edge_scale']:.4f} "
                f"scale={rec['spectral_scale']:.4f}"
            )

    H_final, edge_scale = model()
    eig_raw = safe_eigvalsh(H_final)
    if calibration is not None:
        eig_scaled, spec_scale, spec_shift = calibration(eig_raw)
    else:
        eig_scaled = eig_raw
        spec_scale = torch.tensor(1.0, device=device)
        spec_shift = torch.tensor(0.0, device=device)

    result = {
        "W": W,
        "L": L,
        "H": H_final.detach().cpu().numpy(),
        "V": model.V.detach().cpu().numpy(),
        "eig_raw": eig_raw.detach().cpu().numpy(),
        "eig_scaled": eig_scaled.detach().cpu().numpy(),
        "history": history,
        "edge_scale": float(edge_scale.detach().cpu().reshape(-1)[0]),
        "spectral_scale": float(spec_scale.detach().cpu().reshape(-1)[0]),
        "spectral_shift": float(spec_shift.detach().cpu().reshape(-1)[0]),
    }
    return result
