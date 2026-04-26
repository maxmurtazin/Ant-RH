from __future__ import annotations

"""Learnable DTES geometry utilities.

This module learns a finite embedding whose induced graph Laplacian defines a
self-adjoint DTES operator. It is a numerical geometry learning experiment, not
a proof method for the Riemann hypothesis.
"""

import numpy as np
import torch
import torch.nn as nn

from core.adaptive_loss_controller import AdaptiveLossController


class DTESGeometry(nn.Module):
    def __init__(self, n, dim=2):
        super().__init__()
        t = torch.linspace(0.0, 1.0, n)
        Z_init = torch.zeros(n, dim)
        Z_init[:, 0] = t
        if dim > 1:
            Z_init[:, 1] = 0.1 * torch.sin(10 * t)
        if dim > 2:
            Z_init[:, 2] = 0.1 * torch.cos(10 * t)
        self.Z = nn.Parameter(Z_init)

    def forward(self):
        return self.Z


def build_graph_from_embedding(Z, sigma=1.0, k_neighbors=8, valid_mask=None):
    dists = torch.cdist(Z, Z)
    dists = dists**1.5

    W = torch.exp(-dists / sigma)
    W = W * (1 - torch.eye(W.shape[0], device=W.device, dtype=W.dtype))
    if valid_mask is not None:
        mask = valid_mask.to(device=W.device, dtype=W.dtype)
        W = W * mask[:, None] * mask[None, :]

    if k_neighbors is not None:
        k = min(int(k_neighbors), max(W.shape[0] - 1, 0))
        if k > 0:
            vals, idx = torch.topk(W, k=k, dim=1)
            W_sparse = torch.zeros_like(W)
            W_sparse.scatter_(1, idx, vals)
            W = 0.5 * (W_sparse + W_sparse.T)
        else:
            W = torch.zeros_like(W)

    W = 0.5 * (W + W.T)
    if valid_mask is not None:
        mask = valid_mask.to(device=W.device, dtype=W.dtype)
        W = W * mask[:, None] * mask[None, :]

    return W


def graph_laplacian_torch(W, eps=1e-5):
    D = torch.diag(W.sum(dim=1))
    L = D - W
    L = 0.5 * (L + L.T)
    L = L + eps * torch.eye(L.shape[0], device=W.device, dtype=W.dtype)
    return L


def safe_eigvalsh(H):
    try:
        return torch.linalg.eigvalsh(H)
    except RuntimeError:
        H = 0.5 * (H + H.T)
        H = H + 1e-5 * torch.eye(H.shape[0], device=H.device, dtype=H.dtype)
        return torch.linalg.eigvalsh(H)


def pauli_hole_loss(Z_valid, Z_invalid, margin=0.1):
    """
    Push valid geometry away from forbidden Pauli nodal holes.
    """
    if Z_invalid is None or Z_invalid.numel() == 0:
        return torch.tensor(0.0, device=Z_valid.device)

    d = torch.cdist(Z_valid, Z_invalid)
    return torch.mean(torch.relu(margin - d) ** 2)


class DTESGraphOperator(nn.Module):
    def __init__(self, n, dim=2, valid_mask=None):
        super().__init__()
        self.geometry = DTESGeometry(n, dim)
        self.V = nn.Parameter(0.01 * torch.randn(n))
        self.edge_logits = nn.Parameter(0.01 * torch.randn(n, n))
        self.log_edge_scale = nn.Parameter(torch.zeros(1))
        if valid_mask is None:
            self.valid_mask = None
        else:
            mask = torch.as_tensor(valid_mask, dtype=torch.bool).reshape(-1)
            if mask.numel() != n:
                raise ValueError("valid_mask length must match n")
            self.register_buffer("valid_mask", mask)

    def forward(self):
        Z = self.geometry()
        W = build_graph_from_embedding(Z, valid_mask=self.valid_mask)
        E = torch.sigmoid(0.5 * (self.edge_logits + self.edge_logits.T))
        W = W * (0.5 + E)
        W = W * (1 - torch.eye(W.shape[0], device=W.device, dtype=W.dtype))
        if self.valid_mask is not None:
            mask = self.valid_mask.to(device=W.device, dtype=W.dtype)
            W = W * mask[:, None] * mask[None, :]
        L = graph_laplacian_torch(W)

        edge_scale = torch.exp(self.log_edge_scale).clamp(0.1, 100.0)
        H = edge_scale * L + torch.diag(self.V)
        H = 0.5 * (H + H.T)

        return H, W, Z, edge_scale


def normalize(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def spectral_loss(eig, zeta, k):
    k = min(int(k), eig.numel(), zeta.numel())
    if k <= 0:
        return eig.new_tensor(float("inf"))
    e = normalize(eig[:k])
    z = normalize(zeta[:k])
    return torch.mean((e - z) ** 2)


def spacing_loss(eig, zeta, k):
    k = min(int(k), eig.numel(), zeta.numel())
    if k < 2:
        return eig.new_tensor(0.0)
    e = torch.diff(torch.sort(eig[:k]).values)
    z = torch.diff(torch.sort(zeta[:k]).values)
    m = min(e.numel(), z.numel())
    if m <= 1:
        return eig.new_tensor(0.0)
    return torch.mean(torch.abs(normalize(e[:m]) - normalize(z[:m])))


def spectral_spread_loss(eig, k=50, min_std=0.3):
    vals = eig[:k]
    std = torch.std(vals)
    return torch.relu(torch.tensor(min_std, device=eig.device) - std) ** 2


def normalize_range(x):
    return (x - x[0]) / (x[-1] - x[0] + 1e-8)


def spectral_growth_loss(eig, zeta_zeros, k=50):
    k = min(k, eig.numel(), zeta_zeros.numel())
    e = normalize_range(torch.sort(eig[:k]).values)
    z = normalize_range(torch.sort(zeta_zeros[:k]).values)
    return torch.mean((e - z) ** 2)


def _as_finite_1d(name, values):
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least one finite value")
    return arr


def _embedding_repulsion(Z):
    n = Z.shape[0]
    if n < 2:
        return Z.new_tensor(0.0)
    d = torch.cdist(Z, Z) + 1e-3
    return torch.mean(1.0 / d)


def train_dtes_geometry(
    t_grid,
    zeta_zeros,
    steps=3000,
    lr=1e-3,
    k=50,
    dim=2,
    max_n=400,
    device="cpu",
    valid_mask=None,
    pauli_weight=1.0,
    adaptive_weights=False,
    adaptive_lr=0.05,
):
    t_grid = _as_finite_1d("t_grid", t_grid)
    zeta_zeros = np.sort(np.abs(_as_finite_1d("zeta_zeros", zeta_zeros)))
    zeta_zeros = zeta_zeros[zeta_zeros > 0.0]
    if zeta_zeros.size == 0:
        raise ValueError("zeta_zeros must contain at least one positive finite value")

    valid_mask_arr = None
    if valid_mask is not None:
        valid_mask_arr = np.asarray(valid_mask, dtype=bool).reshape(-1)
        if valid_mask_arr.size != len(t_grid):
            raise ValueError("valid_mask length must match t_grid length")

    if len(t_grid) > max_n:
        idx = np.linspace(0, len(t_grid) - 1, max_n).astype(int)
        t_grid = t_grid[idx]
        if valid_mask_arr is not None:
            valid_mask_arr = valid_mask_arr[idx]

    n = len(t_grid)

    model = DTESGraphOperator(n, dim, valid_mask=valid_mask_arr).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    zeta = torch.tensor(zeta_zeros, dtype=torch.float32, device=device)

    history = []
    base_weights = {
        "spectral": 5.0,
        "spacing": 1.0,
        "growth": 3.0,
        "spread": 10.0,
        "geom": 0.01,
        "repulsion": 0.001,
        "amplitude": 0.001,
    }
    pauli_loss_available = valid_mask_arr is not None and not bool(np.all(valid_mask_arr))
    if pauli_loss_available:
        base_weights["pauli"] = float(pauli_weight)
    loss_names = list(base_weights)
    controller = None
    if adaptive_weights:
        controller = AdaptiveLossController(
            names=loss_names,
            init_weights=base_weights,
            lr=adaptive_lr,
            min_weight=1e-4,
            max_weight=20.0,
        )

    for step in range(int(steps)):
        opt.zero_grad()

        H, W, Z, edge_scale = model()
        eig = safe_eigvalsh(H)

        loss_spec = spectral_loss(eig, zeta, k)
        loss_spacing = spacing_loss(eig, zeta, k)
        loss_spread = spectral_spread_loss(eig, k=k, min_std=0.3)
        loss_growth = spectral_growth_loss(eig, zeta, k=k)

        geom_reg = torch.mean(Z**2)
        d = torch.cdist(Z, Z) + 1e-3
        repulsion = torch.mean(1.0 / d)
        amp = torch.mean(model.V**2)
        embedding_spread = torch.std(Z)
        loss_pauli = torch.tensor(0.0, device=device)
        eig_window = eig[: min(int(k), eig.numel())]
        eig_std = torch.std(eig_window)

        loss_dict = {
            "spectral": loss_spec.detach().item(),
            "spacing": loss_spacing.detach().item(),
            "growth": loss_growth.detach().item(),
            "spread": loss_spread.detach().item(),
            "geom": geom_reg.detach().item(),
            "repulsion": repulsion.detach().item(),
            "amplitude": amp.detach().item(),
        }
        if pauli_loss_available:
            loss_dict["pauli"] = loss_pauli.detach().item()
        if controller is not None:
            weights = controller.update(loss_dict)
        else:
            weights = dict(base_weights)

        loss = (
            weights["spectral"] * loss_spec
            + weights["spacing"] * loss_spacing
            + weights["growth"] * loss_growth
            + weights["spread"] * loss_spread
            + weights["geom"] * geom_reg
            + weights["repulsion"] * repulsion
            + weights["amplitude"] * amp
        )
        if pauli_loss_available:
            loss = loss + weights["pauli"] * loss_pauli
        if not torch.isfinite(loss):
            raise FloatingPointError("DTES geometry loss became NaN or inf")

        loss.backward()
        opt.step()

        if step % 50 == 0:
            if controller is not None:
                print(
                    f"[V10-AUTO] step={step} loss={loss.item():.6f} "
                    f"spec={loss_spec.item():.4f} spacing={loss_spacing.item():.4f} "
                    f"w_spec={weights['spectral']:.3f} "
                    f"w_spacing={weights['spacing']:.3f}"
                )
            else:
                print(
                    f"[V10-GRAPH-FIX] step={step} "
                    f"loss={loss.item():.6f} "
                    f"eig_std={eig_std.item():.4f} "
                )

        history.append(
            {
                "step": int(step),
                "loss": float(loss.detach().cpu()),
                "spectral_loss": float(loss_spec.detach().cpu()),
                "spacing_loss": float(loss_spacing.detach().cpu()),
                "spread_loss": float(loss_spread.detach().cpu()),
                "growth_loss": float(loss_growth.detach().cpu()),
                "geom_loss": float(geom_reg.detach().cpu()),
                "repulsion_loss": float(repulsion.detach().cpu()),
                "amplitude_loss": float(amp.detach().cpu()),
                "pauli_loss": float(loss_pauli.detach().cpu()),
                "spread": float(embedding_spread.detach().cpu()),
                "eig_mean": float(eig.detach().mean().cpu()),
                "eig_std": float(eig_std.detach().cpu()),
                "eig_min": float(eig_window.min().detach().cpu()),
                "eig_max": float(eig_window.max().detach().cpu()),
                "weights": {k: float(v) for k, v in weights.items()},
            }
        )

    with torch.no_grad():
        H, W, Z, edge_scale = model()
        eig = safe_eigvalsh(H)

    return {
        "Z": Z.detach().cpu().numpy(),
        "W": W.detach().cpu().numpy(),
        "V": model.V.detach().cpu().numpy(),
        "eig": eig.detach().cpu().numpy(),
        "history": history,
        "final_weights": history[-1]["weights"] if history else dict(base_weights),
        "edge_scale": float(edge_scale.detach().cpu().reshape(-1)[0]),
    }
