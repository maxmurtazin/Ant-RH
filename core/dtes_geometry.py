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
        self.n = n
        self.dim = dim

        t = torch.linspace(0.0, 1.0, n)
        self.register_buffer("t_fixed", t)

        if dim > 1:
            self.y_raw = nn.Parameter(0.05 * torch.sin(2 * torch.pi * t))
        else:
            self.y_raw = None

        if dim > 2:
            self.z_raw = nn.Parameter(0.05 * torch.cos(2 * torch.pi * t))
        else:
            self.z_raw = None

    def forward(self):
        coords = [self.t_fixed]

        if self.dim > 1:
            coords.append(self.y_raw)

        if self.dim > 2:
            coords.append(self.z_raw)

        return torch.stack(coords, dim=1)


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
    trace = torch.trace(L)
    L = L / (trace / L.shape[0] + 1e-8)
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

        edge_scale = 1.0
        H = L + torch.diag(self.V)
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


def raw_spectral_loss(eig, zeta, k):
    k = min(int(k), eig.numel(), zeta.numel())
    if k <= 0:
        return eig.new_tensor(float("inf"))
    return torch.mean((eig[:k] - zeta[:k]) ** 2)


def anchor_loss(eig, zeta, n_anchor=5):
    k = min(int(n_anchor), eig.numel(), zeta.numel())
    if k <= 0:
        return eig.new_tensor(0.0)
    return torch.mean((eig[:k] - zeta[:k]) ** 2)


def spacing_anchor_loss(eig, zeta, n_anchor=5):
    k = min(int(n_anchor), eig.numel(), zeta.numel())

    if k < 2:
        return torch.tensor(0.0, device=eig.device)

    de = eig[1:k] - eig[: k - 1]
    dz = zeta[1:k] - zeta[: k - 1]

    return torch.mean((de - dz) ** 2)


def affine_penalty(eig, zeta, k):
    k = min(int(k), eig.numel(), zeta.numel())
    if k <= 1:
        return eig.new_tensor(0.0)
    x = eig[:k]
    y = zeta[:k]

    A = torch.stack([x, torch.ones_like(x)], dim=1)
    sol = torch.linalg.lstsq(A, y.unsqueeze(1)).solution

    a = sol[0, 0]
    b = sol[1, 0]

    return (a - 1.0) ** 2 + b**2


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


def uniform_spacing_loss(Z):
    z = torch.sort(Z[:, 0]).values
    diffs = z[1:] - z[:-1]
    return torch.var(diffs)


def ordering_loss(Z):
    z = Z[:, 0]
    z_sorted, _ = torch.sort(z)
    return torch.mean((z - z_sorted) ** 2)


def center_loss(Z):
    return torch.mean(Z[:, 0] ** 2)


def curve_smoothness_loss(Z):
    if Z.shape[0] < 3 or Z.shape[1] < 2:
        return Z.new_tensor(0.0)

    loss = Z.new_tensor(0.0)

    for d in range(1, Z.shape[1]):
        y = Z[:, d]
        loss = loss + torch.mean((y[2:] - 2 * y[1:-1] + y[:-2]) ** 2)

    return loss


def curve_amplitude_loss(Z):
    if Z.shape[1] < 2:
        return Z.new_tensor(0.0)

    return torch.mean(Z[:, 1:] ** 2)


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
    no_scaling=False,
    generalization_split=False,
    anchor_loss_enabled=False,
    anchor_weight=10.0,
    spacing_anchor_weight=5.0,
    n_anchor=5,
    line_geometry=False,
    spacing_weight=1.0,
    ordering_weight=0.5,
    center_weight=0.1,
    parametric_line=False,
    curve_smooth_weight=1.0,
    curve_amp_weight=0.1,
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
    train_idx = None
    test_idx = None
    zeta_train = zeta
    zeta_test = None
    if no_scaling and generalization_split:
        split_n = min(zeta.numel(), n)
        idx_all = torch.arange(split_n, device=device)
        train_idx = idx_all[0::2]
        test_idx = idx_all[1::2]
        zeta_train = zeta[train_idx]
        zeta_test = zeta[test_idx] if test_idx.numel() else None

    history = []
    base_weights = {
        "spectral": 5.0,
        "spacing": 1.0,
        "growth": 3.0,
        "spread": 10.0,
        "geom": 0.01,
        "repulsion": 0.0001,
        "amplitude": 0.001,
    }
    pauli_loss_available = valid_mask_arr is not None and not bool(np.all(valid_mask_arr))
    if pauli_loss_available:
        base_weights["pauli"] = float(pauli_weight)
    loss_names = list(base_weights)
    controller = None
    if adaptive_weights and not no_scaling:
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

        if no_scaling and train_idx is not None:
            eig_train = eig[train_idx]
            loss_spec = raw_spectral_loss(eig_train, zeta_train, k)
            loss_spacing = spacing_loss(eig_train, zeta_train, k)
            loss_growth = spectral_growth_loss(eig_train, zeta_train, k=k)
            loss_affine = affine_penalty(eig_train, zeta_train, k)
            if zeta_test is not None and test_idx is not None and test_idx.numel():
                test_error_t = raw_spectral_loss(eig[test_idx], zeta_test, k)
            else:
                test_error_t = eig.new_tensor(float("nan"))
        elif no_scaling:
            loss_spec = raw_spectral_loss(eig, zeta, k)
            loss_spacing = spacing_loss(eig, zeta, k)
            loss_growth = spectral_growth_loss(eig, zeta, k=k)
            loss_affine = affine_penalty(eig, zeta, k)
            test_error_t = eig.new_tensor(float("nan"))
        else:
            loss_spec = spectral_loss(eig, zeta, k)
            loss_spacing = spacing_loss(eig, zeta, k)
            loss_growth = spectral_growth_loss(eig, zeta, k=k)
            loss_affine = eig.new_tensor(0.0)
            test_error_t = eig.new_tensor(float("nan"))
        loss_spread = spectral_spread_loss(eig, k=k, min_std=0.3)

        geom_reg = torch.mean(Z**2)
        d = torch.cdist(Z, Z) + 1e-3
        repulsion = torch.mean(1.0 / d)
        amp = torch.mean(model.V**2)
        embedding_spread = torch.std(Z)
        loss_pauli = torch.tensor(0.0, device=device)
        eig_window = eig[: min(int(k), eig.numel())]
        eig_std = torch.std(eig_window)
        loss_line_spacing = torch.tensor(0.0, device=device)
        loss_ordering = torch.tensor(0.0, device=device)
        loss_center = torch.tensor(0.0, device=device)
        loss_curve_smooth = torch.tensor(0.0, device=device)
        loss_curve_amp = torch.tensor(0.0, device=device)
        curve_geometry = parametric_line or line_geometry
        if line_geometry and not curve_geometry:
            loss_line_spacing = uniform_spacing_loss(Z)
            loss_ordering = ordering_loss(Z)
            loss_center = center_loss(Z)
        if curve_geometry:
            loss_curve_smooth = curve_smoothness_loss(Z)
            loss_curve_amp = curve_amplitude_loss(Z)
        loss_anchor = torch.tensor(0.0, device=device)
        loss_spacing_anchor = torch.tensor(0.0, device=device)
        if anchor_loss_enabled:
            loss_anchor = anchor_loss(eig, zeta, n_anchor=n_anchor)
            loss_spacing_anchor = spacing_anchor_loss(eig, zeta, n_anchor=n_anchor)

        loss_dict = {
            "spectral": loss_spec.detach().item(),
            "spacing": loss_spacing.detach().item(),
            "growth": loss_growth.detach().item(),
            "spread": loss_spread.detach().item(),
            "geom": geom_reg.detach().item(),
            "repulsion": repulsion.detach().item(),
            "amplitude": amp.detach().item(),
        }
        if no_scaling:
            loss_dict["affine"] = loss_affine.detach().item()
        if pauli_loss_available:
            loss_dict["pauli"] = loss_pauli.detach().item()
        if controller is not None:
            weights = controller.update(loss_dict)
        else:
            weights = dict(base_weights)

        if no_scaling:
            loss = (
                5.0 * loss_spec
                + 2.0 * loss_spacing
                + 3.0 * loss_growth
                + 5.0 * loss_affine
                + 2.0 * loss_spread
                + 0.01 * geom_reg
                + 0.0001 * repulsion
            )
            weights = {
                "raw_spectral": 5.0,
                "spacing": 2.0,
                "growth": 3.0,
                "affine": 5.0,
                "spread": 2.0,
                "geom": 0.01,
                "repulsion": 0.0001,
            }
            if anchor_loss_enabled:
                weights["anchor"] = float(anchor_weight)
                weights["spacing_anchor"] = float(spacing_anchor_weight)
            if pauli_loss_available:
                weights["pauli"] = float(pauli_weight)
        else:
            loss = (
                weights["spectral"] * loss_spec
                + weights["spacing"] * loss_spacing
                + weights["growth"] * loss_growth
                + weights["spread"] * loss_spread
                + weights["geom"] * geom_reg
                + weights["repulsion"] * repulsion
                + weights["amplitude"] * amp
            )
            if anchor_loss_enabled:
                weights["anchor"] = float(anchor_weight)
                weights["spacing_anchor"] = float(spacing_anchor_weight)
        loss = (
            loss
            + float(anchor_weight) * loss_anchor
            + float(spacing_anchor_weight) * loss_spacing_anchor
        )
        if line_geometry and not curve_geometry:
            loss = (
                loss
                + float(spacing_weight) * loss_line_spacing
                + float(ordering_weight) * loss_ordering
                + float(center_weight) * loss_center
            )
            weights["line_spacing"] = float(spacing_weight)
            weights["ordering"] = float(ordering_weight)
            weights["center"] = float(center_weight)
        if curve_geometry:
            loss = (
                loss
                + float(curve_smooth_weight) * loss_curve_smooth
                + float(curve_amp_weight) * loss_curve_amp
            )
            weights["curve_smooth"] = float(curve_smooth_weight)
            weights["curve_amp"] = float(curve_amp_weight)
        if pauli_loss_available:
            loss = loss + weights["pauli"] * loss_pauli
        if not torch.isfinite(loss):
            raise FloatingPointError("DTES geometry loss became NaN or inf")

        loss.backward()
        opt.step()

        if step % 50 == 0:
            if curve_geometry:
                print(
                    f"[V10-PARAM-LINE] step={step} "
                    f"curve_smooth={loss_curve_smooth.item():.6f} "
                    f"curve_amp={loss_curve_amp.item():.6f}"
                )
            elif line_geometry:
                print(
                    f"[V10-LINE] step={step} "
                    f"loss={loss.item():.6f} "
                    f"spacing={loss_line_spacing.item():.4f} "
                    f"ordering={loss_ordering.item():.4f}"
                )
            elif anchor_loss_enabled:
                print(
                    f"[V10-ANCHOR] step={step} "
                    f"loss={loss.item():.6f} "
                    f"anchor={loss_anchor.item():.4f} "
                    f"spacing_anchor={loss_spacing_anchor.item():.4f}"
                )
            elif no_scaling:
                print(
                    f"[V10.3] step={step} "
                    f"loss={loss.item():.6f} "
                    f"train_mse={loss_spec.item():.4f} "
                    f"test_mse={test_error_t.item():.4f}"
                )
            elif controller is not None:
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
                "anchor_loss": float(loss_anchor.detach().cpu()),
                "spacing_anchor_loss": float(loss_spacing_anchor.detach().cpu()),
                "n_anchor": int(n_anchor),
                "line_spacing_loss": float(loss_line_spacing.detach().cpu()),
                "ordering_loss": float(loss_ordering.detach().cpu()),
                "center_loss": float(loss_center.detach().cpu()),
                "curve_smoothness_loss": float(loss_curve_smooth.detach().cpu()),
                "curve_amplitude_loss": float(loss_curve_amp.detach().cpu()),
                "affine_penalty": float(loss_affine.detach().cpu()),
                "train_error": float(loss_spec.detach().cpu()),
                "test_error": float(test_error_t.detach().cpu()),
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
        if no_scaling and train_idx is not None:
            final_train_error = raw_spectral_loss(eig[train_idx], zeta_train, k)
            if zeta_test is not None and test_idx is not None and test_idx.numel():
                final_test_error = raw_spectral_loss(eig[test_idx], zeta_test, k)
            else:
                final_test_error = eig.new_tensor(float("nan"))
            final_affine_penalty = affine_penalty(eig[train_idx], zeta_train, k)
        elif no_scaling:
            final_train_error = raw_spectral_loss(eig, zeta, k)
            final_test_error = eig.new_tensor(float("nan"))
            final_affine_penalty = affine_penalty(eig, zeta, k)
        else:
            final_train_error = eig.new_tensor(float("nan"))
            final_test_error = eig.new_tensor(float("nan"))
            final_affine_penalty = eig.new_tensor(float("nan"))

    return {
        "Z": Z.detach().cpu().numpy(),
        "W": W.detach().cpu().numpy(),
        "V": model.V.detach().cpu().numpy(),
        "eig": eig.detach().cpu().numpy(),
        "history": history,
        "final_weights": history[-1]["weights"] if history else dict(base_weights),
        "train_error": float(final_train_error.detach().cpu()),
        "test_error": float(final_test_error.detach().cpu()),
        "affine_penalty": float(final_affine_penalty.detach().cpu()),
        "edge_scale": float(edge_scale),
    }
