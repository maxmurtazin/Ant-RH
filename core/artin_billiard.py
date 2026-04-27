from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def sample_artin_domain(n_x=64, n_y=64, y_min=0.15, y_max=4.0):
    """
    Approximate PSL(2,Z)\\H fundamental domain:
        |x| <= 1/2
        x^2 + y^2 >= 1
        y > 0

    Returns:
        points: tensor [N, 2] with columns x,y
    """
    xs = np.linspace(-0.5, 0.5, n_x)
    ys = np.linspace(y_min, y_max, n_y)

    pts = []
    for x in xs:
        for y in ys:
            if x * x + y * y >= 1.0 and y > 0:
                pts.append([x, y])

    if not pts:
        raise ValueError("Artin domain sampler produced no points")

    return torch.tensor(pts, dtype=torch.float32)


def hyperbolic_distance(p, q, eps=1e-8):
    """
    Distance on upper half-plane:
    cosh d = 1 + |p-q|^2 / (2 y_p y_q)
    """
    dx = p[..., 0] - q[..., 0]
    dy = p[..., 1] - q[..., 1]

    y1 = torch.clamp(p[..., 1], min=eps)
    y2 = torch.clamp(q[..., 1], min=eps)

    arg = 1.0 + (dx * dx + dy * dy) / (2.0 * y1 * y2)
    arg = torch.clamp(arg, min=1.0 + eps)

    return torch.acosh(arg)


def pairwise_hyperbolic_distance(points):
    P = points[:, None, :]
    Q = points[None, :, :]
    return hyperbolic_distance(P, Q)


def build_hyperbolic_graph(points, sigma=0.35, k_neighbors=12):
    """
    Build weighted graph on sampled Artin domain using hyperbolic distance.
    """
    d = pairwise_hyperbolic_distance(points)
    n = points.shape[0]
    eye = torch.eye(n, device=points.device, dtype=points.dtype)

    W = torch.exp(-d / sigma)
    W = W * (1.0 - eye)

    if k_neighbors is not None:
        k = min(int(k_neighbors), max(n - 1, 0))
        if k > 0:
            vals, idx = torch.topk(W, k=k, dim=1)
            W_sparse = torch.zeros_like(W)
            W_sparse.scatter_(1, idx, vals)
            W = 0.5 * (W_sparse + W_sparse.T)
        else:
            W = torch.zeros_like(W)

    return W


def graph_laplacian(W, eps=1e-8):
    D = torch.diag(W.sum(dim=1))
    L = D - W
    L = 0.5 * (L + L.T)
    L = L + eps * torch.eye(L.shape[0], device=W.device, dtype=W.dtype)
    return L


class ArtinDTESOperator(nn.Module):
    def __init__(self, points, sigma=0.35, k_neighbors=12):
        super().__init__()
        self.register_buffer("points", points)

        W = build_hyperbolic_graph(points, sigma=sigma, k_neighbors=k_neighbors)
        L = graph_laplacian(W)

        self.register_buffer("W_base", W)
        self.register_buffer("L_base", L)

        n = points.shape[0]

        self.V = nn.Parameter(0.01 * torch.randn(n))
        self.log_cL = nn.Parameter(torch.tensor([4.5], dtype=torch.float32))
        self.log_cV = nn.Parameter(torch.tensor([0.0], dtype=torch.float32))
        self.energy_shift = nn.Parameter(torch.tensor([100.0], dtype=torch.float32))

    def forward(self):
        cL = torch.exp(self.log_cL).clamp(1e-4, 1e5)
        cV = torch.exp(self.log_cV).clamp(1e-4, 1e5)
        b = self.energy_shift.clamp(-1e5, 1e5)

        H = cL * self.L_base + cV * torch.diag(self.V)
        H = H + b * torch.eye(H.shape[0], device=H.device, dtype=H.dtype)
        H = 0.5 * (H + H.T)

        return H, self.W_base, self.points, cL, cV, b


def primitive_geodesic_length_proxy(points, max_pairs=2000):
    """
    Simple proxy for closed geodesic length statistics:
    use pairwise hyperbolic distances as candidate orbit lengths.

    This is not a full Selberg trace formula yet.
    It is a computational proxy for billiard path lengths.
    """
    d = pairwise_hyperbolic_distance(points)
    vals = d.flatten()
    vals = vals[torch.isfinite(vals)]
    vals = vals[vals > 1e-6]

    if vals.numel() > max_pairs:
        idx = torch.linspace(
            0,
            vals.numel() - 1,
            max_pairs,
            device=vals.device,
        ).long()
        vals = vals[idx]

    return vals


def selberg_trace_proxy_loss(eig, lengths, k=50, tau=0.1):
    """
    Compare heat trace from eigenvalues with length-orbit proxy.

    Spectral heat trace:
        Tr exp(-tau H)

    Geodesic proxy:
        sum exp(-ell^2 / tau)
    """
    kk = min(int(k), eig.numel())
    if kk <= 0 or lengths.numel() == 0:
        return eig.new_tensor(0.0)

    e = eig[:kk]
    heat_spec = torch.sum(torch.exp(-tau * e))

    lengths = lengths.to(eig.device)
    heat_geo = torch.sum(torch.exp(-(lengths**2) / (tau + 1e-8)))

    heat_spec = heat_spec / (kk + 1e-8)
    heat_geo = heat_geo / (lengths.numel() + 1e-8)

    return (heat_spec - heat_geo) ** 2


def raw_spectral_loss(eig, zeta, k=50):
    k = min(int(k), eig.numel(), zeta.numel())
    if k <= 0:
        return eig.new_tensor(float("inf"))
    return torch.mean((eig[:k] - zeta[:k]) ** 2)


def spacing_loss(eig, zeta, k=50):
    k = min(int(k), eig.numel(), zeta.numel())
    if k < 2:
        return eig.new_tensor(0.0)
    de = eig[1:k] - eig[: k - 1]
    dz = zeta[1:k] - zeta[: k - 1]
    return torch.mean((de - dz) ** 2)


def range_loss(eig, zeta, k=50):
    k = min(int(k), eig.numel(), zeta.numel())
    if k <= 0:
        return eig.new_tensor(float("inf"))
    er = eig[:k].max() - eig[:k].min()
    zr = zeta[:k].max() - zeta[:k].min()
    return ((er - zr) / (zr + 1e-8)) ** 2


def potential_smoothness_loss(V, W):
    diff = V[:, None] - V[None, :]
    return torch.sum(W * diff**2) / (torch.sum(W) + 1e-8)
