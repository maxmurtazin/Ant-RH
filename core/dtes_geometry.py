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


def build_multiscale_graph_from_embedding(
    Z,
    sigmas=(0.02, 0.10, 0.35),
    k_neighbors=(4, 16, 64),
    scale_logits=None,
    valid_mask=None,
):
    dists = torch.cdist(Z, Z)
    n = Z.shape[0]
    eye = torch.eye(n, device=Z.device, dtype=Z.dtype)

    Ws = []

    for sigma, k in zip(sigmas, k_neighbors):
        W = torch.exp(-dists / sigma)
        W = W * (1.0 - eye)

        if valid_mask is not None:
            mask = valid_mask.to(device=W.device, dtype=W.dtype)
            W = W * mask[:, None] * mask[None, :]

        if k is not None and k > 0:
            kk = min(int(k), max(n - 1, 0))
            if kk > 0:
                vals, idx = torch.topk(W, k=kk, dim=1)
                W_sparse = torch.zeros_like(W)
                W_sparse.scatter_(1, idx, vals)
                W = 0.5 * (W_sparse + W_sparse.T)
            else:
                W = torch.zeros_like(W)

        Ws.append(W)

    if scale_logits is None:
        weights = torch.ones(len(Ws), device=Z.device, dtype=Z.dtype) / len(Ws)
    else:
        weights = torch.softmax(scale_logits, dim=0)

    W_total = torch.zeros_like(Ws[0])
    for w, W in zip(weights, Ws):
        W_total = W_total + w * W

    W_total = 0.5 * (W_total + W_total.T)
    if valid_mask is not None:
        mask = valid_mask.to(device=W_total.device, dtype=W_total.dtype)
        W_total = W_total * mask[:, None] * mask[None, :]

    return W_total, weights


def build_nested_graphs(Z, valid_mask=None):
    dists = torch.cdist(Z, Z)
    n = Z.shape[0]
    eye = torch.eye(n, device=Z.device, dtype=Z.dtype)

    configs = [
        (0.02, 4),  # local
        (0.10, 16),  # medium
        (0.35, 64),  # global
    ]

    Ws = []

    for sigma, k in configs:
        W = torch.exp(-dists / sigma)
        W = W * (1.0 - eye)

        if valid_mask is not None:
            mask = valid_mask.to(device=W.device, dtype=W.dtype)
            W = W * mask[:, None] * mask[None, :]

        kk = min(int(k), max(n - 1, 0))
        if kk > 0:
            vals, idx = torch.topk(W, k=kk, dim=1)
            W_sparse = torch.zeros_like(W)
            W_sparse.scatter_(1, idx, vals)
            W = 0.5 * (W_sparse + W_sparse.T)
        else:
            W = torch.zeros_like(W)

        if valid_mask is not None:
            mask = valid_mask.to(device=W.device, dtype=W.dtype)
            W = W * mask[:, None] * mask[None, :]

        Ws.append(W)

    return Ws


def graph_laplacian_torch(W, eps=1e-5, normalize_trace=True):
    D = torch.diag(W.sum(dim=1))
    L = D - W
    L = 0.5 * (L + L.T)
    if normalize_trace:
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


def build_magnetic_hermitian_operator(W, phase, beta=1.0, diagonal_potential=None):
    """
    Build complex Hermitian magnetic graph operator.

    Off-diagonal:
        H_ij = - W_ij * exp(i beta (S_i - S_j))

    Diagonal:
        H_ii = sum_j W_ij + V_i
    """
    dphi = phase[:, None] - phase[None, :]
    U = torch.exp(1j * float(beta) * dphi)

    Wc = W.to(torch.complex64)
    H = -Wc * U

    degree = W.sum(dim=1)

    if diagonal_potential is None:
        diag = degree
    else:
        diag = degree + diagonal_potential

    H = H + torch.diag(diag.to(torch.complex64))

    # enforce Hermitian symmetry
    H = 0.5 * (H + H.conj().T)

    return H


def safe_eigvalsh_complex(H, jitter=1e-6, max_tries=6):
    n = H.shape[0]
    eye = torch.eye(n, dtype=H.dtype, device=H.device)

    H = 0.5 * (H + H.conj().T)
    real = torch.nan_to_num(H.real, nan=0.0, posinf=1e6, neginf=-1e6)
    imag = torch.nan_to_num(H.imag, nan=0.0, posinf=1e6, neginf=-1e6)
    H = torch.complex(real, imag)

    last_err = None

    for i in range(max_tries):
        try:
            jj = jitter * (10.0**i)
            return torch.linalg.eigvalsh(H + jj * eye)
        except RuntimeError as e:
            last_err = e

    raise RuntimeError(f"complex eigvalsh failed: {last_err}")


def wavefunction_fields(log_amplitude, phase):
    A = torch.exp(log_amplitude)
    psi_real = A * torch.cos(phase)
    psi_imag = A * torch.sin(phase)
    psi_abs = torch.sqrt(psi_real**2 + psi_imag**2 + 1e-12)
    return A, psi_real, psi_imag, psi_abs


def amplitude_energy_loss(log_amplitude, V):
    pred = log_amplitude
    target = -V.detach()

    pred = (pred - pred.mean()) / (pred.std() + 1e-8)
    target = (target - target.mean()) / (target.std() + 1e-8)

    return torch.mean((pred - target) ** 2)


def phase_flow_loss(phase, W):
    diff = phase[:, None] - phase[None, :]
    return torch.sum(W * diff**2) / (torch.sum(W) + 1e-8)


def phase_coupled_weights(W, phase, coupling=1.0):
    dphi = phase[:, None] - phase[None, :]
    phase_factor = torch.cos(coupling * dphi)
    Wp = W * phase_factor
    Wp = torch.nan_to_num(Wp, nan=0.0, posinf=1e6, neginf=-1e6)
    Wp = 0.5 * (Wp + Wp.T)
    return Wp


def phase_coupled_weights_positive(W, phase, coupling=1.0):
    dphi = phase[:, None] - phase[None, :]
    phase_factor = 0.5 * (1.0 + torch.cos(coupling * dphi))
    Wp = W * phase_factor
    Wp = 0.5 * (Wp + Wp.T)
    return Wp


def phase_activity_loss(phase, min_std=0.02):
    return torch.relu(phase.new_tensor(min_std) - torch.std(phase)) ** 2


def phase_activity_target_loss(phase, target_std=1.0):
    if phase.numel() < 2:
        return phase.new_tensor(0.0)
    dphi = phase[1:] - phase[:-1]
    std = torch.std(dphi)
    return (std - target_std) ** 2


def phase_curvature_loss(phase):
    if phase.numel() < 3:
        return phase.new_tensor(0.0)
    return torch.mean((phase[2:] - 2.0 * phase[1:-1] + phase[:-2]) ** 2)


def project_phase_gradient_std_(phase, target_std=0.5):
    if phase.numel() < 2:
        return
    dphi = phase[1:] - phase[:-1]
    std = torch.std(dphi)
    if not torch.isfinite(std) or std <= 1e-8:
        return
    center = phase.mean()
    phase.sub_(center).mul_(float(target_std) / (std + 1e-8)).add_(center)


def nodal_sparsity_loss(log_amplitude, target_fraction=0.05):
    A = torch.exp(log_amplitude)
    threshold = torch.quantile(A.detach(), target_fraction)
    node_score = torch.sigmoid(20.0 * (threshold - A))
    return torch.mean((node_score.mean() - target_fraction) ** 2)


def amplitude_collapse_loss(log_amplitude):
    A = torch.exp(log_amplitude)
    return torch.relu(0.1 - torch.std(A)) ** 2


def nodal_score(log_amplitude, target_fraction=0.05):
    A = torch.exp(log_amplitude)
    threshold = torch.quantile(A.detach(), target_fraction)
    return torch.sigmoid(20.0 * (threshold - A))


def pauli_hole_loss(Z_valid, Z_invalid, margin=0.1):
    """
    Push valid geometry away from forbidden Pauli nodal holes.
    """
    if Z_invalid is None or Z_invalid.numel() == 0:
        return torch.tensor(0.0, device=Z_valid.device)

    d = torch.cdist(Z_valid, Z_invalid)
    return torch.mean(torch.relu(margin - d) ** 2)


class DTESGraphOperator(nn.Module):
    def __init__(
        self,
        n,
        dim=2,
        valid_mask=None,
        multiscale_geometry=False,
        nested_geometry=False,
        phase_coupled_operator=False,
        phase_coupling=1.0,
        phase_coupling_positive=False,
        magnetic_operator=False,
        magnetic_beta=1.0,
    ):
        super().__init__()
        self.geometry = DTESGeometry(n, dim)
        self.multiscale_geometry = bool(multiscale_geometry)
        self.nested_geometry = bool(nested_geometry)
        self.phase_coupled_operator = bool(phase_coupled_operator)
        self.phase_coupling = float(phase_coupling)
        self.phase_coupling_positive = bool(phase_coupling_positive)
        self.magnetic_operator = bool(magnetic_operator)
        self.magnetic_beta = float(magnetic_beta)
        self.V = nn.Parameter(0.01 * torch.randn(n))
        self.edge_logits = nn.Parameter(0.01 * torch.randn(n, n))
        self.multiscale_logits = nn.Parameter(torch.zeros(3))
        self.level_logits = nn.Parameter(torch.zeros(3))
        self.log_amplitude = nn.Parameter(torch.zeros(n))
        t = torch.linspace(0.0, 1.0, n)
        self.phase = nn.Parameter(0.05 * torch.sin(2 * torch.pi * t))
        if valid_mask is None:
            self.valid_mask = None
        else:
            mask = torch.as_tensor(valid_mask, dtype=torch.bool).reshape(-1)
            if mask.numel() != n:
                raise ValueError("valid_mask length must match n")
            self.register_buffer("valid_mask", mask)

    def _apply_nodal_barrier(self, W):
        A, _, _, _ = wavefunction_fields(self.log_amplitude, self.phase)
        barrier = torch.exp(-2.0 * (A[:, None] + A[None, :]))
        W = W * (1.0 - barrier)
        return 0.5 * (W + W.T)

    def _apply_phase_coupling(self, W):
        if self.phase_coupling_positive:
            return phase_coupled_weights_positive(
                W,
                self.phase,
                coupling=self.phase_coupling,
            )
        return phase_coupled_weights(
            W,
            self.phase,
            coupling=self.phase_coupling,
        )

    def forward(self, wavefunction_topology=False):
        Z = self.geometry()
        if self.nested_geometry:
            Ws = build_nested_graphs(Z, valid_mask=self.valid_mask)
            if wavefunction_topology:
                Ws = [self._apply_nodal_barrier(W_level) for W_level in Ws]
            if self.phase_coupled_operator:
                Ws = [self._apply_phase_coupling(W_level) for W_level in Ws]
            levels = []
            for W_level in Ws:
                L_level = graph_laplacian_torch(W_level, normalize_trace=False)
                levels.append(L_level)

            raw_weights = torch.softmax(self.level_logits, dim=0)
            level_floor = 0.05
            ms_weights = level_floor + (1.0 - len(Ws) * level_floor) * raw_weights
            H = torch.zeros_like(levels[0])
            W = torch.zeros_like(Ws[0])
            for w, L_level, W_level in zip(ms_weights, levels, Ws):
                H = H + w * L_level
                W = W + w * W_level

            edge_scale = 1.0
            H = H + torch.diag(self.V)
            H = 0.5 * (H + H.T)
            W = 0.5 * (W + W.T)
            if self.magnetic_operator:
                H = build_magnetic_hermitian_operator(
                    W,
                    self.phase,
                    beta=self.magnetic_beta,
                    diagonal_potential=self.V,
                )

            return H, W, Z, edge_scale, ms_weights

        if self.multiscale_geometry:
            W, ms_weights = build_multiscale_graph_from_embedding(
                Z,
                scale_logits=self.multiscale_logits,
                valid_mask=self.valid_mask,
            )
        else:
            W = build_graph_from_embedding(Z, valid_mask=self.valid_mask)
            ms_weights = torch.softmax(self.multiscale_logits.detach(), dim=0)
        if wavefunction_topology:
            W = self._apply_nodal_barrier(W)
        if self.phase_coupled_operator:
            W = self._apply_phase_coupling(W)
        E = torch.sigmoid(0.5 * (self.edge_logits + self.edge_logits.T))
        W = W * (0.5 + E)
        W = W * (1 - torch.eye(W.shape[0], device=W.device, dtype=W.dtype))
        if self.valid_mask is not None:
            mask = self.valid_mask.to(device=W.device, dtype=W.dtype)
            W = W * mask[:, None] * mask[None, :]
        if self.magnetic_operator:
            edge_scale = 1.0
            H = build_magnetic_hermitian_operator(
                W,
                self.phase,
                beta=self.magnetic_beta,
                diagonal_potential=self.V,
            )
            return H, W, Z, edge_scale, ms_weights
        L = graph_laplacian_torch(W, normalize_trace=not self.multiscale_geometry)

        edge_scale = 1.0
        H = L + torch.diag(self.V)
        H = 0.5 * (H + H.T)

        return H, W, Z, edge_scale, ms_weights


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


def spectral_range_loss(eig, zeta, k=50):
    k = min(int(k), eig.numel(), zeta.numel())
    if k <= 0:
        return eig.new_tensor(float("inf"))
    erange = eig[:k].max() - eig[:k].min()
    zrange = zeta[:k].max() - zeta[:k].min()
    return ((erange - zrange) / (zrange + 1e-8)) ** 2


def multiscale_entropy_loss(weights):
    return torch.sum(weights * torch.log(weights + 1e-8))


def level_entropy_loss(weights):
    return torch.sum(weights * torch.log(weights + 1e-8))


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
    multiscale_geometry=False,
    range_weight=10.0,
    nested_geometry=False,
    wavefunction_topology=False,
    wave_amp_weight=0.2,
    phase_weight=0.05,
    nodal_weight=0.1,
    amplitude_collapse_weight=0.1,
    phase_activity_weight=1.0,
    phase_target_std=0.5,
    phase_curvature_weight=0.05,
    phase_coupled_operator=False,
    phase_coupling=1.0,
    phase_coupling_positive=False,
    magnetic_operator=False,
    magnetic_beta=1.0,
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

    model = DTESGraphOperator(
        n,
        dim,
        valid_mask=valid_mask_arr,
        multiscale_geometry=multiscale_geometry,
        nested_geometry=nested_geometry,
        phase_coupled_operator=phase_coupled_operator,
        phase_coupling=phase_coupling,
        phase_coupling_positive=phase_coupling_positive,
        magnetic_operator=magnetic_operator,
        magnetic_beta=magnetic_beta,
    ).to(device)
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

        H, W, Z, edge_scale, ms_weights = model(
            wavefunction_topology=wavefunction_topology
        )
        eig = safe_eigvalsh_complex(H).real if magnetic_operator else safe_eigvalsh(H)
        hermitian_error = torch.mean(torch.abs(H - H.conj().T))

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
        elif nested_geometry:
            loss_spec = raw_spectral_loss(eig, zeta, k)
            loss_spacing = spacing_loss(eig, zeta, k)
            loss_growth = spectral_growth_loss(eig, zeta, k=k)
            loss_affine = eig.new_tensor(0.0)
            test_error_t = eig.new_tensor(float("nan"))
        else:
            loss_spec = spectral_loss(eig, zeta, k)
            loss_spacing = spacing_loss(eig, zeta, k)
            loss_growth = spectral_growth_loss(eig, zeta, k=k)
            loss_affine = eig.new_tensor(0.0)
            test_error_t = eig.new_tensor(float("nan"))
        loss_spread = spectral_spread_loss(eig, k=k, min_std=0.3)
        loss_range = spectral_range_loss(eig, zeta, k)
        loss_ms_entropy = (
            level_entropy_loss(ms_weights)
            if nested_geometry
            else multiscale_entropy_loss(ms_weights)
        )
        A, psi_real, psi_imag, psi_abs = wavefunction_fields(
            model.log_amplitude,
            model.phase,
        )
        loss_amp_energy = amplitude_energy_loss(model.log_amplitude, model.V)
        loss_phase = phase_flow_loss(model.phase, W)
        loss_phase_activity = phase_activity_loss(model.phase)
        loss_phase_activity_target = phase_activity_target_loss(
            model.phase,
            target_std=phase_target_std,
        )
        loss_phase_curvature = phase_curvature_loss(model.phase)
        loss_nodal = nodal_sparsity_loss(model.log_amplitude)
        loss_amp_collapse = amplitude_collapse_loss(model.log_amplitude)

        geom_reg = torch.mean(Z**2)
        d = torch.cdist(Z, Z) + 1e-3
        repulsion = torch.mean(1.0 / d)
        amp = torch.mean(model.V**2)
        embedding_spread = torch.std(Z)
        loss_pauli = torch.tensor(0.0, device=device)
        eig_window = eig[: min(int(k), eig.numel())]
        eig_std = torch.std(eig_window)
        zeta_window = zeta[: min(int(k), zeta.numel())]
        eig_range = eig_window.max() - eig_window.min()
        zeta_range = zeta_window.max() - zeta_window.min()
        phase_grad = model.phase[1:] - model.phase[:-1]
        phase_grad_std = (
            torch.std(phase_grad) if phase_grad.numel() > 1 else model.phase.new_tensor(0.0)
        )
        phase_pair_diff = model.phase[:, None] - model.phase[None, :]
        if phase_coupling_positive:
            phase_weight_factor = 0.5 * (
                1.0 + torch.cos(float(phase_coupling) * phase_pair_diff)
            )
        else:
            phase_weight_factor = torch.cos(float(phase_coupling) * phase_pair_diff)
        phase_weight_min = phase_weight_factor.min()
        phase_weight_max = phase_weight_factor.max()
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
        if multiscale_geometry or nested_geometry:
            loss = (
                loss
                + float(range_weight) * loss_range
                + 0.01 * loss_ms_entropy
            )
            weights["range"] = float(range_weight)
            if nested_geometry:
                weights["level_entropy"] = 0.01
            else:
                weights["multiscale_entropy"] = 0.01
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
        if wavefunction_topology:
            loss = (
                loss
                + float(wave_amp_weight) * loss_amp_energy
                + float(phase_weight) * loss_phase
                + float(phase_weight) * loss_phase_activity
                + float(phase_activity_weight) * loss_phase_activity_target
                + float(phase_curvature_weight) * loss_phase_curvature
                + float(nodal_weight) * loss_nodal
                + float(amplitude_collapse_weight) * loss_amp_collapse
            )
            weights["wave_amp"] = float(wave_amp_weight)
            weights["phase"] = float(phase_weight)
            weights["phase_activity"] = float(phase_activity_weight)
            weights["phase_curvature"] = float(phase_curvature_weight)
            weights["nodal"] = float(nodal_weight)
            weights["amplitude_collapse"] = float(amplitude_collapse_weight)
        if pauli_loss_available:
            loss = loss + weights["pauli"] * loss_pauli
        if not torch.isfinite(loss):
            raise FloatingPointError("DTES geometry loss became NaN or inf")

        loss.backward()
        opt.step()
        if wavefunction_topology and (phase_coupled_operator or magnetic_operator):
            with torch.no_grad():
                project_phase_gradient_std_(
                    model.phase,
                    target_std=phase_target_std,
                )
                phase_grad = model.phase[1:] - model.phase[:-1]
                phase_grad_std = (
                    torch.std(phase_grad)
                    if phase_grad.numel() > 1
                    else model.phase.new_tensor(0.0)
                )
                phase_pair_diff = model.phase[:, None] - model.phase[None, :]
                if phase_coupling_positive:
                    phase_weight_factor = 0.5 * (
                        1.0 + torch.cos(float(phase_coupling) * phase_pair_diff)
                    )
                else:
                    phase_weight_factor = torch.cos(
                        float(phase_coupling) * phase_pair_diff
                    )
                phase_weight_min = phase_weight_factor.min()
                phase_weight_max = phase_weight_factor.max()

        if step % 50 == 0:
            if phase_coupled_operator:
                print(
                    f"[V10.7-PHASE-COUPLED] "
                    f"eig_range={eig_range.item():.2f} "
                    f"phase_coupling={float(phase_coupling):.2f}"
                )
            elif magnetic_operator:
                print(
                    f"[V10.8-MAGNETIC] step={step} "
                    f"eig_range={eig_range.item():.2f} "
                    f"herm={hermitian_error.detach().cpu().item():.2e} "
                    f"beta={float(magnetic_beta):.2f}"
                )
            elif wavefunction_topology:
                print(
                    f"[V10.6-PHASE] step={step} "
                    f"phase_grad_std={phase_grad_std.item():.4f} "
                    f"eig_range={eig_range.item():.2f}"
                )
            elif nested_geometry:
                print(
                    f"[V10.5-NESTED] step={step} "
                    f"weights={ms_weights.detach().cpu().numpy()} "
                    f"eig_range={eig_range.item():.2f}"
                )
            elif multiscale_geometry:
                print(
                    f"[V10.4-MULTISCALE] step={step} "
                    f"range={loss_range.item():.4f} "
                    f"eig_range={eig_range.item():.2f} "
                    f"weights={ms_weights.detach().cpu().numpy()}"
                )
            elif curve_geometry:
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
                "range_loss": float(loss_range.detach().cpu()),
                "multiscale_entropy_loss": float(loss_ms_entropy.detach().cpu()),
                "wave_amp_loss": float(loss_amp_energy.detach().cpu()),
                "phase_loss": float(loss_phase.detach().cpu()),
                "phase_activity_loss": float(loss_phase_activity.detach().cpu()),
                "phase_activity_target_loss": float(
                    loss_phase_activity_target.detach().cpu()
                ),
                "phase_curvature_loss": float(loss_phase_curvature.detach().cpu()),
                "phase_grad_std": float(phase_grad_std.detach().cpu()),
                "phase_coupled_operator": bool(phase_coupled_operator),
                "phase_coupling": float(phase_coupling),
                "phase_weight_min": float(phase_weight_min.detach().cpu()),
                "phase_weight_max": float(phase_weight_max.detach().cpu()),
                "magnetic_operator": bool(magnetic_operator),
                "magnetic_beta": float(magnetic_beta),
                "hermitian_error": float(hermitian_error.detach().cpu()),
                "nodal_loss": float(loss_nodal.detach().cpu()),
                "amplitude_collapse_loss": float(loss_amp_collapse.detach().cpu()),
                "amplitude_min": float(A.min().detach().cpu()),
                "amplitude_max": float(A.max().detach().cpu()),
                "amplitude_std": float(A.std().detach().cpu()),
                "multiscale_weights": [
                    float(x) for x in ms_weights.detach().cpu()
                ],
                "nested_weights": [
                    float(x) for x in ms_weights.detach().cpu()
                ]
                if nested_geometry
                else [],
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
                "eig_range": float(eig_range.detach().cpu()),
                "zeta_range": float(zeta_range.detach().cpu()),
                "eig_min": float(eig_window.min().detach().cpu()),
                "eig_max": float(eig_window.max().detach().cpu()),
                "weights": {k: float(v) for k, v in weights.items()},
            }
        )

    with torch.no_grad():
        H, W, Z, edge_scale, ms_weights = model(
            wavefunction_topology=wavefunction_topology
        )
        eig = safe_eigvalsh_complex(H).real if magnetic_operator else safe_eigvalsh(H)
        final_hermitian_error = torch.mean(torch.abs(H - H.conj().T))
        A, psi_real, psi_imag, psi_abs = wavefunction_fields(
            model.log_amplitude,
            model.phase,
        )
        final_nodal_score = nodal_score(model.log_amplitude)
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
        "H": H.detach().cpu().numpy(),
        "V": model.V.detach().cpu().numpy(),
        "eig": eig.detach().cpu().numpy(),
        "amplitude": A.detach().cpu().numpy(),
        "phase": model.phase.detach().cpu().numpy(),
        "psi_real": psi_real.detach().cpu().numpy(),
        "psi_imag": psi_imag.detach().cpu().numpy(),
        "psi_abs": psi_abs.detach().cpu().numpy(),
        "nodal_score": final_nodal_score.detach().cpu().numpy(),
        "history": history,
        "final_weights": history[-1]["weights"] if history else dict(base_weights),
        "train_error": float(final_train_error.detach().cpu()),
        "test_error": float(final_test_error.detach().cpu()),
        "affine_penalty": float(final_affine_penalty.detach().cpu()),
        "edge_scale": float(edge_scale),
        "hermitian_error": float(final_hermitian_error.detach().cpu()),
        "multiscale_weights": [float(x) for x in ms_weights.detach().cpu()],
        "nested_weights": [float(x) for x in ms_weights.detach().cpu()]
        if nested_geometry
        else [],
    }
