from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import scipy.linalg as _scipy_linalg  # type: ignore
    import scipy.sparse as _scipy_sparse  # type: ignore
    import scipy.sparse.linalg as _scipy_sparse_linalg  # type: ignore

    _HAVE_SCIPY = True
except Exception:  # pragma: no cover
    _HAVE_SCIPY = False
    _scipy_linalg = None
    _scipy_sparse = None
    _scipy_sparse_linalg = None


_DTF = np.float64


def _nan_inf_counts(H: np.ndarray) -> tuple[int, int]:
    H = np.asarray(H)
    nan_count = int(np.isnan(H).sum())
    inf_count = int(np.isinf(H).sum())
    return nan_count, inf_count


def _symmetry_error(H: np.ndarray) -> float:
    H = np.asarray(H, dtype=_DTF)
    return float(np.linalg.norm(H - H.T, ord="fro"))


def _fro_norm(H: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(H, dtype=_DTF), ord="fro"))


def _gershgorin_bounds(H: np.ndarray) -> tuple[float, float]:
    H = np.asarray(H, dtype=_DTF)
    diag = np.diag(H)
    r = np.sum(np.abs(H), axis=1) - np.abs(diag)
    gmin = float(np.min(diag - r)) if diag.size else 0.0
    gmax = float(np.max(diag + r)) if diag.size else 0.0
    return gmin, gmax


def _power_iteration_norm(H: np.ndarray, iters: int = 30, seed: int = 42) -> float:
    H = np.asarray(H, dtype=_DTF)
    n = H.shape[0]
    if n == 0:
        return 0.0
    rng = np.random.default_rng(int(seed))
    v = rng.normal(size=(n,)).astype(_DTF)
    v /= np.linalg.norm(v) + 1e-12
    for _ in range(int(iters)):
        v = H @ v
        nv = np.linalg.norm(v) + 1e-12
        v /= nv
    Hv = H @ v
    return float(np.linalg.norm(Hv) / (np.linalg.norm(v) + 1e-12))


def stabilize_operator(
    H,
    method: str = "auto",
    eps: float = 1e-6,
    jitter: float = 1e-8,
    diagonal_shift: float = 1e-6,
    normalize: bool = True,
    project_self_adjoint: bool = True,
    condition_cap: float = 1e8,
    backend: str = "numpy",
    *,
    seed: int = 42,
) -> tuple[np.ndarray, Dict[str, Any]]:
    """
    Returns:
        H_stable, report
    """
    t0 = time.perf_counter()
    H0 = np.asarray(H)
    report: Dict[str, Any] = {
        "input_shape": list(H0.shape),
        "dtype": str(H0.dtype),
    }

    nan_count, inf_count = _nan_inf_counts(H0)
    report["nan_count"] = nan_count
    report["inf_count"] = inf_count

    Hf = np.asarray(H0, dtype=_DTF)
    report["symmetry_error_before"] = _symmetry_error(Hf)
    report["fro_norm_before"] = _fro_norm(Hf)

    Hf = np.nan_to_num(Hf, nan=0.0, posinf=1e6, neginf=-1e6)

    if project_self_adjoint:
        Hf = 0.5 * (Hf + Hf.T)
    report["symmetry_error_after"] = _symmetry_error(Hf)

    n = int(Hf.shape[0])
    if n > 0:
        Hf = Hf + float(diagonal_shift) * np.eye(n, dtype=_DTF)

    # normalization
    norm_method = "frobenius" if (method == "auto") else str(method)
    if normalize:
        if norm_method in ("auto", "frobenius"):
            denom = _fro_norm(Hf) + float(eps)
            Hf = Hf / denom
            report["normalization"] = "frobenius"
            report["normalization_scale"] = float(denom)
        elif norm_method == "trace":
            tr = float(np.trace(Hf))
            denom = abs(tr) + float(eps)
            Hf = Hf / denom
            report["normalization"] = "trace"
            report["normalization_scale"] = float(denom)
        elif norm_method == "spectral":
            est = _power_iteration_norm(Hf, iters=30, seed=seed) + float(eps)
            Hf = Hf / est
            report["normalization"] = "spectral"
            report["normalization_scale"] = float(est)
        else:
            denom = _fro_norm(Hf) + float(eps)
            Hf = Hf / denom
            report["normalization"] = "frobenius"
            report["normalization_scale"] = float(denom)
    else:
        report["normalization"] = "none"
        report["normalization_scale"] = 1.0

    # jitter (deterministic)
    if n > 0 and float(jitter) > 0.0:
        rng = np.random.default_rng(int(seed))
        diag_noise = rng.normal(size=(n,)).astype(_DTF)
        diag_noise = diag_noise / (np.std(diag_noise) + 1e-12)
        Hf = Hf + float(jitter) * np.diag(diag_noise)

    # diagnostics
    report["fro_norm_after"] = _fro_norm(Hf)
    gmin, gmax = _gershgorin_bounds(Hf)
    report["gershgorin_min"] = float(gmin)
    report["gershgorin_max"] = float(gmax)
    report["diagonal_shift"] = float(diagonal_shift)
    report["jitter"] = float(jitter)

    # condition proxy and adaptive diagonal shift
    cond_before = float("inf")
    try:
        pinv = np.linalg.pinv(Hf, rcond=1e-12)
        cond_before = float(_fro_norm(Hf) * _fro_norm(pinv))
    except Exception:
        cond_before = float("inf")
    report["condition_proxy_before"] = float(cond_before)

    extra_shift = 0.0
    if not np.isfinite(cond_before) or cond_before > float(condition_cap):
        extra_shift = float(diagonal_shift) * 10.0
        if n > 0:
            Hf = Hf + extra_shift * np.eye(n, dtype=_DTF)
        try:
            pinv2 = np.linalg.pinv(Hf, rcond=1e-12)
            cond_after = float(_fro_norm(Hf) * _fro_norm(pinv2))
        except Exception:
            cond_after = float("inf")
        report["condition_proxy_after"] = float(cond_after)
        report["diagonal_shift_effective"] = float(diagonal_shift + extra_shift)
    else:
        report["condition_proxy_after"] = float(cond_before)
        report["diagonal_shift_effective"] = float(diagonal_shift)

    report["stabilization_time_s"] = float(time.perf_counter() - t0)
    report["backend"] = str(backend)
    return Hf, report


def safe_eigh(
    H,
    k: Optional[int] = None,
    which: str = "SA",
    return_eigenvectors: bool = False,
    backend: str = "numpy",
    stabilize: bool = True,
    seed: int = 42,
) -> tuple[np.ndarray, Optional[np.ndarray], Dict[str, Any]]:
    """
    Stable eigenvalue computation.

    Returns:
        eigvals, eigvecs_or_none, report
    """
    rep: Dict[str, Any] = {}
    t0 = time.perf_counter()

    H_in = np.asarray(H)
    rep["input_shape"] = list(H_in.shape)
    rep["dtype"] = str(H_in.dtype)

    if stabilize:
        Hs, srep = stabilize_operator(H_in, seed=seed)
        rep.update({f"stabilize_{k}": v for k, v in srep.items()})
    else:
        Hs = np.asarray(H_in, dtype=_DTF)
        Hs = np.nan_to_num(Hs, nan=0.0, posinf=1e6, neginf=-1e6)
        Hs = 0.5 * (Hs + Hs.T)

    n = int(Hs.shape[0])
    rep["n"] = n
    rep["requested_k"] = None if k is None else int(k)

    eigvals: np.ndarray
    eigvecs: Optional[np.ndarray] = None

    success = False
    backend_used = "numpy"
    err_msg = ""

    try:
        if k is None:
            if return_eigenvectors:
                eigvals, eigvecs = np.linalg.eigh(Hs)
            else:
                eigvals = np.linalg.eigvalsh(Hs)
            backend_used = "numpy"
            success = True
        else:
            kk = int(k)
            if n <= 2000 or kk >= n - 1:
                if return_eigenvectors:
                    eigvals, eigvecs = np.linalg.eigh(Hs)
                else:
                    eigvals = np.linalg.eigvalsh(Hs)
                backend_used = "numpy_dense_fallback"
                success = True
            else:
                if _HAVE_SCIPY:
                    A = _scipy_sparse.csr_matrix(Hs)
                    w, v = _scipy_sparse_linalg.eigsh(A, k=kk, which=str(which))
                    eigvals = np.asarray(w, dtype=_DTF)
                    eigvals.sort()
                    eigvecs = np.asarray(v, dtype=_DTF) if return_eigenvectors else None
                    backend_used = "scipy_eigsh"
                    success = True
                else:
                    if return_eigenvectors:
                        eigvals, eigvecs = np.linalg.eigh(Hs)
                    else:
                        eigvals = np.linalg.eigvalsh(Hs)
                    backend_used = "numpy_no_scipy"
                    success = True
    except Exception as e:
        err_msg = repr(e)
        success = False

    # ladder fallback
    if not success and _HAVE_SCIPY:
        try:
            if return_eigenvectors:
                eigvals, eigvecs = _scipy_linalg.eigh(Hs, check_finite=True)
            else:
                eigvals = _scipy_linalg.eigvalsh(Hs, check_finite=True)
                eigvecs = None
            backend_used = "scipy_dense"
            success = True
        except Exception as e:
            err_msg = err_msg + " | " + repr(e)
            success = False

    if not success:
        try:
            shift = 1e-6
            Ht = Hs + shift * np.eye(n, dtype=_DTF)
            eigvals = np.linalg.eigvalsh(Ht)
            eigvecs = None
            backend_used = "numpy_shifted"
            success = True
        except Exception as e:
            err_msg = err_msg + " | " + repr(e)
            success = False

    rep["eigh_backend_used"] = backend_used
    rep["eigh_success"] = bool(success)
    if err_msg:
        rep["eigh_error"] = err_msg

    if not success:
        rep["eig_time_s"] = float(time.perf_counter() - t0)
        return np.array([], dtype=_DTF), None, rep

    eigvals = np.asarray(eigvals, dtype=_DTF).reshape(-1)
    eigvals = eigvals[np.isfinite(eigvals)]

    # repeated eigenvalue estimate (near-equality after sorting)
    if eigvals.size >= 2:
        diffs = np.diff(np.sort(eigvals))
        tol = 1e-10 + 1e-8 * float(np.std(eigvals) + 1e-12)
        rep["num_repeated_eigenvalues_estimate"] = int(np.sum(diffs < tol))
    else:
        rep["num_repeated_eigenvalues_estimate"] = 0

    rep["eig_time_s"] = float(time.perf_counter() - t0)
    return eigvals, eigvecs, rep


def safe_torch_eigh(
    H_torch,
    k: Optional[int] = None,
    which: str = "SA",
    return_eigenvectors: bool = False,
    stabilize: bool = True,
    seed: int = 42,
) -> tuple["np.ndarray | Any", Optional["np.ndarray | Any"], Dict[str, Any]]:
    """
    Safe torch eigensolve with CPU fallback; returns torch tensors on original device.
    If all torch paths fail, falls back to numpy (non-differentiable).
    """
    import torch

    rep: Dict[str, Any] = {"torch_device": str(H_torch.device), "torch_dtype": str(H_torch.dtype)}
    t0 = time.perf_counter()

    dev = H_torch.device
    H = H_torch
    try:
        H = H.to(dtype=torch.float64)
    except Exception:
        H = H.double()

    H = 0.5 * (H + H.T)
    H = torch.nan_to_num(H, nan=0.0, posinf=1e6, neginf=-1e6)

    n = int(H.shape[0])
    eye = torch.eye(n, dtype=H.dtype, device=H.device)

    success = False
    backend_used = "torch"
    err_msg = ""
    eigvals = None
    eigvecs = None

    # jitter ladder on current device
    if stabilize:
        torch.manual_seed(int(seed))
    for i in range(6):
        try:
            jj = float(1e-8) * (10.0 ** i)
            Hs = H + jj * eye
            if return_eigenvectors:
                w, v = torch.linalg.eigh(Hs)
                eigvals, eigvecs = w, v
            else:
                eigvals = torch.linalg.eigvalsh(Hs)
                eigvecs = None
            backend_used = "torch_device"
            success = True
            rep["torch_jitter_used"] = jj
            break
        except Exception as e:
            err_msg = repr(e)
            success = False

    # CPU fallback
    if not success:
        try:
            Hc = H.detach().to("cpu")
            eye_c = torch.eye(n, dtype=Hc.dtype, device=Hc.device)
            for i in range(6):
                jj = float(1e-8) * (10.0 ** i)
                Hs = Hc + jj * eye_c
                if return_eigenvectors:
                    w, v = torch.linalg.eigh(Hs)
                    eigvals, eigvecs = w.to(dev), v.to(dev)
                else:
                    eigvals = torch.linalg.eigvalsh(Hs).to(dev)
                    eigvecs = None
                backend_used = "torch_cpu"
                success = True
                rep["torch_jitter_used"] = jj
                break
        except Exception as e:
            err_msg = err_msg + " | " + repr(e)
            success = False

    # numpy fallback (non-differentiable)
    if not success:
        try:
            Hn = H.detach().to("cpu").numpy()
            w, v, nrep = safe_eigh(Hn, k=k, which=which, return_eigenvectors=return_eigenvectors, stabilize=True, seed=seed)
            eigvals = torch.tensor(w, dtype=torch.float64, device=dev)
            eigvecs = torch.tensor(v, dtype=torch.float64, device=dev) if (return_eigenvectors and v is not None) else None
            rep.update({f"numpy_{k}": vv for k, vv in nrep.items()})
            backend_used = "numpy_fallback"
            success = True
        except Exception as e:
            err_msg = err_msg + " | " + repr(e)
            success = False

    rep["eigh_backend_used"] = backend_used
    rep["eigh_success"] = bool(success)
    if err_msg:
        rep["eigh_error"] = err_msg
    rep["eig_time_s"] = float(time.perf_counter() - t0)
    return eigvals, eigvecs, rep


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=_DTF).reshape(-1)
    mu = float(np.mean(x)) if x.size else 0.0
    sd = float(np.std(x)) if x.size else 1.0
    sd = sd if sd > 1e-12 else 1.0
    return (x - mu) / sd


def stable_spectral_loss(
    H,
    target_zeros,
    k: int,
    normalize_spectrum: bool = True,
    spacing_loss: bool = True,
    *,
    seed: int = 42,
) -> tuple[float, Dict[str, Any]]:
    """
    Computes robust spectral loss.
    """
    rep: Dict[str, Any] = {}
    k = int(k)
    eigvals, _, erep = safe_eigh(H, k=None, return_eigenvectors=False, stabilize=True, seed=seed)
    rep.update({f"eigh_{kk}": vv for kk, vv in erep.items()})
    if eigvals.size == 0 or not np.all(np.isfinite(eigvals)):
        rep["spectral_loss"] = float("inf")
        rep["spacing_loss"] = float("inf")
        rep["total_loss"] = float("inf")
        return float("inf"), rep

    eigvals = np.sort(eigvals)
    tz = np.asarray(target_zeros, dtype=_DTF).reshape(-1)
    tz = tz[np.isfinite(tz)]
    if tz.size == 0:
        rep["spectral_loss"] = float("inf")
        rep["spacing_loss"] = float("inf")
        rep["total_loss"] = float("inf")
        return float("inf"), rep

    kk = min(k, eigvals.size, tz.size)
    e = eigvals[:kk]
    z = tz[:kk]

    if normalize_spectrum:
        e_n = _zscore(e)
        z_n = _zscore(z)
        rep["spectrum_normalization"] = "zscore"
    else:
        e_n = e
        z_n = z
        rep["spectrum_normalization"] = "none"

    L_spec = float(np.mean((e_n - z_n) ** 2)) if kk > 0 else float("inf")
    rep["spectral_loss"] = float(L_spec)

    L_spacing = 0.0
    if spacing_loss and kk >= 3:
        de = np.diff(np.sort(e))
        dz = np.diff(np.sort(z))
        me = float(np.mean(de)) if de.size else 1.0
        mz = float(np.mean(dz)) if dz.size else 1.0
        me = me if me > 1e-12 else 1.0
        mz = mz if mz > 1e-12 else 1.0
        de = de / me
        dz = dz / mz
        m = min(de.size, dz.size)
        L_spacing = float(np.mean((de[:m] - dz[:m]) ** 2)) if m > 0 else 0.0
    rep["spacing_loss"] = float(L_spacing)

    rep["total_loss"] = float(L_spec + L_spacing)
    return float(L_spec + L_spacing), rep

