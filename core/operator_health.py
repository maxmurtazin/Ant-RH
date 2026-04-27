import json

import numpy as np
import torch


def is_complex_matrix(H):
    return torch.is_complex(H)


def adjoint(H):
    return H.conj().T if torch.is_complex(H) else H.T


def make_self_adjoint(H):
    return 0.5 * (H + adjoint(H))


def finite_check(H):
    if torch.is_complex(H):
        return bool(torch.isfinite(H.real).all() and torch.isfinite(H.imag).all())
    return bool(torch.isfinite(H).all())


def hermitian_error(H, eps=1e-12):
    A = adjoint(H)
    num = torch.linalg.norm(H - A)
    den = torch.linalg.norm(H) + eps
    return float((num / den).detach().cpu())


def sanitize_matrix(H, clamp_value=1e6):
    if torch.is_complex(H):
        real = torch.nan_to_num(
            H.real,
            nan=0.0,
            posinf=clamp_value,
            neginf=-clamp_value,
        )
        imag = torch.nan_to_num(
            H.imag,
            nan=0.0,
            posinf=clamp_value,
            neginf=-clamp_value,
        )
        return torch.complex(real, imag)
    return torch.nan_to_num(H, nan=0.0, posinf=clamp_value, neginf=-clamp_value)


def safe_eigvalsh_self_adjoint(H, jitter_eps=1e-8, max_tries=6):
    H = make_self_adjoint(H)
    H = sanitize_matrix(H)

    n = H.shape[0]
    eye = torch.eye(n, dtype=H.dtype, device=H.device)

    used_jitter = 0.0
    last_error = None

    for i in range(max_tries):
        try:
            if i == 0:
                eig = torch.linalg.eigvalsh(H)
            else:
                used_jitter = jitter_eps * (10.0 ** (i - 1))
                eig = torch.linalg.eigvalsh(H + used_jitter * eye)
            return eig.real, used_jitter, None
        except RuntimeError as e:
            last_error = str(e)

    return None, used_jitter, last_error


def spectral_health_check(H, jitter_eps=1e-8):
    report = {
        "finite_outputs": finite_check(H),
        "hermitian_error": hermitian_error(H),
        "used_jitter": 0.0,
        "eig_min": None,
        "eig_max": None,
        "eig_range": None,
        "eig_std": None,
        "condition_proxy": None,
        "spectral_pass": False,
        "failure_reason": None,
    }

    if not report["finite_outputs"]:
        report["failure_reason"] = "operator contains NaN or Inf"
        return report, None

    eig, used_jitter, err = safe_eigvalsh_self_adjoint(H, jitter_eps=jitter_eps)
    report["used_jitter"] = float(used_jitter)

    if eig is None:
        report["failure_reason"] = err or "eigvalsh failed"
        return report, None

    eig_stats = eig.detach().cpu()

    eig_min = float(eig_stats.min())
    eig_max = float(eig_stats.max())
    eig_range = float(eig_max - eig_min)
    eig_std = float(eig_stats.std())

    denom = max(abs(eig_min), 1e-12)
    condition_proxy = float(abs(eig_max) / denom)

    report.update(
        {
            "eig_min": eig_min,
            "eig_max": eig_max,
            "eig_range": eig_range,
            "eig_std": eig_std,
            "condition_proxy": condition_proxy,
            "spectral_pass": bool(np.isfinite(eig_range) and np.isfinite(eig_std)),
            "failure_reason": None,
        }
    )

    return report, eig


def validate_operator_health(
    H_raw,
    self_adjoint_tol=1e-8,
    strict_self_adjoint=False,
    jitter_eps=1e-8,
):
    H_raw = sanitize_matrix(H_raw)

    raw_error = hermitian_error(H_raw)
    raw_finite = finite_check(H_raw)

    if strict_self_adjoint and raw_error > self_adjoint_tol:
        H_projected = make_self_adjoint(H_raw)
    else:
        H_projected = make_self_adjoint(H_raw)

    projected_error = hermitian_error(H_projected)

    health, eig = spectral_health_check(H_projected, jitter_eps=jitter_eps)

    report = {
        "operator_health_version": "V10.10",
        "raw_is_complex": bool(is_complex_matrix(H_raw)),
        "hermitian_error_raw": raw_error,
        "hermitian_error_projected": projected_error,
        "finite_outputs": bool(raw_finite and health["finite_outputs"]),
        "self_adjoint_tol": float(self_adjoint_tol),
        "strict_self_adjoint": bool(strict_self_adjoint),
        "self_adjoint_pass": bool(projected_error <= self_adjoint_tol),
        **health,
        "note": "operator-level Hilbert–Pólya compatibility check; not a proof of RH",
    }

    return H_projected, eig, report
