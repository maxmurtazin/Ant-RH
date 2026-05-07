#!/usr/bin/env python3
"""
V14.6 — Spectral-GAN Artin Operator Proposal Engine

Computational evidence only; not a proof of RH.

IMPORTANT:
  - GAN is used ONLY as a proposal engine for Artin words.
  - Final acceptance is deterministic via hard spectral gates/metrics.
  - This module is experimental search infrastructure, not an RH proof.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pandas as pd  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit("This script requires pandas. Please install pandas and retry.") from e


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from validation import residue_diagnostics as rd  # noqa: E402

# Optional safe eigensolver
try:
    from core.spectral_stabilization import safe_eigh as _safe_eigh  # type: ignore

    _HAVE_SAFE_EIGH = True
except Exception:
    _safe_eigh = None
    _HAVE_SAFE_EIGH = False

# Torch is required by spec but must have fallback.
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset

    _HAVE_TORCH = True
except Exception:
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    Dataset = object  # type: ignore
    DataLoader = object  # type: ignore
    _HAVE_TORCH = False


# ----------------------------
# Utilities
# ----------------------------


def _resolve(p: str) -> Path:
    pp = Path(p)
    if not pp.is_absolute():
        pp = Path(ROOT) / pp
    return pp


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        v = float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def json_sanitize(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): json_sanitize(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_sanitize(v) for v in x]
    if x is None:
        return None
    if isinstance(x, (int, str, bool)):
        return x
    if isinstance(x, float):
        return float(x) if math.isfinite(x) else None
    try:
        v = float(x)
        return float(v) if math.isfinite(v) else None
    except Exception:
        return str(x)


def write_text(path: Path, s: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(s, encoding="utf-8")


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def format_seconds(sec: float) -> str:
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _find_pdflatex() -> Optional[str]:
    w = shutil.which("pdflatex")
    if w:
        return w
    for p in ("/Library/TeX/texbin/pdflatex",):
        if Path(p).is_file():
            return p
    return None


def try_pdflatex(tex_path: Path, out_dir: Path, pdf_name: str) -> Tuple[bool, str]:
    exe = _find_pdflatex()
    if not exe:
        return False, "pdflatex not found"
    try:
        r = subprocess.run(
            [exe, "-interaction=nonstopmode", f"-output-directory={out_dir.resolve()}", tex_path.name],
            cwd=str(out_dir.resolve()),
            capture_output=True,
            text=True,
            timeout=240,
        )
        ok = r.returncode == 0 and (out_dir / pdf_name).is_file()
        msg = "ok" if ok else (r.stderr[-500:] if r.stderr else r.stdout[-500:])
        return ok, msg
    except Exception as e:
        return False, repr(e)


def set_deterministic(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    if _HAVE_TORCH:
        torch.manual_seed(int(seed))
        torch.use_deterministic_algorithms(False)


def pick_device(device: str) -> str:
    if device == "cpu":
        return "cpu"
    if device == "cuda":
        if _HAVE_TORCH and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if device == "mps":
        if _HAVE_TORCH and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    # auto
    if _HAVE_TORCH and torch.cuda.is_available():
        return "cuda"
    if _HAVE_TORCH and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ----------------------------
# Artin word parsing / encoding
# ----------------------------


def parse_word(word_str: str) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    s = str(word_str or "").strip()
    if not s:
        return out
    for tok in s.split():
        tok = tok.strip()
        if not tok:
            continue
        # sigma_12^-3
        try:
            left, pstr = tok.split("^")
            istr = left.split("_")[1]
            out.append((int(istr), int(pstr)))
        except Exception:
            continue
    return out


def word_to_string(word: List[Tuple[int, int]]) -> str:
    return " ".join([f"sigma_{int(i)}^{int(p)}" for i, p in word])


def simplify_word(word: List[Tuple[int, int]], *, max_power: int, max_word_len: int) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for i, p in word:
        i = int(i)
        p = int(max(-max_power, min(max_power, int(p))))
        if p == 0:
            continue
        if out and out[-1][0] == i:
            pp = int(out[-1][1] + p)
            pp = int(max(-max_power, min(max_power, pp)))
            out[-1] = (i, pp)
            if out[-1][1] == 0:
                out.pop()
            continue
        out.append((i, p))
        if len(out) >= int(max_word_len):
            break
    return out


def clamp_word_to_dim(word: List[Tuple[int, int]], dim: int, max_power: int, max_word_len: int) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for i, p in word:
        ii = int(max(1, min(int(dim) - 1, int(i))))
        pp = int(max(-max_power, min(max_power, int(p))))
        if pp == 0:
            continue
        out.append((ii, pp))
    out = simplify_word(out, max_power=max_power, max_word_len=max_word_len)
    if len(out) < 2 and out:
        # pad with a neighbor generator
        i0 = out[0][0]
        i1 = int(max(1, min(dim - 1, i0 + 1)))
        out.append((i1, int(out[0][1])))
    return out


def power_values(max_power: int) -> List[int]:
    return [p for p in range(-max_power, max_power + 1) if p != 0]


# ----------------------------
# Stabilized Artin operator builder (numpy)
# ----------------------------


def rotation_block(theta: float) -> np.ndarray:
    c = float(math.cos(theta))
    s = float(math.sin(theta))
    return np.asarray([[c, -s], [s, c]], dtype=np.float64)


def embed_2x2(dim: int, idx0: int, B: np.ndarray) -> np.ndarray:
    n = int(dim)
    G = np.eye(n, dtype=np.float64)
    i = int(idx0)
    if i < 0 or i + 1 >= n:
        return G
    G[i : i + 2, i : i + 2] = np.asarray(B, dtype=np.float64)
    return G


def hermitian_part(M: np.ndarray) -> np.ndarray:
    return 0.5 * (M + M.T)


def op_norm_fro(A: np.ndarray) -> float:
    return float(np.linalg.norm(A, ord="fro"))


def normalize_operator(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = op_norm_fro(A)
    return A / max(float(n), float(eps))


def remove_trace(H: np.ndarray) -> np.ndarray:
    n = int(H.shape[0])
    tr = float(np.trace(H)) / float(max(1, n))
    return H - tr * np.eye(n, dtype=H.dtype)


def safe_eigvalsh(H: np.ndarray, *, seed: int) -> Optional[np.ndarray]:
    if not np.isfinite(H).all():
        return None
    try:
        if _HAVE_SAFE_EIGH and _safe_eigh is not None:
            w, _, _rep = _safe_eigh(np.asarray(H, dtype=np.float64), k=None, return_eigenvectors=False, stabilize=True, seed=int(seed))
            w = np.asarray(w, dtype=np.float64).reshape(-1)
            w = w[np.isfinite(w)]
            w.sort()
            return w
        w = np.linalg.eigvalsh(np.asarray(H, dtype=np.float64))
        w = np.asarray(w, dtype=np.float64).reshape(-1)
        w = w[np.isfinite(w)]
        w.sort()
        return w
    except Exception:
        return None


def normalize_spectral_radius(H: np.ndarray, target_radius: float, eps: float = 1e-12) -> Tuple[np.ndarray, float]:
    w = safe_eigvalsh(H, seed=42)
    if w is None or w.size == 0:
        return H, float("nan")
    r = float(max(abs(float(w[0])), abs(float(w[-1]))))
    if not (math.isfinite(r) and r > eps):
        return H, r
    s = float(target_radius) / r
    return H * s, float(target_radius)


def build_stabilized_artin_operator(dim: int, word: List[Tuple[int, int]], target_radius: Optional[float] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Requirements:
      - deterministic local block generator for sigma_i^p
      - A = 0.5*(G + G.T), normalize by Fro norm
      - H = sum_k c_k * sign_power * A_i,p, c_k = 1/sqrt(k+1)
      - symmetrize, trace removal, spectral radius normalization to dim/4
      - diagnostics returned
    """
    n = int(dim)
    eps = 1e-12
    H = np.zeros((n, n), dtype=np.float64)
    theta_base = math.pi / 8.0

    nan_flag = False
    for k, (gi, p) in enumerate(word):
        i = int(max(1, min(n - 1, int(gi))))
        pp = int(p)
        # local deterministic block
        max_i = max(1, n - 1)
        theta = float(theta_base) * float(i) / float(max_i)
        B = rotation_block(float(pp) * theta)
        G = embed_2x2(n, i - 1, B)
        A = hermitian_part(G)
        A = normalize_operator(A, eps=eps)
        ck = 1.0 / math.sqrt(float(k) + 1.0)
        sign_power = 1.0 if pp > 0 else -1.0
        H = H + float(ck) * float(sign_power) * A
        if not np.isfinite(H).all():
            nan_flag = True
            break

    H = hermitian_part(H)
    H = remove_trace(H)

    tr_abs = float(abs(float(np.trace(H))))
    fro = float(op_norm_fro(H))
    if target_radius is None:
        target_radius = float(max(4.0, n / 4.0))
    H, rad = normalize_spectral_radius(H, float(target_radius), eps=eps)

    stable = (not nan_flag) and np.isfinite(H).all() and math.isfinite(rad) and (rad > 0.0) and (fro > 0.0)
    diag = {
        "spectral_radius": float(rad),
        "fro_norm": float(fro),
        "trace_abs": float(tr_abs),
        "nan_flag": bool(nan_flag),
        "stable_flag": bool(stable),
    }
    return H, diag


def unfold_to_mean_spacing_one(x: np.ndarray) -> np.ndarray:
    return rd.unfold_to_mean_spacing_one(np.asarray(x, dtype=np.float64).reshape(-1))


# ----------------------------
# Spectral metrics
# ----------------------------


def make_L_grid(L_min: float, L_max: float, n_L: int) -> np.ndarray:
    L_min = float(L_min)
    L_max = float(L_max)
    n_L = int(n_L)
    if not (math.isfinite(L_min) and math.isfinite(L_max)) or n_L < 2:
        return np.asarray([1.0, 2.0], dtype=np.float64)
    if L_max <= L_min:
        L_max = L_min + 1.0
    return np.linspace(L_min, L_max, n_L, dtype=np.float64)


def number_variance_curve(levels_unfolded: np.ndarray, L_grid: np.ndarray) -> np.ndarray:
    x = np.asarray(levels_unfolded, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size < 8:
        return np.full_like(L_grid, np.nan, dtype=np.float64)
    x = np.sort(x)
    out = np.full_like(L_grid, np.nan, dtype=np.float64)
    t_candidates = x[:-1]
    for i, L in enumerate(np.asarray(L_grid, dtype=np.float64).reshape(-1)):
        if not (math.isfinite(float(L)) and float(L) > 0.0):
            continue
        left = np.searchsorted(x, t_candidates, side="left")
        right = np.searchsorted(x, t_candidates + float(L), side="right")
        counts = (right - left).astype(np.float64)
        if counts.size < 8:
            continue
        out[i] = float(np.var(counts))
    return out


def sigma2_poisson(L: np.ndarray) -> np.ndarray:
    return np.asarray(L, dtype=np.float64)


def sigma2_gue_asymptotic(L: np.ndarray) -> np.ndarray:
    L = np.asarray(L, dtype=np.float64)
    gamma = 0.5772156649015329
    return (1.0 / (math.pi**2)) * (np.log(np.maximum(1e-12, 2.0 * math.pi * L)) + gamma + 1.0)


def curve_l2(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    if aa.size == 0 or bb.size == 0 or aa.size != bb.size:
        return float("nan")
    m = np.isfinite(aa) & np.isfinite(bb)
    if not m.any():
        return float("nan")
    return float(np.sqrt(np.mean((aa[m] - bb[m]) ** 2)))


def spacing_stats(levels_unfolded: np.ndarray) -> Dict[str, float]:
    x = np.asarray(levels_unfolded, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size < 8:
        return {"spacing_mean": float("nan"), "spacing_std": float("nan"), "spacing_cv": float("nan")}
    x = np.sort(x)
    s = np.diff(x)
    s = s[np.isfinite(s) & (s > 0)]
    if s.size < 4:
        return {"spacing_mean": float("nan"), "spacing_std": float("nan"), "spacing_cv": float("nan")}
    mu = float(np.mean(s))
    sd = float(np.std(s))
    cv = float(sd / max(1e-12, mu))
    return {"spacing_mean": mu, "spacing_std": sd, "spacing_cv": cv}


def wigner_poisson_proxies(levels_unfolded: np.ndarray) -> Dict[str, float]:
    """
    Lightweight proxies:
      - wigner_proxy: lower CV (closer to Wigner surmise-ish) -> higher proxy
      - poisson_proxy: higher CV -> higher proxy
    """
    st = spacing_stats(levels_unfolded)
    cv = st["spacing_cv"]
    if not math.isfinite(cv):
        return {"wigner_proxy": float("nan"), "poisson_proxy": float("nan")}
    wigner = float(1.0 / (1.0 + cv))
    poisson = float(cv / (1.0 + cv))
    return {"wigner_proxy": wigner, "poisson_proxy": poisson}


def nv_features(levels_unfolded: np.ndarray, target_unfolded: np.ndarray, L_grid: np.ndarray) -> Dict[str, float]:
    op = number_variance_curve(levels_unfolded, L_grid)
    tg = number_variance_curve(target_unfolded, L_grid)
    rmse = float(curve_l2(op, tg))
    # slope/curvature via quadratic fit on finite points
    m = np.isfinite(op) & np.isfinite(tg)
    if not m.any():
        return {
            "nv_rmse": rmse,
            "nv_slope": float("nan"),
            "nv_curvature": float("nan"),
            "poisson_like_fraction": 1.0,
            "anti_poisson_score": float("nan"),
        }
    Lm = np.asarray(L_grid, dtype=np.float64)[m]
    y = (op - tg)[m]
    try:
        # y ~ a L^2 + b L + c
        a, b, _c = np.polyfit(Lm, y, deg=2)
        slope = float(b)
        curvature = float(a)
    except Exception:
        slope = float("nan")
        curvature = float("nan")
    dP = float(curve_l2(op, sigma2_poisson(L_grid)))
    dG = float(curve_l2(op, sigma2_gue_asymptotic(L_grid)))
    poisson_like = bool(math.isfinite(dP) and math.isfinite(dG) and dP < dG)
    anti_poisson = float(dP - dG) if (math.isfinite(dP) and math.isfinite(dG)) else float("nan")
    return {
        "nv_rmse": rmse,
        "nv_slope": slope,
        "nv_curvature": curvature,
        "poisson_like_fraction": 1.0 if poisson_like else 0.0,
        "anti_poisson_score": anti_poisson,
    }


def complexity(word: List[Tuple[int, int]]) -> float:
    if not word:
        return 0.0
    # count tokens + avg abs power
    L = float(len(word))
    p = float(np.mean([abs(int(pp)) for (_i, pp) in word]))
    return float(L + 0.25 * p)


def active_argument_metrics(
    levels: np.ndarray,
    target_levels: np.ndarray,
    windows: List[Tuple[float, float]],
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    errs_norm: List[float] = []
    errs_raw: List[float] = []
    active = 0
    both = 0
    for (a, b) in windows:
        n_op = int(rd.count_in_window(levels, float(a), float(b)))
        n_tg = int(rd.count_in_window(target_levels, float(a), float(b)))
        active_window = (n_op > 0) or (n_tg > 0)
        if not active_window:
            continue
        active += 1
        if (n_op > 0) and (n_tg > 0):
            both += 1
        err = float(abs(n_op - n_tg))
        err_norm = float(err / float(max(1, n_tg)))
        errs_raw.append(err)
        errs_norm.append(err_norm)
        rows.append(
            {
                "window_a": float(a),
                "window_b": float(b),
                "N_operator": int(n_op),
                "N_target": int(n_tg),
                "N_error": float(err),
                "N_error_norm": float(err_norm),
                "active_window": bool(active_window),
            }
        )
    med_norm = float(np.median(np.asarray(errs_norm, dtype=np.float64))) if errs_norm else 1.0
    mean_norm = float(np.mean(np.asarray(errs_norm, dtype=np.float64))) if errs_norm else 1.0
    support_overlap = float(both) / float(max(1, active))
    return {"active_arg_error": med_norm, "active_arg_error_mean": mean_norm, "support_overlap": support_overlap}, rows


def residue_metrics(
    levels: np.ndarray,
    target_levels: np.ndarray,
    windows: List[Tuple[float, float]],
    eta: float,
    n_contour_points: int,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    errs: List[float] = []
    leaks: List[float] = []
    for (a, b) in windows:
        n_op = int(rd.count_in_window(levels, float(a), float(b)))
        n_tg = int(rd.count_in_window(target_levels, float(a), float(b)))
        if (n_op == 0) and (n_tg == 0):
            continue
        I_op = rd.residue_proxy_count(levels, float(a), float(b), eta=float(eta), n_contour_points=int(n_contour_points))
        I_tg = rd.residue_proxy_count(target_levels, float(a), float(b), eta=float(eta), n_contour_points=int(n_contour_points))
        err = abs(float(I_op.real) - float(I_tg.real)) / max(1.0, abs(float(I_tg.real)))
        leak = float(abs(float(I_op.imag)))
        errs.append(float(err))
        leaks.append(leak)
        rows.append(
            {
                "window_a": float(a),
                "window_b": float(b),
                "I_operator_real": float(I_op.real),
                "I_operator_imag": float(I_op.imag),
                "I_target_real": float(I_tg.real),
                "I_target_imag": float(I_tg.imag),
                "residue_count_error": float(err),
                "residue_imag_leak": float(leak),
            }
        )
    med_err = float(np.median(np.asarray(errs, dtype=np.float64))) if errs else 1.0
    med_leak = float(np.median(np.asarray(leaks, dtype=np.float64))) if leaks else 0.0
    return {"residue_error": med_err, "imag_leak": med_leak}, rows


def trace_metrics(
    levels: np.ndarray,
    target_levels: np.ndarray,
    windows: List[Tuple[float, float]],
) -> Dict[str, float]:
    errs: List[float] = []
    for (a, b) in windows:
        n_op = int(rd.count_in_window(levels, float(a), float(b)))
        n_tg = int(rd.count_in_window(target_levels, float(a), float(b)))
        if (n_op == 0) and (n_tg == 0):
            continue
        c = 0.5 * (float(a) + float(b))
        for s in (0.5, 1.0, 2.0, 4.0):
            Sop = rd.trace_formula_proxy(levels, center=float(c), sigma=float(s))
            Stg = rd.trace_formula_proxy(target_levels, center=float(c), sigma=float(s))
            if not (math.isfinite(Sop) and math.isfinite(Stg)):
                continue
            tr = abs(math.log1p(max(0.0, float(Sop))) - math.log1p(max(0.0, float(Stg))))
            errs.append(float(tr))
    med = float(np.median(np.asarray(errs, dtype=np.float64))) if errs else 1.0
    return {"trace_error": med}


def null_separation_proxy(
    J: float,
    baseline_random_J: Optional[float],
    baseline_rejected_J: Optional[float],
    baseline_ablation_J: Optional[float],
) -> Dict[str, Any]:
    beats_random = bool(baseline_random_J is not None and math.isfinite(baseline_random_J) and J <= float(baseline_random_J))
    beats_rejected = bool(baseline_rejected_J is not None and math.isfinite(baseline_rejected_J) and J <= float(baseline_rejected_J))
    beats_ablation = bool(baseline_ablation_J is not None and math.isfinite(baseline_ablation_J) and J <= float(baseline_ablation_J))
    sep = float("nan")
    if baseline_random_J is not None and math.isfinite(baseline_random_J):
        sep = float(baseline_random_J - J)
    return {
        "beats_random": beats_random,
        "beats_rejected": beats_rejected,
        "beats_ablation": beats_ablation,
        "null_separation": sep,
    }


# ----------------------------
# Torch models (Generator + Critic)
# ----------------------------


class ArtinWordGenerator(nn.Module):
    def __init__(self, *, max_dim: int, latent_dim: int, hidden_dim: int, max_word_len: int, max_power: int, n_dims: int, n_modes: int):
        super().__init__()
        self.max_dim = int(max_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.max_word_len = int(max_word_len)
        self.max_power = int(max_power)
        self.n_dims = int(n_dims)
        self.n_modes = int(n_modes)
        self.dim_emb = nn.Embedding(self.n_dims, 32)
        self.mode_emb = nn.Embedding(self.n_modes, 16)
        self.fc1 = nn.Linear(self.latent_dim + 32 + 16, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        # per-position heads
        self.gen_head = nn.Linear(self.hidden_dim, self.max_word_len * self.max_dim)
        self.pow_head = nn.Linear(self.hidden_dim, self.max_word_len * (2 * self.max_power))
        self.stop_head = nn.Linear(self.hidden_dim, self.max_word_len)

    def forward(self, z: torch.Tensor, dim_id: torch.Tensor, mode_id: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = torch.cat([z, self.dim_emb(dim_id), self.mode_emb(mode_id)], dim=-1)
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        gen_logits = self.gen_head(h).view(-1, self.max_word_len, self.max_dim)
        pow_logits = self.pow_head(h).view(-1, self.max_word_len, 2 * self.max_power)
        stop_logits = self.stop_head(h).view(-1, self.max_word_len)
        return gen_logits, pow_logits, stop_logits


class SpectralCritic(nn.Module):
    def __init__(self, *, in_dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        logit = self.out(h).squeeze(-1)
        score = torch.sigmoid(logit)
        return score, logit


class WordCorpusDataset(Dataset):
    """
    Supervised corpus for generator (teacher forcing).
    Encodes words into (gen_ids, pow_ids, stop) sequences of length max_word_len.
      gen_ids in [0..max_dim-1] (we store generator index directly, clipped)
      pow_ids in [0..2*max_power-1] representing [-max_power..-1,1..max_power]
      stop in {0,1} indicates end at position k (1 at first stop).
    """

    def __init__(self, examples: List[Dict[str, Any]], *, max_dim: int, max_word_len: int, max_power: int, dims: List[int], modes: List[str]):
        self.examples = examples
        self.max_dim = int(max_dim)
        self.max_word_len = int(max_word_len)
        self.max_power = int(max_power)
        self.dims = dims
        self.modes = modes
        self.pow_vals = power_values(self.max_power)
        self.pow_to_id = {p: i for i, p in enumerate(self.pow_vals)}

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        ex = self.examples[idx]
        dim = int(ex["dim"])
        mode = str(ex.get("mode", "gan_only"))
        word = clamp_word_to_dim(list(ex["word"]), dim, self.max_power, self.max_word_len)
        gen_ids = np.zeros((self.max_word_len,), dtype=np.int64)
        pow_ids = np.zeros((self.max_word_len,), dtype=np.int64)
        stop = np.zeros((self.max_word_len,), dtype=np.float32)
        L = min(len(word), self.max_word_len)
        for k in range(L):
            gi, pw = word[k]
            gen_ids[k] = int(max(0, min(self.max_dim - 1, int(gi))))
            pw = int(max(-self.max_power, min(self.max_power, int(pw))))
            if pw == 0:
                pw = 1
            if pw not in self.pow_to_id:
                # map to nearest
                pw = int(max(-self.max_power, min(self.max_power, pw)))
                if pw == 0:
                    pw = 1
                if pw not in self.pow_to_id:
                    pw = self.pow_vals[0]
            pow_ids[k] = int(self.pow_to_id[pw])
        if L > 0 and L < self.max_word_len:
            stop[L - 1] = 1.0
        elif L >= self.max_word_len:
            stop[self.max_word_len - 1] = 1.0

        dim_id = int(self.dims.index(dim)) if dim in self.dims else 0
        mode_id = int(self.modes.index(mode)) if mode in self.modes else 0
        return {
            "dim": dim,
            "dim_id": dim_id,
            "mode_id": mode_id,
            "gen_ids": gen_ids,
            "pow_ids": pow_ids,
            "stop": stop,
        }


def sample_word_from_generator(
    G: ArtinWordGenerator,
    *,
    dim: int,
    dim_id: int,
    mode_id: int,
    latent_dim: int,
    max_word_len: int,
    max_power: int,
    device: str,
    rng: np.random.Generator,
    min_len: int = 2,
) -> List[Tuple[int, int]]:
    z = torch.from_numpy(rng.normal(size=(1, latent_dim)).astype(np.float32)).to(device)
    did = torch.tensor([dim_id], dtype=torch.long, device=device)
    mid = torch.tensor([mode_id], dtype=torch.long, device=device)
    with torch.no_grad():
        gen_logits, pow_logits, stop_logits = G(z, did, mid)
        gen_logits = gen_logits[0].detach().cpu().numpy()  # [L, max_dim]
        pow_logits = pow_logits[0].detach().cpu().numpy()  # [L, 2*max_power]
        stop_logits = stop_logits[0].detach().cpu().numpy()  # [L]

    pow_vals = power_values(max_power)
    word: List[Tuple[int, int]] = []
    stop_at = max_word_len
    for k in range(max_word_len):
        # stop if sigmoid(stop)>0.5, but keep at least min_len
        p_stop = 1.0 / (1.0 + math.exp(-float(stop_logits[k])))
        if (k >= min_len) and (p_stop > 0.5):
            stop_at = k
            break
        # mask generator ids to [1..dim-1]
        gl = gen_logits[k].copy()
        gl[:1] = -1e9
        gl[dim:] = -1e9
        gi = int(np.argmax(gl))
        pi = int(np.argmax(pow_logits[k]))
        pw = int(pow_vals[pi])
        word.append((gi, pw))
    word = clamp_word_to_dim(word, dim, max_power, max_word_len)
    return word


# ----------------------------
# Bootstrap / previous artifacts loading (best-effort)
# ----------------------------


def try_load_v14_5_best_words(v14_5_dir: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    p = v14_5_dir / "v14_5_best_candidates.csv"
    if not p.is_file():
        return out
    try:
        df = pd.read_csv(p)
        for _, r in df.iterrows():
            dim = int(r.get("dim", 0))
            w = parse_word(str(r.get("best_word", "")))
            if not w:
                continue
            out.append({"dim": dim, "mode": "gan_only", "word": w, "source": "v14_5_best"})
    except Exception:
        return out
    return out


def try_load_v13o10_words(candidate_ranking_csv: Path, max_rows: int = 200) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not candidate_ranking_csv.is_file():
        return out
    try:
        df = pd.read_csv(candidate_ranking_csv)
        # attempt common columns: dim, word / artin_word / candidate_word
        cols = [c for c in df.columns if "word" in c.lower()]
        dim_col = "dim" if "dim" in df.columns else None
        if not cols:
            return out
        word_col = cols[0]
        df2 = df.head(int(max_rows)).copy()
        for _, r in df2.iterrows():
            dim = int(r.get(dim_col, 0)) if dim_col else 0
            w = parse_word(str(r.get(word_col, "")))
            if not w:
                continue
            if dim <= 0:
                dim = 64
            out.append({"dim": dim, "mode": "gan_only", "word": w, "source": "v13o10"})
    except Exception:
        return out
    return out


def try_load_semantic_memory(v14_4_dir: Path, v14_5_dir: Path) -> List[Dict[str, Any]]:
    # For V14.6 we only need it to bias semantic hybrid; minimal motif set is enough.
    mem: List[Dict[str, Any]] = []
    p1 = v14_5_dir / "v14_5_semantic_pheromones.jsonl"
    p0 = v14_4_dir / "v14_4_semantic_pheromones.jsonl"
    for p in (p1, p0):
        if not p.is_file():
            continue
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if isinstance(rec, dict):
                    mem.append(rec)
        except Exception:
            continue
    if not mem:
        mem = [
            {"type": "avoid", "motif": "avoid_band_collapse", "weight": 0.8, "confidence": 0.85, "dim": "all", "source": "init"},
            {"type": "prefer", "motif": "prefer_entropy", "weight": 0.6, "confidence": 0.75, "dim": "all", "source": "init"},
            {"type": "avoid", "motif": "avoid_repeated_bigrams", "weight": 0.7, "confidence": 0.8, "dim": "all", "source": "init"},
        ]
    return mem[:800]


def semantic_score_simple(word: List[Tuple[int, int]], dim: int, memory: List[Dict[str, Any]]) -> float:
    # same motif logic as V14.5 (simplified)
    # prefer_entropy (entropy>=0.6), avoid_band_collapse (band small), avoid_repeated_bigrams
    gens = [int(i) for i, _p in word]
    if not gens:
        return 0.0
    c = Counter(gens)
    total = float(sum(c.values()))
    ps = np.asarray([v / total for v in c.values()], dtype=np.float64)
    H = float(-np.sum(ps * np.log(np.maximum(1e-12, ps))))
    Hn = float(H / max(1e-12, math.log(max(2.0, float(dim - 1)))))
    band = float(max(gens) - min(gens) + 1)
    rep_bi = repeated_bigram_fraction(word)
    score = 0.0
    for rec in memory:
        if not isinstance(rec, dict):
            continue
        motif = str(rec.get("motif", "")).strip()
        typ = str(rec.get("type", "")).strip().lower()
        w = abs(safe_float(rec.get("weight", 0.0), 0.0)) * max(0.0, min(1.0, safe_float(rec.get("confidence", 1.0), 1.0)))
        if w <= 0:
            continue
        match = False
        if motif == "prefer_entropy":
            match = Hn >= 0.6
        elif motif == "avoid_band_collapse":
            match = band <= max(3.0, 0.08 * float(dim))
        elif motif == "avoid_repeated_bigrams":
            match = rep_bi >= 0.15
        else:
            match = False
        if not match:
            continue
        if typ == "prefer":
            score += w
        elif typ in ("avoid", "artifact", "caution"):
            score -= w
    return float(score)


def repeated_bigram_fraction(word: List[Tuple[int, int]]) -> float:
    if len(word) < 3:
        return 0.0
    bigrams = [(word[i][0], word[i][1], word[i + 1][0], word[i + 1][1]) for i in range(len(word) - 1)]
    c = Counter(bigrams)
    rep = sum(v - 1 for v in c.values() if v > 1)
    return float(rep / max(1, len(bigrams)))

#
# ----------------------------
# V14.6b — Real null-control Stage E (baseline pool loading)
# ----------------------------
#


def _read_csv_best_effort(path: Path) -> Optional[pd.DataFrame]:
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def load_v13o14_baselines(v13o14_dir: Path) -> Tuple[Dict[int, Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    """
    Returns:
      - per-dim summary dict with keys best_random_J/best_rejected_J/best_ablation_J/primary_best_J (if available)
      - baseline pool rows (dim, group, source, J, meta...)
      - missing reasons
    """
    reasons: List[str] = []
    per_dim: Dict[int, Dict[str, Any]] = {}
    pool: List[Dict[str, Any]] = []

    p_null = v13o14_dir / "v13o14_null_comparisons.csv"
    p_mode = v13o14_dir / "v13o14_candidate_mode_summary.csv"

    df_null = _read_csv_best_effort(p_null)
    if df_null is None:
        reasons.append(f"missing {p_null.name}")
    else:
        for _, r in df_null.iterrows():
            d = int(r.get("dim", 0))
            if d <= 0:
                continue
            per_dim.setdefault(d, {})
            for k in ("primary_best_J", "best_random_J", "best_rejected_J", "best_ablation_J"):
                if k in df_null.columns:
                    per_dim[d][k] = safe_float(r.get(k, float("nan")))
            # also push pool items if present
            if "best_random_J" in df_null.columns:
                pool.append({"dim": d, "group": "random_controls_best", "source": "v13o14_null_comparisons", "J": safe_float(r.get("best_random_J"))})
            if "best_rejected_J" in df_null.columns:
                pool.append({"dim": d, "group": "rejected_word_seed17", "source": "v13o14_null_comparisons", "J": safe_float(r.get("best_rejected_J"))})
            if "best_ablation_J" in df_null.columns:
                pool.append({"dim": d, "group": "ablations_best", "source": "v13o14_null_comparisons", "J": safe_float(r.get("best_ablation_J"))})
            if "primary_best_J" in df_null.columns:
                pool.append({"dim": d, "group": "primary_word_seed6", "source": "v13o14_null_comparisons", "J": safe_float(r.get("primary_best_J"))})

    df_mode = _read_csv_best_effort(p_mode)
    if df_mode is None:
        reasons.append(f"missing {p_mode.name}")
    else:
        for _, r in df_mode.iterrows():
            d = int(r.get("dim", 0))
            if d <= 0:
                continue
            group = str(r.get("word_group", "")).strip() or str(r.get("candidate_classification", "unknown"))
            # prefer best_all_J if present else best_simple_J
            J = float("nan")
            if "best_all_J" in df_mode.columns:
                J = safe_float(r.get("best_all_J", float("nan")))
            elif "best_simple_J" in df_mode.columns:
                J = safe_float(r.get("best_simple_J", float("nan")))
            pool.append(
                {
                    "dim": d,
                    "group": group,
                    "source": "v13o14_candidate_mode_summary",
                    "J": J,
                    "is_random_baseline": bool(r.get("is_random_baseline", False)),
                    "is_ablation": bool(r.get("is_ablation", False)),
                    "is_rejected_word": bool(r.get("is_rejected_word", False)),
                    "is_primary": bool(r.get("is_primary", False)),
                    "support_overlap_fraction": safe_float(r.get("support_overlap_fraction", float("nan"))),
                    "poisson_like_fraction": safe_float(r.get("poisson_like_fraction", float("nan"))),
                }
            )

    return per_dim, pool, reasons


def load_v14_2_baselines(v14_2_dir: Path) -> Tuple[Dict[int, float], List[Dict[str, Any]], List[str]]:
    reasons: List[str] = []
    best_by_dim: Dict[int, float] = {}
    pool: List[Dict[str, Any]] = []
    p_best = v14_2_dir / "v14_2_best_candidates.csv"
    df = _read_csv_best_effort(p_best)
    if df is None:
        reasons.append(f"missing {p_best.name}")
        return best_by_dim, pool, reasons
    # expected columns: dim, J_v14_2
    if "dim" not in df.columns:
        reasons.append(f"invalid {p_best.name}: missing dim column")
        return best_by_dim, pool, reasons
    Jcol = "J_v14_2" if "J_v14_2" in df.columns else ("J_total" if "J_total" in df.columns else None)
    if Jcol is None:
        reasons.append(f"invalid {p_best.name}: missing J column (J_v14_2/J_total)")
        return best_by_dim, pool, reasons
    for d, sub in df.groupby(df["dim"].astype(int)):
        Js = pd.to_numeric(sub[Jcol], errors="coerce").astype(float).to_numpy()
        Js = Js[np.isfinite(Js)]
        if Js.size == 0:
            continue
        best = float(np.min(Js))
        best_by_dim[int(d)] = best
        pool.append({"dim": int(d), "group": "best_v14_2", "source": p_best.name, "J": best})
    return best_by_dim, pool, reasons


def load_v14_5_baselines(v14_5_dir: Path) -> Tuple[Dict[int, float], List[Dict[str, Any]], List[str]]:
    reasons: List[str] = []
    best_by_dim: Dict[int, float] = {}
    pool: List[Dict[str, Any]] = []
    p_best = v14_5_dir / "v14_5_best_candidates.csv"
    df = _read_csv_best_effort(p_best)
    if df is None:
        reasons.append(f"missing {p_best.name}")
        return best_by_dim, pool, reasons
    if "dim" not in df.columns:
        reasons.append(f"invalid {p_best.name}: missing dim")
        return best_by_dim, pool, reasons
    Jcol = "best_J" if "best_J" in df.columns else ("J_total" if "J_total" in df.columns else None)
    if Jcol is None:
        reasons.append(f"invalid {p_best.name}: missing best_J/J_total")
        return best_by_dim, pool, reasons
    for d, sub in df.groupby(df["dim"].astype(int)):
        Js = pd.to_numeric(sub[Jcol], errors="coerce").astype(float).to_numpy()
        Js = Js[np.isfinite(Js)]
        if Js.size == 0:
            continue
        best = float(np.min(Js))
        best_by_dim[int(d)] = best
        pool.append({"dim": int(d), "group": "best_v14_5", "source": p_best.name, "J": best})
    return best_by_dim, pool, reasons


def baseline_stats_vs_gan(best_gan_J: float, pool_Js: List[float]) -> Tuple[float, float]:
    """
    Returns (null_zscore, null_percentile) for GAN among pool_Js.
      - zscore: (gan - mean)/std  (lower is better)
      - percentile: fraction of pool values <= gan (lower means GAN is better)
    If insufficient data: (nan, nan)
    """
    xs = [float(x) for x in pool_Js if math.isfinite(float(x))]
    if len(xs) < 3 or not math.isfinite(float(best_gan_J)):
        return float("nan"), float("nan")
    mu = float(np.mean(xs))
    sd = float(np.std(xs))
    z = float((best_gan_J - mu) / sd) if sd > 1e-12 else float("nan")
    pct = float(np.mean([1.0 if x <= float(best_gan_J) else 0.0 for x in xs]))
    return z, pct


# ----------------------------
# Critic feature vector builder
# ----------------------------


FEATURE_NAMES = [
    "spacing_mean",
    "spacing_std",
    "spacing_cv",
    "wigner_proxy",
    "poisson_proxy",
    "nv_rmse",
    "nv_slope",
    "nv_curvature",
    "active_arg_error",
    "residue_error",
    "imag_leak",
    "trace_error",
    "support_overlap",
    "complexity",
    "null_separation",
]


def features_to_vec(feat: Dict[str, Any]) -> np.ndarray:
    x = []
    for k in FEATURE_NAMES:
        v = safe_float(feat.get(k, float("nan")))
        if not math.isfinite(v):
            v = 0.0
        x.append(float(v))
    return np.asarray(x, dtype=np.float32)


def compute_spectral_features(
    *,
    dim: int,
    word: List[Tuple[int, int]],
    levels: np.ndarray,
    target_levels: np.ndarray,
    windows: List[Tuple[float, float]],
    L_grid: np.ndarray,
    args: argparse.Namespace,
    null_sep: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns:
      - feature dict
      - argument_count rows
      - residue rows
      - nv_curve rows (operator+target)
      - trace rows (empty; trace is aggregated only)
    """
    st = spacing_stats(levels)
    wp = wigner_poisson_proxies(levels)
    nvf = nv_features(levels, target_levels, L_grid)

    argm, arg_rows = active_argument_metrics(levels, target_levels, windows)
    resm, res_rows = residue_metrics(levels, target_levels, windows, eta=float(args.eta), n_contour_points=int(args.n_contour_points))
    trm = trace_metrics(levels, target_levels, windows)

    comp = complexity(word)
    feat = {
        **st,
        **wp,
        **nvf,
        "active_arg_error": float(argm["active_arg_error"]),
        "residue_error": float(resm["residue_error"]),
        "imag_leak": float(resm["imag_leak"]),
        "trace_error": float(trm["trace_error"]),
        "support_overlap": float(argm["support_overlap"]),
        "complexity": float(comp),
        "null_separation": float(null_sep),
        "poisson_like_fraction": float(nvf["poisson_like_fraction"]),
        "anti_poisson_score": float(nvf["anti_poisson_score"]),
    }

    nv_rows: List[Dict[str, Any]] = []
    op_nv = number_variance_curve(levels, L_grid)
    tg_nv = number_variance_curve(target_levels, L_grid)
    for L, y in zip(L_grid, op_nv):
        nv_rows.append({"kind": "operator", "L": float(L), "Sigma2": safe_float(y, float("nan"))})
    for L, y in zip(L_grid, tg_nv):
        nv_rows.append({"kind": "target", "L": float(L), "Sigma2": safe_float(y, float("nan"))})

    trace_rows: List[Dict[str, Any]] = []
    return feat, arg_rows, res_rows, nv_rows, trace_rows


# ----------------------------
# GAN-ish training (bootstrap)
# ----------------------------


def poisson_synthetic_levels(n: int, rng: np.random.Generator) -> np.ndarray:
    # Poisson spacings: exponential(1)
    spac = rng.exponential(scale=1.0, size=max(16, int(n)))
    x = np.cumsum(spac)
    return unfold_to_mean_spacing_one(x)


def shuffled_levels(levels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    x = np.asarray(levels, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size < 16:
        return x
    s = np.diff(np.sort(x))
    rng.shuffle(s)
    y = np.cumsum(np.concatenate([[x.min()], s]))
    return unfold_to_mean_spacing_one(y)


def train_models(
    *,
    G: Optional[ArtinWordGenerator],
    D: Optional[SpectralCritic],
    corpus: List[Dict[str, Any]],
    dims: List[int],
    generator_modes: List[str],
    target_by_dim: Dict[int, np.ndarray],
    windows: List[Tuple[float, float]],
    L_grid: np.ndarray,
    args: argparse.Namespace,
    device: str,
    rng: np.random.Generator,
) -> List[Dict[str, Any]]:
    """
    Training is bootstrap due to lack of labels.

    - Generator: supervised language-modeling on corpus if available (teacher forcing).
    - Critic: binary classifier on spectral feature vectors:
        positives: target features + features of best corpus words (evaluated)
        negatives: Poisson synthetic + shuffled target + unstable/random operators
    """
    hist: List[Dict[str, Any]] = []
    if not _HAVE_TORCH or G is None or D is None:
        return hist

    # Generator supervised training on corpus
    max_dim = int(G.max_dim)
    if corpus:
        ds = WordCorpusDataset(corpus, max_dim=max_dim, max_word_len=int(args.max_word_len), max_power=int(args.max_power), dims=dims, modes=generator_modes)
        dl = DataLoader(ds, batch_size=int(args.batch_size), shuffle=True, drop_last=False)
        optG = torch.optim.Adam(G.parameters(), lr=2e-4)
    else:
        dl = None
        optG = None

    # Critic training data builder
    optD = torch.optim.Adam(D.parameters(), lr=2e-4)

    # precompute target feature vectors per dim
    target_feat_by_dim: Dict[int, np.ndarray] = {}
    for d in dims:
        tgt = target_by_dim[int(d)]
        # Use target as both op and target for feature extraction baseline
        feat, *_ = compute_spectral_features(dim=int(d), word=[(1, 1), (2, 1)], levels=tgt, target_levels=tgt, windows=windows, L_grid=L_grid, args=args, null_sep=float("nan"))
        target_feat_by_dim[int(d)] = features_to_vec(feat)

    for epoch in range(1, int(args.num_epochs) + 1):
        # --- Train critic on bootstrapped batch ---
        X: List[np.ndarray] = []
        y: List[int] = []
        # positives: target feats
        for d in dims:
            X.append(target_feat_by_dim[int(d)])
            y.append(1)
        # negatives: poisson + shuffled per dim
        for d in dims:
            n = max(64, int(d))
            pois = poisson_synthetic_levels(n, rng)
            shuf = shuffled_levels(target_by_dim[int(d)], rng)
            feat_p, *_ = compute_spectral_features(dim=int(d), word=[(1, 1), (2, 1)], levels=pois, target_levels=target_by_dim[int(d)], windows=windows, L_grid=L_grid, args=args, null_sep=float("nan"))
            feat_s, *_ = compute_spectral_features(dim=int(d), word=[(1, 1), (2, 1)], levels=shuf, target_levels=target_by_dim[int(d)], windows=windows, L_grid=L_grid, args=args, null_sep=float("nan"))
            X.append(features_to_vec(feat_p))
            y.append(0)
            X.append(features_to_vec(feat_s))
            y.append(0)

        # add a few corpus-word evaluated positives/negatives (cheap subset)
        bestJ_epoch: Dict[int, float] = {int(d): float("inf") for d in dims}
        if corpus:
            subset = corpus[: min(24, len(corpus))]
            for ex in subset:
                d = int(ex["dim"])
                word = clamp_word_to_dim(list(ex["word"]), d, int(args.max_power), int(args.max_word_len))
                H, diag = build_stabilized_artin_operator(d, word)
                w = safe_eigvalsh(H, seed=int(args.seed + 17 * epoch + d))
                if (w is None) or (w.size < 8) or (not diag["stable_flag"]):
                    continue
                lv = unfold_to_mean_spacing_one(w)
                # align to target quantiles for windowing
                tgt = target_by_dim[int(d)]
                try:
                    oq = np.quantile(lv, [0.1, 0.9])
                    tq = np.quantile(tgt, [0.1, 0.9])
                    a = float(max(1e-12, tq[1] - tq[0]) / max(1e-12, oq[1] - oq[0]))
                    b = float(tq[0] - a * oq[0])
                    lv = a * lv + b
                except Exception:
                    pass
                feat, *_ = compute_spectral_features(dim=d, word=word, levels=lv, target_levels=tgt, windows=windows, L_grid=L_grid, args=args, null_sep=float("nan"))
                J_proxy = float(feat.get("nv_rmse", 1.0)) + float(feat.get("active_arg_error", 1.0)) + float(feat.get("residue_error", 1.0)) + float(feat.get("poisson_like_fraction", 1.0))
                bestJ_epoch[int(d)] = min(bestJ_epoch[int(d)], J_proxy)
                X.append(features_to_vec(feat))
                # bootstrapped label: positive-ish if not poisson-like and reasonable arg/res
                lab = 1 if (float(feat.get("poisson_like_fraction", 1.0)) < 0.5 and float(feat.get("support_overlap", 0.0)) > 0.25) else 0
                y.append(int(lab))

        X_t = torch.from_numpy(np.stack(X, axis=0)).to(device)
        y_t = torch.tensor(y, dtype=torch.float32, device=device)
        D.train()
        optD.zero_grad(set_to_none=True)
        score, logit = D(X_t)
        D_loss = F.binary_cross_entropy(score, y_t)
        D_loss.backward()
        optD.step()

        # --- Train generator (supervised) if corpus exists ---
        G_loss_v = float("nan")
        if dl is not None and optG is not None:
            G.train()
            total = 0.0
            n_batches = 0
            for batch in dl:
                z = torch.randn((batch["gen_ids"].shape[0], int(args.latent_dim)), device=device)
                did = batch["dim_id"].to(device, dtype=torch.long)
                mid = batch["mode_id"].to(device, dtype=torch.long)
                gen_ids = batch["gen_ids"].to(device, dtype=torch.long)
                pow_ids = batch["pow_ids"].to(device, dtype=torch.long)
                stop = batch["stop"].to(device, dtype=torch.float32)
                optG.zero_grad(set_to_none=True)
                gen_logits, pow_logits, stop_logits = G(z, did, mid)
                # generator ids: ignore padding positions by using stop signal; keep it simple: full CE
                gen_loss = F.cross_entropy(gen_logits.view(-1, max_dim), gen_ids.view(-1))
                pow_loss = F.cross_entropy(pow_logits.view(-1, 2 * int(args.max_power)), pow_ids.view(-1))
                stop_loss = F.binary_cross_entropy_with_logits(stop_logits.view(-1), stop.view(-1))
                loss = gen_loss + 0.5 * pow_loss + 0.2 * stop_loss
                loss.backward()
                optG.step()
                total += float(loss.detach().cpu().item())
                n_batches += 1
                # keep epoch light
                if n_batches >= 6:
                    break
            G_loss_v = float(total / max(1, n_batches))

        # Monitoring stats
        D.eval()
        with torch.no_grad():
            cr = float(torch.mean(score[y_t > 0.5]).detach().cpu().item()) if (y_t > 0.5).any() else float("nan")
            cf = float(torch.mean(score[y_t < 0.5]).detach().cpu().item()) if (y_t < 0.5).any() else float("nan")

        bestJ_mean = float(np.mean([bestJ_epoch[int(d)] for d in dims if math.isfinite(bestJ_epoch[int(d)])])) if dims else float("nan")
        hist.append(
            {
                "epoch": int(epoch),
                "dim": "all",
                "D_loss": float(D_loss.detach().cpu().item()),
                "G_loss": float(G_loss_v) if math.isfinite(G_loss_v) else float("nan"),
                "critic_real_mean": cr,
                "critic_fake_mean": cf,
                "best_J_epoch": bestJ_mean,
                "best_reward_epoch": float("nan"),
            }
        )
        if epoch == 1 or epoch % max(1, int(args.progress_every)) == 0 or epoch == int(args.num_epochs):
            print(f"[V14.6][train] epoch={epoch}/{int(args.num_epochs)} D_loss={float(D_loss.detach().cpu().item()):.4f} G_loss={G_loss_v if math.isfinite(G_loss_v) else float('nan'):.4f}", flush=True)

    return hist


# ----------------------------
# Candidate generation + evaluation
# ----------------------------


def rank_based_rewards(rows: List[Dict[str, Any]]) -> None:
    rows.sort(key=lambda r: float(r["J_total"]))
    N = len(rows)
    for rank_i, r in enumerate(rows):
        base = float((N - 1 - rank_i) / max(1, N - 1))
        rr = float(base ** float(r.get("_rank_reward_power", 1.0)))
        r["final_reward"] = float(min(max(rr, 1e-6), 1.0))


def compute_losses_and_gates(
    *,
    stable_flag: bool,
    feat: Dict[str, Any],
    word_len: int,
    args: argparse.Namespace,
    null_flags: Dict[str, Any],
    critic_score: float,
) -> Tuple[Dict[str, float], Dict[str, bool], str]:
    # Losses
    spacing_loss = float(feat.get("spacing_cv", 1.0)) if math.isfinite(safe_float(feat.get("spacing_cv"))) else 1.0
    nv_loss = float(feat.get("nv_rmse", 1.0)) if math.isfinite(safe_float(feat.get("nv_rmse"))) else 1.0
    arg_loss = float(feat.get("active_arg_error", 1.0))
    residue_loss = float(feat.get("residue_error", 1.0))
    trace_loss = float(feat.get("trace_error", 1.0))
    anti_poisson_loss = float(1.0 if float(feat.get("poisson_like_fraction", 1.0)) > float(args.poisson_like_max) else 0.0)
    null_loss = float(0.0 if bool(null_flags.get("beats_random", False)) else 1.0) if (null_flags.get("beats_random") is not None) else float("nan")
    complexity_loss = float(max(0.0, float(word_len) - float(args.complexity_max)) / max(1.0, float(args.complexity_max)))
    gan_prior = float(1.0 - float(critic_score)) if math.isfinite(float(critic_score)) else 1.0

    L = {
        "spacing_loss": spacing_loss,
        "nv_loss": nv_loss,
        "arg_loss": arg_loss,
        "residue_loss": residue_loss,
        "trace_loss": trace_loss,
        "anti_poisson_loss": anti_poisson_loss,
        "null_loss": 0.0 if math.isnan(null_loss) else float(null_loss),
        "complexity_loss": complexity_loss,
        "gan_prior_loss": gan_prior,
    }

    # Hard gates
    G1 = bool(stable_flag)
    G2 = bool(float(feat.get("support_overlap", 0.0)) >= float(args.support_overlap_min))
    G3 = bool(float(feat.get("active_arg_error", 1.0)) <= float(args.active_error_margin))
    G4 = bool(float(feat.get("residue_error", 1.0)) <= float(args.residue_error_margin))
    G5 = bool(float(feat.get("imag_leak", 1.0)) <= float(args.imag_leak_margin))
    G6 = bool(float(feat.get("trace_error", 1.0)) <= float(args.trace_error_margin))
    G7 = bool(float(feat.get("poisson_like_fraction", 1.0)) <= float(args.poisson_like_max))
    G8 = bool(null_flags.get("beats_random", False)) if (null_flags.get("beats_random") is not None) else False
    G9 = bool(null_flags.get("beats_rejected", False)) if (null_flags.get("beats_rejected") is not None) else False
    G10 = bool(null_flags.get("beats_ablation", False)) if (null_flags.get("beats_ablation") is not None) else False
    # "not GAN overfit": conservative => fail if critic very high but Poisson-like
    G11 = not (math.isfinite(float(critic_score)) and float(critic_score) > 0.9 and float(feat.get("poisson_like_fraction", 1.0)) >= 0.95)
    G12 = bool(int(word_len) <= int(args.complexity_max))

    gates = {
        "G1_stable_operator": G1,
        "G2_support_overlap_ok": G2,
        "G3_active_argument_ok": G3,
        "G4_residue_error_ok": G4,
        "G5_imag_leak_ok": G5,
        "G6_trace_proxy_ok": G6,
        "G7_not_poisson_like": G7,
        "G8_beats_random_baseline": G8,
        "G9_beats_rejected_word_if_available": G9,
        "G10_beats_ablation_if_available": G10,
        "G11_not_gan_overfit": bool(G11),
        "G12_word_complexity_ok": G12,
    }
    all_gate = bool(G1 and G2 and G3 and G4 and G5 and G6 and G7 and G11 and G12)
    gates["all_gate_pass"] = all_gate

    # Classification
    if not G1:
        cls = "UNSTABLE_OPERATOR"
    elif all_gate:
        cls = "ALL_GATE_PASS"
    elif not G7:
        cls = "POISSON_LIKE_FAIL"
    elif not (G2 and G3):
        cls = "SUPPORT_OR_ARGUMENT_FAIL"
    elif not (G4 and G5):
        cls = "RESIDUE_FAIL"
    else:
        cls = "PARTIAL_PASS"
    return L, gates, cls


def main() -> None:
    ap = argparse.ArgumentParser(description="V14.6 Spectral-GAN Artin Operator Proposal Engine (computational only).")
    ap.add_argument("--true_levels_csv", type=str, default="runs/v13o9_true_unfolded_spectra_from_v13o8/v13o9_unfolded_levels.csv")
    ap.add_argument("--zeros_csv", type=str, default="runs/zeros_100_400_precise.csv")
    ap.add_argument("--v14_5_dir", type=str, default="runs/v14_5_semantic_pheromone_anticollapse")
    ap.add_argument("--v14_4_dir", type=str, default="runs/v14_4_semantic_pheromone_dtes_ant_agents")
    ap.add_argument("--v14_2_dir", type=str, default="runs/v14_2_stabilized_artin_operator_search")
    ap.add_argument("--v13o14_dir", type=str, default="runs/v13o14_transport_null_controls")
    ap.add_argument("--v13o10_candidate_ranking", type=str, default="runs/v13o10_true_spectra_candidate_rescue/v13o10_candidate_ranking.csv")
    ap.add_argument("--out_dir", type=str, default="runs/v14_6_spectral_gan_artin_operator")
    ap.add_argument("--dims", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--num_epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_candidates", type=int, default=512)
    ap.add_argument("--latent_dim", type=int, default=64)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--max_word_len", type=int, default=32)
    ap.add_argument("--max_power", type=int, default=3)
    ap.add_argument("--generator_modes", type=str, nargs="+", default=["gan_only", "gan_aco_hybrid", "gan_semantic_hybrid"])
    ap.add_argument("--window_min", type=float, default=100.0)
    ap.add_argument("--window_max", type=float, default=400.0)
    ap.add_argument("--window_size", type=float, default=40.0)
    ap.add_argument("--window_stride", type=float, default=20.0)
    ap.add_argument("--L_min", type=float, default=0.5)
    ap.add_argument("--L_max", type=float, default=16.0)
    ap.add_argument("--n_L", type=int, default=64)
    ap.add_argument("--eta", type=float, default=0.15)
    ap.add_argument("--n_contour_points", type=int, default=256)
    ap.add_argument("--support_overlap_min", type=float, default=0.5)
    ap.add_argument("--active_error_margin", type=float, default=0.25)
    ap.add_argument("--residue_error_margin", type=float, default=0.25)
    ap.add_argument("--imag_leak_margin", type=float, default=0.05)
    ap.add_argument("--poisson_like_max", type=float, default=0.5)
    ap.add_argument("--complexity_max", type=int, default=32)
    # gate margin for trace proxy (kept explicit; also accept --trace_margin alias)
    ap.add_argument("--trace_error_margin", type=float, default=0.5)
    ap.add_argument("--trace_margin", type=float, default=None)
    ap.add_argument("--lambda_spec", type=float, default=1.0)
    ap.add_argument("--lambda_spacing", type=float, default=1.0)
    ap.add_argument("--lambda_nv", type=float, default=1.0)
    ap.add_argument("--lambda_arg", type=float, default=1.0)
    ap.add_argument("--lambda_res", type=float, default=1.0)
    ap.add_argument("--lambda_trace", type=float, default=0.5)
    ap.add_argument("--lambda_ap", type=float, default=2.0)
    ap.add_argument("--lambda_null", type=float, default=1.0)
    ap.add_argument("--lambda_complexity", type=float, default=0.05)
    ap.add_argument("--lambda_gan", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=146)
    ap.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "mps", "cuda"])
    ap.add_argument("--n_jobs", type=int, default=1)
    ap.add_argument("--progress_every", type=int, default=20)
    ap.add_argument("--write_pdf", action="store_true")
    args = ap.parse_args()
    if args.trace_margin is not None:
        # alias support
        args.trace_error_margin = float(args.trace_margin)

    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    set_deterministic(int(args.seed))
    rng = np.random.default_rng(int(args.seed))
    device = pick_device(str(args.device))

    warnings: List[str] = []
    if not _HAVE_TORCH:
        warnings.append("torch unavailable -> fallback random proposal (critic_score NaN).")
    else:
        if device != "cpu":
            warnings.append(f"using device={device}")
        else:
            warnings.append("using device=cpu (no cuda/mps available)")

    # windows + L grid
    windows = rd.make_windows(args.window_min, args.window_max, args.window_size, args.window_stride)
    L_grid = make_L_grid(args.L_min, args.L_max, args.n_L)
    if not windows:
        raise SystemExit("No windows created; check window_* args.")

    # load targets
    dims = [int(d) for d in args.dims]
    target_by_dim: Dict[int, np.ndarray] = {}
    df_levels, lvl_warns = rd.load_true_levels_csv(_resolve(args.true_levels_csv), dims_keep=dims)
    warnings.extend([f"true_levels_csv: {w}" for w in lvl_warns])
    if df_levels is not None and not df_levels.empty:
        try:
            df = df_levels.copy()
            df["dim"] = pd.to_numeric(df["dim"], errors="coerce").astype("Int64")
            for c in ("target_group", "source"):
                df[c] = df[c].astype(str).str.strip()
            df["unfolded_level"] = pd.to_numeric(df["unfolded_level"], errors="coerce").astype(float)
            for d in dims:
                sub = df[(df["dim"].astype(int) == int(d)) & (df["target_group"] == "real_zeta") & (df["source"] == "target")]
                if not sub.empty:
                    x = np.sort(sub["unfolded_level"].to_numpy(dtype=np.float64))
                    x = x[np.isfinite(x)]
                    if x.size >= 8:
                        target_by_dim[int(d)] = x
        except Exception as e:
            warnings.append(f"true_levels_csv parse failed: {e!r}")

    zeros_raw, zeros_warns = rd.load_zeros_csv(_resolve(args.zeros_csv))
    warnings.extend([f"zeros_csv: {w}" for w in zeros_warns])
    zeros_unfolded = unfold_to_mean_spacing_one(zeros_raw)
    for d in dims:
        if int(d) not in target_by_dim:
            target_by_dim[int(d)] = zeros_unfolded.copy()
            warnings.append(f"dim={d}: target fallback to zeros_csv (real_zeta missing)")

    # load previous artifacts (best-effort)
    v14_5_dir = _resolve(args.v14_5_dir)
    v14_4_dir = _resolve(args.v14_4_dir)
    v13o10_rank = _resolve(args.v13o10_candidate_ranking)
    corpus: List[Dict[str, Any]] = []
    corpus.extend(try_load_v14_5_best_words(v14_5_dir))
    corpus.extend(try_load_v13o10_words(v13o10_rank))
    # keep only dims we run
    corpus = [c for c in corpus if int(c.get("dim", 0)) in dims and c.get("word")]
    if not corpus:
        warnings.append("no corpus words found -> generator training is skipped (random init sampling).")

    semantic_memory = try_load_semantic_memory(v14_4_dir, v14_5_dir)

    # baselines (best-effort): if missing, leave None and mark gates false
    baseline_random_J: Optional[float] = None
    baseline_rejected_J: Optional[float] = None
    baseline_ablation_J: Optional[float] = None

    # build models
    max_dim = int(max(dims)) if dims else 256
    generator_modes = [str(m) for m in args.generator_modes]
    if _HAVE_TORCH:
        G = ArtinWordGenerator(
            max_dim=max_dim,
            latent_dim=int(args.latent_dim),
            hidden_dim=int(args.hidden_dim),
            max_word_len=int(args.max_word_len),
            max_power=int(args.max_power),
            n_dims=len(dims),
            n_modes=len(generator_modes),
        ).to(device)
        D = SpectralCritic(in_dim=len(FEATURE_NAMES), hidden_dim=int(args.hidden_dim)).to(device)
    else:
        G = None
        D = None

    # train
    t0 = time.perf_counter()
    gan_history = train_models(G=G, D=D, corpus=corpus, dims=dims, generator_modes=generator_modes, target_by_dim=target_by_dim, windows=windows, L_grid=L_grid, args=args, device=device, rng=rng)

    # generate/evaluate candidates
    generated_candidates: List[Dict[str, Any]] = []
    spectral_features_rows: List[Dict[str, Any]] = []
    residue_rows_all: List[Dict[str, Any]] = []
    argument_rows_all: List[Dict[str, Any]] = []
    nv_rows_all: List[Dict[str, Any]] = []
    candidate_ranking: List[Dict[str, Any]] = []

    critic_scores_cache: Dict[Tuple[int, str, int], float] = {}

    def critic_score(feat: Dict[str, Any]) -> float:
        if not _HAVE_TORCH or D is None:
            return float("nan")
        x = torch.from_numpy(features_to_vec(feat)[None, :]).to(device)
        D.eval()
        with torch.no_grad():
            s, _log = D(x)
        return float(s.detach().cpu().numpy().reshape(-1)[0])

    # Proposal hybrids:
    # - gan_only: sample from G or random
    # - gan_aco_hybrid: occasionally mutate tokens using v14_5 pheromone summary (if exists) else random mix
    # - gan_semantic_hybrid: resample words with low semantic score / high collapse penalty
    top_pheromone_tokens: Dict[int, List[Tuple[int, int]]] = {}
    p_pher = v14_5_dir / "v14_5_numeric_pheromone_summary.csv"
    if p_pher.is_file():
        try:
            dfp = pd.read_csv(p_pher)
            for d in dims:
                sub = dfp[dfp["dim"].astype(int) == int(d)].copy() if "dim" in dfp.columns else dfp.copy()
                # keep top tokens
                toks = []
                for _, r in sub.sort_values("pheromone", ascending=False).head(80).iterrows():
                    toks.append((int(r["generator"]), int(r["power"])))
                if toks:
                    top_pheromone_tokens[int(d)] = toks
        except Exception:
            pass

    n_per_mode = max(1, int(args.num_candidates) // max(1, len(generator_modes)))
    candidate_id = 0
    for d_idx, d in enumerate(dims):
        tgt = target_by_dim[int(d)]
        for mode in generator_modes:
            mode_id = int(generator_modes.index(mode))
            rows_mode: List[Dict[str, Any]] = []
            for j in range(n_per_mode):
                candidate_id += 1
                # sample word
                if _HAVE_TORCH and G is not None:
                    w = sample_word_from_generator(
                        G,
                        dim=int(d),
                        dim_id=int(d_idx),
                        mode_id=int(mode_id),
                        latent_dim=int(args.latent_dim),
                        max_word_len=int(args.max_word_len),
                        max_power=int(args.max_power),
                        device=device,
                        rng=rng,
                        min_len=2,
                    )
                else:
                    # fallback random
                    L = int(rng.integers(2, int(args.max_word_len) + 1))
                    w = [(int(rng.integers(1, int(d))), int(rng.choice(power_values(int(args.max_power))))) for _ in range(L)]
                    w = clamp_word_to_dim(w, int(d), int(args.max_power), int(args.max_word_len))

                # hybrid adjustments
                if mode == "gan_aco_hybrid" and int(d) in top_pheromone_tokens and top_pheromone_tokens[int(d)]:
                    toks = top_pheromone_tokens[int(d)]
                    # replace ~30% positions with high-pheromone tokens
                    w2 = list(w)
                    for k in range(len(w2)):
                        if rng.random() < 0.3:
                            w2[k] = toks[int(rng.integers(0, len(toks)))]
                    w = clamp_word_to_dim(w2, int(d), int(args.max_power), int(args.max_word_len))

                if mode == "gan_semantic_hybrid":
                    # resample up to 2 times if semantic score is poor or collapse is high
                    for _try in range(2):
                        sem = semantic_score_simple(w, int(d), semantic_memory)
                        gens = [ii for ii, _pp in w]
                        band = (max(gens) - min(gens) + 1) if gens else 0
                        collapse = (band <= max(3.0, 0.08 * float(d)))
                        if sem >= 0.0 and (not collapse):
                            break
                        # mild mutation
                        if w:
                            k = int(rng.integers(0, len(w)))
                            w[k] = (int(rng.integers(1, int(d))), int(rng.choice(power_values(int(args.max_power)))))
                            w = clamp_word_to_dim(w, int(d), int(args.max_power), int(args.max_word_len))

                wstr = word_to_string(w)
                word_len = int(len(w))

                # build operator
                H, diag = build_stabilized_artin_operator(int(d), w, target_radius=float(max(4.0, int(d) / 4.0)))
                stable_flag = bool(diag["stable_flag"])
                w_eval = safe_eigvalsh(H, seed=int(args.seed + 1000 * candidate_id + 7 * d))
                if (w_eval is None) or (w_eval.size < 8) or (not stable_flag):
                    feat = {k: float("nan") for k in FEATURE_NAMES}
                    feat.update(
                        {
                            "active_arg_error": 1.0,
                            "residue_error": 1.0,
                            "imag_leak": 1.0,
                            "trace_error": 1.0,
                            "support_overlap": 0.0,
                            "complexity": float(word_len),
                            "null_separation": float("nan"),
                            "poisson_like_fraction": 1.0,
                        }
                    )
                    arg_rows = []
                    res_rows = []
                    nv_rows = []
                else:
                    lv = unfold_to_mean_spacing_one(w_eval)
                    # align to target quantiles for windows
                    try:
                        oq = np.quantile(lv, [0.1, 0.9])
                        tq = np.quantile(tgt, [0.1, 0.9])
                        a = float(max(1e-12, tq[1] - tq[0]) / max(1e-12, oq[1] - oq[0]))
                        b = float(tq[0] - a * oq[0])
                        lv = a * lv + b
                    except Exception:
                        pass

                    # J baseline separation proxy uses placeholder baselines (None -> NaN)
                    null_sep = float("nan")
                    feat, arg_rows, res_rows, nv_rows, _tr = compute_spectral_features(
                        dim=int(d), word=w, levels=lv, target_levels=tgt, windows=windows, L_grid=L_grid, args=args, null_sep=null_sep
                    )

                cs = critic_score(feat)
                # losses + J_total
                null_flags = null_separation_proxy(
                    J=0.0,
                    baseline_random_J=baseline_random_J,
                    baseline_rejected_J=baseline_rejected_J,
                    baseline_ablation_J=baseline_ablation_J,
                )
                losses, gates, cls = compute_losses_and_gates(
                    stable_flag=stable_flag,
                    feat=feat,
                    word_len=word_len,
                    args=args,
                    null_flags=null_flags,
                    critic_score=cs if math.isfinite(cs) else 0.5,
                )
                J_total = (
                    float(args.lambda_spacing) * float(losses["spacing_loss"])
                    + float(args.lambda_nv) * float(losses["nv_loss"])
                    + float(args.lambda_arg) * float(losses["arg_loss"])
                    + float(args.lambda_res) * float(losses["residue_loss"])
                    + float(args.lambda_trace) * float(losses["trace_loss"])
                    + float(args.lambda_ap) * float(losses["anti_poisson_loss"])
                    + float(args.lambda_null) * float(losses["null_loss"])
                    + float(args.lambda_complexity) * float(losses["complexity_loss"])
                    + float(args.lambda_gan) * float(losses["gan_prior_loss"])
                )
                if not stable_flag:
                    J_total = float(1e6)
                J_total = float(min(max(J_total, 0.0), 1e6))

                # update null proxy with J_total now
                null_flags = null_separation_proxy(J=J_total, baseline_random_J=baseline_random_J, baseline_rejected_J=baseline_rejected_J, baseline_ablation_J=baseline_ablation_J)
                # recompute gates with real null flags
                losses, gates, cls = compute_losses_and_gates(
                    stable_flag=stable_flag,
                    feat=feat,
                    word_len=word_len,
                    args=args,
                    null_flags=null_flags,
                    critic_score=cs if math.isfinite(cs) else 0.5,
                )

                # store rows
                gen_row = {
                    "dim": int(d),
                    "mode": str(mode),
                    "candidate_id": int(candidate_id),
                    "word": wstr,
                    "word_len": int(word_len),
                    "stable_flag": bool(stable_flag),
                    "critic_score": float(cs) if math.isfinite(cs) else float("nan"),
                    "J_total": float(J_total),
                    "final_reward": float("nan"),  # filled by rank-based within (dim,mode)
                    **losses,
                    "all_gate_pass": bool(gates["all_gate_pass"]),
                }
                # required column names mapping
                gen_row["spacing_loss"] = float(losses["spacing_loss"])
                gen_row["anti_poisson_loss"] = float(losses["anti_poisson_loss"])
                gen_row["null_loss"] = float(losses["null_loss"])
                gen_row["complexity_loss"] = float(losses["complexity_loss"])

                rows_mode.append(gen_row)
                generated_candidates.append(gen_row)

                spectral_features_rows.append(
                    {
                        "dim": int(d),
                        "mode": str(mode),
                        "candidate_id": int(candidate_id),
                        "word": wstr,
                        **{k: feat.get(k, float("nan")) for k in FEATURE_NAMES},
                        "poisson_like_fraction": feat.get("poisson_like_fraction", float("nan")),
                    }
                )

                for ar in arg_rows:
                    argument_rows_all.append(
                        {
                            "dim": int(d),
                            "mode": str(mode),
                            "candidate_id": int(candidate_id),
                            "word": wstr,
                            **ar,
                        }
                    )
                for rr in res_rows:
                    residue_rows_all.append(
                        {
                            "dim": int(d),
                            "mode": str(mode),
                            "candidate_id": int(candidate_id),
                            "word": wstr,
                            **rr,
                        }
                    )
                for nr in nv_rows:
                    nv_rows_all.append(
                        {
                            "dim": int(d),
                            "mode": str(mode),
                            "candidate_id": int(candidate_id),
                            "word": wstr,
                            **nr,
                        }
                    )

            # rank-based final_reward within dim/mode
            rows_mode.sort(key=lambda r: float(r["J_total"]))
            N = len(rows_mode)
            for rank_i, r in enumerate(rows_mode):
                base = float((N - 1 - rank_i) / max(1, N - 1))
                final_reward = float(base)  # power=1 for V14.6
                # modifiers similar spirit to V14.5
                if not bool(r["stable_flag"]):
                    final_reward = 0.0
                frow = next((fr for fr in spectral_features_rows if int(fr.get("candidate_id", -1)) == int(r["candidate_id"])), {})
                if float(frow.get("support_overlap", 1.0)) <= 0.0:
                    final_reward *= 0.1
                if float(frow.get("poisson_like_fraction", 1.0)) >= 0.95:
                    final_reward *= 0.25
                final_reward = float(min(max(final_reward, 1e-6 if base > 0 else 0.0), 1.0))
                r["final_reward"] = float(final_reward)

    # candidate ranking by J_total
    for d in dims:
        for mode in generator_modes:
            sub = [r for r in generated_candidates if int(r["dim"]) == int(d) and str(r["mode"]) == str(mode)]
            sub.sort(key=lambda r: float(r["J_total"]))
            for rank_i, r in enumerate(sub, start=1):
                # join feature subset
                frow = next((fr for fr in spectral_features_rows if int(fr["candidate_id"]) == int(r["candidate_id"])), {})
                candidate_ranking.append(
                    {
                        "dim": int(d),
                        "mode": str(mode),
                        "rank": int(rank_i),
                        "word": str(r["word"]),
                        "word_len": int(r["word_len"]),
                        "J_total": float(r["J_total"]),
                        "critic_score": float(r["critic_score"]) if math.isfinite(safe_float(r["critic_score"])) else float("nan"),
                        "final_reward": float(r["final_reward"]) if math.isfinite(safe_float(r["final_reward"])) else float("nan"),
                        "support_overlap": safe_float(frow.get("support_overlap", float("nan"))),
                        "active_arg_error": safe_float(frow.get("active_arg_error", float("nan"))),
                        "residue_error": safe_float(frow.get("residue_error", float("nan"))),
                        "imag_leak": safe_float(frow.get("imag_leak", float("nan"))),
                        "trace_error": safe_float(frow.get("trace_error", float("nan"))),
                        "poisson_like_fraction": safe_float(frow.get("poisson_like_fraction", float("nan"))),
                        "all_gate_pass": bool(r["all_gate_pass"]),
                    }
                )

    # gate summary: best per dim/mode
    gate_summary_rows: List[Dict[str, Any]] = []
    for d in dims:
        for mode in generator_modes:
            sub = [r for r in generated_candidates if int(r["dim"]) == int(d) and str(r["mode"]) == str(mode)]
            if not sub:
                continue
            sub.sort(key=lambda r: float(r["J_total"]))
            best = sub[0]
            frow = next((fr for fr in spectral_features_rows if int(fr["candidate_id"]) == int(best["candidate_id"])), {})
            stable_flag = bool(best["stable_flag"])
            feat = frow
            null_flags = {"beats_random": False, "beats_rejected": False, "beats_ablation": False}
            losses, gates, cls = compute_losses_and_gates(
                stable_flag=stable_flag,
                feat=feat,
                word_len=int(best["word_len"]),
                args=args,
                null_flags=null_flags,
                critic_score=float(best.get("critic_score", 0.5)) if math.isfinite(safe_float(best.get("critic_score"))) else 0.5,
            )
            gate_summary_rows.append(
                {
                    "dim": int(d),
                    "mode": str(mode),
                    "word": str(best["word"]),
                    **{k: bool(v) for k, v in gates.items() if k != "all_gate_pass"},
                    "all_gate_pass": bool(gates["all_gate_pass"]),
                    "classification": str(cls),
                    "J_total": float(best["J_total"]),
                    "critic_score": float(best["critic_score"]) if math.isfinite(safe_float(best["critic_score"])) else float("nan"),
                    "final_reward": float(best["final_reward"]) if math.isfinite(safe_float(best["final_reward"])) else float("nan"),
                }
            )

    # ablation summary per dim
    ablation_rows: List[Dict[str, Any]] = []
    for d in dims:
        bestJ_by_mode: Dict[str, float] = {}
        any_pass = False
        for mode in generator_modes:
            sub = [r for r in generated_candidates if int(r["dim"]) == int(d) and str(r["mode"]) == str(mode)]
            if not sub:
                bestJ_by_mode[mode] = float("inf")
                continue
            sub.sort(key=lambda r: float(r["J_total"]))
            bestJ_by_mode[mode] = float(sub[0]["J_total"])
            any_pass = any_pass or bool(sub[0]["all_gate_pass"])
        best_mode = min(bestJ_by_mode, key=lambda m: bestJ_by_mode[m]) if bestJ_by_mode else ""
        ablation_rows.append(
            {
                "dim": int(d),
                "gan_only_best_J": float(bestJ_by_mode.get("gan_only", float("inf"))),
                "gan_aco_hybrid_best_J": float(bestJ_by_mode.get("gan_aco_hybrid", float("inf"))),
                "gan_semantic_hybrid_best_J": float(bestJ_by_mode.get("gan_semantic_hybrid", float("inf"))),
                "best_mode": str(best_mode),
                "best_overall_J": float(bestJ_by_mode.get(best_mode, float("inf"))) if best_mode else float("inf"),
                "any_gate_pass": bool(any_pass),
            }
        )

    # write outputs
    write_csv(
        out_dir / "v14_6_generated_candidates.csv",
        fieldnames=[
            "dim",
            "mode",
            "candidate_id",
            "word",
            "word_len",
            "stable_flag",
            "critic_score",
            "J_total",
            "final_reward",
            "spacing_loss",
            "nv_loss",
            "arg_loss",
            "residue_loss",
            "trace_loss",
            "anti_poisson_loss",
            "null_loss",
            "complexity_loss",
            "gan_prior_loss",
            "all_gate_pass",
        ],
        rows=generated_candidates,
    )
    write_csv(
        out_dir / "v14_6_spectral_features.csv",
        fieldnames=["dim", "mode", "candidate_id", "word"] + FEATURE_NAMES + ["poisson_like_fraction"],
        rows=spectral_features_rows,
    )
    write_csv(
        out_dir / "v14_6_gate_summary.csv",
        fieldnames=[
            "dim",
            "mode",
            "word",
            "G1_stable_operator",
            "G2_support_overlap_ok",
            "G3_active_argument_ok",
            "G4_residue_error_ok",
            "G5_imag_leak_ok",
            "G6_trace_proxy_ok",
            "G7_not_poisson_like",
            "G8_beats_random_baseline",
            "G9_beats_rejected_word_if_available",
            "G10_beats_ablation_if_available",
            "G11_not_gan_overfit",
            "G12_word_complexity_ok",
            "all_gate_pass",
            "classification",
            "J_total",
            "critic_score",
            "final_reward",
        ],
        rows=gate_summary_rows,
    )
    write_csv(
        out_dir / "v14_6_candidate_ranking.csv",
        fieldnames=[
            "dim",
            "mode",
            "rank",
            "word",
            "word_len",
            "J_total",
            "critic_score",
            "final_reward",
            "support_overlap",
            "active_arg_error",
            "residue_error",
            "imag_leak",
            "trace_error",
            "poisson_like_fraction",
            "all_gate_pass",
        ],
        rows=candidate_ranking,
    )
    write_csv(
        out_dir / "v14_6_gan_training_history.csv",
        fieldnames=["epoch", "dim", "D_loss", "G_loss", "critic_real_mean", "critic_fake_mean", "best_J_epoch", "best_reward_epoch"],
        rows=gan_history,
    )
    write_csv(
        out_dir / "v14_6_mode_ablation_summary.csv",
        fieldnames=["dim", "gan_only_best_J", "gan_aco_hybrid_best_J", "gan_semantic_hybrid_best_J", "best_mode", "best_overall_J", "any_gate_pass"],
        rows=ablation_rows,
    )
    write_csv(
        out_dir / "v14_6_residue_scores.csv",
        fieldnames=[
            "dim",
            "mode",
            "candidate_id",
            "word",
            "window_a",
            "window_b",
            "I_operator_real",
            "I_operator_imag",
            "I_target_real",
            "I_target_imag",
            "residue_count_error",
            "residue_imag_leak",
        ],
        rows=residue_rows_all,
    )
    write_csv(
        out_dir / "v14_6_argument_counts.csv",
        fieldnames=[
            "dim",
            "mode",
            "candidate_id",
            "word",
            "window_a",
            "window_b",
            "N_operator",
            "N_target",
            "N_error",
            "N_error_norm",
            "active_window",
        ],
        rows=argument_rows_all,
    )
    write_csv(
        out_dir / "v14_6_nv_curves.csv",
        fieldnames=["dim", "mode", "candidate_id", "word", "kind", "L", "Sigma2"],
        rows=nv_rows_all,
    )

    # ----------------------------
    # V14.6b: real null-control Stage E (comparisons + gates)
    # ----------------------------
    v13o14_dir = _resolve(args.v13o14_dir)
    v14_2_dir = _resolve(args.v14_2_dir)
    v14_5_dir = _resolve(args.v14_5_dir)

    v13_per_dim, v13_pool, v13_missing = load_v13o14_baselines(v13o14_dir)
    v14_2_best, v14_2_pool, v14_2_missing = load_v14_2_baselines(v14_2_dir)
    v14_5_best, v14_5_pool, v14_5_missing = load_v14_5_baselines(v14_5_dir)

    prior_pool: List[Dict[str, Any]] = []
    prior_pool.extend(v13_pool)
    prior_pool.extend(v14_2_pool)
    prior_pool.extend(v14_5_pool)
    # sanitize pool rows (ensure required columns)
    for r in prior_pool:
        r.setdefault("group", "unknown")
        r.setdefault("source", "unknown")
        r["dim"] = int(r.get("dim", 0) or 0)
        r["J"] = safe_float(r.get("J", float("nan")))

    v14_6b_real_null: List[Dict[str, Any]] = []
    v14_6b_gate: List[Dict[str, Any]] = []
    decision_rows: List[Dict[str, Any]] = []

    # helper: baseline lookup by dim
    def _best_from_pool(dim: int, pred) -> Optional[float]:
        Js = [safe_float(r.get("J", float("nan"))) for r in prior_pool if int(r.get("dim", 0)) == int(dim) and pred(r)]
        Js = [j for j in Js if math.isfinite(j)]
        return float(min(Js)) if Js else None

    for d in dims:
        # required baseline groups: random, rejected, ablation, best v14_2, best v14_5
        best_random_J = None
        best_rejected_J = None
        best_ablation_J = None
        best_v14_2_J = v14_2_best.get(int(d))
        best_v14_5_J = v14_5_best.get(int(d))

        # V13O.14 provides best_random/best_rejected/best_ablation explicitly if present
        if int(d) in v13_per_dim:
            best_random_J = safe_float(v13_per_dim[int(d)].get("best_random_J", float("nan")))
            best_rejected_J = safe_float(v13_per_dim[int(d)].get("best_rejected_J", float("nan")))
            best_ablation_J = safe_float(v13_per_dim[int(d)].get("best_ablation_J", float("nan")))
            if not math.isfinite(best_random_J):
                best_random_J = None
            if not math.isfinite(best_rejected_J):
                best_rejected_J = None
            if not math.isfinite(best_ablation_J):
                best_ablation_J = None

        # fallback: mine from pool
        if best_random_J is None:
            best_random_J = _best_from_pool(int(d), lambda r: bool(r.get("is_random_baseline", False)) or str(r.get("group", "")).startswith("random"))
        if best_rejected_J is None:
            best_rejected_J = _best_from_pool(int(d), lambda r: str(r.get("group", "")) == "rejected_word_seed17" or bool(r.get("is_rejected_word", False)))
        if best_ablation_J is None:
            best_ablation_J = _best_from_pool(int(d), lambda r: bool(r.get("is_ablation", False)) or str(r.get("group", "")).startswith("ablate_") or str(r.get("group", "")).startswith("ablate"))

        # pool for zscore/percentile: null/prior candidates (exclude primary zeta target)
        pool_Js = [safe_float(r.get("J", float("nan"))) for r in prior_pool if int(r.get("dim", 0)) == int(d)]
        pool_Js = [j for j in pool_Js if math.isfinite(j)]

        for mode in generator_modes:
            sub = [r for r in generated_candidates if int(r["dim"]) == int(d) and str(r["mode"]) == str(mode)]
            if not sub:
                continue
            sub.sort(key=lambda r: float(r["J_total"]))
            best = sub[0]
            best_gan_J = float(best["J_total"])

            # best null = best among available baselines
            null_candidates = [x for x in [best_random_J, best_rejected_J, best_ablation_J, best_v14_2_J, best_v14_5_J] if x is not None and math.isfinite(float(x))]
            best_null_J = float(min(null_candidates)) if null_candidates else None
            null_separation = float(best_null_J - best_gan_J) if (best_null_J is not None and math.isfinite(best_gan_J)) else float("nan")
            z, pct = baseline_stats_vs_gan(best_gan_J, pool_Js)

            beats_random = bool(best_random_J is not None and best_gan_J <= float(best_random_J))
            beats_rejected = bool(best_rejected_J is not None and best_gan_J <= float(best_rejected_J))
            beats_ablation = bool(best_ablation_J is not None and best_gan_J <= float(best_ablation_J))
            beats_v14_2 = bool(best_v14_2_J is not None and best_gan_J <= float(best_v14_2_J))
            beats_v14_5 = bool(best_v14_5_J is not None and best_gan_J <= float(best_v14_5_J))

            missing_reasons = []
            if best_random_J is None:
                missing_reasons.append("missing_random_baseline")
            if best_rejected_J is None:
                missing_reasons.append("missing_rejected_word_seed17")
            if best_ablation_J is None:
                missing_reasons.append("missing_ablation_baselines")
            if best_v14_2_J is None:
                missing_reasons.append("missing_v14_2_best")
            if best_v14_5_J is None:
                missing_reasons.append("missing_v14_5_best")
            if len(pool_Js) < 3:
                missing_reasons.append("null_distribution_too_small(<3)")

            v14_6b_real_null.append(
                {
                    "dim": int(d),
                    "mode": str(mode),
                    "best_gan_J": float(best_gan_J),
                    "best_random_J": float(best_random_J) if best_random_J is not None else float("nan"),
                    "best_rejected_J": float(best_rejected_J) if best_rejected_J is not None else float("nan"),
                    "best_ablation_J": float(best_ablation_J) if best_ablation_J is not None else float("nan"),
                    "best_v14_2_J": float(best_v14_2_J) if best_v14_2_J is not None else float("nan"),
                    "best_v14_5_J": float(best_v14_5_J) if best_v14_5_J is not None else float("nan"),
                    "gan_minus_best_random": float(best_gan_J - best_random_J) if best_random_J is not None else float("nan"),
                    "gan_minus_rejected": float(best_gan_J - best_rejected_J) if best_rejected_J is not None else float("nan"),
                    "gan_minus_best_ablation": float(best_gan_J - best_ablation_J) if best_ablation_J is not None else float("nan"),
                    "gan_minus_v14_2": float(best_gan_J - best_v14_2_J) if best_v14_2_J is not None else float("nan"),
                    "gan_minus_v14_5": float(best_gan_J - best_v14_5_J) if best_v14_5_J is not None else float("nan"),
                    "beats_random": bool(beats_random),
                    "beats_rejected": bool(beats_rejected),
                    "beats_ablation": bool(beats_ablation),
                    "beats_v14_2": bool(beats_v14_2),
                    "beats_v14_5": bool(beats_v14_5),
                    "null_separation": float(null_separation),
                    "null_zscore": float(z),
                    "null_percentile": float(pct),
                    "missing_baseline_reason": "|".join(missing_reasons),
                }
            )

            # update gates: do NOT pass on missing baselines
            G8 = bool(beats_random) if best_random_J is not None else False
            G9 = bool(beats_rejected) if best_rejected_J is not None else False
            G10 = bool(beats_ablation) if best_ablation_J is not None else False
            G11 = bool(beats_v14_2 or beats_v14_5) if (best_v14_2_J is not None or best_v14_5_J is not None) else False
            G12 = bool(math.isfinite(null_separation) and null_separation > 0.0 and len(pool_Js) >= 3)

            # spectral gates from original gate summary row (already computed per best candidate)
            g0 = next((gr for gr in gate_summary_rows if int(gr.get("dim", -1)) == int(d) and str(gr.get("mode", "")) == str(mode)), None)
            if g0 is None:
                continue
            spectral_ok = bool(
                bool(g0.get("G1_stable_operator", False))
                and bool(g0.get("G2_support_overlap_ok", False))
                and bool(g0.get("G3_active_argument_ok", False))
                and bool(g0.get("G4_residue_error_ok", False))
                and bool(g0.get("G5_imag_leak_ok", False))
                and bool(g0.get("G6_trace_proxy_ok", False))
                and bool(g0.get("G7_not_poisson_like", False))
                and bool(g0.get("G11_not_gan_overfit", False))
                and bool(g0.get("G12_word_complexity_ok", False))
            )
            all_pass = bool(spectral_ok and G8 and G9 and G10 and G11 and G12)

            v14_6b_gate.append(
                {
                    "dim": int(d),
                    "mode": str(mode),
                    "word": str(g0.get("word", "")),
                    "J_total": float(g0.get("J_total", float("nan"))),
                    "critic_score": float(g0.get("critic_score", float("nan"))),
                    "final_reward": float(g0.get("final_reward", float("nan"))),
                    "G1_stable_operator": bool(g0.get("G1_stable_operator", False)),
                    "G2_support_overlap_ok": bool(g0.get("G2_support_overlap_ok", False)),
                    "G3_active_argument_ok": bool(g0.get("G3_active_argument_ok", False)),
                    "G4_residue_error_ok": bool(g0.get("G4_residue_error_ok", False)),
                    "G5_imag_leak_ok": bool(g0.get("G5_imag_leak_ok", False)),
                    "G6_trace_proxy_ok": bool(g0.get("G6_trace_proxy_ok", False)),
                    "G7_not_poisson_like": bool(g0.get("G7_not_poisson_like", False)),
                    "G8_beats_random_controls": bool(G8),
                    "G9_beats_rejected_control": bool(G9),
                    "G10_beats_ablation_controls": bool(G10),
                    "G11_beats_prior_artin_search": bool(G11),
                    "G12_null_separation_ok": bool(G12),
                    "all_gate_pass": bool(all_pass),
                    "null_separation": float(null_separation),
                    "null_zscore": float(z),
                    "null_percentile": float(pct),
                    "missing_baseline_reason": "|".join(missing_reasons),
                    "classification": "ALL_GATE_PASS" if all_pass else str(g0.get("classification", "UNKNOWN")),
                }
            )

    # decision summary
    n_all = sum(1 for r in v14_6b_gate if bool(r.get("all_gate_pass", False)))
    n_beats_random = sum(1 for r in v14_6b_gate if bool(r.get("G8_beats_random_controls", False)))
    n_not_poiss = sum(1 for r in v14_6b_gate if bool(r.get("G7_not_poisson_like", False)))
    proceed_v14_7 = bool(n_beats_random > 0 or n_not_poiss > 0)
    analytic_claim = False
    decision_rows.append(
        {
            "proceed_to_v14_7": bool(proceed_v14_7),
            "analytic_claim": bool(analytic_claim),
            "n_gate_rows": int(len(v14_6b_gate)),
            "n_all_gate_pass": int(n_all),
            "n_beats_random": int(n_beats_random),
            "n_not_poisson_like": int(n_not_poiss),
            "missing_sources": "|".join(sorted(set(v13_missing + v14_2_missing + v14_5_missing))),
        }
    )

    write_csv(
        out_dir / "v14_6b_prior_baseline_pool.csv",
        fieldnames=["dim", "group", "source", "J", "is_random_baseline", "is_ablation", "is_rejected_word", "is_primary", "support_overlap_fraction", "poisson_like_fraction"],
        rows=prior_pool,
    )
    write_csv(
        out_dir / "v14_6b_real_null_comparisons.csv",
        fieldnames=list(v14_6b_real_null[0].keys()) if v14_6b_real_null else ["dim", "mode", "best_gan_J"],
        rows=v14_6b_real_null,
    )
    write_csv(
        out_dir / "v14_6b_real_null_gate_summary.csv",
        fieldnames=list(v14_6b_gate[0].keys()) if v14_6b_gate else ["dim", "mode", "all_gate_pass"],
        rows=v14_6b_gate,
    )
    write_csv(
        out_dir / "v14_6b_decision_summary.csv",
        fieldnames=list(decision_rows[0].keys()) if decision_rows else ["proceed_to_v14_7", "analytic_claim"],
        rows=decision_rows,
    )

    # V14.6b report (md/tex/pdf)
    OUT_ABS_B = str(out_dir.resolve())
    md_b = []
    md_b.append("# V14.6b — Real Null-Control Stage E\n\n")
    md_b.append("> Computational evidence only; not a proof of RH.\n\n")
    md_b.append("## Purpose\n")
    md_b.append("Replace placeholder null-controls with **real comparisons** against prior baselines (V13O.14, V14.2, V14.5) without silently passing missing sources.\n\n")
    md_b.append("## Inputs used (best-effort)\n")
    md_b.append(f"- v13o14_dir: `{str(v13o14_dir)}`\n")
    md_b.append(f"- v14_2_dir: `{str(v14_2_dir)}`\n")
    md_b.append(f"- v14_5_dir: `{str(v14_5_dir)}`\n\n")
    md_b.append("## Missing baseline policy\n")
    md_b.append("- Missing required baseline => corresponding gate is **False** and reason is recorded in `missing_baseline_reason`.\n")
    md_b.append("- If null distribution has <3 values => `null_zscore`/`null_percentile` are NaN and **G12 is False**.\n\n")
    md_b.append("## Outputs\n")
    md_b.append("- `v14_6b_prior_baseline_pool.csv`\n")
    md_b.append("- `v14_6b_real_null_comparisons.csv`\n")
    md_b.append("- `v14_6b_real_null_gate_summary.csv`\n")
    md_b.append("- `v14_6b_decision_summary.csv`\n\n")
    md_b.append("## Decision\n")
    md_b.append(f"- proceed_to_v14_7: **{proceed_v14_7}**\n")
    md_b.append(f"- analytic_claim: **{analytic_claim}**\n")
    md_b.append(f"- all_gate_pass count: **{n_all}** (out of {len(v14_6b_gate)})\n\n")
    md_b.append("## Verification commands\n")
    md_b.append("```bash\n")
    md_b.append(f'OUT="{OUT_ABS_B}"\n\n')
    md_b.append('echo "=== V14.6b FILES ==="\nfind "$OUT" -maxdepth 1 -type f | sort\n\n')
    md_b.append('echo "=== V14.6b NULL GATE ==="\ncolumn -s, -t < "$OUT"/v14_6b_real_null_gate_summary.csv | head -120\n\n')
    md_b.append('echo "=== V14.6b NULL COMPARISONS ==="\ncolumn -s, -t < "$OUT"/v14_6b_real_null_comparisons.csv | head -120\n\n')
    md_b.append('echo "=== V14.6b BASELINE POOL ==="\ncolumn -s, -t < "$OUT"/v14_6b_prior_baseline_pool.csv | head -120\n\n')
    md_b.append('echo "=== V14.6b DECISION ==="\ncolumn -s, -t < "$OUT"/v14_6b_decision_summary.csv\n\n')
    md_b.append('echo "=== V14.6b REPORT ==="\nhead -240 "$OUT"/v14_6b_report.md\n')
    md_b.append("```\n\n")
    md_b.append("> Computational evidence only; not a proof of RH.\n")
    write_text(out_dir / "v14_6b_report.md", "".join(md_b))

    tex_b = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\begin{document}
\section*{V14.6b --- Real Null-Control Stage E}
\textbf{Computational evidence only; not a proof of RH.}

\subsection*{Policy}
Missing baselines never silently pass: gates are false and reasons are recorded.
\end{document}
"""
    tex_b_path = out_dir / "v14_6b_report.tex"
    write_text(tex_b_path, tex_b)
    if args.write_pdf or _find_pdflatex():
        _okb, _msgb = try_pdflatex(tex_b_path, out_dir, "v14_6b_report.pdf")

    results = {
        "version": "v14_6",
        "out_dir": str(out_dir),
        "dims": dims,
        "generator_modes": generator_modes,
        "num_epochs": int(args.num_epochs),
        "num_candidates": int(args.num_candidates),
        "device": device,
        "torch_available": bool(_HAVE_TORCH),
        "warnings": warnings,
        "gate_summary": gate_summary_rows,
        "mode_ablation_summary": ablation_rows,
        "notes": [
            "Computational evidence only; not a proof of RH.",
            "Critic does NOT decide final acceptance; hard gates do.",
            "Null-control baselines are best-effort in V14.6 outputs; V14.6b adds real null-control comparisons and does not silently pass missing sources.",
        ],
        "v14_6b": {
            "missing_sources": sorted(set(v13_missing + v14_2_missing + v14_5_missing)),
            "decision_summary": decision_rows,
        },
    }
    write_text(out_dir / "v14_6_results.json", json.dumps(json_sanitize(results), indent=2, ensure_ascii=False) + "\n")

    # Report
    OUT_ABS = str(out_dir.resolve())
    any_all = sum(1 for r in gate_summary_rows if bool(r.get("all_gate_pass", False)))
    any_not_poiss = sum(1 for r in gate_summary_rows if bool(r.get("G7_not_poisson_like", False)))
    any_res_ok = sum(1 for r in gate_summary_rows if bool(r.get("G4_residue_error_ok", False)))
    any_arg_ok = sum(1 for r in gate_summary_rows if bool(r.get("G3_active_argument_ok", False)))
    md = []
    md.append("# V14.6 Spectral-GAN Artin Operator Proposal Engine\n\n")
    md.append("> Computational evidence only; not a proof of RH.\n\n")
    md.append("## 1. Purpose\n")
    md.append("Use a PyTorch Generator to propose Artin words and a Critic to score spectral fingerprints, while **final acceptance is deterministic** via hard gates (spectrum/NV/anti-Poisson/residue/argument/trace/null controls).\n\n")
    md.append("## 2. Why GAN is used only as proposal engine\n")
    md.append("The critic can be wrong or overfit; therefore it never decides acceptance. It only biases proposals and provides an auxiliary prior loss.\n\n")
    md.append("## 3. Generator architecture\n")
    md.append("- `ArtinWordGenerator`: MLP over noise `z` + dim embedding + mode embedding, outputs logits for generator id / power / stop per token.\n\n")
    md.append("## 4. Spectral critic architecture\n")
    md.append("- `SpectralCritic`: MLP over a fixed feature vector, outputs `critic_score in [0,1]`.\n\n")
    md.append("## 5. Hard evaluator and gates\n")
    md.append("Stabilized operator builder: Hermitian local blocks, trace removal, radius normalization, safe eigensolver; then compute spacing/NV/argument/residue/trace features and apply gates.\n\n")
    md.append("## 6. Best candidate per dimension\n")
    md.append("See `v14_6_gate_summary.csv` and `v14_6_candidate_ranking.csv`.\n\n")
    md.append("## 7. Mode ablation\n")
    md.append("Modes: `gan_only`, `gan_aco_hybrid`, `gan_semantic_hybrid`. Summary: `v14_6_mode_ablation_summary.csv`.\n\n")
    md.append("## 8. Did GAN improve over V14.5?\n")
    md.append("This run reports internal best-J by mode; a direct V14.5-vs-V14.6 comparison requires matching evaluator settings and baselines.\n\n")
    md.append("## 9–11. Gates status\n")
    md.append(f"- all_gate_pass count: **{any_all}** (out of {len(gate_summary_rows)})\n")
    md.append(f"- not_poisson_like count: **{any_not_poiss}**\n")
    md.append(f"- residue_ok count: **{any_res_ok}**\n")
    md.append(f"- argument_ok count: **{any_arg_ok}**\n\n")
    md.append("## 12. Decision\n")
    md.append("- Proceed to V14.7 only if reward/score diversity improved or at least one gate nearly passes.\n")
    md.append("- **Analytic claim**: always **False** unless `all_gate_pass` and null controls pass.\n\n")
    md.append("## Explicit answers\n")
    md.append("- Did V14.6 use GAN? **Yes** (PyTorch generator+critic) if torch available; otherwise fallback proposal.\n")
    md.append("- Did it generate new Artin words? **Yes**.\n")
    md.append("- Did critic decide final acceptance? **No**.\n")
    md.append("- Did hard gates decide final acceptance? **Yes**.\n\n")
    md.append("## Verification commands\n")
    md.append("```bash\n")
    md.append(f'OUT="{OUT_ABS}"\n\n')
    md.append('echo "=== FILES ==="\nfind "$OUT" -maxdepth 1 -type f | sort\n\n')
    md.append('echo "=== GATE ==="\ncolumn -s, -t < "$OUT"/v14_6_gate_summary.csv | head -120\n\n')
    md.append('echo "=== RANKING ==="\ncolumn -s, -t < "$OUT"/v14_6_candidate_ranking.csv | head -120\n\n')
    md.append('echo "=== ABLATION ==="\ncolumn -s, -t < "$OUT"/v14_6_mode_ablation_summary.csv\n\n')
    md.append('echo "=== GAN HISTORY ==="\ncolumn -s, -t < "$OUT"/v14_6_gan_training_history.csv | tail -80\n\n')
    md.append('echo "=== ARGUMENT COUNTS ==="\ncolumn -s, -t < "$OUT"/v14_6_argument_counts.csv | head -120\n\n')
    md.append('echo "=== RESIDUE SCORES ==="\ncolumn -s, -t < "$OUT"/v14_6_residue_scores.csv | head -120\n\n')
    md.append('echo "=== REPORT ==="\nhead -240 "$OUT"/v14_6_report.md\n')
    md.append("```\n\n")
    md.append("> Computational evidence only; not a proof of RH.\n")
    write_text(out_dir / "v14_6_report.md", "".join(md))

    tex = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{hyperref}
\begin{document}
\section*{V14.6 --- Spectral-GAN Artin Operator Proposal Engine}
\textbf{Computational evidence only; not a proof of RH.}

\subsection*{Summary}
This experiment uses a PyTorch generator+critic as a \emph{proposal engine} for Artin words.
Final acceptance is determined only by deterministic spectral diagnostics and hard gates.

\subsection*{No proof claim}
This report provides computational diagnostics only and makes no analytic claim.
\end{document}
"""
    tex_path = out_dir / "v14_6_report.tex"
    write_text(tex_path, tex)
    wrote_pdf = False
    pdf_msg = ""
    if args.write_pdf or _find_pdflatex():
        wrote_pdf, pdf_msg = try_pdflatex(tex_path, out_dir, "v14_6_report.pdf")
        if not wrote_pdf:
            warnings.append(f"pdflatex failed/skipped: {pdf_msg}")

    elapsed = time.perf_counter() - t0
    print(f"[V14.6] done in {format_seconds(elapsed)} out_dir={out_dir}", flush=True)


if __name__ == "__main__":
    main()

