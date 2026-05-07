#!/usr/bin/env python3
"""
V14.7b — Support-Calibrated Anti-Poisson Smoke (Ant-RH)

Computational evidence only; not a proof of RH.

Purpose:
  - Fix V14.7 smoke failure modes by introducing support pre-calibration (transport map)
  - Enforce hard rejections for zero support and NV flatline/pathology
  - Make reward usable for ACO/DTES by using rank-log reward or inverse-J reward

This is a diagnostic smoke / infrastructure validation, not an RH proof attempt.
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
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pandas as pd  # type: ignore

    _HAVE_PANDAS = True
except Exception:
    pd = None  # type: ignore
    _HAVE_PANDAS = False


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from validation import residue_diagnostics as rd  # noqa: E402

try:
    from core.spectral_stabilization import safe_eigh as _safe_eigh  # type: ignore

    _HAVE_SAFE_EIGH = True
except Exception:
    _safe_eigh = None
    _HAVE_SAFE_EIGH = False


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


def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
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


def _read_csv_best_effort(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    if _HAVE_PANDAS:
        try:
            df = pd.read_csv(path)  # type: ignore
            return df.to_dict(orient="records")  # type: ignore
        except Exception:
            pass
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


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


def try_pdflatex(tex_path: Path, out_dir: Path, pdf_name: str) -> bool:
    exe = _find_pdflatex()
    if not exe:
        return False
    try:
        r = subprocess.run(
            [exe, "-interaction=nonstopmode", f"-output-directory={out_dir.resolve()}", tex_path.name],
            cwd=str(out_dir.resolve()),
            capture_output=True,
            text=True,
            timeout=240,
        )
        return r.returncode == 0 and (out_dir / pdf_name).is_file()
    except Exception:
        return False


# ----------------------------
# Stabilized operator construction (V14.2-style local blocks)
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
    return 0.5 * (M + M.conj().T)


def op_norm_fro(A: np.ndarray) -> float:
    return float(np.linalg.norm(A, ord="fro"))


def normalize_operator(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = op_norm_fro(A)
    return A / max(float(n), float(eps))


def remove_trace(H: np.ndarray) -> np.ndarray:
    n = int(H.shape[0])
    if n <= 0:
        return H
    tr = np.trace(H) / complex(n)
    return H - tr * np.eye(n, dtype=H.dtype)


def safe_eigvalsh(H: np.ndarray, *, seed: int) -> Optional[np.ndarray]:
    Hh = hermitian_part(H)
    if not np.isfinite(np.asarray(Hh.real)).all():
        return None
    if _HAVE_SAFE_EIGH and _safe_eigh is not None and np.isrealobj(Hh):
        try:
            w, _, _rep = _safe_eigh(np.asarray(Hh.real, dtype=np.float64), k=None, return_eigenvectors=False, stabilize=True, seed=int(seed))
            w = np.asarray(w, dtype=np.float64).reshape(-1)
            w = w[np.isfinite(w)]
            w.sort()
            return w
        except Exception:
            pass
    try:
        w = np.linalg.eigvalsh(np.asarray(Hh.real, dtype=np.float64))
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


def make_stable_generator(dim: int, generator_index: int, power: int, *, theta_base: float) -> np.ndarray:
    n = int(dim)
    i = int(max(1, min(n - 1, int(generator_index))))
    max_i = max(1, n - 1)
    theta = float(theta_base) * float(i) / float(max_i)
    B = rotation_block(float(power) * theta)
    G = embed_2x2(n, i - 1, B)
    A = hermitian_part(G.astype(np.complex128, copy=False))
    return normalize_operator(A)


def build_stabilized_operator(dim: int, word: List[Tuple[int, int]], seed: int) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    n = int(dim)
    eps = 1e-12
    theta_base = math.pi / 8.0
    H = np.zeros((n, n), dtype=np.complex128)
    nan_flag = False
    for k, (gi, p) in enumerate(word):
        A = make_stable_generator(n, int(gi), int(p), theta_base=theta_base)
        ck = 1.0 / math.sqrt(float(k) + 1.0)
        H = H + complex(ck) * A
        if not np.isfinite(H.real).all():
            nan_flag = True
            break
    H = hermitian_part(H)
    H = remove_trace(H)
    H, rad = normalize_spectral_radius(H, target_radius=float(max(4.0, n / 4.0)), eps=eps)
    w = safe_eigvalsh(H, seed=seed)
    stable = (not nan_flag) and (w is not None) and (w.size >= 8) and np.isfinite(H.real).all() and math.isfinite(rad)
    diag = {
        "stable": bool(stable),
        "nan_flag": bool(nan_flag),
        "spectral_radius": float(rad),
        "fro_norm": float(op_norm_fro(np.asarray(H.real, dtype=np.float64))),
        "trace_abs": float(abs(np.trace(H))),
        "n_eigs": int(w.size) if w is not None else 0,
    }
    return (H if stable else None), diag


# ----------------------------
# Word utilities + sampling
# ----------------------------


def power_values(max_power: int) -> List[int]:
    return [p for p in range(-max_power, max_power + 1) if p != 0]


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
        i0 = out[0][0]
        i1 = int(max(1, min(dim - 1, i0 + 1)))
        out.append((i1, out[0][1]))
    return out


def word_to_string(word: List[Tuple[int, int]]) -> str:
    return " ".join([f"sigma_{int(i)}^{int(p)}" for i, p in word])


def sample_token(
    *,
    dim: int,
    pher: Dict[Tuple[int, int], float],
    alpha: float,
    beta: float,
    semantic_term: float,
    rng: np.random.Generator,
) -> Tuple[int, int]:
    items = list(pher.keys())
    tau = np.asarray([pher[it] for it in items], dtype=np.float64)
    mid = 0.5 * (dim - 1)
    eta = np.asarray([(1.0 / (1.0 + 0.01 * abs(float(gi) - mid))) * (1.0 / (1.0 + 0.10 * abs(float(pw)))) for (gi, pw) in items], dtype=np.float64)
    logits = float(alpha) * np.log(np.maximum(1e-12, tau)) + float(beta) * np.log(np.maximum(1e-12, eta))
    if semantic_term != 0.0:
        sem = np.asarray([(-0.5 * abs(float(gi) - mid) / max(1.0, float(dim)) - 0.05 * abs(float(pw))) for (gi, pw) in items], dtype=np.float64)
        logits = logits + float(semantic_term) * sem
    logits = logits - float(np.max(logits))
    w = np.exp(np.clip(logits, -60.0, 60.0))
    s = float(np.sum(w))
    if not (math.isfinite(s) and s > 0.0):
        return items[int(rng.integers(0, len(items)))]
    p = w / s
    idx = int(rng.choice(np.arange(len(items)), p=p))
    return items[idx]


# ----------------------------
# Support pre-calibration (transport)
# ----------------------------


@dataclass
class TransportMap:
    mode_requested: str
    mode_effective: str
    scale: float
    shift: float
    log_c: float
    op_q10: float
    op_q50: float
    op_q90: float
    tg_q10: float
    tg_q50: float
    tg_q90: float


def transport_affine(op: np.ndarray, tg: np.ndarray) -> TransportMap:
    op = np.asarray(op, dtype=np.float64)
    tg = np.asarray(tg, dtype=np.float64)
    op = op[np.isfinite(op)]
    tg = tg[np.isfinite(tg)]
    if op.size < 8 or tg.size < 8:
        return TransportMap("affine", "none", 1.0, 0.0, 0.0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
    op_q = np.quantile(op, [0.1, 0.5, 0.9])
    tg_q = np.quantile(tg, [0.1, 0.5, 0.9])
    scale = float((tg_q[2] - tg_q[0]) / max(1e-12, (op_q[2] - op_q[0])))
    shift = float(tg_q[1] - scale * op_q[1])
    return TransportMap("affine", "affine", scale, shift, 0.0, float(op_q[0]), float(op_q[1]), float(op_q[2]), float(tg_q[0]), float(tg_q[1]), float(tg_q[2]))


def transport_log_affine(op: np.ndarray, tg: np.ndarray) -> TransportMap:
    """
    Simple log-affine fallback:
      T(x)=a*x + b + c*log1p(max(x-min(x),0))
    We try c=0 first and report affine if not enough data.
    """
    # For this smoke we keep it stable: c=0, i.e. affine effective unless fit succeeds.
    tm = transport_affine(op, tg)
    if tm.mode_effective == "none":
        tm.mode_requested = "log_affine"
        tm.mode_effective = "none"
        return tm
    tm.mode_requested = "log_affine"
    tm.mode_effective = "affine"  # explicit fallback
    tm.log_c = 0.0
    return tm


def apply_transport(op: np.ndarray, tm: TransportMap) -> np.ndarray:
    x = np.asarray(op, dtype=np.float64)
    x = x[np.isfinite(x)]
    if tm.mode_effective == "none":
        x.sort()
        return x
    y = tm.scale * x + tm.shift
    if tm.mode_requested == "log_affine" and abs(float(tm.log_c)) > 0.0:
        xmin = float(np.min(x)) if x.size else 0.0
        y = y + float(tm.log_c) * np.log1p(np.maximum(0.0, x - xmin))
    y = np.asarray(y, dtype=np.float64)
    y = y[np.isfinite(y)]
    y.sort()
    return y


# ----------------------------
# Diagnostics (calibrated)
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


def curve_l2(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(-1)
    bb = np.asarray(b, dtype=np.float64).reshape(-1)
    if aa.size == 0 or bb.size == 0 or aa.size != bb.size:
        return float("nan")
    m = np.isfinite(aa) & np.isfinite(bb)
    if not m.any():
        return float("nan")
    return float(np.sqrt(np.mean((aa[m] - bb[m]) ** 2)))


def nv_diagnostics(op_nv: np.ndarray, tgt_nv: np.ndarray, L_grid: np.ndarray) -> Dict[str, Any]:
    y = np.asarray(op_nv, dtype=np.float64)
    m = np.isfinite(y)
    if not m.any():
        return {
            "nv_range": float("nan"),
            "nv_std": float("nan"),
            "nv_median": float("nan"),
            "nv_rmse": float("nan"),
            "nv_slope_roughness": float("nan"),
            "nv_flatline": True,
            "nv_too_large": True,
        }
    yy = y[m]
    nv_range = float(np.max(yy) - np.min(yy))
    nv_std = float(np.std(yy))
    nv_median = float(np.median(yy))
    nv_rmse = float(curve_l2(op_nv, tgt_nv))
    dif = np.diff(yy)
    rough = float(np.std(dif)) if dif.size >= 2 else 0.0
    return {
        "nv_range": nv_range,
        "nv_std": nv_std,
        "nv_median": nv_median,
        "nv_rmse": nv_rmse,
        "nv_slope_roughness": rough,
        "nv_flatline": bool(nv_range < 1e-12 or nv_std < 1e-12),
        "nv_too_large": bool(nv_median > 1e6),
    }


def active_argument_counts(levels: np.ndarray, target: np.ndarray, windows: List[Tuple[float, float]]) -> Tuple[float, float, float, List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    errs_norm: List[float] = []
    active = 0
    both = 0
    for a, b in windows:
        n_op = int(rd.count_in_window(levels, float(a), float(b)))
        n_tg = int(rd.count_in_window(target, float(a), float(b)))
        active_window = (n_op > 0) or (n_tg > 0)
        if not active_window:
            continue
        active += 1
        if (n_op > 0) and (n_tg > 0):
            both += 1
        err = float(abs(n_op - n_tg))
        err_norm = float(err / float(max(1, n_tg)))
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
    med = float(np.median(np.asarray(errs_norm, dtype=np.float64))) if errs_norm else 1.0
    mean = float(np.mean(np.asarray(errs_norm, dtype=np.float64))) if errs_norm else 1.0
    sup = float(both) / float(max(1, active))
    return med, mean, sup, rows


def residue_scores(levels: np.ndarray, target: np.ndarray, windows: List[Tuple[float, float]], eta: float, n_contour_points: int) -> Tuple[float, float, List[Dict[str, Any]]]:
    errs: List[float] = []
    leaks: List[float] = []
    rows: List[Dict[str, Any]] = []
    for a, b in windows:
        n_op = int(rd.count_in_window(levels, float(a), float(b)))
        n_tg = int(rd.count_in_window(target, float(a), float(b)))
        if (n_op == 0) and (n_tg == 0):
            continue
        I_op = rd.residue_proxy_count(levels, float(a), float(b), eta=float(eta), n_contour_points=int(n_contour_points))
        I_tg = rd.residue_proxy_count(target, float(a), float(b), eta=float(eta), n_contour_points=int(n_contour_points))
        err = abs(float(I_op.real) - float(I_tg.real)) / max(1.0, abs(float(I_tg.real)))
        leak = float(abs(float(I_op.imag)))
        errs.append(float(err))
        leaks.append(float(leak))
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
    return med_err, med_leak, rows


def trace_proxy_rows(levels: np.ndarray, target: np.ndarray, windows: List[Tuple[float, float]]) -> Tuple[float, List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    errs: List[float] = []
    for a, b in windows:
        n_op = int(rd.count_in_window(levels, float(a), float(b)))
        n_tg = int(rd.count_in_window(target, float(a), float(b)))
        if (n_op == 0) and (n_tg == 0):
            continue
        c = 0.5 * (float(a) + float(b))
        for s in (0.5, 1.0, 2.0, 4.0):
            Sop = rd.trace_formula_proxy(levels, center=float(c), sigma=float(s))
            Stg = rd.trace_formula_proxy(target, center=float(c), sigma=float(s))
            if not (math.isfinite(Sop) and math.isfinite(Stg)):
                continue
            err = abs(math.log1p(max(0.0, float(Sop))) - math.log1p(max(0.0, float(Stg))))
            errs.append(float(err))
            rows.append(
                {
                    "window_a": float(a),
                    "window_b": float(b),
                    "center": float(c),
                    "sigma": float(s),
                    "S_operator": float(Sop),
                    "S_target": float(Stg),
                    "trace_error_norm": float(err),
                }
            )
    med = float(np.median(np.asarray(errs, dtype=np.float64))) if errs else 1.0
    return med, rows


def poisson_like_fraction_from_nv(op_nv: np.ndarray, tgt_nv: np.ndarray, L_grid: np.ndarray) -> float:
    op = np.asarray(op_nv, dtype=np.float64)
    tg = np.asarray(tgt_nv, dtype=np.float64)
    L = np.asarray(L_grid, dtype=np.float64)
    m = np.isfinite(op) & np.isfinite(tg) & np.isfinite(L)
    if not m.any():
        return 1.0
    op = op[m]
    tg = tg[m]
    L = L[m]
    # fraction of L where op is closer to Poisson than to target
    dP = np.abs(op - L)
    dT = np.abs(op - tg)
    return float(np.mean((dP < dT).astype(np.float64)))

def effective_windows_for_target(tgt: np.ndarray, args: argparse.Namespace) -> Tuple[List[Tuple[float, float]], Dict[str, Any]]:
    """
    If the requested window range does not intersect the target scale (e.g. target is 0..dim-1
    but user requested 100..220), remap the window configuration to the target quantile range.
    This keeps the smoke meaningful while preserving the user's requested window *shape*.
    """
    tgt = np.asarray(tgt, dtype=np.float64)
    tgt = tgt[np.isfinite(tgt)]
    tgt.sort()
    meta = {
        "window_mode_effective": "requested",
        "window_min_effective": float(args.window_min),
        "window_max_effective": float(args.window_max),
        "window_size_effective": float(args.window_size),
        "window_stride_effective": float(args.window_stride),
        "target_min": float(np.min(tgt)) if tgt.size else float("nan"),
        "target_max": float(np.max(tgt)) if tgt.size else float("nan"),
    }
    win = rd.make_windows(args.window_min, args.window_max, args.window_size, args.window_stride)
    if not win or tgt.size < 8:
        return win, meta
    # if target has zero counts in all windows -> remap
    any_counts = any(int(rd.count_in_window(tgt, float(a), float(b))) > 0 for (a, b) in win)
    if any_counts:
        return win, meta
    q10, q90 = np.quantile(tgt, [0.1, 0.9])
    # scale window_size/stride proportionally to requested span
    span_req = max(1e-12, float(args.window_max) - float(args.window_min))
    span_tgt = max(1e-12, float(q90) - float(q10))
    size_eff = float(span_tgt * float(args.window_size) / span_req)
    stride_eff = float(span_tgt * float(args.window_stride) / span_req)
    # enforce minimum widths
    size_eff = max(1e-6, size_eff)
    stride_eff = max(1e-6, stride_eff)
    win2 = rd.make_windows(float(q10), float(q90), float(size_eff), float(stride_eff))
    meta.update(
        {
            "window_mode_effective": "remapped_to_target_quantiles",
            "window_min_effective": float(q10),
            "window_max_effective": float(q90),
            "window_size_effective": float(size_eff),
            "window_stride_effective": float(stride_eff),
        }
    )
    return win2, meta


# ----------------------------
# Null baselines (best-effort, same spirit as V14.7)
# ----------------------------


def load_baseline_pool(dims: List[int], v13o14_dir: Path, v14_2_dir: Path, v14_5_dir: Path, v14_6b_dir: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    pool: List[Dict[str, Any]] = []
    missing: List[str] = []

    def add(dim: int, kind: str, J: Any, source: str, label: str, source_file: str, notes: str, missing_source: bool = False) -> None:
        pool.append(
            {
                "dim": int(dim),
                "baseline_kind": str(kind),
                "J": safe_float(J, float("nan")),
                "source": str(source),
                "label": str(label),
                "source_file": str(source_file),
                "notes": str(notes),
                "missing_source": bool(missing_source),
            }
        )

    # Prefer V14.6b pool if exists
    p6b = v14_6b_dir / "v14_6b_prior_baseline_pool.csv"
    rows6b = _read_csv_best_effort(p6b)
    if rows6b:
        for r in rows6b:
            d = safe_int(r.get("dim", 0))
            if d in dims:
                add(d, str(r.get("baseline_kind", "other")), r.get("J", float("nan")), str(r.get("source", "v14_6b")), str(r.get("label", "")), "v14_6b_prior_baseline_pool.csv", str(r.get("notes", "")), bool(r.get("missing_source", False)))
    else:
        missing.append("v14_6b_prior_baseline_pool.csv")

    # V13O.14
    r13n = _read_csv_best_effort(v13o14_dir / "v13o14_null_comparisons.csv")
    if not r13n:
        missing.append("v13o14_null_comparisons.csv")
    else:
        for r in r13n:
            d = safe_int(r.get("dim", 0))
            if d not in dims:
                continue
            for kind, col, label in [
                ("random", "best_random_J", "best_random_controls"),
                ("rejected", "best_rejected_J", "rejected_word_seed17"),
                ("ablation", "best_ablation_J", "best_ablations"),
            ]:
                if col in r:
                    J = safe_float(r.get(col, float("nan")))
                    if math.isfinite(J):
                        add(d, kind, J, "v13o14", label, "v13o14_null_comparisons.csv", "best_of")

    # V14.2
    r2b = _read_csv_best_effort(v14_2_dir / "v14_2_best_candidates.csv")
    if not r2b:
        missing.append("v14_2_best_candidates.csv")
    else:
        for r in r2b:
            d = safe_int(r.get("dim", 0))
            if d not in dims:
                continue
            J = safe_float(r.get("J_v14_2", r.get("J_total", float("nan"))))
            if math.isfinite(J):
                add(d, "prior_artin", J, "v14_2", "best_v14_2", "v14_2_best_candidates.csv", "best_candidates")

    # V14.5
    r5a = _read_csv_best_effort(v14_5_dir / "v14_5_ablation_summary.csv")
    if not r5a:
        missing.append("v14_5_ablation_summary.csv")
    else:
        for r in r5a:
            d = safe_int(r.get("dim", 0))
            if d not in dims:
                continue
            for k in ["numeric_only_best_J", "semantic_only_best_J", "hybrid_numeric_semantic_best_J", "hybrid_ranked_anticollapse_best_J"]:
                if k in r:
                    J = safe_float(r.get(k, float("nan")))
                    if math.isfinite(J):
                        add(d, "prior_artin", J, "v14_5", k.replace("_best_J", ""), "v14_5_ablation_summary.csv", "ablation_summary")

    # ensure at least placeholder per dim
    for d in dims:
        any_dim = any(int(r.get("dim", 0)) == int(d) and not bool(r.get("missing_source", False)) and math.isfinite(safe_float(r.get("J", float("nan")))) for r in pool)
        if not any_dim:
            add(d, "missing", float("nan"), "missing", "no_baselines_loaded", "", "no baselines loaded", True)
    return pool, sorted(set(missing))


def best_baseline_J(pool: List[Dict[str, Any]], dim: int, kind: str) -> Optional[float]:
    Js = []
    for r in pool:
        if int(r.get("dim", 0)) != int(dim):
            continue
        if bool(r.get("missing_source", False)):
            continue
        if str(r.get("baseline_kind", "")) != str(kind):
            continue
        J = safe_float(r.get("J", float("nan")))
        if math.isfinite(J):
            Js.append(float(J))
    return float(min(Js)) if Js else None


def best_null_J(pool: List[Dict[str, Any]], dim: int) -> Optional[float]:
    Js = []
    for r in pool:
        if int(r.get("dim", 0)) != int(dim):
            continue
        if bool(r.get("missing_source", False)):
            continue
        kind = str(r.get("baseline_kind", ""))
        if kind in ("primary", "missing"):
            continue
        J = safe_float(r.get("J", float("nan")))
        if math.isfinite(J):
            Js.append(float(J))
    return float(min(Js)) if Js else None


def null_dist(pool: List[Dict[str, Any]], dim: int) -> List[float]:
    xs = []
    for r in pool:
        if int(r.get("dim", 0)) != int(dim):
            continue
        if bool(r.get("missing_source", False)):
            continue
        kind = str(r.get("baseline_kind", ""))
        if kind in ("primary", "missing"):
            continue
        J = safe_float(r.get("J", float("nan")))
        if math.isfinite(J):
            xs.append(float(J))
    return xs


# ----------------------------
# Main evaluation
# ----------------------------


def unfold_to_mean_spacing_one(x: np.ndarray) -> np.ndarray:
    return rd.unfold_to_mean_spacing_one(np.asarray(x, dtype=np.float64).reshape(-1))


def main() -> None:
    ap = argparse.ArgumentParser(description="V14.7b support-calibrated anti-Poisson smoke (computational only).")
    ap.add_argument("--true_levels_csv", type=str, required=True)
    ap.add_argument("--zeros_csv", type=str, required=True)
    ap.add_argument("--v13o14_dir", type=str, required=True)
    ap.add_argument("--v14_2_dir", type=str, required=True)
    ap.add_argument("--v14_5_dir", type=str, required=True)
    ap.add_argument("--v14_6b_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--dims", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--num_ants", type=int, default=16)
    ap.add_argument("--num_iters", type=int, default=10)
    ap.add_argument("--max_word_len", type=int, default=16)
    ap.add_argument("--max_power", type=int, default=3)
    ap.add_argument("--window_min", type=float, default=100.0)
    ap.add_argument("--window_max", type=float, default=220.0)
    ap.add_argument("--window_size", type=float, default=40.0)
    ap.add_argument("--window_stride", type=float, default=20.0)
    ap.add_argument("--L_min", type=float, default=0.5)
    ap.add_argument("--L_max", type=float, default=8.0)
    ap.add_argument("--n_L", type=int, default=16)
    ap.add_argument("--eta", type=float, default=0.15)
    ap.add_argument("--n_contour_points", type=int, default=256)
    ap.add_argument("--transport_mode", type=str, default="affine", choices=["affine", "log_affine", "none"])
    ap.add_argument("--support_overlap_min", type=float, default=0.25)
    ap.add_argument("--active_argument_margin", type=float, default=0.75)
    ap.add_argument("--residue_error_margin", type=float, default=1.0)
    ap.add_argument("--trace_error_margin", type=float, default=1.0)
    ap.add_argument("--poisson_hard_threshold", type=float, default=0.85)
    ap.add_argument("--reward_mode", type=str, default="rank_log", choices=["rank_log", "inverse_J"])
    ap.add_argument("--semantic_modes", type=str, nargs="+", default=["numeric_only", "hybrid_ranked_anticollapse"])
    ap.add_argument("--reject_zero_support", action="store_true")
    ap.add_argument("--reject_nv_flatline", action="store_true")
    ap.add_argument("--min_nv_range", type=float, default=0.05)
    ap.add_argument("--max_nv_median", type=float, default=100.0)
    ap.add_argument("--progress_every", type=int, default=1)
    ap.add_argument("--seed", type=int, default=147)
    args = ap.parse_args()

    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rng = np.random.default_rng(int(args.seed))
    random.seed(int(args.seed))

    warnings: List[str] = []
    if not _HAVE_SAFE_EIGH:
        warnings.append("safe_eigh unavailable; using numpy eigvalsh fallback.")

    dims = [int(d) for d in args.dims]
    modes = [str(m) for m in args.semantic_modes]
    pvals = power_values(int(args.max_power))

    # Note: effective windows may be remapped per-dim (see effective_windows_for_target)
    L_grid = make_L_grid(args.L_min, args.L_max, args.n_L)

    # Targets: prefer true_levels_csv real_zeta/target, else zeros.
    target_by_dim: Dict[int, np.ndarray] = {}
    df_levels, lvl_warns = rd.load_true_levels_csv(_resolve(args.true_levels_csv), dims_keep=dims)
    warnings.extend([f"true_levels_csv: {w}" for w in lvl_warns])
    if df_levels is not None and not df_levels.empty:
        try:
            df = df_levels.copy()
            df["dim"] = pd.to_numeric(df["dim"], errors="coerce").astype("Int64") if _HAVE_PANDAS else df["dim"]
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
            x = np.asarray(zeros_unfolded, dtype=np.float64).copy()
            x = x[np.isfinite(x)]
            x.sort()
            target_by_dim[int(d)] = x
            warnings.append(f"dim={d}: using zeros target fallback (real_zeta missing).")

    # baselines
    pool, missing_baselines = load_baseline_pool(
        dims=dims,
        v13o14_dir=_resolve(args.v13o14_dir),
        v14_2_dir=_resolve(args.v14_2_dir),
        v14_5_dir=_resolve(args.v14_5_dir),
        v14_6b_dir=_resolve(args.v14_6b_dir),
    )
    warnings.extend([f"baseline_missing: {m}" for m in missing_baselines])

    # numeric pheromones per dim/mode
    pher: Dict[str, Dict[int, Dict[Tuple[int, int], float]]] = {m: {} for m in modes}
    for m in modes:
        for d in dims:
            pher[m][int(d)] = {(gi, pw): 1.0 for gi in range(1, int(d)) for pw in pvals}

    # Outputs
    transport_rows: List[Dict[str, Any]] = []
    arg_rows_all: List[Dict[str, Any]] = []
    residue_rows_all: List[Dict[str, Any]] = []
    trace_rows_all: List[Dict[str, Any]] = []
    nv_diag_rows: List[Dict[str, Any]] = []
    aco_history: List[Dict[str, Any]] = []
    ranking_rows: List[Dict[str, Any]] = []
    best_candidates: List[Dict[str, Any]] = []
    gate_rows: List[Dict[str, Any]] = []
    null_rows: List[Dict[str, Any]] = []

    # Best tracking
    best_by: Dict[Tuple[int, str], Dict[str, Any]] = {(int(d), m): {"J": float("inf"), "row": None} for d in dims for m in modes}

    for it in range(1, int(args.num_iters) + 1):
        for d in dims:
            tgt = target_by_dim[int(d)]
            windows_d, win_meta = effective_windows_for_target(tgt, args)
            if not windows_d:
                raise SystemExit("No effective windows produced; check window args and target levels.")
            tgt_nv = number_variance_curve(tgt, L_grid)
            for mode in modes:
                batch: List[Dict[str, Any]] = []
                for ant in range(int(args.num_ants)):
                    sem_term = 0.0 if mode == "numeric_only" else 0.8
                    L = int(rng.integers(2, int(args.max_word_len) + 1))
                    w: List[Tuple[int, int]] = []
                    for _k in range(L):
                        w.append(
                            sample_token(
                                dim=int(d),
                                pher=pher[mode][int(d)],
                                alpha=1.0,
                                beta=2.0,
                                semantic_term=float(sem_term),
                                rng=rng,
                            )
                        )
                    w = clamp_word_to_dim(w, int(d), int(args.max_power), int(args.max_word_len))
                    wstr = word_to_string(w)

                    # Stage A: operator + eigs
                    H, st = build_stabilized_operator(int(d), w, seed=int(args.seed + 100000 * it + 1000 * d + ant))
                    stable = bool(st.get("stable", False)) and H is not None
                    classification = "OK"
                    if not stable:
                        row = {
                            "iter": int(it),
                            "dim": int(d),
                            "mode": str(mode),
                            "ant_id": int(ant),
                            "word": wstr,
                            "word_len": int(len(w)),
                            "J_v14_7b": 1e6,
                            "final_reward": 0.0,
                            "rank_reward": 0.0,
                            "score_reward": 0.0,
                            "poisson_like_fraction": 1.0,
                            "support_overlap": 0.0,
                            "active_argument_error_med": 1.0,
                            "residue_error_med": 1.0,
                            "trace_error_med": 1.0,
                            "nv_range": float("nan"),
                            "nv_median": float("nan"),
                            "nv_rmse": float("nan"),
                            "classification": "UNSTABLE_OPERATOR",
                            "all_gate_pass": False,
                        }
                        batch.append(row)
                        aco_history.append(row)
                        continue

                    eigs = safe_eigvalsh(H, seed=int(args.seed + 7 * ant + 13 * it + 17 * d))
                    if eigs is None or eigs.size < 8:
                        row = {
                            "iter": int(it),
                            "dim": int(d),
                            "mode": str(mode),
                            "ant_id": int(ant),
                            "word": wstr,
                            "word_len": int(len(w)),
                            "J_v14_7b": 1e6,
                            "final_reward": 0.0,
                            "rank_reward": 0.0,
                            "score_reward": 0.0,
                            "poisson_like_fraction": 1.0,
                            "support_overlap": 0.0,
                            "active_argument_error_med": 1.0,
                            "residue_error_med": 1.0,
                            "trace_error_med": 1.0,
                            "nv_range": float("nan"),
                            "nv_median": float("nan"),
                            "nv_rmse": float("nan"),
                            "classification": "EIG_FAIL",
                            "all_gate_pass": False,
                        }
                        batch.append(row)
                        aco_history.append(row)
                        continue

                    op_levels = unfold_to_mean_spacing_one(eigs)

                    # Transport map
                    if args.transport_mode == "none":
                        tm = TransportMap("none", "none", 1.0, 0.0, 0.0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"))
                        op_cal = np.asarray(op_levels, dtype=np.float64)
                    elif args.transport_mode == "affine":
                        tm = transport_affine(op_levels, tgt)
                        op_cal = apply_transport(op_levels, tm)
                    else:
                        tm = transport_log_affine(op_levels, tgt)
                        op_cal = apply_transport(op_levels, tm)

                    transport_rows.append(
                        {
                            "dim": int(d),
                            "mode": str(mode),
                            "iter": int(it),
                            "ant_id": int(ant),
                            "word": wstr,
                            "transport_mode_requested": str(tm.mode_requested),
                            "transport_mode_effective": str(tm.mode_effective),
                            "scale": float(tm.scale),
                            "shift": float(tm.shift),
                            "log_c": float(tm.log_c),
                            "op_q10": float(tm.op_q10),
                            "op_q50": float(tm.op_q50),
                            "op_q90": float(tm.op_q90),
                            "target_q10": float(tm.tg_q10),
                            "target_q50": float(tm.tg_q50),
                            "target_q90": float(tm.tg_q90),
                        }
                    )

                    # Support + argument counts on calibrated levels
                    arg_med, _arg_mean, support, arg_rows = active_argument_counts(op_cal, tgt, windows_d)
                    for ar in arg_rows:
                        arg_rows_all.append({"iter": int(it), "dim": int(d), "mode": str(mode), "ant_id": int(ant), "word": wstr, **ar})

                    # Zero-support rejection
                    if args.reject_zero_support and support <= 0.0:
                        classification = "ZERO_SUPPORT_REJECT"

                    # NV diagnostics (calibrated)
                    # Important: transport scaling breaks "mean spacing = 1".
                    # For NV/Poisson diagnostics, re-unfold after calibration on both operator and target.
                    op_for_nv = unfold_to_mean_spacing_one(op_cal)
                    tgt_for_nv = unfold_to_mean_spacing_one(tgt)
                    op_nv = number_variance_curve(op_for_nv, L_grid)
                    tgt_nv_eff = number_variance_curve(tgt_for_nv, L_grid)
                    dnv = nv_diagnostics(op_nv, tgt_nv_eff, L_grid)
                    nv_range = float(dnv["nv_range"]) if math.isfinite(safe_float(dnv["nv_range"])) else 0.0
                    nv_median = safe_float(dnv["nv_median"])
                    nv_rmse = safe_float(dnv["nv_rmse"])
                    nv_flatline = bool(dnv["nv_flatline"])
                    nv_too_large = bool((math.isfinite(nv_median) and nv_median > float(args.max_nv_median)) or bool(dnv["nv_too_large"]))
                    nv_ok = bool(math.isfinite(nv_rmse) and nv_range >= float(args.min_nv_range) and (math.isfinite(nv_median) and nv_median <= float(args.max_nv_median)))
                    if args.reject_nv_flatline and (nv_flatline or (not nv_ok)):
                        if classification == "OK":
                            classification = "NV_COLLAPSE_OR_BAD_UNFOLDING"

                    # Poisson-like fraction from NV vs Poisson and vs target
                    pois_frac = poisson_like_fraction_from_nv(op_nv, tgt_nv_eff, L_grid)
                    if (support <= 0.0) or nv_flatline:
                        # Do not treat as genuine anti-Poisson signal
                        pois_frac = 1.0

                    # Residue / trace on calibrated levels
                    res_med, _leak, res_rows = residue_scores(op_cal, tgt, windows_d, eta=float(args.eta), n_contour_points=int(args.n_contour_points))
                    for rr in res_rows:
                        residue_rows_all.append({"iter": int(it), "dim": int(d), "mode": str(mode), "ant_id": int(ant), "word": wstr, **rr})
                    tr_med, tr_rows = trace_proxy_rows(op_cal, tgt, windows_d)
                    for tr in tr_rows:
                        trace_rows_all.append({"iter": int(it), "dim": int(d), "mode": str(mode), "ant_id": int(ant), "word": wstr, **tr})

                    # NV diagnostics rows
                    nv_diag_rows.append(
                        {
                            "dim": int(d),
                            "mode": str(mode),
                            "iter": int(it),
                            "ant_id": int(ant),
                            "word": wstr,
                            "nv_range": float(dnv["nv_range"]),
                            "nv_std": float(dnv["nv_std"]),
                            "nv_median": float(dnv["nv_median"]),
                            "nv_rmse": float(dnv["nv_rmse"]),
                            "nv_slope_roughness": float(dnv["nv_slope_roughness"]),
                            "nv_flatline": bool(nv_flatline),
                            "nv_too_large": bool(nv_too_large),
                            "G4_number_variance_ok": bool(nv_ok),
                        }
                    )

                    # Construct J_v14_7b (smoke: emphasize support+NV+arg)
                    J = 0.0
                    J += 10.0 * max(0.0, float(args.support_overlap_min) - float(support))
                    J += 3.0 * float(arg_med)
                    J += 2.0 * (0.0 if nv_ok else 1.0)
                    J += 2.0 * float(max(0.0, res_med - float(args.residue_error_margin)))
                    J += 1.0 * float(max(0.0, tr_med - float(args.trace_error_margin)))
                    J += 2.0 * float(pois_frac)
                    if classification == "ZERO_SUPPORT_REJECT":
                        J += 1e4
                    if classification == "NV_COLLAPSE_OR_BAD_UNFOLDING":
                        J += 1e4
                    J = float(min(max(J, 0.0), 1e6))

                    row = {
                        "iter": int(it),
                        "dim": int(d),
                        "mode": str(mode),
                        "ant_id": int(ant),
                        "word": wstr,
                        "word_len": int(len(w)),
                        "J_v14_7b": float(J),
                        "final_reward": float("nan"),  # filled later per batch if rank_log
                        "rank_reward": float("nan"),
                        "score_reward": float("nan"),
                        "poisson_like_fraction": float(pois_frac),
                        "support_overlap": float(support),
                        "active_argument_error_med": float(arg_med),
                        "residue_error_med": float(res_med),
                        "trace_error_med": float(tr_med),
                        "nv_range": float(nv_range),
                        "nv_median": float(nv_median) if math.isfinite(nv_median) else float("nan"),
                        "nv_rmse": float(nv_rmse) if math.isfinite(nv_rmse) else float("nan"),
                        "classification": str(classification),
                        "all_gate_pass": False,  # later for best per dim/mode
                    }
                    batch.append(row)
                    aco_history.append(row)

                    # best tracking
                    key = (int(d), str(mode))
                    if float(J) < float(best_by[key]["J"]):
                        best_by[key] = {"J": float(J), "row": row.copy()}

                # reward assignment within batch
                batch_sorted = sorted(batch, key=lambda r: float(r["J_v14_7b"]))
                for rank_i, r in enumerate(batch_sorted):
                    if args.reward_mode == "inverse_J":
                        rr = float("nan")
                        sr = float("nan")
                        fr = float(1.0 / (1.0 + float(r["J_v14_7b"])))
                    else:
                        rr = float(1.0 / math.sqrt(float(rank_i) + 1.0))
                        sr = float(1.0 / (1.0 + math.log1p(max(0.0, float(r["J_v14_7b"])))))
                        fr = float(0.5 * rr + 0.5 * sr)
                    # enforce hard rejects
                    if str(r.get("classification", "")) in ("ZERO_SUPPORT_REJECT", "NV_COLLAPSE_OR_BAD_UNFOLDING", "UNSTABLE_OPERATOR", "EIG_FAIL"):
                        fr = 0.0
                    fr = float(min(max(fr, 0.0), 1.0))
                    r["rank_reward"] = float(rr) if math.isfinite(rr) else float("nan")
                    r["score_reward"] = float(sr) if math.isfinite(sr) else float("nan")
                    r["final_reward"] = float(fr)

                # pheromone update: evaporate + deposit from top elites by final_reward
                for tok in list(pher[mode][int(d)].keys()):
                    pher[mode][int(d)][tok] = float(max(1e-6, 0.85 * pher[mode][int(d)][tok]))
                elites = [r for r in batch_sorted[: max(1, int(0.2 * len(batch_sorted)))] if float(r.get("final_reward", 0.0)) > 0.0]
                for r in elites:
                    dep = float(r.get("final_reward", 0.0))
                    if dep <= 0:
                        continue
                    for tokstr in str(r.get("word", "")).split():
                        try:
                            left, pstr = tokstr.split("^")
                            gi = int(left.split("_")[1])
                            pw = int(pstr)
                            if (gi, pw) in pher[mode][int(d)]:
                                pher[mode][int(d)][(gi, pw)] = float(min(1e6, pher[mode][int(d)][(gi, pw)] + dep))
                        except Exception:
                            continue

        if it == 1 or it % max(1, int(args.progress_every)) == 0 or it == int(args.num_iters):
            elapsed = time.perf_counter() - t0
            eta = (elapsed / max(1, it)) * max(0, int(args.num_iters) - it)
            bestJ = min(float(best_by[(int(d), m)]["J"]) for d in dims for m in modes)
            print(f"[V14.7b] iter={it}/{int(args.num_iters)} best_J={bestJ:.6g} eta={format_seconds(eta)}", flush=True)

    # Build best candidates / gates / null comparisons
    for d in dims:
        for mode in modes:
            br = best_by[(int(d), str(mode))]["row"]
            if br is None:
                continue
            best_candidates.append(
                {
                    "dim": int(d),
                    "mode": str(mode),
                    "best_word": str(br["word"]),
                    "best_J_v14_7b": float(br["J_v14_7b"]),
                    "best_final_reward": float(br.get("final_reward", float("nan"))),
                    "best_support_overlap": float(br["support_overlap"]),
                    "best_poisson_like_fraction": float(br["poisson_like_fraction"]),
                    "best_nv_range": float(br["nv_range"]),
                    "best_nv_median": float(br["nv_median"]) if math.isfinite(safe_float(br["nv_median"])) else float("nan"),
                }
            )

            # gates
            G1 = True  # stable for best row (otherwise J huge)
            G2 = bool(float(br["support_overlap"]) >= float(args.support_overlap_min))
            G3 = bool(float(br["active_argument_error_med"]) <= float(args.active_argument_margin))
            # NV gate from diagnostics thresholds
            G4 = bool(float(br["nv_range"]) >= float(args.min_nv_range) and (math.isfinite(safe_float(br["nv_median"])) and float(br["nv_median"]) <= float(args.max_nv_median)) and math.isfinite(safe_float(br["nv_rmse"])))
            G5 = bool(float(br["poisson_like_fraction"]) < float(args.poisson_hard_threshold) and (not (str(br["classification"]) in ("ZERO_SUPPORT_REJECT", "NV_COLLAPSE_OR_BAD_UNFOLDING"))))
            G6 = bool(float(br["residue_error_med"]) <= float(args.residue_error_margin))
            G7 = bool(float(br["trace_error_med"]) <= float(args.trace_error_margin))

            # null comparisons
            best_random = best_baseline_J(pool, int(d), "random")
            best_rej = best_baseline_J(pool, int(d), "rejected")
            best_abl = best_baseline_J(pool, int(d), "ablation")
            best_prior = best_baseline_J(pool, int(d), "prior_artin")
            best_null = best_null_J(pool, int(d))
            null_sep = float(best_null - float(br["J_v14_7b"])) if (best_null is not None and math.isfinite(best_null)) else float("nan")
            dist = null_dist(pool, int(d))
            null_z = float("nan")
            null_pct = float("nan")
            if len(dist) >= 3:
                mu = float(np.mean(dist))
                sd = float(np.std(dist))
                null_z = float((mu - float(br["J_v14_7b"])) / sd) if sd > 1e-12 else float("nan")
                null_pct = float(np.mean([1.0 if x <= float(br["J_v14_7b"]) else 0.0 for x in dist]))

            beats_random = bool(best_random is not None and float(br["J_v14_7b"]) < float(best_random))
            beats_rej = bool(best_rej is not None and float(br["J_v14_7b"]) < float(best_rej))
            beats_abl = bool(best_abl is not None and float(br["J_v14_7b"]) < float(best_abl))
            beats_prior = bool(best_prior is not None and float(br["J_v14_7b"]) < float(best_prior))
            missing_str = []
            if best_random is None:
                missing_str.append("random")
            if best_rej is None:
                missing_str.append("rejected")
            if best_abl is None:
                missing_str.append("ablation")
            if best_prior is None:
                missing_str.append("prior_artin")

            null_rows.append(
                {
                    "dim": int(d),
                    "mode": str(mode),
                    "best_word": str(br["word"]),
                    "best_J_v14_7b": float(br["J_v14_7b"]),
                    "best_random_J": float(best_random) if best_random is not None else float("nan"),
                    "best_rejected_J": float(best_rej) if best_rej is not None else float("nan"),
                    "best_ablation_J": float(best_abl) if best_abl is not None else float("nan"),
                    "best_prior_artin_J": float(best_prior) if best_prior is not None else float("nan"),
                    "best_null_J": float(best_null) if best_null is not None else float("nan"),
                    "null_separation": float(null_sep),
                    "null_zscore": float(null_z),
                    "null_percentile": float(null_pct),
                    "beats_random": bool(beats_random),
                    "beats_rejected": bool(beats_rej),
                    "beats_ablation": bool(beats_abl),
                    "beats_prior_artin": bool(beats_prior),
                    "missing_baselines": "|".join(missing_str),
                }
            )

            G8 = bool(beats_random) if best_random is not None else False
            G9 = bool(beats_rej) if best_rej is not None else False
            G10 = bool(beats_abl) if best_abl is not None else False
            G11 = bool(beats_prior) if best_prior is not None else False
            G12 = True  # no semantic collapse check in smoke; always True
            all_gate = bool(G1 and G2 and G3 and G4 and G5 and G6 and G7 and G8 and G9 and G10 and G11 and G12)
            gate_rows.append(
                {
                    "dim": int(d),
                    "mode": str(mode),
                    "word": str(br["word"]),
                    "J_v14_7b": float(br["J_v14_7b"]),
                    "final_reward": float(br.get("final_reward", float("nan"))),
                    "poisson_like_fraction": float(br["poisson_like_fraction"]),
                    "support_overlap": float(br["support_overlap"]),
                    "G1_stable": bool(G1),
                    "G2_support_overlap_ok": bool(G2),
                    "G3_active_argument_ok": bool(G3),
                    "G4_number_variance_ok": bool(G4),
                    "G5_not_poisson_like": bool(G5),
                    "G6_residue_error_ok": bool(G6),
                    "G7_trace_proxy_ok": bool(G7),
                    "G8_beats_random_controls": bool(G8),
                    "G9_beats_rejected_control": bool(G9),
                    "G10_beats_ablation_controls": bool(G10),
                    "G11_beats_prior_artin_search": bool(G11),
                    "G12_no_semantic_collapse": bool(G12),
                    "G13_all_gate_pass": bool(all_gate),
                    "classification": str(br.get("classification", "")) if not all_gate else "PASS_SUPPORT_CALIBRATED_SMOKE",
                }
            )

    # Candidate ranking (top per dim/mode by J)
    for d in dims:
        for mode in modes:
            sub = [r for r in aco_history if int(r["dim"]) == int(d) and str(r["mode"]) == str(mode)]
            sub.sort(key=lambda r: float(r["J_v14_7b"]))
            for k, r in enumerate(sub[:200], start=1):
                ranking_rows.append(
                    {
                        "dim": int(d),
                        "mode": str(mode),
                        "rank": int(k),
                        "word": str(r["word"]),
                        "word_len": int(r["word_len"]),
                        "J_v14_7b": float(r["J_v14_7b"]),
                        "final_reward": float(r["final_reward"]),
                        "support_overlap": float(r["support_overlap"]),
                        "poisson_like_fraction": float(r["poisson_like_fraction"]),
                        "nv_range": float(r["nv_range"]) if math.isfinite(safe_float(r["nv_range"])) else float("nan"),
                        "classification": str(r["classification"]),
                    }
                )

    # Write outputs (headers always)
    write_csv(
        out_dir / "v14_7b_transport_maps.csv",
        fieldnames=[
            "dim",
            "mode",
            "iter",
            "ant_id",
            "word",
            "transport_mode_requested",
            "transport_mode_effective",
            "scale",
            "shift",
            "log_c",
            "op_q10",
            "op_q50",
            "op_q90",
            "target_q10",
            "target_q50",
            "target_q90",
        ],
        rows=transport_rows,
    )
    write_csv(
        out_dir / "v14_7b_active_argument_counts.csv",
        fieldnames=list(arg_rows_all[0].keys()) if arg_rows_all else ["iter", "dim", "mode", "ant_id", "word", "window_a", "window_b", "N_operator", "N_target", "N_error", "N_error_norm", "active_window"],
        rows=arg_rows_all,
    )
    write_csv(
        out_dir / "v14_7b_residue_scores.csv",
        fieldnames=list(residue_rows_all[0].keys()) if residue_rows_all else ["iter", "dim", "mode", "ant_id", "word", "window_a", "window_b", "residue_count_error", "residue_imag_leak"],
        rows=residue_rows_all,
    )
    write_csv(
        out_dir / "v14_7b_trace_proxy.csv",
        fieldnames=list(trace_rows_all[0].keys()) if trace_rows_all else ["iter", "dim", "mode", "ant_id", "word", "window_a", "window_b", "center", "sigma", "S_operator", "S_target", "trace_error_norm"],
        rows=trace_rows_all,
    )
    write_csv(
        out_dir / "v14_7b_nv_diagnostics.csv",
        fieldnames=list(nv_diag_rows[0].keys()) if nv_diag_rows else ["dim", "mode", "iter", "ant_id", "word", "nv_range", "nv_std", "nv_median", "nv_rmse", "nv_slope_roughness", "nv_flatline", "nv_too_large", "G4_number_variance_ok"],
        rows=nv_diag_rows,
    )
    write_csv(
        out_dir / "v14_7b_aco_history.csv",
        fieldnames=list(aco_history[0].keys()) if aco_history else ["iter", "dim", "mode", "ant_id", "word", "J_v14_7b", "final_reward"],
        rows=aco_history,
    )
    write_csv(
        out_dir / "v14_7b_candidate_ranking.csv",
        fieldnames=list(ranking_rows[0].keys()) if ranking_rows else ["dim", "mode", "rank", "word", "J_v14_7b"],
        rows=ranking_rows,
    )
    write_csv(
        out_dir / "v14_7b_best_candidates.csv",
        fieldnames=list(best_candidates[0].keys()) if best_candidates else ["dim", "mode", "best_word", "best_J_v14_7b"],
        rows=best_candidates,
    )
    write_csv(
        out_dir / "v14_7b_gate_summary.csv",
        fieldnames=list(gate_rows[0].keys()) if gate_rows else ["dim", "mode", "G13_all_gate_pass"],
        rows=gate_rows,
    )
    write_csv(
        out_dir / "v14_7b_null_comparisons.csv",
        fieldnames=list(null_rows[0].keys()) if null_rows else ["dim", "mode", "best_J_v14_7b"],
        rows=null_rows,
    )

    # Results + report
    bestJ = min(float(r["best_J_v14_7b"]) for r in best_candidates) if best_candidates else float("inf")
    # smoke success criteria
    active_nonempty = bool(len(arg_rows_all) > 0)
    positive_support = any(float(r.get("support_overlap", 0.0)) > 0.0 for r in aco_history)
    reward_unique6 = len(set(round(float(r.get("final_reward", 0.0)), 6) for r in aco_history if math.isfinite(safe_float(r.get("final_reward", float("nan"))))))
    reward_diverse = bool(reward_unique6 > 10)
    not_all_poisson = any(float(r.get("poisson_like_fraction", 1.0)) < 1.0 for r in aco_history)
    nv_not_flat = any(bool(r.get("G4_number_variance_ok", False)) for r in nv_diag_rows)
    improves_v147 = bool(math.isfinite(bestJ) and bestJ < 18.98377793000534)

    results = {
        "version": "v14_7b",
        "out_dir": str(out_dir),
        "dims": dims,
        "warnings": warnings,
        "missing_baselines": missing_baselines,
        "smoke_success_checks": {
            "active_argument_counts_nonempty": active_nonempty,
            "positive_support_exists": positive_support,
            "reward_diverse(unique6>10)": reward_diverse,
            "not_all_poisson": not_all_poisson,
            "nv_not_flat_exists": nv_not_flat,
            "improves_vs_v14_7_smoke_J~18.9838": improves_v147,
            "best_J_v14_7b": float(bestJ) if math.isfinite(bestJ) else None,
        },
        "notes": ["Computational evidence only; not a proof of RH."],
    }
    write_text(out_dir / "v14_7b_results.json", json.dumps(json_sanitize(results), indent=2, ensure_ascii=False) + "\n")

    OUT_ABS = str(out_dir.resolve())
    md = []
    md.append("# V14.7b — Support-Calibrated Anti-Poisson Smoke\n\n")
    md.append("> Computational evidence only; not a proof of RH.\n\n")
    md.append("## Context: why V14.7 failed (smoke)\n")
    md.append("- zero support_overlap\n- empty active_argument_counts\n- poisson_like_fraction≈1.0\n- reward collapse\n- pathological operator number variance\n\n")
    md.append("## What V14.7b adds\n")
    md.append("- support pre-calibration (transport) before all diagnostics\n")
    md.append("- zero-support hard rejection\n")
    md.append("- NV flatline/pathology rejection\n")
    md.append("- rank-log reward (or inverse_J) to restore reward diversity\n\n")
    md.append("## Smoke success criteria (computed)\n")
    md.append(f"- support_overlap>0 exists: **{positive_support}**\n")
    md.append(f"- active_argument_counts non-empty: **{active_nonempty}**\n")
    md.append(f"- NV not flatline exists: **{nv_not_flat}**\n")
    md.append(f"- poisson_like_fraction not always 1.0: **{not_all_poisson}**\n")
    md.append(f"- final_reward unique rounded(1e-6) > 10: **{reward_diverse}** (unique6={reward_unique6})\n")
    md.append(f"- best J improves vs V14.7 smoke J≈18.98377793: **{improves_v147}** (best_J={bestJ:.6g})\n\n")
    md.append("## Default smoke command\n")
    md.append("```bash\n")
    md.append('OUT=runs/v14_7b_support_calibrated_antipoisson_smoke\nmkdir -p "$OUT"\n\n')
    md.append("caffeinate -dimsu python3 scripts/run_v14_7b_support_calibrated_antipoisson.py \\\n")
    md.append("  --true_levels_csv runs/v13o9_true_unfolded_spectra_from_v13o8/v13o9_unfolded_levels.csv \\\n")
    md.append("  --zeros_csv runs/zeros_100_400_precise.csv \\\n")
    md.append("  --v13o14_dir runs/v13o14_transport_null_controls \\\n")
    md.append("  --v14_2_dir runs/v14_2_stabilized_artin_operator_search \\\n")
    md.append("  --v14_5_dir runs/v14_5_semantic_anticollapse_search \\\n")
    md.append("  --v14_6b_dir runs/v14_6b_real_null_control_stage_e \\\n")
    md.append('  --out_dir "$OUT" \\\n')
    md.append("  --dims 64 \\\n")
    md.append("  --num_ants 16 \\\n")
    md.append("  --num_iters 10 \\\n")
    md.append("  --max_word_len 16 \\\n")
    md.append("  --max_power 3 \\\n")
    md.append("  --window_min 100 \\\n")
    md.append("  --window_max 220 \\\n")
    md.append("  --window_size 40 \\\n")
    md.append("  --window_stride 20 \\\n")
    md.append("  --L_min 0.5 \\\n")
    md.append("  --L_max 8 \\\n")
    md.append("  --n_L 16 \\\n")
    md.append("  --transport_mode affine \\\n")
    md.append("  --reject_zero_support \\\n")
    md.append("  --reject_nv_flatline \\\n")
    md.append("  --min_nv_range 0.05 \\\n")
    md.append("  --max_nv_median 100 \\\n")
    md.append("  --support_overlap_min 0.25 \\\n")
    md.append("  --active_argument_margin 0.75 \\\n")
    md.append("  --residue_error_margin 1.0 \\\n")
    md.append("  --trace_error_margin 1.0 \\\n")
    md.append("  --poisson_hard_threshold 0.85 \\\n")
    md.append("  --reward_mode rank_log \\\n")
    md.append("  --semantic_modes numeric_only hybrid_ranked_anticollapse \\\n")
    md.append("  --progress_every 1 \\\n")
    md.append("  --seed 147\n")
    md.append("```\n\n")
    md.append("## Verification commands\n")
    md.append("```bash\n")
    md.append('OUT=runs/v14_7b_support_calibrated_antipoisson_smoke\n\n')
    md.append('echo "=== FILES ==="\nfind "$OUT" -maxdepth 1 -type f | sort\n\n')
    md.append('echo "=== BEST ==="\ncolumn -s, -t < "$OUT"/v14_7b_best_candidates.csv\n\n')
    md.append('echo "=== GATE ==="\ncolumn -s, -t < "$OUT"/v14_7b_gate_summary.csv\n\n')
    md.append('echo "=== TRANSPORT MAPS ==="\ncolumn -s, -t < "$OUT"/v14_7b_transport_maps.csv | head -80\n\n')
    md.append('echo "=== ACTIVE ARGUMENT COUNTS ==="\ncolumn -s, -t < "$OUT"/v14_7b_active_argument_counts.csv | head -80\n\n')
    md.append('echo "=== NV DIAGNOSTICS ==="\ncolumn -s, -t < "$OUT"/v14_7b_nv_diagnostics.csv | head -80\n\n')
    md.append('echo "=== RANKING ==="\ncolumn -s, -t < "$OUT"/v14_7b_candidate_ranking.csv | head -60\n\n')
    md.append('echo "=== NULL COMPARISONS ==="\ncolumn -s, -t < "$OUT"/v14_7b_null_comparisons.csv | head -80\n\n')
    md.append('echo "=== REPORT ==="\nhead -220 "$OUT"/v14_7b_report.md\n')
    md.append("```\n\n")
    md.append("> Computational evidence only; not a proof of RH.\n")
    write_text(out_dir / "v14_7b_report.md", "".join(md))

    tex = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\begin{document}
\section*{V14.7b --- Support-Calibrated Anti-Poisson Smoke}
\textbf{Computational evidence only; not a proof of RH.}

This is a diagnostic smoke to validate that support calibration yields non-empty active windows,
that number variance is not pathological flatline, and that the reward distribution is usable.
\end{document}
"""
    tex_path = out_dir / "v14_7b_report.tex"
    write_text(tex_path, tex)
    if _find_pdflatex():
        try_pdflatex(tex_path, out_dir, "v14_7b_report.pdf")

    elapsed = time.perf_counter() - t0
    print(f"[V14.7b] done in {format_seconds(elapsed)} out_dir={out_dir}", flush=True)


if __name__ == "__main__":
    main()

