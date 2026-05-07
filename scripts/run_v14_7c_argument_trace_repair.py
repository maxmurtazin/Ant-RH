#!/usr/bin/env python3
"""
V14.7c — Argument-Trace Repair for Support-Calibrated Artin-DTES Candidates

Computational evidence only; not a proof of RH.

This script performs a local repair search around the best V14.7b candidates
to improve the two diagnostics that still fail the all-gate in V14.7b:
  - active argument count error
  - trace proxy error

It intentionally does NOT restart from scratch; it mutates Artin words locally
while preserving properties already working (support overlap, anti-Poisson, etc.).
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
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
# Small utilities
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
# Stabilized operator construction (match V14.7b)
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
# Word parsing / formatting
# ----------------------------


def word_to_string(word: List[Tuple[int, int]]) -> str:
    return " ".join([f"sigma_{int(i)}^{int(p)}" for i, p in word])


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
    return out


def parse_artin_word(s: str, *, dim: int, max_power: int, warnings: List[str]) -> List[Tuple[int, int]]:
    toks = str(s or "").strip().split()
    out: List[Tuple[int, int]] = []
    for t in toks:
        tt = t.strip()
        if not tt:
            continue
        try:
            left, pstr = tt.split("^")
            if not left.startswith("sigma_"):
                warnings.append(f"skip_token(no_sigma_prefix): {tt}")
                continue
            gi = int(left.split("_", 1)[1])
            pw = int(pstr)
            if gi < 1 or gi >= int(dim):
                warnings.append(f"skip_token(generator_out_of_range dim={dim}): {tt}")
                continue
            if pw == 0 or abs(pw) > int(max_power):
                warnings.append(f"skip_token(power_out_of_range max_power={max_power}): {tt}")
                continue
            out.append((gi, pw))
        except Exception:
            warnings.append(f"skip_token(parse_fail): {tt}")
            continue
    out = clamp_word_to_dim(out, int(dim), int(max_power), max_word_len=10**9)
    return out


# ----------------------------
# Diversity / anti-collapse
# ----------------------------


def generator_entropy(word: List[Tuple[int, int]], n_generators: int) -> float:
    gens = [int(i) for i, _p in word]
    if not gens or n_generators <= 1:
        return 0.0
    c = Counter(gens)
    total = float(sum(c.values()))
    ps = np.asarray([v / total for v in c.values()], dtype=np.float64)
    H = float(-np.sum(ps * np.log(np.maximum(1e-12, ps))))
    return float(H / max(1e-12, math.log(float(n_generators))))


def power_entropy(word: List[Tuple[int, int]], max_power: int) -> float:
    ps = [int(p) for _i, p in word]
    if not ps:
        return 0.0
    n_bins = max(2, 2 * int(max_power))
    c = Counter(ps)
    total = float(sum(c.values()))
    probs = np.asarray([v / total for v in c.values()], dtype=np.float64)
    H = float(-np.sum(probs * np.log(np.maximum(1e-12, probs))))
    return float(H / max(1e-12, math.log(float(n_bins))))


def collapse_penalty(word: List[Tuple[int, int]], *, dim: int) -> Tuple[float, Dict[str, Any]]:
    L = int(len(word))
    gens = [int(i) for i, _p in word]
    uniq = int(len(set(gens)))
    ent = float(generator_entropy(word, max(1, int(dim) - 1)))
    low_len = float(max(0, 4 - L))
    one_gen = 1.0 if uniq <= 1 else 0.0
    low_uniq = float(max(0, 3 - uniq))
    low_ent = float(max(0.0, 0.25 - ent))
    # repeated generator >4 consecutively
    rep_run = 0
    cur = 1
    for k in range(1, len(gens)):
        if gens[k] == gens[k - 1]:
            cur += 1
        else:
            rep_run = max(rep_run, cur)
            cur = 1
    rep_run = max(rep_run, cur) if gens else 0
    rep_pen = float(max(0, rep_run - 4))
    penalty = float(low_len + 2.0 * one_gen + 1.0 * low_uniq + 3.0 * low_ent + 0.5 * rep_pen)
    return penalty, {
        "word_len": L,
        "unique_generator_count": uniq,
        "generator_entropy": ent,
        "max_consecutive_generator_run": int(rep_run),
        "collapse_penalty": float(penalty),
    }


# ----------------------------
# Edit distance (token-level, cutoff)
# ----------------------------


def edit_distance_cutoff(a: List[Tuple[int, int]], b: List[Tuple[int, int]], cutoff: int) -> int:
    """
    Levenshtein distance over tokens (generator,power). Stops early above cutoff.
    """
    la, lb = len(a), len(b)
    if abs(la - lb) > cutoff:
        return cutoff + 1
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur0 = i
        cur = [cur0]
        row_min = cur0
        ai = a[i - 1]
        for j in range(1, lb + 1):
            cost = 0 if ai == b[j - 1] else 1
            v = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            cur.append(v)
            row_min = min(row_min, v)
        prev = cur
        if row_min > cutoff:
            return cutoff + 1
    return prev[-1]


# ----------------------------
# Diagnostics (match V14.7b calibrated approach)
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


def apply_transport(op: np.ndarray, tm: TransportMap) -> np.ndarray:
    x = np.asarray(op, dtype=np.float64)
    x = x[np.isfinite(x)]
    if tm.mode_effective == "none":
        x.sort()
        return x
    y = tm.scale * x + tm.shift
    y = np.asarray(y, dtype=np.float64)
    y = y[np.isfinite(y)]
    y.sort()
    return y


def unfold_to_mean_spacing_one(x: np.ndarray) -> np.ndarray:
    return rd.unfold_to_mean_spacing_one(np.asarray(x, dtype=np.float64).reshape(-1))


def make_L_grid() -> np.ndarray:
    return np.linspace(0.5, 8.0, 16, dtype=np.float64)


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
        return {"nv_range": float("nan"), "nv_median": float("nan"), "nv_rmse": float("nan")}
    yy = y[m]
    return {
        "nv_range": float(np.max(yy) - np.min(yy)),
        "nv_median": float(np.median(yy)),
        "nv_rmse": float(curve_l2(op_nv, tgt_nv)),
    }


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
    dP = np.abs(op - L)
    dT = np.abs(op - tg)
    return float(np.mean((dP < dT).astype(np.float64)))


def effective_windows_for_target(tgt: np.ndarray) -> Tuple[List[Tuple[float, float]], Dict[str, Any]]:
    """
    Use the V14.7b windowing defaults, but remap if target has zero counts in all windows.
    """
    window_min, window_max, window_size, window_stride = (100.0, 220.0, 40.0, 20.0)
    tgt = np.asarray(tgt, dtype=np.float64)
    tgt = tgt[np.isfinite(tgt)]
    tgt.sort()
    meta = {
        "window_mode_effective": "requested",
        "window_min_effective": float(window_min),
        "window_max_effective": float(window_max),
        "window_size_effective": float(window_size),
        "window_stride_effective": float(window_stride),
        "target_min": float(np.min(tgt)) if tgt.size else float("nan"),
        "target_max": float(np.max(tgt)) if tgt.size else float("nan"),
    }
    win = rd.make_windows(window_min, window_max, window_size, window_stride)
    if not win or tgt.size < 8:
        return win, meta
    any_counts = any(int(rd.count_in_window(tgt, float(a), float(b))) > 0 for (a, b) in win)
    if any_counts:
        return win, meta
    q10, q90 = np.quantile(tgt, [0.1, 0.9])
    span_req = max(1e-12, float(window_max) - float(window_min))
    span_tgt = max(1e-12, float(q90) - float(q10))
    size_eff = float(span_tgt * float(window_size) / span_req)
    stride_eff = float(span_tgt * float(window_stride) / span_req)
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


# ----------------------------
# Baselines / null comparisons (best-effort, match V14.7b spirit)
# ----------------------------


def load_baseline_pool(dims: List[int], v13o14_dir: Path, v14_2_dir: Path, v14_5_dir: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
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

    for d in dims:
        any_dim = any(int(r.get("dim", 0)) == int(d) and not bool(r.get("missing_source", False)) and math.isfinite(safe_float(r.get("J", float("nan")))) for r in pool)
        if not any_dim:
            add(d, "missing", float("nan"), "missing", "no_baselines_loaded", "", "no baselines loaded", True)
    return pool, sorted(set(missing))


def best_baseline_J(pool: List[Dict[str, Any]], dim: int, kind: str) -> Optional[float]:
    Js: List[float] = []
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
    Js: List[float] = []
    for r in pool:
        if int(r.get("dim", 0)) != int(dim):
            continue
        if bool(r.get("missing_source", False)):
            continue
        kind = str(r.get("baseline_kind", ""))
        if kind in ("missing", "primary"):
            continue
        J = safe_float(r.get("J", float("nan")))
        if math.isfinite(J):
            Js.append(float(J))
    return float(min(Js)) if Js else None


def null_dist(pool: List[Dict[str, Any]], dim: int) -> List[float]:
    xs: List[float] = []
    for r in pool:
        if int(r.get("dim", 0)) != int(dim):
            continue
        if bool(r.get("missing_source", False)):
            continue
        kind = str(r.get("baseline_kind", ""))
        if kind in ("missing", "primary"):
            continue
        J = safe_float(r.get("J", float("nan")))
        if math.isfinite(J):
            xs.append(float(J))
    return xs


# ----------------------------
# Mutation operators (local repair)
# ----------------------------


@dataclass(frozen=True)
class MutationConfig:
    mutation_insert_prob: float
    mutation_delete_prob: float
    mutation_replace_prob: float
    mutation_power_flip_prob: float
    mutation_local_generator_jitter_prob: float
    local_generator_radius: int


def _biased_generator(rng: random.Random, *, dim: int, center: int, radius: int, preferred: Sequence[int]) -> int:
    if preferred and rng.random() < 0.7:
        g0 = int(preferred[int(rng.randrange(0, len(preferred)))])
        g = g0 + int(rng.randint(-radius, radius))
    else:
        g = int(center + rng.randint(-radius, radius))
    return int(max(1, min(int(dim) - 1, g)))


def mutate_word(
    parent: List[Tuple[int, int]],
    *,
    dim: int,
    max_power: int,
    max_word_len: int,
    preferred_generators: Sequence[int],
    cfg: MutationConfig,
    rng: random.Random,
) -> Tuple[List[Tuple[int, int]], str]:
    w = list(parent)
    if not w:
        return w, "noop_empty"
    mut_kind = "replace"
    r = rng.random()
    if r < cfg.mutation_delete_prob:
        mut_kind = "delete"
    elif r < cfg.mutation_delete_prob + cfg.mutation_insert_prob:
        mut_kind = "insert"
    elif r < cfg.mutation_delete_prob + cfg.mutation_insert_prob + cfg.mutation_power_flip_prob:
        mut_kind = "power_flip"
    elif r < cfg.mutation_delete_prob + cfg.mutation_insert_prob + cfg.mutation_power_flip_prob + cfg.mutation_local_generator_jitter_prob:
        mut_kind = "local_generator_jitter"
    else:
        mut_kind = "replace"

    pvals = power_values(int(max_power))
    center = int(round(0.5 * (dim - 1)))

    if mut_kind == "delete":
        if len(w) <= 2:
            mut_kind = "replace"
        else:
            idx = int(rng.randrange(0, len(w)))
            del w[idx]
            return simplify_word(clamp_word_to_dim(w, dim, max_power, max_word_len), max_power=max_power, max_word_len=max_word_len), "delete"

    if mut_kind == "insert":
        if len(w) >= int(max_word_len):
            mut_kind = "replace"
        else:
            idx = int(rng.randrange(0, len(w) + 1))
            if w:
                anchor = w[max(0, min(len(w) - 1, idx - 1))][0]
            else:
                anchor = center
            gi = _biased_generator(rng, dim=dim, center=int(anchor), radius=int(cfg.local_generator_radius), preferred=preferred_generators)
            pw = int(pvals[int(rng.randrange(0, len(pvals)))])
            w.insert(idx, (int(gi), int(pw)))
            return simplify_word(clamp_word_to_dim(w, dim, max_power, max_word_len), max_power=max_power, max_word_len=max_word_len), "insert"

    idx = int(rng.randrange(0, len(w)))
    gi, pw = w[idx]

    if mut_kind == "local_generator_jitter":
        gi2 = int(gi) + int(rng.randint(-int(cfg.local_generator_radius), int(cfg.local_generator_radius)))
        gi2 = int(max(1, min(int(dim) - 1, gi2)))
        w[idx] = (gi2, int(pw))
        return simplify_word(clamp_word_to_dim(w, dim, max_power, max_word_len), max_power=max_power, max_word_len=max_word_len), "local_generator_jitter"

    if mut_kind == "power_flip":
        if rng.random() < 0.5:
            pw2 = -int(pw)
        else:
            pw2 = int(math.copysign(int(rng.randint(1, int(max_power))), int(pw)))
        pw2 = int(max(-int(max_power), min(int(max_power), pw2)))
        if pw2 == 0:
            pw2 = 1
        w[idx] = (int(gi), int(pw2))
        return simplify_word(clamp_word_to_dim(w, dim, max_power, max_word_len), max_power=max_power, max_word_len=max_word_len), "power_flip"

    # replace: swap generator and/or power, with motif bias
    anchor = int(gi)
    gi2 = _biased_generator(rng, dim=dim, center=anchor, radius=int(cfg.local_generator_radius), preferred=preferred_generators)
    if rng.random() < 0.4:
        pw2 = int(pvals[int(rng.randrange(0, len(pvals)))])
    else:
        pw2 = int(pw)
    w[idx] = (int(gi2), int(pw2))
    return simplify_word(clamp_word_to_dim(w, dim, max_power, max_word_len), max_power=max_power, max_word_len=max_word_len), "replace"


# ----------------------------
# Candidate evaluation for V14.7c
# ----------------------------


@dataclass
class EvalOut:
    stable: bool
    metric_nan_flag: bool
    word_len: int
    generator_entropy: float
    power_entropy: float
    support_overlap: float
    poisson_like_fraction: float
    active_argument_error_med: float
    residue_error_med: float
    trace_error_med: float
    nv_range: float
    nv_median: float
    null_fail_penalty: int
    beats_random: bool
    beats_rejected: bool
    beats_ablation: bool
    beats_prior_artin: bool
    classification: str
    J_v14_7c: float
    final_reward: float
    # details (for best-candidate outputs)
    arg_rows: List[Dict[str, Any]]
    trace_rows: List[Dict[str, Any]]
    residue_rows: List[Dict[str, Any]]
    nv_rows: List[Dict[str, Any]]


def finite_or_penalty(x: float, *, penalty: float, flags: Dict[str, bool], key: str) -> float:
    if not math.isfinite(float(x)):
        flags[key] = True
        return float(penalty)
    return float(x)


def evaluate_candidate(
    *,
    dim: int,
    word: List[Tuple[int, int]],
    seed_word: List[Tuple[int, int]],
    target_levels: np.ndarray,
    windows: List[Tuple[float, float]],
    L_grid: np.ndarray,
    baselines: List[Dict[str, Any]],
    args: argparse.Namespace,
    rng_seed: int,
    missing_baselines_by_dim: Dict[int, List[str]],
) -> EvalOut:
    nan_flags: Dict[str, bool] = {}
    penalty_big = 1e3

    w_clamped = clamp_word_to_dim(word, int(dim), int(args.max_power), int(args.max_word_len))
    if not w_clamped:
        return EvalOut(
            stable=False,
            metric_nan_flag=True,
            word_len=0,
            generator_entropy=0.0,
            power_entropy=0.0,
            support_overlap=0.0,
            poisson_like_fraction=1.0,
            active_argument_error_med=1.0,
            residue_error_med=1.0,
            trace_error_med=1.0,
            nv_range=0.0,
            nv_median=float("nan"),
            null_fail_penalty=4,
            beats_random=False,
            beats_rejected=False,
            beats_ablation=False,
            beats_prior_artin=False,
            classification="UNSTABLE_OPERATOR",
            J_v14_7c=1e6,
            final_reward=0.0,
            arg_rows=[],
            trace_rows=[],
            residue_rows=[],
            nv_rows=[],
        )

    # anti-collapse
    collapse_pen, collapse_meta = collapse_penalty(w_clamped, dim=int(dim))
    gen_ent = float(collapse_meta["generator_entropy"])
    pow_ent = float(power_entropy(w_clamped, int(args.max_power)))
    no_collapse = bool(
        int(collapse_meta["word_len"]) >= 4 and int(collapse_meta["unique_generator_count"]) >= 3 and float(gen_ent) >= 0.25 and float(collapse_meta["max_consecutive_generator_run"]) <= 8
    )

    # operator
    H, st = build_stabilized_operator(int(dim), w_clamped, seed=int(rng_seed))
    stable = bool(st.get("stable", False)) and H is not None
    if not stable:
        return EvalOut(
            stable=False,
            metric_nan_flag=True,
            word_len=int(len(w_clamped)),
            generator_entropy=float(gen_ent),
            power_entropy=float(pow_ent),
            support_overlap=0.0,
            poisson_like_fraction=1.0,
            active_argument_error_med=1.0,
            residue_error_med=1.0,
            trace_error_med=1.0,
            nv_range=0.0,
            nv_median=float("nan"),
            null_fail_penalty=4,
            beats_random=False,
            beats_rejected=False,
            beats_ablation=False,
            beats_prior_artin=False,
            classification="UNSTABLE_OPERATOR",
            J_v14_7c=1e6,
            final_reward=0.0,
            arg_rows=[],
            trace_rows=[],
            residue_rows=[],
            nv_rows=[],
        )

    eigs = safe_eigvalsh(H, seed=int(rng_seed + 17))
    if eigs is None or eigs.size < 8:
        return EvalOut(
            stable=False,
            metric_nan_flag=True,
            word_len=int(len(w_clamped)),
            generator_entropy=float(gen_ent),
            power_entropy=float(pow_ent),
            support_overlap=0.0,
            poisson_like_fraction=1.0,
            active_argument_error_med=1.0,
            residue_error_med=1.0,
            trace_error_med=1.0,
            nv_range=0.0,
            nv_median=float("nan"),
            null_fail_penalty=4,
            beats_random=False,
            beats_rejected=False,
            beats_ablation=False,
            beats_prior_artin=False,
            classification="EIG_FAIL",
            J_v14_7c=1e6,
            final_reward=0.0,
            arg_rows=[],
            trace_rows=[],
            residue_rows=[],
            nv_rows=[],
        )

    op_levels = unfold_to_mean_spacing_one(eigs)
    tgt = np.asarray(target_levels, dtype=np.float64)
    tgt = tgt[np.isfinite(tgt)]
    tgt.sort()

    # Support + arg counts (support-calibrated via transport map, as in V14.7b)
    tm = transport_affine(op_levels, tgt)
    op_cal = apply_transport(op_levels, tm)

    arg_med, _arg_mean, support_overlap, arg_rows = active_argument_counts(op_cal, tgt, windows)
    res_med, _leak, residue_rows = residue_scores(op_cal, tgt, windows, eta=0.15, n_contour_points=256)
    tr_med, trace_rows = trace_proxy_rows(op_cal, tgt, windows)

    # NV / Poisson-like diagnostics: unfold again for NV invariance
    op_for_nv = unfold_to_mean_spacing_one(op_cal)
    tgt_for_nv = unfold_to_mean_spacing_one(tgt)
    op_nv = number_variance_curve(op_for_nv, L_grid)
    tgt_nv = number_variance_curve(tgt_for_nv, L_grid)
    dnv = nv_diagnostics(op_nv, tgt_nv, L_grid)
    nv_range = safe_float(dnv.get("nv_range", float("nan")), 0.0)
    nv_med = safe_float(dnv.get("nv_median", float("nan")), float("nan"))
    pois_frac = float(poisson_like_fraction_from_nv(op_nv, tgt_nv, L_grid))
    nv_rows = [{"kind": "operator", "L": float(L), "Sigma2": safe_float(y)} for L, y in zip(L_grid, op_nv)]

    # null gates: only true if baseline exists
    best_random = best_baseline_J(baselines, int(dim), "random")
    best_rej = best_baseline_J(baselines, int(dim), "rejected")
    best_abl = best_baseline_J(baselines, int(dim), "ablation")
    best_prior = best_baseline_J(baselines, int(dim), "prior_artin")
    beats_random = bool(best_random is not None and math.isfinite(best_random)) and False
    beats_rejected = bool(best_rej is not None and math.isfinite(best_rej)) and False
    beats_ablation = bool(best_abl is not None and math.isfinite(best_abl)) and False
    beats_prior = bool(best_prior is not None and math.isfinite(best_prior)) and False
    # compare after J computed (since J depends on null_fail_penalty); approximate: compare using candidate J once computed below

    # edit distance
    ed = int(edit_distance_cutoff(seed_word, w_clamped, cutoff=int(args.edit_distance_max)))

    # metric finiteness
    arg_med2 = finite_or_penalty(float(arg_med), penalty=penalty_big, flags=nan_flags, key="active_argument_error_med")
    tr_med2 = finite_or_penalty(float(tr_med), penalty=penalty_big, flags=nan_flags, key="trace_error_med")
    res_med2 = finite_or_penalty(float(res_med), penalty=penalty_big, flags=nan_flags, key="residue_error_med")
    sup2 = finite_or_penalty(float(support_overlap), penalty=0.0, flags=nan_flags, key="support_overlap")
    pois2 = finite_or_penalty(float(pois_frac), penalty=1.0, flags=nan_flags, key="poisson_like_fraction")
    nv_range2 = finite_or_penalty(float(nv_range), penalty=0.0, flags=nan_flags, key="nv_range")
    nv_med2 = finite_or_penalty(float(nv_med), penalty=0.0, flags=nan_flags, key="nv_median")

    # determine null fail penalty (after gates computed)
    # missing baseline => gate is False and counts as fail (do not silently pass)
    missing_kinds: List[str] = []
    if best_random is None:
        missing_kinds.append("random")
    if best_rej is None:
        missing_kinds.append("rejected")
    if best_abl is None:
        missing_kinds.append("ablation")
    if best_prior is None:
        missing_kinds.append("prior_artin")
    missing_baselines_by_dim[int(dim)] = sorted(set(missing_baselines_by_dim.get(int(dim), [])) | set(missing_kinds))

    # objective J_v14_7c (null_fail_penalty filled after beats_* computed)
    # first pass: pessimistic beats False
    null_fail_penalty = int((0 if beats_random else 1) + (0 if beats_rejected else 1) + (0 if beats_ablation else 1) + (0 if beats_prior else 1))
    J = 0.0
    J += float(args.lambda_arg) * float(arg_med2)
    J += float(args.lambda_trace) * float(tr_med2)
    J += float(args.lambda_support) * float(max(0.0, float(args.support_min) - float(sup2)))
    J += float(args.lambda_poisson) * float(pois2)
    J += float(args.lambda_residue) * float(res_med2)
    J += float(args.lambda_nv) * float(max(0.0, float(args.nv_range_min) - float(nv_range2)) / max(1.0, float(args.nv_range_min)))
    J += float(args.lambda_null) * float(null_fail_penalty)
    J += float(args.lambda_edit) * float(ed)
    # anti-collapse soft penalty to avoid trivial collapse even if not hard-gated
    J += 0.25 * float(collapse_pen)

    # now compute beats_* using J comparison (only if baseline exists)
    beats_random = bool(best_random is not None and math.isfinite(best_random) and float(J) < float(best_random))
    beats_rejected = bool(best_rej is not None and math.isfinite(best_rej) and float(J) < float(best_rej))
    beats_ablation = bool(best_abl is not None and math.isfinite(best_abl) and float(J) < float(best_abl))
    beats_prior = bool(best_prior is not None and math.isfinite(best_prior) and float(J) < float(best_prior))
    null_fail_penalty = int((0 if beats_random else 1) + (0 if beats_rejected else 1) + (0 if beats_ablation else 1) + (0 if beats_prior else 1))

    # update J with correct null_fail_penalty
    J = float(J - float(args.lambda_null) * float((0 if False else 0)) + 0.0)  # no-op; keep structure explicit
    J = float(
        float(args.lambda_arg) * float(arg_med2)
        + float(args.lambda_trace) * float(tr_med2)
        + float(args.lambda_support) * float(max(0.0, float(args.support_min) - float(sup2)))
        + float(args.lambda_poisson) * float(pois2)
        + float(args.lambda_residue) * float(res_med2)
        + float(args.lambda_nv) * float(max(0.0, float(args.nv_range_min) - float(nv_range2)) / max(1.0, float(args.nv_range_min)))
        + float(args.lambda_null) * float(null_fail_penalty)
        + float(args.lambda_edit) * float(ed)
        + 0.25 * float(collapse_pen)
    )

    metric_nan_flag = bool(any(nan_flags.values()))
    J = float(min(max(J, 0.0), 1e6))
    final_reward = float(1.0 / (1.0 + float(J))) if math.isfinite(J) else 0.0
    final_reward = float(final_reward) if (math.isfinite(final_reward) and final_reward >= 0.0) else 0.0

    # gates -> classification
    G2 = bool(sup2 >= float(args.support_min))
    G3 = bool(arg_med2 <= float(args.active_argument_error_max))
    G4 = bool(nv_range2 >= float(args.nv_range_min) and pois2 <= float(args.poisson_like_max))
    G5 = bool(pois2 <= float(args.poisson_like_max))
    G6 = bool(res_med2 <= float(args.residue_error_max))
    G7 = bool(tr_med2 <= float(args.trace_error_max))
    G12 = bool(no_collapse)
    G13 = bool((not metric_nan_flag) and math.isfinite(J) and math.isfinite(final_reward))

    if metric_nan_flag or (not G13):
        classification = "METRIC_NAN_FAIL"
    elif sup2 <= 0.0:
        classification = "ZERO_SUPPORT_REJECT"
    elif not G5:
        classification = "POISSON_LIKE_FAIL"
    elif not G3:
        classification = "ARGUMENT_FAIL"
    elif not G7:
        classification = "TRACE_FAIL"
    elif not G6:
        classification = "RESIDUE_FAIL"
    elif null_fail_penalty > 0:
        classification = "NULL_FAIL"
    elif not G12:
        classification = "COLLAPSE_FAIL"
    elif bool(stable and G2 and G3 and G4 and G5 and G6 and G7 and (null_fail_penalty == 0) and G12 and G13):
        classification = "ALL_GATE_PASS"
    else:
        classification = "OK"

    # enforce: if any metric NaN -> never claim all gates pass
    if classification == "ALL_GATE_PASS" and metric_nan_flag:
        classification = "METRIC_NAN_FAIL"

    return EvalOut(
        stable=bool(stable),
        metric_nan_flag=bool(metric_nan_flag),
        word_len=int(len(w_clamped)),
        generator_entropy=float(gen_ent),
        power_entropy=float(pow_ent),
        support_overlap=float(sup2),
        poisson_like_fraction=float(pois2),
        active_argument_error_med=float(arg_med2),
        residue_error_med=float(res_med2),
        trace_error_med=float(tr_med2),
        nv_range=float(nv_range2),
        nv_median=float(nv_med2),
        null_fail_penalty=int(null_fail_penalty),
        beats_random=bool(beats_random),
        beats_rejected=bool(beats_rejected),
        beats_ablation=bool(beats_ablation),
        beats_prior_artin=bool(beats_prior),
        classification=str(classification),
        J_v14_7c=float(J),
        final_reward=float(final_reward),
        arg_rows=list(arg_rows),
        trace_rows=list(trace_rows),
        residue_rows=list(residue_rows),
        nv_rows=list(nv_rows),
    )


# ----------------------------
# Seed selection
# ----------------------------


def _column_exists(rows: List[Dict[str, Any]], col: str) -> bool:
    return bool(rows) and (col in rows[0])


def select_seeds(
    ranking_rows: List[Dict[str, Any]],
    *,
    dims: List[int],
    modes: List[str],
    top_k: int,
    support_min: float,
    poisson_like_max: float,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    missing: List[str] = []
    out: List[Dict[str, Any]] = []
    have_class = _column_exists(ranking_rows, "classification")

    for d in dims:
        for mode in modes:
            sub = [r for r in ranking_rows if safe_int(r.get("dim", -1)) == int(d) and str(r.get("mode", "")) == str(mode)]
            if not sub:
                out.append(
                    {
                        "dim": int(d),
                        "mode": str(mode),
                        "seed_rank": None,
                        "seed_word": None,
                        "seed_J_v14_7b": None,
                        "seed_support_overlap": None,
                        "seed_poisson_like_fraction": None,
                        "seed_nv_range": None,
                        "seed_selection_empty": True,
                    }
                )
                continue
            # strict filter
            strict = []
            for r in sub:
                if have_class and str(r.get("classification", "")).strip() != "OK":
                    continue
                if safe_float(r.get("support_overlap", float("nan"))) < float(support_min):
                    continue
                if safe_float(r.get("poisson_like_fraction", float("nan"))) > float(poisson_like_max):
                    continue
                strict.append(r)
            if not strict:
                # relax only classification
                strict = [r for r in sub if safe_float(r.get("support_overlap", float("nan"))) >= float(support_min) and safe_float(r.get("poisson_like_fraction", float("nan"))) <= float(poisson_like_max)]
            strict.sort(key=lambda r: safe_float(r.get("J_v14_7b", float("inf")), float("inf")))
            if not strict:
                out.append(
                    {
                        "dim": int(d),
                        "mode": str(mode),
                        "seed_rank": None,
                        "seed_word": None,
                        "seed_J_v14_7b": None,
                        "seed_support_overlap": None,
                        "seed_poisson_like_fraction": None,
                        "seed_nv_range": None,
                        "seed_selection_empty": True,
                    }
                )
                continue
            for k, r in enumerate(strict[: int(top_k)], start=1):
                out.append(
                    {
                        "dim": int(d),
                        "mode": str(mode),
                        "seed_rank": int(k),
                        "seed_word": str(r.get("word", "")),
                        "seed_J_v14_7b": safe_float(r.get("J_v14_7b", float("nan"))),
                        "seed_support_overlap": safe_float(r.get("support_overlap", float("nan"))),
                        "seed_poisson_like_fraction": safe_float(r.get("poisson_like_fraction", float("nan"))),
                        "seed_nv_range": safe_float(r.get("nv_range", float("nan"))),
                        "seed_selection_empty": False,
                    }
                )
    return out, missing


# ----------------------------
# Reports
# ----------------------------


def render_report_md(
    *,
    out_dir: Path,
    config: Dict[str, Any],
    missing_sources: List[str],
    best_rows: List[Dict[str, Any]],
    gate_rows: List[Dict[str, Any]],
    null_rows: List[Dict[str, Any]],
    repair_summary: Dict[str, Any],
    proceed_to_v14_8: bool,
) -> str:
    md: List[str] = []
    md.append("# V14.7c — Argument-Trace Repair for Support-Calibrated Artin-DTES Candidates\n\n")
    md.append("> Warning: Computational evidence only; not a proof of RH.\n\n")
    md.append("## 1. Purpose\n")
    md.append("Improve the two failed V14.7b diagnostics (active argument count + trace proxy) via local word repair, without restarting the search.\n\n")
    md.append("## 2. Why V14.7c exists\n")
    md.append("V14.7b produced nontrivial support-calibrated anti-Poisson candidates, but still failed gates G3 and G7 for top dim=64 candidates.\n\n")
    md.append("## 3. Inputs\n")
    md.append(f"- Seeds: `{config.get('v14_7b_dir','')}/v14_7b_candidate_ranking.csv`\n")
    md.append(f"- Optional baselines: `{config.get('v13o14_dir','')}`, `{config.get('v14_5_dir','')}`, `{config.get('v14_2_dir','')}`\n")
    md.append(f"- Missing optional sources (recorded): `{', '.join(missing_sources) if missing_sources else 'none'}`\n\n")
    md.append("## 4. Repair search method\n")
    md.append("- Local mutations: replace / insert / delete / generator jitter / power flip.\n")
    md.append("- Motif bias: for dim=64, bias generator choices toward the most frequent generators in top seeds.\n")
    md.append("- Anti-collapse: penalize short/low-entropy/single-generator collapse; keep hard gate `G12_no_collapse`.\n\n")
    md.append("## 5. Objective\n")
    md.append("Minimize `J_v14_7c` as a weighted sum of argument, trace, support, Poisson-likeness, residue, NV, null-fail, and edit distance penalties.\n\n")
    md.append("## 6. Gate definitions\n")
    md.append("Gates G1–G14 follow the V14.7c spec; `G14_all_gate_pass` is true only if all scientific and null gates pass and all metrics are finite.\n\n")
    md.append("## 7. Best candidates\n")
    if best_rows:
        for r in best_rows:
            md.append(f"- dim={r.get('dim')} mode={r.get('mode')} J={r.get('J_v14_7c'):.6g} reward={r.get('final_reward'):.6g} support={r.get('support_overlap'):.3g} poisson={r.get('poisson_like_fraction'):.3g} arg={r.get('active_argument_error_med'):.3g} trace={r.get('trace_error_med'):.3g}\n")
    else:
        md.append("- (none)\n")
    md.append("\n")
    md.append("## 8. Gate summary\n")
    if gate_rows:
        for r in gate_rows:
            md.append(f"- dim={r.get('dim')} mode={r.get('mode')} all_gate={r.get('G14_all_gate_pass')} class={r.get('classification')}\n")
    else:
        md.append("- (none)\n")
    md.append("\n")
    md.append("## 9. Null-control comparison\n")
    if null_rows:
        for r in null_rows:
            md.append(f"- dim={r.get('dim')} mode={r.get('mode')} beats_random={r.get('beats_random')} beats_rejected={r.get('beats_rejected')} beats_ablation={r.get('beats_ablation')} beats_prior_artin={r.get('beats_prior_artin')} missing_baselines={r.get('missing_baselines')}\n")
    else:
        md.append("- (none)\n")
    md.append("\n")
    md.append("## 10. Argument-count diagnostics\n")
    md.append("See `v14_7c_active_argument_counts.csv` for per-window counts/errors.\n\n")
    md.append("## 11. Trace-proxy diagnostics\n")
    md.append("See `v14_7c_trace_proxy.csv` for per-window/sigma proxy errors.\n\n")
    md.append("## 12. Failure analysis\n")
    md.append(f"- Repair attempts: {repair_summary.get('n_evals_total', 0)} evals, {repair_summary.get('n_accept_total', 0)} accepted moves.\n\n")
    md.append("## 13. Decision\n")
    md.append(f"- Did V14.7c start from V14.7b support-calibrated candidates? **True**\n")
    md.append(f"- Did repair preserve anti-Poisson behavior? **{'True' if repair_summary.get('preserved_antipoisson', False) else 'False'}**\n")
    md.append(f"- Did repair improve active argument count? **{'True' if repair_summary.get('improved_argument', False) else 'False'}**\n")
    md.append(f"- Did repair improve trace proxy? **{'True' if repair_summary.get('improved_trace', False) else 'False'}**\n")
    md.append(f"- Did any candidate pass all gates? **{any(str(r.get('G14_all_gate_pass','')).lower()=='true' for r in gate_rows)}**\n")
    md.append(f"- Should proceed to V14.8? **{proceed_to_v14_8}**\n")
    md.append("- Should make analytic claim? **False**\n\n")
    md.append("> Warning: Computational evidence only; not a proof of RH.\n")
    return "".join(md)


# ----------------------------
# Main
# ----------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="V14.7c argument/trace repair around V14.7b seeds (computational only).")
    ap.add_argument("--v14_7b_dir", type=str, default="runs/v14_7b_support_calibrated_antipoisson_full")
    ap.add_argument("--v13o14_dir", type=str, default="runs/v13o14_transport_null_controls")
    ap.add_argument("--v14_5_dir", type=str, default="runs/v14_5_semantic_anticollapse_search")
    ap.add_argument("--v14_2_dir", type=str, default="runs/v14_2_stabilized_artin_operator_search")
    ap.add_argument("--out_dir", type=str, default="runs/v14_7c_argument_trace_repair")
    ap.add_argument("--dims", type=int, nargs="+", default=[64])
    ap.add_argument("--modes", type=str, nargs="+", default=["numeric_only", "hybrid_ranked_anticollapse"])
    ap.add_argument("--top_k_seeds", type=int, default=25)
    ap.add_argument("--num_repair_iters", type=int, default=80)
    ap.add_argument("--num_mutations_per_seed", type=int, default=16)
    ap.add_argument("--max_word_len", type=int, default=32)
    ap.add_argument("--max_power", type=int, default=3)
    ap.add_argument("--support_min", type=float, default=0.625)
    ap.add_argument("--poisson_like_max", type=float, default=0.25)
    ap.add_argument("--active_argument_error_max", type=float, default=0.25)
    ap.add_argument("--trace_error_max", type=float, default=0.35)
    ap.add_argument("--residue_error_max", type=float, default=0.35)
    ap.add_argument("--nv_range_min", type=float, default=32.0)
    ap.add_argument("--edit_distance_max", type=int, default=8)
    ap.add_argument("--lambda_arg", type=float, default=4.0)
    ap.add_argument("--lambda_trace", type=float, default=3.0)
    ap.add_argument("--lambda_support", type=float, default=8.0)
    ap.add_argument("--lambda_poisson", type=float, default=10.0)
    ap.add_argument("--lambda_residue", type=float, default=2.0)
    ap.add_argument("--lambda_nv", type=float, default=1.0)
    ap.add_argument("--lambda_null", type=float, default=2.0)
    ap.add_argument("--lambda_edit", type=float, default=0.05)
    ap.add_argument("--mutation_insert_prob", type=float, default=0.20)
    ap.add_argument("--mutation_delete_prob", type=float, default=0.10)
    ap.add_argument("--mutation_replace_prob", type=float, default=0.35)
    ap.add_argument("--mutation_power_flip_prob", type=float, default=0.20)
    ap.add_argument("--mutation_local_generator_jitter_prob", type=float, default=0.35)
    ap.add_argument("--local_generator_radius", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260507)
    ap.add_argument("--progress_every", type=int, default=10)
    args = ap.parse_args()

    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    rng_np = np.random.default_rng(int(args.seed))
    rng = random.Random(int(args.seed))

    warnings: List[str] = []
    if not _HAVE_SAFE_EIGH:
        warnings.append("safe_eigh unavailable; using numpy eigvalsh fallback.")

    # Inputs
    v14_7b_dir = _resolve(args.v14_7b_dir)
    ranking_path = v14_7b_dir / "v14_7b_candidate_ranking.csv"
    ranking_rows = _read_csv_best_effort(ranking_path)
    if not ranking_rows:
        raise SystemExit(f"Missing/empty required input: {ranking_path}")

    # Optional V14.7b extra files (load if present; for provenance only)
    optional_sources = [
        "v14_7b_best_candidates.csv",
        "v14_7b_gate_summary.csv",
        "v14_7b_active_argument_counts.csv",
        "v14_7b_trace_proxy.csv",
        "v14_7b_residue_scores.csv",
        "v14_7b_null_comparisons.csv",
    ]
    missing_sources: List[str] = []
    for name in optional_sources:
        if not (v14_7b_dir / name).is_file():
            missing_sources.append(f"v14_7b:{name}")

    dims = [int(d) for d in args.dims]
    modes = [str(m) for m in args.modes]

    # Baselines pool (best-effort)
    baselines, missing_baseline_files = load_baseline_pool(dims, _resolve(args.v13o14_dir), _resolve(args.v14_2_dir), _resolve(args.v14_5_dir))
    for m in missing_baseline_files:
        missing_sources.append(m)

    # Targets: best-effort consistent defaults (record if missing)
    true_levels_csv = _resolve("runs/v13o9_true_unfolded_spectra_from_v13o8/v13o9_unfolded_levels.csv")
    zeros_csv = _resolve("runs/zeros_100_400_precise.csv")
    target_by_dim: Dict[int, np.ndarray] = {}
    if true_levels_csv.is_file():
        df_levels, lvl_warns = rd.load_true_levels_csv(true_levels_csv, dims_keep=dims)
        warnings.extend([f"true_levels_csv: {w}" for w in lvl_warns])
        if df_levels is not None and (not getattr(df_levels, "empty", False)) and _HAVE_PANDAS:
            try:
                df = df_levels.copy()
                df["dim"] = pd.to_numeric(df["dim"], errors="coerce").astype("Int64")  # type: ignore
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
    else:
        missing_sources.append("true_levels_csv_missing:runs/v13o9_true_unfolded_spectra_from_v13o8/v13o9_unfolded_levels.csv")

    zeros_raw: Optional[np.ndarray] = None
    if zeros_csv.is_file():
        z, z_warns = rd.load_zeros_csv(zeros_csv)
        warnings.extend([f"zeros_csv: {w}" for w in z_warns])
        zeros_raw = np.asarray(z, dtype=np.float64)
    else:
        missing_sources.append("zeros_csv_missing:runs/zeros_100_400_precise.csv")

    for d in dims:
        if int(d) not in target_by_dim:
            if zeros_raw is None or zeros_raw.size < 8:
                # hard fallback: synthetic equally spaced target
                x = np.linspace(0.0, float(d) - 1.0, max(64, int(d)), dtype=np.float64)
                target_by_dim[int(d)] = unfold_to_mean_spacing_one(x)
                warnings.append(f"dim={d}: target fallback synthetic (zeros missing/too small).")
            else:
                target_by_dim[int(d)] = unfold_to_mean_spacing_one(zeros_raw)
                warnings.append(f"dim={d}: using zeros target fallback (real_zeta missing).")

    # Seed selection
    seed_pool_rows, _seed_missing = select_seeds(
        ranking_rows,
        dims=dims,
        modes=modes,
        top_k=int(args.top_k_seeds),
        support_min=float(args.support_min),
        poisson_like_max=float(args.poisson_like_max),
    )

    # motif bias: detect top generators among selected seeds (dim=64 especially)
    preferred_generators_by: Dict[Tuple[int, str], List[int]] = {}
    seed_parse_warnings: List[str] = []
    for d in dims:
        for mode in modes:
            seeds_here = [r for r in seed_pool_rows if int(r.get("dim", 0)) == int(d) and str(r.get("mode", "")) == str(mode) and not bool(r.get("seed_selection_empty", False))]
            counter: Counter[int] = Counter()
            for r in seeds_here:
                w = parse_artin_word(str(r.get("seed_word", "")), dim=int(d), max_power=int(args.max_power), warnings=seed_parse_warnings)
                for gi, _p in w:
                    counter[int(gi)] += 1
            top = [g for g, _c in counter.most_common(16)]
            preferred_generators_by[(int(d), str(mode))] = top

    missing_baselines_by_dim: Dict[int, List[str]] = {}

    # Search loop
    L_grid = make_L_grid()
    repair_history_rows: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []

    best_by_dim_mode: Dict[Tuple[int, str], Dict[str, Any]] = {}
    best_details_by_dim_mode: Dict[Tuple[int, str], EvalOut] = {}

    cfg = MutationConfig(
        mutation_insert_prob=float(args.mutation_insert_prob),
        mutation_delete_prob=float(args.mutation_delete_prob),
        mutation_replace_prob=float(args.mutation_replace_prob),
        mutation_power_flip_prob=float(args.mutation_power_flip_prob),
        mutation_local_generator_jitter_prob=float(args.mutation_local_generator_jitter_prob),
        local_generator_radius=int(args.local_generator_radius),
    )

    n_evals_total = 0
    n_accept_total = 0

    # Initialize per-seed state
    seeds_state: List[Dict[str, Any]] = []
    for i, sr in enumerate(seed_pool_rows):
        if bool(sr.get("seed_selection_empty", False)):
            continue
        d = int(sr["dim"])
        mode = str(sr["mode"])
        seed_word_str = str(sr.get("seed_word", ""))
        w0 = parse_artin_word(seed_word_str, dim=int(d), max_power=int(args.max_power), warnings=seed_parse_warnings)
        if not w0:
            continue
        windows, _win_meta = effective_windows_for_target(target_by_dim[int(d)])
        ev0 = evaluate_candidate(
            dim=int(d),
            word=w0,
            seed_word=w0,
            target_levels=target_by_dim[int(d)],
            windows=windows,
            L_grid=L_grid,
            baselines=baselines,
            args=args,
            rng_seed=int(args.seed + 10000 * i + 17),
            missing_baselines_by_dim=missing_baselines_by_dim,
        )
        n_evals_total += 1
        seeds_state.append(
            {
                "seed_id": int(len(seeds_state)),
                "dim": int(d),
                "mode": str(mode),
                "seed_word": w0,
                "seed_word_str": word_to_string(w0),
                "seed_eval": ev0,
                "current_word": w0,
                "current_eval": ev0,
                "best_word": w0,
                "best_eval": ev0,
            }
        )
        candidate_rows.append(
            {
                "dim": int(d),
                "mode": str(mode),
                "word": word_to_string(w0),
                "word_len": int(ev0.word_len),
                "edit_distance_from_seed": 0,
                "seed_word": word_to_string(w0),
                "J_v14_7c": float(ev0.J_v14_7c),
                "final_reward": float(ev0.final_reward),
                "support_overlap": float(ev0.support_overlap),
                "poisson_like_fraction": float(ev0.poisson_like_fraction),
                "active_argument_error_med": float(ev0.active_argument_error_med),
                "residue_error_med": float(ev0.residue_error_med),
                "trace_error_med": float(ev0.trace_error_med),
                "nv_range": float(ev0.nv_range),
                "nv_median": float(ev0.nv_median),
                "generator_entropy": float(ev0.generator_entropy),
                "power_entropy": float(ev0.power_entropy),
                "beats_random": bool(ev0.beats_random),
                "beats_rejected": bool(ev0.beats_rejected),
                "beats_ablation": bool(ev0.beats_ablation),
                "beats_prior_artin": bool(ev0.beats_prior_artin),
                "classification": str(ev0.classification),
            }
        )

    # If nothing to do, still write empty outputs
    if not seeds_state:
        missing_sources.append("seed_pool_empty_after_parse")

    for it in range(1, int(args.num_repair_iters) + 1):
        for st_i, st in enumerate(seeds_state):
            d = int(st["dim"])
            mode = str(st["mode"])
            seed_word = list(st["seed_word"])
            seed_word_str = str(st["seed_word_str"])
            parent_word = list(st["current_word"])
            parent_eval: EvalOut = st["current_eval"]
            best_eval: EvalOut = st["best_eval"]
            windows, _win_meta = effective_windows_for_target(target_by_dim[int(d)])

            for m_id in range(int(args.num_mutations_per_seed)):
                mut_w, mut_kind = mutate_word(
                    parent_word,
                    dim=int(d),
                    max_power=int(args.max_power),
                    max_word_len=int(args.max_word_len),
                    preferred_generators=preferred_generators_by.get((int(d), str(mode)), []),
                    cfg=cfg,
                    rng=rng,
                )
                ed = int(edit_distance_cutoff(seed_word, mut_w, cutoff=int(args.edit_distance_max)))
                if ed > int(args.edit_distance_max):
                    continue
                ev = evaluate_candidate(
                    dim=int(d),
                    word=mut_w,
                    seed_word=seed_word,
                    target_levels=target_by_dim[int(d)],
                    windows=windows,
                    L_grid=L_grid,
                    baselines=baselines,
                    args=args,
                    rng_seed=int(args.seed + 100000 * it + 1000 * d + 7 * st_i + 31 * m_id),
                    missing_baselines_by_dim=missing_baselines_by_dim,
                )
                n_evals_total += 1

                # acceptance: preserve key properties and no semantic collapse
                accept = False
                if (
                    (not ev.metric_nan_flag)
                    and ev.stable
                    and (ev.support_overlap >= float(args.support_min))
                    and (ev.poisson_like_fraction <= float(args.poisson_like_max))
                    and (ev.word_len >= 4)
                    and (ev.classification not in ("COLLAPSE_FAIL", "METRIC_NAN_FAIL", "UNSTABLE_OPERATOR", "EIG_FAIL", "ZERO_SUPPORT_REJECT"))
                    and (float(ev.J_v14_7c) + 1e-12 < float(parent_eval.J_v14_7c))
                ):
                    accept = True

                if accept:
                    st["current_word"] = mut_w
                    st["current_eval"] = ev
                    parent_word = mut_w
                    parent_eval = ev
                    n_accept_total += 1

                is_best = False
                if float(ev.J_v14_7c) + 1e-12 < float(best_eval.J_v14_7c) and (ev.support_overlap >= float(args.support_min)) and (ev.poisson_like_fraction <= float(args.poisson_like_max)):
                    st["best_word"] = mut_w
                    st["best_eval"] = ev
                    best_eval = ev
                    is_best = True

                repair_history_rows.append(
                    {
                        "iter": int(it),
                        "dim": int(d),
                        "mode": str(mode),
                        "seed_id": int(st["seed_id"]),
                        "mutation_id": int(m_id),
                        "mutation_kind": str(mut_kind),
                        "parent_word": word_to_string(parent_word),
                        "candidate_word": word_to_string(mut_w),
                        "edit_distance": int(ed),
                        "word_len": int(ev.word_len),
                        "J_v14_7c": float(ev.J_v14_7c),
                        "final_reward": float(ev.final_reward),
                        "support_overlap": float(ev.support_overlap),
                        "poisson_like_fraction": float(ev.poisson_like_fraction),
                        "active_argument_error_med": float(ev.active_argument_error_med),
                        "residue_error_med": float(ev.residue_error_med),
                        "trace_error_med": float(ev.trace_error_med),
                        "nv_range": float(ev.nv_range),
                        "null_fail_penalty": int(ev.null_fail_penalty),
                        "classification": str(ev.classification),
                        "accepted": bool(accept),
                        "is_best_so_far": bool(is_best),
                    }
                )

                candidate_rows.append(
                    {
                        "dim": int(d),
                        "mode": str(mode),
                        "word": word_to_string(mut_w),
                        "word_len": int(ev.word_len),
                        "edit_distance_from_seed": int(ed),
                        "seed_word": seed_word_str,
                        "J_v14_7c": float(ev.J_v14_7c),
                        "final_reward": float(ev.final_reward),
                        "support_overlap": float(ev.support_overlap),
                        "poisson_like_fraction": float(ev.poisson_like_fraction),
                        "active_argument_error_med": float(ev.active_argument_error_med),
                        "residue_error_med": float(ev.residue_error_med),
                        "trace_error_med": float(ev.trace_error_med),
                        "nv_range": float(ev.nv_range),
                        "nv_median": float(ev.nv_median),
                        "generator_entropy": float(ev.generator_entropy),
                        "power_entropy": float(ev.power_entropy),
                        "beats_random": bool(ev.beats_random),
                        "beats_rejected": bool(ev.beats_rejected),
                        "beats_ablation": bool(ev.beats_ablation),
                        "beats_prior_artin": bool(ev.beats_prior_artin),
                        "classification": str(ev.classification),
                    }
                )

        if it == 1 or it % max(1, int(args.progress_every)) == 0 or it == int(args.num_repair_iters):
            elapsed = time.perf_counter() - t0
            eta = (elapsed / max(1, it)) * max(0, int(args.num_repair_iters) - it)
            # best overall so far
            best_J = float("inf")
            best_row = None
            for st in seeds_state:
                ev: EvalOut = st["best_eval"]
                if float(ev.J_v14_7c) < best_J:
                    best_J = float(ev.J_v14_7c)
                    best_row = (st, ev)
            if best_row is not None:
                st, ev = best_row
                print(
                    f"[V14.7c] iter={it}/{int(args.num_repair_iters)} dim={int(st['dim'])} mode={st['mode']} "
                    f"best_J={float(ev.J_v14_7c):.6g} best_reward={float(ev.final_reward):.6g} "
                    f"support={float(ev.support_overlap):.3g} poisson={float(ev.poisson_like_fraction):.3g} "
                    f"arg={float(ev.active_argument_error_med):.3g} trace={float(ev.trace_error_med):.3g} "
                    f"eta={format_seconds(eta)}",
                    flush=True,
                )
            else:
                print(f"[V14.7c] iter={it}/{int(args.num_repair_iters)} best_J=nan eta={format_seconds(eta)}", flush=True)

    # best per dim/mode from seeds_state
    best_candidates_rows: List[Dict[str, Any]] = []
    gate_rows: List[Dict[str, Any]] = []
    null_rows: List[Dict[str, Any]] = []
    active_argument_best_rows: List[Dict[str, Any]] = []
    trace_best_rows: List[Dict[str, Any]] = []
    residue_best_rows: List[Dict[str, Any]] = []
    nv_best_rows: List[Dict[str, Any]] = []

    for d in dims:
        for mode in modes:
            best_ev = None
            best_word = None
            for st in seeds_state:
                if int(st["dim"]) == int(d) and str(st["mode"]) == str(mode):
                    ev: EvalOut = st["best_eval"]
                    if best_ev is None or float(ev.J_v14_7c) < float(best_ev.J_v14_7c):
                        best_ev = ev
                        best_word = st["best_word"]
            if best_ev is None or best_word is None:
                continue
            wstr = word_to_string(best_word)
            best_by_dim_mode[(int(d), str(mode))] = {"word": wstr, "J_v14_7c": float(best_ev.J_v14_7c)}
            best_details_by_dim_mode[(int(d), str(mode))] = best_ev

            best_candidates_rows.append(
                {
                    "dim": int(d),
                    "mode": str(mode),
                    "best_word": str(wstr),
                    "J_v14_7c": float(best_ev.J_v14_7c),
                    "final_reward": float(best_ev.final_reward),
                    "support_overlap": float(best_ev.support_overlap),
                    "poisson_like_fraction": float(best_ev.poisson_like_fraction),
                    "active_argument_error_med": float(best_ev.active_argument_error_med),
                    "residue_error_med": float(best_ev.residue_error_med),
                    "trace_error_med": float(best_ev.trace_error_med),
                    "nv_range": float(best_ev.nv_range),
                    "nv_median": float(best_ev.nv_median),
                    "generator_entropy": float(best_ev.generator_entropy),
                    "power_entropy": float(best_ev.power_entropy),
                    "classification": str(best_ev.classification),
                }
            )

            # gates
            G1 = bool(best_ev.stable)
            G2 = bool(best_ev.support_overlap >= float(args.support_min))
            G3 = bool(best_ev.active_argument_error_med <= float(args.active_argument_error_max))
            G4 = bool(best_ev.nv_range >= float(args.nv_range_min) and best_ev.poisson_like_fraction <= float(args.poisson_like_max))
            G5 = bool(best_ev.poisson_like_fraction <= float(args.poisson_like_max))
            G6 = bool(best_ev.residue_error_med <= float(args.residue_error_max))
            G7 = bool(best_ev.trace_error_med <= float(args.trace_error_max))
            # null gates only if baseline exists (otherwise false)
            best_random = best_baseline_J(baselines, int(d), "random")
            best_rej = best_baseline_J(baselines, int(d), "rejected")
            best_abl = best_baseline_J(baselines, int(d), "ablation")
            best_prior = best_baseline_J(baselines, int(d), "prior_artin")
            G8 = bool(best_ev.beats_random) if best_random is not None else False
            G9 = bool(best_ev.beats_rejected) if best_rej is not None else False
            G10 = bool(best_ev.beats_ablation) if best_abl is not None else False
            G11 = bool(best_ev.beats_prior_artin) if best_prior is not None else False
            # collapse
            collapse_pen, meta = collapse_penalty(best_word, dim=int(d))
            G12 = bool(int(meta["word_len"]) >= 4 and int(meta["unique_generator_count"]) >= 3 and float(meta["generator_entropy"]) >= 0.25)
            G13 = bool((not best_ev.metric_nan_flag) and math.isfinite(best_ev.J_v14_7c) and math.isfinite(best_ev.final_reward))
            all_gate = bool(G1 and G2 and G3 and G4 and G5 and G6 and G7 and G8 and G9 and G10 and G11 and G12 and G13)

            gate_rows.append(
                {
                    "dim": int(d),
                    "mode": str(mode),
                    "word": str(wstr),
                    "J_v14_7c": float(best_ev.J_v14_7c),
                    "final_reward": float(best_ev.final_reward),
                    "support_overlap": float(best_ev.support_overlap),
                    "poisson_like_fraction": float(best_ev.poisson_like_fraction),
                    "active_argument_error_med": float(best_ev.active_argument_error_med),
                    "residue_error_med": float(best_ev.residue_error_med),
                    "trace_error_med": float(best_ev.trace_error_med),
                    "nv_range": float(best_ev.nv_range),
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
                    "G12_no_collapse": bool(G12),
                    "G13_metric_finite": bool(G13),
                    "G14_all_gate_pass": bool(all_gate),
                    "classification": str(best_ev.classification if not all_gate else "ALL_GATE_PASS"),
                }
            )

            # null comparisons row
            best_null = best_null_J(baselines, int(d))
            dist = null_dist(baselines, int(d))
            null_sep = float(best_null - float(best_ev.J_v14_7c)) if (best_null is not None and math.isfinite(best_null)) else float("nan")
            null_z = float("nan")
            null_pct = float("nan")
            if len(dist) >= 3:
                mu = float(np.mean(dist))
                sd = float(np.std(dist))
                null_z = float((mu - float(best_ev.J_v14_7c)) / sd) if sd > 1e-12 else float("nan")
                null_pct = float(np.mean([1.0 if x <= float(best_ev.J_v14_7c) else 0.0 for x in dist]))
            missing_str = "|".join(sorted(set(missing_baselines_by_dim.get(int(d), []))))
            null_rows.append(
                {
                    "dim": int(d),
                    "mode": str(mode),
                    "best_word": str(wstr),
                    "best_J_v14_7c": float(best_ev.J_v14_7c),
                    "best_random_J": float(best_random) if best_random is not None else float("nan"),
                    "best_rejected_J": float(best_rej) if best_rej is not None else float("nan"),
                    "best_ablation_J": float(best_abl) if best_abl is not None else float("nan"),
                    "best_prior_artin_J": float(best_prior) if best_prior is not None else float("nan"),
                    "best_null_J": float(best_null) if best_null is not None else float("nan"),
                    "null_separation": float(null_sep),
                    "null_zscore": float(null_z),
                    "null_percentile": float(null_pct),
                    "beats_random": bool(best_ev.beats_random),
                    "beats_rejected": bool(best_ev.beats_rejected),
                    "beats_ablation": bool(best_ev.beats_ablation),
                    "beats_prior_artin": bool(best_ev.beats_prior_artin),
                    "missing_baselines": str(missing_str),
                }
            )

            # detail outputs (best only)
            for r in best_ev.arg_rows:
                active_argument_best_rows.append({"dim": int(d), "mode": str(mode), "word": str(wstr), **r})
            for r in best_ev.trace_rows:
                trace_best_rows.append({"dim": int(d), "mode": str(mode), "word": str(wstr), **r})
            for r in best_ev.residue_rows:
                residue_best_rows.append({"dim": int(d), "mode": str(mode), "word": str(wstr), **r})
            for r in best_ev.nv_rows:
                nv_best_rows.append({"dim": int(d), "mode": str(mode), "word": str(wstr), **r})

    # Candidate ranking with per-dim/mode ranks
    ranked_rows: List[Dict[str, Any]] = []
    for d in dims:
        for mode in modes:
            sub = [r for r in candidate_rows if int(r.get("dim", -1)) == int(d) and str(r.get("mode", "")) == str(mode)]
            sub.sort(key=lambda r: safe_float(r.get("J_v14_7c", float("inf")), float("inf")))
            for k, r in enumerate(sub, start=1):
                rr = dict(r)
                rr["rank"] = int(k)
                ranked_rows.append(rr)

    # repair summary decisions
    # Compare best vs seed metrics (first evaluated per seed is included as candidate row with edit_distance=0)
    improved_argument = False
    improved_trace = False
    preserved_antipoisson = True
    for st in seeds_state:
        seed_ev: EvalOut = st["seed_eval"] if "seed_eval" in st else st["current_eval"]
        best_ev: EvalOut = st["best_eval"]
        improved_argument = improved_argument or (float(best_ev.active_argument_error_med) + 1e-12 < float(seed_ev.active_argument_error_med))
        improved_trace = improved_trace or (float(best_ev.trace_error_med) + 1e-12 < float(seed_ev.trace_error_med))
        preserved_antipoisson = preserved_antipoisson and (float(best_ev.poisson_like_fraction) <= float(args.poisson_like_max) + 1e-12)

    any_all_gate = any(str(r.get("G14_all_gate_pass", "")).lower() == "true" for r in gate_rows)
    proceed_to_v14_8 = bool(any_all_gate or (improved_argument and improved_trace))

    # Write outputs
    write_csv(
        out_dir / "v14_7c_seed_pool.csv",
        fieldnames=[
            "dim",
            "mode",
            "seed_rank",
            "seed_word",
            "seed_J_v14_7b",
            "seed_support_overlap",
            "seed_poisson_like_fraction",
            "seed_nv_range",
            "seed_selection_empty",
        ],
        rows=seed_pool_rows,
    )
    write_csv(
        out_dir / "v14_7c_repair_history.csv",
        fieldnames=list(repair_history_rows[0].keys()) if repair_history_rows else ["iter", "dim", "mode", "seed_id", "mutation_id", "candidate_word", "J_v14_7c", "accepted"],
        rows=repair_history_rows,
    )
    write_csv(
        out_dir / "v14_7c_candidate_ranking.csv",
        fieldnames=[
            "dim",
            "mode",
            "rank",
            "word",
            "word_len",
            "edit_distance_from_seed",
            "seed_word",
            "J_v14_7c",
            "final_reward",
            "support_overlap",
            "poisson_like_fraction",
            "active_argument_error_med",
            "residue_error_med",
            "trace_error_med",
            "nv_range",
            "nv_median",
            "generator_entropy",
            "power_entropy",
            "beats_random",
            "beats_rejected",
            "beats_ablation",
            "beats_prior_artin",
            "classification",
        ],
        rows=ranked_rows,
    )
    write_csv(
        out_dir / "v14_7c_best_candidates.csv",
        fieldnames=list(best_candidates_rows[0].keys()) if best_candidates_rows else ["dim", "mode", "best_word", "J_v14_7c", "final_reward"],
        rows=best_candidates_rows,
    )
    write_csv(
        out_dir / "v14_7c_gate_summary.csv",
        fieldnames=list(gate_rows[0].keys()) if gate_rows else ["dim", "mode", "G14_all_gate_pass"],
        rows=gate_rows,
    )
    write_csv(
        out_dir / "v14_7c_null_comparisons.csv",
        fieldnames=list(null_rows[0].keys()) if null_rows else ["dim", "mode", "best_J_v14_7c"],
        rows=null_rows,
    )
    write_csv(
        out_dir / "v14_7c_active_argument_counts.csv",
        fieldnames=list(active_argument_best_rows[0].keys()) if active_argument_best_rows else ["dim", "mode", "word", "window_a", "window_b", "N_operator", "N_target", "N_error", "N_error_norm", "active_window"],
        rows=active_argument_best_rows,
    )
    write_csv(
        out_dir / "v14_7c_trace_proxy.csv",
        fieldnames=list(trace_best_rows[0].keys()) if trace_best_rows else ["dim", "mode", "word", "window_a", "window_b", "center", "sigma", "S_operator", "S_target", "trace_error_norm"],
        rows=trace_best_rows,
    )
    write_csv(
        out_dir / "v14_7c_residue_scores.csv",
        fieldnames=list(residue_best_rows[0].keys()) if residue_best_rows else ["dim", "mode", "word", "window_a", "window_b", "residue_count_error", "residue_imag_leak"],
        rows=residue_best_rows,
    )
    write_csv(
        out_dir / "v14_7c_nv_diagnostics.csv",
        fieldnames=list(nv_best_rows[0].keys()) if nv_best_rows else ["dim", "mode", "word", "kind", "L", "Sigma2"],
        rows=nv_best_rows,
    )

    # results.json
    results = {
        "version": "v14_7c",
        "config": {k: json_sanitize(getattr(args, k)) for k in vars(args).keys()},
        "missing_sources": sorted(set(missing_sources)),
        "warnings": warnings,
        "best_by_dim_mode": {f"{d}|{m}": v for (d, m), v in best_by_dim_mode.items()},
        "n_all_gate_pass": int(sum(1 for r in gate_rows if bool(r.get("G14_all_gate_pass", False)))),
        "proceed_to_v14_8": bool(proceed_to_v14_8),
        "analytic_claim": False,
        "short_interpretation": "Local repair search around V14.7b seeds; computational evidence only.",
        "repair_summary": {
            "n_evals_total": int(n_evals_total),
            "n_accept_total": int(n_accept_total),
            "improved_argument": bool(improved_argument),
            "improved_trace": bool(improved_trace),
            "preserved_antipoisson": bool(preserved_antipoisson),
        },
    }
    write_text(out_dir / "v14_7c_results.json", json.dumps(json_sanitize(results), indent=2, ensure_ascii=False) + "\n")

    # report.md / report.tex / report.pdf
    report_md = render_report_md(
        out_dir=out_dir,
        config=results["config"],
        missing_sources=sorted(set(missing_sources)),
        best_rows=best_candidates_rows,
        gate_rows=gate_rows,
        null_rows=null_rows,
        repair_summary=results["repair_summary"],
        proceed_to_v14_8=bool(proceed_to_v14_8),
    )
    write_text(out_dir / "v14_7c_report.md", report_md)

    tex = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\begin{document}
\section*{V14.7c --- Argument-Trace Repair for Support-Calibrated Artin-DTES Candidates}
\textbf{Warning: Computational evidence only; not a proof of RH.}

This run performs a local word repair search around V14.7b support-calibrated candidates, optimizing
active argument count error and trace proxy error while preserving support overlap and anti-Poisson behavior.
\end{document}
"""
    tex_path = out_dir / "v14_7c_report.tex"
    write_text(tex_path, tex)
    if _find_pdflatex():
        try_pdflatex(tex_path, out_dir, "v14_7c_report.pdf")

    elapsed = time.perf_counter() - t0
    print(f"[V14.7c] done in {format_seconds(elapsed)} out_dir={out_dir}", flush=True)


if __name__ == "__main__":
    main()

