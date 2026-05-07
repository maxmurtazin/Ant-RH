#!/usr/bin/env python3
"""
V14.8b — Spectral Support Rescaling for Braid-Graph Laplacian (Ant-RH)

Computational evidence only; not a proof of RH.

This is a *post-processing calibration layer* on top of V14.8 operators:
  - take top V14.8 candidates (dim/family/mode)
  - reconstruct raw operator spectrum by rebuilding the V14.8 braid-graph Laplacian operator
  - fit monotone rescaling:
        E_scaled = a E + b
    and optionally:
        E_scaled = a E + b + c log(1 + max(E, 0))
  - evaluate support/argument/residue/trace/NV/Hadamard + null comparisons

Missing baseline files must not crash the run: gates become False and sources recorded.
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
# Token / word utils (reuse V14.8 representation)
# ----------------------------


Token = Tuple[int, int]


def power_values(max_power: int) -> List[int]:
    return [p for p in range(-max_power, max_power + 1) if p != 0]


def simplify_word(word: List[Token], *, max_power: int, max_word_len: int) -> List[Token]:
    out: List[Token] = []
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


def clamp_word_to_dim(word: List[Token], dim: int, max_power: int, max_word_len: int) -> List[Token]:
    out: List[Token] = []
    for i, p in word:
        ii = int(max(1, min(int(dim) - 1, int(i))))
        pp = int(max(-max_power, min(max_power, int(p))))
        if pp == 0:
            continue
        out.append((ii, pp))
    return simplify_word(out, max_power=max_power, max_word_len=max_word_len)


def word_to_string(word: List[Token]) -> str:
    return " ".join([f"sigma_{int(i)}^{int(p)}" for i, p in word])


def parse_braid_word(s: str, *, dim: int, max_power: int, warnings: List[str]) -> List[Token]:
    toks = str(s or "").strip().split()
    out: List[Token] = []
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
            if pw == 0 or abs(pw) > int(max_power):
                warnings.append(f"skip_token(power_out_of_range max_power={max_power}): {tt}")
                continue
            out.append((int(gi), int(pw)))
        except Exception:
            warnings.append(f"skip_token(parse_fail): {tt}")
            continue
    return simplify_word(out, max_power=int(max_power), max_word_len=10**9)


# ----------------------------
# Braid graph + operator (same as V14.8 core)
# ----------------------------


@dataclass
class GraphStats:
    n_nodes: int
    n_edges_nonzero: int
    total_weight: float
    total_signed_weight: float
    visit_count: int
    repeat_count: int
    long_range_edge_count: int
    curvature_score: float


def build_braid_graph(dim: int, word: List[Token]) -> Tuple[np.ndarray, np.ndarray, GraphStats]:
    n = int(dim)
    W_abs = np.zeros((n, n), dtype=np.float64)
    W_sgn = np.zeros((n, n), dtype=np.float64)
    visits = 0
    repeats = 0
    long_edges = 0
    prev_i_eff: Optional[int] = None
    for gi, p in word:
        visits += 1
        i_eff = int(gi) % max(1, (n - 1))
        j_eff = i_eff + 1
        w = float(abs(int(p)))
        s = float(1.0 if int(p) > 0 else -1.0)
        W_abs[i_eff, j_eff] += w
        W_abs[j_eff, i_eff] += w
        W_sgn[i_eff, j_eff] += s * w
        W_sgn[j_eff, i_eff] += s * w
        if prev_i_eff is not None:
            if prev_i_eff == i_eff:
                repeats += 1
            if prev_i_eff != i_eff:
                a = int(max(0, min(n - 1, prev_i_eff)))
                b = int(max(0, min(n - 1, i_eff)))
                if a != b:
                    mem = 0.15 * min(1.0, w)
                    W_abs[a, b] += mem
                    W_abs[b, a] += mem
                    W_sgn[a, b] += mem * (1.0 if (i_eff - prev_i_eff) > 0 else -1.0)
                    W_sgn[b, a] += W_sgn[a, b]
                    long_edges += 1
        prev_i_eff = i_eff
    nz = int(np.sum(W_abs > 0.0))
    gen_hist = Counter([int(gi) % max(1, (n - 1)) for gi, _p in word])
    total = float(sum(gen_hist.values())) if gen_hist else 1.0
    pmax = float(max(gen_hist.values()) / total) if gen_hist else 1.0
    curvature = float(repeats / max(1, visits)) + float(pmax)
    stats = GraphStats(
        n_nodes=n,
        n_edges_nonzero=nz,
        total_weight=float(np.sum(W_abs)),
        total_signed_weight=float(np.sum(W_sgn)),
        visit_count=int(visits),
        repeat_count=int(repeats),
        long_range_edge_count=int(long_edges),
        curvature_score=float(curvature),
    )
    return W_abs, W_sgn, stats


def make_laplacian_from_adjacency(A: np.ndarray, *, use_abs_degree: bool = True) -> np.ndarray:
    A = np.asarray(A)
    deg = np.sum(np.abs(A), axis=1) if use_abs_degree else np.sum(A, axis=1)
    D = np.diag(deg.astype(np.float64, copy=False))
    return (D - A).astype(np.complex128, copy=False)


def magnetic_adjacency(W_abs: np.ndarray, W_sgn: np.ndarray, *, theta_scale: float = 0.35) -> np.ndarray:
    W = np.asarray(W_abs, dtype=np.float64)
    S = np.asarray(W_sgn, dtype=np.float64)
    n = int(W.shape[0])
    A = np.zeros((n, n), dtype=np.complex128)
    for i in range(n):
        for j in range(i + 1, n):
            w = float(W[i, j])
            if w <= 0:
                continue
            orient = float(np.tanh(S[i, j]))
            theta = float(theta_scale) * orient
            val = complex(w * math.cos(theta), w * math.sin(theta))
            A[i, j] = val
            A[j, i] = np.conjugate(val)
    return A


def build_operator_from_word(*, dim: int, word: List[Token], family: str, seed: int) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    n = int(dim)
    w = clamp_word_to_dim(word, dim=n, max_power=3, max_word_len=10**9)
    W_abs, W_sgn, gs = build_braid_graph(n, w)
    diag: Dict[str, Any] = {
        "stable": False,
        "graph_n_edges_nonzero": int(gs.n_edges_nonzero),
        "graph_total_weight": float(gs.total_weight),
        "graph_total_signed_weight": float(gs.total_signed_weight),
        "graph_visit_count": int(gs.visit_count),
        "graph_repeat_count": int(gs.repeat_count),
        "graph_long_range_edge_count": int(gs.long_range_edge_count),
        "graph_curvature_score": float(gs.curvature_score),
    }

    mode = str(family).strip()
    if mode == "plain_graph_laplacian":
        A = W_abs.astype(np.complex128, copy=False)
        L = make_laplacian_from_adjacency(A.real, use_abs_degree=True)
    elif mode == "signed_braid_laplacian":
        A = W_sgn.astype(np.complex128, copy=False)
        L = make_laplacian_from_adjacency(A.real, use_abs_degree=True)
    elif mode == "magnetic_graph_laplacian":
        A = magnetic_adjacency(W_abs, W_sgn, theta_scale=0.45)
        L = make_laplacian_from_adjacency(A, use_abs_degree=True)
    elif mode == "curvature_regularized_laplacian":
        A = W_abs.astype(np.complex128, copy=False)
        L = make_laplacian_from_adjacency(A.real, use_abs_degree=True)
        lam_curv = 0.25
        deg = np.sum(W_abs, axis=1)
        curv_diag = (deg / (np.max(deg) + 1e-12)).astype(np.float64) if np.max(deg) > 1e-12 else np.zeros((n,), dtype=np.float64)
        L = L + complex(lam_curv) * np.diag(curv_diag)
        diag["lambda_curv"] = float(lam_curv)
    elif mode == "hybrid_braid_graph_laplacian":
        A_mag = magnetic_adjacency(W_abs, W_sgn, theta_scale=0.40)
        L0 = make_laplacian_from_adjacency(W_sgn, use_abs_degree=True)
        Lm = make_laplacian_from_adjacency(A_mag, use_abs_degree=True)
        lam_mag = 0.35
        lam_curv = 0.20
        deg = np.sum(W_abs, axis=1)
        curv_diag = (deg / (np.max(deg) + 1e-12)).astype(np.float64) if np.max(deg) > 1e-12 else np.zeros((n,), dtype=np.float64)
        L = L0 + complex(lam_mag) * Lm + complex(lam_curv) * np.diag(curv_diag)
        diag["lambda_mag"] = float(lam_mag)
        diag["lambda_curv"] = float(lam_curv)
    else:
        diag["reason"] = f"unknown_family={family!r}"
        return None, diag

    # stabilize: Hermitian symm, remove trace, normalize spectral radius to dim/4
    H = 0.5 * (L + L.conj().T)
    tr = np.trace(H) / complex(n)
    H = H - tr * np.eye(n, dtype=np.complex128)
    try:
        eigs = np.linalg.eigvalsh(np.asarray(H.real, dtype=np.float64))
        eigs = np.asarray(eigs, dtype=np.float64).reshape(-1)
        eigs = eigs[np.isfinite(eigs)]
        if eigs.size < 8:
            diag["reason"] = "eig_fail_pre_norm"
            return None, diag
        r = float(max(abs(float(eigs[0])), abs(float(eigs[-1]))))
    except Exception:
        diag["reason"] = "eig_exception_pre_norm"
        return None, diag
    if not (math.isfinite(r) and r > 1e-12):
        diag["reason"] = "radius_nonfinite"
        return None, diag
    target_radius = float(max(4.0, n / 4.0))
    H = H * (target_radius / r)
    herm_err = float(np.linalg.norm(H - H.conj().T, ord="fro"))
    diag.update({"stable": True, "trace_abs": float(abs(np.trace(H))), "spectral_radius": float(target_radius), "hermitian_fro_error": float(herm_err)})
    return H, diag


# ----------------------------
# Rescaling models
# ----------------------------


@dataclass
class FitResult:
    fit_mode: str
    a: float
    b: float
    c: float
    J_fast: float


def apply_rescaling(E: np.ndarray, *, a: float, b: float, c: float, fit_mode: str) -> np.ndarray:
    x = np.asarray(E, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return x
    y = a * x + b
    if str(fit_mode) == "log_affine" and abs(float(c)) > 0.0:
        y = y + float(c) * np.log1p(np.maximum(0.0, x))
    y = np.asarray(y, dtype=np.float64)
    y = y[np.isfinite(y)]
    y.sort()
    return y


def scale_complexity_penalty(*, fit_mode: str, a: float, b: float, c: float) -> float:
    # prefer affine when comparable
    pen = 0.0
    pen += 0.05 * abs(float(math.log(max(1e-12, abs(a)))))  # mild
    pen += 0.005 * abs(float(b))
    if str(fit_mode) == "log_affine":
        pen += 1.0 + 0.25 * abs(float(c))
    return float(pen)


def make_windows(args: argparse.Namespace) -> List[Tuple[float, float]]:
    return rd.make_windows(float(args.window_min), float(args.window_max), float(args.window_size), float(args.window_stride))


def unfold_to_mean_spacing_one(x: np.ndarray) -> np.ndarray:
    return rd.unfold_to_mean_spacing_one(np.asarray(x, dtype=np.float64).reshape(-1))


def make_L_grid(L_min: float, L_max: float, n_L: int) -> np.ndarray:
    if n_L < 2:
        return np.asarray([1.0, 2.0], dtype=np.float64)
    L_min = float(L_min)
    L_max = float(L_max)
    if not (math.isfinite(L_min) and math.isfinite(L_max)) or L_max <= L_min:
        L_min, L_max = 0.5, 8.0
    return np.linspace(L_min, L_max, int(n_L), dtype=np.float64)


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


def support_overlap(levels: np.ndarray, target: np.ndarray, windows: List[Tuple[float, float]]) -> float:
    active = 0
    both = 0
    for a, b in windows:
        nt = int(rd.count_in_window(target, float(a), float(b)))
        if nt <= 0:
            continue
        active += 1
        no = int(rd.count_in_window(levels, float(a), float(b)))
        if no > 0:
            both += 1
    return float(both) / float(max(1, active))


def active_argument_counts(levels: np.ndarray, target: np.ndarray, windows: List[Tuple[float, float]]) -> Tuple[float, List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    errs_norm: List[float] = []
    for a, b in windows:
        nt = int(rd.count_in_window(target, float(a), float(b)))
        no = int(rd.count_in_window(levels, float(a), float(b)))
        active = (nt > 0) or (no > 0)
        if not active:
            continue
        err = float(abs(no - nt))
        err_norm = float(err / float(max(1, nt)))
        errs_norm.append(err_norm)
        rows.append(
            {
                "window_a": float(a),
                "window_b": float(b),
                "N_operator": int(no),
                "N_target": int(nt),
                "N_error": float(err),
                "N_error_norm": float(err_norm),
                "active_window": bool(active),
            }
        )
    med = float(np.median(np.asarray(errs_norm, dtype=np.float64))) if errs_norm else float("nan")
    return med, rows


def residue_scores(levels: np.ndarray, target: np.ndarray, windows: List[Tuple[float, float]]) -> Tuple[float, List[Dict[str, Any]]]:
    errs: List[float] = []
    rows: List[Dict[str, Any]] = []
    for a, b in windows:
        no = int(rd.count_in_window(levels, float(a), float(b)))
        nt = int(rd.count_in_window(target, float(a), float(b)))
        if (no == 0) and (nt == 0):
            continue
        I_op = rd.residue_proxy_count(levels, float(a), float(b), eta=0.15, n_contour_points=256)
        I_tg = rd.residue_proxy_count(target, float(a), float(b), eta=0.15, n_contour_points=256)
        err = abs(float(I_op.real) - float(I_tg.real)) / max(1.0, abs(float(I_tg.real)))
        rows.append(
            {
                "window_a": float(a),
                "window_b": float(b),
                "I_operator_real": float(I_op.real),
                "I_operator_imag": float(I_op.imag),
                "I_target_real": float(I_tg.real),
                "I_target_imag": float(I_tg.imag),
                "residue_count_error": float(err),
                "residue_imag_leak": float(abs(float(I_op.imag))),
            }
        )
        errs.append(float(err))
    med = float(np.median(np.asarray(errs, dtype=np.float64))) if errs else float("nan")
    return med, rows


def trace_proxy_rows(levels: np.ndarray, target: np.ndarray, windows: List[Tuple[float, float]]) -> Tuple[float, List[Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    errs: List[float] = []
    for a, b in windows:
        no = int(rd.count_in_window(levels, float(a), float(b)))
        nt = int(rd.count_in_window(target, float(a), float(b)))
        if (no == 0) and (nt == 0):
            continue
        c0 = 0.5 * (float(a) + float(b))
        for s in (0.5, 1.0, 2.0, 4.0):
            Sop = rd.trace_formula_proxy(levels, center=float(c0), sigma=float(s))
            Stg = rd.trace_formula_proxy(target, center=float(c0), sigma=float(s))
            if not (math.isfinite(Sop) and math.isfinite(Stg)):
                continue
            lop = math.log1p(max(0.0, float(Sop)))
            ltg = math.log1p(max(0.0, float(Stg)))
            denom = max(1.0, abs(float(ltg)))
            err = abs(float(lop) - float(ltg)) / denom
            errs.append(float(err))
            rows.append(
                {
                    "window_a": float(a),
                    "window_b": float(b),
                    "center": float(c0),
                    "sigma": float(s),
                    "S_operator": float(Sop),
                    "S_target": float(Stg),
                    "trace_error_norm": float(err),
                }
            )
    med = float(np.median(np.asarray(errs, dtype=np.float64))) if errs else float("nan")
    return med, rows


def mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    m = float(np.median(x))
    return float(np.median(np.abs(x - m)))


def normalize_curve(y: np.ndarray) -> Tuple[np.ndarray, float, float]:
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return np.asarray([], dtype=np.float64), float("nan"), float("nan")
    m = float(np.median(y))
    s = float(mad(y))
    if not (math.isfinite(s) and s > 1e-12):
        s = float(np.std(y))
    if not (math.isfinite(s) and s > 1e-12):
        s = 1.0
    return (y - m) / s, m, s


def hadamard_error(
    *,
    scaled_levels: np.ndarray,
    target: np.ndarray,
    z_grid: np.ndarray,
    eps: float,
) -> Tuple[float, Dict[str, Any], List[Dict[str, Any]]]:
    cand = np.asarray(scaled_levels, dtype=np.float64).reshape(-1)
    tgt = np.asarray(target, dtype=np.float64).reshape(-1)
    cand = cand[np.isfinite(cand) & (cand > 0)]
    tgt = tgt[np.isfinite(tgt) & (tgt > 0)]
    cand.sort()
    tgt.sort()
    N = int(min(cand.size, tgt.size))
    meta = {"N": int(N)}
    if N < 8:
        return float("nan"), {**meta, "reason": "insufficient_ordinates"}, []
    cand = cand[:N]
    tgt = tgt[:N]
    z = np.asarray(z_grid, dtype=np.float64)
    z2 = z * z

    def logxi(vals: np.ndarray) -> np.ndarray:
        v2 = vals * vals + float(eps)
        M = 1.0 - (z2.reshape(-1, 1) / v2.reshape(1, -1))
        Y = np.log(np.abs(M) + float(eps))
        return np.sum(Y, axis=1).astype(np.float64, copy=False)

    y_c = logxi(cand)
    y_t = logxi(tgt)
    y_c_n, _mc, _sc = normalize_curve(y_c)
    y_t_n, _mt, _st = normalize_curve(y_t)
    if y_c_n.size == 0 or y_t_n.size == 0 or y_c_n.size != y_t_n.size:
        return float("nan"), {**meta, "reason": "normalize_fail"}, []

    m = np.isfinite(y_c_n) & np.isfinite(y_t_n) & np.isfinite(z)
    if m.sum() < 8:
        return float("nan"), {**meta, "reason": "finite_mask_fail"}, []
    yc = y_c_n[m]
    yt = y_t_n[m]
    zz = z[m]
    rmse = float(np.sqrt(np.mean((yc - yt) ** 2)))
    yc0 = yc - float(np.mean(yc))
    yt0 = yt - float(np.mean(yt))
    denom = float(np.sqrt(np.mean(yc0**2) * np.mean(yt0**2)))
    corr = float(np.mean(yc0 * yt0) / denom) if denom > 1e-12 else float("nan")
    try:
        zc = float(zz[int(np.argmin(yc))])
        zt = float(zz[int(np.argmin(yt))])
        peak = float(abs(zc - zt) / max(1e-12, float(np.max(zz) - float(np.min(zz)))))
    except Exception:
        peak = float("nan")
    L_had = float(rmse + 0.25 * (1.0 - (corr if math.isfinite(corr) else 0.0)) + 0.25 * (peak if math.isfinite(peak) else 1.0))
    rows = []
    for z0, a0, b0, an, bn in zip(z, y_c, y_t, y_c_n, y_t_n):
        rows.append(
            {
                "z": float(z0),
                "logdet_candidate": safe_float(a0),
                "logdet_target": safe_float(b0),
                "logdet_candidate_norm": safe_float(an),
                "logdet_target_norm": safe_float(bn),
            }
        )
    meta.update({"hadamard_rmse": float(rmse), "hadamard_corr": float(corr), "hadamard_peak_alignment_error": float(peak), "L_had": float(L_had)})
    return float(L_had), meta, rows


# ----------------------------
# Null baselines (V13o14 + V14.7b best-effort)
# ----------------------------


def load_baseline_pool(*, dims: List[int], v13o14_dir: Path, v14_7b_dir: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    pool: List[Dict[str, Any]] = []
    missing: List[str] = []

    def add(dim: int, kind: str, J: Any, source: str, label: str, source_file: str, missing_source: bool) -> None:
        pool.append(
            {
                "dim": int(dim),
                "baseline_kind": str(kind),
                "J": safe_float(J, float("nan")),
                "source": str(source),
                "label": str(label),
                "source_file": str(source_file),
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
                        add(d, kind, J, "v13o14", label, "v13o14_null_comparisons.csv", False)

    r7bn = _read_csv_best_effort(v14_7b_dir / "v14_7b_null_comparisons.csv")
    if not r7bn:
        missing.append("v14_7b_null_comparisons.csv")
    else:
        for r in r7bn:
            d = safe_int(r.get("dim", 0))
            if d not in dims:
                continue
            if "best_prior_artin_J" in r:
                J = safe_float(r.get("best_prior_artin_J", float("nan")))
                if math.isfinite(J):
                    add(d, "prior_artin", J, "v14_7b", "best_prior_artin_J", "v14_7b_null_comparisons.csv", False)

    for d in dims:
        any_dim = any(int(rr.get("dim", 0)) == int(d) and not bool(rr.get("missing_source", False)) and math.isfinite(safe_float(rr.get("J", float("nan")))) for rr in pool)
        if not any_dim:
            add(d, "missing", float("nan"), "missing", "no_baselines_loaded", "", True)

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
# Fit search per candidate
# ----------------------------


def coarse_fit_affine(
    E: np.ndarray,
    target: np.ndarray,
    windows: List[Tuple[float, float]],
    args: argparse.Namespace,
) -> FitResult:
    # coarse grid over a,b evaluating only support + argument (fast)
    a_grid = np.linspace(float(args.scale_grid_min), float(args.scale_grid_max), int(args.scale_grid_n), dtype=np.float64)
    b_grid = np.linspace(float(args.shift_grid_min), float(args.shift_grid_max), int(args.shift_grid_n), dtype=np.float64)
    best = FitResult("affine", a=1.0, b=0.0, c=0.0, J_fast=float("inf"))
    for ai, a in enumerate(a_grid):
        for bi, b in enumerate(b_grid):
            lvl = apply_rescaling(E, a=float(a), b=float(b), c=0.0, fit_mode="affine")
            sup = support_overlap(lvl, target, windows)
            arg_med, _ = active_argument_counts(lvl, target, windows)
            if not math.isfinite(arg_med):
                continue
            support_loss = float(max(0.0, 0.6 - sup))
            J_fast = 20.0 * support_loss + 10.0 * float(arg_med) + 0.25 * scale_complexity_penalty(fit_mode="affine", a=float(a), b=float(b), c=0.0)
            if J_fast < best.J_fast:
                best = FitResult("affine", a=float(a), b=float(b), c=0.0, J_fast=float(J_fast))
    return best


def refine_fit_affine(
    E: np.ndarray,
    target: np.ndarray,
    windows: List[Tuple[float, float]],
    args: argparse.Namespace,
    seed_fit: FitResult,
) -> FitResult:
    # local refinement around best coarse (small neighborhood with finer steps)
    a0, b0 = float(seed_fit.a), float(seed_fit.b)
    da = max(1e-6, 0.05 * max(1.0, abs(a0)))
    db = max(1e-6, 0.05 * max(1.0, abs(b0)))
    a_grid = np.linspace(a0 - 2.0 * da, a0 + 2.0 * da, 9, dtype=np.float64)
    b_grid = np.linspace(b0 - 2.0 * db, b0 + 2.0 * db, 9, dtype=np.float64)
    best = FitResult("affine", a=a0, b=b0, c=0.0, J_fast=float("inf"))
    for a in a_grid:
        if a <= 0:
            continue
        for b in b_grid:
            lvl = apply_rescaling(E, a=float(a), b=float(b), c=0.0, fit_mode="affine")
            sup = support_overlap(lvl, target, windows)
            arg_med, _ = active_argument_counts(lvl, target, windows)
            if not math.isfinite(arg_med):
                continue
            support_loss = float(max(0.0, 0.6 - sup))
            J_fast = 20.0 * support_loss + 10.0 * float(arg_med) + 0.25 * scale_complexity_penalty(fit_mode="affine", a=float(a), b=float(b), c=0.0)
            if J_fast < best.J_fast:
                best = FitResult("affine", a=float(a), b=float(b), c=0.0, J_fast=float(J_fast))
    return best


def try_log_affine(
    E: np.ndarray,
    target: np.ndarray,
    windows: List[Tuple[float, float]],
    args: argparse.Namespace,
    affine_best: FitResult,
) -> FitResult:
    # small c sweep, keep a/b near affine optimum, penalize complexity
    a0, b0 = float(affine_best.a), float(affine_best.b)
    c_grid = [-1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0]
    best = FitResult("log_affine", a=a0, b=b0, c=0.0, J_fast=float("inf"))
    for c in c_grid:
        lvl = apply_rescaling(E, a=a0, b=b0, c=float(c), fit_mode="log_affine")
        sup = support_overlap(lvl, target, windows)
        arg_med, _ = active_argument_counts(lvl, target, windows)
        if not math.isfinite(arg_med):
            continue
        support_loss = float(max(0.0, 0.6 - sup))
        J_fast = 20.0 * support_loss + 10.0 * float(arg_med) + 0.25 * scale_complexity_penalty(fit_mode="log_affine", a=a0, b=b0, c=float(c))
        if J_fast < best.J_fast:
            best = FitResult("log_affine", a=a0, b=b0, c=float(c), J_fast=float(J_fast))
    return best


# ----------------------------
# Full metrics + objective + gates
# ----------------------------


def compute_full_metrics(
    *,
    dim: int,
    family: str,
    mode: str,
    word: List[Token],
    eig_raw: np.ndarray,
    zeros: np.ndarray,
    windows: List[Tuple[float, float]],
    L_grid: np.ndarray,
    z_grid: np.ndarray,
    fit: FitResult,
    args: argparse.Namespace,
    baseline_pool: List[Dict[str, Any]],
    missing_baselines_set: set[str],
) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    lvl = apply_rescaling(eig_raw, a=float(fit.a), b=float(fit.b), c=float(fit.c), fit_mode=str(fit.fit_mode))
    tgt = np.asarray(zeros, dtype=np.float64)
    tgt = tgt[np.isfinite(tgt)]
    tgt.sort()

    sup = float(support_overlap(lvl, tgt, windows))
    arg_med, arg_rows = active_argument_counts(lvl, tgt, windows)
    res_med, res_rows = residue_scores(lvl, tgt, windows)
    tr_med, tr_rows = trace_proxy_rows(lvl, tgt, windows)

    # hadamard
    L_had, had_meta, had_curve = hadamard_error(scaled_levels=lvl, target=tgt, z_grid=z_grid, eps=float(1e-9))
    had_err = float(L_had) if math.isfinite(L_had) else float("nan")

    # NV
    lv_nv = unfold_to_mean_spacing_one(lvl)
    tg_nv = unfold_to_mean_spacing_one(tgt)
    op_nv = number_variance_curve(lv_nv, L_grid)
    tgt_nv = number_variance_curve(tg_nv, L_grid)
    nv_rmse = float(curve_l2(op_nv, tgt_nv))
    m_nv = np.isfinite(op_nv)
    nv_range = float(np.max(op_nv[m_nv]) - np.min(op_nv[m_nv])) if m_nv.any() else float("nan")
    nv_median = float(np.median(op_nv[m_nv])) if m_nv.any() else float("nan")
    pois = float(poisson_like_fraction_from_nv(op_nv, tgt_nv, L_grid)) if m_nv.any() else float("nan")

    # null comparisons vs baseline pool (best-effort)
    best_random = best_baseline_J(baseline_pool, int(dim), "random")
    best_rej = best_baseline_J(baseline_pool, int(dim), "rejected")
    best_abl = best_baseline_J(baseline_pool, int(dim), "ablation")
    best_prior = best_baseline_J(baseline_pool, int(dim), "prior_artin")
    best_null = best_null_J(baseline_pool, int(dim))
    dist = null_dist(baseline_pool, int(dim))

    for kind, val in [("random", best_random), ("rejected", best_rej), ("ablation", best_abl), ("prior_artin", best_prior)]:
        if val is None:
            missing_baselines_set.add(kind)

    # objective J_v14_8b
    lambda_support = 20.0
    lambda_arg = 10.0
    lambda_res = 5.0
    lambda_trace = 2.0
    lambda_had = 5.0
    lambda_nv = 2.0
    lambda_pois = 5.0
    lambda_scale = 0.25
    lambda_null = 10.0

    support_loss = float(max(0.0, 0.6 - sup))
    nv_loss = float(nv_rmse) if math.isfinite(nv_rmse) else 1e3
    scale_pen = float(scale_complexity_penalty(fit_mode=str(fit.fit_mode), a=float(fit.a), b=float(fit.b), c=float(fit.c)))

    # null_failure_penalty: count failed null gates among those with baselines present
    null_fail = 0
    beats_random = False if best_random is None else False
    beats_rejected = False if best_rej is None else False
    beats_ablation = False if best_abl is None else False
    beats_prior = False if best_prior is None else False

    # compute J first without null penalty? spec uses null_failure_penalty; use J itself comparison
    # We'll compute J_base, then beats vs baselines using J_base, then null_fail and final J.
    arg_term = float(arg_med) if math.isfinite(arg_med) else 1e3
    res_term = float(res_med) if math.isfinite(res_med) else 1e3
    tr_term = float(tr_med) if math.isfinite(tr_med) else 1e3
    had_term = float(had_err) if math.isfinite(had_err) else 1e3
    pois_term = float(pois) if math.isfinite(pois) else 1.0

    J_base = (
        lambda_support * support_loss
        + lambda_arg * arg_term
        + lambda_res * res_term
        + lambda_trace * tr_term
        + lambda_had * had_term
        + lambda_nv * nv_loss
        + lambda_pois * pois_term
        + lambda_scale * scale_pen
    )

    beats_random = bool(best_random is not None and math.isfinite(best_random) and float(J_base) < float(best_random))
    beats_rejected = bool(best_rej is not None and math.isfinite(best_rej) and float(J_base) < float(best_rej))
    beats_ablation = bool(best_abl is not None and math.isfinite(best_abl) and float(J_base) < float(best_abl))
    beats_prior = bool(best_prior is not None and math.isfinite(best_prior) and float(J_base) < float(best_prior))

    for present, ok in [
        (best_random is not None, beats_random),
        (best_rej is not None, beats_rejected),
        (best_abl is not None, beats_ablation),
        (best_prior is not None, beats_prior),
    ]:
        if present and (not ok):
            null_fail += 1
        if (not present):
            null_fail += 1  # do not silently pass

    J = float(J_base + lambda_null * float(null_fail))
    final_reward = float(1.0 / (1.0 + J)) if math.isfinite(J) else 0.0
    if not math.isfinite(final_reward):
        final_reward = 0.0

    null_sep = float(best_null - J) if (best_null is not None and math.isfinite(best_null)) else float("nan")
    null_z = float("nan")
    null_pct = float("nan")
    if len(dist) >= 3 and math.isfinite(J):
        mu = float(np.mean(dist))
        sd = float(np.std(dist))
        null_z = float((mu - float(J)) / sd) if sd > 1e-12 else float("nan")
        null_pct = float(np.mean([1.0 if x <= float(J) else 0.0 for x in dist]))

    # gates
    G1 = bool(math.isfinite(J)) and (eig_raw.size >= 8)
    G2 = bool(sup >= 0.6)
    G3 = bool(math.isfinite(arg_med) and float(arg_med) <= 0.25)
    G4 = bool(math.isfinite(res_med) and float(res_med) <= 0.25)
    G5 = bool(math.isfinite(tr_med) and float(tr_med) <= 0.75)
    G6 = bool(math.isfinite(had_err) and float(had_err) <= 1.0)
    G7 = bool(math.isfinite(nv_range) and math.isfinite(nv_median))
    G8 = bool(math.isfinite(pois) and float(pois) <= 0.5)
    G9 = bool(0.05 <= float(fit.a) <= 200.0 and abs(float(fit.b)) <= 500.0)
    G10 = bool(beats_random) if best_random is not None else False
    G11 = bool(beats_rejected) if best_rej is not None else False
    G12 = bool(beats_ablation) if best_abl is not None else False
    G13 = bool(beats_prior) if best_prior is not None else False
    all_gate = bool(G1 and G2 and G3 and G4 and G5 and G6 and G7 and G8 and G9 and G10 and G11 and G12 and G13)

    if not G1:
        classification = "UNSTABLE"
    elif not G2:
        classification = "ZERO_SUPPORT_REJECT"
    elif not G3:
        classification = "ARGUMENT_FAIL"
    elif not G4:
        classification = "RESIDUE_FAIL"
    elif not G5:
        classification = "TRACE_FAIL"
    elif not G6:
        classification = "HADAMARD_FAIL"
    elif not G8:
        classification = "POISSON_LIKE_FAIL"
    elif not (G10 and G11 and G12 and G13):
        classification = "NULL_CONTROL_FAIL"
    else:
        classification = "PASS"

    summary = {
        "dim": int(dim),
        "family": str(family),
        "mode": str(mode),
        "word": word_to_string(word),
        "fit_mode": str(fit.fit_mode),
        "a": float(fit.a),
        "b": float(fit.b),
        "c": float(fit.c),
        "support_overlap": float(sup),
        "active_argument_error_med": float(arg_med) if math.isfinite(arg_med) else float("nan"),
        "residue_error_med": float(res_med) if math.isfinite(res_med) else float("nan"),
        "trace_error_med": float(tr_med) if math.isfinite(tr_med) else float("nan"),
        "hadamard_error_med": float(had_err) if math.isfinite(had_err) else float("nan"),
        "nv_range": float(nv_range) if math.isfinite(nv_range) else float("nan"),
        "nv_median": float(nv_median) if math.isfinite(nv_median) else float("nan"),
        "poisson_like_fraction": float(pois) if math.isfinite(pois) else float("nan"),
        "null_failures": int(null_fail),
        "beats_random": bool(beats_random),
        "beats_rejected": bool(beats_rejected),
        "beats_ablation": bool(beats_ablation),
        "beats_prior_artin": bool(beats_prior),
        "null_separation": float(null_sep),
        "null_zscore": float(null_z),
        "null_percentile": float(null_pct),
        "missing_baselines": "|".join(sorted(missing_baselines_set)),
        "J_v14_8b": float(J),
        "final_reward": float(final_reward),
        "G1_stable": bool(G1),
        "G2_support_overlap_ok": bool(G2),
        "G3_active_argument_ok": bool(G3),
        "G4_residue_error_ok": bool(G4),
        "G5_trace_proxy_ok": bool(G5),
        "G6_hadamard_determinant_ok": bool(G6),
        "G7_number_variance_ok": bool(G7),
        "G8_not_poisson_like": bool(G8),
        "G9_scale_not_degenerate": bool(G9),
        "G10_beats_random_controls": bool(G10),
        "G11_beats_rejected_control": bool(G11),
        "G12_beats_ablation_controls": bool(G12),
        "G13_beats_prior_artin_search": bool(G13),
        "G14_all_gate_pass": bool(all_gate),
        "classification": str(classification),
    }
    detail = {
        "active_argument_counts": arg_rows,
        "residue_scores": res_rows,
        "trace_proxy": tr_rows,
        "hadamard_curve": had_curve,
        "nv_curve": [{"kind": "operator", "L": float(L), "Sigma2": safe_float(y)} for L, y in zip(L_grid, op_nv)],
    }
    return summary, had_meta, detail["hadamard_curve"]


# ----------------------------
# Report rendering
# ----------------------------


def render_report_md(
    *,
    out_dir: Path,
    config: Dict[str, Any],
    missing_sources: List[str],
    best_rows: List[Dict[str, Any]],
    gate_rows: List[Dict[str, Any]],
    improves_zero_support: Optional[bool],
) -> str:
    md: List[str] = []
    md.append("# V14.8b — Spectral Support Rescaling for Braid-Graph Laplacian\n\n")
    md.append("> Computational evidence only; not a proof of RH.\n\n")
    md.append("## 1. Purpose\n")
    md.append("Apply monotone spectral rescaling on top of V14.8 braid-graph Laplacian candidates to improve active-window support and downstream argument/residue/Hadamard diagnostics.\n\n")
    md.append("## 2. Why V14.8 failed\n")
    md.append("V14.8 failures were dominated by `ZERO_SUPPORT_REJECT` and argument-count mismatch (active windows). Trace proxy was comparatively stable.\n\n")
    md.append("## 3. What spectral support rescaling does\n")
    md.append("Fit affine or log-affine rescaling (E→aE+b(+c·log(1+max(E,0)))) to align spectrum to the active zeta window range, increasing support overlap and improving window-based counts.\n\n")
    md.append("## 4. Best candidate per dim/family/mode\n")
    if best_rows:
        for r in best_rows:
            md.append(
                f"- dim={r.get('dim')} family={r.get('family')} mode={r.get('mode')} "
                f"J={safe_float(r.get('J_v14_8b')):.6g} reward={safe_float(r.get('final_reward')):.6g} "
                f"support={safe_float(r.get('support_overlap')):.3g} arg={safe_float(r.get('active_argument_error_med')):.3g} "
                f"had={safe_float(r.get('hadamard_error_med')):.3g} fit={r.get('fit_mode')} a={safe_float(r.get('a')):.3g} b={safe_float(r.get('b')):.3g}\n"
            )
    else:
        md.append("- (none)\n")
    md.append("\n")
    md.append("## 5. Whether active support improved\n")
    if improves_zero_support is None:
        md.append("- (could not compare: V14.8 inputs missing)\n")
    else:
        md.append(f"- Improves over V14.8 on ZERO_SUPPORT_REJECT frequency: **{improves_zero_support}**\n")
    md.append("\n")
    md.append("## 6–8. Argument-count / Hadamard / Null controls\n")
    md.append("See `v14_8b_gate_summary.csv` and `v14_8b_null_comparisons.csv`.\n\n")
    md.append("## 9. Decision\n")
    md.append(f"- proceed_to_v14_9 = True only if any `G14_all_gate_pass` is True.\n")
    md.append("- analytic_claim = False unless all gates + independent null validation pass.\n\n")
    md.append("## Verification commands\n")
    md.append("```bash\n")
    md.append('OUT=runs/v14_8b_spectral_support_rescaling\n\n')
    md.append('echo "=== FILES ==="\nfind "$OUT" -maxdepth 1 -type f | sort\n\n')
    md.append('echo "=== BEST ==="\ncolumn -s, -t < "$OUT"/v14_8b_best_candidates.csv\n\n')
    md.append('echo "=== GATE ==="\ncolumn -s, -t < "$OUT"/v14_8b_gate_summary.csv\n\n')
    md.append('echo "=== RANKING TOP ==="\ncolumn -s, -t < "$OUT"/v14_8b_candidate_ranking.csv | head -80\n\n')
    md.append('echo "=== REPORT ==="\nhead -240 "$OUT"/v14_8b_report.md\n')
    md.append("```\n\n")
    md.append("> Computational evidence only; not a proof of RH.\n")
    return "".join(md)


# ----------------------------
# Main
# ----------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="V14.8b spectral support rescaling (computational only).")
    ap.add_argument("--v14_8_dir", type=str, default="runs/v14_8_braid_graph_laplacian_hadamard_gate")
    ap.add_argument("--v13o14_dir", type=str, default="runs/v13o14_transport_null_controls")
    ap.add_argument("--v14_7b_dir", type=str, default="runs/v14_7b_support_calibrated_antipoisson_full")
    ap.add_argument("--zeros_csv", type=str, default="runs/zeros_100_400_precise.csv")
    ap.add_argument("--out_dir", type=str, default="runs/v14_8b_spectral_support_rescaling")
    ap.add_argument("--dims", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument(
        "--families",
        type=str,
        nargs="+",
        default=[
            "plain_graph_laplacian",
            "magnetic_graph_laplacian",
            "signed_braid_laplacian",
            "curvature_regularized_laplacian",
            "hybrid_braid_graph_laplacian",
        ],
    )
    ap.add_argument("--modes", type=str, nargs="+", default=["numeric_only", "hybrid_ranked_anticollapse"])
    ap.add_argument("--fit_modes", type=str, nargs="+", default=["affine", "log_affine"])
    ap.add_argument("--window_min", type=float, default=100.0)
    ap.add_argument("--window_max", type=float, default=300.0)
    ap.add_argument("--window_size", type=float, default=40.0)
    ap.add_argument("--window_stride", type=float, default=20.0)
    ap.add_argument("--scale_grid_min", type=float, default=0.1)
    ap.add_argument("--scale_grid_max", type=float, default=100.0)
    ap.add_argument("--scale_grid_n", type=int, default=80)
    ap.add_argument("--shift_grid_min", type=float, default=-200.0)
    ap.add_argument("--shift_grid_max", type=float, default=200.0)
    ap.add_argument("--shift_grid_n", type=int, default=80)
    ap.add_argument("--top_k_candidates", type=int, default=10)
    ap.add_argument("--progress_every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260507)
    args = ap.parse_args()

    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    rng = random.Random(int(args.seed))
    np_rng = np.random.default_rng(int(args.seed))

    dims = [int(d) for d in args.dims]
    families = [str(f) for f in args.families]
    modes = [str(m) for m in args.modes]
    fit_modes = [str(m) for m in args.fit_modes]

    missing_sources: List[str] = []
    warnings: List[str] = []

    # resolve v14_8_dir: if default doesn't exist, fall back to known v14_8 path
    v14_8_dir = _resolve(args.v14_8_dir)
    if not v14_8_dir.is_dir():
        alt = _resolve("runs/v14_8_braid_graph_laplacian_hadamard")
        if alt.is_dir():
            v14_8_dir = alt
            warnings.append(f"v14_8_dir fallback used: {alt}")
        else:
            raise SystemExit(f"v14_8_dir not found: {v14_8_dir}")

    # Load V14.8 ranking for seed candidates
    v14_8_ranking_path = v14_8_dir / "v14_8_candidate_ranking.csv"
    v14_8_rows = _read_csv_best_effort(v14_8_ranking_path)
    if not v14_8_rows:
        raise SystemExit(f"Missing/empty required input: {v14_8_ranking_path}")

    # Load zeros
    zeros_csv = _resolve(args.zeros_csv)
    if not zeros_csv.is_file():
        raise SystemExit(f"zeros_csv missing: {zeros_csv}")
    zeros_raw, zeros_warns = rd.load_zeros_csv(zeros_csv)
    warnings.extend([f"zeros_csv: {w}" for w in zeros_warns])
    zeros_unfolded = unfold_to_mean_spacing_one(np.asarray(zeros_raw, dtype=np.float64))

    windows = make_windows(args)
    if not windows:
        raise SystemExit("No windows produced; check window args.")
    L_grid = make_L_grid(0.5, 8.0, 16)
    z_grid = np.linspace(0.5, 32.0, 128, dtype=np.float64)

    # Null baselines pool
    baseline_pool, missing_baselines_files = load_baseline_pool(dims=dims, v13o14_dir=_resolve(args.v13o14_dir), v14_7b_dir=_resolve(args.v14_7b_dir))
    for m in missing_baselines_files:
        missing_sources.append(f"baseline_missing:{m}")

    # outputs
    scaled_spectra_rows: List[Dict[str, Any]] = []
    transport_rows: List[Dict[str, Any]] = []
    arg_rows_all: List[Dict[str, Any]] = []
    residue_rows_all: List[Dict[str, Any]] = []
    trace_rows_all: List[Dict[str, Any]] = []
    had_rows_all: List[Dict[str, Any]] = []
    nv_rows_all: List[Dict[str, Any]] = []
    ranking_rows: List[Dict[str, Any]] = []
    best_rows: List[Dict[str, Any]] = []
    gate_rows: List[Dict[str, Any]] = []
    null_rows: List[Dict[str, Any]] = []

    # compare ZERO_SUPPORT_REJECT freq vs V14.8
    zero_support_v148 = sum(1 for r in v14_8_rows if str(r.get("classification", "")).strip() == "ZERO_SUPPORT_REJECT")

    # process candidates
    n_done = 0
    for d in dims:
        for fam in families:
            for mode in modes:
                # select top_k candidates from V14.8 for this dim/family/mode
                sub = [r for r in v14_8_rows if safe_int(r.get("dim", -1)) == int(d) and str(r.get("mode", "")) == str(fam) and str(r.get("search_mode", "")) == str(mode)]
                sub.sort(key=lambda r: safe_float(r.get("J_v14_8", float("inf")), float("inf")))
                seeds = sub[: int(args.top_k_candidates)]
                if not seeds:
                    continue

                for seed_rank, r in enumerate(seeds, start=1):
                    w_warns: List[str] = []
                    w = parse_braid_word(str(r.get("word", "")), dim=int(d), max_power=3, warnings=w_warns)
                    if not w:
                        continue

                    # reconstruct raw spectrum via V14.8 operator
                    H, gdiag = build_operator_from_word(dim=int(d), word=w, family=str(fam), seed=int(args.seed + 1000 * d + 17 * seed_rank))
                    if H is None or not bool(gdiag.get("stable", False)):
                        # unstable -> still record a row with UNSTABLE
                        ranking_rows.append(
                            {
                                "dim": int(d),
                                "family": str(fam),
                                "mode": str(mode),
                                "seed_rank": int(seed_rank),
                                "word": str(r.get("word", "")),
                                "fit_mode": "none",
                                "a": float("nan"),
                                "b": float("nan"),
                                "c": float("nan"),
                                "support_overlap": 0.0,
                                "active_argument_error_med": float("nan"),
                                "residue_error_med": float("nan"),
                                "trace_error_med": float("nan"),
                                "hadamard_error_med": float("nan"),
                                "nv_range": float("nan"),
                                "nv_median": float("nan"),
                                "poisson_like_fraction": float("nan"),
                                "null_failures": 4,
                                "beats_random": False,
                                "beats_rejected": False,
                                "beats_ablation": False,
                                "beats_prior_artin": False,
                                "missing_baselines": "unknown",
                                "J_v14_8b": 1e6,
                                "final_reward": 0.0,
                                "classification": "UNSTABLE",
                            }
                        )
                        continue

                    try:
                        eig = np.linalg.eigvalsh(np.asarray(H.real, dtype=np.float64))
                        eig = np.asarray(eig, dtype=np.float64).reshape(-1)
                        eig = eig[np.isfinite(eig)]
                        eig.sort()
                    except Exception:
                        eig = np.asarray([], dtype=np.float64)
                    if eig.size < 8:
                        continue

                    # choose fit (affine + optional log_affine)
                    fit0 = coarse_fit_affine(eig, zeros_unfolded, windows, args)
                    fit1 = refine_fit_affine(eig, zeros_unfolded, windows, args, fit0)
                    best_fit = fit1
                    if "log_affine" in set(fit_modes):
                        fit_log = try_log_affine(eig, zeros_unfolded, windows, args, best_fit)
                        # prefer affine if comparable (complexity penalty baked into J_fast)
                        if float(fit_log.J_fast) + 1e-9 < float(best_fit.J_fast):
                            best_fit = fit_log

                    missing_baselines_set: set[str] = set()
                    summary, had_meta, had_curve = compute_full_metrics(
                        dim=int(d),
                        family=str(fam),
                        mode=str(mode),
                        word=w,
                        eig_raw=eig,
                        zeros=zeros_unfolded,
                        windows=windows,
                        L_grid=L_grid,
                        z_grid=z_grid,
                        fit=best_fit,
                        args=args,
                        baseline_pool=baseline_pool,
                        missing_baselines_set=missing_baselines_set,
                    )

                    ranking_rows.append(summary)
                    transport_rows.append(
                        {
                            "dim": int(d),
                            "family": str(fam),
                            "mode": str(mode),
                            "word": str(summary["word"]),
                            "fit_mode": str(summary["fit_mode"]),
                            "a": float(summary["a"]),
                            "b": float(summary["b"]),
                            "c": float(summary["c"]),
                            "J_fast": float(best_fit.J_fast),
                            "seed_rank": int(seed_rank),
                        }
                    )

                    # store a compact spectrum dump (first 2*dim values max)
                    lvl_scaled = apply_rescaling(eig, a=float(summary["a"]), b=float(summary["b"]), c=float(summary["c"]), fit_mode=str(summary["fit_mode"]))
                    for idx in range(int(min(2 * int(d), eig.size, lvl_scaled.size))):
                        scaled_spectra_rows.append(
                            {
                                "dim": int(d),
                                "family": str(fam),
                                "mode": str(mode),
                                "word": str(summary["word"]),
                                "idx": int(idx),
                                "E_raw": float(eig[idx]),
                                "E_scaled": float(lvl_scaled[idx]),
                            }
                        )

                    # detail rows (best effort) – append with identifiers
                    # argument/residue/trace: recompute rows directly for consistency (using scaled levels)
                    lvl = lvl_scaled
                    arg_med, arows = active_argument_counts(lvl, zeros_unfolded, windows)
                    for rr in arows:
                        arg_rows_all.append({"dim": int(d), "family": str(fam), "mode": str(mode), "word": str(summary["word"]), **rr})
                    res_med, rrows = residue_scores(lvl, zeros_unfolded, windows)
                    for rr in rrows:
                        residue_rows_all.append({"dim": int(d), "family": str(fam), "mode": str(mode), "word": str(summary["word"]), **rr})
                    tr_med, trows = trace_proxy_rows(lvl, zeros_unfolded, windows)
                    for rr in trows:
                        trace_rows_all.append({"dim": int(d), "family": str(fam), "mode": str(mode), "word": str(summary["word"]), **rr})
                    # hadamard curves (subsample)
                    for rr in had_curve[:: max(1, int(len(had_curve) / 64))]:
                        had_rows_all.append({"dim": int(d), "family": str(fam), "mode": str(mode), "word": str(summary["word"]), **rr})
                    # NV diagnostics
                    lv_nv = unfold_to_mean_spacing_one(lvl)
                    tg_nv = unfold_to_mean_spacing_one(zeros_unfolded)
                    op_nv = number_variance_curve(lv_nv, L_grid)
                    for L, y in zip(L_grid, op_nv):
                        nv_rows_all.append({"dim": int(d), "family": str(fam), "mode": str(mode), "word": str(summary["word"]), "kind": "operator", "L": float(L), "Sigma2": safe_float(y)})

                    n_done += 1
                    if n_done == 1 or (n_done % max(1, int(args.progress_every)) == 0):
                        elapsed = time.perf_counter() - t0
                        print(
                            f"[V14.8b] done={n_done} dim={d} family={fam} mode={mode} "
                            f"J={float(summary['J_v14_8b']):.6g} support={float(summary['support_overlap']):.3g} "
                            f"arg={safe_float(summary['active_argument_error_med']):.3g} had={safe_float(summary['hadamard_error_med']):.3g} "
                            f"eta={format_seconds(max(0.0, (elapsed/max(1,n_done))*max(0, (len(dims)*len(families)*len(modes)*int(args.top_k_candidates))-n_done)))}",
                            flush=True,
                        )

    # ranking: sort per dim/family/mode by J
    ranked: List[Dict[str, Any]] = []
    for d in dims:
        for fam in families:
            for mode in modes:
                sub = [r for r in ranking_rows if int(r.get("dim", -1)) == int(d) and str(r.get("family", "")) == str(fam) and str(r.get("mode", "")) == str(mode)]
                sub.sort(key=lambda r: safe_float(r.get("J_v14_8b", float("inf")), float("inf")))
                for k, rr in enumerate(sub, start=1):
                    row = dict(rr)
                    row["rank"] = int(k)
                    ranked.append(row)
                if sub:
                    best = sub[0]
                    best_rows.append(
                        {
                            "dim": int(d),
                            "family": str(fam),
                            "mode": str(mode),
                            "best_word": str(best.get("word", "")),
                            "J_v14_8b": safe_float(best.get("J_v14_8b", float("nan"))),
                            "final_reward": safe_float(best.get("final_reward", float("nan"))),
                            "support_overlap": safe_float(best.get("support_overlap", float("nan"))),
                            "active_argument_error_med": safe_float(best.get("active_argument_error_med", float("nan"))),
                            "hadamard_error_med": safe_float(best.get("hadamard_error_med", float("nan"))),
                            "fit_mode": str(best.get("fit_mode", "")),
                            "a": safe_float(best.get("a", float("nan"))),
                            "b": safe_float(best.get("b", float("nan"))),
                            "c": safe_float(best.get("c", float("nan"))),
                            "classification": str(best.get("classification", "")),
                        }
                    )

    # Gate + null summaries derived from best rows
    for r in ranked:
        gate_rows.append(
            {
                "dim": int(r.get("dim", 0)),
                "family": str(r.get("family", "")),
                "mode": str(r.get("mode", "")),
                "word": str(r.get("word", "")),
                "J_v14_8b": safe_float(r.get("J_v14_8b", float("nan"))),
                "final_reward": safe_float(r.get("final_reward", float("nan"))),
                "support_overlap": safe_float(r.get("support_overlap", float("nan"))),
                "active_argument_error_med": safe_float(r.get("active_argument_error_med", float("nan"))),
                "residue_error_med": safe_float(r.get("residue_error_med", float("nan"))),
                "trace_error_med": safe_float(r.get("trace_error_med", float("nan"))),
                "hadamard_error_med": safe_float(r.get("hadamard_error_med", float("nan"))),
                "poisson_like_fraction": safe_float(r.get("poisson_like_fraction", float("nan"))),
                "a": safe_float(r.get("a", float("nan"))),
                "b": safe_float(r.get("b", float("nan"))),
                "fit_mode": str(r.get("fit_mode", "")),
                "G1_stable": bool(r.get("G1_stable", False)),
                "G2_support_overlap_ok": bool(r.get("G2_support_overlap_ok", False)),
                "G3_active_argument_ok": bool(r.get("G3_active_argument_ok", False)),
                "G4_residue_error_ok": bool(r.get("G4_residue_error_ok", False)),
                "G5_trace_proxy_ok": bool(r.get("G5_trace_proxy_ok", False)),
                "G6_hadamard_determinant_ok": bool(r.get("G6_hadamard_determinant_ok", False)),
                "G7_number_variance_ok": bool(r.get("G7_number_variance_ok", False)),
                "G8_not_poisson_like": bool(r.get("G8_not_poisson_like", False)),
                "G9_scale_not_degenerate": bool(r.get("G9_scale_not_degenerate", False)),
                "G10_beats_random_controls": bool(r.get("G10_beats_random_controls", False)),
                "G11_beats_rejected_control": bool(r.get("G11_beats_rejected_control", False)),
                "G12_beats_ablation_controls": bool(r.get("G12_beats_ablation_controls", False)),
                "G13_beats_prior_artin_search": bool(r.get("G13_beats_prior_artin_search", False)),
                "G14_all_gate_pass": bool(r.get("G14_all_gate_pass", False)),
                "classification": str(r.get("classification", "")),
            }
        )
        null_rows.append(
            {
                "dim": int(r.get("dim", 0)),
                "family": str(r.get("family", "")),
                "mode": str(r.get("mode", "")),
                "word": str(r.get("word", "")),
                "J_v14_8b": safe_float(r.get("J_v14_8b", float("nan"))),
                "null_separation": safe_float(r.get("null_separation", float("nan"))),
                "null_zscore": safe_float(r.get("null_zscore", float("nan"))),
                "null_percentile": safe_float(r.get("null_percentile", float("nan"))),
                "beats_random": bool(r.get("beats_random", False)),
                "beats_rejected": bool(r.get("beats_rejected", False)),
                "beats_ablation": bool(r.get("beats_ablation", False)),
                "beats_prior_artin": bool(r.get("beats_prior_artin", False)),
                "missing_baselines": str(r.get("missing_baselines", "")),
            }
        )

    # improvement check
    zero_support_v148b = sum(1 for r in ranked if str(r.get("classification", "")).strip() == "ZERO_SUPPORT_REJECT")
    improves_zero_support = None
    if zero_support_v148 is not None:
        improves_zero_support = bool(zero_support_v148b < zero_support_v148)

    # proceed decision
    proceed_to_v14_9 = any(bool(r.get("G14_all_gate_pass", False)) for r in gate_rows)

    # write outputs
    write_csv(out_dir / "v14_8b_scaled_spectra.csv", fieldnames=list(scaled_spectra_rows[0].keys()) if scaled_spectra_rows else ["dim", "family", "mode", "word", "idx", "E_raw", "E_scaled"], rows=scaled_spectra_rows)
    write_csv(out_dir / "v14_8b_transport_maps.csv", fieldnames=list(transport_rows[0].keys()) if transport_rows else ["dim", "family", "mode", "word", "fit_mode", "a", "b", "c"], rows=transport_rows)
    write_csv(out_dir / "v14_8b_active_argument_counts.csv", fieldnames=list(arg_rows_all[0].keys()) if arg_rows_all else ["dim", "family", "mode", "word", "window_a", "window_b"], rows=arg_rows_all)
    write_csv(out_dir / "v14_8b_residue_scores.csv", fieldnames=list(residue_rows_all[0].keys()) if residue_rows_all else ["dim", "family", "mode", "word", "window_a", "window_b"], rows=residue_rows_all)
    write_csv(out_dir / "v14_8b_trace_proxy.csv", fieldnames=list(trace_rows_all[0].keys()) if trace_rows_all else ["dim", "family", "mode", "word", "window_a", "window_b"], rows=trace_rows_all)
    write_csv(out_dir / "v14_8b_hadamard_scores.csv", fieldnames=list(had_rows_all[0].keys()) if had_rows_all else ["dim", "family", "mode", "word", "z"], rows=had_rows_all)
    write_csv(out_dir / "v14_8b_nv_diagnostics.csv", fieldnames=list(nv_rows_all[0].keys()) if nv_rows_all else ["dim", "family", "mode", "word", "kind", "L", "Sigma2"], rows=nv_rows_all)
    write_csv(out_dir / "v14_8b_candidate_ranking.csv", fieldnames=list(ranked[0].keys()) if ranked else ["dim", "family", "mode", "rank", "word", "J_v14_8b"], rows=ranked)
    write_csv(out_dir / "v14_8b_best_candidates.csv", fieldnames=list(best_rows[0].keys()) if best_rows else ["dim", "family", "mode", "best_word", "J_v14_8b"], rows=best_rows)
    write_csv(out_dir / "v14_8b_gate_summary.csv", fieldnames=list(gate_rows[0].keys()) if gate_rows else ["dim", "family", "mode", "G14_all_gate_pass"], rows=gate_rows)
    write_csv(out_dir / "v14_8b_null_comparisons.csv", fieldnames=list(null_rows[0].keys()) if null_rows else ["dim", "family", "mode", "word", "beats_random"], rows=null_rows)

    results = {
        "version": "v14_8b",
        "config": {k: json_sanitize(getattr(args, k)) for k in vars(args).keys()},
        "missing_sources": sorted(set(missing_sources)),
        "warnings": warnings,
        "n_all_gate_pass": int(sum(1 for r in gate_rows if bool(r.get("G14_all_gate_pass", False)))),
        "proceed_to_v14_9": bool(proceed_to_v14_9),
        "analytic_claim": False,
        "comparison": {"v14_8_zero_support_reject_rows": int(zero_support_v148), "v14_8b_zero_support_reject_rows": int(zero_support_v148b), "improves_zero_support_reject_frequency": improves_zero_support},
        "short_interpretation": "Spectral support rescaling layer on top of V14.8 operators; computational evidence only.",
    }
    write_text(out_dir / "v14_8b_results.json", json.dumps(json_sanitize(results), indent=2, ensure_ascii=False) + "\n")

    report_md = render_report_md(
        out_dir=out_dir,
        config=results["config"],
        missing_sources=sorted(set(missing_sources)),
        best_rows=best_rows,
        gate_rows=gate_rows,
        improves_zero_support=improves_zero_support,
    )
    write_text(out_dir / "v14_8b_report.md", report_md)

    tex = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\begin{document}
\section*{V14.8b --- Spectral Support Rescaling for Braid-Graph Laplacian}
\textbf{Computational evidence only; not a proof of RH.}

This run fits a monotone spectral rescaling on top of V14.8 braid-graph Laplacian candidates to increase
active-window support overlap and improve window-based diagnostics (argument/residue/trace) and the Hadamard determinant proxy.
\end{document}
"""
    tex_path = out_dir / "v14_8b_report.tex"
    write_text(tex_path, tex)
    if _find_pdflatex():
        try_pdflatex(tex_path, out_dir, "v14_8b_report.pdf")

    elapsed = time.perf_counter() - t0
    print(f"[V14.8b] done in {format_seconds(elapsed)} out_dir={out_dir}", flush=True)


if __name__ == "__main__":
    main()

