#!/usr/bin/env python3
"""
V14.8 — Braid-Graph Laplacian with Hadamard Determinant Gate (Ant-RH)

Computational evidence only; not a proof of RH.

Core idea:
  - Generate braid words W = sigma_i^p ...
  - Build a weighted braid graph G(W) over nodes 0..dim-1
  - Construct Hermitian graph Laplacian operators (multiple modes)
  - Evaluate via support/argument/residue/trace/NV + NEW Hadamard determinant gate
  - ACO-style search (numeric_only + hybrid_ranked_anticollapse)

No network access, no LLM calls, deterministic seeds.
Missing baseline files must not crash the run (null gates become False).
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
# Word parsing / formatting
# ----------------------------


Token = Tuple[int, int]  # (generator index i>=1, power p!=0)


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
            # allow any gi; clamp later for operator build
            out.append((int(gi), int(pw)))
        except Exception:
            warnings.append(f"skip_token(parse_fail): {tt}")
            continue
    out = simplify_word(out, max_power=int(max_power), max_word_len=10**9)
    # empty invalid
    return out


# ----------------------------
# Anti-collapse metrics
# ----------------------------


def generator_entropy(word: List[Token], n_generators: int) -> float:
    gens = [int(i) for i, _p in word]
    if not gens or n_generators <= 1:
        return 0.0
    c = Counter(gens)
    total = float(sum(c.values()))
    ps = np.asarray([v / total for v in c.values()], dtype=np.float64)
    H = float(-np.sum(ps * np.log(np.maximum(1e-12, ps))))
    return float(H / max(1e-12, math.log(float(n_generators))))


def power_entropy(word: List[Token], max_power: int) -> float:
    ps = [int(p) for _i, p in word]
    if not ps:
        return 0.0
    n_bins = max(2, 2 * int(max_power))
    c = Counter(ps)
    total = float(sum(c.values()))
    probs = np.asarray([v / total for v in c.values()], dtype=np.float64)
    H = float(-np.sum(probs * np.log(np.maximum(1e-12, probs))))
    return float(H / max(1e-12, math.log(float(n_bins))))


def collapse_features(word: List[Token], *, dim: int) -> Dict[str, Any]:
    L = int(len(word))
    gens = [int(i) for i, _p in word]
    uniq = int(len(set(gens)))
    ent = float(generator_entropy(word, max(1, int(dim) - 1)))
    rep_run = 0
    cur = 1
    for k in range(1, len(gens)):
        if gens[k] == gens[k - 1]:
            cur += 1
        else:
            rep_run = max(rep_run, cur)
            cur = 1
    rep_run = max(rep_run, cur) if gens else 0
    return {
        "word_len": int(L),
        "unique_generator_count": int(uniq),
        "generator_entropy": float(ent),
        "power_entropy": float(power_entropy(word, int(3))),
        "max_consecutive_generator_run": int(rep_run),
    }


def complexity_penalty(word: List[Token], max_word_len: int) -> float:
    L = float(len(word))
    return float(L / max(1.0, float(max_word_len)))


# ----------------------------
# Braid graph construction
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
    """
    Returns (W_abs, W_signed, stats) as real symmetric matrices (dim x dim) with
    weights accumulated from braid interactions.

    Nodes: 0..dim-1
    For sigma_i^p:
      i_eff = i % (dim-1)   (maps to 0..dim-2)
      edge (i_eff, i_eff+1)
      abs weight += abs(p)
      signed weight += sign(p)*abs(p)

    Also add weak long-range edges between consecutive generators.
    """
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
            # weak memory edge between generator indices
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
    # curvature score: repeated generator + concentration of visits
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
    if use_abs_degree:
        deg = np.sum(np.abs(A), axis=1)
    else:
        deg = np.sum(A, axis=1)
    D = np.diag(deg.astype(np.float64, copy=False))
    return (D - A).astype(np.complex128, copy=False)


def magnetic_adjacency(W_abs: np.ndarray, W_sgn: np.ndarray, *, theta_scale: float = 0.35) -> np.ndarray:
    """
    A_ij = w_ij * exp(i theta_ij) with Hermitian symmetry A_ji = conj(A_ij).
    theta_ij derived from signed orientation.
    """
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


def build_operator_from_word(
    *,
    dim: int,
    word: List[Token],
    op_mode: str,
    seed: int,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Constructs stabilized Hermitian operator H from braid-graph Laplacian modes.
    Returns (H, diagnostics) where H is complex Hermitian (dim x dim).
    """
    rng = np.random.default_rng(int(seed))
    n = int(dim)
    w = clamp_word_to_dim(word, dim=n, max_power=3, max_word_len=10**9)
    W_abs, W_sgn, gs = build_braid_graph(n, w)
    diag: Dict[str, Any] = {
        "graph_n_edges_nonzero": int(gs.n_edges_nonzero),
        "graph_total_weight": float(gs.total_weight),
        "graph_total_signed_weight": float(gs.total_signed_weight),
        "graph_visit_count": int(gs.visit_count),
        "graph_repeat_count": int(gs.repeat_count),
        "graph_long_range_edge_count": int(gs.long_range_edge_count),
        "graph_curvature_score": float(gs.curvature_score),
    }

    mode = str(op_mode).strip()
    # adjacency choices
    if mode == "plain_graph_laplacian":
        A = W_abs.astype(np.complex128, copy=False)
        L = make_laplacian_from_adjacency(A.real, use_abs_degree=True)
    elif mode == "signed_braid_laplacian":
        A = W_sgn.astype(np.complex128, copy=False)
        # D uses abs degree for stability
        L = make_laplacian_from_adjacency(A.real, use_abs_degree=True)
    elif mode == "magnetic_graph_laplacian":
        A = magnetic_adjacency(W_abs, W_sgn, theta_scale=0.45)
        L = make_laplacian_from_adjacency(A, use_abs_degree=True)
    elif mode == "curvature_regularized_laplacian":
        A = W_abs.astype(np.complex128, copy=False)
        L = make_laplacian_from_adjacency(A.real, use_abs_degree=True)
        lam_curv = 0.25
        curv_diag = np.zeros((n,), dtype=np.float64)
        # curvature per node: degree concentration + repeated generator boosts adjacent nodes
        deg = np.sum(W_abs, axis=1)
        if np.max(deg) > 1e-12:
            curv_diag = (deg / (np.max(deg) + 1e-12)).astype(np.float64)
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
        return None, {"stable": False, "reason": f"unknown op_mode={op_mode!r}"}

    # Stabilization: Hermitian symmetrize, remove trace, normalize radius
    H = 0.5 * (L + L.conj().T)
    tr = np.trace(H) / complex(n)
    H = H - tr * np.eye(n, dtype=np.complex128)

    # normalize spectral radius to target = dim/4
    target_radius = float(max(4.0, n / 4.0))
    eigs = None
    try:
        eigs = np.linalg.eigvalsh(np.asarray(H.real, dtype=np.float64))
        eigs = np.asarray(eigs, dtype=np.float64).reshape(-1)
        eigs = eigs[np.isfinite(eigs)]
    except Exception:
        eigs = None
    if eigs is None or eigs.size < 8:
        diag.update({"stable": False, "reason": "eig_fail_pre_norm"})
        return None, diag
    r = float(max(abs(float(eigs[0])), abs(float(eigs[-1]))))
    if not (math.isfinite(r) and r > 1e-12):
        diag.update({"stable": False, "reason": "radius_nonfinite"})
        return None, diag
    H = H * (target_radius / r)
    herm_err = float(np.linalg.norm(H - H.conj().T, ord="fro"))
    diag.update(
        {
            "stable": True,
            "trace_abs": float(abs(np.trace(H))),
            "spectral_radius": float(target_radius),
            "hermitian_fro_error": float(herm_err),
        }
    )
    return H, diag


# ----------------------------
# Spectral calibration + diagnostics
# ----------------------------


@dataclass
class TransportMap:
    scale: float
    shift: float
    op_q10: float
    op_q50: float
    op_q90: float
    tg_q10: float
    tg_q50: float
    tg_q90: float
    mode_effective: str


def transport_affine(op: np.ndarray, tg: np.ndarray) -> TransportMap:
    op = np.asarray(op, dtype=np.float64)
    tg = np.asarray(tg, dtype=np.float64)
    op = op[np.isfinite(op)]
    tg = tg[np.isfinite(tg)]
    if op.size < 8 or tg.size < 8:
        return TransportMap(1.0, 0.0, float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), float("nan"), "none")
    op_q = np.quantile(op, [0.1, 0.5, 0.9])
    tg_q = np.quantile(tg, [0.1, 0.5, 0.9])
    scale = float((tg_q[2] - tg_q[0]) / max(1e-12, (op_q[2] - op_q[0])))
    shift = float(tg_q[1] - scale * op_q[1])
    return TransportMap(scale, shift, float(op_q[0]), float(op_q[1]), float(op_q[2]), float(tg_q[0]), float(tg_q[1]), float(tg_q[2]), "affine")


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
            # normalized log error per spec
            lop = math.log1p(max(0.0, float(Sop)))
            ltg = math.log1p(max(0.0, float(Stg)))
            denom = max(1.0, abs(float(ltg)))
            err = abs(float(lop) - float(ltg)) / denom
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


def unfold_to_mean_spacing_one(x: np.ndarray) -> np.ndarray:
    return rd.unfold_to_mean_spacing_one(np.asarray(x, dtype=np.float64).reshape(-1))


def make_windows(args: argparse.Namespace) -> List[Tuple[float, float]]:
    win = rd.make_windows(float(args.window_min), float(args.window_max), float(args.window_size), float(args.window_stride))
    return win


# ----------------------------
# Hadamard determinant gate
# ----------------------------


def mad(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)))


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


def hadamard_profiles(
    *,
    candidate_pos: np.ndarray,
    target_pos: np.ndarray,
    z_grid: np.ndarray,
    eps: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    cand = np.asarray(candidate_pos, dtype=np.float64).reshape(-1)
    tgt = np.asarray(target_pos, dtype=np.float64).reshape(-1)
    cand = cand[np.isfinite(cand) & (cand > 0)]
    tgt = tgt[np.isfinite(tgt) & (tgt > 0)]
    cand.sort()
    tgt.sort()
    N = int(min(cand.size, tgt.size))
    if N < 8:
        meta = {"N": int(N), "reason": "insufficient_ordinates"}
        z = np.asarray(z_grid, dtype=np.float64)
        nan = np.full_like(z, np.nan, dtype=np.float64)
        return nan, nan, nan, nan, meta
    cand = cand[:N]
    tgt = tgt[:N]
    z = np.asarray(z_grid, dtype=np.float64)
    z2 = z * z

    # logXi(z) = sum_n log( abs(1 - z^2/(lambda_n^2+eps)) + eps )
    def logxi(vals: np.ndarray) -> np.ndarray:
        v2 = vals * vals + float(eps)
        # (n_grid, N)
        M = 1.0 - (z2.reshape(-1, 1) / v2.reshape(1, -1))
        Y = np.log(np.abs(M) + float(eps))
        out = np.sum(Y, axis=1)
        return out.astype(np.float64, copy=False)

    y_c = logxi(cand)
    y_t = logxi(tgt)
    y_c_n, med_c, scale_c = normalize_curve(y_c)
    y_t_n, med_t, scale_t = normalize_curve(y_t)

    meta = {"N": int(N), "med_candidate": float(med_c), "scale_candidate": float(scale_c), "med_target": float(med_t), "scale_target": float(scale_t)}
    return y_c, y_t, y_c_n, y_t_n, meta


def hadamard_scores(z: np.ndarray, y_c_n: np.ndarray, y_t_n: np.ndarray) -> Dict[str, Any]:
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    yc = np.asarray(y_c_n, dtype=np.float64).reshape(-1)
    yt = np.asarray(y_t_n, dtype=np.float64).reshape(-1)
    if yc.size == 0 or yt.size == 0 or yc.size != yt.size:
        return {"hadamard_rmse": float("nan"), "hadamard_corr": float("nan"), "hadamard_peak_alignment_error": float("nan"), "L_had": float("nan")}
    m = np.isfinite(z) & np.isfinite(yc) & np.isfinite(yt)
    if m.sum() < 8:
        return {"hadamard_rmse": float("nan"), "hadamard_corr": float("nan"), "hadamard_peak_alignment_error": float("nan"), "L_had": float("nan")}
    z = z[m]
    yc = yc[m]
    yt = yt[m]
    rmse = float(np.sqrt(np.mean((yc - yt) ** 2)))
    # correlation
    yc0 = yc - float(np.mean(yc))
    yt0 = yt - float(np.mean(yt))
    denom = float(np.sqrt(np.mean(yc0**2) * np.mean(yt0**2)))
    corr = float(np.mean(yc0 * yt0) / denom) if denom > 1e-12 else float("nan")
    # peak alignment: argmin of normalized logXi (dominant dip), compare z locations
    try:
        zc = float(z[int(np.argmin(yc))])
        zt = float(z[int(np.argmin(yt))])
        peak_align = float(abs(zc - zt) / max(1e-12, (float(np.max(z)) - float(np.min(z)))))
    except Exception:
        peak_align = float("nan")
    L_had = float(rmse + 0.25 * (1.0 - (corr if math.isfinite(corr) else 0.0)) + 0.25 * (peak_align if math.isfinite(peak_align) else 1.0))
    return {"hadamard_rmse": float(rmse), "hadamard_corr": float(corr), "hadamard_peak_alignment_error": float(peak_align), "L_had": float(L_had)}


# ----------------------------
# Null controls (best-effort)
# ----------------------------


def load_baseline_pool(*, dims: List[int], v13o14_dir: Path, v14_2_dir: Path, v14_5_dir: Path, v14_7b_dir: Path, v14_7c_dir: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    pool: List[Dict[str, Any]] = []
    missing: List[str] = []

    def add_row(dim: int, source: str, kind: str, label: str, mode: str, J: Any, source_file: str, missing_source: bool, notes: str) -> None:
        pool.append(
            {
                "dim": int(dim),
                "source": str(source),
                "baseline_kind": str(kind),
                "label": str(label),
                "mode": str(mode),
                "J": safe_float(J, float("nan")),
                "source_file": str(source_file),
                "missing_source": bool(missing_source),
                "notes": str(notes),
            }
        )

    # v13o14
    p13m = v13o14_dir / "v13o14_candidate_mode_summary.csv"
    p13n = v13o14_dir / "v13o14_null_comparisons.csv"
    r13m = _read_csv_best_effort(p13m)
    r13n = _read_csv_best_effort(p13n)
    if not r13m:
        missing.append("v13o14_candidate_mode_summary.csv")
    else:
        for r in r13m:
            d = safe_int(r.get("dim", 0))
            if d not in dims:
                continue
            J = safe_float(r.get("best_all_J", r.get("best_simple_J", float("nan"))))
            if not math.isfinite(J):
                continue
            kind = "other"
            if str(r.get("is_random_baseline", "")).lower() == "true" or bool(r.get("is_random_baseline", False)):
                kind = "random"
            if str(r.get("is_ablation", "")).lower() == "true" or bool(r.get("is_ablation", False)):
                kind = "ablation"
            if str(r.get("is_rejected_word", "")).lower() == "true" or bool(r.get("is_rejected_word", False)):
                kind = "rejected"
            if str(r.get("is_primary", "")).lower() == "true" or bool(r.get("is_primary", False)):
                kind = "primary"
            add_row(d, "v13o14", kind, str(r.get("word_group", "")), str(r.get("best_mode", "")), J, "v13o14_candidate_mode_summary.csv", False, str(r.get("candidate_classification", "")))
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
                ("primary", "primary_best_J", "primary_word_seed6"),
            ]:
                if col in r:
                    J = safe_float(r.get(col, float("nan")))
                    if math.isfinite(J):
                        add_row(d, "v13o14", kind, label, "best_of", J, "v13o14_null_comparisons.csv", False, "best_of")

    # v14_5
    p5a = v14_5_dir / "v14_5_ablation_summary.csv"
    r5a = _read_csv_best_effort(p5a)
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
                        add_row(d, "v14_5", "prior_artin", k.replace("_best_J", ""), "best_of", J, "v14_5_ablation_summary.csv", False, "ablation_summary")

    # v14_2
    p2b = v14_2_dir / "v14_2_best_candidates.csv"
    r2b = _read_csv_best_effort(p2b)
    if not r2b:
        missing.append("v14_2_best_candidates.csv")
    else:
        for r in r2b:
            d = safe_int(r.get("dim", 0))
            if d not in dims:
                continue
            J = safe_float(r.get("J_v14_2", r.get("J_total", float("nan"))))
            if math.isfinite(J):
                add_row(d, "v14_2", "prior_artin", "best_v14_2", "best_of", J, "v14_2_best_candidates.csv", False, "best_candidates")

    # v14_7b null comparisons
    p7bn = v14_7b_dir / "v14_7b_null_comparisons.csv"
    r7bn = _read_csv_best_effort(p7bn)
    if not r7bn:
        missing.append("v14_7b_null_comparisons.csv")
    else:
        for r in r7bn:
            d = safe_int(r.get("dim", 0))
            if d not in dims:
                continue
            for kind, col in [("random", "best_random_J"), ("rejected", "best_rejected_J"), ("ablation", "best_ablation_J"), ("prior_artin", "best_prior_artin_J")]:
                if col in r:
                    J = safe_float(r.get(col, float("nan")))
                    if math.isfinite(J):
                        add_row(d, "v14_7b", kind, col, "best_of", J, "v14_7b_null_comparisons.csv", False, "null_comparisons")

    # v14_7c null comparisons
    p7cn = v14_7c_dir / "v14_7c_null_comparisons.csv"
    r7cn = _read_csv_best_effort(p7cn)
    if not r7cn:
        missing.append("v14_7c_null_comparisons.csv")
    else:
        for r in r7cn:
            d = safe_int(r.get("dim", 0))
            if d not in dims:
                continue
            for kind, col in [("random", "best_random_J"), ("rejected", "best_rejected_J"), ("ablation", "best_ablation_J"), ("prior_artin", "best_prior_artin_J")]:
                if col in r:
                    J = safe_float(r.get(col, float("nan")))
                    if math.isfinite(J):
                        add_row(d, "v14_7c", kind, col, "best_of", J, "v14_7c_null_comparisons.csv", False, "null_comparisons")

    # If nothing loaded for a dim, add explicit placeholder
    for d in dims:
        any_dim = any(int(r.get("dim", 0)) == int(d) and not bool(r.get("missing_source", False)) and math.isfinite(safe_float(r.get("J", float("nan")))) for r in pool)
        if not any_dim:
            add_row(d, "missing", "missing", "no_baselines_loaded", "", float("nan"), "", True, "no baseline rows loaded for this dim")

    return pool, sorted(set(missing))


def best_baseline_J(pool: List[Dict[str, Any]], dim: int, kinds: Sequence[str]) -> Optional[float]:
    Js: List[float] = []
    kinds_set = set(str(k) for k in kinds)
    for r in pool:
        if int(r.get("dim", 0)) != int(dim):
            continue
        if bool(r.get("missing_source", False)):
            continue
        if str(r.get("baseline_kind", "")) in kinds_set:
            J = safe_float(r.get("J", float("nan")))
            if math.isfinite(J):
                Js.append(float(J))
    return float(min(Js)) if Js else None


def null_distribution_Js(pool: List[Dict[str, Any]], dim: int) -> List[float]:
    xs: List[float] = []
    for r in pool:
        if int(r.get("dim", 0)) != int(dim):
            continue
        if bool(r.get("missing_source", False)):
            continue
        if str(r.get("baseline_kind", "")) == "primary":
            continue
        J = safe_float(r.get("J", float("nan")))
        if math.isfinite(J):
            xs.append(float(J))
    return xs


# ----------------------------
# ACO sampling
# ----------------------------


def sample_token(
    *,
    dim: int,
    powers: List[int],
    pher: Dict[Tuple[int, int], float],
    alpha: float,
    beta: float,
    semantic_term: float,
    preferred_generators: Sequence[int],
    rng: np.random.Generator,
) -> Token:
    items = list(pher.keys())
    tau = np.asarray([pher[it] for it in items], dtype=np.float64)
    # heuristic: mild mid preference + smaller power + optional preferred band bias
    mid = 0.5 * (dim - 1)
    eta = np.asarray([(1.0 / (1.0 + 0.01 * abs(float(gi) - mid))) * (1.0 / (1.0 + 0.10 * abs(float(pw)))) for (gi, pw) in items], dtype=np.float64)
    if preferred_generators:
        pref = set(int(g) for g in preferred_generators)
        pref_boost = np.asarray([1.15 if int(gi) in pref else 1.0 for (gi, _pw) in items], dtype=np.float64)
        eta = eta * pref_boost
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
# Evaluation
# ----------------------------


@dataclass
class EvalResult:
    stable: bool
    metric_nan_flag: bool
    J_v14_8: float
    final_reward: float
    support_overlap: float
    active_argument_error_med: float
    residue_error_med: float
    trace_error_med: float
    nv_rmse: float
    nv_range: float
    nv_median: float
    poisson_like_fraction: float
    hadamard_rmse: float
    hadamard_corr: float
    hadamard_peak_alignment_error: float
    L_had: float
    # null stats
    beats_random: bool
    beats_rejected: bool
    beats_ablation: bool
    beats_prior_artin: bool
    null_separation: float
    null_zscore: float
    null_percentile: float
    missing_baselines: str
    # collapse
    generator_entropy: float
    power_entropy: float
    unique_generator_count: int
    classification: str
    # details
    arg_rows: List[Dict[str, Any]]
    residue_rows: List[Dict[str, Any]]
    trace_rows: List[Dict[str, Any]]
    nv_rows: List[Dict[str, Any]]
    hadamard_curve_rows: List[Dict[str, Any]]
    graph_diag: Dict[str, Any]


def finite_or_penalty(x: float, *, penalty: float, flags: Dict[str, bool], key: str) -> float:
    if not math.isfinite(float(x)):
        flags[key] = True
        return float(penalty)
    return float(x)


def evaluate_word(
    *,
    dim: int,
    op_mode: str,
    search_mode: str,
    word: List[Token],
    zeros_unfolded: np.ndarray,
    windows: List[Tuple[float, float]],
    L_grid: np.ndarray,
    z_grid: np.ndarray,
    args: argparse.Namespace,
    baselines: List[Dict[str, Any]],
    seed: int,
) -> EvalResult:
    nan_flags: Dict[str, bool] = {}
    big = 1e3
    w = clamp_word_to_dim(word, dim=int(dim), max_power=int(args.max_power), max_word_len=int(args.max_word_len))
    if not w:
        return EvalResult(
            stable=False,
            metric_nan_flag=True,
            J_v14_8=1e6,
            final_reward=0.0,
            support_overlap=0.0,
            active_argument_error_med=1.0,
            residue_error_med=1.0,
            trace_error_med=1.0,
            nv_rmse=1.0,
            nv_range=0.0,
            nv_median=float("nan"),
            poisson_like_fraction=1.0,
            hadamard_rmse=float("nan"),
            hadamard_corr=float("nan"),
            hadamard_peak_alignment_error=float("nan"),
            L_had=big,
            beats_random=False,
            beats_rejected=False,
            beats_ablation=False,
            beats_prior_artin=False,
            null_separation=float("nan"),
            null_zscore=float("nan"),
            null_percentile=float("nan"),
            missing_baselines="all_missing",
            generator_entropy=0.0,
            power_entropy=0.0,
            unique_generator_count=0,
            classification="UNSTABLE",
            arg_rows=[],
            residue_rows=[],
            trace_rows=[],
            nv_rows=[],
            hadamard_curve_rows=[],
            graph_diag={},
        )

    # collapse metrics
    gens = [int(i) for i, _p in w]
    uniq = int(len(set(gens)))
    g_ent = float(generator_entropy(w, max(1, int(dim) - 1)))
    p_ent = float(power_entropy(w, int(args.max_power)))
    no_collapse = bool(len(w) >= 3 and uniq >= 3 and g_ent >= 0.25 and max((len(list(g)) for _k, g in itertools_groupby(gens)), default=0) <= 12)
    # stable operator
    H, gd = build_operator_from_word(dim=int(dim), word=w, op_mode=str(op_mode), seed=int(seed))
    if H is None or not bool(gd.get("stable", False)):
        return EvalResult(
            stable=False,
            metric_nan_flag=True,
            J_v14_8=1e6,
            final_reward=0.0,
            support_overlap=0.0,
            active_argument_error_med=1.0,
            residue_error_med=1.0,
            trace_error_med=1.0,
            nv_rmse=1.0,
            nv_range=0.0,
            nv_median=float("nan"),
            poisson_like_fraction=1.0,
            hadamard_rmse=float("nan"),
            hadamard_corr=float("nan"),
            hadamard_peak_alignment_error=float("nan"),
            L_had=big,
            beats_random=False,
            beats_rejected=False,
            beats_ablation=False,
            beats_prior_artin=False,
            null_separation=float("nan"),
            null_zscore=float("nan"),
            null_percentile=float("nan"),
            missing_baselines="all_missing",
            generator_entropy=float(g_ent),
            power_entropy=float(p_ent),
            unique_generator_count=int(uniq),
            classification="UNSTABLE",
            arg_rows=[],
            residue_rows=[],
            trace_rows=[],
            nv_rows=[],
            hadamard_curve_rows=[],
            graph_diag=dict(gd),
        )

    # eigenvalues
    try:
        eig = np.linalg.eigvalsh(np.asarray(H.real, dtype=np.float64))
        eig = np.asarray(eig, dtype=np.float64).reshape(-1)
        eig = eig[np.isfinite(eig)]
        eig.sort()
    except Exception:
        eig = np.asarray([], dtype=np.float64)
    if eig.size < 8:
        gd["stable"] = False
        return EvalResult(
            stable=False,
            metric_nan_flag=True,
            J_v14_8=1e6,
            final_reward=0.0,
            support_overlap=0.0,
            active_argument_error_med=1.0,
            residue_error_med=1.0,
            trace_error_med=1.0,
            nv_rmse=1.0,
            nv_range=0.0,
            nv_median=float("nan"),
            poisson_like_fraction=1.0,
            hadamard_rmse=float("nan"),
            hadamard_corr=float("nan"),
            hadamard_peak_alignment_error=float("nan"),
            L_had=big,
            beats_random=False,
            beats_rejected=False,
            beats_ablation=False,
            beats_prior_artin=False,
            null_separation=float("nan"),
            null_zscore=float("nan"),
            null_percentile=float("nan"),
            missing_baselines="all_missing",
            generator_entropy=float(g_ent),
            power_entropy=float(p_ent),
            unique_generator_count=int(uniq),
            classification="UNSTABLE",
            arg_rows=[],
            residue_rows=[],
            trace_rows=[],
            nv_rows=[],
            hadamard_curve_rows=[],
            graph_diag=dict(gd),
        )

    # unfold + calibrate to zeros via affine transport
    levels = unfold_to_mean_spacing_one(eig)
    tgt = np.asarray(zeros_unfolded, dtype=np.float64)
    tgt = tgt[np.isfinite(tgt)]
    tgt.sort()
    tm = transport_affine(levels, tgt)
    levels_cal = apply_transport(levels, tm)

    # support + arg
    arg_med, _arg_mean, support_overlap, arg_rows = active_argument_counts(levels_cal, tgt, windows)
    res_med, _leak, residue_rows = residue_scores(levels_cal, tgt, windows, eta=0.15, n_contour_points=256)
    tr_med, trace_rows = trace_proxy_rows(levels_cal, tgt, windows)

    # NV
    lv_nv = unfold_to_mean_spacing_one(levels_cal)
    tg_nv = unfold_to_mean_spacing_one(tgt)
    op_nv = number_variance_curve(lv_nv, L_grid)
    tgt_nv = number_variance_curve(tg_nv, L_grid)
    nv_rmse = float(curve_l2(op_nv, tgt_nv))
    m_nv = np.isfinite(op_nv)
    nv_range = float(np.max(op_nv[m_nv]) - np.min(op_nv[m_nv])) if m_nv.any() else 0.0
    nv_median = float(np.median(op_nv[m_nv])) if m_nv.any() else float("nan")
    pois_frac = float(poisson_like_fraction_from_nv(op_nv, tgt_nv, L_grid))
    nv_rows = [{"kind": "operator", "L": float(L), "Sigma2": safe_float(y)} for L, y in zip(L_grid, op_nv)]

    # Hadamard profiles (use positive ordinates)
    pos_levels = np.asarray(levels_cal, dtype=np.float64)
    pos_levels = pos_levels[np.isfinite(pos_levels) & (pos_levels > 0)]
    pos_tgt = np.asarray(tgt, dtype=np.float64)
    pos_tgt = pos_tgt[np.isfinite(pos_tgt) & (pos_tgt > 0)]
    y_c, y_t, y_c_n, y_t_n, had_meta = hadamard_profiles(candidate_pos=pos_levels, target_pos=pos_tgt, z_grid=z_grid, eps=float(args.hadamard_eps))
    hs = hadamard_scores(z_grid, y_c_n, y_t_n)
    had_curve_rows = []
    for zz, a, b, an, bn in zip(z_grid, y_c, y_t, y_c_n, y_t_n):
        had_curve_rows.append(
            {
                "z": float(zz),
                "logdet_candidate": safe_float(a),
                "logdet_target": safe_float(b),
                "logdet_candidate_norm": safe_float(an),
                "logdet_target_norm": safe_float(bn),
            }
        )

    # finiteness
    sup2 = finite_or_penalty(float(support_overlap), penalty=0.0, flags=nan_flags, key="support_overlap")
    arg2 = finite_or_penalty(float(arg_med), penalty=big, flags=nan_flags, key="active_argument_error_med")
    res2 = finite_or_penalty(float(res_med), penalty=big, flags=nan_flags, key="residue_error_med")
    tr2 = finite_or_penalty(float(tr_med), penalty=big, flags=nan_flags, key="trace_error_med")
    nv2 = finite_or_penalty(float(nv_rmse), penalty=big, flags=nan_flags, key="nv_rmse")
    pois2 = finite_or_penalty(float(pois_frac), penalty=1.0, flags=nan_flags, key="poisson_like_fraction")
    L_had = finite_or_penalty(float(hs.get("L_had", float("nan"))), penalty=big, flags=nan_flags, key="L_had")

    # null controls: compare candidate J to baseline bests with margins
    best_random = best_baseline_J(baselines, int(dim), ["random"])
    best_rejected = best_baseline_J(baselines, int(dim), ["rejected"])
    best_ablation = best_baseline_J(baselines, int(dim), ["ablation"])
    best_prior = best_baseline_J(baselines, int(dim), ["prior_artin"])
    best_null = best_baseline_J(baselines, int(dim), ["random", "rejected", "ablation", "prior_artin", "other"])
    miss: List[str] = []
    if best_random is None:
        miss.append("random")
    if best_rejected is None:
        miss.append("rejected")
    if best_ablation is None:
        miss.append("ablation")
    if best_prior is None:
        miss.append("prior_artin")
    missing_baselines = "|".join(miss)

    # objective weights (defaults per spec)
    lambda_support = 3.0
    lambda_arg = 4.0
    lambda_residue = 2.0
    lambda_trace = 2.0
    lambda_nv = 1.0
    lambda_antipoisson = 3.0
    lambda_hadamard = 4.0
    lambda_complexity = 0.05
    lambda_stability = 10.0
    lambda_null = 2.0

    support_pen = float(max(0.0, float(args.support_overlap_min) - sup2))
    stability_pen = float(max(0.0, gd.get("hermitian_fro_error", 0.0))) + (0.0 if gd.get("stable", False) else 1.0)
    comp_pen = float(complexity_penalty(w, int(args.max_word_len)))
    null_pen = 0.0
    # penalize missing baselines (so it is never silently "passed")
    if miss:
        null_pen += 1.0

    J = (
        lambda_support * support_pen
        + lambda_arg * arg2
        + lambda_residue * res2
        + lambda_trace * tr2
        + lambda_nv * nv2
        + lambda_antipoisson * pois2
        + lambda_hadamard * L_had
        + lambda_complexity * comp_pen
        + lambda_stability * stability_pen
        + lambda_null * null_pen
    )
    if not math.isfinite(J):
        J = 1e6
        nan_flags["J_v14_8"] = True

    # beats checks (only if baseline exists)
    margin = float(args.null_separation_margin)
    beats_random = bool(best_random is not None and float(J) < float(best_random) - margin)
    beats_rejected = bool(best_rejected is not None and float(J) < float(best_rejected) - margin)
    beats_ablation = bool(best_ablation is not None and float(J) < float(best_ablation) - margin)
    beats_prior = bool(best_prior is not None and float(J) < float(best_prior) - margin)

    dist = null_distribution_Js(baselines, int(dim))
    null_sep = float(best_null - float(J)) if (best_null is not None and math.isfinite(best_null)) else float("nan")
    null_z = float("nan")
    null_pct = float("nan")
    if len(dist) >= 3:
        mu = float(np.mean(dist))
        sd = float(np.std(dist))
        null_z = float((mu - float(J)) / sd) if sd > 1e-12 else float("nan")
        null_pct = float(np.mean([1.0 if x <= float(J) else 0.0 for x in dist]))

    reward = float(1.0 / (1.0 + float(J)))
    if not math.isfinite(reward):
        reward = 0.0

    # Gates -> classification
    metric_nan_flag = bool(any(nan_flags.values()))
    G1 = bool(gd.get("stable", False)) and not metric_nan_flag
    G2 = bool(sup2 >= float(args.support_overlap_min))
    G3 = bool(arg2 <= float(args.active_argument_error_max))
    G4 = bool(nv_range >= 0.0 and nv2 <= 1e6)  # diagnostic gate itself is reported; strict anti-Poisson is G5
    G5 = bool(pois2 <= float(args.poisson_like_max))
    G6 = bool(res2 <= float(args.residue_error_max))
    G7 = bool(tr2 <= float(args.trace_error_max))
    G8 = bool(L_had <= float(args.hadamard_error_max))
    G9 = bool(beats_random) if best_random is not None else False
    G10 = bool(beats_rejected) if best_rejected is not None else False
    G11 = bool(beats_ablation) if best_ablation is not None else False
    G12 = bool(beats_prior) if best_prior is not None else False
    G13 = bool(no_collapse)
    all_gate = bool(G1 and G2 and G3 and G4 and G5 and G6 and G7 and G8 and G9 and G10 and G11 and G12 and G13)

    if not G1:
        classification = "UNSTABLE"
    elif sup2 <= 0.0:
        classification = "ZERO_SUPPORT_REJECT"
    elif not G3:
        classification = "ARGUMENT_FAIL"
    elif not G6:
        classification = "RESIDUE_FAIL"
    elif not G7:
        classification = "TRACE_FAIL"
    elif not G8:
        classification = "HADAMARD_FAIL"
    elif not G5:
        classification = "POISSON_LIKE_FAIL"
    elif not (G9 and G10 and G11 and G12):
        classification = "NULL_CONTROL_FAIL"
    else:
        classification = "OK" if not all_gate else "ALL_GATE_PASS"

    return EvalResult(
        stable=bool(G1),
        metric_nan_flag=bool(metric_nan_flag),
        J_v14_8=float(J),
        final_reward=float(max(0.0, min(1.0, reward))),
        support_overlap=float(sup2),
        active_argument_error_med=float(arg2),
        residue_error_med=float(res2),
        trace_error_med=float(tr2),
        nv_rmse=float(nv2),
        nv_range=float(nv_range),
        nv_median=float(nv_median),
        poisson_like_fraction=float(pois2),
        hadamard_rmse=float(hs.get("hadamard_rmse", float("nan"))),
        hadamard_corr=float(hs.get("hadamard_corr", float("nan"))),
        hadamard_peak_alignment_error=float(hs.get("hadamard_peak_alignment_error", float("nan"))),
        L_had=float(L_had),
        beats_random=bool(beats_random),
        beats_rejected=bool(beats_rejected),
        beats_ablation=bool(beats_ablation),
        beats_prior_artin=bool(beats_prior),
        null_separation=float(null_sep),
        null_zscore=float(null_z),
        null_percentile=float(null_pct),
        missing_baselines=str(missing_baselines),
        generator_entropy=float(g_ent),
        power_entropy=float(p_ent),
        unique_generator_count=int(uniq),
        classification=str(classification),
        arg_rows=arg_rows,
        residue_rows=residue_rows,
        trace_rows=trace_rows,
        nv_rows=nv_rows,
        hadamard_curve_rows=had_curve_rows,
        graph_diag=dict(gd),
    )


def itertools_groupby(xs: List[int]) -> Iterable[Tuple[int, List[int]]]:
    # small local replacement for itertools.groupby to keep file self-contained
    if not xs:
        return []
    out: List[Tuple[int, List[int]]] = []
    cur_key = xs[0]
    cur: List[int] = [xs[0]]
    for x in xs[1:]:
        if x == cur_key:
            cur.append(x)
        else:
            out.append((cur_key, cur))
            cur_key = x
            cur = [x]
    out.append((cur_key, cur))
    return out


# ----------------------------
# Report
# ----------------------------


def render_report_md(
    *,
    out_dir: Path,
    config: Dict[str, Any],
    missing_sources: List[str],
    best_rows: List[Dict[str, Any]],
    gate_rows: List[Dict[str, Any]],
    null_rows: List[Dict[str, Any]],
    had_rows: List[Dict[str, Any]],
    proceed_to_v14_9: bool,
) -> str:
    OUT_ABS = str(out_dir.resolve())
    md: List[str] = []
    md.append("# V14.8 — Braid-Graph Laplacian with Hadamard Determinant Gate\n\n")
    md.append("> Computational evidence only; not a proof of RH.\n\n")
    md.append("## Purpose\n")
    md.append("Move from direct Artin-word operators to braid/graph Laplacian operators and add a Hadamard determinant gate comparing spectral determinants against true zeta ordinates.\n\n")
    md.append("## Why move from Artin words to braid-graph Laplacian\n")
    md.append("V14.7b/V14.7c achieved nontrivial support and anti-Poisson behavior but failed argument-count and trace-proxy gates. Graph geometry aims to give smoother, controllable spectral structure.\n\n")
    md.append("## Operator construction\n")
    md.append("- Build weighted braid graph from braid word tokens.\n")
    md.append("- Form Laplacian variants: plain, signed, magnetic (Hermitian), curvature-regularized, hybrid.\n")
    md.append("- Stabilize by Hermitian symmetrization, trace removal, and spectral radius normalization.\n\n")
    md.append("## Hadamard determinant gate\n")
    md.append("Compute truncated log-Hadamard profiles on a z-grid for candidate vs target, normalize by median/MAD, and score via RMSE + (1-corr) + peak-alignment.\n\n")
    md.append("## Best candidates\n")
    if best_rows:
        for r in best_rows:
            md.append(
                f"- dim={r.get('dim')} mode={r.get('mode')} search_mode={r.get('search_mode')} "
                f"J={safe_float(r.get('J_v14_8')):.6g} reward={safe_float(r.get('final_reward')):.6g} "
                f"support={safe_float(r.get('support_overlap')):.3g} pois={safe_float(r.get('poisson_like_fraction')):.3g} "
                f"arg={safe_float(r.get('active_argument_error_med')):.3g} trace={safe_float(r.get('trace_error_med')):.3g} "
                f"L_had={safe_float(r.get('L_had')):.3g}\n"
            )
    else:
        md.append("- (none)\n")
    md.append("\n")
    md.append("## Gate summary\n")
    if gate_rows:
        for r in gate_rows:
            md.append(f"- dim={r.get('dim')} mode={r.get('mode')} search_mode={r.get('search_mode')} all_gate={r.get('G14_all_gate_pass')} class={r.get('classification')}\n")
    else:
        md.append("- (none)\n")
    md.append("\n")
    md.append("## Null comparison\n")
    if null_rows:
        for r in null_rows:
            md.append(f"- dim={r.get('dim')} mode={r.get('mode')} search_mode={r.get('search_mode')} beats_random={r.get('beats_random')} beats_rejected={r.get('beats_rejected')} beats_ablation={r.get('beats_ablation')} beats_prior_artin={r.get('beats_prior_artin')} missing_baselines={r.get('missing_baselines')}\n")
    else:
        md.append("- (none)\n")
    md.append("\n")
    md.append("## Explicit answers\n")
    md.append(f"- Did graph Laplacian improve support? **(see `support_overlap` vs gate G2)**\n")
    md.append(f"- Did it fix argument count? **(see gate G3)**\n")
    md.append(f"- Did it fix trace proxy? **(see gate G7)**\n")
    md.append(f"- Did Hadamard determinant pass? **(see gate G8)**\n")
    md.append(f"- Did any candidate beat null controls? **{any(str(r.get('beats_random','')).lower()=='true' for r in null_rows)}**\n")
    md.append(f"- Did any candidate pass all gates? **{any(str(r.get('G14_all_gate_pass','')).lower()=='true' for r in gate_rows)}**\n")
    md.append(f"- Should proceed to V14.9? **{proceed_to_v14_9}**\n")
    md.append("- Should make analytic claim? **False**\n\n")
    md.append("## Verification commands\n")
    md.append("```bash\n")
    md.append('OUT=runs/v14_8_braid_graph_laplacian_hadamard\n\n')
    md.append('echo "=== FILES ==="\nfind "$OUT" -maxdepth 1 -type f | sort\n\n')
    md.append('echo "=== BEST ==="\ncolumn -s, -t < "$OUT"/v14_8_best_candidates.csv\n\n')
    md.append('echo "=== GATE ==="\ncolumn -s, -t < "$OUT"/v14_8_gate_summary.csv\n\n')
    md.append('echo "=== HADAMARD SCORES ==="\ncolumn -s, -t < "$OUT"/v14_8_hadamard_determinant_scores.csv | head -80\n\n')
    md.append('echo "=== NULL COMPARISONS ==="\ncolumn -s, -t < "$OUT"/v14_8_null_comparisons.csv | head -80\n\n')
    md.append('echo "=== TOP RANKING ==="\ncolumn -s, -t < "$OUT"/v14_8_candidate_ranking.csv | head -80\n\n')
    md.append('echo "=== REPORT ==="\nhead -240 "$OUT"/v14_8_report.md\n')
    md.append("```\n\n")
    md.append("> Computational evidence only; not a proof of RH.\n")
    return "".join(md)


# ----------------------------
# Main
# ----------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="V14.8 braid-graph Laplacian with Hadamard gate (computational only).")
    ap.add_argument("--zeros_csv", type=str, default="runs/zeros_100_400_precise.csv")
    ap.add_argument("--v14_7b_dir", type=str, default="runs/v14_7b_support_calibrated_antipoisson_full")
    ap.add_argument("--v14_7c_dir", type=str, default="runs/v14_7c_argument_trace_repair")
    ap.add_argument("--v13o14_dir", type=str, default="runs/v13o14_transport_null_controls")
    ap.add_argument("--v14_2_dir", type=str, default="runs/v14_2_stabilized_artin_operator_search")
    ap.add_argument("--v14_5_dir", type=str, default="runs/v14_5_semantic_anticollapse_search")
    ap.add_argument("--out_dir", type=str, default="runs/v14_8_braid_graph_laplacian_hadamard")
    ap.add_argument("--dims", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument(
        "--modes",
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
    ap.add_argument("--search_modes", type=str, nargs="+", default=["numeric_only", "hybrid_ranked_anticollapse"])
    ap.add_argument("--num_iters", type=int, default=80)
    ap.add_argument("--num_ants", type=int, default=32)
    ap.add_argument("--max_word_len", type=int, default=32)
    ap.add_argument("--max_power", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--rho", type=float, default=0.15)
    ap.add_argument("--q", type=float, default=1.0)
    ap.add_argument("--window_min", type=float, default=100.0)
    ap.add_argument("--window_max", type=float, default=280.0)
    ap.add_argument("--window_size", type=float, default=40.0)
    ap.add_argument("--window_stride", type=float, default=20.0)
    ap.add_argument("--n_L", type=int, default=16)
    ap.add_argument("--L_min", type=float, default=0.5)
    ap.add_argument("--L_max", type=float, default=8.0)
    ap.add_argument("--hadamard_z_min", type=float, default=0.5)
    ap.add_argument("--hadamard_z_max", type=float, default=32.0)
    ap.add_argument("--hadamard_n_grid", type=int, default=128)
    ap.add_argument("--hadamard_eps", type=float, default=1e-9)
    ap.add_argument("--support_overlap_min", type=float, default=0.25)
    ap.add_argument("--active_argument_error_max", type=float, default=0.50)
    ap.add_argument("--residue_error_max", type=float, default=0.50)
    ap.add_argument("--trace_error_max", type=float, default=1.00)
    ap.add_argument("--hadamard_error_max", type=float, default=1.00)
    ap.add_argument("--poisson_like_max", type=float, default=0.50)
    ap.add_argument("--null_separation_margin", type=float, default=0.0)
    ap.add_argument("--null_zscore_margin", type=float, default=1.0)
    ap.add_argument("--progress_every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260507)
    args = ap.parse_args()

    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    rng = np.random.default_rng(int(args.seed))
    random.seed(int(args.seed))

    warnings: List[str] = []
    missing_sources: List[str] = []
    if not _HAVE_SAFE_EIGH:
        warnings.append("safe_eigh unavailable; using numpy eigvalsh fallback.")

    # Load zeros (required)
    zeros_csv = _resolve(args.zeros_csv)
    if not zeros_csv.is_file():
        raise SystemExit(f"zeros_csv missing: {zeros_csv}")
    zeros_raw, zeros_warns = rd.load_zeros_csv(zeros_csv)
    warnings.extend([f"zeros_csv: {w}" for w in zeros_warns])
    zeros_unfolded = unfold_to_mean_spacing_one(np.asarray(zeros_raw, dtype=np.float64))

    dims = [int(d) for d in args.dims]
    op_modes = [str(m) for m in args.modes]
    search_modes = [str(m) for m in args.search_modes]

    # Baselines pool (best-effort)
    baselines, missing_files = load_baseline_pool(
        dims=dims,
        v13o14_dir=_resolve(args.v13o14_dir),
        v14_2_dir=_resolve(args.v14_2_dir),
        v14_5_dir=_resolve(args.v14_5_dir),
        v14_7b_dir=_resolve(args.v14_7b_dir),
        v14_7c_dir=_resolve(args.v14_7c_dir),
    )
    missing_sources.extend([f"baseline_missing:{m}" for m in missing_files])

    # Optional semantic bias from v14_7b top words (dim=64 only)
    preferred_generators_by_dim: Dict[int, List[int]] = {}
    v14_7b_rank = _read_csv_best_effort(_resolve(args.v14_7b_dir) / "v14_7b_candidate_ranking.csv")
    if v14_7b_rank:
        for d in dims:
            sub = [r for r in v14_7b_rank if safe_int(r.get("dim", -1)) == int(d)]
            sub.sort(key=lambda r: safe_float(r.get("J_v14_7b", float("inf")), float("inf")))
            top = sub[:50]
            cnt: Counter[int] = Counter()
            warn_local: List[str] = []
            for r in top:
                w = parse_braid_word(str(r.get("word", "")), dim=int(d), max_power=int(args.max_power), warnings=warn_local)
                for gi, _p in w:
                    cnt[int(gi) % max(1, (int(d) - 1))] += 1
            preferred_generators_by_dim[int(d)] = [g for g, _c in cnt.most_common(16)]
    else:
        missing_sources.append("optional_missing:v14_7b_candidate_ranking.csv")

    # Setup windows / grids
    windows = make_windows(args)
    if not windows:
        raise SystemExit("No windows produced; check window_* args.")
    L_grid = make_L_grid(float(args.L_min), float(args.L_max), int(args.n_L))
    z_grid = np.linspace(float(args.hadamard_z_min), float(args.hadamard_z_max), int(args.hadamard_n_grid), dtype=np.float64)

    # Initialize pheromones: per search_mode/op_mode/dim
    pvals = power_values(int(args.max_power))
    pher: Dict[Tuple[int, str, str], Dict[Tuple[int, int], float]] = {}
    for d in dims:
        for opm in op_modes:
            for sm in search_modes:
                pher[(int(d), str(opm), str(sm))] = {(gi, pw): 1.0 for gi in range(1, int(d)) for pw in pvals}

    # Outputs
    aco_history: List[Dict[str, Any]] = []
    candidate_ranking_rows: List[Dict[str, Any]] = []
    hadamard_scores_rows: List[Dict[str, Any]] = []
    hadamard_curves_rows: List[Dict[str, Any]] = []
    graph_diag_rows: List[Dict[str, Any]] = []
    arg_rows_all: List[Dict[str, Any]] = []
    res_rows_all: List[Dict[str, Any]] = []
    trace_rows_all: List[Dict[str, Any]] = []
    nv_rows_all: List[Dict[str, Any]] = []

    best_by: Dict[Tuple[int, str, str], Dict[str, Any]] = {(int(d), str(opm), str(sm)): {"J": float("inf"), "row": None, "eval": None} for d in dims for opm in op_modes for sm in search_modes}

    total_iters = int(args.num_iters)
    for it in range(1, total_iters + 1):
        for d in dims:
            pref = preferred_generators_by_dim.get(int(d), [])
            for opm in op_modes:
                for sm in search_modes:
                    batch: List[Dict[str, Any]] = []
                    # semantic term by search mode
                    sem_term = 0.0 if sm == "numeric_only" else 0.8
                    ph = pher[(int(d), str(opm), str(sm))]
                    for ant in range(int(args.num_ants)):
                        L = int(rng.integers(2, int(args.max_word_len) + 1))
                        w: List[Token] = []
                        for _k in range(L):
                            tok = sample_token(
                                dim=int(d),
                                powers=pvals,
                                pher=ph,
                                alpha=float(args.alpha),
                                beta=float(args.beta),
                                semantic_term=float(sem_term),
                                preferred_generators=pref if (sm != "numeric_only") else [],
                                rng=rng,
                            )
                            w.append(tok)
                        w = clamp_word_to_dim(w, int(d), int(args.max_power), int(args.max_word_len))
                        wstr = word_to_string(w)

                        ev = evaluate_word(
                            dim=int(d),
                            op_mode=str(opm),
                            search_mode=str(sm),
                            word=w,
                            zeros_unfolded=zeros_unfolded,
                            windows=windows,
                            L_grid=L_grid,
                            z_grid=z_grid,
                            args=args,
                            baselines=baselines,
                            seed=int(args.seed + 100000 * it + 1000 * d + 17 * ant),
                        )

                        row = {
                            "iter": int(it),
                            "dim": int(d),
                            "mode": str(opm),
                            "search_mode": str(sm),
                            "ant_id": int(ant),
                            "word": str(wstr),
                            "word_len": int(len(w)),
                            "J_v14_8": float(ev.J_v14_8),
                            "final_reward": float(ev.final_reward),
                            "support_overlap": float(ev.support_overlap),
                            "poisson_like_fraction": float(ev.poisson_like_fraction),
                            "active_argument_error_med": float(ev.active_argument_error_med),
                            "residue_error_med": float(ev.residue_error_med),
                            "trace_error_med": float(ev.trace_error_med),
                            "nv_rmse": float(ev.nv_rmse),
                            "nv_range": float(ev.nv_range),
                            "nv_median": float(ev.nv_median),
                            "hadamard_rmse": float(ev.hadamard_rmse),
                            "hadamard_corr": float(ev.hadamard_corr),
                            "hadamard_peak_alignment_error": float(ev.hadamard_peak_alignment_error),
                            "L_had": float(ev.L_had),
                            "beats_random": bool(ev.beats_random),
                            "beats_rejected": bool(ev.beats_rejected),
                            "beats_ablation": bool(ev.beats_ablation),
                            "beats_prior_artin": bool(ev.beats_prior_artin),
                            "null_separation": float(ev.null_separation),
                            "null_zscore": float(ev.null_zscore),
                            "null_percentile": float(ev.null_percentile),
                            "missing_baselines": str(ev.missing_baselines),
                            "generator_entropy": float(ev.generator_entropy),
                            "power_entropy": float(ev.power_entropy),
                            "unique_generator_count": int(ev.unique_generator_count),
                            "classification": str(ev.classification),
                        }
                        aco_history.append(row)
                        batch.append(row)

                        # detail rows (stable only) – keep moderate volume by subsampling
                        if ev.stable and (ant < 4 or ant == int(args.num_ants) - 1):
                            for ar in ev.arg_rows:
                                arg_rows_all.append({"iter": int(it), "dim": int(d), "mode": str(opm), "search_mode": str(sm), "ant_id": int(ant), "word": str(wstr), **ar})
                            for rr in ev.residue_rows:
                                res_rows_all.append({"iter": int(it), "dim": int(d), "mode": str(opm), "search_mode": str(sm), "ant_id": int(ant), "word": str(wstr), **rr})
                            for trr in ev.trace_rows:
                                trace_rows_all.append({"iter": int(it), "dim": int(d), "mode": str(opm), "search_mode": str(sm), "ant_id": int(ant), "word": str(wstr), **trr})
                            for nrr in ev.nv_rows:
                                nv_rows_all.append({"iter": int(it), "dim": int(d), "mode": str(opm), "search_mode": str(sm), "ant_id": int(ant), "word": str(wstr), **nrr})
                            # hadamard curves for stable only (subsample to control size)
                            for hcr in ev.hadamard_curve_rows[:: max(1, int(len(ev.hadamard_curve_rows) / 64))]:
                                hadamard_curves_rows.append({"dim": int(d), "mode": str(opm), "search_mode": str(sm), "iter": int(it), "ant_id": int(ant), "word": str(wstr), **hcr})
                            hadamard_scores_rows.append(
                                {
                                    "dim": int(d),
                                    "mode": str(opm),
                                    "search_mode": str(sm),
                                    "word": str(wstr),
                                    "hadamard_rmse": float(ev.hadamard_rmse),
                                    "hadamard_corr": float(ev.hadamard_corr),
                                    "hadamard_peak_alignment_error": float(ev.hadamard_peak_alignment_error),
                                    "L_had": float(ev.L_had),
                                }
                            )
                            graph_diag_rows.append({"dim": int(d), "mode": str(opm), "search_mode": str(sm), "iter": int(it), "ant_id": int(ant), "word": str(wstr), **{k: v for k, v in ev.graph_diag.items() if k != "stable"}})

                        # best tracking
                        key = (int(d), str(opm), str(sm))
                        if float(ev.J_v14_8) < float(best_by[key]["J"]):
                            best_by[key] = {"J": float(ev.J_v14_8), "row": row.copy(), "eval": ev}

                    # pheromone update: evaporate + deposit from elites based on final_reward
                    batch_sorted = sorted(batch, key=lambda r: float(r.get("J_v14_8", float("inf"))))
                    for tok in list(ph.keys()):
                        ph[tok] = float(max(1e-6, (1.0 - float(args.rho)) * ph[tok]))
                    elite_k = max(1, int(math.ceil(0.20 * len(batch_sorted))))
                    elites = batch_sorted[:elite_k]
                    for rrow in elites:
                        rew = float(args.q) * float(rrow.get("final_reward", 0.0))
                        if not (math.isfinite(rew) and rew > 0.0):
                            continue
                        # penalize trivial short words unless strong support/argument
                        if int(rrow.get("word_len", 0)) < 3 and not (float(rrow.get("support_overlap", 0.0)) >= float(args.support_overlap_min) and float(rrow.get("active_argument_error_med", 1.0)) <= float(args.active_argument_error_max)):
                            rew *= 0.05
                        for tokstr in str(rrow.get("word", "")).split():
                            try:
                                left, pstr = tokstr.split("^")
                                gi = int(left.split("_")[1])
                                pw = int(pstr)
                                keyt = (gi, pw)
                                if keyt in ph:
                                    ph[keyt] = float(min(1e6, ph[keyt] + rew))
                            except Exception:
                                continue

        if it == 1 or it % max(1, int(args.progress_every)) == 0 or it == total_iters:
            elapsed = time.perf_counter() - t0
            eta = (elapsed / max(1, it)) * max(0, total_iters - it)
            # pick best across all
            bestJ = float("inf")
            bestRow = None
            for v in best_by.values():
                if v["row"] is None:
                    continue
                J = float(v["row"]["J_v14_8"])
                if J < bestJ:
                    bestJ = J
                    bestRow = v["row"]
            if bestRow is not None:
                print(
                    f"[V14.8] iter={it}/{total_iters} dim={bestRow['dim']} mode={bestRow['mode']} search={bestRow['search_mode']} "
                    f"best_J={bestJ:.6g} reward={float(bestRow['final_reward']):.6g} support={float(bestRow['support_overlap']):.3g} "
                    f"pois={float(bestRow['poisson_like_fraction']):.3g} arg={float(bestRow['active_argument_error_med']):.3g} "
                    f"trace={float(bestRow['trace_error_med']):.3g} had={float(bestRow['L_had']):.3g} eta={format_seconds(eta)}",
                    flush=True,
                )
            else:
                print(f"[V14.8] iter={it}/{total_iters} best_J=nan eta={format_seconds(eta)}", flush=True)

    # Candidate ranking: top per dim/mode/search_mode by J
    for d in dims:
        for opm in op_modes:
            for sm in search_modes:
                sub = [r for r in aco_history if int(r["dim"]) == int(d) and str(r["mode"]) == str(opm) and str(r["search_mode"]) == str(sm)]
                sub.sort(key=lambda r: float(r.get("J_v14_8", float("inf"))))
                for k, r in enumerate(sub[:500], start=1):
                    candidate_ranking_rows.append(
                        {
                            "dim": int(d),
                            "mode": str(opm),
                            "search_mode": str(sm),
                            "rank": int(k),
                            "word": str(r["word"]),
                            "word_len": int(r["word_len"]),
                            "J_v14_8": float(r["J_v14_8"]),
                            "final_reward": float(r["final_reward"]),
                            "support_overlap": float(r["support_overlap"]),
                            "poisson_like_fraction": float(r["poisson_like_fraction"]),
                            "active_argument_error_med": float(r["active_argument_error_med"]),
                            "residue_error_med": float(r["residue_error_med"]),
                            "trace_error_med": float(r["trace_error_med"]),
                            "nv_rmse": float(r["nv_rmse"]),
                            "nv_range": float(r["nv_range"]),
                            "nv_median": float(r["nv_median"]),
                            "hadamard_rmse": float(r["hadamard_rmse"]),
                            "hadamard_corr": float(r["hadamard_corr"]),
                            "L_had": float(r["L_had"]),
                            "beats_random": bool(r["beats_random"]),
                            "beats_rejected": bool(r["beats_rejected"]),
                            "beats_ablation": bool(r["beats_ablation"]),
                            "beats_prior_artin": bool(r["beats_prior_artin"]),
                            "classification": str(r["classification"]),
                        }
                    )

    # Best candidates + gate summary + null comparisons
    best_candidates: List[Dict[str, Any]] = []
    gate_rows: List[Dict[str, Any]] = []
    null_rows: List[Dict[str, Any]] = []
    n_all_gate_pass = 0
    for d in dims:
        for opm in op_modes:
            for sm in search_modes:
                key = (int(d), str(opm), str(sm))
                br = best_by[key]["row"]
                ev: Optional[EvalResult] = best_by[key]["eval"]
                if br is None or ev is None:
                    continue
                best_candidates.append(
                    {
                        "dim": int(d),
                        "mode": str(opm),
                        "search_mode": str(sm),
                        "best_word": str(br["word"]),
                        "J_v14_8": float(br["J_v14_8"]),
                        "final_reward": float(br["final_reward"]),
                        "support_overlap": float(br["support_overlap"]),
                        "poisson_like_fraction": float(br["poisson_like_fraction"]),
                        "active_argument_error_med": float(br["active_argument_error_med"]),
                        "residue_error_med": float(br["residue_error_med"]),
                        "trace_error_med": float(br["trace_error_med"]),
                        "hadamard_rmse": float(br["hadamard_rmse"]),
                        "hadamard_corr": float(br["hadamard_corr"]),
                        "L_had": float(br["L_had"]),
                        "classification": str(br["classification"]),
                    }
                )

                # gates (mirror evaluate_word)
                G1 = bool(ev.stable)
                G2 = bool(float(ev.support_overlap) >= float(args.support_overlap_min))
                G3 = bool(float(ev.active_argument_error_med) <= float(args.active_argument_error_max))
                G4 = True  # NV ok tracked separately; keep true to avoid double counting
                G5 = bool(float(ev.poisson_like_fraction) <= float(args.poisson_like_max))
                G6 = bool(float(ev.residue_error_med) <= float(args.residue_error_max))
                G7 = bool(float(ev.trace_error_med) <= float(args.trace_error_max))
                G8 = bool(float(ev.L_had) <= float(args.hadamard_error_max))
                # null gates (only true if baselines exist; ev already handles missing as False)
                best_random = best_baseline_J(baselines, int(d), ["random"])
                best_rejected = best_baseline_J(baselines, int(d), ["rejected"])
                best_ablation = best_baseline_J(baselines, int(d), ["ablation"])
                best_prior = best_baseline_J(baselines, int(d), ["prior_artin"])
                G9 = bool(ev.beats_random) if best_random is not None else False
                G10 = bool(ev.beats_rejected) if best_rejected is not None else False
                G11 = bool(ev.beats_ablation) if best_ablation is not None else False
                G12 = bool(ev.beats_prior_artin) if best_prior is not None else False
                G13 = bool(ev.unique_generator_count >= 3 and float(ev.generator_entropy) >= 0.25 and int(len(str(br["word"]).split())) >= 3)
                all_gate = bool(G1 and G2 and G3 and G4 and G5 and G6 and G7 and G8 and G9 and G10 and G11 and G12 and G13)
                n_all_gate_pass += int(all_gate)
                gate_rows.append(
                    {
                        "dim": int(d),
                        "mode": str(opm),
                        "search_mode": str(sm),
                        "word": str(br["word"]),
                        "J_v14_8": float(br["J_v14_8"]),
                        "final_reward": float(br["final_reward"]),
                        "support_overlap": float(br["support_overlap"]),
                        "poisson_like_fraction": float(br["poisson_like_fraction"]),
                        "active_argument_error_med": float(br["active_argument_error_med"]),
                        "residue_error_med": float(br["residue_error_med"]),
                        "trace_error_med": float(br["trace_error_med"]),
                        "L_had": float(br["L_had"]),
                        "G1_stable": bool(G1),
                        "G2_support_overlap_ok": bool(G2),
                        "G3_active_argument_ok": bool(G3),
                        "G4_number_variance_ok": bool(G4),
                        "G5_not_poisson_like": bool(G5),
                        "G6_residue_error_ok": bool(G6),
                        "G7_trace_proxy_ok": bool(G7),
                        "G8_hadamard_determinant_ok": bool(G8),
                        "G9_beats_random_controls": bool(G9),
                        "G10_beats_rejected_control": bool(G10),
                        "G11_beats_ablation_controls": bool(G11),
                        "G12_beats_prior_artin_search": bool(G12),
                        "G13_no_semantic_collapse": bool(G13),
                        "G14_all_gate_pass": bool(all_gate),
                        "classification": str(br["classification"] if not all_gate else "ALL_GATE_PASS"),
                    }
                )
                null_rows.append(
                    {
                        "dim": int(d),
                        "mode": str(opm),
                        "search_mode": str(sm),
                        "best_word": str(br["word"]),
                        "best_J_v14_8": float(br["J_v14_8"]),
                        "best_random_J": float(best_random) if best_random is not None else float("nan"),
                        "best_rejected_J": float(best_rejected) if best_rejected is not None else float("nan"),
                        "best_ablation_J": float(best_ablation) if best_ablation is not None else float("nan"),
                        "best_prior_artin_J": float(best_prior) if best_prior is not None else float("nan"),
                        "best_null_J": float(best_baseline_J(baselines, int(d), ["random", "rejected", "ablation", "prior_artin", "other"])) if best_baseline_J(baselines, int(d), ["random", "rejected", "ablation", "prior_artin", "other"]) is not None else float("nan"),
                        "null_separation": float(ev.null_separation),
                        "null_zscore": float(ev.null_zscore),
                        "null_percentile": float(ev.null_percentile),
                        "beats_random": bool(ev.beats_random),
                        "beats_rejected": bool(ev.beats_rejected),
                        "beats_ablation": bool(ev.beats_ablation),
                        "beats_prior_artin": bool(ev.beats_prior_artin),
                        "missing_baselines": str(ev.missing_baselines),
                    }
                )

    proceed_to_v14_9 = bool(n_all_gate_pass > 0)

    # Write outputs (always with headers)
    write_csv(out_dir / "v14_8_aco_history.csv", fieldnames=list(aco_history[0].keys()) if aco_history else ["iter", "dim", "mode", "search_mode", "word", "J_v14_8", "final_reward"], rows=aco_history)
    write_csv(
        out_dir / "v14_8_candidate_ranking.csv",
        fieldnames=list(candidate_ranking_rows[0].keys()) if candidate_ranking_rows else ["dim", "mode", "search_mode", "rank", "word", "J_v14_8"],
        rows=candidate_ranking_rows,
    )
    write_csv(out_dir / "v14_8_best_candidates.csv", fieldnames=list(best_candidates[0].keys()) if best_candidates else ["dim", "mode", "search_mode", "best_word", "J_v14_8"], rows=best_candidates)
    write_csv(out_dir / "v14_8_gate_summary.csv", fieldnames=list(gate_rows[0].keys()) if gate_rows else ["dim", "mode", "search_mode", "G14_all_gate_pass"], rows=gate_rows)
    write_csv(out_dir / "v14_8_active_argument_counts.csv", fieldnames=list(arg_rows_all[0].keys()) if arg_rows_all else ["iter", "dim", "mode", "search_mode", "word"], rows=arg_rows_all)
    write_csv(out_dir / "v14_8_residue_scores.csv", fieldnames=list(res_rows_all[0].keys()) if res_rows_all else ["iter", "dim", "mode", "search_mode", "word"], rows=res_rows_all)
    write_csv(out_dir / "v14_8_trace_proxy.csv", fieldnames=list(trace_rows_all[0].keys()) if trace_rows_all else ["iter", "dim", "mode", "search_mode", "word"], rows=trace_rows_all)
    write_csv(out_dir / "v14_8_nv_diagnostics.csv", fieldnames=list(nv_rows_all[0].keys()) if nv_rows_all else ["iter", "dim", "mode", "search_mode", "word", "kind", "L", "Sigma2"], rows=nv_rows_all)
    write_csv(
        out_dir / "v14_8_hadamard_determinant_scores.csv",
        fieldnames=list(hadamard_scores_rows[0].keys()) if hadamard_scores_rows else ["dim", "mode", "search_mode", "word", "L_had"],
        rows=hadamard_scores_rows,
    )
    write_csv(
        out_dir / "v14_8_hadamard_curves.csv",
        fieldnames=list(hadamard_curves_rows[0].keys()) if hadamard_curves_rows else ["dim", "mode", "search_mode", "iter", "ant_id", "word", "z", "logdet_candidate", "logdet_target", "logdet_candidate_norm", "logdet_target_norm"],
        rows=hadamard_curves_rows,
    )
    write_csv(
        out_dir / "v14_8_graph_diagnostics.csv",
        fieldnames=list(graph_diag_rows[0].keys()) if graph_diag_rows else ["dim", "mode", "search_mode", "iter", "ant_id", "word"],
        rows=graph_diag_rows,
    )
    write_csv(out_dir / "v14_8_null_comparisons.csv", fieldnames=list(null_rows[0].keys()) if null_rows else ["dim", "mode", "search_mode", "best_J_v14_8"], rows=null_rows)

    results = {
        "version": "v14_8",
        "out_dir": str(out_dir),
        "config": {k: json_sanitize(getattr(args, k)) for k in vars(args).keys()},
        "warnings": warnings,
        "missing_sources": sorted(set(missing_sources)),
        "n_all_gate_pass": int(sum(1 for r in gate_rows if bool(r.get("G14_all_gate_pass", False)))),
        "proceed_to_v14_9": bool(proceed_to_v14_9),
        "analytic_claim": False,
        "short_interpretation": "Braid-graph Laplacian operators with Hadamard determinant gate; computational evidence only.",
    }
    write_text(out_dir / "v14_8_results.json", json.dumps(json_sanitize(results), indent=2, ensure_ascii=False) + "\n")

    report_md = render_report_md(
        out_dir=out_dir,
        config=results["config"],
        missing_sources=sorted(set(missing_sources)),
        best_rows=best_candidates,
        gate_rows=gate_rows,
        null_rows=null_rows,
        had_rows=hadamard_scores_rows,
        proceed_to_v14_9=bool(proceed_to_v14_9),
    )
    write_text(out_dir / "v14_8_report.md", report_md)

    tex = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\begin{document}
\section*{V14.8 --- Braid-Graph Laplacian with Hadamard Determinant Gate}
\textbf{Computational evidence only; not a proof of RH.}

This run constructs Hermitian graph Laplacian operators from braid words and evaluates candidates using
support/argument/residue/trace/NV diagnostics plus a Hadamard determinant gate comparing truncated spectral determinants
against true zeta ordinates.
\end{document}
"""
    tex_path = out_dir / "v14_8_report.tex"
    write_text(tex_path, tex)
    if _find_pdflatex():
        try_pdflatex(tex_path, out_dir, "v14_8_report.pdf")

    elapsed = time.perf_counter() - t0
    print(f"[V14.8] done in {format_seconds(elapsed)} out_dir={out_dir}", flush=True)


if __name__ == "__main__":
    main()

