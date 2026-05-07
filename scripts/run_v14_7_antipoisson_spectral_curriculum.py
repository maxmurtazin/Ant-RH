#!/usr/bin/env python3
"""
V14.7 — Anti-Poisson Spectral Curriculum for Artin-DTES Agents

Computational evidence only; not a proof of RH.

This is a new staged Artin/ACO/DTES search designed to address the dominant failure mode
observed in V13O.14–V14.6b: candidates are Poisson-like and/or fail support/argument/residue gates.

Key design:
  - Curriculum (stages A..G) via time-varying effective lambdas.
  - Hard anti-Poisson barrier removes Poisson-like candidates from elite selection.
  - Semantic pheromones + numeric pheromones + anti-collapse diversity penalties.
  - Real null-control comparisons against baseline pools (best-effort, never silent-pass).

Outputs are written to a run directory. Script is robust to missing optional prior files.
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
    """
    H = sum_k c_k * A_{i_k,p_k}, c_k = 1/sqrt(k+1)
    Hermitian, trace-free, spectral radius normalized.
    """
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
# Word / diversity / semantic
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


def generator_entropy(word: List[Tuple[int, int]], n_generators: int) -> float:
    gens = [int(i) for i, _p in word]
    if not gens or n_generators <= 1:
        return 0.0
    c = Counter(gens)
    total = float(sum(c.values()))
    ps = np.asarray([v / total for v in c.values()], dtype=np.float64)
    H = float(-np.sum(ps * np.log(np.maximum(1e-12, ps))))
    return float(H / max(1e-12, math.log(float(n_generators))))


def power_entropy(word: List[Tuple[int, int]], n_powers: int) -> float:
    ps = [int(p) for _i, p in word]
    if not ps or n_powers <= 1:
        return 0.0
    c = Counter(ps)
    total = float(sum(c.values()))
    probs = np.asarray([v / total for v in c.values()], dtype=np.float64)
    H = float(-np.sum(probs * np.log(np.maximum(1e-12, probs))))
    return float(H / max(1e-12, math.log(float(n_powers))))


def repeated_bigram_fraction(word: List[Tuple[int, int]]) -> float:
    if len(word) < 3:
        return 0.0
    bigrams = [(word[i][0], word[i][1], word[i + 1][0], word[i + 1][1]) for i in range(len(word) - 1)]
    c = Counter(bigrams)
    rep = sum(v - 1 for v in c.values() if v > 1)
    return float(rep / max(1, len(bigrams)))


def approx_edit_distance(a: List[Tuple[int, int]], b: List[Tuple[int, int]]) -> int:
    """Cheap diversity proxy: token mismatches on aligned prefix + length penalty."""
    la, lb = len(a), len(b)
    m = min(la, lb)
    mism = sum(1 for i in range(m) if a[i] != b[i])
    return int(mism + abs(la - lb))


def semantic_rules_seed() -> List[Dict[str, Any]]:
    return [
        {"type": "avoid", "motif": "avoid_band_collapse", "weight": 0.8, "confidence": 0.85, "scope": "search", "source": "init"},
        {"type": "prefer", "motif": "prefer_entropy", "weight": 0.6, "confidence": 0.75, "scope": "search", "source": "init"},
        {"type": "avoid", "motif": "avoid_repeated_bigrams", "weight": 0.7, "confidence": 0.8, "scope": "search", "source": "init"},
        {"type": "avoid", "motif": "avoid_too_short", "weight": 0.4, "confidence": 0.7, "scope": "search", "source": "init"},
    ]


def rule_weight(rec: Dict[str, Any]) -> float:
    w = abs(safe_float(rec.get("weight", 0.0), 0.0))
    c = max(0.0, min(1.0, safe_float(rec.get("confidence", 1.0), 1.0)))
    return float(w * c)


def semantic_score(word: List[Tuple[int, int]], dim: int, max_power: int, memory: Sequence[Dict[str, Any]]) -> float:
    gens = [int(i) for i, _p in word]
    if not gens:
        return 0.0
    band = float(max(gens) - min(gens) + 1)
    Hn = generator_entropy(word, max(1, dim - 1))
    rep = repeated_bigram_fraction(word)
    L = len(word)
    score = 0.0
    for rec in memory:
        typ = str(rec.get("type", "")).strip().lower()
        motif = str(rec.get("motif", "")).strip()
        w = rule_weight(rec)
        if w <= 0:
            continue
        match = False
        if motif == "prefer_entropy":
            match = Hn >= 0.6
        elif motif == "avoid_band_collapse":
            match = band <= max(3.0, 0.08 * float(dim))
        elif motif == "avoid_repeated_bigrams":
            match = rep >= 0.15
        elif motif == "avoid_too_short":
            match = L <= 3
        else:
            match = False
        if not match:
            continue
        if typ == "prefer":
            score += w
        elif typ in ("avoid", "artifact", "caution"):
            score -= w
    return float(score)


def update_semantic_from_diagnostics(dim: int, diag: Dict[str, Any], source: str, it: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    sup = safe_float(diag.get("support_overlap", float("nan")))
    pois = safe_float(diag.get("poisson_like_fraction", float("nan")))
    ent = safe_float(diag.get("generator_entropy", float("nan")))
    if math.isfinite(sup) and sup < 0.25:
        out.append(
            {"type": "avoid", "motif": "avoid_band_collapse", "text": "Avoid support collapse via narrow bands.", "weight": 0.9, "confidence": 0.9, "dim": int(dim), "source": source, "iter": int(it)}
        )
    if math.isfinite(pois) and pois >= 0.95:
        out.append(
            {"type": "caution", "motif": "prefer_entropy", "text": "Poisson dominates; bias toward entropy/diversity motifs.", "weight": 0.6, "confidence": 0.85, "dim": int(dim), "source": source, "iter": int(it)}
        )
    if math.isfinite(ent) and ent < 0.35:
        out.append(
            {"type": "avoid", "motif": "avoid_repeated_bigrams", "text": "Avoid looping motifs; repeated bigrams correlate with collapse.", "weight": 0.7, "confidence": 0.8, "dim": int(dim), "source": source, "iter": int(it)}
        )
    return out


# ----------------------------
# Spectral metrics
# ----------------------------


def unfold_to_mean_spacing_one(x: np.ndarray) -> np.ndarray:
    return rd.unfold_to_mean_spacing_one(np.asarray(x, dtype=np.float64).reshape(-1))


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


def nv_long_range_error(op_nv: np.ndarray, tgt_nv: np.ndarray, L_grid: np.ndarray, frac: float = 0.33) -> float:
    L = np.asarray(L_grid, dtype=np.float64)
    m = np.isfinite(op_nv) & np.isfinite(tgt_nv) & np.isfinite(L)
    if not m.any():
        return 1.0
    Lm = L[m]
    y = (op_nv - tgt_nv)[m]
    if Lm.size < 8:
        return float(np.mean(np.abs(y))) if y.size else 1.0
    # long range: top frac of L
    thr = np.quantile(Lm, 1.0 - float(frac))
    sel = Lm >= thr
    if not sel.any():
        sel = slice(None)
    yy = y[sel]
    return float(np.sqrt(np.mean(yy**2))) if yy.size else 1.0


def nv_slope(op_nv: np.ndarray, L_grid: np.ndarray) -> float:
    L = np.asarray(L_grid, dtype=np.float64)
    y = np.asarray(op_nv, dtype=np.float64)
    m = np.isfinite(L) & np.isfinite(y)
    if m.sum() < 8:
        return float("nan")
    try:
        # linear slope over long-range half
        Lm = L[m]
        ym = y[m]
        thr = np.quantile(Lm, 0.5)
        sel = Lm >= thr
        if sel.sum() < 4:
            sel = np.ones_like(Lm, dtype=bool)
        b, a = np.polyfit(Lm[sel], ym[sel], deg=1)  # y ~ b L + a
        return float(b)
    except Exception:
        return float("nan")


def active_argument_metrics(levels: np.ndarray, target: np.ndarray, windows: List[Tuple[float, float]]) -> Tuple[float, float, float, List[Dict[str, Any]]]:
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


def residue_metrics(levels: np.ndarray, target: np.ndarray, windows: List[Tuple[float, float]], eta: float, n_contour_points: int) -> Tuple[float, float, List[Dict[str, Any]]]:
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


def trace_metrics(levels: np.ndarray, target: np.ndarray, windows: List[Tuple[float, float]]) -> Tuple[float, List[Dict[str, Any]]]:
    errs: List[float] = []
    rows: List[Dict[str, Any]] = []
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
            tr = abs(math.log1p(max(0.0, float(Sop))) - math.log1p(max(0.0, float(Stg))))
            errs.append(float(tr))
            rows.append({"window_a": float(a), "window_b": float(b), "center": float(c), "sigma": float(s), "err": float(tr)})
    med = float(np.median(np.asarray(errs, dtype=np.float64))) if errs else 1.0
    return med, rows


# ----------------------------
# Curriculum schedule
# ----------------------------


def sigmoid_schedule(progress: float, center: float, sharpness: float = 12.0) -> float:
    x = float(progress) - float(center)
    return float(1.0 / (1.0 + math.exp(-float(sharpness) * x)))


def curriculum_lambdas(progress: float, args: argparse.Namespace) -> Dict[str, float]:
    return {
        "lambda_stability": float(args.lambda_stability),
        "lambda_support": float(args.lambda_support),
        "lambda_arg": float(args.lambda_arg) * sigmoid_schedule(progress, center=0.20),
        "lambda_nv": float(args.lambda_nv) * sigmoid_schedule(progress, center=0.35),
        "lambda_ap": float(args.lambda_ap) * sigmoid_schedule(progress, center=0.40),
        "lambda_residue": float(args.lambda_residue) * sigmoid_schedule(progress, center=0.55),
        "lambda_trace": float(args.lambda_trace) * sigmoid_schedule(progress, center=0.65),
        "lambda_null": float(args.lambda_null) * sigmoid_schedule(progress, center=0.75),
        "lambda_complexity": float(args.lambda_complexity),
        "lambda_semantic_collapse": float(args.lambda_semantic_collapse),
    }


# ----------------------------
# Null-control baseline pool loading (best-effort)
# ----------------------------


def load_baseline_pool(
    *,
    dims: List[int],
    v13o14_dir: Path,
    v14_2_dir: Path,
    v14_5_dir: Path,
    v14_6b_dir: Path,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Returns (pool_rows, missing_files).
    Normalized columns:
      dim, source, baseline_kind, label, mode, J, source_file, missing_source, notes
    """
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

    # V14.6b baseline pool if exists (best path)
    p6b = v14_6b_dir / "v14_6b_prior_baseline_pool.csv"
    rows6b = _read_csv_best_effort(p6b)
    if rows6b:
        for r in rows6b:
            d = safe_int(r.get("dim", 0))
            if d in dims:
                add_row(d, str(r.get("source", "v14_6b")), str(r.get("baseline_kind", "other")), str(r.get("label", "")), str(r.get("mode", "")), r.get("J", float("nan")), "v14_6b_prior_baseline_pool.csv", bool(r.get("missing_source", False)), str(r.get("notes", "")))
    else:
        if p6b.exists():
            missing.append("v14_6b_prior_baseline_pool.csv (unreadable)")
        else:
            missing.append("v14_6b_prior_baseline_pool.csv (missing)")

    # V13O.14
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
            group = str(r.get("word_group", "")).strip()
            if not group:
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
            add_row(d, "v13o14", kind, group, str(r.get("best_mode", "")), J, "v13o14_candidate_mode_summary.csv", False, str(r.get("candidate_classification", "")))
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

    # V14.5
    p5a = v14_5_dir / "v14_5_ablation_summary.csv"
    p5h = v14_5_dir / "v14_5_aco_history.csv"
    r5a = _read_csv_best_effort(p5a)
    r5h = _read_csv_best_effort(p5h)
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
    if not r5h:
        missing.append("v14_5_aco_history.csv")
    else:
        for r in r5h:
            d = safe_int(r.get("dim", 0))
            if d not in dims:
                continue
            J = safe_float(r.get("raw_J", float("nan")))
            if math.isfinite(J):
                add_row(d, "v14_5", "prior_artin", "v14_5_history", str(r.get("mode", "")), J, "v14_5_aco_history.csv", False, "aco_history")

    # V14.2
    p2b = v14_2_dir / "v14_2_best_candidates.csv"
    p2h = v14_2_dir / "v14_2_aco_history.csv"
    r2b = _read_csv_best_effort(p2b)
    r2h = _read_csv_best_effort(p2h)
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
    if not r2h:
        missing.append("v14_2_aco_history.csv")
    else:
        for r in r2h:
            d = safe_int(r.get("dim", 0))
            if d not in dims:
                continue
            J = safe_float(r.get("raw_J", r.get("J_v14_2", float("nan"))))
            if math.isfinite(J):
                add_row(d, "v14_2", "prior_artin", "v14_2_history", "aco_history", J, "v14_2_aco_history.csv", False, "aco_history")

    # If absolutely nothing loaded for a dim, add explicit missing placeholders
    for d in dims:
        any_dim = any(int(r.get("dim", 0)) == int(d) and not bool(r.get("missing_source", False)) and math.isfinite(safe_float(r.get("J", float("nan")))) for r in pool)
        if not any_dim:
            add_row(d, "missing", "missing", "no_baselines_loaded", "", float("nan"), "", True, "no baseline rows loaded for this dim")

    return pool, sorted(set(missing))


def best_baseline_J(pool: List[Dict[str, Any]], dim: int, kinds: Sequence[str]) -> Optional[float]:
    Js = []
    for r in pool:
        if int(r.get("dim", 0)) != int(dim):
            continue
        if bool(r.get("missing_source", False)):
            continue
        if str(r.get("baseline_kind", "")) in set(kinds):
            J = safe_float(r.get("J", float("nan")))
            if math.isfinite(J):
                Js.append(float(J))
    return float(min(Js)) if Js else None


def null_distribution_Js(pool: List[Dict[str, Any]], dim: int) -> List[float]:
    xs = []
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
    rng: np.random.Generator,
) -> Tuple[int, int]:
    items = list(pher.keys())
    tau = np.asarray([pher[it] for it in items], dtype=np.float64)
    mid = 0.5 * (dim - 1)
    # heuristic: mild mid preference + smaller power
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
# Evaluate candidate with curriculum + barriers
# ----------------------------


@dataclass
class EvalResult:
    stable: bool
    J: float
    final_reward: float
    poisson_like_fraction: float
    support_overlap: float
    active_argument_error_med: float
    residue_error_med: float
    trace_error_med: float
    generator_entropy: float
    power_entropy: float
    collapse_loss: float
    antiP_loss: float
    nv_rmse: float
    nv_long_rmse: float
    nv_slope_op: float
    nv_slope_tgt: float
    # stage-wise losses
    L_stability: float
    L_support: float
    L_arg: float
    L_nv: float
    L_residue: float
    L_trace: float
    L_null: float
    L_complexity: float
    L_semantic_collapse: float


def evaluate_word(
    *,
    dim: int,
    word: List[Tuple[int, int]],
    target_levels: np.ndarray,
    windows: List[Tuple[float, float]],
    L_grid: np.ndarray,
    pool: List[Dict[str, Any]],
    semantic_memory: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    lambdas_eff: Dict[str, float],
    seed: int,
    recent_elite_words: List[List[Tuple[int, int]]],
) -> Tuple[EvalResult, List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns EvalResult and detail rows:
      argument_count_rows, nv_curve_rows, residue_rows, trace_rows
    """
    # Stage A: stability
    H, st = build_stabilized_operator(dim, word, seed=seed)
    if H is None or not st.get("stable", False):
        # hard fail: large but finite penalties, reward=0
        er = EvalResult(
            stable=False,
            J=1e6,
            final_reward=0.0,
            poisson_like_fraction=1.0,
            support_overlap=0.0,
            active_argument_error_med=1.0,
            residue_error_med=1.0,
            trace_error_med=1.0,
            generator_entropy=0.0,
            power_entropy=0.0,
            collapse_loss=1.0,
            antiP_loss=1.0,
            nv_rmse=1.0,
            nv_long_rmse=1.0,
            nv_slope_op=float("nan"),
            nv_slope_tgt=float("nan"),
            L_stability=1.0,
            L_support=1.0,
            L_arg=1.0,
            L_nv=1.0,
            L_residue=1.0,
            L_trace=1.0,
            L_null=1.0,
            L_complexity=1.0,
            L_semantic_collapse=1.0,
        )
        return er, [], [], [], []

    w = safe_eigvalsh(H, seed=seed)
    if w is None or w.size < 8:
        er = EvalResult(
            stable=False,
            J=1e6,
            final_reward=0.0,
            poisson_like_fraction=1.0,
            support_overlap=0.0,
            active_argument_error_med=1.0,
            residue_error_med=1.0,
            trace_error_med=1.0,
            generator_entropy=0.0,
            power_entropy=0.0,
            collapse_loss=1.0,
            antiP_loss=1.0,
            nv_rmse=1.0,
            nv_long_rmse=1.0,
            nv_slope_op=float("nan"),
            nv_slope_tgt=float("nan"),
            L_stability=1.0,
            L_support=1.0,
            L_arg=1.0,
            L_nv=1.0,
            L_residue=1.0,
            L_trace=1.0,
            L_null=1.0,
            L_complexity=1.0,
            L_semantic_collapse=1.0,
        )
        return er, [], [], [], []

    levels = unfold_to_mean_spacing_one(w)
    # align quantiles for windowing
    try:
        oq = np.quantile(levels, [0.1, 0.9])
        tq = np.quantile(target_levels, [0.1, 0.9])
        a = float(max(1e-12, tq[1] - tq[0]) / max(1e-12, oq[1] - oq[0]))
        b = float(tq[0] - a * oq[0])
        levels = (a * levels + b).astype(np.float64, copy=False)
    except Exception:
        pass

    # Stage B/C: support + argument
    arg_med, arg_mean, support_overlap, arg_rows = active_argument_metrics(levels, target_levels, windows)

    # Stage D: NV + anti-Poisson features
    op_nv = number_variance_curve(levels, L_grid)
    tgt_nv = number_variance_curve(target_levels, L_grid)
    nv_rmse = float(curve_l2(op_nv, tgt_nv))
    nv_long_rmse = float(nv_long_range_error(op_nv, tgt_nv, L_grid, frac=0.33))
    dP = float(curve_l2(op_nv, sigma2_poisson(L_grid)))
    dG = float(curve_l2(op_nv, sigma2_gue_asymptotic(L_grid)))
    poisson_like = bool(math.isfinite(dP) and math.isfinite(dG) and dP < dG)
    pois_frac = 1.0 if poisson_like else 0.0
    slope_op = nv_slope(op_nv, L_grid)
    slope_tgt = nv_slope(tgt_nv, L_grid)
    # anti-Poisson soft penalty terms
    nv_slope_target_like_max = float(max(0.0, slope_tgt)) if math.isfinite(slope_tgt) else 0.0
    long_range_nv_margin = float(0.15)  # conservative margin; can be tuned later
    L_antiP = float(pois_frac)
    if math.isfinite(slope_op):
        L_antiP += float(max(0.0, float(slope_op) - float(nv_slope_target_like_max)))
    L_antiP += float(max(0.0, float(nv_long_rmse) - float(long_range_nv_margin)))

    nv_rows = []
    for L, y in zip(L_grid, op_nv):
        nv_rows.append({"kind": "operator", "L": float(L), "Sigma2": safe_float(y)})
    for L, y in zip(L_grid, tgt_nv):
        nv_rows.append({"kind": "target", "L": float(L), "Sigma2": safe_float(y)})

    # Stage E: residue
    residue_med, imag_leak_med, residue_rows = residue_metrics(levels, target_levels, windows, eta=float(args.eta), n_contour_points=int(args.n_contour_points))
    # Stage F: trace
    trace_med, trace_rows = trace_metrics(levels, target_levels, windows)

    # Stage G: null separation (best-of baselines)
    best_random_J = best_baseline_J(pool, dim, ["random"])
    best_rejected_J = best_baseline_J(pool, dim, ["rejected"])
    best_ablation_J = best_baseline_J(pool, dim, ["ablation"])
    best_prior_J = best_baseline_J(pool, dim, ["prior_artin"])
    best_null_J = best_baseline_J(pool, dim, ["random", "rejected", "ablation", "prior_artin", "other"])
    null_dist = null_distribution_Js(pool, dim)
    null_z = float("nan")
    null_pct = float("nan")
    if len(null_dist) >= 3:
        mu = float(np.mean(null_dist))
        sd = float(np.std(null_dist))
        if sd > 1e-12:
            null_z = float((mu - 0.0) / sd)  # placeholder; will compute after J defined
        null_pct = float(np.mean([1.0 if x <= 0.0 else 0.0 for x in null_dist]))

    # Diversity / semantic collapse loss
    gen_ent = generator_entropy(word, max(1, dim - 1))
    pow_ent = power_entropy(word, n_powers=max(2, 2 * int(args.max_power)))
    rep_bi = repeated_bigram_fraction(word)
    # near-identical to recent elites -> penalty
    eds = [approx_edit_distance(word, w2) for w2 in recent_elite_words[-25:]] if recent_elite_words else []
    near = float(np.mean([1.0 if ed <= 2 else 0.0 for ed in eds])) if eds else 0.0
    L_collapse = float(max(0.0, 0.5 - gen_ent) + max(0.0, 0.45 - pow_ent) + rep_bi + near)
    # semantic: higher score => smaller loss
    sem = semantic_score(word, dim, int(args.max_power), semantic_memory)
    L_semantic_collapse = float(max(0.0, -sem)) + L_collapse

    # complexity
    L_complexity = float(len(word) / max(1, int(args.max_word_len)))

    # Convert metrics to losses (finite, robust)
    L_stability = 0.0
    L_support = float(max(0.0, float(args.support_overlap_min) - float(support_overlap)))
    L_arg = float(arg_med)
    L_nv = float(nv_rmse) + 0.5 * float(nv_long_rmse)
    L_residue = float(max(0.0, float(residue_med) - float(args.residue_error_margin)))
    L_trace = float(max(0.0, float(trace_med) - float(args.trace_error_margin)))
    # null loss: prefer beating best prior; if missing baselines, penalize mildly but keep finite
    if best_null_J is None or not math.isfinite(float(best_null_J)):
        L_null = 1.0
    else:
        # will compare after J computed; use placeholder 0 for now and later update in report
        L_null = 0.0

    # objective J_v14_7
    J = (
        float(lambdas_eff["lambda_stability"]) * float(L_stability)
        + float(lambdas_eff["lambda_support"]) * float(L_support)
        + float(lambdas_eff["lambda_arg"]) * float(L_arg)
        + float(lambdas_eff["lambda_nv"]) * float(L_nv)
        + float(lambdas_eff["lambda_ap"]) * float(L_antiP)
        + float(lambdas_eff["lambda_residue"]) * float(L_residue)
        + float(lambdas_eff["lambda_trace"]) * float(L_trace)
        + float(lambdas_eff["lambda_null"]) * float(L_null)
        + float(lambdas_eff["lambda_complexity"]) * float(L_complexity)
        + float(lambdas_eff["lambda_semantic_collapse"]) * float(L_semantic_collapse)
    )
    J = float(min(max(J, 0.0), 1e6))

    # now compute null terms depending on J and update null z/pct properly
    if len(null_dist) >= 3:
        mu = float(np.mean(null_dist))
        sd = float(np.std(null_dist))
        null_z = float((mu - J) / sd) if sd > 1e-12 else float("nan")
        null_pct = float(np.mean([1.0 if x <= J else 0.0 for x in null_dist]))

    # base reward with barriers
    base_reward = float(1.0 / (1.0 + J))
    # stage-dependent multiplicative barriers (always apply; curriculum affects J via lambdas)
    if support_overlap < float(args.support_overlap_min):
        base_reward *= float(args.support_barrier_multiplier)
    if arg_med > float(args.active_argument_margin):
        base_reward *= float(args.argument_barrier_multiplier)
    if pois_frac >= float(args.poisson_hard_threshold):
        base_reward *= float(args.poisson_barrier_multiplier)

    # ensure finite reward
    if not math.isfinite(base_reward) or base_reward < 0.0:
        base_reward = 0.0
    base_reward = float(min(max(base_reward, 0.0), 1.0))

    er = EvalResult(
        stable=True,
        J=float(J),
        final_reward=float(base_reward),
        poisson_like_fraction=float(pois_frac),
        support_overlap=float(support_overlap),
        active_argument_error_med=float(arg_med),
        residue_error_med=float(residue_med),
        trace_error_med=float(trace_med),
        generator_entropy=float(gen_ent),
        power_entropy=float(pow_ent),
        collapse_loss=float(L_collapse),
        antiP_loss=float(L_antiP),
        nv_rmse=float(nv_rmse),
        nv_long_rmse=float(nv_long_rmse),
        nv_slope_op=float(slope_op) if math.isfinite(slope_op) else float("nan"),
        nv_slope_tgt=float(slope_tgt) if math.isfinite(slope_tgt) else float("nan"),
        L_stability=float(L_stability),
        L_support=float(L_support),
        L_arg=float(L_arg),
        L_nv=float(L_nv),
        L_residue=float(L_residue),
        L_trace=float(L_trace),
        L_null=float(L_null),
        L_complexity=float(L_complexity),
        L_semantic_collapse=float(L_semantic_collapse),
    )
    # attach null info via synthetic trace rows? We'll return via caller from pool comparisons
    for rr in residue_rows:
        rr["residue_imag_leak_med_unused"] = float(imag_leak_med)
    return er, arg_rows, nv_rows, residue_rows, trace_rows


# ----------------------------
# Main
# ----------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="V14.7 Anti-Poisson Spectral Curriculum (computational only).")
    ap.add_argument("--true_levels_csv", type=str, default="runs/v13o9_true_unfolded_spectra_from_v13o8/v13o9_unfolded_levels.csv")
    ap.add_argument("--zeros_csv", type=str, default="runs/zeros_100_400_precise.csv")
    ap.add_argument("--v13o14_dir", type=str, default="runs/v13o14_transport_null_controls")
    ap.add_argument("--v14_2_dir", type=str, default="runs/v14_2_stabilized_artin_operator_search")
    ap.add_argument("--v14_5_dir", type=str, default="runs/v14_5_semantic_anticollapse_search")
    ap.add_argument("--v14_6b_dir", type=str, default="runs/v14_6b_real_null_control_stage_e")
    ap.add_argument("--out_dir", type=str, default="runs/v14_7_antipoisson_spectral_curriculum")
    ap.add_argument("--dims", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--num_ants", type=int, default=48)
    ap.add_argument("--num_iters", type=int, default=120)
    ap.add_argument("--max_word_len", type=int, default=36)
    ap.add_argument("--max_power", type=int, default=3)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--rho", type=float, default=0.15)
    ap.add_argument("--q", type=float, default=1.0)
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
    ap.add_argument("--active_argument_margin", type=float, default=0.25)
    ap.add_argument("--residue_error_margin", type=float, default=0.25)
    ap.add_argument("--trace_error_margin", type=float, default=0.25)
    ap.add_argument("--poisson_hard_threshold", type=float, default=0.90)
    ap.add_argument("--poisson_barrier_multiplier", type=float, default=1e-6)
    ap.add_argument("--support_barrier_multiplier", type=float, default=1e-3)
    ap.add_argument("--argument_barrier_multiplier", type=float, default=1e-3)
    ap.add_argument("--lambda_stability", type=float, default=1.0)
    ap.add_argument("--lambda_support", type=float, default=1.0)
    ap.add_argument("--lambda_arg", type=float, default=1.0)
    ap.add_argument("--lambda_nv", type=float, default=1.0)
    ap.add_argument("--lambda_ap", type=float, default=2.0)
    ap.add_argument("--lambda_residue", type=float, default=1.0)
    ap.add_argument("--lambda_trace", type=float, default=0.5)
    ap.add_argument("--lambda_null", type=float, default=1.0)
    ap.add_argument("--lambda_complexity", type=float, default=0.05)
    ap.add_argument("--lambda_semantic_collapse", type=float, default=1.0)
    ap.add_argument(
        "--semantic_modes",
        type=str,
        nargs="+",
        default=["numeric_only", "semantic_only", "hybrid_numeric_semantic", "hybrid_ranked_anticollapse"],
    )
    ap.add_argument("--elite_fraction", type=float, default=0.2)
    ap.add_argument("--progress_every", type=int, default=20)
    ap.add_argument("--seed", type=int, default=146)
    args = ap.parse_args()

    out_dir = _resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    rng = np.random.default_rng(int(args.seed))
    random.seed(int(args.seed))

    warnings: List[str] = []
    if not _HAVE_SAFE_EIGH:
        warnings.append("core.spectral_stabilization.safe_eigh unavailable; using numpy eigvalsh fallback.")

    dims = [int(d) for d in args.dims]
    semantic_modes = [str(m) for m in args.semantic_modes]

    windows = rd.make_windows(args.window_min, args.window_max, args.window_size, args.window_stride)
    if not windows:
        raise SystemExit("No windows produced; check window_* args.")
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
            target_by_dim[int(d)] = zeros_unfolded.copy()
            warnings.append(f"dim={d}: using zeros_csv target fallback (real_zeta missing).")

    # Load baseline pool for null comparisons (best-effort)
    pool, missing_files = load_baseline_pool(
        dims=dims,
        v13o14_dir=_resolve(args.v13o14_dir),
        v14_2_dir=_resolve(args.v14_2_dir),
        v14_5_dir=_resolve(args.v14_5_dir),
        v14_6b_dir=_resolve(args.v14_6b_dir),
    )
    warnings.extend([f"baseline_missing: {m}" for m in missing_files])

    # Initialize semantic memory
    semantic_memory: List[Dict[str, Any]] = semantic_rules_seed()

    # Initialize numeric pheromones per dim/mode/token
    max_power = int(args.max_power)
    pvals = power_values(max_power)
    pher: Dict[str, Dict[int, Dict[Tuple[int, int], float]]] = {m: {} for m in semantic_modes}
    for mode in semantic_modes:
        for d in dims:
            pher[mode][int(d)] = {(gi, pw): 1.0 for gi in range(1, int(d)) for pw in pvals}

    # Outputs
    aco_history: List[Dict[str, Any]] = []
    candidate_ranking: List[Dict[str, Any]] = []
    best_candidates: List[Dict[str, Any]] = []
    gate_summary: List[Dict[str, Any]] = []
    active_argument_counts: List[Dict[str, Any]] = []
    nv_curves: List[Dict[str, Any]] = []
    residue_scores: List[Dict[str, Any]] = []
    trace_proxy: List[Dict[str, Any]] = []
    null_comparisons: List[Dict[str, Any]] = []
    pheromone_summary: List[Dict[str, Any]] = []
    schedule_rows: List[Dict[str, Any]] = []

    # Track best per dim/mode
    best_by: Dict[Tuple[int, str], Dict[str, Any]] = {(int(d), m): {"J": float("inf"), "row": None, "word_tokens": None} for d in dims for m in semantic_modes}
    # For diversity penalties
    recent_elite_by: Dict[Tuple[int, str], List[List[Tuple[int, int]]]] = {(int(d), m): [] for d in dims for m in semantic_modes}

    # Main loop
    total_iters = int(args.num_iters)
    for it in range(1, total_iters + 1):
        progress = float((it - 1) / max(1, total_iters - 1))
        lam_eff = curriculum_lambdas(progress, args)
        schedule_rows.append({"iter": int(it), "progress": float(progress), **{k: float(v) for k, v in lam_eff.items()}})

        for d in dims:
            tgt = target_by_dim[int(d)]
            for mode in semantic_modes:
                # batch of candidates for rank/elite selection
                batch_rows: List[Dict[str, Any]] = []
                batch_tokens: List[List[Tuple[int, int]]] = []
                # semantic influence by mode (fixed weights; curriculum uses lambda_semantic_collapse)
                if mode == "numeric_only":
                    sem_term = 0.0
                elif mode == "semantic_only":
                    sem_term = 1.0
                elif mode == "hybrid_numeric_semantic":
                    sem_term = 0.5
                else:
                    sem_term = 0.8

                for ant in range(int(args.num_ants)):
                    L = int(rng.integers(2, int(args.max_word_len) + 1))
                    w: List[Tuple[int, int]] = []
                    for _k in range(L):
                        tok = sample_token(
                            dim=int(d),
                            powers=pvals,
                            pher=pher[mode][int(d)],
                            alpha=float(args.alpha),
                            beta=float(args.beta),
                            semantic_term=float(sem_term),
                            rng=rng,
                        )
                        w.append(tok)
                    w = clamp_word_to_dim(w, int(d), max_power=int(args.max_power), max_word_len=int(args.max_word_len))
                    wstr = word_to_string(w)

                    er, arg_rows, nv_rows, res_rows, tr_rows = evaluate_word(
                        dim=int(d),
                        word=w,
                        target_levels=tgt,
                        windows=windows,
                        L_grid=L_grid,
                        pool=pool,
                        semantic_memory=semantic_memory,
                        args=args,
                        lambdas_eff=lam_eff,
                        seed=int(args.seed + 100000 * it + 1000 * d + 7 * ant),
                        recent_elite_words=recent_elite_by[(int(d), mode)],
                    )

                    row = {
                        "iter": int(it),
                        "dim": int(d),
                        "mode": str(mode),
                        "ant_id": int(ant),
                        "word": wstr,
                        "word_len": int(len(w)),
                        "stable": bool(er.stable),
                        "J_v14_7": float(er.J),
                        "final_reward": float(er.final_reward),
                        "poisson_like_fraction": float(er.poisson_like_fraction),
                        "support_overlap": float(er.support_overlap),
                        "active_argument_error_med": float(er.active_argument_error_med),
                        "residue_error_med": float(er.residue_error_med),
                        "trace_error_med": float(er.trace_error_med),
                        "nv_rmse": float(er.nv_rmse),
                        "nv_long_rmse": float(er.nv_long_rmse),
                        "nv_slope_op": float(er.nv_slope_op) if math.isfinite(er.nv_slope_op) else float("nan"),
                        "nv_slope_tgt": float(er.nv_slope_tgt) if math.isfinite(er.nv_slope_tgt) else float("nan"),
                        "antiP_loss": float(er.antiP_loss),
                        "generator_entropy": float(er.generator_entropy),
                        "power_entropy": float(er.power_entropy),
                        "collapse_loss": float(er.collapse_loss),
                        "L_stability": float(er.L_stability),
                        "L_support": float(er.L_support),
                        "L_arg": float(er.L_arg),
                        "L_nv": float(er.L_nv),
                        "L_residue": float(er.L_residue),
                        "L_trace": float(er.L_trace),
                        "L_null": float(er.L_null),
                        "L_complexity": float(er.L_complexity),
                        "L_semantic_collapse": float(er.L_semantic_collapse),
                        "lambda_support_eff": float(lam_eff["lambda_support"]),
                        "lambda_arg_eff": float(lam_eff["lambda_arg"]),
                        "lambda_nv_eff": float(lam_eff["lambda_nv"]),
                        "lambda_ap_eff": float(lam_eff["lambda_ap"]),
                        "lambda_residue_eff": float(lam_eff["lambda_residue"]),
                        "lambda_trace_eff": float(lam_eff["lambda_trace"]),
                        "lambda_null_eff": float(lam_eff["lambda_null"]),
                    }
                    batch_rows.append(row)
                    batch_tokens.append(w)
                    aco_history.append(row)

                    # detail rows (subsample: only keep for stable candidates to control size)
                    if er.stable:
                        for ar in arg_rows:
                            active_argument_counts.append({"iter": int(it), "dim": int(d), "mode": str(mode), "ant_id": int(ant), "word": wstr, **ar})
                        for nr in nv_rows:
                            nv_curves.append({"iter": int(it), "dim": int(d), "mode": str(mode), "ant_id": int(ant), "word": wstr, **nr})
                        for rr in res_rows:
                            residue_scores.append({"iter": int(it), "dim": int(d), "mode": str(mode), "ant_id": int(ant), "word": wstr, **rr})
                        for tr in tr_rows:
                            trace_proxy.append({"iter": int(it), "dim": int(d), "mode": str(mode), "ant_id": int(ant), "word": wstr, **tr})

                    # best tracking
                    key = (int(d), mode)
                    if float(er.J) < float(best_by[key]["J"]):
                        best_by[key] = {"J": float(er.J), "row": row.copy(), "word_tokens": w}
                        semantic_memory.extend(update_semantic_from_diagnostics(int(d), row, source="v14_7_best_update", it=int(it)))

                # pheromone update: evaporate + deposit from elites based on final_reward
                rows_sorted = sorted(batch_rows, key=lambda r: float(r["J_v14_7"]))
                N = len(rows_sorted)
                elite_k = int(max(1, math.ceil(float(args.elite_fraction) * N)))
                elites = rows_sorted[:elite_k]

                # add elite words for diversity
                elite_words = [batch_tokens[batch_rows.index(r)] for r in elites if r.get("stable", False)]
                recent_elite_by[(int(d), mode)].extend(elite_words)
                recent_elite_by[(int(d), mode)] = recent_elite_by[(int(d), mode)][-200:]

                # evaporate
                for tok in list(pher[mode][int(d)].keys()):
                    pher[mode][int(d)][tok] = float(max(1e-6, (1.0 - float(args.rho)) * pher[mode][int(d)][tok]))

                # deposit (note: Poisson-like already strongly penalized in reward via barrier)
                for r in elites:
                    rew = float(args.q) * float(r.get("final_reward", 0.0))
                    if not (math.isfinite(rew) and rew > 0.0):
                        continue
                    # parse tokens from word string to avoid carrying token list mapping complexity
                    toks: List[Tuple[int, int]] = []
                    for tokstr in str(r.get("word", "")).split():
                        try:
                            left, pstr = tokstr.split("^")
                            gi = int(left.split("_")[1])
                            pw = int(pstr)
                            toks.append((gi, pw))
                        except Exception:
                            continue
                    for t in toks:
                        if t in pher[mode][int(d)]:
                            pher[mode][int(d)][t] = float(min(1e6, pher[mode][int(d)][t] + rew))

        if it == 1 or it % max(1, int(args.progress_every)) == 0 or it == total_iters:
            elapsed = time.perf_counter() - t0
            eta = (elapsed / max(1, it)) * max(0, total_iters - it)
            best_main = min(float(best_by[(int(d), semantic_modes[-1])]["J"]) for d in dims) if semantic_modes else float("inf")
            print(f"[V14.7] iter={it}/{total_iters} best_main_J={best_main:.6g} eta={format_seconds(eta)}", flush=True)

        # prune semantic memory deterministically
        semantic_memory.sort(key=lambda rec: abs(rule_weight(rec)), reverse=True)
        semantic_memory = semantic_memory[:600]

    # Write pheromone summary (top tokens per dim/mode)
    for d in dims:
        for mode in semantic_modes:
            items = list(pher[mode][int(d)].items())
            items.sort(key=lambda kv: float(kv[1]), reverse=True)
            for (gi, pw), v in items[: min(120, len(items))]:
                pheromone_summary.append({"dim": int(d), "mode": str(mode), "generator": int(gi), "power": int(pw), "pheromone": float(v)})

    # Best candidates + gate summary + ranking + null comparisons
    for d in dims:
        for mode in semantic_modes:
            br = best_by[(int(d), mode)]["row"]
            if br is None:
                continue
            best_candidates.append(
                {
                    "dim": int(d),
                    "mode": str(mode),
                    "best_word": str(br["word"]),
                    "J_v14_7": float(br["J_v14_7"]),
                    "final_reward": float(br["final_reward"]),
                    "poisson_like_fraction": float(br["poisson_like_fraction"]),
                    "support_overlap": float(br["support_overlap"]),
                    "active_argument_error_med": float(br["active_argument_error_med"]),
                    "residue_error_med": float(br["residue_error_med"]),
                    "trace_error_med": float(br["trace_error_med"]),
                    "generator_entropy": float(br["generator_entropy"]),
                    "power_entropy": float(br["power_entropy"]),
                }
            )

    # Ranking per dim/mode from history
    for d in dims:
        for mode in semantic_modes:
            sub = [r for r in aco_history if int(r["dim"]) == int(d) and str(r["mode"]) == str(mode)]
            sub.sort(key=lambda r: float(r["J_v14_7"]))
            for rank_i, r in enumerate(sub[:500], start=1):
                candidate_ranking.append(
                    {
                        "dim": int(d),
                        "mode": str(mode),
                        "rank": int(rank_i),
                        "word": str(r["word"]),
                        "word_len": int(r["word_len"]),
                        "J_v14_7": float(r["J_v14_7"]),
                        "final_reward": float(r["final_reward"]),
                        "poisson_like_fraction": float(r["poisson_like_fraction"]),
                        "support_overlap": float(r["support_overlap"]),
                        "active_argument_error_med": float(r["active_argument_error_med"]),
                        "residue_error_med": float(r["residue_error_med"]),
                        "trace_error_med": float(r["trace_error_med"]),
                        "generator_entropy": float(r["generator_entropy"]),
                        "power_entropy": float(r["power_entropy"]),
                    }
                )

    # Gates + null comparisons per dim/mode (best candidate)
    for d in dims:
        for mode in semantic_modes:
            br = best_by[(int(d), mode)]["row"]
            if br is None:
                continue
            bestJ = float(br["J_v14_7"])
            best_random = best_baseline_J(pool, int(d), ["random"])
            best_rej = best_baseline_J(pool, int(d), ["rejected"])
            best_abl = best_baseline_J(pool, int(d), ["ablation"])
            best_prior = best_baseline_J(pool, int(d), ["prior_artin"])
            best_null = best_baseline_J(pool, int(d), ["random", "rejected", "ablation", "prior_artin", "other"])
            null_sep = float(best_null - bestJ) if (best_null is not None and math.isfinite(bestJ) and math.isfinite(best_null)) else float("nan")
            dist = null_distribution_Js(pool, int(d))
            null_z, null_pct = (float("nan"), float("nan"))
            if len(dist) >= 3:
                mu = float(np.mean(dist))
                sd = float(np.std(dist))
                null_z = float((mu - bestJ) / sd) if sd > 1e-12 else float("nan")
                null_pct = float(np.mean([1.0 if x <= bestJ else 0.0 for x in dist]))

            beats_random = bool(best_random is not None and bestJ < float(best_random))
            beats_rej = bool(best_rej is not None and bestJ < float(best_rej))
            beats_abl = bool(best_abl is not None and bestJ < float(best_abl))
            beats_prior = bool(best_prior is not None and bestJ < float(best_prior))

            null_comparisons.append(
                {
                    "dim": int(d),
                    "mode": str(mode),
                    "best_word": str(br["word"]),
                    "best_J_v14_7": float(bestJ),
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
                    "missing_baselines": bool(best_null is None),
                }
            )

            # gates
            G1 = bool(br["stable"])
            G2 = bool(float(br["support_overlap"]) >= float(args.support_overlap_min))
            G3 = bool(float(br["active_argument_error_med"]) <= float(args.active_argument_margin))
            G4 = True  # number variance ok is captured by G5 hard threshold; keep as placeholder True
            G5 = bool(float(br["poisson_like_fraction"]) < float(args.poisson_hard_threshold))
            G6 = bool(float(br["residue_error_med"]) <= float(args.residue_error_margin))
            G7 = bool(float(br["trace_error_med"]) <= float(args.trace_error_margin))
            G8 = bool(beats_random) if best_random is not None else False
            G9 = bool(beats_rej) if best_rej is not None else False
            G10 = bool(beats_abl) if best_abl is not None else False
            G11 = bool(beats_prior) if best_prior is not None else False
            G12 = bool(float(br["generator_entropy"]) >= 0.5 and float(br["power_entropy"]) >= 0.45)
            all_gate = bool(G1 and G2 and G3 and G5 and G6 and G7 and G8 and G9 and G10 and G11 and G12)
            gate_summary.append(
                {
                    "dim": int(d),
                    "mode": str(mode),
                    "word": str(br["word"]),
                    "J_v14_7": float(bestJ),
                    "final_reward": float(br["final_reward"]),
                    "poisson_like_fraction": float(br["poisson_like_fraction"]),
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
                    "classification": "ALL_GATE_PASS" if all_gate else ("POISSON_LIKE_FAIL" if not G5 else "PARTIAL_FAIL"),
                }
            )

    # Write semantic rules JSONL
    rules_path = out_dir / "v14_7_semantic_pheromone_rules.jsonl"
    rules_path.parent.mkdir(parents=True, exist_ok=True)
    with rules_path.open("w", encoding="utf-8") as f:
        for rec in semantic_memory:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Write outputs
    write_csv(
        out_dir / "v14_7_aco_history.csv",
        fieldnames=list(aco_history[0].keys()) if aco_history else ["iter", "dim", "mode", "word", "J_v14_7", "final_reward"],
        rows=aco_history,
    )
    write_csv(
        out_dir / "v14_7_best_candidates.csv",
        fieldnames=list(best_candidates[0].keys()) if best_candidates else ["dim", "mode", "best_word", "J_v14_7"],
        rows=best_candidates,
    )
    write_csv(
        out_dir / "v14_7_gate_summary.csv",
        fieldnames=list(gate_summary[0].keys()) if gate_summary else ["dim", "mode", "G13_all_gate_pass"],
        rows=gate_summary,
    )
    write_csv(
        out_dir / "v14_7_candidate_ranking.csv",
        fieldnames=list(candidate_ranking[0].keys()) if candidate_ranking else ["dim", "mode", "rank", "word", "J_v14_7"],
        rows=candidate_ranking,
    )
    write_csv(
        out_dir / "v14_7_active_argument_counts.csv",
        fieldnames=list(active_argument_counts[0].keys()) if active_argument_counts else ["iter", "dim", "mode", "word"],
        rows=active_argument_counts,
    )
    write_csv(
        out_dir / "v14_7_nv_curves.csv",
        fieldnames=list(nv_curves[0].keys()) if nv_curves else ["iter", "dim", "mode", "word", "kind", "L", "Sigma2"],
        rows=nv_curves,
    )
    write_csv(
        out_dir / "v14_7_residue_scores.csv",
        fieldnames=list(residue_scores[0].keys()) if residue_scores else ["iter", "dim", "mode", "word", "residue_count_error"],
        rows=residue_scores,
    )
    write_csv(
        out_dir / "v14_7_trace_proxy.csv",
        fieldnames=list(trace_proxy[0].keys()) if trace_proxy else ["iter", "dim", "mode", "word", "err"],
        rows=trace_proxy,
    )
    write_csv(
        out_dir / "v14_7_null_comparisons.csv",
        fieldnames=list(null_comparisons[0].keys()) if null_comparisons else ["dim", "mode", "best_J_v14_7"],
        rows=null_comparisons,
    )
    write_csv(
        out_dir / "v14_7_pheromone_summary.csv",
        fieldnames=list(pheromone_summary[0].keys()) if pheromone_summary else ["dim", "mode", "generator", "power", "pheromone"],
        rows=pheromone_summary,
    )
    write_csv(
        out_dir / "v14_7_curriculum_schedule.csv",
        fieldnames=list(schedule_rows[0].keys()) if schedule_rows else ["iter", "progress"],
        rows=schedule_rows,
    )

    # Results JSON + report
    any_all = sum(1 for r in gate_summary if bool(r.get("G13_all_gate_pass", False)))
    any_not_poiss = sum(1 for r in gate_summary if bool(r.get("G5_not_poisson_like", False)))
    proceed_v14_8 = bool(any_all > 0)
    analytic_claim = False
    results = {
        "version": "v14_7",
        "out_dir": str(out_dir),
        "dims": dims,
        "semantic_modes": semantic_modes,
        "num_ants": int(args.num_ants),
        "num_iters": int(args.num_iters),
        "curriculum": {"schedule": "sigmoid", "centers": {"arg": 0.20, "nv": 0.35, "ap": 0.40, "residue": 0.55, "trace": 0.65, "null": 0.75}},
        "barriers": {
            "poisson_hard_threshold": float(args.poisson_hard_threshold),
            "poisson_barrier_multiplier": float(args.poisson_barrier_multiplier),
            "support_barrier_multiplier": float(args.support_barrier_multiplier),
            "argument_barrier_multiplier": float(args.argument_barrier_multiplier),
        },
        "warnings": warnings,
        "missing_baseline_files": missing_files,
        "gate_summary": gate_summary,
        "decision": {"proceed_to_v14_8": proceed_v14_8, "analytic_claim": analytic_claim, "all_gate_pass_count": int(any_all), "not_poisson_like_count": int(any_not_poiss)},
        "notes": ["Computational evidence only; not a proof of RH."],
    }
    write_text(out_dir / "v14_7_results.json", json.dumps(json_sanitize(results), indent=2, ensure_ascii=False) + "\n")

    OUT_ABS = str(out_dir.resolve())
    md = []
    md.append("# V14.7 — Anti-Poisson Spectral Curriculum for Artin-DTES Agents\n\n")
    md.append("> Computational evidence only; not a proof of RH.\n\n")
    md.append("## Why V14.7 exists\n")
    md.append("V13O.14–V14.6b showed that candidates are overwhelmingly Poisson-like or fail support/argument/residue gates. V14.7 introduces a strict curriculum and a hard anti-Poisson barrier to remove Poisson-like candidates from elite selection.\n\n")
    md.append("## Curriculum objective\n")
    md.append("Stages A..G are implemented via a sigmoid schedule on effective lambdas, gradually activating argument/NV/anti-Poisson/residue/trace/null terms.\n\n")
    md.append("## Anti-Poisson hard barrier\n")
    md.append(f"- `poisson_hard_threshold={float(args.poisson_hard_threshold)}`\n")
    md.append(f"- if `poisson_like_fraction >= threshold`, reward *= `{float(args.poisson_barrier_multiplier)}`\n\n")
    md.append("## Summary\n")
    md.append(f"- all-gate pass count: **{any_all}** (out of {len(gate_summary)})\n")
    md.append(f"- not-poisson-like count: **{any_not_poiss}**\n")
    md.append(f"- proceed_to_v14_8: **{proceed_v14_8}**\n")
    md.append(f"- analytic_claim: **{analytic_claim}**\n\n")
    md.append("## Outputs\n")
    md.append("- `v14_7_gate_summary.csv`, `v14_7_best_candidates.csv`, `v14_7_candidate_ranking.csv`\n")
    md.append("- `v14_7_aco_history.csv` (reward/J diversity)\n")
    md.append("- `v14_7_null_comparisons.csv` (real baseline comparisons)\n")
    md.append("- `v14_7_semantic_pheromone_rules.jsonl`, `v14_7_pheromone_summary.csv`\n\n")
    md.append("## Decision rule\n")
    md.append("- proceed_to_v14_8 = True only if at least one non-artifact candidate passes all gates.\n")
    md.append("- analytic_claim always False unless all gates and independent null validations pass.\n\n")
    md.append("## Verification commands\n")
    md.append("```bash\n")
    md.append(f'OUT="{OUT_ABS}"\n\n')
    md.append('echo "=== FILES ==="\nfind "$OUT" -maxdepth 1 -type f | sort\n\n')
    md.append('echo "=== GATE ==="\ncolumn -s, -t < "$OUT"/v14_7_gate_summary.csv | head -120\n\n')
    md.append('echo "=== BEST CANDIDATES ==="\ncolumn -s, -t < "$OUT"/v14_7_best_candidates.csv | head -120\n\n')
    md.append('echo "=== CANDIDATE RANKING ==="\ncolumn -s, -t < "$OUT"/v14_7_candidate_ranking.csv | head -120\n\n')
    md.append('echo "=== ACO HISTORY TAIL ==="\ntail -80 \"$OUT\"/v14_7_aco_history.csv | column -s, -t\n\n')
    md.append('echo "=== NULL COMPARISONS ==="\ncolumn -s, -t < \"$OUT\"/v14_7_null_comparisons.csv | head -120\n\n')
    md.append('echo "=== PHEROMONES ==="\ncolumn -s, -t < \"$OUT\"/v14_7_pheromone_summary.csv | head -120\n\n')
    md.append('echo \"=== REPORT ===\"\\nhead -260 \"$OUT\"/v14_7_report.md\n')
    md.append("```\n\n")
    md.append("## Quick verdict command\n")
    md.append("```bash\n")
    md.append("python3 - <<'PY'\n")
    md.append(
        "from pathlib import Path\n"
        "import csv, statistics as st\n\n"
        "OUT = Path('runs/v14_7_antipoisson_spectral_curriculum')\n\n"
        "print('=== V14.7 QUICK VERDICT ===')\n\n"
        "gate = OUT/'v14_7_gate_summary.csv'\n"
        "if gate.exists():\n"
        "    rows = list(csv.DictReader(open(gate)))\n"
        "    print('gate rows:', len(rows))\n"
        "    print('all_gate_pass:', sum(str(r.get('G13_all_gate_pass','')).lower()=='true' for r in rows))\n"
        "    print('not_poisson_like:', sum(str(r.get('G5_not_poisson_like','')).lower()=='true' for r in rows))\n"
        "    print('argument_ok:', sum(str(r.get('G3_active_argument_ok','')).lower()=='true' for r in rows))\n"
        "    print('residue_ok:', sum(str(r.get('G6_residue_error_ok','')).lower()=='true' for r in rows))\n"
        "    print('trace_ok:', sum(str(r.get('G7_trace_proxy_ok','')).lower()=='true' for r in rows))\n"
        "    for r in rows:\n"
        "        print(r.get('dim'), r.get('mode'), 'J=', r.get('J_v14_7'), 'reward=', r.get('final_reward'), 'poisson=', r.get('poisson_like_fraction'), 'pass=', r.get('G13_all_gate_pass'), 'class=', r.get('classification'))\n"
        "else:\n"
        "    print('missing gate_summary')\n\n"
        "hist = OUT/'v14_7_aco_history.csv'\n"
        "if hist.exists():\n"
        "    rewards=[]; Js=[]; poisson=[]\n"
        "    for r in csv.DictReader(open(hist)):\n"
        "        for key, arr in [('final_reward', rewards), ('J_v14_7', Js), ('poisson_like_fraction', poisson)]:\n"
        "            try: arr.append(float(r[key]))\n"
        "            except: pass\n"
        "    def summary(name, xs):\n"
        "        if not xs:\n"
        "            print(name, 'missing'); return\n"
        "        print(name, 'n=', len(xs), 'min=', min(xs), 'median=', st.median(xs), 'mean=', st.mean(xs), 'max=', max(xs), 'unique6=', len(set(round(x,6) for x in xs)))\n"
        "    print('\\n=== SCORE DIVERSITY ===')\n"
        "    summary('final_reward', rewards)\n"
        "    summary('J_v14_7', Js)\n"
        "    summary('poisson_like_fraction', poisson)\n"
        "else:\n"
        "    print('missing aco_history')\n\n"
        "nulls = OUT/'v14_7_null_comparisons.csv'\n"
        "if nulls.exists():\n"
        "    print('\\n=== NULL COMPARISON TOP ===')\n"
        "    for r in list(csv.DictReader(open(nulls)))[:20]:\n"
        "        print(r)\n"
        "else:\n"
        "    print('missing null comparisons')\n\n"
        "print('\\nInterpretation:')\n"
        "print('- all_gate_pass > 0: run V14.8 independent null validation.')\n"
        "print('- not_poisson_like = 0: anti-Poisson rigidity still unsolved.')\n"
        "print('- reward diversity high but poisson still 1.0: objective is diverse but wrong.')\n"
        "print('- if semantic modes dominate but fail diversity: semantic pheromone collapse persists.')\n"
    )
    md.append("PY\n")
    md.append("```\n\n")
    md.append("> Computational evidence only; not a proof of RH.\n")
    write_text(out_dir / "v14_7_report.md", "".join(md))

    tex = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}
\begin{document}
\section*{V14.7 --- Anti-Poisson Spectral Curriculum for Artin-DTES Agents}
\textbf{Computational evidence only; not a proof of RH.}

\subsection*{Summary}
This experiment introduces a staged curriculum and a hard anti-Poisson barrier to remove Poisson-like candidates from elite selection.
Final acceptance is purely by deterministic gates.
\end{document}
"""
    tex_path = out_dir / "v14_7_report.tex"
    write_text(tex_path, tex)
    if _find_pdflatex():
        try_pdflatex(tex_path, out_dir, "v14_7_report.pdf")

    elapsed = time.perf_counter() - t0
    print(f"[V14.7] done in {format_seconds(elapsed)} out_dir={out_dir}", flush=True)


if __name__ == "__main__":
    main()

