from __future__ import annotations

"""
Residue / argument-principle inspired diagnostics (computational proxy).

Computational evidence only; not a proof of the Riemann Hypothesis.

This module is used by scripts/run_v13o11_residue_argument_principle.py.
It intentionally avoids any claim that these proxies implement the true argument principle
for zeta; they are numerical diagnostics over finite unfolded level lists.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import pandas as pd  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit("This module requires pandas. Please install pandas and retry.") from e


def finite_series(x: "pd.Series") -> "pd.Series":
    return pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan)


def unfold_to_mean_spacing_one(x: np.ndarray) -> np.ndarray:
    x = np.sort(np.asarray(x, dtype=np.float64).reshape(-1))
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.zeros((0,), dtype=np.float64)
    g = np.diff(x)
    g = g[np.isfinite(g) & (g > 0.0)]
    ms = float(np.mean(g)) if g.size else 1.0
    ms = ms if (math.isfinite(ms) and ms > 0.0) else 1.0
    return ((x - float(x[0])) / max(ms, 1e-12)).astype(np.float64, copy=False)


def load_true_levels_csv(path: Path, *, dims_keep: Sequence[int]) -> Tuple[Optional["pd.DataFrame"], List[str]]:
    """
    Load explicit unfolded levels from V13O.9 source-format OR V13O.8 kind-format.
    Returns normalized dataframe with:
      dim,V_mode,word_group,target_group,seed,source,level_index,unfolded_level
    """
    warns: List[str] = []
    if not path.is_file():
        return None, [f"true_levels_csv missing: {path}"]
    try:
        df = pd.read_csv(path)
    except Exception as e:
        return None, [f"true_levels_csv read failed: {e!r}"]
    df.columns = [str(c).strip() for c in df.columns]
    if df.empty:
        return None, ["true_levels_csv is empty"]

    if "source" not in df.columns and "kind" in df.columns:
        df = df.copy()
        df["source"] = df["kind"].astype(str).str.strip()

    required = ["dim", "V_mode", "word_group", "target_group", "seed", "source", "level_index", "unfolded_level"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return None, [f"true_levels_csv missing required columns: {missing}"]

    out = df[required].copy()
    out["dim"] = pd.to_numeric(out["dim"], errors="coerce").astype("Int64")
    out["seed"] = pd.to_numeric(out["seed"], errors="coerce").astype("Int64")
    out["level_index"] = pd.to_numeric(out["level_index"], errors="coerce").astype("Int64")
    for k in ("V_mode", "word_group", "target_group", "source"):
        out[k] = out[k].astype(str).str.strip()
    out["unfolded_level"] = finite_series(out["unfolded_level"])
    out = out.dropna(subset=["dim", "seed", "level_index", "unfolded_level"])

    dims_keep_set = {int(d) for d in dims_keep}
    out = out[out["dim"].astype(int).isin(dims_keep_set)]
    if out.empty:
        return None, ["true_levels_csv has no rows after filtering dims / NaNs"]

    bad = sorted({s for s in out["source"].astype(str).unique().tolist() if s not in ("operator", "target", "control")})
    if bad:
        warns.append(f"nonstandard source labels: {bad} (allowed: operator/target/control)")

    out = out.sort_values(["dim", "V_mode", "word_group", "target_group", "seed", "source", "level_index"], ascending=True)
    return out, warns


def load_zeros_csv(path: Path) -> Tuple[np.ndarray, List[str]]:
    """
    Load precise zeta zeros from CSV/TXT.

    Accepts columns named: gamma, zero, t, imag, height (case-insensitive),
    or falls back to first numeric column.
    """
    warns: List[str] = []
    if not path.is_file():
        raise FileNotFoundError(str(path))
    try:
        # try CSV first
        df = pd.read_csv(path)
        df.columns = [str(c).strip() for c in df.columns]
    except Exception:
        # try whitespace-delimited
        df = pd.read_csv(path, delim_whitespace=True, header=None)
        df.columns = [f"col_{i}" for i in range(len(df.columns))]
        warns.append("zeros_csv parsed as whitespace-delimited without headers")

    cols_l = {str(c).strip().lower(): str(c) for c in df.columns}
    cand_names = ["gamma", "zero", "t", "imag", "height"]
    col = None
    for nm in cand_names:
        if nm in cols_l:
            col = cols_l[nm]
            break
    if col is None:
        # first numeric-ish column
        best = None
        best_rate = -1.0
        for c in df.columns:
            s = finite_series(df[c])
            rate = float(s.notna().mean())
            if rate > best_rate:
                best_rate = rate
                best = str(c)
        col = best
        warns.append(f"zeros column inferred as: {col}")
    if col is None:
        return np.asarray([], dtype=np.float64), warns + ["no usable numeric column found"]

    x = finite_series(df[col]).dropna().astype(float).to_numpy(dtype=np.float64)
    x = x[np.isfinite(x)]
    x = np.sort(x)
    return x, warns


def make_windows(window_min: float, window_max: float, window_size: float, window_stride: float) -> List[Tuple[float, float]]:
    a0 = float(window_min)
    bmax = float(window_max)
    w = float(window_size)
    s = float(window_stride)
    if not (math.isfinite(a0) and math.isfinite(bmax) and math.isfinite(w) and math.isfinite(s)):
        return []
    if w <= 0.0 or s <= 0.0 or bmax <= a0:
        return []
    out: List[Tuple[float, float]] = []
    a = a0
    while a + w <= bmax + 1e-12:
        out.append((a, a + w))
        a += s
        if len(out) > 50000:
            break
    return out


def count_in_window(levels: np.ndarray, a: float, b: float) -> int:
    x = np.asarray(levels, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return 0
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0
    lo = float(a)
    hi = float(b)
    left = int(np.searchsorted(x, lo, side="left"))
    right = int(np.searchsorted(x, hi, side="right"))
    return int(max(0, right - left))


def argument_principle_proxy(levels_operator: np.ndarray, levels_target: np.ndarray, a: float, b: float) -> Tuple[float, float, int, int]:
    n_op = count_in_window(levels_operator, a, b)
    n_tg = count_in_window(levels_target, a, b)
    err = abs(int(n_op) - int(n_tg))
    norm = float(err) / float(max(1, int(n_tg)))
    return float(err), float(norm), int(n_op), int(n_tg)


def resolvent_cauchy_sum(z: np.ndarray, levels: np.ndarray, *, eta: float) -> np.ndarray:
    """
    R(z) = sum_k 1 / (z - lambda_k + i*eta)
    Vectorized over z; levels 1D.
    """
    lam = np.asarray(levels, dtype=np.float64).reshape(-1)
    lam = lam[np.isfinite(lam)]
    zc = np.asarray(z, dtype=np.complex128).reshape(-1)
    if lam.size == 0 or zc.size == 0:
        return np.zeros_like(zc, dtype=np.complex128)
    denom = zc[:, None] - lam[None, :] + 1j * float(eta)
    return np.sum(1.0 / denom, axis=1)


def rectangle_contour(a: float, b: float, *, height: float, n_per_edge: int) -> np.ndarray:
    """
    Rectangle around [a,b] with imaginary extent ±height.
    Returns closed contour points (last point repeats first).
    """
    a = float(a)
    b = float(b)
    h = float(height)
    n = int(max(8, n_per_edge))
    # edges (exclude last point per edge to avoid duplicates, then close)
    bottom = np.linspace(a, b, n, endpoint=False) + (-1j * h)
    right = (b + 1j * np.linspace(-h, h, n, endpoint=False))
    top = np.linspace(b, a, n, endpoint=False) + (1j * h)
    left = (a + 1j * np.linspace(h, -h, n, endpoint=False))
    pts = np.concatenate([bottom, right, top, left, bottom[:1]])
    return pts.astype(np.complex128, copy=False)


def contour_integral_trapezoid(z: np.ndarray, fz: np.ndarray) -> complex:
    z = np.asarray(z, dtype=np.complex128).reshape(-1)
    fz = np.asarray(fz, dtype=np.complex128).reshape(-1)
    if z.size < 2 or fz.size != z.size:
        return 0.0 + 0.0j
    dz = z[1:] - z[:-1]
    return complex(np.sum(0.5 * (fz[:-1] + fz[1:]) * dz))


def residue_proxy_count(levels: np.ndarray, a: float, b: float, *, eta: float, n_contour_points: int) -> complex:
    """
    I = (1/(2π i)) ∮ R(z) dz where R is the resolvent-like Cauchy sum.
    Returns complex I.
    """
    # choose contour height somewhat larger than eta and window size
    w = float(b - a)
    h = max(4.0 * float(eta), 0.25 * max(w, 1e-6), 0.25)
    z = rectangle_contour(a, b, height=h, n_per_edge=int(max(8, n_contour_points // 4)))
    fz = resolvent_cauchy_sum(z, levels, eta=float(eta))
    integ = contour_integral_trapezoid(z, fz)
    return complex(integ / (2.0 * math.pi * 1j))


def trace_formula_proxy(levels: np.ndarray, *, center: float, sigma: float) -> float:
    """
    S(σ) = sum_k exp(-(lambda_k - c)^2/(2σ^2))
    """
    x = np.asarray(levels, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    s = float(sigma)
    if not (math.isfinite(s) and s > 0.0):
        return float("nan")
    c = float(center)
    return float(np.sum(np.exp(-((x - c) ** 2) / (2.0 * s * s))))


@dataclass(frozen=True)
class WindowDiagnostics:
    a: float
    b: float
    N_error: float
    N_error_norm: float
    N_operator: int
    N_target: int
    I_operator: complex
    I_target: complex
    residue_count_error: float
    residue_imag_leak: float
    residue_mass_error: float

