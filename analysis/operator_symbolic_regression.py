#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.artin_operator import hyperbolic_distance_matrix, sample_domain
from core.spectral_stabilization import safe_eigh, stable_spectral_loss


DT = np.float64


def _read_lengths_csv(path: Path) -> np.ndarray:
    if not path.exists():
        return np.array([], dtype=DT)
    arr = np.loadtxt(str(path), delimiter=",", skiprows=1, usecols=(0,), dtype=DT)
    arr = np.asarray(arr, dtype=DT).reshape(-1)
    arr = arr[np.isfinite(arr)]
    arr = arr[arr > 0.0]
    return arr


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _write_text(path: Path, s: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(s, encoding="utf-8")


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=DT).reshape(-1)
    mu = float(np.mean(x)) if x.size else 0.0
    sd = float(np.std(x)) if x.size else 1.0
    if sd < 1e-12:
        sd = 1.0
    return (x - mu) / sd


def _feature_names(scales: List[float]) -> List[str]:
    base = ["bias", "diag", "d", "d2", "exp(-d2)", "local_density"]
    for k in scales:
        base.append(f"exp(-d2/{k}^2)")
    base += ["g_mean", "g_std", "g_p50", "g_p90"]
    return base


def build_dataset(
    H: np.ndarray,
    z: np.ndarray,
    lengths: np.ndarray,
    *,
    eps_density: float = 0.6,
    kernel_scales: List[float] = None,
    max_samples: int = 50_000,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if kernel_scales is None:
        kernel_scales = [0.1, 0.3, 1.0]
    kernel_scales = [float(s) for s in kernel_scales]

    H = np.asarray(H, dtype=DT)
    n = int(H.shape[0])
    d = hyperbolic_distance_matrix(z, z).astype(DT, copy=False)
    d2 = d * d

    # local density: sum_j exp(-d(i,j)^2 / eps_density)
    A = np.exp(-d2 / (float(eps_density) + 1e-12))
    np.fill_diagonal(A, 0.0)
    density = np.sum(A, axis=1).astype(DT, copy=False)
    density = (density - float(np.mean(density))) / (float(np.std(density)) + 1e-12)

    # geodesic-length global features (available even without per-(i,j) path attribution)
    g_mean = float(np.mean(lengths)) if lengths.size else 0.0
    g_std = float(np.std(lengths)) if lengths.size else 1.0
    g_p50 = float(np.percentile(lengths, 50.0)) if lengths.size else 0.0
    g_p90 = float(np.percentile(lengths, 90.0)) if lengths.size else 0.0

    # sample indices
    rng = np.random.default_rng(int(seed))
    total = n * n
    m = int(min(max_samples, total))
    idx = rng.choice(total, size=m, replace=False) if m < total else np.arange(total, dtype=np.int64)
    ii = (idx // n).astype(np.int64, copy=False)
    jj = (idx % n).astype(np.int64, copy=False)

    di = d[ii, jj]
    di2 = d2[ii, jj]
    diag = (ii == jj).astype(DT, copy=False)
    dens = density[ii]

    feats: List[np.ndarray] = []
    feats.append(np.ones((m,), dtype=DT))
    feats.append(diag)
    feats.append(di)
    feats.append(di2)
    feats.append(np.exp(-di2))
    feats.append(dens)
    for k in kernel_scales:
        feats.append(np.exp(-di2 / (k * k + 1e-12)))

    feats.append(np.full((m,), g_mean, dtype=DT))
    feats.append(np.full((m,), g_std, dtype=DT))
    feats.append(np.full((m,), g_p50, dtype=DT))
    feats.append(np.full((m,), g_p90, dtype=DT))

    X = np.stack(feats, axis=1)
    y = H[ii, jj].astype(DT, copy=False)

    names = _feature_names(kernel_scales)
    return X, y, names


def _ridge_fit(X: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray:
    X = np.asarray(X, dtype=DT)
    y = np.asarray(y, dtype=DT).reshape(-1)
    nfeat = int(X.shape[1])
    XtX = X.T @ X
    XtX = XtX + float(l2) * np.eye(nfeat, dtype=DT)
    Xty = X.T @ y
    try:
        w = np.linalg.solve(XtX, Xty)
    except Exception:
        w = np.linalg.lstsq(XtX, Xty, rcond=None)[0]
    return np.asarray(w, dtype=DT).reshape(-1)


def _linear_predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    return (np.asarray(X, dtype=DT) @ np.asarray(w, dtype=DT).reshape(-1)).reshape(-1)


def _fit_linear_multistage(X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    stages = [
        {"name": "simple", "l2": 1e-1},
        {"name": "medium", "l2": 1e-2},
        {"name": "full", "l2": 1e-3},
    ]
    best = None
    frontier = []
    for s in stages:
        w = _ridge_fit(X, y, l2=float(s["l2"]))
        pred = _linear_predict(X, w)
        mse = float(np.mean((pred - y) ** 2))
        complexity = int(np.sum(np.abs(w) > 1e-8))
        rec = {"stage": s["name"], "l2": float(s["l2"]), "mse": mse, "complexity": complexity, "weights": w.tolist()}
        frontier.append({"stage": s["name"], "mse": mse, "complexity": complexity})
        if best is None or mse < best["mse"]:
            best = rec
    return {"best": best, "frontier": frontier}


def _attempt_pysr_fit(X: np.ndarray, y: np.ndarray, feature_names: List[str], seed: int) -> Optional[Dict[str, Any]]:
    try:
        from pysr import PySRRegressor  # type: ignore
    except Exception:
        return None

    model = PySRRegressor(
        niterations=60,
        populations=10,
        population_size=100,
        maxsize=30,
        maxdepth=10,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["exp", "log", "sin", "cos"],
        loss="loss(x, y) = (x - y)^2",
        model_selection="best",
        parsimony=1e-3,
        random_state=int(seed),
        verbosity=0,
    )
    model.fit(X, y, variable_names=feature_names)
    try:
        best = model.get_best()
        expr = str(best["equation"])
        loss = float(best.get("loss", np.nan))
        complexity = int(best.get("complexity", -1))
    except Exception:
        expr = str(model)
        loss = float("nan")
        complexity = -1
    return {
        "backend": "pysr",
        "expression_raw": expr,
        "loss": loss,
        "complexity": complexity,
    }


def _gemma_simplify(expression: str, backend: str = "ollama", model: str = "gemma") -> Dict[str, str]:
    prompt = (
        "You are a mathematical physicist.\n\n"
        "Given this expression:\n\n"
        f"{expression}\n\n"
        "Simplify it and express it as a clean operator form.\n\n"
        "Try to identify:\n"
        "- Laplacian terms\n"
        "- kernel terms\n"
        "- potential terms\n\n"
        "Return LaTeX and simplified Python form.\n"
        "Output JSON:\n"
        '{ "latex": "...", "python": "..." }\n'
    )
    if backend != "ollama":
        return {"latex": expression, "python": expression}
    try:
        p = subprocess.run(["ollama", "run", model, prompt], capture_output=True, text=True)
        if p.returncode != 0:
            return {"latex": expression, "python": expression}
        txt = p.stdout.strip()
        m = re.search(r"\{[\s\S]*\}", txt)
        if not m:
            return {"latex": expression, "python": expression}
        obj = json.loads(m.group(0))
        if isinstance(obj, dict) and "latex" in obj and "python" in obj:
            return {"latex": str(obj["latex"]), "python": str(obj["python"])}
        return {"latex": expression, "python": expression}
    except Exception:
        return {"latex": expression, "python": expression}


def main() -> None:
    ap = argparse.ArgumentParser(description="V13.1 Operator symbolic regression (structured operator -> formula)")
    ap.add_argument("--max_samples", type=int, default=50_000)
    ap.add_argument("--backend", type=str, default="pysr", choices=["pysr", "gplearn", "linear"])
    ap.add_argument("--use_llm", type=str, default="False")
    ap.add_argument("--llm_backend", type=str, default="ollama")
    ap.add_argument("--llm_model", type=str, default="gemma")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_points", type=int, default=128)
    ap.add_argument("--operator", type=str, default="runs/artin_operator_structured.npy")
    ap.add_argument("--weights", type=str, default="runs/artin_structured_weights.npy")
    ap.add_argument("--spectrum_csv", type=str, default="runs/artin_structured_spectrum.csv")
    ap.add_argument("--lengths", type=str, default="runs/artin_lengths.csv")
    ap.add_argument("--zeros", type=str, default="data/zeta_zeros.txt")
    ap.add_argument("--out_json", type=str, default="runs/operator_symbolic.json")
    ap.add_argument("--out_txt", type=str, default="runs/operator_formula.txt")
    ap.add_argument("--out_tex", type=str, default="runs/operator_formula.tex")
    args = ap.parse_args()

    op_path = Path(args.operator)
    if not op_path.exists():
        raise FileNotFoundError(f"missing operator: {op_path}")
    H = np.asarray(np.load(str(op_path)), dtype=DT)
    n = int(H.shape[0])

    # Reconstruct z_points deterministically (same domain distribution as V12 code)
    z = sample_domain(n_points=n, seed=int(args.seed))

    lengths = _read_lengths_csv(Path(args.lengths))

    X, y, feat_names = build_dataset(
        H=H,
        z=z,
        lengths=lengths,
        eps_density=0.6,
        kernel_scales=[0.1, 0.3, 1.0],
        max_samples=int(args.max_samples),
        seed=int(args.seed),
    )

    backend = str(args.backend).lower()
    use_llm = str(args.use_llm).lower() in ("1", "true", "yes", "y", "t")

    expression_raw = ""
    expression_simplified = ""
    latex = ""
    fit_info: Dict[str, Any] = {"backend": backend}

    if backend == "pysr":
        res = _attempt_pysr_fit(X, y, feat_names, seed=int(args.seed))
        if res is None:
            print("PySR not installed. Install with:", file=sys.stderr)
            print("  pip install pysr", file=sys.stderr)
            print("Falling back to linear regression with nonlinear features.", file=sys.stderr)
            backend = "linear"
        else:
            fit_info.update(res)
            expression_raw = str(res.get("expression_raw", ""))

    if backend == "gplearn":
        print("gplearn backend not bundled; falling back to linear regression.", file=sys.stderr)
        backend = "linear"

    if backend == "linear":
        multi = _fit_linear_multistage(X, y)
        best = multi["best"]
        w = np.asarray(best["weights"], dtype=DT)
        # compact expression string
        terms = []
        for wi, name in zip(w, feat_names):
            if abs(wi) > 1e-6:
                terms.append(f"({wi:.6g})*{name}")
        expression_raw = " + ".join(terms) if terms else "0.0"
        fit_info.update({"backend": "linear", "multistage": multi})

    if use_llm and expression_raw:
        simp = _gemma_simplify(expression_raw, backend=str(args.llm_backend), model=str(args.llm_model))
        latex = simp.get("latex", "")
        expression_simplified = simp.get("python", "")
    else:
        latex = expression_raw
        expression_simplified = expression_raw

    # Validation: MSE and spectral error using a linear symbolic reconstruction if available.
    # If backend is linear, we can reconstruct H_hat by predicting for all pairs.
    mse = float("inf")
    spectral_error = float("inf")
    try:
        if backend == "linear":
            # build full features for all entries (may be big but n<=512 typical)
            d = hyperbolic_distance_matrix(z, z).astype(DT, copy=False)
            d2 = d * d
            A = np.exp(-d2 / (0.6 + 1e-12))
            np.fill_diagonal(A, 0.0)
            density = np.sum(A, axis=1).astype(DT, copy=False)
            density = (density - float(np.mean(density))) / (float(np.std(density)) + 1e-12)

            g_mean = float(np.mean(lengths)) if lengths.size else 0.0
            g_std = float(np.std(lengths)) if lengths.size else 1.0
            g_p50 = float(np.percentile(lengths, 50.0)) if lengths.size else 0.0
            g_p90 = float(np.percentile(lengths, 90.0)) if lengths.size else 0.0

            diag = np.eye(n, dtype=DT)
            feats = [
                np.ones((n, n), dtype=DT),
                diag,
                d,
                d2,
                np.exp(-d2),
                density[:, None] * np.ones((1, n), dtype=DT),
                np.exp(-d2 / (0.1 * 0.1 + 1e-12)),
                np.exp(-d2 / (0.3 * 0.3 + 1e-12)),
                np.exp(-d2 / (1.0 * 1.0 + 1e-12)),
                np.full((n, n), g_mean, dtype=DT),
                np.full((n, n), g_std, dtype=DT),
                np.full((n, n), g_p50, dtype=DT),
                np.full((n, n), g_p90, dtype=DT),
            ]
            Phi = np.stack(feats, axis=2)  # (n,n,f)
            w_best = np.asarray(fit_info["multistage"]["best"]["weights"], dtype=DT).reshape(-1)
            H_hat = np.tensordot(Phi, w_best, axes=([2], [0]))
            H_hat = 0.5 * (H_hat + H_hat.T)
            mse = float(np.mean((H_hat - H) ** 2))

            # spectral error: compare normalized first k eigenvalues
            k_eval = min(128, n)
            eig_H, _, _ = safe_eigh(H, stabilize=True, seed=int(args.seed))
            eig_S, _, _ = safe_eigh(H_hat, stabilize=True, seed=int(args.seed))
            eig_H = np.sort(np.asarray(eig_H, dtype=DT).reshape(-1))[:k_eval]
            eig_S = np.sort(np.asarray(eig_S, dtype=DT).reshape(-1))[:k_eval]
            eig_Hn = _zscore(eig_H)
            eig_Sn = _zscore(eig_S)
            spectral_error = float(np.mean((eig_Hn - eig_Sn) ** 2))
        else:
            mse = float("nan")
            spectral_error = float("nan")
    except Exception:
        pass

    out = {
        "expression_raw": expression_raw,
        "expression_simplified": expression_simplified,
        "latex": latex,
        "mse": float(mse),
        "spectral_error": float(spectral_error),
        "fit_info": fit_info,
        "feature_names": feat_names,
        "n_points": int(n),
        "max_samples": int(args.max_samples),
    }

    out_json = Path(args.out_json)
    _write_json(out_json, out)
    _write_text(Path(args.out_txt), expression_simplified.strip() + "\n")
    _write_text(Path(args.out_tex), latex.strip() + "\n")


if __name__ == "__main__":
    main()

