#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.llm_runner import LLMRunner
from core.pde_feature_library import build_feature_context, reconstruct_points, stack_eigenpair_system
from core.spectral_stabilization import safe_eigh

DT = np.float64


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _read_text(path: Path) -> Optional[str]:
    try:
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _write_text(path: Path, s: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(s, encoding="utf-8")


def _write_terms_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["term", "coefficient", "abs_coefficient", "active"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _load_lengths(path: Path) -> np.ndarray:
    try:
        if not path.exists():
            return np.array([], dtype=DT)
        out = np.loadtxt(str(path), delimiter=",", skiprows=1, usecols=(0,), dtype=DT)
        out = np.asarray(out, dtype=DT).reshape(-1)
        out = out[np.isfinite(out)]
        return out
    except Exception:
        return np.array([], dtype=DT)


def _bool_arg(x: Any) -> bool:
    return str(x).lower() in {"1", "true", "yes", "y", "t"}


def _ridge_solve(X: np.ndarray, y: np.ndarray, ridge: float) -> np.ndarray:
    X = np.asarray(X, dtype=DT)
    y = np.asarray(y, dtype=DT).reshape(-1)
    p = X.shape[1]
    XtX = X.T @ X + float(ridge) * np.eye(p, dtype=DT)
    Xty = X.T @ y
    try:
        coef = np.linalg.solve(XtX, Xty)
    except Exception:
        coef = np.linalg.lstsq(XtX, Xty, rcond=None)[0]
    return np.asarray(coef, dtype=DT).reshape(-1)


def _stlsq(X: np.ndarray, y: np.ndarray, threshold: float, ridge: float, n_iter: int = 5) -> np.ndarray:
    X = np.asarray(X, dtype=DT)
    y = np.asarray(y, dtype=DT).reshape(-1)
    coef = _ridge_solve(X, y, ridge=ridge)
    for _ in range(int(n_iter)):
        active = np.abs(coef) >= float(threshold)
        if not np.any(active):
            return np.zeros_like(coef)
        Xa = X[:, active]
        coef_a = _ridge_solve(Xa, y, ridge=ridge)
        coef_new = np.zeros_like(coef)
        coef_new[active] = coef_a
        coef = coef_new
    return coef


def _fit_lasso_if_available(X: np.ndarray, y: np.ndarray, alpha: float) -> Optional[np.ndarray]:
    try:
        from sklearn.linear_model import Lasso  # type: ignore
    except Exception:
        return None
    try:
        model = Lasso(alpha=float(alpha), fit_intercept=False, max_iter=10000)
        model.fit(X, y)
        return np.asarray(model.coef_, dtype=DT).reshape(-1)
    except Exception:
        return None


def _norm_error(y: np.ndarray, yhat: np.ndarray) -> float:
    den = float(np.mean(y**2)) + 1e-12
    return float(np.mean((y - yhat) ** 2) / den)


def _term_specs() -> Dict[str, Tuple[str, str]]:
    return {
        "psi": (r"\psi", "c0 psi"),
        "laplacian_psi": (r"\Delta_G \psi", "c1 Delta_G psi"),
        "rbf_kernel_psi": (r"\mathcal{K}_{\mathrm{rbf}}\psi", "c2 K_rbf psi"),
        "inv_distance_kernel_psi": (r"\mathcal{K}_{\mathrm{inv}}\psi", "c3 K_inv psi"),
        "potential_psi": (r"V(z)\psi", "c4 V(z) psi"),
        "density_psi": (r"\rho(z)\psi", "c5 rho(z) psi"),
        "mean_length_psi": (r"\bar{\ell}\,\psi", "c6 mean_length psi"),
        "std_length_psi": (r"\sigma_{\ell}\,\psi", "c7 std_length psi"),
        "psi_cubed": (r"\psi^3", "c8 psi^3"),
        "abs_psi_times_psi": (r"|\psi|\psi", "c9 |psi|psi"),
        "inv_im_psi": (r"y^{-1}\psi", "c10 y^{-1}psi"),
        "log_im_psi": (r"\log(y)\psi", "c11 log(y)psi"),
    }


def _build_formula(names: List[str], coef: np.ndarray, tol: float = 1e-12) -> Tuple[str, str, List[Dict[str, Any]]]:
    specs = _term_specs()
    text_terms: List[str] = []
    tex_terms: List[str] = []
    rows: List[Dict[str, Any]] = []
    for n, c in zip(names, coef):
        c = float(c)
        active = abs(c) > float(tol)
        rows.append({"term": n, "coefficient": c, "abs_coefficient": abs(c), "active": bool(active)})
        if not active:
            continue
        tex_name, txt_name = specs.get(n, (n, n))
        tex_terms.append(f"{c:+.6g}\\,{tex_name}")
        text_terms.append(f"{c:+.6g} * {txt_name}")
    if not tex_terms:
        return "Hpsi ~= 0", r"H\psi \approx 0", rows
    text_formula = "Hpsi ~= " + " ".join(text_terms)
    tex_formula = r"H\psi \approx " + " ".join(tex_terms)
    return text_formula, tex_formula, rows


def _extract_json_blob(text: str) -> Optional[Dict[str, Any]]:
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except Exception:
        return None
    return obj if isinstance(obj, dict) else None


def _llm_interpret(latex_formula: str, metrics: Dict[str, Any], llama_cli: str, model_path: str) -> Optional[str]:
    prompt = (
        "You are a mathematical physicist.\n\n"
        "We discovered the following approximate operator equation:\n\n"
        f"{latex_formula}\n\n"
        "Metrics:\n"
        f"{json.dumps(metrics, indent=2)}\n\n"
        "Explain:\n"
        "1. What type of operator this resembles\n"
        "2. Whether it is local, nonlocal, or mixed\n"
        "3. Which terms dominate\n"
        "4. Whether this is meaningful for Hilbert-Polya style search\n"
        "5. Main limitations\n\n"
        "Be honest if the fit is weak."
    )
    try:
        runner = LLMRunner(model_path=str(model_path), llama_cli=str(llama_cli), n_ctx=4096, timeout_s=60.0)
        out = runner.generate(prompt, max_tokens=400, temperature=0.2).strip()
        return out if out else None
    except Exception:
        return None


def _report_markdown(obj: Dict[str, Any]) -> str:
    inp = obj.get("inputs", {})
    eig = obj.get("eigenpair_quality", {})
    selected = obj.get("selected_model", {})
    eq_tex = obj.get("equation_latex", "")
    terms = obj.get("terms", [])
    fit = obj.get("fit_quality", {})
    interp = obj.get("interpretation", "LLM interpretation unavailable.")
    limitations = obj.get("limitations", [])
    next_steps = obj.get("next_steps", [])
    return (
        "# Operator PDE Discovery Report\n\n"
        "## Inputs\n\n"
        "```json\n"
        f"{json.dumps(inp, indent=2)}\n"
        "```\n\n"
        "## Eigenpair quality\n\n"
        "```json\n"
        f"{json.dumps(eig, indent=2)}\n"
        "```\n\n"
        "## Candidate term library\n\n"
        "- psi\n- laplacian_psi\n- rbf_kernel_psi\n- inv_distance_kernel_psi\n- potential_psi\n"
        "- density_psi\n- mean_length_psi\n- std_length_psi\n- psi_cubed\n- abs_psi_times_psi\n"
        "- inv_im_psi\n- log_im_psi\n\n"
        "## Selected equation\n\n"
        "```tex\n"
        f"{eq_tex}\n"
        "```\n\n"
        "## Coefficients\n\n"
        "```json\n"
        f"{json.dumps(terms, indent=2)}\n"
        "```\n\n"
        "## Fit quality\n\n"
        "```json\n"
        f"{json.dumps(fit, indent=2)}\n"
        "```\n\n"
        "## Interpretation\n\n"
        f"{interp}\n\n"
        "## Limitations\n\n"
        + "\n".join(f"- {x}" for x in limitations)
        + "\n\n## Next steps\n\n"
        + "\n".join(f"- {x}" for x in next_steps)
        + "\n"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="V13.2 Operator PDE discovery for Ant-RH")
    ap.add_argument("--operator", type=str, default="runs/artin_operator_structured.npy")
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--threshold", type=float, default=1e-4)
    ap.add_argument("--ridge", type=float, default=1e-8)
    ap.add_argument("--alpha_complexity", type=float, default=1e-3)
    ap.add_argument("--use_lasso", type=str, default="False")
    ap.add_argument("--use_llm", type=str, default="True")
    ap.add_argument("--llama_cli", type=str, default="llama-cli")
    ap.add_argument("--model_path", type=str, default="/Users/machome/models/gemma/gemma-3-1b-it-Q5_K_M.gguf")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sigma", type=float, default=0.6)
    ap.add_argument("--out_json", type=str, default="runs/operator_pde_discovery.json")
    ap.add_argument("--out_tex", type=str, default="runs/operator_pde_formula.tex")
    ap.add_argument("--out_md", type=str, default="runs/operator_pde_report.md")
    ap.add_argument("--out_terms_csv", type=str, default="runs/operator_pde_terms.csv")
    args = ap.parse_args()

    root = Path(ROOT)
    op_primary = root / args.operator
    op_fallback = root / "runs/artin_operator.npy"
    op_path = op_primary if op_primary.exists() else op_fallback
    if not op_path.exists():
        raise FileNotFoundError(
            f"Missing operator file. Tried '{op_primary.as_posix()}' and fallback '{op_fallback.as_posix()}'."
        )

    H = np.asarray(np.load(str(op_path)), dtype=DT)
    if H.ndim != 2 or H.shape[0] != H.shape[1]:
        raise ValueError(f"Operator must be square matrix, got shape={H.shape}")
    H = np.nan_to_num(H, nan=0.0, posinf=1e6, neginf=-1e6)
    H = 0.5 * (H + H.T)

    n = int(H.shape[0])
    k_req = int(args.k)
    k_use = max(1, min(k_req, n))

    eigvals, eigvecs, erep = safe_eigh(H, k=None, return_eigenvectors=True, stabilize=True, seed=int(args.seed))
    if eigvecs is None or eigvals.size == 0:
        eigvals, eigvecs, erep = safe_eigh(H, k=None, return_eigenvectors=True, stabilize=False, seed=int(args.seed))
    if eigvecs is None or eigvals.size == 0:
        raise RuntimeError("Eigensolver failed; could not obtain eigenpairs.")

    eigvals = np.asarray(eigvals, dtype=DT).reshape(-1)
    eigvecs = np.asarray(eigvecs, dtype=DT)
    sort_idx = np.argsort(eigvals)
    eigvals = eigvals[sort_idx]
    eigvecs = eigvecs[:, sort_idx]
    k_use = min(k_use, eigvals.size, eigvecs.shape[1])

    lengths = _load_lengths(root / "runs/artin_lengths.csv")
    cfg_text = _read_text(root / "runs/v12_config_used.yaml") or ""
    cfg_seed_match = re.search(r"^\s*seed:\s*(\d+)\s*$", cfg_text, flags=re.M)
    seed_use = int(cfg_seed_match.group(1)) if cfg_seed_match else int(args.seed)

    z = reconstruct_points(n=n, seed=seed_use)
    ctx = build_feature_context(H=H, z=z, lengths=lengths, sigma=float(args.sigma))
    X, y, names = stack_eigenpair_system(eigvals=eigvals, eigvecs=eigvecs, ctx=ctx, k_use=k_use)
    if X.size == 0 or y.size == 0:
        raise RuntimeError("Feature stack is empty; cannot fit PDE model.")

    thresholds = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    if float(args.threshold) not in thresholds:
        thresholds.append(float(args.threshold))
        thresholds = sorted(set(thresholds))
    use_lasso = _bool_arg(args.use_lasso)

    candidates: List[Dict[str, Any]] = []
    for thr in thresholds:
        coef = None
        fit_backend = "stlsq"
        if use_lasso:
            coef = _fit_lasso_if_available(X, y, alpha=float(thr))
            if coef is not None:
                fit_backend = "lasso"
        if coef is None:
            coef = _stlsq(X, y, threshold=float(thr), ridge=float(args.ridge), n_iter=5)

        yhat = X @ coef
        mse = float(np.mean((yhat - y) ** 2))
        nerr = _norm_error(y, yhat)
        active = int(np.sum(np.abs(coef) > 1e-12))
        score = float(nerr + float(args.alpha_complexity) * active)
        candidates.append(
            {
                "threshold": float(thr),
                "backend": fit_backend,
                "mse": mse,
                "normalized_error": nerr,
                "active_terms": active,
                "score": score,
                "coef": coef.tolist(),
            }
        )

    candidates = sorted(candidates, key=lambda r: r["score"])
    best = candidates[0]
    coef_best = np.asarray(best["coef"], dtype=DT)

    formula_text, formula_tex, term_rows = _build_formula(names, coef_best, tol=1e-12)
    active_terms = [r for r in term_rows if r["active"]]
    top_terms = sorted(active_terms, key=lambda r: r["abs_coefficient"], reverse=True)[:5]

    fit_quality = {
        "mse": float(best["mse"]),
        "normalized_error": float(best["normalized_error"]),
        "active_terms": int(best["active_terms"]),
        "score": float(best["score"]),
        "k_requested": int(k_req),
        "k_used": int(k_use),
    }

    interpretation = "LLM interpretation unavailable."
    if _bool_arg(args.use_llm):
        llm = _llm_interpret(
            latex_formula=formula_tex,
            metrics={
                "fit_quality": fit_quality,
                "top_terms": top_terms,
                "eigh_success": erep.get("eigh_success"),
                "operator_dim": n,
            },
            llama_cli=str(args.llama_cli),
            model_path=str(args.model_path),
        )
        if llm:
            interpretation = llm

    limitations = [
        "Regression is performed on reconstructed geometry, not exact stored z-points.",
        "Feature library is finite and may miss true operator components.",
        "Shared coefficients across eigenstates impose a strong model bias.",
    ]
    if fit_quality["normalized_error"] > 0.5:
        limitations.append("Fit quality is weak; discovered equation should be treated as preliminary.")
    if fit_quality["active_terms"] >= 8:
        limitations.append("Model is relatively dense; interpretability is limited.")

    next_steps = [
        "Compare fits across alternative kernels and Laplacian constructions.",
        "Condition on spectral windows rather than global first-k stacking.",
        "Validate discovered terms on held-out eigenpairs.",
    ]

    out = {
        "inputs": {
            "operator_path_used": op_path.as_posix(),
            "operator_requested": op_primary.as_posix(),
            "operator_fallback": op_fallback.as_posix(),
            "n": int(n),
            "seed_used": int(seed_use),
            "optional_files": {
                "runs/artin_structured_spectrum.csv": (root / "runs/artin_structured_spectrum.csv").exists(),
                "runs/artin_structured_report.json": (root / "runs/artin_structured_report.json").exists(),
                "runs/operator_symbolic.json": (root / "runs/operator_symbolic.json").exists(),
                "runs/operator_stability_report.json": (root / "runs/operator_stability_report.json").exists(),
                "runs/v12_config_used.yaml": (root / "runs/v12_config_used.yaml").exists(),
            },
        },
        "eigenpair_quality": {
            "eigh_report": erep,
            "eig_min": float(np.min(eigvals[:k_use])),
            "eig_max": float(np.max(eigvals[:k_use])),
            "eig_std": float(np.std(eigvals[:k_use])),
        },
        "selected_model": {
            "threshold": float(best["threshold"]),
            "backend": best["backend"],
            "ridge": float(args.ridge),
            "alpha_complexity": float(args.alpha_complexity),
            "active_terms": int(best["active_terms"]),
        },
        "candidate_models": [{k: v for k, v in c.items() if k != "coef"} for c in candidates],
        "terms": term_rows,
        "top_terms": top_terms,
        "fit_quality": fit_quality,
        "equation_text": formula_text,
        "equation_latex": formula_tex,
        "interpretation": interpretation,
        "limitations": limitations,
        "next_steps": next_steps,
    }

    _write_json(root / args.out_json, out)
    _write_text(root / args.out_tex, formula_tex + "\n")
    _write_terms_csv(root / args.out_terms_csv, term_rows)
    _write_text(root / args.out_md, _report_markdown(out))


if __name__ == "__main__":
    main()

