#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.artin_operator import build_laplacian, load_geodesics_words_json, sample_domain, select_top_k_geodesics
from core.spectral_stabilization import safe_eigh


DTYPE = np.float64
EPS = 1e-12


def _stable_hash_phase(word: List[int]) -> float:
    """
    Deterministic phase in [0, 2π] from the word entries.
    """
    payload = ",".join(str(int(a)) for a in word).encode("utf-8")
    h = hashlib.sha256(payload).digest()
    u64 = int.from_bytes(h[:8], byteorder="little", signed=False)
    # map to [0, 2π]
    return float((u64 / float(2**64 - 1)) * (2.0 * np.pi))


def _word_signature(word: List[int]) -> Dict[str, float]:
    w = [int(a) for a in word]
    length = float(len(w))
    abs_sum = float(np.sum(np.abs(np.asarray(w, dtype=np.int64)))) if w else 0.0
    s = np.sign(np.asarray(w, dtype=DTYPE)) if w else np.asarray([], dtype=DTYPE)
    sign_sum = float(np.sum(s)) if s.size else 0.0
    if s.size >= 2:
        alternating_score = float(np.sum(s[:-1] * s[1:]))
    else:
        alternating_score = 0.0
    hash_phase = float(_stable_hash_phase(w))
    return {
        "length": float(length),
        "abs_sum": float(abs_sum),
        "sign_sum": float(sign_sum),
        "alternating_score": float(alternating_score),
        "hash_phase": float(hash_phase),
    }


def _build_word_sensitive_core(
    *,
    z_points: np.ndarray,
    distances: Optional[np.ndarray] = None,
    geodesics: List[Dict[str, Any]],
    eps: float = 0.6,
    geo_sigma: float = 0.6,
    kernel_normalization: str = "max",
    laplacian_weight: float = 1.0,
    geo_weight: float = 10.0,
    potential_weight: float = 0.25,
    diag_shift: float = 1e-6,
) -> Dict[str, Any]:
    """
    Single implementation for word-sensitive H (V13J decomposition + public builder).

    Additive structure before symmetrization:
        H_tilde = laplacian_weight * L + geo_weight * K_norm + potential_weight * V_norm
    where K_norm, V_norm are max-normalized kernel / potential factors (``kernel_normalization='max'``).
    There is **no** separate braid matrix B in H; word data enter only through (K_norm, V_norm).
    Ramsey / Nijenhuis are **not** added to H (loss regularizers elsewhere).
    """
    z = np.asarray(z_points)
    n = int(z.shape[0])
    if distances is None:
        L, _, d = build_laplacian(z, eps=float(eps))
    else:
        d = np.asarray(distances, dtype=DTYPE, copy=False)
        L, _, _ = build_laplacian(z, eps=float(eps))

    d = np.asarray(d, dtype=DTYPE, copy=False)
    d2 = d * d
    y = np.imag(z).astype(DTYPE, copy=False).reshape(-1)

    K_raw = np.zeros((n, n), dtype=DTYPE)
    V_vec = np.zeros((n,), dtype=DTYPE)

    word_reports: List[Dict[str, Any]] = []
    for g in geodesics:
        a_list = g.get("a_list", None)
        if not isinstance(a_list, list) or len(a_list) == 0:
            continue
        w = [int(a) for a in a_list]
        sig = _word_signature(w)

        length = float(sig["length"])
        abs_sum = float(sig["abs_sum"])
        sign_sum = float(sig["sign_sum"])
        alternating_score = float(sig["alternating_score"])
        hash_phase = float(sig["hash_phase"])

        tr = g.get("trace", 0.0)
        try:
            trace_val = float(tr)
        except Exception:
            trace_val = 0.0
        trace_scale = 1.0 + 0.05 * float(np.tanh(abs(trace_val)))

        A_w = (1.0 + 0.1 * abs_sum + 0.2 * abs(sign_sum)) * trace_scale
        sigma_w = float(geo_sigma) * (1.0 + 0.05 * length)
        sigma_w2 = float(sigma_w * sigma_w + 1e-12)
        orientation_w = float(sign_sum / (length + 1e-8))
        phase_offset = 0.15 * alternating_score + 0.1 * float(np.tanh(trace_val))

        K_w = A_w * np.exp(-d2 / sigma_w2, dtype=DTYPE) * np.cos(hash_phase + phase_offset + orientation_w * d, dtype=DTYPE)
        K_raw += K_w

        V_vec += A_w * np.sin(hash_phase + y, dtype=DTYPE)

        word_reports.append(
            {
                "a_list": w,
                "length": float(length),
                "abs_sum": float(abs_sum),
                "sign_sum": float(sign_sum),
                "alternating_score": float(alternating_score),
                "trace": float(trace_val),
                "hash_phase": float(hash_phase),
                "A_w": float(A_w),
                "sigma_w": float(sigma_w),
                "orientation_w": float(orientation_w),
            }
        )

    norm_kind = str(kernel_normalization).strip().lower()
    k_scale = 1.0
    v_scale = 1.0
    if norm_kind == "max":
        k_scale = float(np.max(np.abs(K_raw))) if K_raw.size else 0.0
        v_scale = float(np.max(np.abs(V_vec))) if V_vec.size else 0.0
        K_norm = K_raw / (k_scale + EPS)
        V_vec_n = V_vec / (v_scale + EPS)
    elif norm_kind in ("none", "off", "false", "no"):
        K_norm = K_raw
        V_vec_n = V_vec
    else:
        raise ValueError(f"unknown kernel_normalization={kernel_normalization!r}; expected 'max' or 'none'")

    V_norm = np.diag(V_vec_n.astype(DTYPE, copy=False))
    lw = float(laplacian_weight)
    gw = float(geo_weight)
    pw = float(potential_weight)
    Lm = np.asarray(L, dtype=DTYPE, copy=False)
    H0 = lw * Lm
    G_w = gw * K_norm
    V_w = pw * V_norm
    B = np.zeros((n, n), dtype=DTYPE)
    H_tilde = H0 + G_w + V_w
    H_sym = 0.5 * (H_tilde + H_tilde.T)
    H_final = H_sym + float(diag_shift) * np.eye(n, dtype=DTYPE)

    word_reports.sort(key=lambda r: abs(float(r.get("A_w", 0.0))), reverse=True)
    top_word_signatures = word_reports[: min(10, len(word_reports))]

    rep: Dict[str, Any] = {
        "word_sensitivity_enabled": True,
        "n_points": int(n),
        "n_words_used": int(len(word_reports)),
        "eps": float(eps),
        "geo_weight": float(geo_weight),
        "geo_sigma": float(geo_sigma),
        "kernel_normalization": str(kernel_normalization),
        "laplacian_weight": float(laplacian_weight),
        "potential_weight": float(potential_weight),
        "k_word_norm": float(np.linalg.norm(K_norm, ord="fro")),
        "v_word_norm": float(np.linalg.norm(V_norm, ord="fro")),
        "operator_fro_norm": float(np.linalg.norm(H_final, ord="fro")),
        "top_word_signatures": top_word_signatures,
        "k_scale": float(k_scale),
        "v_scale": float(v_scale),
    }
    return {
        "H0": H0,
        "B": B,
        "G": G_w,
        "G_normalized": K_norm,
        "V": V_w,
        "V_normalized": V_norm,
        "H_tilde": H_tilde,
        "H_sym": H_sym,
        "H_final": H_final,
        "L": Lm,
        "K_raw": K_raw,
        "rep": rep,
    }


def build_word_sensitive_components(
    *,
    z_points: np.ndarray,
    distances: Optional[np.ndarray] = None,
    geodesics: List[Dict[str, Any]],
    eps: float = 0.6,
    geo_sigma: float = 0.6,
    kernel_normalization: str = "max",
    laplacian_weight: float = 1.0,
    geo_weight: float = 10.0,
    potential_weight: float = 0.25,
    diag_shift: float = 1e-6,
) -> Dict[str, Any]:
    """
    V13J: explicit additive components matching ``build_word_sensitive_operator`` numerically.

    Returns numpy arrays plus ``formula_text`` / ``latex_formula`` strings and Frobenius norms.
    """
    C = _build_word_sensitive_core(
        z_points=z_points,
        distances=distances,
        geodesics=geodesics,
        eps=eps,
        geo_sigma=geo_sigma,
        kernel_normalization=kernel_normalization,
        laplacian_weight=laplacian_weight,
        geo_weight=geo_weight,
        potential_weight=potential_weight,
        diag_shift=diag_shift,
    )
    H0, Gn, Vn = C["H0"], C["G_normalized"], C["V_normalized"]
    lw, gw, pw = float(laplacian_weight), float(geo_weight), float(potential_weight)
    formula_text = (
        "H_w = Sym(H_tilde_w) + diag_shift * I,  Sym(M) = (M + M^T) / 2,  diag_shift = 1e-6.\n"
        "H_tilde_w = H0 + geo_weight * G_norm + potential_weight * V_norm,\n"
        "  H0 = laplacian_weight * L  (graph Laplacian on hyperbolic distance kernel, eps).\n"
        "  G_norm[p,q] = (sum_w K_w[p,q]) / max|K_raw|  (max normalization; K_raw built below).\n"
        "  V_norm = diag(v / max|v|)  with v_q = sum_w A_w * sin(hash_phase(w) + Im(z_q)).\n"
        "Per word w (geodesic entry):\n"
        "  A_w = (1 + 0.1*|sum a| + 0.2*|sign_sum|) * (1 + 0.05*tanh(|trace|))  (trace from geodesic dict, else 0).\n"
        "  sigma_w = geo_sigma * (1 + 0.05*len(w)),  sigma_w2 = sigma_w^2 + 1e-12.\n"
        "  orientation_w = sign_sum / (len(w) + 1e-8).\n"
        "  phase_offset = 0.15*alternating_score + 0.1*tanh(trace).\n"
        "  K_w[p,q] = A_w * exp(-d_{pq}^2 / sigma_w2) * cos(hash_phase + phase_offset + orientation_w * d_{pq}),\n"
        "    d_{pq} hyperbolic distance between sample points z_p, z_q.\n"
        "  contribution to v_q: A_w * sin(hash_phase + Im(z_q)).\n"
        "There is no separate braid matrix B in H; Ramsey/Nijenhuis are not added to H."
    )
    latex_formula = (
        r"H_w = \mathrm{Sym}\bigl(\lambda_L L + \lambda_g \widehat{K}_w + \lambda_p \widehat{V}_w\bigr) + \varepsilon I,\quad "
        r"\mathrm{Sym}(M)=\tfrac12(M+M^{\mathsf T}),\ \varepsilon=10^{-6}."
        r"\\ \widehat{K}_w = K_{\mathrm{raw}}/\|K_{\mathrm{raw}}\|_\infty,\ "
        r"\widehat{V}_w=\mathrm{diag}(\mathbf{v}/\|\mathbf{v}\|_\infty)."
        r"\\ K^{(w)}_{pq}=A_w\exp(-d_{pq}^2/\sigma_w^2)\cos(\phi_w+\omega_w d_{pq}),\ "
        r"\phi_w=\mathrm{hash\_phase}(w)+\mathrm{phase\_offset}(w),\ \omega_w=\mathrm{orientation}_w."
    )
    comp_norms = {
        "fro_H0": float(np.linalg.norm(H0, ord="fro")),
        "fro_B": float(np.linalg.norm(C["B"], ord="fro")),
        "fro_G_normalized": float(np.linalg.norm(Gn, ord="fro")),
        "fro_G_weighted": float(np.linalg.norm(C["G"], ord="fro")),
        "fro_V_normalized": float(np.linalg.norm(Vn, ord="fro")),
        "fro_V_weighted": float(np.linalg.norm(C["V"], ord="fro")),
        "fro_H_tilde": float(np.linalg.norm(C["H_tilde"], ord="fro")),
        "fro_H_sym": float(np.linalg.norm(C["H_sym"], ord="fro")),
        "fro_H_final": float(np.linalg.norm(C["H_final"], ord="fro")),
        "coefficients": {"laplacian_weight": lw, "geo_weight": gw, "potential_weight": pw},
    }
    out = dict(C)
    out["component_norms"] = comp_norms
    out["formula_text"] = formula_text
    out["latex_formula"] = latex_formula
    return out


def build_word_sensitive_operator(
    *,
    z_points: np.ndarray,
    distances: Optional[np.ndarray] = None,
    geodesics: List[Dict[str, Any]],
    eps: float = 0.6,
    geo_sigma: float = 0.6,
    kernel_normalization: str = "max",
    laplacian_weight: float = 1.0,
    geo_weight: float = 10.0,
    potential_weight: float = 0.25,
    diag_shift: float = 1e-6,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Operator Builder V2 with guaranteed word sensitivity.

    H = Sym( laplacian_weight * L + geo_weight * K_norm + potential_weight * V_norm ) + diag_shift * I

    K_norm and V_norm are max-normalized when ``kernel_normalization=='max'`` (same as legacy).
    """
    C = _build_word_sensitive_core(
        z_points=z_points,
        distances=distances,
        geodesics=geodesics,
        eps=eps,
        geo_sigma=geo_sigma,
        kernel_normalization=kernel_normalization,
        laplacian_weight=laplacian_weight,
        geo_weight=geo_weight,
        potential_weight=potential_weight,
        diag_shift=diag_shift,
    )
    return C["H_final"], C["rep"]


def _write_spectrum_csv(path: Path, eigvals: np.ndarray) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("idx,eigval\n")
        for i, ev in enumerate(eigvals.reshape(-1)):
            f.write(f"{i},{float(ev)}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Operator Builder V2: word-sensitive Artin operator")
    ap.add_argument("--n_points", type=int, default=128)
    ap.add_argument("--eps", type=float, default=0.6)
    ap.add_argument("--geo_weight", type=float, default=25.0)
    ap.add_argument("--geo_sigma", type=float, default=0.4)
    ap.add_argument("--kernel_normalization", type=str, default="max")
    ap.add_argument("--laplacian_weight", type=float, default=1.0)
    ap.add_argument("--potential_weight", type=float, default=0.25)
    ap.add_argument("--top_k_geodesics", type=int, default=500)
    ap.add_argument("--geodesics", type=str, default="runs/artin_words.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=str, default="runs/artin_operator_word_sensitive.npy")
    ap.add_argument("--out_dir", type=str, default="runs")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = out_dir / out_path

    t0 = time.perf_counter()
    z = sample_domain(int(args.n_points), seed=int(args.seed))
    L, _, d = build_laplacian(z, eps=float(args.eps))
    words = load_geodesics_words_json(str(args.geodesics))
    geodesics = select_top_k_geodesics(words, top_k=int(args.top_k_geodesics))

    H, rep = build_word_sensitive_operator(
        z_points=z,
        distances=d,
        geodesics=geodesics,
        eps=float(args.eps),
        geo_sigma=float(args.geo_sigma),
        kernel_normalization=str(args.kernel_normalization),
        laplacian_weight=float(args.laplacian_weight),
        geo_weight=float(args.geo_weight),
        potential_weight=float(args.potential_weight),
    )
    rep["build_time_s"] = float(time.perf_counter() - t0)
    rep["out_path"] = str(out_path)
    rep["geodesics_path"] = str(args.geodesics)

    np.save(out_path, H.astype(DTYPE, copy=False))

    eigvals, _, erep = safe_eigh(H, stabilize=True, seed=int(args.seed))
    eigvals = np.sort(np.asarray(eigvals, dtype=DTYPE).reshape(-1))
    rep["eigh_report"] = erep

    _write_spectrum_csv(out_dir / "artin_word_sensitive_spectrum.csv", eigvals)

    with (out_dir / "artin_word_sensitive_report.json").open("w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2)


if __name__ == "__main__":
    main()

