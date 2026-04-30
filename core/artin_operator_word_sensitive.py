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

    H = laplacian_weight * L + geo_weight * K_word + potential_weight * V_word

    K_word and V_word explicitly depend on word entries, signs, length, trace (if present),
    orientation, and a deterministic hash phase.
    """
    z = np.asarray(z_points)
    n = int(z.shape[0])
    if distances is None:
        L, _, d = build_laplacian(z, eps=float(eps))
    else:
        d = np.asarray(distances, dtype=DTYPE, copy=False)
        # still build L in a consistent way to avoid relying on caller's laplacian
        L, _, _ = build_laplacian(z, eps=float(eps))

    d = np.asarray(d, dtype=DTYPE, copy=False)
    d2 = d * d
    y = np.imag(z).astype(DTYPE, copy=False).reshape(-1)

    K_word = np.zeros((n, n), dtype=DTYPE)
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

        # incorporate trace/length explicitly (trace comes from upstream word feature)
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
        K_word += K_w

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

    # Normalize only at the end
    norm_kind = str(kernel_normalization).strip().lower()
    if norm_kind == "max":
        k_scale = float(np.max(np.abs(K_word))) if K_word.size else 0.0
        v_scale = float(np.max(np.abs(V_vec))) if V_vec.size else 0.0
        K_word = K_word / (k_scale + EPS)
        V_vec = V_vec / (v_scale + EPS)
    elif norm_kind in ("none", "off", "false", "no"):
        pass
    else:
        raise ValueError(f"unknown kernel_normalization={kernel_normalization!r}; expected 'max' or 'none'")

    V_word = np.diag(V_vec.astype(DTYPE, copy=False))

    H = float(laplacian_weight) * np.asarray(L, dtype=DTYPE) + float(geo_weight) * K_word + float(potential_weight) * V_word
    H = 0.5 * (H + H.T)
    H = H + float(diag_shift) * np.eye(n, dtype=DTYPE)

    # report: top signatures by amplitude
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
        "k_word_norm": float(np.linalg.norm(K_word, ord="fro")),
        "v_word_norm": float(np.linalg.norm(V_word, ord="fro")),
        "operator_fro_norm": float(np.linalg.norm(H, ord="fro")),
        "top_word_signatures": top_word_signatures,
    }
    return H, rep


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

