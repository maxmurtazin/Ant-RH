#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
import math
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.artin_operator import build_geodesic_kernel, build_laplacian, sample_domain
from core.artin_symbolic_billiard import (
    build_word,
    hyperbolic_length_from_trace,
    is_hyperbolic_matrix,
    precompute_T_powers,
    trace_2x2,
)
from core.spectral_stabilization import safe_eigh


def _physics_diagnostics(H_raw: np.ndarray, eigvals: np.ndarray) -> Dict[str, Any]:
    # 1) Self-adjoint diagnostics
    hn = float(np.linalg.norm(H_raw, ord="fro"))
    hdiff = float(np.linalg.norm(H_raw - H_raw.T, ord="fro")) / (hn + 1e-8)
    if hdiff < 1e-6:
        sa_status = "ok"
    elif hdiff < 1e-3:
        sa_status = "approx"
    else:
        sa_status = "broken"

    # 2) Spectral diagnostics
    eig = np.asarray(eigvals)
    imag_max = float(np.max(np.abs(np.imag(eig)))) if eig.size else float("inf")
    spectrum_real = bool(imag_max < 1e-8)
    eig_sorted = np.sort(np.real(eig)) if eig.size else np.array([], dtype=np.float64)
    spacing = np.diff(eig_sorted) if eig_sorted.size >= 2 else np.array([], dtype=np.float64)
    spacing_std = float(np.std(spacing)) if spacing.size else 0.0
    spectral_status = "ok" if (spectrum_real and spacing_std > 0.0) else "degenerate"

    # 3) OTOC-like proxy via r-statistic
    r_mean = None
    otoc_indicator = "integrable"
    if spacing.size >= 3:
        s = np.asarray(spacing, dtype=np.float64)
        s0 = np.abs(s[:-1])
        s1 = np.abs(s[1:])
        denom = np.maximum(np.maximum(s0, s1), 1e-12)
        r = np.minimum(s0, s1) / denom
        r_mean_val = float(np.mean(r)) if r.size else None
        r_mean = r_mean_val
        if r_mean_val is None:
            otoc_indicator = "integrable"
        elif r_mean_val < 0.4:
            otoc_indicator = "integrable"
        elif r_mean_val < 0.5:
            otoc_indicator = "intermediate"
        else:
            otoc_indicator = "chaotic"
    else:
        r_mean = None
        otoc_indicator = "integrable"

    return {
        "self_adjoint_error": float(hdiff),
        "self_adjoint_status": sa_status,
        "spectral_status": spectral_status,
        "otoc_indicator": otoc_indicator,
        "r_mean": (float(r_mean) if isinstance(r_mean, (int, float)) else None),
    }


def braid_tokens_to_artin_word(tokens) -> list[int]:
    word: List[int] = []
    for tok in tokens:
        s = str(tok)
        if s.startswith("SIGMA_") and (s.endswith("_PLUS") or s.endswith("_MINUS")):
            parts = s.split("_")
            if len(parts) != 3:
                continue
            try:
                idx = int(parts[1])
            except Exception:
                continue
            sign = 1 if parts[2] == "PLUS" else -1
            if idx > 0:
                word.append(sign * idx)
    return word


def evaluate_braid_candidate(word: list[int], seen_words: Optional[Set[Tuple[int, ...]]] = None) -> dict:
    result: Dict[str, Any] = {
        "valid": False,
        "word": [int(a) for a in word if int(a) != 0] if word else [],
        "reward": -10.0,
        "validity_score": -1.0,
        "length_score": -2.0,
        "spectral_score": -10.0,
        "stability_score": -10.0,
        "diversity_score": 0.0,
        "spectral_error": float("inf"),
        "self_adjoint_error": float("inf"),
        "self_adjoint_status": "broken",
        "spectral_status": "degenerate",
        "otoc_indicator": "integrable",
        "r_mean": None,
        "primitive": False,
        "hyperbolic": False,
        "reason": "",
    }
    seen_set = seen_words if seen_words is not None else set()
    try:
        clean = [int(a) for a in word]
    except Exception:
        result["reason"] = "non_integer_word"
        return result
    if any(a == 0 for a in clean):
        result["reason"] = "zero_action"
        return result
    if len(clean) < 3:
        result["reason"] = "too_short"
        return result
    max_power = max(abs(a) for a in clean)
    try:
        t_stack, offset = precompute_T_powers(int(max_power))
        M = build_word(clean, t_stack, offset)
        tr = trace_2x2(M)
        hyperbolic = bool(is_hyperbolic_matrix(M))
        result["hyperbolic"] = hyperbolic
        if not hyperbolic:
            result["reason"] = "non_hyperbolic"
            return result
        length = hyperbolic_length_from_trace(abs(tr))
        primitive = len(clean) == len(result["word"]) and len(set(clean)) > 1
        geodesics = [{"a_list": clean, "length": float(length), "is_hyperbolic": True, "primitive": primitive}]
        z = sample_domain(64, seed=42)
        L, _, _ = build_laplacian(z, eps=0.6)
        K, _ = build_geodesic_kernel(z, geodesics, sigma=0.3)
        H_raw = (-L + K).astype(np.float64, copy=False)
        H = 0.5 * (H_raw + H_raw.T)
        eigvals, _, erep = safe_eigh(H, stabilize=True, seed=42)
        phys = _physics_diagnostics(H_raw, eigvals)
        spectral_error = float(np.std(eigvals[: min(16, len(eigvals))])) if len(eigvals) else float("inf")
        target_length = 4.0
        validity_score = 1.0 + (0.5 if primitive else 0.0)
        length_score = max(-2.0, min(1.0, -abs(float(length) - target_length) / target_length))
        spectral_score = -math.log1p(max(float(spectral_error), 0.0)) if np.isfinite(spectral_error) else -10.0
        if float(phys.get("self_adjoint_error", float("inf"))) < 1e-6:
            stability_score = 1.0
        else:
            stability_score = -math.log1p(max(float(phys.get("self_adjoint_error", 0.0)), 0.0))
        repetition_count = sum(1 for i in range(1, len(clean)) if clean[i] == clean[i - 1])
        diversity_ratio = len(set(clean)) / max(len(clean), 1)
        diversity_score = max(0.0, min(1.0, diversity_ratio - repetition_count / max(len(clean) - 1, 1)))
        word_key = tuple(clean)
        if word_key in seen_set:
            diversity_score -= 1.0
        else:
            seen_set.add(word_key)
        reward = (
            1.0 * validity_score
            + 0.5 * length_score
            + 1.0 * spectral_score
            + 0.5 * stability_score
            + 0.2 * diversity_score
        )
        reward = max(-10.0, min(10.0, float(reward)))
        result.update(
            {
                "valid": True,
                "reason": "ok",
                "length": float(length),
                "trace": float(tr),
                "primitive": bool(primitive),
                "spectral_error": float(spectral_error),
                "self_adjoint_error": float(phys.get("self_adjoint_error", float("inf"))),
                "self_adjoint_status": str(phys.get("self_adjoint_status", "broken")),
                "spectral_status": str(phys.get("spectral_status", "degenerate")),
                "otoc_indicator": str(phys.get("otoc_indicator", "integrable")),
                "r_mean": phys.get("r_mean", None),
                "validity_score": float(validity_score),
                "length_score": float(length_score),
                "spectral_score": float(spectral_score),
                "stability_score": float(stability_score),
                "diversity_score": float(diversity_score),
                "reward": float(reward),
            }
        )
        return result
    except Exception as e:
        result["reason"] = repr(e)
        return result
