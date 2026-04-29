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
        H = (-L + K).astype(np.float64, copy=False)
        H = 0.5 * (H + H.T)
        fro_sym = float(np.linalg.norm(H - H.T, ord="fro"))
        eigvals, _, erep = safe_eigh(H, stabilize=True, seed=42)
        spectral_error = float(np.std(eigvals[: min(16, len(eigvals))])) if len(eigvals) else float("inf")
        target_length = 4.0
        validity_score = 1.0 + (0.5 if primitive else 0.0)
        length_score = max(-2.0, min(1.0, -abs(float(length) - target_length) / target_length))
        spectral_score = -math.log1p(max(float(spectral_error), 0.0)) if np.isfinite(spectral_error) else -10.0
        if fro_sym < 1e-6:
            stability_score = 1.0
        else:
            stability_score = -math.log1p(max(fro_sym, 0.0))
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
                "self_adjoint_error": float(fro_sym),
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
