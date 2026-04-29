#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.artin_operator import build_geodesic_kernel, build_laplacian, load_zeros, sample_domain
from core.artin_symbolic_billiard import (
    build_word,
    hyperbolic_length_from_trace,
    is_hyperbolic_matrix,
    precompute_T_powers,
    trace_2x2,
)
from core.spectral_stabilization import safe_eigh, stable_spectral_loss

try:
    from core.artin_operator_structured import build_operator_basis

    _HAVE_STRUCTURED = True
except Exception:
    build_operator_basis = None
    _HAVE_STRUCTURED = False


DTYPE = np.float64
EPS = 1e-12


def _valid_word(word: List[int], max_length: int, max_power: int) -> bool:
    if len(word) < 3 or len(word) > int(max_length):
        return False
    last = None
    for a in word:
        try:
            ai = int(a)
        except Exception:
            return False
        if ai == 0 or abs(ai) > int(max_power):
            return False
        if last is not None and ai == last:
            return False
        last = ai
    return True


def _random_word(rng: np.random.Generator, min_length: int, max_length: int, max_power: int) -> List[int]:
    vals = [a for a in range(-int(max_power), int(max_power) + 1) if a != 0]
    L = int(rng.integers(min_length, max_length + 1))
    word: List[int] = []
    while len(word) < L:
        nxt = int(rng.choice(vals))
        if word and word[-1] == nxt:
            continue
        word.append(nxt)
    return word


def _word_feature(word: List[int], t_stack: np.ndarray, offset: int) -> Optional[Dict[str, Any]]:
    try:
        M = build_word(word, t_stack, offset)
        tr = trace_2x2(M)
        if not is_hyperbolic_matrix(M):
            return None
        ell = hyperbolic_length_from_trace(abs(tr))
        if not np.isfinite(ell) or ell <= 0.0:
            return None
        return {
            "a_list": [int(a) for a in word],
            "length": float(ell),
            "trace": float(tr),
            "is_hyperbolic": True,
            "primitive": True,
        }
    except Exception:
        return None


def _make_word_set(
    kind: str,
    rng: np.random.Generator,
    words_per_set: int,
    max_length: int,
    max_power: int,
    set_idx: int,
) -> List[List[int]]:
    out: List[List[int]] = []
    base = [1, -1, 2]
    while len(out) < int(words_per_set):
        if kind == "random_words_small":
            w = _random_word(rng, 3, max(3, min(4, max_length)), max(2, min(max_power, 2)))
        elif kind == "random_words_large":
            w = _random_word(rng, max(3, max_length - 2), max_length, max_power)
        elif kind == "planner_like_words":
            templates = [
                [1, -1, 2],
                [2, -1, 1],
                [-1, 2, -2],
                [1, -2, 2, -1],
            ]
            w = list(templates[len(out) % len(templates)])
            while len(w) < min(max_length, 5 + (len(out) % 2)):
                nxt = int(rng.choice([-2, -1, 1, 2]))
                if w[-1] != nxt:
                    w.append(nxt)
        elif kind == "repeated_words":
            rep = 1 + (set_idx + len(out)) % max(1, max_length // len(base))
            w = (base * max(1, rep))[: max(3, min(max_length, 3 * rep))]
        else:  # shuffled_words
            source = [1, -1, 2, -2, 3, -3][: max(3, min(6, 2 * max_power))]
            rng.shuffle(source)
            w = source[: max(3, min(max_length, len(source)))]
        if _valid_word(w, max_length, max_power):
            out.append([int(a) for a in w])
    return out


def _build_geodesics(
    words: List[List[int]],
    max_power: int,
    top_k_geodesics: int,
) -> List[Dict[str, Any]]:
    t_stack, offset = precompute_T_powers(int(max_power))
    geodesics: List[Dict[str, Any]] = []
    seen = set()
    for word in words:
        feat = _word_feature(word, t_stack, offset)
        if feat is None:
            continue
        key = tuple(feat["a_list"])
        if key in seen:
            continue
        seen.add(key)
        geodesics.append(feat)
    geodesics.sort(key=lambda g: float(g["length"]))
    return geodesics[: int(top_k_geodesics)]


def _build_operator(
    z_points: np.ndarray,
    geodesics: List[Dict[str, Any]],
    eps: float,
    sigma: float,
    use_structured: bool,
    top_k_geodesics: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    L, _, distances = build_laplacian(z_points, eps=float(eps))
    rep: Dict[str, Any] = {"builder": "basic", "used_geodesics": int(len(geodesics))}

    if use_structured and _HAVE_STRUCTURED and build_operator_basis is not None:
        try:
            tmp_path = Path("runs") / "_operator_sensitivity_words.json"
            tmp_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path.write_text(json.dumps(geodesics, ensure_ascii=False), encoding="utf-8")
            basis, names, basis_rep = build_operator_basis(
                z_points=z_points,
                distances=distances,
                eps=float(eps),
                geo_sigma=float(sigma),
                top_k_geodesics=int(top_k_geodesics),
                geodesics_path=str(tmp_path),
                seed=42,
            )
            H = np.zeros_like(L, dtype=DTYPE)
            for B, name in zip(basis, names):
                if str(name) == "K_geo":
                    H = H + np.asarray(B, dtype=DTYPE)
            if not np.any(H):
                H = -np.asarray(L, dtype=DTYPE)
            else:
                H = -np.asarray(L, dtype=DTYPE) + np.asarray(H, dtype=DTYPE)
            H = 0.5 * (H + H.T)
            rep.update({"builder": "structured", "basis_report": basis_rep})
            return H, rep
        except Exception as e:
            rep["structured_error"] = repr(e)

    K, used = build_geodesic_kernel(z_points, geodesics, sigma=float(sigma))
    H = -np.asarray(L, dtype=DTYPE) + np.asarray(K, dtype=DTYPE)
    H = 0.5 * (H + H.T)
    rep["used_geodesics"] = int(used)
    return H, rep


def _loss_against_zeros(H: np.ndarray, zeros: Optional[np.ndarray]) -> float:
    if zeros is None or zeros.size == 0:
        return float("nan")
    try:
        loss, _ = stable_spectral_loss(H, zeros, k=min(H.shape[0], zeros.size), normalize_spectrum=True, spacing_loss=True)
        return float(loss)
    except Exception:
        return float("nan")


def _diagnosis(
    operator_distance_mean: float,
    spectrum_distance_mean: float,
    loss_std: float,
) -> str:
    msgs: List[str] = []
    if np.isfinite(operator_distance_mean) and operator_distance_mean < 1e-4:
        msgs.append("Artin words barely affect operator.")
    if np.isfinite(spectrum_distance_mean) and spectrum_distance_mean < 1e-4:
        msgs.append("Spectrum insensitive to word changes.")
    if np.isfinite(loss_std) and loss_std < 1e-5:
        msgs.append("Loss is flat; ACO cannot learn.")
    if len(msgs) == 3:
        msgs.append("Operator/loss pipeline is insensitive. Need stronger geodesic coupling.")
    if not msgs:
        msgs.append("Operator and spectrum respond to word changes at measurable scale.")
    return " ".join(msgs)


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnostic test: operator sensitivity to Artin words")
    ap.add_argument("--n_sets", type=int, default=8)
    ap.add_argument("--words_per_set", type=int, default=50)
    ap.add_argument("--max_length", type=int, default=6)
    ap.add_argument("--max_power", type=int, default=4)
    ap.add_argument("--n_points", type=int, default=128)
    ap.add_argument("--top_k_geodesics", type=int, default=500)
    ap.add_argument("--eps", type=float, default=0.6)
    ap.add_argument("--sigma", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--zeros", type=str, default="data/zeta_zeros.txt")
    ap.add_argument("--out_dir", type=str, default="runs")
    args = ap.parse_args()

    rng = np.random.default_rng(int(args.seed))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        zeros = load_zeros(args.zeros)
    except Exception:
        zeros = None

    set_types = [
        "random_words_small",
        "random_words_large",
        "planner_like_words",
        "repeated_words",
        "shuffled_words",
    ]
    z_points = sample_domain(int(args.n_points), seed=int(args.seed))

    rows: List[Dict[str, Any]] = []
    operators: List[np.ndarray] = []
    eigens: List[np.ndarray] = []
    losses: List[float] = []

    for idx in range(int(args.n_sets)):
        kind = set_types[idx % len(set_types)]
        words = _make_word_set(
            kind=kind,
            rng=rng,
            words_per_set=int(args.words_per_set),
            max_length=int(args.max_length),
            max_power=int(args.max_power),
            set_idx=idx,
        )
        geodesics = _build_geodesics(words, int(args.max_power), int(args.top_k_geodesics))
        H, build_rep = _build_operator(
            z_points=z_points,
            geodesics=geodesics,
            eps=float(args.eps),
            sigma=float(args.sigma),
            use_structured=True,
            top_k_geodesics=int(args.top_k_geodesics),
        )
        fro_norm = float(np.linalg.norm(H, ord="fro"))
        eigvals, _, eig_rep = safe_eigh(H, stabilize=True, seed=int(args.seed) + idx)
        eigvals = np.sort(np.asarray(eigvals, dtype=DTYPE).reshape(-1))
        loss = _loss_against_zeros(H, zeros)

        rows.append(
            {
                "set_index": int(idx),
                "set_type": kind,
                "n_words": int(len(words)),
                "n_geodesics": int(len(geodesics)),
                "fro_norm": fro_norm,
                "loss": loss,
                "builder": build_rep.get("builder", "basic"),
                "eigh_success": bool(eig_rep.get("eigh_success", False)),
            }
        )
        operators.append(H)
        eigens.append(eigvals)
        losses.append(loss)

    op_dists: List[float] = []
    spec_dists: List[float] = []
    for i in range(len(operators)):
        for j in range(i + 1, len(operators)):
            Ha = operators[i]
            Hb = operators[j]
            denom = float(np.linalg.norm(Ha, ord="fro")) + EPS
            op_dists.append(float(np.linalg.norm(Ha - Hb, ord="fro") / denom))

            ea = eigens[i]
            eb = eigens[j]
            k = min(len(ea), len(eb))
            if k > 0:
                spec_denom = float(np.linalg.norm(ea[:k])) + EPS
                spec_dists.append(float(np.linalg.norm(ea[:k] - eb[:k]) / spec_denom))

    finite_losses = np.asarray([x for x in losses if np.isfinite(x)], dtype=DTYPE)
    operator_distance_mean = float(np.mean(op_dists)) if op_dists else float("nan")
    operator_distance_max = float(np.max(op_dists)) if op_dists else float("nan")
    spectrum_distance_mean = float(np.mean(spec_dists)) if spec_dists else float("nan")
    spectrum_distance_max = float(np.max(spec_dists)) if spec_dists else float("nan")
    loss_std = float(np.std(finite_losses)) if finite_losses.size else float("nan")
    loss_range = float(np.max(finite_losses) - np.min(finite_losses)) if finite_losses.size else float("nan")
    diagnosis = _diagnosis(operator_distance_mean, spectrum_distance_mean, loss_std)

    report = {
        "operator_distance_mean": operator_distance_mean,
        "operator_distance_max": operator_distance_max,
        "spectrum_distance_mean": spectrum_distance_mean,
        "spectrum_distance_max": spectrum_distance_max,
        "loss_std": loss_std,
        "loss_range": loss_range,
        "diagnosis": diagnosis,
        "set_reports": rows,
    }

    with (out_dir / "operator_sensitivity_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    with (out_dir / "operator_sensitivity_matrix.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["set_index", "set_type", "n_words", "n_geodesics", "fro_norm", "loss", "builder", "eigh_success"])
        for row in rows:
            writer.writerow(
                [
                    row["set_index"],
                    row["set_type"],
                    row["n_words"],
                    row["n_geodesics"],
                    row["fro_norm"],
                    row["loss"],
                    row["builder"],
                    row["eigh_success"],
                ]
            )

    summary_lines = [
        "# Operator Sensitivity Summary",
        "",
        f"- operator_distance_mean: `{operator_distance_mean:.6g}`" if np.isfinite(operator_distance_mean) else "- operator_distance_mean: `nan`",
        f"- operator_distance_max: `{operator_distance_max:.6g}`" if np.isfinite(operator_distance_max) else "- operator_distance_max: `nan`",
        f"- spectrum_distance_mean: `{spectrum_distance_mean:.6g}`" if np.isfinite(spectrum_distance_mean) else "- spectrum_distance_mean: `nan`",
        f"- spectrum_distance_max: `{spectrum_distance_max:.6g}`" if np.isfinite(spectrum_distance_max) else "- spectrum_distance_max: `nan`",
        f"- loss_std: `{loss_std:.6g}`" if np.isfinite(loss_std) else "- loss_std: `nan`",
        f"- loss_range: `{loss_range:.6g}`" if np.isfinite(loss_range) else "- loss_range: `nan`",
        "",
        "## Diagnosis",
        diagnosis,
        "",
        "## Set Overview",
    ]
    for row in rows:
        summary_lines.append(
            f"- set `{row['set_index']}` `{row['set_type']}`: "
            f"geodesics={row['n_geodesics']} fro_norm={row['fro_norm']:.6g} loss={row['loss']}"
        )
    summary_lines.append("")
    (out_dir / "operator_sensitivity_summary.md").write_text("\n".join(summary_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
