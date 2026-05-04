#!/usr/bin/env python3
"""
V13C: validate ACO candidate words — spectral loss, spacing, Wigner/GUE KS, Ramsey/Nijenhuis, commutator proxy.

Run from repo root (same operator path as ACO ``operator_builder=word_sensitive``):

  python3 scripts/validate_candidates_v13.py --zeros 128 --dim 128 --out_dir runs/v13_candidate_validation_word_sensitive
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Matplotlib reads MPLCONFIGDIR at import; default ~/.matplotlib may be unwritable (CI/sandbox).
if not os.environ.get("MPLCONFIGDIR"):
    _mpl_cfg = Path(ROOT) / ".mpl_cache"
    try:
        _mpl_cfg.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(_mpl_cfg)
    except OSError:
        pass

from core.artin_aco import resolve_zeros_cli  # noqa: E402
from core.artin_operator import sample_domain  # noqa: E402

try:
    from core.artin_operator_word_sensitive import build_word_sensitive_operator  # noqa: E402
except ImportError:  # pragma: no cover
    build_word_sensitive_operator = None  # type: ignore[misc, assignment]
from core.ramsey_nijenhuis import (  # noqa: E402
    nijenhuis_defect,
    ramsey_score_word,
    word_to_shift_operator,
)

_DTF = np.float64

DEFAULT_CANDIDATES: List[Dict[str, Any]] = [
    {"id": "A_seed3_best", "word": [4, 1, -1, -4, -1, -1, 1, 1]},
    {"id": "B_seed12_long", "word": [-6, 1, 2, -1, 4, 1, -1]},
    {"id": "C_seed6_long", "word": [4, 2, -3, -3, 1, -1]},
]


def _load_zeros(n: int) -> np.ndarray:
    """First ``n`` positive zeta ordinates: try file, else mpmath, else synthetic."""
    p = Path(ROOT) / "data" / "zeta_zeros.txt"
    if p.exists():
        z = np.loadtxt(str(p), dtype=_DTF)
        z = np.asarray(z, dtype=_DTF).reshape(-1)
        z = z[np.isfinite(z)]
        z = np.sort(z[z > 0.0])
        if int(z.size) >= int(n):
            return z[: int(n)].astype(_DTF, copy=False)
    try:
        import mpmath as mp  # type: ignore

        out: List[float] = []
        for j in range(1, int(n) + 1):
            out.append(float(mp.im(mp.zetazero(j))))
        return np.asarray(out, dtype=_DTF)
    except Exception:
        return resolve_zeros_cli(str(int(n)))


def _geodesic_entry_for_word(word: List[int]) -> Dict[str, Any]:
    """Single-geodesic dict aligned with ``ArtinACOTrainer._bank_top_geodesics`` shape."""
    w = [int(x) for x in word]
    return {
        "a_list": w,
        "length": float(max(1.0, len(w))),
        "is_hyperbolic": True,
        "primitive": True,
    }


def build_operator_aco_word_sensitive(
    word: List[int],
    *,
    n_points: int,
    eps: float,
    geo_weight: float,
    geo_sigma: float,
    potential_weight: float,
    seed: int,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Same operator path as ``ArtinACOTrainer`` when ``operator_builder=word_sensitive``:
    ``build_word_sensitive_operator`` with ACO defaults (kernel_normalization=max, laplacian_weight=1).
    """
    if build_word_sensitive_operator is None:
        raise RuntimeError(
            "core.artin_operator_word_sensitive.build_word_sensitive_operator is unavailable; "
            "cannot run word-sensitive validation."
        )
    Z = sample_domain(int(n_points), seed=int(seed))
    geodesics = [_geodesic_entry_for_word(word)]
    H, rep = build_word_sensitive_operator(
        z_points=Z,
        distances=None,
        geodesics=geodesics,
        eps=float(eps),
        geo_sigma=float(geo_sigma),
        kernel_normalization="max",
        laplacian_weight=1.0,
        geo_weight=float(geo_weight),
        potential_weight=float(potential_weight),
    )
    H = np.asarray(H, dtype=_DTF, copy=False)
    meta: Dict[str, Any] = {"word_sensitive_report": rep, "n_words_used": int(rep.get("n_words_used", 0))}
    return H, meta


def operator_matrix_sha256(H: np.ndarray) -> str:
    hc = np.ascontiguousarray(np.asarray(H, dtype=np.float64))
    return hashlib.sha256(hc.tobytes()).hexdigest()


def pairwise_operator_fro_dists(ids: List[str], mats: List[np.ndarray]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    n = len(ids)
    for i in range(n):
        for j in range(i + 1, n):
            if mats[i] is None or mats[j] is None:
                continue
            d = float(np.linalg.norm(mats[i] - mats[j], ord="fro"))
            out.append({"id_a": ids[i], "id_b": ids[j], "fro_norm_diff": d})
    return out


def warn_identical_operator_hashes(results: List[Dict[str, Any]]) -> None:
    by_hash: Dict[str, List[Tuple[str, Tuple[int, ...]]]] = defaultdict(list)
    for r in results:
        h = r.get("operator_hash")
        if not h or not isinstance(h, str):
            continue
        wid = str(r.get("id", ""))
        w = r.get("word_used") or r.get("word")
        if not isinstance(w, list):
            continue
        key_w = tuple(int(x) for x in w)
        by_hash[h].append((wid, key_w))
    for hsh, entries in by_hash.items():
        distinct_words = {e[1] for e in entries}
        if len(distinct_words) > 1:
            parts = [f"{e[0]}:{list(e[1])}" for e in entries]
            print(f"[validation-warning] identical_operator_hash for hash={hsh} :: {' | '.join(parts)}")


def _symmetrize_torch(H: np.ndarray) -> torch.Tensor:
    t = torch.tensor(H, dtype=torch.float64, device="cpu")
    return 0.5 * (t + t.T)


def _eigvalsh_safe(H_t: torch.Tensor) -> Tuple[np.ndarray, Optional[str]]:
    try:
        if not torch.isfinite(H_t).all():
            return np.array([], dtype=_DTF), "nonfinite_operator_matrix"
        w = torch.linalg.eigvalsh(H_t)
        w = w.detach().cpu().numpy().astype(_DTF, copy=False)
        w = np.sort(w)
        if not np.isfinite(w).all():
            return np.array([], dtype=_DTF), "nonfinite_eigenvalues"
        return w, None
    except Exception as ex:
        return np.array([], dtype=_DTF), repr(ex)


def wigner_gue_cdf(s: np.ndarray) -> np.ndarray:
    s = np.maximum(0.0, np.asarray(s, dtype=np.float64))
    return 1.0 - np.exp(-np.pi * s * s / 4.0)


def wigner_gue_pdf(s: np.ndarray) -> np.ndarray:
    s = np.maximum(0.0, np.asarray(s, dtype=np.float64))
    return (np.pi / 2.0) * s * np.exp(-np.pi * s * s / 4.0)


def ks_against_wigner_gue(spacings_unit_mean: np.ndarray) -> float:
    """One-sample KS distance vs Wigner surmise CDF F(s)=1-exp(-pi s^2/4), s>=0."""
    x = np.sort(spacings_unit_mean[np.isfinite(spacings_unit_mean) & (spacings_unit_mean >= 0)])
    n = int(x.size)
    if n < 2:
        return float("nan")
    F = wigner_gue_cdf(x)
    i = np.arange(1, n + 1, dtype=np.float64)
    d_minus = np.max(np.abs(i / n - F))
    d_plus = np.max(np.abs((i - 1.0) / n - F))
    return float(max(d_minus, d_plus))


def normalized_gaps(sorted_eigs: np.ndarray) -> np.ndarray:
    e = np.asarray(sorted_eigs, dtype=np.float64).reshape(-1)
    if e.size < 2:
        return np.array([], dtype=np.float64)
    g = np.diff(e)
    m = float(np.mean(g))
    if not math.isfinite(m) or m <= 0.0:
        return np.array([], dtype=np.float64)
    return g / m


def spacing_mse_normalized(eig_slice: np.ndarray, zero_slice: np.ndarray) -> float:
    ge = normalized_gaps(np.sort(eig_slice))
    gz = normalized_gaps(np.sort(zero_slice))
    k = min(ge.size, gz.size)
    if k <= 0:
        return float("nan")
    return float(np.mean((ge[:k] - gz[:k]) ** 2))


def commutator_proxy(word: List[int], dim: int, eig_sorted: np.ndarray) -> float:
    """||[A,D]||_F / max(1, ||A||_F ||D||_F) with D diagonal from sorted eigenvalues (or 1..dim)."""
    d = int(dim)
    A = word_to_shift_operator([int(x) for x in word], d, device="cpu").to(dtype=torch.float64)
    ev = np.asarray(eig_sorted, dtype=np.float64).reshape(-1)
    if ev.size >= d:
        diag = ev[:d]
    else:
        diag = np.arange(1, d + 1, dtype=np.float64)
    Dm = torch.diag(torch.tensor(diag, dtype=torch.float64))
    C = A @ Dm - Dm @ A
    nf = float(torch.linalg.matrix_norm(C, ord="fro").item())
    na = float(torch.linalg.matrix_norm(A, ord="fro").item())
    nd = float(torch.linalg.matrix_norm(Dm, ord="fro").item())
    den = max(1.0, na * nd)
    return float(nf / den)


def validate_one(
    cand: Dict[str, Any],
    *,
    zeros: np.ndarray,
    dim: int,
    eps: float,
    seed: int,
    geo_weight: float,
    geo_sigma: float,
    potential_weight: float,
) -> Tuple[Dict[str, Any], Optional[np.ndarray]]:
    cid = str(cand.get("id", "unknown"))
    word = [int(x) for x in cand["word"]]
    row: Dict[str, Any] = {
        "id": cid,
        "word": word,
        "word_used": list(word),
        "operator_builder": "word_sensitive",
        "dim": int(dim),
        "operator_hash": None,
        "operator_fro_norm": None,
        "operator_trace": None,
        "operator_n_words_used": None,
        "operator_used_geodesics": None,
        "eig_error": None,
        "eig_min": None,
        "eig_max": None,
        "n_eig": 0,
        "n_zeros": int(zeros.size),
        "k_align": 0,
        "spectral_raw_mse": float("nan"),
        "spectral_log_mse": float("nan"),
        "spacing_mse_normalized": float("nan"),
        "spacing_mean": float("nan"),
        "spacing_std": float("nan"),
        "ks_wigner": float("nan"),
        "ramsey_score": float("nan"),
        "nijenhuis_defect": float("nan"),
        "comm_norm_proxy": float("nan"),
    }
    H_out: Optional[np.ndarray] = None
    try:
        H, meta = build_operator_aco_word_sensitive(
            word,
            n_points=int(dim),
            eps=float(eps),
            geo_weight=float(geo_weight),
            geo_sigma=float(geo_sigma),
            potential_weight=float(potential_weight),
            seed=int(seed),
        )
        H_out = np.asarray(H, dtype=_DTF, copy=True)
        row["operator_n_words_used"] = int(meta.get("n_words_used", 0))
        row["operator_used_geodesics"] = int(meta.get("n_words_used", 0))
        row["operator_hash"] = operator_matrix_sha256(H)
        row["operator_fro_norm"] = float(np.linalg.norm(H, ord="fro"))
        row["operator_trace"] = float(np.trace(H))
        Ht = _symmetrize_torch(H)
        eig, err = _eigvalsh_safe(Ht)
        row["eig_error"] = err
        row["n_eig"] = int(eig.size)
        if eig.size == 0:
            return row
        row["eig_min"] = float(np.min(eig))
        row["eig_max"] = float(np.max(eig))
        z = np.asarray(zeros, dtype=_DTF).reshape(-1)
        z = z[np.isfinite(z)]
        k = int(min(int(eig.size), int(z.size)))
        row["k_align"] = k
        if k >= 1:
            e_k = np.sort(eig[:k])
            z_k = np.sort(z[:k])
            raw_mse = float(np.mean((e_k - z_k) ** 2))
            row["spectral_raw_mse"] = raw_mse
            row["spectral_log_mse"] = float(np.log1p(max(0.0, raw_mse)))
        if k >= 2:
            sp = np.diff(np.sort(eig[:k]))
            row["spacing_mean"] = float(np.mean(sp))
            row["spacing_std"] = float(np.std(sp))
            row["spacing_mse_normalized"] = spacing_mse_normalized(eig[:k], z[:k])
            nu = normalized_gaps(np.sort(eig[:k]))
            if nu.size >= 2:
                row["ks_wigner"] = ks_against_wigner_gue(nu)
        row["ramsey_score"] = float(ramsey_score_word(word))
        N = word_to_shift_operator(word, int(dim), device="cpu")
        row["nijenhuis_defect"] = float(nijenhuis_defect(N).detach().cpu().item())
        row["comm_norm_proxy"] = float(commutator_proxy(word, int(dim), eig))
    except Exception as ex:
        row["eig_error"] = repr(ex)
    return row, H_out


def _plot_spacing(
    out_path: Path,
    eig_sorted_slice: np.ndarray,
    *,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    e = np.asarray(eig_sorted_slice, dtype=np.float64).reshape(-1)
    if e.size < 2:
        return
    nu = normalized_gaps(np.sort(e))
    if nu.size == 0:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(nu, bins=min(40, max(8, nu.size)), density=True, alpha=0.65, color="steelblue", label="normalized gaps")
    smax = float(max(3.0, np.max(nu) * 1.1))
    xs = np.linspace(0.0, smax, 300)
    ax.plot(xs, wigner_gue_pdf(xs), "r-", lw=2, label="Wigner surmise (GUE) pdf")
    ax.set_xlabel("normalized spacing")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _sort_key(r: Dict[str, Any]) -> Tuple[float, float, float]:
    def f(x: Any) -> float:
        v = float(x) if x is not None and np.isfinite(float(x)) else float("inf")

        return v

    return (f(r.get("spectral_log_mse")), f(r.get("ks_wigner")), f(r.get("nijenhuis_defect")))


def main() -> None:
    ap = argparse.ArgumentParser(description="V13C candidate validation (spectral + GUE + Ramsey/Nijenhuis; not an RH proof).")
    ap.add_argument("--zeros", type=int, default=128, help="Number of zeta ordinates (file or mpmath/synthetic).")
    ap.add_argument("--out_dir", type=str, default="runs/v13_candidate_validation")
    ap.add_argument("--dim", type=int, default=128, help="Operator size n_points (H is dim x dim).")
    ap.add_argument("--eps", type=float, default=0.6, help="Laplacian eps (matches ACO --op_eps).")
    ap.add_argument("--geo_weight", type=float, default=10.0, help="Matches ACO --geo_weight for word_sensitive.")
    ap.add_argument("--geo_sigma", type=float, default=0.6, help="Matches ACO --geo_sigma for word_sensitive.")
    ap.add_argument(
        "--potential_weight",
        type=float,
        default=0.25,
        help="Matches ACO --potential_weight for word_sensitive.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--candidates_json", type=str, default="", help="Optional JSON list of {id, word: [...]}.")
    ap.add_argument(
        "--save_plots",
        type=str,
        default="true",
        help="Write spacing histograms (matplotlib only).",
    )
    args = ap.parse_args()

    if build_word_sensitive_operator is None:
        raise SystemExit(
            "word_sensitive operator builder is required for V13C validation; "
            "import core.artin_operator_word_sensitive failed."
        )

    save_plots = str(args.save_plots).lower() in ("1", "true", "yes", "y")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    zeros = _load_zeros(int(args.zeros))
    if zeros.size < int(args.dim):
        raise ValueError(f"need at least --dim zeros ({args.dim}), got {zeros.size}")

    candidates: List[Dict[str, Any]] = list(DEFAULT_CANDIDATES)
    if str(args.candidates_json).strip():
        p = Path(args.candidates_json)
        if not p.is_file():
            raise FileNotFoundError(f"candidates_json not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if not isinstance(loaded, list):
            raise ValueError("candidates_json must be a JSON list of {id, word}")
        candidates = loaded

    results: List[Dict[str, Any]] = []
    H_mats: List[Optional[np.ndarray]] = []
    for c in candidates:
        if not isinstance(c, dict) or "word" not in c:
            continue
        r, Hm = validate_one(
            c,
            zeros=zeros,
            dim=int(args.dim),
            eps=float(args.eps),
            seed=int(args.seed),
            geo_weight=float(args.geo_weight),
            geo_sigma=float(args.geo_sigma),
            potential_weight=float(args.potential_weight),
        )
        results.append(r)
        H_mats.append(Hm)

    ids_list = [str(r["id"]) for r in results]
    pairwise = pairwise_operator_fro_dists(ids_list, H_mats)
    warn_identical_operator_hashes(results)

    results_sorted = sorted(results, key=_sort_key)

    payload = {
        "zeros_n": int(zeros.size),
        "dim": int(args.dim),
        "operator_builder": "word_sensitive",
        "eps": float(args.eps),
        "geo_weight": float(args.geo_weight),
        "geo_sigma": float(args.geo_sigma),
        "potential_weight": float(args.potential_weight),
        "seed": int(args.seed),
        "pairwise_operator_distances": pairwise,
        "candidates": results_sorted,
    }
    with open(out_dir / "validation_results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=True)

    cols = [
        "id",
        "operator_hash",
        "operator_fro_norm",
        "operator_trace",
        "word_used",
        "spectral_log_mse",
        "ks_wigner",
        "nijenhuis_defect",
        "spectral_raw_mse",
        "ramsey_score",
        "comm_norm_proxy",
        "spacing_mse_normalized",
        "spacing_mean",
        "spacing_std",
        "k_align",
        "n_eig",
        "eig_min",
        "eig_max",
        "eig_error",
    ]
    with open(out_dir / "validation_results.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in results_sorted:
            flat = {k: r.get(k) for k in cols}
            wu = flat.get("word_used")
            if isinstance(wu, list):
                flat["word_used"] = json.dumps(wu)
            w.writerow(flat)

    if save_plots:
        for r in results:
            cid = str(r["id"])
            safe = re.sub(r"[^a-zA-Z0-9_.-]+", "_", cid)
            word = r.get("word_used") or r.get("word") or []
            try:
                H, _ = build_operator_aco_word_sensitive(
                    [int(x) for x in word],
                    n_points=int(args.dim),
                    eps=float(args.eps),
                    geo_weight=float(args.geo_weight),
                    geo_sigma=float(args.geo_sigma),
                    potential_weight=float(args.potential_weight),
                    seed=int(args.seed),
                )
                Ht = _symmetrize_torch(H)
                eig, _ = _eigvalsh_safe(Ht)
                k = int(r.get("k_align") or 0)
                if eig.size >= 2 and k >= 2:
                    _plot_spacing(
                        out_dir / f"spacing_{safe}.png",
                        np.sort(eig[:k]),
                        title=f"{cid}: normalized eig spacings (k={k})",
                    )
            except Exception:
                pass

    print("V13C validation — sorted by spectral_log_mse, ks_wigner, nijenhuis_defect (ascending; nan last)\n")
    for r in results_sorted:
        print(
            f"  {r['id']}: log_mse={r['spectral_log_mse']!s} ks={r['ks_wigner']!s} "
            f"nij={r['nijenhuis_defect']!s} ramsey={r['ramsey_score']!s} comm={r['comm_norm_proxy']!s} "
            f"k={r['k_align']} eig_err={r['eig_error']!s}"
        )
    print(f"\nWrote {out_dir / 'validation_results.json'}")
    print(f"Wrote {out_dir / 'validation_results.csv'}")
    if save_plots:
        print(f"Plots under {out_dir} (spacing_*.png)")


if __name__ == "__main__":
    main()
