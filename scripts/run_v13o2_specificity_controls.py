#!/usr/bin/env python3
"""
V13O.2: specificity controls — target-blind V modes, train/test transfer, destroyed correlations,
long-range statistics. Computational evidence only; not a proof of RH.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

_DTF = np.float64
ALPHA = 0.5
EPS_BUILDER = 0.6
GEO_WEIGHT = 10.0
CLIP_DEFAULT = (0.5, 99.5)
DIAG_SHIFT = 1e-6
ABS_CAP = 5.0

PRIMARY_WORD = [-4, -2, -4, -2, -2, -1, -1]
REJECTED_WORD = [6, 4, -1, 1, 1, 1, 4]

DIM_K_TRAIN = {64: 45, 128: 96, 256: 128}

DIM_PARAM: Dict[int, Dict[str, Any]] = {
    64: {
        "lambda_p": 2.121320343559643,
        "geo_sigma": 0.5045378491522287,
        "smooth_sigma": 0.7071067811865476,
        "beta0": 0.3,
        "tau": 200.0,
        "beta_floor": 0.03,
    },
    128: {
        "lambda_p": 3.0,
        "geo_sigma": 0.6,
        "smooth_sigma": 1.0,
        "beta0": 0.3,
        "tau": 500.0,
        "beta_floor": 0.03,
    },
    256: {
        "lambda_p": 2.121320343559643,
        "geo_sigma": 0.5045378491522287,
        "smooth_sigma": 1.25,
        "beta0": 0.3,
        "tau": 300.0,
        "beta_floor": 0.03,
    },
}

V_MODES_ALL = [
    "full_V",
    "target_blind_V",
    "density_only_V",
    "word_only_V",
    "phase_only_V",
    "frozen_V_after_5",
    "frozen_V_after_10",
    "very_weak_V",
    "weak_V",
    "smooth_V_strong",
]

TARGETS_ALL = [
    "real_zeta",
    "block_shuffled_zeta_block4",
    "block_shuffled_zeta_block8",
    "local_jitter_zeta_small",
    "local_jitter_zeta_medium",
    "density_matched_synthetic",
    "GUE_synthetic",
    "Poisson_synthetic",
    "reversed_zeta",
]

CONTROL_TARGETS_SPECIFICITY = [
    "block_shuffled_zeta_block4",
    "block_shuffled_zeta_block8",
    "local_jitter_zeta_small",
    "local_jitter_zeta_medium",
    "density_matched_synthetic",
    "GUE_synthetic",
    "Poisson_synthetic",
    "reversed_zeta",
]


def format_seconds(sec: float) -> str:
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _resolve(p: str) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = Path(ROOT) / path
    return path


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_pdflatex() -> Optional[str]:
    w = shutil.which("pdflatex")
    if w:
        return w
    for p in ("/Library/TeX/texbin/pdflatex", "/usr/local/texlive/2026/bin/universal-darwin/pdflatex"):
        if Path(p).is_file():
            return p
    return None


def try_pdflatex(tex_path: Path, out_dir: Path, pdf_basename: str) -> bool:
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
        return r.returncode == 0 and (out_dir / pdf_basename).is_file()
    except (OSError, subprocess.TimeoutExpired):
        return False


def latex_escape(s: str) -> str:
    t = str(s)
    t = t.replace("\\", "\\textbackslash{}")
    t = t.replace("{", "\\{").replace("}", "\\}")
    t = t.replace("_", "\\_")
    t = t.replace("%", "\\%")
    return t


def build_h(
    *,
    z_points: np.ndarray,
    word: List[int],
    v13l: Any,
    geo_sigma: float,
    laplacian_weight: float,
    geo_weight: float,
) -> np.ndarray:
    geodesics = [v13l.geodesic_entry_for_word([int(x) for x in word])]
    H_base, _ = v13l.build_h_base_no_potential(
        z_points=z_points,
        geodesics=geodesics,
        eps=float(EPS_BUILDER),
        geo_sigma=float(geo_sigma),
        geo_weight=float(geo_weight),
        laplacian_weight=float(laplacian_weight),
        distances=None,
        diag_shift=float(DIAG_SHIFT),
    )
    return np.asarray(H_base, dtype=_DTF, copy=True)


def gue_ord(n: int, rng: np.random.Generator) -> np.ndarray:
    a = rng.standard_normal((n, n))
    h = (a + a.T) / math.sqrt(2.0)
    w = np.sort(np.linalg.eigvalsh(h).astype(np.float64))
    w = w - float(w[0]) + 1e-6
    return w.astype(_DTF, copy=False)


def poisson_ord(n: int, rng: np.random.Generator) -> np.ndarray:
    s = rng.exponential(1.0, size=(n,)).astype(np.float64)
    c = np.cumsum(s)
    c = c - float(c[0]) + 1e-6
    return c.astype(_DTF, copy=False)


def block_shuffle_zeta(z_sorted: np.ndarray, k_train: int, block: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    need = 2 * int(k_train)
    z = np.sort(np.asarray(z_sorted[:need], dtype=np.float64).reshape(-1))
    gaps = np.diff(z)
    B = int(block)
    n_blocks = max(1, len(gaps) // B)
    blocks = [gaps[i * B : (i + 1) * B].copy() for i in range(n_blocks)]
    rest = gaps[n_blocks * B :].copy() if n_blocks * B < len(gaps) else np.zeros((0,), dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    order = np.arange(len(blocks))
    rng.shuffle(order)
    new_gaps = np.concatenate([blocks[int(i)] for i in order] + ([rest] if rest.size else []))
    recon = np.zeros(need, dtype=np.float64)
    recon[0] = float(z[0])
    for i in range(1, need):
        gidx = i - 1
        if gidx < new_gaps.size:
            recon[i] = recon[i - 1] + float(new_gaps[gidx])
        else:
            recon[i] = recon[i - 1] + 1e-3
    tr = recon[:k_train].astype(_DTF, copy=False)
    te = recon[k_train:need].astype(_DTF, copy=False)
    return tr, te


def local_jitter_zeta(z_sorted: np.ndarray, k_train: int, seed: int, amp_factor: float) -> Tuple[np.ndarray, np.ndarray]:
    need = 2 * int(k_train)
    z = np.sort(np.asarray(z_sorted[:need], dtype=np.float64).reshape(-1))
    gaps = np.diff(z)
    med = float(np.median(gaps)) if gaps.size else 1.0
    rng = np.random.default_rng(int(seed))
    noise = rng.normal(0.0, float(amp_factor) * med, size=gaps.shape)
    gaps2 = np.maximum(gaps + noise, 1e-9)
    gaps2 = gaps2 * (float(np.mean(gaps)) / max(float(np.mean(gaps2)), 1e-12))
    recon = np.zeros(need, dtype=np.float64)
    recon[0] = float(z[0])
    recon[1:] = z[0] + np.cumsum(gaps2)
    tr = recon[:k_train].astype(_DTF, copy=False)
    te = recon[k_train:need].astype(_DTF, copy=False)
    return tr, te


def density_matched_synthetic(z_sorted: np.ndarray, k_train: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    need = 2 * int(k_train)
    z = np.sort(np.asarray(z_sorted[:need], dtype=np.float64).reshape(-1))
    gaps = np.maximum(np.diff(z), 1e-9)
    inv = 1.0 / gaps
    env = np.convolve(inv, np.ones(5) / 5.0, mode="same")
    rng = np.random.default_rng(int(seed))
    g2 = rng.exponential(1.0, size=gaps.shape).astype(np.float64)
    g2 = g2 * (env / (np.mean(g2) / max(np.mean(gaps), 1e-12) + 1e-12))
    g2 = g2 * (float(np.sum(gaps)) / max(float(np.sum(g2)), 1e-12))
    recon = np.zeros(need, dtype=np.float64)
    recon[0] = float(z[0])
    recon[1:] = z[0] + np.cumsum(g2)
    tr = recon[:k_train].astype(_DTF, copy=False)
    te = recon[k_train:need].astype(_DTF, copy=False)
    return tr, te


def build_train_test_targets(
    name: str,
    z_sorted: np.ndarray,
    k_train: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, bool, str]:
    need = 2 * int(k_train)
    base = np.sort(np.asarray(z_sorted[:need], dtype=_DTF).reshape(-1))
    if name == "real_zeta":
        return base[:k_train].copy(), base[k_train:need].copy(), False, "sorted_zeta"
    if name == "reversed_zeta":
        rev = base[::-1].copy()
        return rev[:k_train].copy(), rev[k_train:need].copy(), True, "reversed_prefix"
    if name == "block_shuffled_zeta_block4":
        tr, te = block_shuffle_zeta(z_sorted, k_train, 4, seed)
        return tr, te, False, "block_shuffle_4"
    if name == "block_shuffled_zeta_block8":
        tr, te = block_shuffle_zeta(z_sorted, k_train, 8, seed + 1)
        return tr, te, False, "block_shuffle_8"
    if name == "local_jitter_zeta_small":
        tr, te = local_jitter_zeta(z_sorted, k_train, seed + 2, 0.05)
        return tr, te, False, "jitter_small"
    if name == "local_jitter_zeta_medium":
        tr, te = local_jitter_zeta(z_sorted, k_train, seed + 3, 0.15)
        return tr, te, False, "jitter_medium"
    if name == "density_matched_synthetic":
        tr, te = density_matched_synthetic(z_sorted, k_train, seed + 4)
        return tr, te, False, "density_matched"
    if name == "GUE_synthetic":
        g = gue_ord(need, np.random.default_rng(seed + 5))
        return g[:k_train].copy(), g[k_train:need].copy(), False, "GUE"
    if name == "Poisson_synthetic":
        p = poisson_ord(need, np.random.default_rng(seed + 6))
        return p[:k_train].copy(), p[k_train:need].copy(), False, "Poisson"
    raise ValueError(name)


def meets_transfer(
    *,
    finite: bool,
    sa: float,
    od: float,
    test_sp: float,
    test_ks: float,
) -> bool:
    return (
        finite
        and math.isfinite(sa)
        and sa <= 1e-12
        and math.isfinite(od)
        and od <= 1e-3
        and math.isfinite(test_sp)
        and test_sp <= 1.2
        and math.isfinite(test_ks)
        and test_ks <= 0.25
    )


def meets_train_accept(sp: float, ks: float, od: float, finite: bool, sa: float, eig_err: Optional[str]) -> bool:
    if eig_err:
        return False
    return (
        finite
        and math.isfinite(sa)
        and sa <= 1e-12
        and math.isfinite(od)
        and od <= 1e-3
        and math.isfinite(sp)
        and sp <= 1.2
        and math.isfinite(ks)
        and ks <= 0.25
    )


def meets_test_accept(sp: float, ks: float, finite: bool, sa: float, eig_err: Optional[str]) -> bool:
    if eig_err:
        return False
    return (
        finite
        and math.isfinite(sa)
        and sa <= 1e-12
        and math.isfinite(sp)
        and sp <= 1.2
        and math.isfinite(ks)
        and ks <= 0.25
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="V13O.2 specificity controls (computational evidence only).")
    ap.add_argument("--candidate_json", type=str, default="runs/v13_candidate_theorem/candidate_operators.json")
    ap.add_argument("--formula_json", type=str, default="runs/v13_operator_formula/formula_components_summary.json")
    ap.add_argument("--v13n_summary", type=str, default="runs/v13n_theorem_report/v13n_summary.json")
    ap.add_argument("--v13o1_results", type=str, default="runs/v13o1_oos_controls/v13o1_results.json")
    ap.add_argument("--out_dir", type=str, default="runs/v13o2_specificity_controls")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dims", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--max_iter", type=int, default=300)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--n_controls", type=int, default=30)
    ap.add_argument("--n_random_words", type=int, default=30)
    ap.add_argument("--n_random_sym", type=int, default=5)
    ap.add_argument("--progress_every", type=int, default=1)
    ap.add_argument("--smoke", action="store_true", help="Subset V-modes and targets for a quick run.")
    args = ap.parse_args()

    out_dir = _resolve(args.out_dir)
    ck_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ck_dir.mkdir(parents=True, exist_ok=True)

    import importlib.util

    def _load_v13_validate() -> Any:
        path = Path(ROOT) / "scripts" / "validate_candidates_v13.py"
        spec = importlib.util.spec_from_file_location("_v13_validate_o2", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load {path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_v13_validate_o2"] = mod
        spec.loader.exec_module(mod)
        return mod

    v = _load_v13_validate()
    from core import v13l_self_consistent as v13l
    from core import v13l1_stabilized as v13l1
    from core import v13o2_specificity as v2
    from core.artin_operator import sample_domain

    cand = _load_json(_resolve(args.candidate_json))
    form = _load_json(_resolve(args.formula_json))
    v13n = _load_json(_resolve(args.v13n_summary))
    v13o1 = _load_json(_resolve(args.v13o1_results))

    dims = [int(d) for d in args.dims if int(d) in DIM_K_TRAIN]
    if not dims:
        raise SystemExit("No valid dims in --dims (expected subset of 64,128,256).")

    max_k = max(DIM_K_TRAIN[d] for d in dims)
    z_pool = v._load_zeros(max(512, 2 * max_k))
    z_sorted = np.sort(np.asarray(z_pool, dtype=_DTF).reshape(-1))
    z_sorted = z_sorted[np.isfinite(z_sorted) & (z_sorted > 0.0)]

    v_modes = list(V_MODES_ALL)
    targets = list(TARGETS_ALL)
    n_rw = max(1, int(args.n_random_words))
    n_sym = max(1, int(args.n_random_sym))
    if args.smoke:
        v_modes = ["full_V", "target_blind_V", "word_only_V", "weak_V"]
        targets = [
            "real_zeta",
            "GUE_synthetic",
            "block_shuffled_zeta_block4",
            "local_jitter_zeta_small",
        ]

    def random_word(rng: np.random.Generator) -> List[int]:
        alphabet = list(range(-6, 0)) + list(range(1, 7))
        return [int(rng.choice(alphabet)) for _ in range(len(PRIMARY_WORD))]

    word_jobs_base: List[Dict[str, Any]] = [
        {"id": "primary_word_seed6", "word": list(PRIMARY_WORD), "lap": 1.0, "geo_w": GEO_WEIGHT, "kind": "word"},
        {"id": "rejected_word_seed17", "word": list(REJECTED_WORD), "lap": 1.0, "geo_w": GEO_WEIGHT, "kind": "word"},
        {"id": "ablate_K", "word": list(PRIMARY_WORD), "lap": 1.0, "geo_w": 0.0, "kind": "word"},
        {"id": "ablate_V", "word": list(PRIMARY_WORD), "lap": 1.0, "geo_w": GEO_WEIGHT, "kind": "ablate_V"},
        {"id": "ablate_L", "word": list(PRIMARY_WORD), "lap": 0.0, "geo_w": GEO_WEIGHT, "kind": "word"},
    ]
    for j in range(n_rw):
        word_jobs_base.append(
            {
                "id": "random_words_n30",
                "word": random_word(np.random.default_rng(int(args.seed) + 7919 + j)),
                "lap": 1.0,
                "geo_w": GEO_WEIGHT,
                "kind": "word",
                "sub": f"rw_{j}",
            }
        )

    jobs_per_dim = len(word_jobs_base) + n_sym
    total_configs = len(dims) * jobs_per_dim * len(v_modes) * len(targets)

    base_seed = int(args.seed)
    t0 = time.perf_counter()
    done = 0
    best_J_global = float("inf")
    accepted_ct = 0

    summary_rows: List[Dict[str, Any]] = []
    transfer_rows: List[Dict[str, Any]] = []
    longrange_rows: List[Dict[str, Any]] = []

    prog = max(1, int(args.progress_every))

    for dim in dims:
        k_train = DIM_K_TRAIN[dim]
        if z_sorted.size < 2 * k_train:
            raise RuntimeError(f"Need {2*k_train} zeros; got {z_sorted.size}.")
        cfg = DIM_PARAM[dim]
        Z = sample_domain(dim, seed=base_seed)
        fro_ref = float(
            np.linalg.norm(
                build_h(
                    z_points=Z,
                    word=list(PRIMARY_WORD),
                    v13l=v13l,
                    geo_sigma=float(cfg["geo_sigma"]),
                    laplacian_weight=1.0,
                    geo_weight=float(GEO_WEIGHT),
                ),
                ord="fro",
            )
        )
        sym_matrices: List[np.ndarray] = []
        for j in range(n_sym):
            r = np.random.default_rng(base_seed + dim * 104729 + j).standard_normal((dim, dim))
            S = 0.5 * (r + r.T)
            fs = float(np.linalg.norm(S, ord="fro"))
            sym_matrices.append((S * (fro_ref / max(fs, 1e-12))).astype(_DTF, copy=False))

        word_jobs: List[Dict[str, Any]] = list(word_jobs_base)
        for j in range(n_sym):
            word_jobs.append(
                {
                    "id": "random_symmetric_baseline",
                    "word": [],
                    "kind": "sym",
                    "sub": f"sym_{j}",
                    "matrix": sym_matrices[j],
                }
            )

        for wji, wj in enumerate(word_jobs):
            wid = wj["id"]
            if wj.get("kind") == "sym":
                H_op = np.asarray(wj["matrix"], dtype=_DTF, copy=True)
            elif wj["kind"] == "ablate_V":
                H_op = build_h(
                    z_points=Z,
                    word=list(wj["word"]),
                    v13l=v13l,
                    geo_sigma=float(cfg["geo_sigma"]),
                    laplacian_weight=float(wj["lap"]),
                    geo_weight=float(wj["geo_w"]),
                )
            else:
                H_op = build_h(
                    z_points=Z,
                    word=list(wj["word"]),
                    v13l=v13l,
                    geo_sigma=float(cfg["geo_sigma"]),
                    laplacian_weight=float(wj["lap"]),
                    geo_weight=float(wj["geo_w"]),
                )

            for vm in v_modes:
                smooth_use = float(cfg["smooth_sigma"])
                clip_lo, clip_hi = float(CLIP_DEFAULT[0]), float(CLIP_DEFAULT[1])
                if vm == "smooth_V_strong":
                    smooth_use *= 4.0
                    clip_lo, clip_hi = 2.0, 98.0

                for tg in targets:
                    done += 1
                    elapsed = time.perf_counter() - t0
                    eta = elapsed / max(done, 1) * max(total_configs - done, 0)
                    if done == 1 or done == total_configs or done % prog == 0:
                        print(
                            f"[v13o2-summary] done={done}/{total_configs} accepted={accepted_ct} "
                            f"best_J_min={best_J_global if math.isfinite(best_J_global) else float('nan'):.4g} "
                            f"elapsed={format_seconds(elapsed)} eta={format_seconds(eta)}",
                            flush=True,
                        )

                    seed_t = base_seed + dim * 100003 + hash(vm) % 9973 + hash(tg) % 9973 + wji * 17
                    tr, te, train_ordered, wnote = build_train_test_targets(tg, z_sorted, k_train, seed_t)
                    z_train_m = np.sort(np.asarray(tr, dtype=_DTF).reshape(-1))
                    z_test_m = np.sort(np.asarray(te, dtype=_DTF).reshape(-1))
                    k_al = int(min(dim, k_train, z_train_m.size, z_test_m.size))

                    print(
                        f"[v13o2] ({done}/{total_configs}) dim={dim} mode=train_test_split target={tg} V={vm} "
                        f"word={wid} elapsed={elapsed:.1f}s eta={format_seconds(eta)} best_J={best_J_global if math.isfinite(best_J_global) else float('nan'):.4g}",
                        flush=True,
                    )

                    eig_err: Optional[str] = None
                    try:
                        if wj["kind"] == "ablate_V":
                            H_fin = np.asarray(H_op, dtype=_DTF, copy=True)
                            train_out = {
                                "H_final": H_fin,
                                "H_base": H_fin,
                                "V_diag_last": np.zeros((dim,), _DTF),
                                "meta": {"converged_operator": True, "n_iter": 0, "eig_error": None},
                                "spectral_log_mse_train": float("nan"),
                                "spacing_mse_normalized_train": float("nan"),
                                "ks_wigner_train": float("nan"),
                                "operator_diff_final": 0.0,
                            }
                            try:
                                eig_tr = np.sort(np.linalg.eigvalsh(0.5 * (H_fin + H_fin.T))).astype(_DTF)
                                from core.v13m1_renormalized import spectral_metrics_window as smw

                                m0 = smw(
                                    eig_tr,
                                    z_train_m,
                                    k_al,
                                    spacing_fn=v.spacing_mse_normalized,
                                    ks_fn=v.ks_against_wigner_gue,
                                    norm_gaps_fn=v.normalized_gaps,
                                )
                                train_out["spectral_log_mse_train"] = float(m0["spectral_log_mse"])
                                train_out["spacing_mse_normalized_train"] = float(m0["spacing_mse_normalized"])
                                train_out["ks_wigner_train"] = float(m0["ks_wigner"])
                            except Exception as ex:
                                eig_err = str(ex)
                        else:
                            train_out = v2.train_v13o2_cell(
                                H_base=np.asarray(H_op, dtype=_DTF, copy=True),
                                z_pool_sorted=z_sorted,
                                train_ordinates=tr,
                                train_ordered=train_ordered,
                                dim=dim,
                                k_train=k_train,
                                alpha=float(ALPHA),
                                lambda_p_dim=float(cfg["lambda_p"]),
                                beta0=float(cfg["beta0"]),
                                tau_beta=float(cfg["tau"]),
                                beta_floor=float(cfg["beta_floor"]),
                                smooth_sigma_dim=smooth_use,
                                clip_lo=clip_lo,
                                clip_hi=clip_hi,
                                diag_shift=float(DIAG_SHIFT),
                                abs_cap_factor=float(ABS_CAP),
                                zeros_train_metric=z_train_m,
                                spacing_fn=v.spacing_mse_normalized,
                                ks_fn=v.ks_against_wigner_gue,
                                norm_gaps_fn=v.normalized_gaps,
                                max_iter=int(args.max_iter),
                                tol=float(args.tol),
                                v_mode=vm,
                                z_points=np.asarray(Z, dtype=np.complex128),
                                word=list(wj["word"]),
                                on_train_iter=None,
                            )
                            eig_err = train_out.get("meta", {}).get("eig_error")
                            if isinstance(eig_err, str):
                                pass
                            else:
                                eig_err = None
                    except Exception as ex:
                        eig_err = str(ex)
                        train_out = {
                            "H_final": np.asarray(H_op, dtype=_DTF, copy=True),
                            "H_base": np.asarray(H_op, dtype=_DTF, copy=True),
                            "V_diag_last": np.zeros((dim,), _DTF),
                            "meta": {"eig_error": eig_err},
                            "spectral_log_mse_train": float(v2.LARGE_PEN),
                            "spacing_mse_normalized_train": float(v2.LARGE_PEN),
                            "ks_wigner_train": float(v2.LARGE_PEN),
                            "operator_diff_final": float(v2.LARGE_PEN),
                        }

                    H_fin = np.asarray(train_out["H_final"], dtype=_DTF, copy=True)
                    try:
                        eig = np.sort(np.linalg.eigvalsh(0.5 * (H_fin + H_fin.T))).astype(_DTF)
                    except Exception as ex:
                        eig_err = eig_err or str(ex)
                        eig = np.full((dim,), np.nan, dtype=_DTF)

                    finite = bool(np.isfinite(H_fin).all() and np.isfinite(eig).all())
                    sa = float(v2.self_adjointness_fro(H_fin))
                    od = float(train_out.get("operator_diff_final", float("nan")))

                    lr_tr = v2.long_range_bundle(
                        eig, z_train_m, k_al, spacing_fn=v.spacing_mse_normalized, ks_fn=v.ks_against_wigner_gue, norm_gaps_fn=v.normalized_gaps
                    )
                    lr_te = v2.long_range_bundle(
                        eig, z_test_m, k_al, spacing_fn=v.spacing_mse_normalized, ks_fn=v.ks_against_wigner_gue, norm_gaps_fn=v.normalized_gaps
                    )
                    J_tr = v2.pareto_J_v13o2(lr_tr)
                    J_te = v2.pareto_J_v13o2(lr_te)
                    if math.isfinite(J_te) and J_te < best_J_global:
                        best_J_global = float(J_te)

                    acc_tr = meets_train_accept(
                        lr_tr["spacing_mse_normalized"],
                        lr_tr["ks_wigner"],
                        od,
                        finite,
                        sa,
                        eig_err,
                    )
                    acc_te = meets_test_accept(
                        lr_te["spacing_mse_normalized"],
                        lr_te["ks_wigner"],
                        finite,
                        sa,
                        eig_err,
                    )
                    acc_xfer = meets_transfer(
                        finite=finite,
                        sa=sa,
                        od=od,
                        test_sp=lr_te["spacing_mse_normalized"],
                        test_ks=lr_te["ks_wigner"],
                    )
                    if acc_xfer:
                        accepted_ct += 1
                        h = hashlib.sha256(
                            f"{dim}|{vm}|{tg}|{wid}|{wj.get('sub', '')}".encode("utf-8")
                        ).hexdigest()[:10]
                        fn = ck_dir / f"checkpoint_dim{dim}_{vm}_{tg}_{wid}_{h}.npy"
                        try:
                            np.save(str(fn), H_fin)
                        except OSError:
                            pass

                    gap_sp = float(lr_te["spacing_mse_normalized"] - lr_tr["spacing_mse_normalized"])
                    gap_sl = float(lr_te["spectral_log_mse"] - lr_tr["spectral_log_mse"])
                    gap_ks = float(lr_te["ks_wigner"] - lr_tr["ks_wigner"])

                    transfer_rows.append(
                        {
                            "dim": dim,
                            "V_mode": vm,
                            "word_group": wid,
                            "target_group": tg,
                            "train_k": k_train,
                            "test_k": k_train,
                            "train_spectral_log_mse": lr_tr["spectral_log_mse"],
                            "test_spectral_log_mse": lr_te["spectral_log_mse"],
                            "gap_spectral": gap_sl,
                            "train_spacing_mse": lr_tr["spacing_mse_normalized"],
                            "test_spacing_mse": lr_te["spacing_mse_normalized"],
                            "gap_spacing": gap_sp,
                            "train_ks": lr_tr["ks_wigner"],
                            "test_ks": lr_te["ks_wigner"],
                            "gap_ks": gap_ks,
                            "operator_diff_final": od,
                            "accepted_train": acc_tr,
                            "accepted_test": acc_te,
                            "accepted_transfer": acc_xfer,
                            "J_train": J_tr,
                            "J_test": J_te,
                            "window_note": wnote,
                        }
                    )

                    longrange_rows.append(
                        {
                            "dim": dim,
                            "target_group": tg,
                            "V_mode": vm,
                            "word_group": wid,
                            "number_variance_l2": lr_te["number_variance_l2"],
                            "spectral_rigidity_proxy": lr_te["spectral_rigidity_proxy"],
                            "run_length_score": lr_te["run_length_score"],
                            "two_point_correlation_l2": lr_te["two_point_correlation_l2"],
                            "nearest_neighbor_spacing_mse": lr_te["nearest_neighbor_spacing_mse"],
                        }
                    )

                    summary_rows.append(
                        {
                            "dim": dim,
                            "V_mode": vm,
                            "word_group": wid,
                            "target_group": tg,
                            "finite": finite,
                            "self_adjointness_fro": sa,
                            "operator_diff_final": od,
                            "eig_error": eig_err or "",
                            "J_train": J_tr,
                            "J_test": J_te,
                            "accepted_transfer": acc_xfer,
                            **{f"lr_train_{k}": v for k, v in lr_tr.items()},
                            **{f"lr_test_{k}": v for k, v in lr_te.items()},
                        }
                    )

    n_ctrl_spec = len(CONTROL_TARGETS_SPECIFICITY)
    specificity_rows: List[Dict[str, Any]] = []
    for dim in dims:
        for vm in v_modes:
            for wj in word_jobs:
                wid = wj["id"]
                j_real = [
                    float(r["J_test"])
                    for r in transfer_rows
                    if int(r["dim"]) == dim and r["V_mode"] == vm and r["word_group"] == wid and r["target_group"] == "real_zeta"
                ]
                j_ctrl = [
                    float(r["J_test"])
                    for r in transfer_rows
                    if int(r["dim"]) == dim
                    and r["V_mode"] == vm
                    and r["word_group"] == wid
                    and r["target_group"] in CONTROL_TARGETS_SPECIFICITY
                ]
                if not j_real:
                    continue
                jr = float(np.median(np.asarray(j_real, dtype=np.float64)))
                jc = float(np.median(np.asarray(j_ctrl, dtype=np.float64))) if j_ctrl else float(v2.LARGE_PEN)
                score = jr - jc
                margin = jc - jr
                pool = j_ctrl + [jr]
                rank = int(sum(1 for x in pool if x < jr) + 1)
                thr_rank = max(3, int(0.1 * max(n_ctrl_spec, 1)))
                spass = bool(margin >= 0.5 and rank <= thr_rank)
                specificity_rows.append(
                    {
                        "dim": dim,
                        "V_mode": vm,
                        "word_group": wid,
                        "J_real_test": jr,
                        "median_J_controls_test": jc,
                        "specificity_score": score,
                        "specificity_margin": margin,
                        "real_rank_among_controls": rank,
                        "n_controls": n_ctrl_spec,
                        "specificity_pass": spass,
                    }
                )

    def finite_sa_all() -> bool:
        for r in summary_rows:
            if not r.get("finite"):
                return False
            if float(r.get("self_adjointness_fro", 1.0)) > 1e-12:
                return False
        return bool(summary_rows)

    def median_J_primary_full(dim_: int, tg: str) -> float:
        xs = [
            float(r["J_test"])
            for r in transfer_rows
            if int(r["dim"]) == dim_
            and r["target_group"] == tg
            and r["word_group"] == "primary_word_seed6"
            and r["V_mode"] == "full_V"
        ]
        if not xs:
            return float("nan")
        return float(np.median(np.asarray(xs, dtype=np.float64)))

    def gue_poisson_flags() -> Tuple[bool, bool]:
        g_ok = True
        p_ok = True
        for d in dims:
            mr = median_J_primary_full(d, "real_zeta")
            has_gue = any(
                int(r["dim"]) == d and r["target_group"] == "GUE_synthetic" for r in transfer_rows
            )
            has_poi = any(
                int(r["dim"]) == d and r["target_group"] == "Poisson_synthetic" for r in transfer_rows
            )
            if has_gue:
                mg = median_J_primary_full(d, "GUE_synthetic")
                if not (math.isfinite(mr) and math.isfinite(mg) and mr + 0.5 < mg):
                    g_ok = False
            if has_poi:
                mp = median_J_primary_full(d, "Poisson_synthetic")
                if not (math.isfinite(mr) and math.isfinite(mp) and mr + 0.5 < mp):
                    p_ok = False
        return g_ok, p_ok

    gue_ok, poi_ok = gue_poisson_flags()

    target_blind_survives = any(
        r.get("V_mode") in ("target_blind_V", "density_only_V", "word_only_V", "phase_only_V") and r.get("accepted_transfer")
        for r in transfer_rows
    )

    spec_pass_dims: Dict[int, bool] = {}
    for d in dims:
        spec_pass_dims[d] = any(
            bool(s.get("specificity_pass"))
            and s.get("word_group") == "primary_word_seed6"
            and s.get("V_mode") == "full_V"
            for s in specificity_rows
            if int(s["dim"]) == d
        )

    primary_transfer_blind = any(
        r.get("word_group") == "primary_word_seed6"
        and r.get("V_mode") == "target_blind_V"
        and r.get("accepted_transfer")
        for r in transfer_rows
    )

    overall_specificity_pass = (
        all(spec_pass_dims.get(d, False) for d in dims)
        and target_blind_survives
        and gue_ok
        and poi_ok
    )

    if overall_specificity_pass:
        conclusion = "specificity_pass"
    elif any(spec_pass_dims.values()) or target_blind_survives:
        conclusion = "partial_specificity"
    else:
        conclusion = "specificity_failed_universal_fitter"

    def margin_positive_primary_full(d: int) -> bool:
        return any(
            float(s.get("specificity_margin", -1.0)) >= 0.5
            for s in specificity_rows
            if int(s["dim"]) == d and s.get("word_group") == "primary_word_seed6" and s.get("V_mode") == "full_V"
        )

    flags = {
        "finite_self_adjoint_all": finite_sa_all(),
        "primary_transfer_pass_any_target_blind": primary_transfer_blind,
        "specificity_margin_positive_dim64": margin_positive_primary_full(64),
        "specificity_margin_positive_dim128": margin_positive_primary_full(128),
        "specificity_margin_positive_dim256": margin_positive_primary_full(256),
        "gue_poisson_not_too_competitive": bool(gue_ok and poi_ok),
        "target_blind_survives": bool(target_blind_survives),
        "overall_specificity_pass": bool(overall_specificity_pass),
    }

    group_map: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for r in summary_rows:
        key = (r["dim"], r["V_mode"], r["word_group"], r["target_group"])
        group_map.setdefault(key, []).append(r)

    group_summary: List[Dict[str, Any]] = []
    for key, rs in sorted(group_map.items()):
        d, vm, wg, tg = key
        acc = float(sum(1 for x in rs if x.get("accepted_transfer"))) / max(1, len(rs))
        jts = [float(x["J_test"]) for x in rs if math.isfinite(float(x["J_test"]))]
        group_summary.append(
            {
                "dim": d,
                "V_mode": vm,
                "word_group": wg,
                "target_group": tg,
                "n": len(rs),
                "accepted_transfer_rate": acc,
                "mean_J_test": float(np.mean(jts)) if jts else float(v2.LARGE_PEN),
                "min_J_test": float(np.min(jts)) if jts else float(v2.LARGE_PEN),
            }
        )

    payload = {
        "warning": "Computational evidence only; not a proof of RH.",
        "status": "V13O.2 specificity controls",
        "conclusion": conclusion,
        "overall_specificity_pass": bool(overall_specificity_pass),
        "flags": flags,
        "dims": dims,
        "v_modes": v_modes,
        "targets": targets,
        "inputs": {
            "candidate_loaded": cand is not None,
            "formula_loaded": form is not None,
            "v13n_loaded": v13n is not None,
            "v13o1_loaded": v13o1 is not None,
        },
        "n_transfer_rows": len(transfer_rows),
        "n_specificity_rows": len(specificity_rows),
        "n_summary_rows": len(summary_rows),
    }

    def write_csv(path: Path, fields: List[str], rows: List[Dict[str, Any]]) -> None:
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for row in rows:
                w.writerow({k: row.get(k) for k in fields})

    if summary_rows:
        write_csv(out_dir / "v13o2_summary.csv", list(summary_rows[0].keys()), summary_rows)
    if transfer_rows:
        tf = [
            "dim",
            "V_mode",
            "word_group",
            "target_group",
            "train_k",
            "test_k",
            "train_spectral_log_mse",
            "test_spectral_log_mse",
            "gap_spectral",
            "train_spacing_mse",
            "test_spacing_mse",
            "gap_spacing",
            "train_ks",
            "test_ks",
            "gap_ks",
            "operator_diff_final",
            "accepted_train",
            "accepted_test",
            "accepted_transfer",
            "J_train",
            "J_test",
            "window_note",
        ]
        write_csv(out_dir / "v13o2_transfer_summary.csv", tf, transfer_rows)
    if group_summary:
        write_csv(out_dir / "v13o2_group_summary.csv", list(group_summary[0].keys()), group_summary)
    if specificity_rows:
        write_csv(out_dir / "v13o2_specificity_scores.csv", list(specificity_rows[0].keys()), specificity_rows)
    if longrange_rows:
        write_csv(
            out_dir / "v13o2_long_range_metrics.csv",
            list(longrange_rows[0].keys()),
            longrange_rows,
        )

    with open(out_dir / "v13o2_results.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=True)

    md = [
        "# V13O.2 Specificity controls\n\n",
        "> Computational evidence for a finite-dimensional candidate only; not a proof of RH.\n\n",
        "## 1. Executive conclusion\n\n",
        f"**{conclusion}** (overall_specificity_pass={overall_specificity_pass}).\n\n",
        "## 2. Main transfer table (primary word, full_V, real zeta)\n\n",
        "| dim | best V_mode | test spacing | test KS | test J | specificity margin | pass |\n|---|---:|---:|---:|---:|---:|---|\n",
    ]
    prim_full = [
        r
        for r in transfer_rows
        if r.get("word_group") == "primary_word_seed6" and r.get("V_mode") == "full_V" and r.get("target_group") == "real_zeta"
    ]
    for r in sorted(prim_full, key=lambda x: int(x["dim"])):
        sp = r.get("test_spacing_mse", float("nan"))
        ks = r.get("test_ks", float("nan"))
        jt = r.get("J_test", float("nan"))
        marg = next(
            (
                float(s.get("specificity_margin", float("nan")))
                for s in specificity_rows
                if int(s["dim"]) == int(r["dim"]) and s["V_mode"] == "full_V" and s["word_group"] == "primary_word_seed6"
            ),
            float("nan"),
        )
        spass = next(
            (
                bool(s.get("specificity_pass"))
                for s in specificity_rows
                if int(s["dim"]) == int(r["dim"]) and s["V_mode"] == "full_V" and s["word_group"] == "primary_word_seed6"
            ),
            False,
        )
        md.append(
            f"| {r['dim']} | full_V | {sp} | {ks} | {jt} | {marg} | {spass} |\n"
        )
    md.append("\n## 3. Leakage table (V modes, primary, real)\n\n")
    md.append("| dim | V_mode | train J | test J | gap | accepted_transfer |\n|---|---:|---:|---:|---:|---|\n")
    for r in sorted(
        [x for x in transfer_rows if x.get("word_group") == "primary_word_seed6" and x.get("target_group") == "real_zeta"],
        key=lambda x: (int(x["dim"]), str(x["V_mode"])),
    ):
        md.append(
            f"| {r['dim']} | {r['V_mode']} | {r.get('J_train')} | {r.get('J_test')} | {float(r.get('J_test',0))-float(r.get('J_train',0))} | {r.get('accepted_transfer')} |\n"
        )
    md.append("\n## 4. Controls (real vs synthetic targets)\n\nSee `v13o2_transfer_summary.csv`.\n\n")
    md.append("## 5. Long-range statistics\n\nSee `v13o2_long_range_metrics.csv`.\n\n")
    md.append("## 6. Interpretation\n\n")
    if conclusion == "specificity_pass":
        md.append("Evidence supports proceeding toward V13P-style analytic renormalization as a next formalization step.\n\n")
    elif conclusion == "partial_specificity":
        md.append("Mixed evidence: treat V13P as conditional; consider V13O.3 leakage reduction.\n\n")
    else:
        md.append("The current construction behaves like a broad universal fitter; refactor V(H) to reduce target leakage before V13P/V13Q/V13R.\n\n")
    md.append("## 7. Next step\n\n")
    md.append(
        "If specificity passes: V13P analytic renormalization. If partial: V13O.3 + conditional V13P. If failed: refactor potential before later milestones.\n"
    )
    (out_dir / "v13o2_report.md").write_text("".join(md), encoding="utf-8")

    tex = (
        "\\documentclass[11pt]{article}\n\\usepackage[margin=1in]{geometry}\n"
        "\\begin{document}\n\\title{V13O.2 Specificity}\n\\maketitle\n"
        "\\section{Conclusion}\n" + latex_escape(conclusion) + "\n\\end{document}\n"
    )
    (out_dir / "v13o2_report.tex").write_text(tex, encoding="utf-8")

    if _find_pdflatex() is None:
        print("[v13o2] WARNING: pdflatex not found; skipping PDF.", flush=True)
    elif try_pdflatex(out_dir / "v13o2_report.tex", out_dir, "v13o2_report.pdf"):
        pass
    else:
        print("[v13o2] WARNING: pdflatex failed.", flush=True)

    odisp = str(out_dir.resolve())
    print(f"Wrote {odisp}/v13o2_results.json", flush=True)
    print(f"Wrote {odisp}/v13o2_summary.csv", flush=True)
    print(f"Wrote {odisp}/v13o2_group_summary.csv", flush=True)
    print(f"Wrote {odisp}/v13o2_transfer_summary.csv", flush=True)
    print(f"Wrote {odisp}/v13o2_specificity_scores.csv", flush=True)
    print(f"Wrote {odisp}/v13o2_report.md / .tex / .pdf", flush=True)
    print(f"Conclusion: {conclusion}", flush=True)
    print(f"Overall specificity pass: {overall_specificity_pass}", flush=True)


if __name__ == "__main__":
    main()
