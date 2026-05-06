#!/usr/bin/env python3
"""
V13O.3: strict specificity gate — clean train/test OOS validation with explicit pass/fail.
Computational evidence only; not a proof of RH.
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
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

PRIMARY_CANDIDATE_ID = "seed_6"

_DTF = np.float64
ALPHA = 0.5
EPS_BUILDER = 0.6
GEO_WEIGHT = 10.0
CLIP_DEFAULT = (0.5, 99.5)
DIAG_SHIFT = 1e-6
ABS_CAP = 5.0

# Hardcoded primary braid (validated against candidate_operators.json seed_6)
PRIMARY_WORD_FALLBACK = [-4, -2, -4, -2, -2, -1, -1]
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

# Shared zeta ordinates: main sets _Z_SORTED_MAIN; worker processes set _Z_SORTED_WORKER via pool initializer.
_Z_SORTED_MAIN: Optional[np.ndarray] = None
_Z_SORTED_WORKER: Optional[np.ndarray] = None
_WORKER_V13_VALIDATE_MOD: Any = None

V_MODES_ALL = [
    "full_V",
    "frozen_V_after_5",
    "frozen_V_after_10",
    "weak_V",
    "very_weak_V",
    "target_blind_V",
    "density_only_V",
    "phase_only_V",
    "word_only_V",
]


def pareto_j_v13o3(sl: float, sp: float, ks: float) -> float:
    """V13O.3 Pareto-on-errors (train/test specificity), no operator step term."""
    if not math.isfinite(float(sl)):
        sl = 1e6
    if not math.isfinite(float(sp)):
        sp = 1e6
    if not math.isfinite(float(ks)):
        ks = 1.0
    return float(sl + 0.15 * float(sp) + 0.50 * float(ks))


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


def _require_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        print(f"[v13o3] ERROR: required input missing: {path}", flush=True)
        sys.exit(1)
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


def json_sanitize(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, float):
        if math.isnan(obj) or not math.isfinite(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if math.isnan(v) or not math.isfinite(v):
            return None
        return v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return json_sanitize(obj.tolist())
    if isinstance(obj, dict):
        return {str(k): json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_sanitize(x) for x in obj]
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, str):
        return obj
    if isinstance(obj, (int,)):
        return int(obj)
    try:
        if isinstance(obj, Path):
            return str(obj)
    except Exception:
        pass
    return obj


def csv_cell(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, float) and (math.isnan(v) or not math.isfinite(v)):
        return ""
    if isinstance(v, (np.floating,)):
        x = float(v)
        if math.isnan(x) or not math.isfinite(x):
            return ""
        return x
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


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


def full_shuffle_zeta(z_sorted: np.ndarray, k_train: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Shuffle gap order uniformly (V13O.3 shuffled_zeta control)."""
    need = 2 * int(k_train)
    z = np.sort(np.asarray(z_sorted[:need], dtype=np.float64).reshape(-1))
    gaps = np.maximum(np.diff(z), 1e-9)
    rng = np.random.default_rng(int(seed))
    order = np.arange(gaps.size)
    rng.shuffle(order)
    g2 = gaps[order]
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
    target_rep: int,
) -> Tuple[np.ndarray, np.ndarray, bool, str]:
    need = 2 * int(k_train)
    base = np.sort(np.asarray(z_sorted[:need], dtype=_DTF).reshape(-1))
    sp = int(target_rep)
    if name == "real_zeta_test":
        return base[:k_train].copy(), base[k_train:need].copy(), False, "sorted_zeta"
    if name == "reversed_zeta":
        rev = base[::-1].copy()
        return rev[:k_train].copy(), rev[k_train:need].copy(), True, "reversed_prefix"
    if name == "block_shuffled_zeta_block4":
        tr, te = block_shuffle_zeta(z_sorted, k_train, 4, seed + 11 * sp)
        return tr, te, False, "block_shuffle_4"
    if name == "block_shuffled_zeta_block8":
        tr, te = block_shuffle_zeta(z_sorted, k_train, 8, seed + 13 * sp + 1)
        return tr, te, False, "block_shuffle_8"
    if name == "local_jitter_zeta_small":
        tr, te = local_jitter_zeta(z_sorted, k_train, seed + 17 * sp + 2, 0.05)
        return tr, te, False, "jitter_small"
    if name == "local_jitter_zeta_medium":
        tr, te = local_jitter_zeta(z_sorted, k_train, seed + 19 * sp + 3, 0.15)
        return tr, te, False, "jitter_medium"
    if name == "density_matched_synthetic":
        tr, te = density_matched_synthetic(z_sorted, k_train, seed + 23 * sp + 4)
        return tr, te, False, "density_matched"
    if name == "GUE_synthetic":
        g = gue_ord(need, np.random.default_rng(seed + 29 * sp + 5))
        return g[:k_train].copy(), g[k_train:need].copy(), False, "GUE"
    if name == "Poisson_synthetic":
        p = poisson_ord(need, np.random.default_rng(seed + 31 * sp + 6))
        return p[:k_train].copy(), p[k_train:need].copy(), False, "Poisson"
    if name == "shuffled_zeta":
        tr, te = full_shuffle_zeta(z_sorted, k_train, seed + 37 * sp + 7)
        return tr, te, False, "gap_shuffle_full"
    raise ValueError(name)


def target_has_replicas(name: str) -> bool:
    return name != "real_zeta_test" and name != "reversed_zeta"


def meets_accepted_v13o3(
    *,
    finite: bool,
    sa: float,
    od: float,
    test_sp: float,
    test_ks: float,
    eig_err: Optional[str],
) -> bool:
    if eig_err:
        return False
    return bool(
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


def dedupe_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Set[Tuple[Any, ...]] = set()
    out: List[Dict[str, Any]] = []
    for r in rows:
        key = (r["dim"], r["V_mode"], r["word_group"], r["target_group"], int(r["control_id"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def load_v13_validate() -> Any:
    import importlib.util

    path = Path(ROOT) / "scripts" / "validate_candidates_v13.py"
    spec = importlib.util.spec_from_file_location("_v13_validate_o3", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_v13_validate_o3"] = mod
    spec.loader.exec_module(mod)
    return mod


def count_target_cells(n_controls: int) -> List[Tuple[str, int]]:
    out: List[Tuple[str, int]] = []
    names = [
        "real_zeta_test",
        "reversed_zeta",
        "block_shuffled_zeta_block4",
        "block_shuffled_zeta_block8",
        "local_jitter_zeta_small",
        "local_jitter_zeta_medium",
        "density_matched_synthetic",
        "GUE_synthetic",
        "Poisson_synthetic",
        "shuffled_zeta",
    ]
    for nm in names:
        if target_has_replicas(nm):
            for c in range(int(n_controls)):
                out.append((nm, c))
        else:
            out.append((nm, 0))
    return out


def transfer_row_nan_shell(job: Dict[str, Any]) -> Dict[str, Any]:
    """Minimal transfer-record shape for failures (CSV / dedupe compat)."""
    nan = float("nan")
    return {
        "dim": job.get("dim"),
        "V_mode": job.get("V_mode"),
        "word_group": job.get("word_group"),
        "target_group": job.get("target_group"),
        "control_id": job.get("control_id"),
        "target_rep": job.get("target_rep", 0),
        "word_job_index": job.get("wji"),
        "window_note": "",
        "train_k": int(DIM_K_TRAIN.get(int(job.get("dim") or 0), 0)),
        "test_k": int(DIM_K_TRAIN.get(int(job.get("dim") or 0), 0)),
        "spectral_log_mse_train": nan,
        "spacing_mse_normalized_train": nan,
        "ks_wigner_train": nan,
        "spectral_log_mse_test": nan,
        "spacing_mse_normalized_test": nan,
        "ks_wigner_test": nan,
        "spectral_log_mse": nan,
        "spacing_mse_normalized": nan,
        "ks_wigner": nan,
        "pareto_objective_train": nan,
        "pareto_objective_test": nan,
        "pareto_objective": nan,
        "operator_diff_final": nan,
        "converged_operator": False,
        "finite": False,
        "self_adjointness_fro": nan,
        "accepted": False,
        "train_to_test_gap_spectral": nan,
        "eig_error": "",
    }


def stable_control_id(word_group_id: str, target_group: str, target_rep: int, wj: Dict[str, Any]) -> int:
    stable_cid = int(target_rep if target_has_replicas(target_group) else 0)
    if word_group_id == "random_words_n30":
        stable_cid = int(wj.get("rw_index", 0)) * 1000 + stable_cid
    elif word_group_id == "random_symmetric_baseline":
        stable_cid = int(wj.get("sym_index", 0)) * 1000 + stable_cid
    return stable_cid


def target_seed_legacy(
    base_seed: int, dim: int, vm: str, tg: str, wji: int, target_rep: int
) -> int:
    """Match legacy sequential seed convention (computed once in parent for pickling determinism across workers)."""
    rep_for_seed = int(target_rep if target_has_replicas(tg) else 0)
    return (
        int(base_seed)
        + dim * 100003
        + (hash(vm) % 9973)
        + (hash(tg) % 9973)
        + wji * 17
        + rep_for_seed * 131
    )


def _serializable_job_config(wj: Dict[str, Any]) -> Dict[str, Any]:
    d = dict(wj)
    d.pop("matrix", None)
    return {k: (list(v) if isinstance(v, list) and k == "word" else v) for k, v in d.items()}


def _v13o3_worker_init(z_sorted: np.ndarray) -> None:
    global _Z_SORTED_WORKER
    _Z_SORTED_WORKER = np.asarray(z_sorted, dtype=_DTF).reshape(-1)


def _z_sorted_for_eval() -> np.ndarray:
    if _Z_SORTED_WORKER is not None:
        return _Z_SORTED_WORKER
    if _Z_SORTED_MAIN is not None:
        return _Z_SORTED_MAIN
    raise RuntimeError("z_sorted not configured (set _Z_SORTED_MAIN or worker init)")


def _get_v13_validate_cached() -> Any:
    global _WORKER_V13_VALIDATE_MOD
    if _WORKER_V13_VALIDATE_MOD is None:
        _WORKER_V13_VALIDATE_MOD = load_v13_validate()
    return _WORKER_V13_VALIDATE_MOD


def build_word_jobs_for_dim(
    *,
    word_jobs_base: List[Dict[str, Any]],
    n_ctrl: int,
) -> List[Dict[str, Any]]:
    """Word jobs for one dim: base list + sym baselines (matrices built in workers from seeds)."""
    word_jobs: List[Dict[str, Any]] = list(word_jobs_base)
    for j in range(n_ctrl):
        word_jobs.append(
            {
                "id": "random_symmetric_baseline",
                "word": [],
                "kind": "sym",
                "sym_index": j,
            }
        )
    return word_jobs


def build_job_list(
    *,
    dims: List[int],
    v_modes: List[str],
    target_cells: List[Tuple[str, int]],
    word_jobs_base: List[Dict[str, Any]],
    n_ctrl: int,
    base_seed: int,
    fro_ref_by_dim: Dict[int, float],
    max_iter: int,
    tol: float,
) -> List[Dict[str, Any]]:
    jobs: List[Dict[str, Any]] = []
    job_id = 0
    for dim in dims:
        fro_ref = float(fro_ref_by_dim[dim])
        word_jobs = build_word_jobs_for_dim(word_jobs_base=word_jobs_base, n_ctrl=n_ctrl)
        cfg = dict(DIM_PARAM[dim])
        for wji, wj in enumerate(word_jobs):
            wid = str(wj["id"])
            for vm in v_modes:
                smooth_use = float(cfg["smooth_sigma"])
                clip_lo, clip_hi = float(CLIP_DEFAULT[0]), float(CLIP_DEFAULT[1])
                for tg, target_rep in target_cells:
                    seed_t = target_seed_legacy(base_seed, dim, vm, tg, wji, target_rep)
                    scid = stable_control_id(wid, tg, target_rep, wj)
                    jobs.append(
                        {
                            "job_id": job_id,
                            "dim": int(dim),
                            "V_mode": vm,
                            "word_group": wid,
                            "target_group": tg,
                            "target_rep": int(target_rep),
                            "control_id": int(scid),
                            "wji": int(wji),
                            "wj": _serializable_job_config(wj),
                            "seed": int(base_seed),
                            "seed_t": int(seed_t),
                            "max_iter": int(max_iter),
                            "tol": float(tol),
                            "fro_ref": fro_ref,
                            "dim_config": cfg,
                            "smooth_sigma": smooth_use,
                            "clip_lo": clip_lo,
                            "clip_hi": clip_hi,
                            "config": {
                                "DIM_PARAM_key": dim,
                                "ALPHA": ALPHA,
                                "EPS_BUILDER": EPS_BUILDER,
                                "GEO_WEIGHT": GEO_WEIGHT,
                                "CLIP_DEFAULT": list(CLIP_DEFAULT),
                                "DIAG_SHIFT": DIAG_SHIFT,
                                "ABS_CAP": ABS_CAP,
                            },
                        }
                    )
                    job_id += 1
    return jobs


def evaluate_one_job(job: Dict[str, Any]) -> Dict[str, Any]:
    """Top-level picklable job runner (must stay at module scope for multiprocessing)."""
    job_id = job.get("job_id")
    t0 = time.time()
    try:
        worker_seed = int(job["seed"]) + 1000003 * int(job_id)
        float(np.random.default_rng(worker_seed).random())  # Per-job RNG (targets use legacy seed_t)

        v = _get_v13_validate_cached()
        from core import v13l_self_consistent as v13l
        from core import v13o2_specificity as v2
        from core.artin_operator import sample_domain

        dim = int(job["dim"])
        vm = str(job["V_mode"])
        tg = str(job["target_group"])
        wid = str(job["word_group"])
        target_rep = int(job["target_rep"])
        wji = int(job["wji"])
        wj = dict(job["wj"])
        k_train = int(DIM_K_TRAIN[dim])
        z_sorted = _z_sorted_for_eval()
        if z_sorted.size < 2 * k_train:
            raise RuntimeError(f"Need {2 * k_train} zeros; got {z_sorted.size}.")

        cfg = dict(job["dim_config"])
        base_seed = int(job["seed"])
        seed_t = int(job["seed_t"])
        Z = sample_domain(dim, seed=base_seed)
        fro_ref = float(job["fro_ref"])

        kind = str(wj.get("kind", "word"))
        if kind == "sym":
            sym_index = int(wj["sym_index"])
            r = np.random.default_rng(base_seed + dim * 104729 + sym_index).standard_normal((dim, dim))
            S = 0.5 * (r + r.T)
            fs = float(np.linalg.norm(S, ord="fro"))
            H_op = (S * (fro_ref / max(fs, 1e-12))).astype(_DTF, copy=False)
        elif kind == "ablate_V":
            H_op = build_h(
                z_points=Z,
                word=list(wj.get("word") or []),
                v13l=v13l,
                geo_sigma=float(cfg["geo_sigma"]),
                laplacian_weight=float(wj["lap"]),
                geo_weight=float(wj["geo_w"]),
            )
        else:
            H_op = build_h(
                z_points=Z,
                word=list(wj.get("word") or []),
                v13l=v13l,
                geo_sigma=float(cfg["geo_sigma"]),
                laplacian_weight=float(wj["lap"]),
                geo_weight=float(wj["geo_w"]),
            )

        smooth_use = float(job["smooth_sigma"])
        clip_lo, clip_hi = float(job["clip_lo"]), float(job["clip_hi"])
        train_ordered: bool
        tr, te, train_ordered, wnote = build_train_test_targets(tg, z_sorted, k_train, seed_t, target_rep)

        z_train_m = np.sort(np.asarray(tr, dtype=_DTF).reshape(-1))
        z_test_m = np.sort(np.asarray(te, dtype=_DTF).reshape(-1))
        k_al = int(min(dim, k_train, z_train_m.size, z_test_m.size))

        eig_err: Optional[str] = None
        try:
            if kind == "ablate_V":
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
                    max_iter=int(job["max_iter"]),
                    tol=float(job["tol"]),
                    v_mode=vm,
                    z_points=np.asarray(Z, dtype=np.complex128),
                    word=list(wj.get("word") or []),
                    on_train_iter=None,
                )
                ee = train_out.get("meta", {}).get("eig_error")
                eig_err = ee if isinstance(ee, str) else None
        except Exception as ex:
            eig_err = str(ex)
            train_out = {
                "H_final": np.asarray(H_op, dtype=_DTF, copy=True),
                "H_base": np.asarray(H_op, dtype=_DTF, copy=True),
                "V_diag_last": np.zeros((dim,), _DTF),
                "meta": {"converged_operator": False, "eig_error": eig_err},
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
        meta_c = train_out.get("meta") or {}
        converged_operator = bool(meta_c.get("converged_operator", False))

        lr_tr = v2.long_range_bundle(
            eig,
            z_train_m,
            k_al,
            spacing_fn=v.spacing_mse_normalized,
            ks_fn=v.ks_against_wigner_gue,
            norm_gaps_fn=v.normalized_gaps,
        )
        lr_te = v2.long_range_bundle(
            eig,
            z_test_m,
            k_al,
            spacing_fn=v.spacing_mse_normalized,
            ks_fn=v.ks_against_wigner_gue,
            norm_gaps_fn=v.normalized_gaps,
        )

        J_tr = pareto_j_v13o3(lr_tr["spectral_log_mse"], lr_tr["spacing_mse_normalized"], lr_tr["ks_wigner"])
        J_te = pareto_j_v13o3(lr_te["spectral_log_mse"], lr_te["spacing_mse_normalized"], lr_te["ks_wigner"])

        acc = meets_accepted_v13o3(
            finite=finite,
            sa=sa,
            od=od,
            test_sp=lr_te["spacing_mse_normalized"],
            test_ks=lr_te["ks_wigner"],
            eig_err=eig_err,
        )

        gap_sl = float(lr_te["spectral_log_mse"] - lr_tr["spectral_log_mse"])
        stable_cid = int(job["control_id"])

        rec: Dict[str, Any] = {
            "dim": dim,
            "V_mode": vm,
            "word_group": wid,
            "target_group": tg,
            "control_id": stable_cid,
            "target_rep": int(target_rep),
            "word_job_index": wji,
            "window_note": wnote,
            "train_k": k_train,
            "test_k": k_train,
            "spectral_log_mse_train": lr_tr["spectral_log_mse"],
            "spacing_mse_normalized_train": lr_tr["spacing_mse_normalized"],
            "ks_wigner_train": lr_tr["ks_wigner"],
            "spectral_log_mse_test": lr_te["spectral_log_mse"],
            "spacing_mse_normalized_test": lr_te["spacing_mse_normalized"],
            "ks_wigner_test": lr_te["ks_wigner"],
            "spectral_log_mse": lr_te["spectral_log_mse"],
            "spacing_mse_normalized": lr_te["spacing_mse_normalized"],
            "ks_wigner": lr_te["ks_wigner"],
            "pareto_objective_train": J_tr,
            "pareto_objective_test": J_te,
            "pareto_objective": J_te,
            "operator_diff_final": od,
            "converged_operator": converged_operator,
            "finite": finite,
            "self_adjointness_fro": sa,
            "accepted": acc,
            "train_to_test_gap_spectral": gap_sl,
            "eig_error": eig_err or "",
        }

        out = {
            "job_id": int(job_id),
            "dim": dim,
            "V_mode": vm,
            "word_group": wid,
            "target_group": tg,
            "control_id": stable_cid,
            "runtime_s": time.time() - t0,
            "worker_pid": os.getpid(),
            "error": None,
            "traceback": None,
            "rec": rec,
            "H_fin": H_fin.copy() if acc else None,
        }
        return out
    except Exception as e:
        err_rec = transfer_row_nan_shell(job)
        err_rec["eig_error"] = repr(e)
        jid = job.get("job_id")
        return {
            "job_id": int(jid) if jid is not None else -1,
            "dim": job.get("dim"),
            "V_mode": job.get("V_mode"),
            "word_group": job.get("word_group"),
            "target_group": job.get("target_group"),
            "control_id": job.get("control_id"),
            "error": repr(e),
            "traceback": traceback.format_exc(),
            "runtime_s": time.time() - t0,
            "worker_pid": os.getpid(),
            "rec": err_rec,
            "H_fin": None,
        }


def _jsonl_append(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(json_sanitize(obj), allow_nan=False) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)


def read_jobs_jsonl(path: Path) -> Dict[int, Dict[str, Any]]:
    """Latest line wins per job_id."""
    by_id: Dict[int, Dict[str, Any]] = {}
    if not path.is_file():
        return by_id
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                o = json.loads(ln)
            except json.JSONDecodeError:
                continue
            jid = o.get("job_id")
            if jid is None:
                continue
            by_id[int(jid)] = o
    return by_id


def jsonl_row_to_result_record(o: Dict[str, Any]) -> Dict[str, Any]:
    META = frozenset({"job_id", "runtime_s", "worker_pid", "error", "traceback"})
    rec = {k: v for k, v in o.items() if k not in META}
    return {
        "job_id": int(o["job_id"]),
        "dim": None,
        "V_mode": None,
        "word_group": None,
        "target_group": None,
        "control_id": None,
        "rec": rec,
        "error": o.get("error"),
        "traceback": o.get("traceback"),
        "runtime_s": o.get("runtime_s"),
        "worker_pid": o.get("worker_pid"),
        "H_fin": None,
    }


def _sort_result_records(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def key(r: Dict[str, Any]) -> Tuple:
        rec = r.get("rec") or {}
        cid = rec.get("control_id")
        try:
            ckey = (0, int(cid))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            ckey = (1, str(cid))
        return (
            int(rec["dim"]) if rec.get("dim") is not None else -1,
            str(rec.get("V_mode") or ""),
            str(rec.get("word_group") or ""),
            str(rec.get("target_group") or ""),
            str(ckey),
            int(r["job_id"]) if r.get("job_id") is not None else -1,
        )

    return sorted(results, key=key)


def _record_for_jsonl(r: Dict[str, Any]) -> Dict[str, Any]:
    """JSONL-safe record without large arrays."""
    if r.get("rec") is not None:
        base = dict(r["rec"])
    else:
        base = {
            "dim": r.get("dim"),
            "V_mode": r.get("V_mode"),
            "word_group": r.get("word_group"),
            "target_group": r.get("target_group"),
            "control_id": r.get("control_id"),
            "finite": False,
            "accepted": False,
            "operator_diff_final": float("nan"),
            "pareto_objective_test": float("nan"),
            "eig_error": str(r.get("error") or ""),
        }
    return {
        **base,
        "job_id": r.get("job_id"),
        "runtime_s": r.get("runtime_s"),
        "worker_pid": r.get("worker_pid"),
        "error": r.get("error"),
        "traceback": r.get("traceback"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="V13O.3 strict specificity validation (computational evidence only).")
    ap.add_argument("--candidate_json", type=str, default="runs/v13_candidate_theorem/candidate_operators.json")
    ap.add_argument("--formula_json", type=str, default="runs/v13_operator_formula/formula_components_summary.json")
    ap.add_argument("--v13n_summary", type=str, default="runs/v13n_theorem_report/v13n_summary.json")
    ap.add_argument("--out_dir", type=str, default="runs/v13o3_strict_specificity")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dims", type=int, nargs="+", default=[64, 128, 256])
    ap.add_argument("--max_iter", type=int, default=300)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--n_controls", type=int, default=30)
    ap.add_argument("--n_random_words", type=int, default=30)
    ap.add_argument("--progress_every", type=int, default=1)
    ap.add_argument("--parallel", action="store_true", help="Run independent cells in parallel with ProcessPoolExecutor.")
    nc_def = os.cpu_count() or 1
    ap.add_argument(
        "--num_workers",
        type=int,
        default=max(1, nc_def - 2),
        help=f"Parallel worker processes (default max(1, cpu_count()-2) ⇒ {max(1, nc_def - 2)} on this machine).",
    )
    ap.add_argument(
        "--chunksize",
        type=int,
        default=1,
        help="Submission batch sizing: submits at most num_workers * max(1, chunksize) jobs ahead (default 1).",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing raw JSONL: skip finished job_ids and append new completions.",
    )
    ap.add_argument(
        "--raw_jsonl",
        type=str,
        default="v13o3_raw_jobs.jsonl",
        help="Relative paths resolve under --out_dir; written as one JSON object per finished job.",
    )
    args = ap.parse_args()

    out_dir = _resolve(args.out_dir)
    ck_dir = out_dir / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    ck_dir.mkdir(parents=True, exist_ok=True)

    cand_path = _resolve(args.candidate_json)
    form_path = _resolve(args.formula_json)
    v13n_path = _resolve(args.v13n_summary)
    cand = _require_json(cand_path)
    form = _require_json(form_path)
    v13n = _require_json(v13n_path)

    spect = cand.get("spectral_candidate") or {}
    if spect.get("id") != PRIMARY_CANDIDATE_ID:
        print(
            f"[v13o3] WARNING: expected spectral_candidate.id={PRIMARY_CANDIDATE_ID}, got {spect.get('id')}",
            flush=True,
        )
    primary_word = [int(x) for x in (spect.get("word") or PRIMARY_WORD_FALLBACK)]
    if primary_word != PRIMARY_WORD_FALLBACK:
        print(f"[v13o3] Using primary word from candidate_json (id={spect.get('id')}).", flush=True)

    rej = None
    for rc in cand.get("rejected_candidates") or []:
        if rc.get("id") == "seed_17":
            rej = [int(x) for x in rc.get("word") or REJECTED_WORD]
            break
    rejected_word = rej or REJECTED_WORD

    v = load_v13_validate()
    from core import v13l_self_consistent as v13l
    from core.artin_operator import sample_domain

    dims = [int(d) for d in args.dims if int(d) in DIM_K_TRAIN]
    if not dims:
        raise SystemExit("No valid dims in --dims (expected subset of 64,128,256).")

    n_ctrl = max(1, int(args.n_controls))
    n_rw = max(1, int(args.n_random_words))
    base_seed = int(args.seed)
    prog = max(1, int(args.progress_every))

    max_k = max(DIM_K_TRAIN[d] for d in dims)
    z_pool = v._load_zeros(max(512, 2 * max_k))
    z_sorted = np.sort(np.asarray(z_pool, dtype=_DTF).reshape(-1))
    z_sorted = z_sorted[np.isfinite(z_sorted) & (z_sorted > 0.0)]

    word_jobs_base: List[Dict[str, Any]] = [
        {"id": "primary_word_seed6", "word": list(primary_word), "lap": 1.0, "geo_w": GEO_WEIGHT, "kind": "word"},
        {"id": "rejected_word_seed17", "word": list(rejected_word), "lap": 1.0, "geo_w": GEO_WEIGHT, "kind": "word"},
        {"id": "ablate_K", "word": list(primary_word), "lap": 1.0, "geo_w": 0.0, "kind": "word"},
        {"id": "ablate_V", "word": list(primary_word), "lap": 1.0, "geo_w": GEO_WEIGHT, "kind": "ablate_V"},
        {"id": "ablate_L", "word": list(primary_word), "lap": 0.0, "geo_w": GEO_WEIGHT, "kind": "word"},
    ]
    for j in range(n_rw):
        alphabet = list(range(-6, 0)) + list(range(1, 7))
        rw = [int(np.random.default_rng(base_seed + 7919 + j).choice(alphabet)) for _ in range(len(primary_word))]
        word_jobs_base.append(
            {"id": "random_words_n30", "word": rw, "lap": 1.0, "geo_w": GEO_WEIGHT, "kind": "word", "rw_index": j}
        )

    target_cells = count_target_cells(n_ctrl)
    v_modes = list(V_MODES_ALL)

    raw_jsonl_arg = Path(args.raw_jsonl)
    raw_jobs_path = raw_jsonl_arg if raw_jsonl_arg.is_absolute() else out_dir / raw_jsonl_arg

    fro_ref_cache: Dict[int, float] = {}
    for dim in dims:
        Zpre = sample_domain(dim, seed=base_seed)
        cfg_pre = DIM_PARAM[dim]
        fro_ref_cache[dim] = float(
            np.linalg.norm(
                build_h(
                    z_points=Zpre,
                    word=list(primary_word),
                    v13l=v13l,
                    geo_sigma=float(cfg_pre["geo_sigma"]),
                    laplacian_weight=1.0,
                    geo_weight=float(GEO_WEIGHT),
                ),
                ord="fro",
            )
        )

    for dim in dims:
        k_need = DIM_K_TRAIN[dim]
        if z_sorted.size < 2 * k_need:
            raise RuntimeError(f"Need {2 * k_need} zeros; got {z_sorted.size}.")

    jobs: List[Dict[str, Any]] = build_job_list(
        dims=dims,
        v_modes=v_modes,
        target_cells=target_cells,
        word_jobs_base=word_jobs_base,
        n_ctrl=n_ctrl,
        base_seed=base_seed,
        fro_ref_by_dim=fro_ref_cache,
        max_iter=int(args.max_iter),
        tol=float(args.tol),
    )
    total_jobs = len(jobs)
    n_word_jobs = len(word_jobs_base) + n_ctrl
    print(
        f"[v13o3] workload grid: {len(dims)} dims × {n_word_jobs} word_jobs × {len(v_modes)} V_modes "
        f"× {len(target_cells)} target_cells → {total_jobs} independent jobs",
        flush=True,
    )

    done_ids_pre: Set[int] = set()
    if args.resume:
        done_ids_pre = set(read_jobs_jsonl(raw_jobs_path).keys())

    if not args.resume and raw_jobs_path.exists():
        raw_jobs_path.unlink()

    jobs_to_run = [j for j in jobs if int(j["job_id"]) not in done_ids_pre]

    global _Z_SORTED_MAIN
    _Z_SORTED_MAIN = z_sorted

    num_workers = max(1, int(args.num_workers))
    chunk_mul = max(1, int(args.chunksize))
    max_inflight = max(1, num_workers * chunk_mul)

    print(
        f"[v13o3] jobs total={total_jobs} to_run={len(jobs_to_run)} resume={bool(args.resume)} "
        f"parallel={bool(args.parallel)} num_workers={num_workers if args.parallel else 1} "
        f"chunksize={chunk_mul} max_inflight={max_inflight if args.parallel else 1}",
        flush=True,
    )

    exec_t0 = time.perf_counter()
    done_total = len(done_ids_pre)
    checkpoint_writes: List[Tuple[Dict[str, Any], np.ndarray]] = []

    def process_finished_row(row: Dict[str, Any]) -> None:
        nonlocal done_total
        done_total += 1
        _jsonl_append(raw_jobs_path, _record_for_jsonl(row))

        rec = row.get("rec") or {}
        hf = row.get("H_fin")
        if hf is not None and rec.get("accepted"):
            checkpoint_writes.append((dict(rec), np.asarray(hf, dtype=_DTF)))

        dim_r = int(rec["dim"]) if rec.get("dim") is not None else 0
        vm_r = str(rec.get("V_mode") or "")
        wg_r = str(rec.get("word_group") or "")
        tg_r = str(rec.get("target_group") or "")
        acc_r = bool(rec.get("accepted"))
        jte = rec.get("pareto_objective_test")
        jte_s = f"{float(jte):.4g}" if isinstance(jte, (int, float)) and math.isfinite(float(jte)) else str(jte)

        if done_total == 1 or done_total == total_jobs or done_total % prog == 0:
            elapsed = time.perf_counter() - exec_t0
            session_finished = max(0, done_total - len(done_ids_pre))
            avg = elapsed / max(session_finished, 1)
            eta_sec = avg * max(total_jobs - done_total, 0)
            pct = 100.0 * float(done_total) / max(total_jobs, 1)
            print(
                f"[v13o3] done={done_total}/{total_jobs} {pct:.1f}% elapsed={elapsed:.1f}s "
                f"avg={avg:.2f}s/job ETA={format_seconds(eta_sec)} "
                f"last dim={dim_r} V={vm_r} group={wg_r} target={tg_r} accepted={acc_r} J={jte_s}",
                flush=True,
            )

    if jobs_to_run:
        if args.parallel:
            with ProcessPoolExecutor(
                max_workers=num_workers,
                initializer=_v13o3_worker_init,
                initargs=(z_sorted,),
            ) as ex:
                idx_run = 0
                while idx_run < len(jobs_to_run):
                    wave = jobs_to_run[idx_run : idx_run + max_inflight]
                    idx_run += len(wave)
                    futs = [ex.submit(evaluate_one_job, job) for job in wave]
                    for fut in as_completed(futs):
                        process_finished_row(fut.result())
        else:
            for job in jobs_to_run:
                process_finished_row(evaluate_one_job(job))

    wall_s = time.perf_counter() - exec_t0

    final_by_id = read_jobs_jsonl(raw_jobs_path)
    if len(final_by_id) != total_jobs:
        print(
            f"[v13o3] WARNING: raw JSONL has {len(final_by_id)}/{total_jobs} jobs; "
            "aggregating available rows only.",
            flush=True,
        )

    combined_results = [jsonl_row_to_result_record(o) for o in final_by_id.values()]
    combined_sorted = _sort_result_records(combined_results)
    all_rows = [dict(r["rec"]) for r in combined_sorted if r.get("rec") is not None]

    failed_jobs = sum(1 for r in combined_sorted if r.get("error"))
    rt_vals = [
        float(r["runtime_s"])
        for r in combined_sorted
        if r.get("runtime_s") is not None and math.isfinite(float(r["runtime_s"]))
    ]
    avg_job_rt = float(sum(rt_vals) / len(rt_vals)) if rt_vals else 0.0

    for rec, H_fin in checkpoint_writes:
        dim = int(rec["dim"])
        vm = str(rec["V_mode"])
        tg = str(rec["target_group"])
        wid = str(rec["word_group"])
        scid = rec["control_id"]
        h = hashlib.sha256(f"{dim}|{vm}|{tg}|{wid}|{scid}".encode("utf-8")).hexdigest()[:10]
        fn = ck_dir / f"checkpoint_dim{dim}_{vm}_{tg}_{wid}_{h}.npy"
        try:
            np.save(str(fn), H_fin)
        except OSError:
            pass
    # --- dedupe ---
    dedup_rows = dedupe_rows(all_rows)

    # --- specificity summaries (median controls pooled) ---
    CONTROL_NAMES = [
        "block_shuffled_zeta_block4",
        "block_shuffled_zeta_block8",
        "local_jitter_zeta_small",
        "local_jitter_zeta_medium",
        "density_matched_synthetic",
        "reversed_zeta",
        "GUE_synthetic",
        "Poisson_synthetic",
        "shuffled_zeta",
    ]

    def filt(d: int, vm: str, wg: str, tg: Optional[str] = None) -> List[Dict[str, Any]]:
        rs = [r for r in dedup_rows if int(r["dim"]) == d and r["V_mode"] == vm and r["word_group"] == wg]
        if tg is not None:
            rs = [r for r in rs if r["target_group"] == tg]
        return rs

    specificity_rows: List[Dict[str, Any]] = []
    gate_rows: List[Dict[str, Any]] = []
    STRICT_GATES_ORDER = (
        "G1_lt_median_all_controls_J",
        "G2_lt_median_block4_J",
        "G3_lt_median_block8_J",
        "G4_lt_median_reversed_J",
        "G5_lt_median_density_matched_J",
        "G6_lt_median_GUE_J",
        "G7_lt_median_poisson_J",
        "G8_lt_median_shuffled_zeta_J",
        "G9_accepted_real_zeta_test",
        "G10_train_test_gap_sl_le_2",
    )

    for dim in dims:
        for vm in v_modes:
            for wg in sorted({str(r["word_group"]) for r in dedup_rows if int(r["dim"]) == dim}):
                j_real_ls = [
                    float(r["pareto_objective_test"])
                    for r in filt(dim, vm, wg, "real_zeta_test")
                    if math.isfinite(float(r["pareto_objective_test"]))
                ]
                if not j_real_ls:
                    continue
                j_real = float(np.median(np.asarray(j_real_ls, dtype=np.float64)))

                jc_all_ls: List[float] = []
                for cn in CONTROL_NAMES:
                    jc_all_ls.extend(
                        float(r["pareto_objective_test"])
                        for r in filt(dim, vm, wg, cn)
                        if math.isfinite(float(r["pareto_objective_test"]))
                    )
                med_all = float(np.median(np.asarray(jc_all_ls, dtype=np.float64))) if jc_all_ls else float("nan")

                def med_tg(name: str) -> float:
                    xs = [
                        float(r["pareto_objective_test"])
                        for r in filt(dim, vm, wg, name)
                        if math.isfinite(float(r["pareto_objective_test"]))
                    ]
                    return float(np.median(np.asarray(xs, dtype=np.float64))) if xs else float("nan")

                m_b4 = med_tg("block_shuffled_zeta_block4")
                m_b8 = med_tg("block_shuffled_zeta_block8")
                m_rev = med_tg("reversed_zeta")
                m_den = med_tg("density_matched_synthetic")
                m_gue = med_tg("GUE_synthetic")
                m_poi = med_tg("Poisson_synthetic")
                m_shuf = med_tg("shuffled_zeta")

                margin = med_all - j_real if math.isfinite(med_all) and math.isfinite(j_real) else float("nan")
                score = j_real - med_all if math.isfinite(med_all) and math.isfinite(j_real) else float("nan")

                real_rec = filt(dim, vm, wg, "real_zeta_test")
                acc_real = bool(real_rec) and all(bool(r.get("accepted")) for r in real_rec)
                gap_sl_max = (
                    max(float(r["train_to_test_gap_spectral"]) for r in real_rec)
                    if real_rec
                    else float("nan")
                )

                def lt(a: float, b: float) -> bool:
                    return math.isfinite(a) and math.isfinite(b) and a < b

                g1 = lt(j_real, med_all)
                g2 = lt(j_real, m_b4)
                g3 = lt(j_real, m_b8)
                g4 = lt(j_real, m_rev)
                g5 = lt(j_real, m_den)
                g6 = lt(j_real, m_gue)
                g7 = lt(j_real, m_poi)
                g8 = lt(j_real, m_shuf)
                g9 = acc_real
                g10 = math.isfinite(gap_sl_max) and gap_sl_max <= 2.0

                strict_pass = bool(g1 and g2 and g3 and g4 and g5 and g6 and g7 and g8 and g9 and g10)

                specificity_rows.append(
                    {
                        "dim": dim,
                        "V_mode": vm,
                        "word_group": wg,
                        "J_real_test": j_real,
                        "median_control_J_all": med_all,
                        "specificity_margin": margin,
                        "specificity_score": score,
                        "strict_specificity_pass": strict_pass,
                    }
                )

                if vm == "full_V" and wg == "primary_word_seed6":
                    gate_rows.append(
                        {
                            "dim": dim,
                            "V_mode": vm,
                            "word_group": wg,
                            "j_real": j_real,
                            "median_all_controls": med_all,
                            "median_block4": m_b4,
                            "median_block8": m_b8,
                            "median_reversed": m_rev,
                            "median_density_matched": m_den,
                            "median_gue": m_gue,
                            "median_poisson": m_poi,
                            "median_shuffled_zeta": m_shuf,
                            "accepted_real": acc_real,
                            "train_to_test_gap_spectral_max": gap_sl_max,
                            **{STRICT_GATES_ORDER[i]: (g1, g2, g3, g4, g5, g6, g7, g8, g9, g10)[i] for i in range(10)},
                            "strict_specificity_pass": strict_pass,
                        }
                    )

    # --- interpret classification ---
    primary_gates = [g for g in gate_rows if int(g["dim"]) in dims]
    strict_all_dims = bool(primary_gates) and all(bool(g.get("strict_specificity_pass")) for g in primary_gates)

    def _med_rw_primary(d: int) -> float:
        xs = [
            float(r["pareto_objective_test"])
            for r in dedup_rows
            if int(r["dim"]) == d
            and r["V_mode"] == "full_V"
            and r["word_group"] == "random_words_n30"
            and r["target_group"] == "real_zeta_test"
            and math.isfinite(float(r["pareto_objective_test"]))
        ]
        return float(np.median(np.asarray(xs, dtype=np.float64))) if xs else float("nan")

    partial = False
    if not strict_all_dims and primary_gates:
        for g in primary_gates:
            d = int(g["dim"])
            j_real = float(g["j_real"])
            jr_ok = math.isfinite(j_real) and math.isfinite(_med_rw_primary(d)) and j_real < _med_rw_primary(d)
            shuffle_ok = bool(
                g.get("G2_lt_median_block4_J")
                and g.get("G3_lt_median_block8_J")
                and g.get("G4_lt_median_reversed_J")
                and g.get("G8_lt_median_shuffled_zeta_J")
                and g.get("G9_accepted_real_zeta_test")
            )
            fail_stat = not (
                bool(g.get("G5_lt_median_density_matched_J"))
                and bool(g.get("G6_lt_median_GUE_J"))
                and bool(g.get("G7_lt_median_poisson_J"))
            )
            if shuffle_ok and jr_ok and fail_stat:
                partial = True

    failed_stat_labels: List[str] = []
    for g in primary_gates:
        if not bool(g.get("strict_specificity_pass")):
            if not bool(g.get("G5_lt_median_density_matched_J")):
                failed_stat_labels.append("density-matched")
            if not bool(g.get("G6_lt_median_GUE_J")):
                failed_stat_labels.append("GUE")
            if not bool(g.get("G7_lt_median_poisson_J")):
                failed_stat_labels.append("Poisson")
            break

    uniq_partial_hint: List[str] = sorted(set(failed_stat_labels))
    if strict_all_dims:
        interp = "STRICT_SPECIFICITY_PASS"
        conclusion_human = (
            "V13O.3 supports zeta-specific out-of-sample structure under the tested finite-dimensional approximation."
        )
    elif partial:
        interp = "PARTIAL_SPECIFICITY"
        failed_txt = ", ".join(uniq_partial_hint) if uniq_partial_hint else "(refer to `v13o3_gate_summary.csv`)"
        conclusion_human = (
            "V13O.3 supports structural sensitivity but not full zeta specificity "
            f"(ensemble gate failures: {failed_txt})."
        )
    else:
        interp = "SPECIFICITY_FAIL"
        conclusion_human = (
            "V13O.3 rejects the current zeta-specificity hypothesis; "
            "the model is still fitting generic spectral/statistical structure."
        )

    # --- effect sizes: primary vs other word_groups on real_zeta_test ---
    effect_rows: List[Dict[str, Any]] = []
    for dim in dims:
        ref_ls = [
            float(r["pareto_objective_test"])
            for r in filt(dim, "full_V", "primary_word_seed6", "real_zeta_test")
            if math.isfinite(float(r["pareto_objective_test"]))
        ]
        if not ref_ls:
            continue
        jref = float(np.mean(np.asarray(ref_ls, dtype=np.float64)))
        ogs = sorted({str(r["word_group"]) for r in dedup_rows if int(r["dim"]) == dim})
        for og in ogs:
            if og == "primary_word_seed6":
                continue
            crs = [
                float(r["pareto_objective_test"])
                for r in filt(dim, "full_V", og, "real_zeta_test")
                if math.isfinite(float(r["pareto_objective_test"]))
            ]
            if not crs:
                continue
            mJ = float(np.mean(np.asarray(crs, dtype=np.float64)))
            effect_rows.append(
                {
                    "dim": dim,
                    "operator_word_group": og,
                    "mean_J_test_primary": jref,
                    "mean_J_test_group": mJ,
                    "delta_mean_J_test_vs_primary": mJ - jref,
                }
            )

    # --- best_by_dim ---
    best_by_dim: List[Dict[str, Any]] = []
    for dim in dims:
        cand_r = [
            r
            for r in dedup_rows
            if int(r["dim"]) == dim and r.get("accepted") and math.isfinite(float(r["pareto_objective_test"]))
        ]
        if not cand_r:
            best_by_dim.append({"dim": dim, "note": "no_accepted_rows"})
            continue
        best = min(cand_r, key=lambda x: float(x["pareto_objective_test"]))
        best_by_dim.append(
            {
                "dim": dim,
                "V_mode": best["V_mode"],
                "word_group": best["word_group"],
                "target_group": best["target_group"],
                "pareto_objective_test": best["pareto_objective_test"],
                "accepted": best["accepted"],
            }
        )

    def write_csv(path: Path, fields: List[str], rows: List[Dict[str, Any]]) -> None:
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for row in rows:
                w.writerow({k: csv_cell(row.get(k)) for k in fields})

    transfer_fields = list(dedup_rows[0].keys()) if dedup_rows else []
    if dedup_rows:
        write_csv(out_dir / "v13o3_transfer_summary.csv", transfer_fields, dedup_rows)
    if specificity_rows:
        write_csv(out_dir / "v13o3_specificity_scores.csv", list(specificity_rows[0].keys()), specificity_rows)
    if gate_rows:
        write_csv(out_dir / "v13o3_gate_summary.csv", list(gate_rows[0].keys()), gate_rows)
    if effect_rows:
        write_csv(out_dir / "v13o3_effect_sizes.csv", list(effect_rows[0].keys()), effect_rows)
    if best_by_dim:
        write_csv(out_dir / "v13o3_best_by_dim.csv", list(best_by_dim[0].keys()), best_by_dim)

    payload = {
        "warning": "Computational evidence only; not a proof of the Riemann Hypothesis.",
        "status": "V13O.3 strict specificity",
        "interpretation": interp,
        "strict_specificity_all_dims": strict_all_dims,
        "partial_specificity": bool(partial),
        "conclusion": conclusion_human,
        "partial_failed_stat_gates_hint": uniq_partial_hint if interp == "PARTIAL_SPECIFICITY" else [],
        "pareto_J": "spectral_log_mse + 0.15*spacing_mse_normalized + 0.50*ks_wigner",
        "acceptance_thresholds": {
            "operator_diff_final_max": 1e-3,
            "spacing_mse_normalized_max": 1.2,
            "ks_wigner_max": 0.25,
            "self_adjointness_fro_max": 1e-12,
            "finite_required": True,
            "train_to_test_gap_spectral_max": 2.0,
        },
        "inputs": {
            "candidate_json": str(cand_path),
            "formula_json": str(form_path),
            "v13n_summary": str(v13n_path),
            "candidate_id": spect.get("id"),
            "formula_meta_builder": (form.get("meta") or {}).get("builder_module"),
        },
        "v13n_operator_formula": v13n.get("operator_formula"),
        "dims": dims,
        "n_controls": n_ctrl,
        "n_random_words": n_rw,
        "strict_gates": list(STRICT_GATES_ORDER),
        "gate_summary": json_sanitize(gate_rows),
        "n_runs_raw": len(all_rows),
        "n_runs_deduped": len(dedup_rows),
        "parallel": bool(args.parallel),
        "num_workers": num_workers if args.parallel else 1,
        "total_jobs": total_jobs,
        "completed_jobs": len(combined_sorted),
        "failed_jobs": failed_jobs,
        "total_runtime_s": float(wall_s),
        "avg_job_runtime_s": avg_job_rt,
        "python_pid": os.getpid(),
        "raw_jobs_jsonl": str(raw_jobs_path.resolve()),
        "resume": bool(args.resume),
    }

    with open(out_dir / "v13o3_results.json", "w", encoding="utf-8") as f:
        json.dump(json_sanitize(payload), f, indent=2, allow_nan=False)

    md = [
        "# V13O.3 Strict specificity\n\n",
        "> **Computational evidence only; not a proof of the Riemann Hypothesis.**\n\n",
        "## Interpretation\n\n",
    ]
    if interp == "STRICT_SPECIFICITY_PASS":
        md.append(
            "**STRICT_SPECIFICITY_PASS:** The primary zeta operator beats all strict controls out-of-sample.\n\n"
        )
    elif interp == "PARTIAL_SPECIFICITY":
        md.extend(
            [
                "**PARTIAL_SPECIFICITY:** The primary braid word clears several shuffle/reversal/word-random baselines ",
                "(and may clear some ensembles), **but fails at least one statistical ensemble median gate** ",
                "(see strict gate columns).\n\n",
                "Typical subtype (evidence-guided): retains sensitivity to permutation/jitter structure while remaining ",
                "competitive with GUE/Poisson/density-matched spectra on the same objective.\n\n",
            ]
        )
    else:
        md.append(
            "**SPECIFICITY_FAIL:** The primary zeta operator does not beat the median or key controls.\n\n"
        )

    md.append(f"**Conclusion:** {conclusion_human}\n\n")
    md.append("## Strict gates (primary, full\\_V)\n\n")
    md.append("| dim | pass | j\\_real | median\\_controls |\n|---:|---:|---:|---:|\n")
    for g in sorted(gate_rows, key=lambda x: int(x["dim"])):
        md.append(
            f"| {g['dim']} | {g.get('strict_specificity_pass')} | "
            f"{g.get('j_real')} | {g.get('median_all_controls')} |\n"
        )
    md.append("\n## Outputs\n\n")
    md.append(
        "- `v13o3_results.json`, `v13o3_transfer_summary.csv`, `v13o3_specificity_scores.csv`, "
        "`v13o3_gate_summary.csv`, `v13o3_effect_sizes.csv`, `v13o3_best_by_dim.csv`\n"
    )
    (out_dir / "v13o3_report.md").write_text("".join(md), encoding="utf-8")

    tex_body = (
        "\\documentclass[11pt]{article}\n\\usepackage[margin=1in]{geometry}\n\\begin{document}\n"
        "\\title{V13O.3 Strict specificity}\n\\maketitle\n"
        "\\section{Warning}\nComputational evidence only; not a proof of the Riemann Hypothesis.\n\n"
        "\\section{Interpretation}\n" + latex_escape(interp) + "\n\n"
        "\\section{Conclusion}\n" + latex_escape(conclusion_human) + "\n\\end{document}\n"
    )
    (out_dir / "v13o3_report.tex").write_text(tex_body, encoding="utf-8")

    if _find_pdflatex() is None:
        print("[v13o3] WARNING: pdflatex not found; skipping PDF.", flush=True)
    elif try_pdflatex(out_dir / "v13o3_report.tex", out_dir, "v13o3_report.pdf"):
        print(f"Wrote {out_dir / 'v13o3_report.pdf'}", flush=True)
    else:
        print("[v13o3] WARNING: pdflatex failed or did not produce v13o3_report.pdf.", flush=True)

    odisp = str(out_dir.resolve())
    print(f"Wrote {odisp}/v13o3_results.json ({len(dedup_rows)} deduped transfer rows)", flush=True)
    print(f"Interpretation: {interp}", flush=True)

    print("\n--- Smoke command ---\n")
    print(
        "python3 scripts/run_v13o3_strict_specificity.py \\\n"
        "  --candidate_json runs/v13_candidate_theorem/candidate_operators.json \\\n"
        "  --formula_json runs/v13_operator_formula/formula_components_summary.json \\\n"
        "  --v13n_summary runs/v13n_theorem_report/v13n_summary.json \\\n"
        "  --out_dir runs/v13o3_strict_specificity_smoke_parallel \\\n"
        "  --seed 42 \\\n"
        "  --dims 64 128 256 \\\n"
        "  --max_iter 80 \\\n"
        "  --tol 1e-3 \\\n"
        "  --n_controls 5 \\\n"
        "  --n_random_words 5 \\\n"
        "  --progress_every 1 \\\n"
        "  --parallel \\\n"
        "  --num_workers 4\n"
    )
    print("--- Full command ---\n")
    print(
        "caffeinate -dimsu python3 scripts/run_v13o3_strict_specificity.py \\\n"
        "  --candidate_json runs/v13_candidate_theorem/candidate_operators.json \\\n"
        "  --formula_json runs/v13_operator_formula/formula_components_summary.json \\\n"
        "  --v13n_summary runs/v13n_theorem_report/v13n_summary.json \\\n"
        "  --out_dir runs/v13o3_strict_specificity_parallel \\\n"
        "  --seed 42 \\\n"
        "  --dims 64 128 256 \\\n"
        "  --max_iter 300 \\\n"
        "  --tol 1e-3 \\\n"
        "  --n_controls 30 \\\n"
        "  --n_random_words 30 \\\n"
        "  --progress_every 1 \\\n"
        "  --parallel \\\n"
        "  --num_workers 6 \\\n"
        "  --resume\n"
    )


if __name__ == "__main__":
    main()
